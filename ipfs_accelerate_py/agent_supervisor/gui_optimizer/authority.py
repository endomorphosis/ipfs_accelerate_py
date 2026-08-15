"""Fail-closed patch and browser-host security authority for VerifiedGuiOptimizer.

Interfaces owned by this module (board VGO-009):

* ``GuiPatchAuthority@1`` — allowed repository roots and forbidden change kinds
* ``GuiHostBoundaryPolicy@1`` — browser content cannot select host paths/commands
* ``GuiAcceptanceAuthority@1`` — evidence required for automatic acceptance

This is a pure, provider-free doctrine layer.  It does not alter backend
authorization, credentials, MCP execution, or the SwissKnife browser gateway.
Callers inject explicit path/change/evidence claims; the authority never
elevates UI state, browser policy output, scope declarations, or missing
evidence into permission.

Fail-closed invariants enforced here:

* mapping inputs are closed and strictly typed before coercion (unknown keys
  reject; only real booleans are accepted for boolean fields; identifiers and
  digests accept only nonempty canonical strings; collection fields accept only
  declared JSON array/object types — strings, mappings, numbers, booleans, and
  null never become valid collections);
* browser envelopes cannot hide path/command/credential selectors by nesting,
  placement, casing, URI encoding, or alternate spelling;
* claim-derived change kinds and computed patch/host decisions override
  acceptance input and cannot be replaced by the caller;
* authority evidence has a nonempty identity and, when used to authorize, is
  current and bound to the exact action and canonical nonempty argument digest;
* caller-supplied ``policy_decision_id`` / ``policy_fresh`` have no authority
  without current evidence bound to that exact action and digest;
* a scope declaration alone is never host authority.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Final
from urllib.parse import unquote, unquote_plus

# ---------------------------------------------------------------------------
# Interface / schema identity
# ---------------------------------------------------------------------------

GUI_PATCH_AUTHORITY_INTERFACE: Final[str] = "GuiPatchAuthority@1"
GUI_HOST_BOUNDARY_POLICY_INTERFACE: Final[str] = "GuiHostBoundaryPolicy@1"
GUI_ACCEPTANCE_AUTHORITY_INTERFACE: Final[str] = "GuiAcceptanceAuthority@1"

GUI_PATCH_AUTHORITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/patch-authority@1"
)
GUI_HOST_BOUNDARY_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/host-boundary-policy@1"
)
GUI_ACCEPTANCE_AUTHORITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/acceptance-authority@1"
)
GUI_AUTHORITY_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/authority-decision@1"
)

# Canonical SwissKnife / accelerator surfaces the optimizer may touch by default.
DEFAULT_ALLOWED_ROOTS: Final[tuple[str, ...]] = (
    "swissknife/web/js/apps/",
    "swissknife/web/js/",
    "swissknife/src/services/gui-optimizer/",
    "swissknife/test/fixtures/gui-optimizer/",
    "swissknife/test/unit/services/gui-optimizer/",
    "swissknife/test/browser/",
    "swissknife/test/e2e/",
    "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/gui_optimizer/",
    "external/ipfs_accelerate/test/api/",
    "external/ipfs_accelerate/test/fixtures/gui_optimizer/",
    "external/ipfs_datasets/ipfs_datasets_py/logic/gui_optimizer/",
    "external/ipfs_datasets/tests/unit/logic/gui_optimizer/",
    "external/ipfs_datasets/tests/fixtures/gui_optimizer/",
    "implementation_plan/evidence/verified_gui_optimizer/",
)

# Roots that are never optimizer write targets even when nested under an
# allowed prefix (defense in depth for generated/vendor/archive paths).
DEFAULT_FORBIDDEN_PATH_PARTS: Final[frozenset[str]] = frozenset(
    {
        "node_modules",
        "vendor",
        "vendors",
        "third_party",
        "generated",
        "dist",
        "build",
        "archive",
        "archives",
        "legacy-archive",
        "emergency-archive",
        "cleanup-archive",
        ".git",
    }
)

# Browser payload / fixture keys that must never cross the host boundary.
# Mirrors swissknife/src/services/mcp/all-app-tool-gateway.ts plus explicit
# process/command/path/credential selector aliases used by optimizer doctrine.
# Extended credential aliases that must never cross the browser→host boundary.
_EXTENDED_CREDENTIAL_ALIASES: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "auth_token",
        "client_secret",
        "private_key",
        "session_token",
        "refresh_token",
        "authorization_header",
        "api_token",
        "oauth_token",
        "accesstoken",
        "authtoken",
        "clientsecret",
        "privatekey",
        "sessiontoken",
        "refreshtoken",
        "authorizationheader",
        "apitoken",
        "oauthtoken",
    }
)

FORBIDDEN_BROWSER_PAYLOAD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "authorization",
        "backend_credentials",
        "bearer_token",
        "api_key",
        "password",
        "secret",
        "credential",
        "credentials",
        "host_path",
        "file_path",
        "filesystem_path",
        "host_file_path",
        "host_filesystem_path",
        "working_directory",
        "cwd",
        "file_uri",
        "python_process",
        "process_command",
        "stdio",
        "shell_command",
        "subprocess",
        "executable",
        "argv",
        "cmd",
        "access_token",
        "auth_token",
        "client_secret",
        "private_key",
        "session_token",
        "refresh_token",
        "authorization_header",
        "api_token",
        "oauth_token",
    }
)

# Host-side process/command selectors that browser content must not choose.
# Keep this aligned with the SwissKnife all-app tool gateway; do not forbid
# ordinary application intent fields such as a UI "command" name, but do
# reject abbreviated host selectors such as ``cmd``.
FORBIDDEN_BROWSER_COMMAND_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "process_command",
        "shell_command",
        "subprocess",
        "python_process",
        "stdio",
        "argv",
        "executable",
        "cmd",
    }
)

FORBIDDEN_BROWSER_PATH_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "host_path",
        "file_path",
        "filesystem_path",
        "host_file_path",
        "host_filesystem_path",
        "working_directory",
        "cwd",
        "file_uri",
    }
)

FORBIDDEN_BROWSER_CREDENTIAL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "authorization",
        "backend_credentials",
        "bearer_token",
        "api_key",
        "password",
        "secret",
        "credential",
        "credentials",
        "access_token",
        "auth_token",
        "client_secret",
        "private_key",
        "session_token",
        "refresh_token",
        "authorization_header",
        "api_token",
        "oauth_token",
    }
)

_PATCH_CLAIM_KEYS: Final[frozenset[str]] = frozenset(
    {"path", "declared", "change_kinds"}
)
_BROWSER_HOST_INPUT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "payload",
        "fixture_only",
        "uses_production_credentials",
        "uses_production_services",
        "uses_production_mcp_tools",
        "uses_user_or_legal_data",
        "selected_host_paths",
        "selected_commands",
        "selected_executables",
    }
)
_AUTHORITY_EVIDENCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "kind",
        "valid",
        "evidence_id",
        "binds_action_id",
        "binds_argument_digest",
        "policy_decision_id",
        "policy_fresh",
        "notes",
    }
)
_ACCEPTANCE_REQUEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "intended_action_id",
        "intended_argument_digest",
        "ui_visible",
        "ui_enabled",
        "browser_policy_outcome",
        "browser_policy_authoritative_claim",
        "policy_decision_id",
        "policy_fresh",
        "confirmation_required",
        "confirmation_action_id",
        "confirmation_argument_digest",
        "confirmation_granted",
        "change_kinds",
        "evidence",
        "accessibility_regression",
        "security_regression",
        "host_boundary_decision",
        "patch_authority_decision",
    }
)

_IDENTIFIER_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "evidence_id",
        "binds_action_id",
        "intended_action_id",
        "confirmation_action_id",
        "policy_decision_id",
        "browser_policy_outcome",
    }
)
_DIGEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "binds_argument_digest",
        "intended_argument_digest",
        "confirmation_argument_digest",
    }
)


class GuiAuthorityError(ValueError):
    """Malformed authority input.  Never grants permission."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "invalid_authority_input",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})


class AuthorityVerdict(str, Enum):
    """The only outcomes of an authority evaluation."""

    ALLOW = "allow"
    REJECT = "reject"
    REQUIRE_HUMAN_REVIEW = "require_human_review"


class ForbiddenChangeKind(str, Enum):
    """Sensitive mutation classes that never auto-accept without extra gates."""

    BACKEND_AUTHORIZATION = "backend_authorization"
    CREDENTIALS = "credentials"
    ARBITRARY_HTML_EXECUTION = "arbitrary_html_execution"
    DISABLED_SECURITY_CHECK = "disabled_security_check"
    DELETED_TEST = "deleted_test"
    UNRELATED_APPLICATION = "unrelated_application"
    UNVERIFIED_ACTION_BINDING = "unverified_action_binding"
    OUT_OF_SCOPE_PATH = "out_of_scope_path"
    CONFIRMATION_WEAKENING = "confirmation_weakening"
    POLICY_WEAKENING = "policy_weakening"
    HOST_BOUNDARY_BYPASS = "host_boundary_bypass"
    PRODUCTION_TOOL_ACCESS = "production_tool_access"


class AuthorityEvidenceKind(str, Enum):
    """Evidence classes that may support automatic acceptance."""

    CONTRACT_VERIFICATION = "contract_verification"
    HUMAN_REVIEW = "human_review"
    HOST_POLICY_REEVALUATION = "host_policy_reevaluation"
    EXACT_CONFIRMATION_BINDING = "exact_confirmation_binding"
    FIXTURE_BOUNDARY = "fixture_boundary"
    SCOPE_DECLARATION = "scope_declaration"


class AuthorityReasonCode(str, Enum):
    """Stable reason codes consumed by later patch-scope / acceptance gates."""

    ALLOWED = "allowed"
    PATH_OUTSIDE_ALLOWED_ROOTS = "path_outside_allowed_roots"
    PATH_ABSOLUTE_OR_TRAVERSAL = "path_absolute_or_traversal"
    PATH_FORBIDDEN_SEGMENT = "path_forbidden_segment"
    FORBIDDEN_CHANGE_KIND = "forbidden_change_kind"
    UNDECLARED_PATH = "undeclared_path"
    BROWSER_HOST_PATH_FORBIDDEN = "browser_host_path_forbidden"
    BROWSER_COMMAND_FORBIDDEN = "browser_command_forbidden"
    BROWSER_CREDENTIAL_FORBIDDEN = "browser_credential_forbidden"
    BROWSER_PRODUCTION_INPUT_FORBIDDEN = "browser_production_input_forbidden"
    UI_STATE_NOT_AUTHORIZATION = "ui_state_not_authorization"
    BROWSER_POLICY_NOT_AUTHORITATIVE = "browser_policy_not_authoritative"
    STALE_POLICY_DECISION = "stale_policy_decision"
    CONFIRMATION_BINDING_MISMATCH = "confirmation_binding_mismatch"
    CONFIRMATION_REQUIRED = "confirmation_required"
    MISSING_AUTHORITY_EVIDENCE = "missing_authority_evidence"
    INVALID_AUTHORITY_EVIDENCE = "invalid_authority_evidence"
    SENSITIVE_CHANGE_REQUIRES_REVIEW = "sensitive_change_requires_review"
    SENSITIVE_CHANGE_REQUIRES_CONTRACT = "sensitive_change_requires_contract"
    ACCESSIBILITY_REGRESSION = "accessibility_regression"
    SECURITY_REGRESSION = "security_regression"
    FIXTURE_ONLY_VIOLATION = "fixture_only_violation"
    INVALID_AUTHORITY_INPUT = "invalid_authority_input"
    UNKNOWN_FIELD = "unknown_field"
    SCOPE_DECLARATION_NOT_AUTHORITY = "scope_declaration_not_authority"
    EVIDENCE_BINDING_MISMATCH = "evidence_binding_mismatch"
    EVIDENCE_NOT_CURRENT = "evidence_not_current"
    EVIDENCE_IDENTITY_REQUIRED = "evidence_identity_required"
    CALLER_POLICY_NOT_AUTHORITY = "caller_policy_not_authority"
    NONCANONICAL_ARGUMENT_DIGEST = "noncanonical_argument_digest"
    EMPTY_ARGUMENT_DIGEST = "empty_argument_digest"
    INVALID_COLLECTION_TYPE = "invalid_collection_type"


# Change kinds that always require contract verification or human review.
SENSITIVE_CHANGE_KINDS: Final[frozenset[ForbiddenChangeKind]] = frozenset(
    {
        ForbiddenChangeKind.BACKEND_AUTHORIZATION,
        ForbiddenChangeKind.CREDENTIALS,
        ForbiddenChangeKind.ARBITRARY_HTML_EXECUTION,
        ForbiddenChangeKind.DISABLED_SECURITY_CHECK,
        ForbiddenChangeKind.DELETED_TEST,
        ForbiddenChangeKind.UNRELATED_APPLICATION,
        ForbiddenChangeKind.UNVERIFIED_ACTION_BINDING,
        ForbiddenChangeKind.CONFIRMATION_WEAKENING,
        ForbiddenChangeKind.POLICY_WEAKENING,
        ForbiddenChangeKind.HOST_BOUNDARY_BYPASS,
        ForbiddenChangeKind.PRODUCTION_TOOL_ACCESS,
    }
)

# Sensitive kinds that never auto-accept even with contract verification alone
# (must escalate to human review unless an explicit human-review receipt exists).
ALWAYS_HUMAN_REVIEW_KINDS: Final[frozenset[ForbiddenChangeKind]] = frozenset(
    {
        ForbiddenChangeKind.BACKEND_AUTHORIZATION,
        ForbiddenChangeKind.CREDENTIALS,
        ForbiddenChangeKind.DISABLED_SECURITY_CHECK,
        ForbiddenChangeKind.HOST_BOUNDARY_BYPASS,
        ForbiddenChangeKind.PRODUCTION_TOOL_ACCESS,
        ForbiddenChangeKind.CONFIRMATION_WEAKENING,
        ForbiddenChangeKind.POLICY_WEAKENING,
    }
)

# Evidence kinds that may authorize an intended action when current and bound.
# Scope declarations and fixture-boundary markers never grant host authority.
HOST_AUTHORIZING_EVIDENCE_KINDS: Final[frozenset[AuthorityEvidenceKind]] = frozenset(
    {
        AuthorityEvidenceKind.CONTRACT_VERIFICATION,
        AuthorityEvidenceKind.HUMAN_REVIEW,
        AuthorityEvidenceKind.HOST_POLICY_REEVALUATION,
        AuthorityEvidenceKind.EXACT_CONFIRMATION_BINDING,
    }
)

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])([A-Z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
# Exact canonical argument-digest grammar.  Uppercase hex, whitespace, other
# algorithms, short/long payloads, empty values, and arbitrary equal strings
# never authorize.
_CANONICAL_ARGUMENT_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_WINDOWS_PATH_RE = re.compile(r"^[a-zA-Z]:[\\/]")
_DRIVE_RELATIVE_RE = re.compile(r"^[a-zA-Z]:(?![\\/]|$)")
_RELATIVE_TRAVERSAL_RE = re.compile(
    r"(?:^|[/\\])\.\.(?:[/\\]|$)|(?:^|[/\\])\.(?:[/\\]|$)"
)
_CREDENTIAL_VALUE_RE = re.compile(
    r"^(?:password|secret|token|apikey|api_key|bearer)\s*[=:]",
    re.IGNORECASE,
)
_CREDENTIAL_INLINE_MARKERS: Final[tuple[str, ...]] = (
    "password=",
    "secret=",
    "token=",
    "api_key=",
    "apikey=",
    "bearer ",
    "secret:",
    "password:",
    "token:",
)
_COMMAND_EXECUTABLE_RE = re.compile(
    r"(?:^|[\s;/|&`])"
    r"(?:cmd(?:\.exe)?|powershell(?:\.exe)?|pwsh(?:\.exe)?|bash|sh|zsh|fish|"
    r"python(?:3)?|sudo|curl|wget)"
    r"(?:$|[\s;/|&`])",
    re.IGNORECASE,
)


def _exact_str(value: Any, name: str) -> str:
    """Accept only the exact built-in ``str`` type (never str subclasses/Enums)."""
    if type(value) is not str:
        raise GuiAuthorityError(
            f"{name} must be a string",
            reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _text(value: Any, name: str, *, required: bool = True) -> str:
    """Strict string field.  Never coerces numbers/bools/null into strings."""
    text_value = _exact_str(value, name)
    if "\x00" in text_value:
        raise GuiAuthorityError(
            f"{name} must not contain NUL",
            reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            details={"field": name},
        )
    text = text_value.strip()
    if required and not text:
        raise GuiAuthorityError(
            f"{name} must not be empty",
            reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            details={"field": name},
        )
    return text


def _identifier_field(
    value: Any,
    name: str,
    *,
    allow_empty: bool = False,
) -> str:
    """Identifier fields accept only a canonical nonempty string when set."""
    text_value = _exact_str(value, name)
    if "\x00" in text_value:
        raise GuiAuthorityError(
            f"{name} must not contain NUL",
            reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            details={"field": name},
        )
    if text_value == "":
        if allow_empty:
            return ""
        raise GuiAuthorityError(
            f"{name} must be a nonempty string identifier",
            reason_code=(
                AuthorityReasonCode.EVIDENCE_IDENTITY_REQUIRED.value
                if name == "evidence_id"
                else AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value
            ),
            details={"field": name},
        )
    if text_value != text_value.strip():
        raise GuiAuthorityError(
            f"{name} must be a canonical nonempty string identifier",
            reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            details={"field": name},
        )
    return text_value


def _digest_field(
    value: Any,
    name: str,
    *,
    allow_empty: bool = False,
) -> str:
    """Digest fields accept only exact ``sha256:[0-9a-f]{64}`` when set.

    Omitted optional fields may default to empty via ``allow_empty=True``.
    Present empty strings, whitespace-padded values, uppercase hex, other
    algorithm prefixes, short/long payloads, and arbitrary non-canonical
    strings (even when equal across fields) always reject.
    """
    text_value = _exact_str(value, name)
    if "\x00" in text_value:
        raise GuiAuthorityError(
            f"{name} must not contain NUL",
            reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            details={"field": name},
        )
    if text_value == "":
        if allow_empty:
            return ""
        raise GuiAuthorityError(
            f"{name} must be a nonempty canonical argument digest",
            reason_code=AuthorityReasonCode.EMPTY_ARGUMENT_DIGEST.value,
            details={"field": name},
        )
    if text_value != text_value.strip() or not _CANONICAL_ARGUMENT_DIGEST_RE.fullmatch(
        text_value
    ):
        raise GuiAuthorityError(
            f"{name} must be a canonical argument digest matching "
            "sha256:[0-9a-f]{64}",
            reason_code=AuthorityReasonCode.NONCANONICAL_ARGUMENT_DIGEST.value,
            details={"field": name},
        )
    return text_value


def _reject_present_null(payload: Mapping[str, Any], key: str) -> None:
    """Present null rejects for non-nullable optional scalars; omitted may default."""
    if key in payload and payload[key] is None:
        raise GuiAuthorityError(
            f"{key} must not be null when present",
            reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            details={"field": key, "value_type": "NoneType"},
        )


def _optional_identifier(payload: Mapping[str, Any], key: str) -> str:
    """Omitted → empty default; present null rejects; present value is strict."""
    if key not in payload:
        return ""
    _reject_present_null(payload, key)
    return _identifier_field(payload[key], key, allow_empty=False)


def _optional_digest(payload: Mapping[str, Any], key: str) -> str:
    """Omitted → empty default; present null rejects; present value is canonical."""
    if key not in payload:
        return ""
    _reject_present_null(payload, key)
    return _digest_field(payload[key], key, allow_empty=False)


def _optional_notes(payload: Mapping[str, Any], key: str = "notes") -> str:
    if key not in payload:
        return ""
    _reject_present_null(payload, key)
    value = _exact_str(payload[key], key)
    if "\x00" in value:
        raise GuiAuthorityError(
            f"{key} must not contain NUL",
            reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            details={"field": key},
        )
    return value


def _bool(value: Any, name: str) -> bool:
    """Accept only exact built-in booleans.  Subclasses and coercions reject."""
    if type(value) is not bool:
        raise GuiAuthorityError(
            f"{name} must be a boolean",
            reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _optional_mapping_bool(
    payload: Mapping[str, Any], key: str, default: bool
) -> bool:
    if key not in payload:
        return default
    return _bool(payload[key], key)


def _require_exact_wire_mapping(value: Any, name: str) -> dict[str, Any]:
    """Wire objects must be exact ``dict`` before any attribute introspection."""
    if type(value) is not dict:
        raise GuiAuthorityError(
            f"{name} must be a JSON object",
            reason_code=AuthorityReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    for key in value:
        if type(key) is not str:
            raise GuiAuthorityError(
                f"{name} keys must be strings",
                reason_code=AuthorityReasonCode.INVALID_COLLECTION_TYPE.value,
                details={"field": name, "key_type": type(key).__name__},
            )
    return value


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise GuiAuthorityError(
            f"{name} must be a JSON object/mapping",
            reason_code=AuthorityReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    for key in value:
        if type(key) is not str:
            raise GuiAuthorityError(
                f"{name} keys must be strings",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
                details={"field": name},
            )
    return value


def _require_json_object(value: Any, name: str) -> dict[str, Any]:
    """JSON objects on the wire are exact dicts with exact string keys only."""
    return _require_exact_wire_mapping(value, name)


def _require_json_array(value: Any, name: str) -> list[Any]:
    """Wire collection fields accept only exact JSON arrays (built-in list).

    Python tuples, list subclasses, strings, mappings, numbers, booleans, null,
    and other containers never become valid arrays.
    """
    if type(value) is not list:
        raise GuiAuthorityError(
            f"{name} must be a JSON array (list); "
            f"{type(value).__name__} is not a valid collection",
            reason_code=AuthorityReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _require_python_sequence(value: Any, name: str) -> Sequence[Any]:
    """Python constructor API: exact list or exact tuple only."""
    if type(value) is list or type(value) is tuple:
        return value
    raise GuiAuthorityError(
        f"{name} must be a JSON array/sequence",
        reason_code=AuthorityReasonCode.INVALID_COLLECTION_TYPE.value,
        details={"field": name, "value_type": type(value).__name__},
    )


def _optional_json_array(
    payload: Mapping[str, Any], key: str
) -> list[Any] | None:
    """Return None when absent; reject null, tuples, and non-array types when present."""
    if key not in payload:
        return None
    value = payload[key]
    if value is None:
        raise GuiAuthorityError(
            f"{key} must be a JSON array when present; null is not a collection",
            reason_code=AuthorityReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": key, "value_type": "NoneType"},
        )
    return _require_json_array(value, key)


def _canonicalize_json_value(value: Any, name: str) -> Any:
    """Recursively retain only exact built-in RFC-JSON wire shapes.

    Rejects custom subclasses before any overridable method is invoked, rejects
    non-finite floats, and returns a deep-copied tree of exact builtins.
    """
    if value is None:
        return None
    value_type = type(value)
    if value_type is bool:
        return value
    if value_type is int:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise GuiAuthorityError(
                f"{name} must be a finite JSON number",
                reason_code=AuthorityReasonCode.INVALID_COLLECTION_TYPE.value,
                details={"field": name, "value_type": "float"},
            )
        return value
    if value_type is str:
        return value
    if value_type is list:
        return [
            _canonicalize_json_value(child, f"{name}[{index}]")
            for index, child in enumerate(value)
        ]
    if value_type is dict:
        out: dict[str, Any] = {}
        for key, child in value.items():
            if type(key) is not str:
                raise GuiAuthorityError(
                    f"{name} object keys must be strings",
                    reason_code=AuthorityReasonCode.INVALID_COLLECTION_TYPE.value,
                    details={"field": name, "key_type": type(key).__name__},
                )
            out[key] = _canonicalize_json_value(child, f"{name}.{key}")
        return out
    raise GuiAuthorityError(
        f"{name} must be a JSON value; {type(value).__name__} is not a "
        "JSON array, object, string, number, boolean, or null",
        reason_code=AuthorityReasonCode.INVALID_COLLECTION_TYPE.value,
        details={"field": name, "value_type": type(value).__name__},
    )


def _assert_json_shape(value: Any, name: str) -> None:
    """Recursive browser payloads admit only exact built-in JSON shapes."""
    _canonicalize_json_value(value, name)


def _json_roundtrip_tree(value: Any) -> Any:
    """Serialize with allow_nan=False and re-load into exact built-ins."""
    return json.loads(json.dumps(value, allow_nan=False, separators=(",", ":")))


def _reject_unknown(
    payload: Mapping[str, Any],
    allowed: frozenset[str],
    noun: str,
) -> None:
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise GuiAuthorityError(
            f"{noun} contains unknown fields: {unknown}",
            reason_code=AuthorityReasonCode.UNKNOWN_FIELD.value,
            details={"noun": noun, "unknown_fields": unknown},
        )


def _decode_field_key(key: str) -> str:
    """Undo URI encoding / plus-forms so disguised selectors cannot hide."""
    text = str(key)
    # Repeated unquote covers double-encoding (host%252Fpath → host%2Fpath → ...).
    for _ in range(4):
        decoded = unquote_plus(unquote(text))
        if decoded == text:
            break
        text = decoded
    return text


def _strip_encoded_alias_suffixes(token: str) -> str:
    """Strip only ``encoded`` alias suffixes (host_path_encoded, fileUriEncoded).

    Do not strip bare ``uri``/``path`` suffixes — ``file_uri`` must remain a
    path selector identity.
    """
    text = token
    if text.endswith("_encoded") and len(text) > len("_encoded"):
        return text[: -len("_encoded")].rstrip("_")
    if text.endswith("encoded") and len(text) > len("encoded"):
        # camelCase-normalized forms such as ``credentialencoded`` after
        # compacting are handled by compact tokens; keep snake form here.
        stripped = text[: -len("encoded")].rstrip("_")
        if stripped:
            return stripped
    return text


def _canonical_field_token(key: str) -> str:
    """Normalize key casing/separators so disguised selectors cannot hide."""
    text = _decode_field_key(key).strip()
    text = _CAMEL_BOUNDARY.sub(r"_\1", text)
    text = (
        text.replace("-", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )
    text = text.lower()
    while "__" in text:
        text = text.replace("__", "_")
    text = text.strip("_")
    return _strip_encoded_alias_suffixes(text)


def _compact_field_token(key: str) -> str:
    return _NON_ALNUM.sub("", _canonical_field_token(key))


def _forbidden_token_sets() -> tuple[frozenset[str], frozenset[str]]:
    canonical = set(FORBIDDEN_BROWSER_PAYLOAD_KEYS) | set(
        FORBIDDEN_BROWSER_COMMAND_FIELDS
    ) | set(FORBIDDEN_BROWSER_PATH_FIELDS) | set(FORBIDDEN_BROWSER_CREDENTIAL_FIELDS)
    compact = {_compact_field_token(item) for item in canonical}
    return frozenset(canonical), frozenset(compact)


_FORBIDDEN_CANONICAL_KEYS, _FORBIDDEN_COMPACT_KEYS = _forbidden_token_sets()
_PATH_CANONICAL = frozenset(FORBIDDEN_BROWSER_PATH_FIELDS)
_PATH_COMPACT = frozenset(_compact_field_token(item) for item in _PATH_CANONICAL)
_COMMAND_CANONICAL = frozenset(FORBIDDEN_BROWSER_COMMAND_FIELDS)
_COMMAND_COMPACT = frozenset(
    _compact_field_token(item) for item in _COMMAND_CANONICAL
)
_CREDENTIAL_CANONICAL = frozenset(FORBIDDEN_BROWSER_CREDENTIAL_FIELDS)
_CREDENTIAL_COMPACT = frozenset(
    _compact_field_token(item) for item in _CREDENTIAL_CANONICAL
)


def _classify_forbidden_key(key: str) -> str | None:
    """Return 'path', 'command', 'credential', or None for a payload key."""
    if type(key) is not str:
        return None
    canonical = _canonical_field_token(key)
    compact = _compact_field_token(key)
    if (
        canonical in _CREDENTIAL_CANONICAL
        or compact in _CREDENTIAL_COMPACT
        or canonical in _EXTENDED_CREDENTIAL_ALIASES
        or compact in _EXTENDED_CREDENTIAL_ALIASES
    ):
        return "credential"
    if canonical in _PATH_CANONICAL or compact in _PATH_COMPACT:
        return "path"
    if canonical in _COMMAND_CANONICAL or compact in _COMMAND_COMPACT:
        return "command"
    if canonical in _FORBIDDEN_CANONICAL_KEYS or compact in _FORBIDDEN_COMPACT_KEYS:
        return "path"
    return None


def _normalize_repo_path(value: Any, name: str = "path") -> str:
    """Normalize a repository-relative POSIX path; reject absolute/traversal."""
    raw = _text(value, name).replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or raw.startswith("/")
        or ".." in path.parts
        or raw != path.as_posix()
        or not raw
    ):
        raise GuiAuthorityError(
            f"{name} must be a normalized repository-relative path",
            reason_code=AuthorityReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL.value,
            details={"field": name, "value": str(value)},
        )
    return raw


def _as_change_kind(value: Any, *, wire: bool = False) -> ForbiddenChangeKind:
    """Coerce a change kind.

    Direct typed constructors may pass ``ForbiddenChangeKind`` members.  Wire
    inputs accept only exact built-in strings — Python Enum members reject even
    though they are ``str`` subclasses.
    """
    if not wire and type(value) is ForbiddenChangeKind:
        return value
    if type(value) is not str:
        raise GuiAuthorityError(
            "change_kind must be a string",
            reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            details={"value_type": type(value).__name__},
        )
    text = _text(value, "change_kind")
    try:
        return ForbiddenChangeKind(text)
    except ValueError as exc:
        raise GuiAuthorityError(
            f"unknown change kind: {text}",
            reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            details={"change_kind": text},
        ) from exc


def _as_evidence_kind(value: Any, *, wire: bool = False) -> AuthorityEvidenceKind:
    """Coerce an evidence kind.

    Direct typed constructors may pass ``AuthorityEvidenceKind`` members.  Wire
    inputs accept only exact built-in strings.
    """
    if not wire and type(value) is AuthorityEvidenceKind:
        return value
    if type(value) is not str:
        raise GuiAuthorityError(
            "evidence_kind must be a string",
            reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            details={"value_type": type(value).__name__},
        )
    text = _text(value, "evidence_kind")
    try:
        return AuthorityEvidenceKind(text)
    except ValueError as exc:
        raise GuiAuthorityError(
            f"unknown evidence kind: {text}",
            reason_code=AuthorityReasonCode.INVALID_AUTHORITY_EVIDENCE.value,
            details={"evidence_kind": text},
        ) from exc


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    mapping = _require_mapping(value, "details")
    return MappingProxyType(dict(mapping))


def _coerce_change_kinds(
    value: Any,
    *,
    field_name: str = "change_kinds",
    wire: bool = False,
) -> tuple[ForbiddenChangeKind, ...]:
    """Strict array of change kinds.  Scalars/null/mappings never coerce.

    Wire inputs (``wire=True``) accept only JSON lists; Python constructors may
    pass tuples of already-typed values.
    """
    if value is None:
        # Absent optional default is handled by callers; explicit null rejects.
        raise GuiAuthorityError(
            f"{field_name} must be a JSON array; null is not a collection",
            reason_code=AuthorityReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": field_name, "value_type": "NoneType"},
        )
    sequence: Sequence[Any]
    if wire:
        sequence = _require_json_array(value, field_name)
        return tuple(_as_change_kind(kind, wire=True) for kind in sequence)
    sequence = _require_python_sequence(value, field_name)
    return tuple(_as_change_kind(kind, wire=False) for kind in sequence)


def _optional_change_kinds(
    payload: Mapping[str, Any], key: str = "change_kinds"
) -> tuple[ForbiddenChangeKind, ...]:
    if key not in payload:
        return ()
    return _coerce_change_kinds(payload[key], field_name=key, wire=True)


def _claim_change_kinds(
    claims: Sequence[PatchPathClaim | Mapping[str, Any]],
) -> tuple[ForbiddenChangeKind, ...]:
    merged: list[ForbiddenChangeKind] = []
    for claim in claims:
        if type(claim) is PatchPathClaim:
            merged.extend(claim.change_kinds)
        elif type(claim) is dict:
            merged.extend(_optional_change_kinds(claim))
    # Preserve order while de-duplicating.
    seen: set[ForbiddenChangeKind] = set()
    ordered: list[ForbiddenChangeKind] = []
    for kind in merged:
        if kind not in seen:
            seen.add(kind)
            ordered.append(kind)
    return tuple(ordered)


def path_has_forbidden_segment(
    path: str,
    *,
    forbidden_parts: frozenset[str] = DEFAULT_FORBIDDEN_PATH_PARTS,
) -> bool:
    """Return True when any path segment is a forbidden generated/vendor part."""
    normalized = _normalize_repo_path(path)
    return any(part in forbidden_parts for part in PurePosixPath(normalized).parts)


def path_under_allowed_roots(
    path: str,
    *,
    allowed_roots: Sequence[str] = DEFAULT_ALLOWED_ROOTS,
) -> bool:
    """Return True when ``path`` is under at least one allowed root prefix."""
    normalized = _normalize_repo_path(path)
    for root in allowed_roots:
        prefix = _text(root, "allowed_root")
        if not prefix.endswith("/"):
            prefix = f"{prefix}/"
        if normalized == prefix[:-1] or normalized.startswith(prefix):
            return True
    return False


@dataclass(frozen=True)
class AuthorityDecision:
    """Typed, fail-closed decision shared by all three authority interfaces."""

    verdict: AuthorityVerdict
    reason_codes: tuple[str, ...]
    interface: str
    schema: str = GUI_AUTHORITY_DECISION_SCHEMA
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, AuthorityVerdict):
            object.__setattr__(
                self, "verdict", AuthorityVerdict(str(self.verdict))
            )
        codes = tuple(
            sorted(
                {
                    _text(code, "reason_code")
                    for code in (self.reason_codes or ())
                }
            )
        )
        if not codes:
            codes = (
                AuthorityReasonCode.ALLOWED.value
                if self.verdict is AuthorityVerdict.ALLOW
                else AuthorityReasonCode.MISSING_AUTHORITY_EVIDENCE.value
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
            "schema": self.schema,
            "interface": self.interface,
            "verdict": self.verdict.value,
            "reason_codes": list(self.reason_codes),
            "message": self.message,
            "details": dict(self.details),
            "allowed": self.allowed,
            "rejected": self.rejected,
            "requires_human_review": self.requires_human_review,
        }


def _decision(
    verdict: AuthorityVerdict,
    *reason_codes: AuthorityReasonCode | str,
    interface: str,
    message: str = "",
    details: Mapping[str, Any] | None = None,
    schema: str = GUI_AUTHORITY_DECISION_SCHEMA,
) -> AuthorityDecision:
    codes = tuple(
        code.value if isinstance(code, AuthorityReasonCode) else str(code)
        for code in reason_codes
    )
    return AuthorityDecision(
        verdict=verdict,
        reason_codes=codes,
        interface=interface,
        schema=schema,
        message=message,
        details=details or {},
    )


# ---------------------------------------------------------------------------
# GuiPatchAuthority@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatchPathClaim:
    """One repository path a proposal intends to create, modify, or delete."""

    path: str
    declared: bool = True
    change_kinds: tuple[ForbiddenChangeKind, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_repo_path(self.path, "path"))
        object.__setattr__(self, "declared", _bool(self.declared, "declared"))
        if self.change_kinds is None:
            raise GuiAuthorityError(
                "change_kinds must be a JSON array; null is not a collection",
                reason_code=AuthorityReasonCode.INVALID_COLLECTION_TYPE.value,
                details={"field": "change_kinds"},
            )
        kinds = tuple(
            _as_change_kind(kind)
            for kind in _require_python_sequence(self.change_kinds, "change_kinds")
        )
        object.__setattr__(self, "change_kinds", kinds)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, index: int = 0) -> "PatchPathClaim":
        if type(raw) is not dict:
            raise GuiAuthorityError(
                f"claims[{index}] must be a mapping",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
                details={"value_type": type(raw).__name__},
            )
        _reject_unknown(raw, _PATCH_CLAIM_KEYS, f"claims[{index}]")
        if "path" not in raw:
            raise GuiAuthorityError(
                f"claims[{index}].path is required",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            )
        if raw["path"] is None:
            raise GuiAuthorityError(
                f"claims[{index}].path must not be null when present",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
                details={"field": "path", "value_type": "NoneType"},
            )
        if "declared" in raw and raw["declared"] is None:
            raise GuiAuthorityError(
                f"claims[{index}].declared must not be null when present",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
                details={"field": "declared", "value_type": "NoneType"},
            )
        return cls(
            path=raw["path"],
            declared=_optional_mapping_bool(raw, "declared", True),
            change_kinds=_optional_change_kinds(raw),
        )


@dataclass(frozen=True)
class GuiPatchAuthority:
    """Allowed roots and forbidden change kinds for optimizer patches.

    Interface: ``GuiPatchAuthority@1``.
    """

    allowed_roots: tuple[str, ...] = DEFAULT_ALLOWED_ROOTS
    forbidden_path_parts: frozenset[str] = DEFAULT_FORBIDDEN_PATH_PARTS
    schema: str = GUI_PATCH_AUTHORITY_SCHEMA
    interface: str = GUI_PATCH_AUTHORITY_INTERFACE

    def __post_init__(self) -> None:
        # Exact-type check before any truthiness or iteration so subclass-controlled
        # RuntimeError cannot masquerade as safe rejection evidence.
        roots_value = self.allowed_roots
        if type(roots_value) is not tuple:
            raise GuiAuthorityError(
                "allowed_roots must be an exact tuple of strings",
                reason_code=AuthorityReasonCode.INVALID_COLLECTION_TYPE.value,
                details={
                    "field": "allowed_roots",
                    "value_type": type(roots_value).__name__,
                },
            )
        roots_list: list[str] = []
        for root in roots_value:
            if type(root) is not str:
                raise GuiAuthorityError(
                    "allowed_roots entries must be exact strings",
                    reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
                    details={
                        "field": "allowed_roots",
                        "value_type": type(root).__name__,
                    },
                )
            normalized = _text(root, "allowed_root")
            if not normalized.endswith("/"):
                normalized = f"{normalized}/"
            roots_list.append(normalized)
        if not roots_list:
            raise GuiAuthorityError(
                "allowed_roots must not be empty",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            )
        object.__setattr__(self, "allowed_roots", tuple(roots_list))
        parts_value = self.forbidden_path_parts
        if type(parts_value) is not frozenset:
            raise GuiAuthorityError(
                "forbidden_path_parts must be an exact frozenset",
                reason_code=AuthorityReasonCode.INVALID_COLLECTION_TYPE.value,
                details={
                    "field": "forbidden_path_parts",
                    "value_type": type(parts_value).__name__,
                },
            )
        parts = frozenset(
            _text(part, "forbidden_path_part") for part in parts_value
        )
        object.__setattr__(self, "forbidden_path_parts", parts)
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        object.__setattr__(self, "interface", _text(self.interface, "interface"))

    def evaluate_path(self, path: str, *, declared: bool = True) -> AuthorityDecision:
        """Evaluate a single path against allowed roots and forbidden segments."""
        declared_flag = _bool(declared, "declared")
        try:
            normalized = _normalize_repo_path(path)
        except GuiAuthorityError as exc:
            return _decision(
                AuthorityVerdict.REJECT,
                AuthorityReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL,
                interface=self.interface,
                schema=self.schema,
                message=str(exc),
                details=exc.details,
            )
        if not declared_flag:
            return _decision(
                AuthorityVerdict.REJECT,
                AuthorityReasonCode.UNDECLARED_PATH,
                interface=self.interface,
                schema=self.schema,
                message="undeclared paths are rejected",
                details={"path": normalized},
            )
        if path_has_forbidden_segment(
            normalized, forbidden_parts=self.forbidden_path_parts
        ):
            return _decision(
                AuthorityVerdict.REJECT,
                AuthorityReasonCode.PATH_FORBIDDEN_SEGMENT,
                interface=self.interface,
                schema=self.schema,
                message="path contains a forbidden segment",
                details={"path": normalized},
            )
        if not path_under_allowed_roots(
            normalized, allowed_roots=self.allowed_roots
        ):
            return _decision(
                AuthorityVerdict.REJECT,
                AuthorityReasonCode.PATH_OUTSIDE_ALLOWED_ROOTS,
                ForbiddenChangeKind.OUT_OF_SCOPE_PATH.value,
                interface=self.interface,
                schema=self.schema,
                message="path is outside allowed optimizer roots",
                details={
                    "path": normalized,
                    "allowed_roots": list(self.allowed_roots),
                },
            )
        return _decision(
            AuthorityVerdict.ALLOW,
            AuthorityReasonCode.ALLOWED,
            interface=self.interface,
            schema=self.schema,
            message="path is within allowed optimizer roots",
            details={"path": normalized},
        )

    def evaluate_change_kinds(
        self, change_kinds: Sequence[ForbiddenChangeKind | str] | None
    ) -> AuthorityDecision:
        """Classify sensitive change kinds as reject / review / allow.

        Explicit null rejects rather than interpreting as an empty safe set.
        """
        kinds = _coerce_change_kinds(change_kinds)
        if not kinds:
            return _decision(
                AuthorityVerdict.ALLOW,
                AuthorityReasonCode.ALLOWED,
                interface=self.interface,
                schema=self.schema,
                message="no sensitive change kinds declared",
            )
        always_review = sorted(
            kind.value for kind in kinds if kind in ALWAYS_HUMAN_REVIEW_KINDS
        )
        sensitive = sorted(
            kind.value for kind in kinds if kind in SENSITIVE_CHANGE_KINDS
        )
        if always_review:
            return _decision(
                AuthorityVerdict.REQUIRE_HUMAN_REVIEW,
                AuthorityReasonCode.SENSITIVE_CHANGE_REQUIRES_REVIEW,
                AuthorityReasonCode.FORBIDDEN_CHANGE_KIND,
                *always_review,
                interface=self.interface,
                schema=self.schema,
                message="sensitive change kinds require human review",
                details={"change_kinds": list(always_review)},
            )
        if sensitive:
            return _decision(
                AuthorityVerdict.REQUIRE_HUMAN_REVIEW,
                AuthorityReasonCode.SENSITIVE_CHANGE_REQUIRES_CONTRACT,
                AuthorityReasonCode.FORBIDDEN_CHANGE_KIND,
                *sensitive,
                interface=self.interface,
                schema=self.schema,
                message=(
                    "sensitive change kinds require contract verification "
                    "or human review"
                ),
                details={"change_kinds": list(sensitive)},
            )
        return _decision(
            AuthorityVerdict.ALLOW,
            AuthorityReasonCode.ALLOWED,
            interface=self.interface,
            schema=self.schema,
            message="change kinds are not classified as sensitive",
            details={"change_kinds": [kind.value for kind in kinds]},
        )

    def evaluate_claims(
        self, claims: Sequence[PatchPathClaim | Mapping[str, Any]]
    ) -> AuthorityDecision:
        """Evaluate a batch of path claims; fail closed on the first hard reject."""
        # Exact list/tuple only — sequence subclasses are rejected before any
        # overridable method (truthiness, iteration, indexing) is invoked.
        if type(claims) is not list and type(claims) is not tuple:
            raise GuiAuthorityError(
                "claims must be a sequence",
                reason_code=AuthorityReasonCode.INVALID_COLLECTION_TYPE.value,
                details={"field": "claims", "value_type": type(claims).__name__},
            )
        if len(claims) == 0:
            return _decision(
                AuthorityVerdict.REJECT,
                AuthorityReasonCode.MISSING_AUTHORITY_EVIDENCE,
                interface=self.interface,
                schema=self.schema,
                message="patch authority requires at least one path claim",
            )

        review_codes: list[str] = []
        review_details: dict[str, Any] = {"claims": []}
        for index, raw in enumerate(claims):
            claim = self._coerce_claim(raw, index)
            path_decision = self.evaluate_path(claim.path, declared=claim.declared)
            if path_decision.rejected:
                return path_decision
            kind_decision = self.evaluate_change_kinds(claim.change_kinds)
            if kind_decision.rejected:
                return kind_decision
            if kind_decision.requires_human_review:
                review_codes.extend(kind_decision.reason_codes)
                review_details["claims"].append(
                    {
                        "path": claim.path,
                        "reason_codes": list(kind_decision.reason_codes),
                    }
                )
        if review_codes:
            return _decision(
                AuthorityVerdict.REQUIRE_HUMAN_REVIEW,
                *sorted(set(review_codes)),
                interface=self.interface,
                schema=self.schema,
                message="one or more path claims require human review",
                details=review_details,
            )
        return _decision(
            AuthorityVerdict.ALLOW,
            AuthorityReasonCode.ALLOWED,
            interface=self.interface,
            schema=self.schema,
            message="all path claims are within patch authority",
            details={"claim_count": len(claims)},
        )

    def _coerce_claim(
        self, raw: PatchPathClaim | Mapping[str, Any], index: int
    ) -> PatchPathClaim:
        if type(raw) is PatchPathClaim:
            return raw
        if type(raw) is not dict:
            raise GuiAuthorityError(
                f"claims[{index}] must be a mapping",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
                details={"value_type": type(raw).__name__},
            )
        return PatchPathClaim.from_mapping(raw, index=index)


# ---------------------------------------------------------------------------
# GuiHostBoundaryPolicy@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrowserHostInput:
    """Browser-origin content presented to the host boundary.

    Fixture-only doctrine: production credentials, services, MCP tools, host
    paths, and process commands are forbidden.

    ``payload`` is retained as a recursively canonicalized exact-JSON tree.
    Each access returns a fresh deep copy so callers cannot mutate retained
    state, and accepted payloads always ``json.dumps(..., allow_nan=False)``
    round-trip through ``json.loads``.
    """

    fixture_only: bool = True
    uses_production_credentials: bool = False
    uses_production_services: bool = False
    uses_production_mcp_tools: bool = False
    uses_user_or_legal_data: bool = False
    selected_host_paths: tuple[str, ...] = ()
    selected_commands: tuple[str, ...] = ()
    selected_executables: tuple[str, ...] = ()
    _payload_json: str = field(default="{}", repr=False, compare=True)

    def __init__(
        self,
        payload: Mapping[str, Any] | None = None,
        fixture_only: bool = True,
        uses_production_credentials: bool = False,
        uses_production_services: bool = False,
        uses_production_mcp_tools: bool = False,
        uses_user_or_legal_data: bool = False,
        selected_host_paths: tuple[str, ...] = (),
        selected_commands: tuple[str, ...] = (),
        selected_executables: tuple[str, ...] = (),
    ) -> None:
        if payload is None:
            payload = {}
        if type(payload) is not dict:
            raise GuiAuthorityError(
                "payload must be a JSON object",
                reason_code=AuthorityReasonCode.INVALID_COLLECTION_TYPE.value,
                details={"field": "payload", "value_type": type(payload).__name__},
            )
        tree = _canonicalize_json_value(payload, "payload")
        if type(tree) is not dict:
            raise GuiAuthorityError(
                "payload must be a JSON object",
                reason_code=AuthorityReasonCode.INVALID_COLLECTION_TYPE.value,
                details={"field": "payload"},
            )
        # Round-trip through json to prove allow_nan=False serializability and
        # retain only exact built-in types.
        encoded = json.dumps(
            tree, allow_nan=False, separators=(",", ":"), sort_keys=True
        )
        object.__setattr__(self, "_payload_json", encoded)
        object.__setattr__(self, "fixture_only", _bool(fixture_only, "fixture_only"))
        object.__setattr__(
            self,
            "uses_production_credentials",
            _bool(uses_production_credentials, "uses_production_credentials"),
        )
        object.__setattr__(
            self,
            "uses_production_services",
            _bool(uses_production_services, "uses_production_services"),
        )
        object.__setattr__(
            self,
            "uses_production_mcp_tools",
            _bool(uses_production_mcp_tools, "uses_production_mcp_tools"),
        )
        object.__setattr__(
            self,
            "uses_user_or_legal_data",
            _bool(uses_user_or_legal_data, "uses_user_or_legal_data"),
        )
        object.__setattr__(
            self,
            "selected_host_paths",
            BrowserHostInput._coerce_string_tuple(
                selected_host_paths, "selected_host_paths", wire=False
            ),
        )
        object.__setattr__(
            self,
            "selected_commands",
            BrowserHostInput._coerce_string_tuple(
                selected_commands, "selected_commands", wire=False
            ),
        )
        object.__setattr__(
            self,
            "selected_executables",
            BrowserHostInput._coerce_string_tuple(
                selected_executables, "selected_executables", wire=False
            ),
        )

    @property
    def payload(self) -> dict[str, Any]:
        """Fresh exact-JSON deep copy of the retained canonical payload tree."""
        return json.loads(self._payload_json)

    @staticmethod
    def _coerce_string_tuple(
        value: Any, name: str, *, wire: bool
    ) -> tuple[str, ...]:
        if value is None:
            raise GuiAuthorityError(
                f"{name} must be a JSON array; null is not a collection",
                reason_code=AuthorityReasonCode.INVALID_COLLECTION_TYPE.value,
                details={"field": name},
            )
        if wire:
            sequence: Sequence[Any] = _require_json_array(value, name)
        else:
            sequence = _require_python_sequence(value, name)
        return tuple(_text(item, f"{name}[]", required=False) for item in sequence)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BrowserHostInput":
        if type(raw) is not dict:
            raise GuiAuthorityError(
                "browser_input must be a mapping",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
                details={"value_type": type(raw).__name__},
            )
        _reject_unknown(raw, _BROWSER_HOST_INPUT_KEYS, "browser_input")
        for flag in (
            "fixture_only",
            "uses_production_credentials",
            "uses_production_services",
            "uses_production_mcp_tools",
            "uses_user_or_legal_data",
        ):
            if flag in raw and raw[flag] is None:
                raise GuiAuthorityError(
                    f"{flag} must not be null when present",
                    reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
                    details={"field": flag, "value_type": "NoneType"},
                )
        if "payload" not in raw:
            payload: dict[str, Any] = {}
        elif raw["payload"] is None:
            raise GuiAuthorityError(
                "payload must be a JSON object; null is not a collection",
                reason_code=AuthorityReasonCode.INVALID_COLLECTION_TYPE.value,
                details={"field": "payload"},
            )
        else:
            payload = _require_json_object(raw["payload"], "payload")
            payload = _canonicalize_json_value(payload, "payload")

        selected_host_paths = _optional_json_array(raw, "selected_host_paths")
        selected_commands = _optional_json_array(raw, "selected_commands")
        selected_executables = _optional_json_array(raw, "selected_executables")
        return cls(
            payload=payload,
            fixture_only=_optional_mapping_bool(raw, "fixture_only", True),
            uses_production_credentials=_optional_mapping_bool(
                raw, "uses_production_credentials", False
            ),
            uses_production_services=_optional_mapping_bool(
                raw, "uses_production_services", False
            ),
            uses_production_mcp_tools=_optional_mapping_bool(
                raw, "uses_production_mcp_tools", False
            ),
            uses_user_or_legal_data=_optional_mapping_bool(
                raw, "uses_user_or_legal_data", False
            ),
            selected_host_paths=(
                ()
                if selected_host_paths is None
                else BrowserHostInput._coerce_string_tuple(
                    selected_host_paths, "selected_host_paths", wire=True
                )
            ),
            selected_commands=(
                ()
                if selected_commands is None
                else BrowserHostInput._coerce_string_tuple(
                    selected_commands, "selected_commands", wire=True
                )
            ),
            selected_executables=(
                ()
                if selected_executables is None
                else BrowserHostInput._coerce_string_tuple(
                    selected_executables, "selected_executables", wire=True
                )
            ),
        )


def _fully_decode_text(value: str) -> str:
    """Undo percent/plus encoding (including double encoding) on string values."""
    text = str(value)
    for _ in range(4):
        decoded = unquote_plus(unquote(text))
        if decoded == text:
            break
        text = decoded
    return text


def _walk_forbidden_payload_keys(
    value: Any,
    *,
    path: str = "",
    found: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    """Return (json_path, classification) for forbidden selectors.

    Operates only on exact built-in dict/list trees (already canonicalized).
    """
    hits = found if found is not None else []
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                continue
            child_path = f"{path}.{key}" if path else key
            classification = _classify_forbidden_key(key)
            if classification is not None:
                hits.append((child_path, classification))
            _walk_forbidden_payload_keys(child, path=child_path, found=hits)
    elif type(value) is list:
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            _walk_forbidden_payload_keys(child, path=child_path, found=hits)
    return hits


def _path_candidates(value: str) -> tuple[str, ...]:
    """Original and fully-decoded forms for path/command/credential inspection."""
    original = value.strip()
    decoded = _fully_decode_text(value).strip()
    if decoded == original:
        return (original,) if original else ()
    return tuple(item for item in (original, decoded) if item)


def _looks_like_host_path(value: str) -> bool:
    if type(value) is not str:
        return False
    for text in _path_candidates(value):
        if not text:
            continue
        lowered = text.lower()
        if text.startswith("/") or text.startswith("~"):
            return True
        if text.startswith("\\\\") or text.startswith("//"):
            return True
        # Single leading backslash after decode of percent-encoded UNC/Windows.
        if text.startswith("\\"):
            return True
        if _WINDOWS_PATH_RE.match(text):
            return True
        # Drive-relative forms such as C:secret (no slash after the colon).
        if _DRIVE_RELATIVE_RE.match(text):
            return True
        if lowered.startswith("file:"):
            return True
        # Relative traversal and dotted parent segments.
        if ".." in text.replace("\\", "/").split("/"):
            return True
        if _RELATIVE_TRAVERSAL_RE.search(text.replace("\\", "/")):
            # Allow ordinary single-dot path segments only when they are pure
            # relative traversal patterns (../, ./.., etc.).
            if ".." in text:
                return True
        normalized = text.replace("\\", "/")
        if normalized.startswith("../") or normalized.startswith("./../"):
            return True
        if "/../" in normalized or normalized.endswith("/.."):
            return True
        if normalized == ".." or normalized.startswith("..\\"):
            return True
    return False


def _looks_like_command(value: str) -> bool:
    if type(value) is not str:
        return False
    markers = (
        "&&",
        "||",
        ";",
        "`",
        "$(",
        "\n",
        "\t",
        "|",
        ">",
        "<",
        "sudo ",
        "rm -",
        "curl ",
        "wget ",
        "python ",
        "python3 ",
        "bash ",
        "sh ",
        "powershell ",
        "powershell.exe",
        "cmd.exe",
        "cmd /",
        "cmd.exe ",
    )
    for text in _path_candidates(value):
        if not text:
            continue
        lowered = text.lower()
        if any(marker in lowered for marker in markers):
            return True
        if _COMMAND_EXECUTABLE_RE.search(lowered):
            return True
        # Bare cmd /c and powershell -Command forms.
        if lowered.startswith("cmd ") or lowered.startswith("cmd.exe"):
            return True
        if lowered.startswith("powershell") or lowered.startswith("pwsh"):
            return True
        if "\t" in text or "|" in text or ">" in text or "<" in text:
            return True
    return False


def _looks_like_credential(value: str) -> bool:
    if type(value) is not str:
        return False
    for text in _path_candidates(value):
        if not text:
            continue
        lowered = text.lower()
        if _CREDENTIAL_VALUE_RE.match(text):
            return True
        if any(marker in lowered for marker in _CREDENTIAL_INLINE_MARKERS):
            return True
    return False


@dataclass(frozen=True)
class GuiHostBoundaryPolicy:
    """Browser-to-host choke doctrine for the optimizer.

    Interface: ``GuiHostBoundaryPolicy@1``.

    Mirrors the SwissKnife gateway invariant: browser content never selects
    host filesystem paths, process commands, or credentials.  Presentation and
    UI state are not authorization.
    """

    schema: str = GUI_HOST_BOUNDARY_POLICY_SCHEMA
    interface: str = GUI_HOST_BOUNDARY_POLICY_INTERFACE
    forbid_absolute_path_strings: bool = True
    forbid_command_like_strings: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        object.__setattr__(self, "interface", _text(self.interface, "interface"))
        # Doctrine cannot be weakened through construction: only literal True is
        # accepted.  False raises rather than silently becoming True.
        if self.forbid_absolute_path_strings is not True:
            raise GuiAuthorityError(
                "forbid_absolute_path_strings must be literal True; "
                "path value inspection executes unconditionally",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
                details={
                    "field": "forbid_absolute_path_strings",
                    "value": self.forbid_absolute_path_strings,
                },
            )
        if self.forbid_command_like_strings is not True:
            raise GuiAuthorityError(
                "forbid_command_like_strings must be literal True; "
                "command value inspection executes unconditionally",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
                details={
                    "field": "forbid_command_like_strings",
                    "value": self.forbid_command_like_strings,
                },
            )
        object.__setattr__(self, "forbid_absolute_path_strings", True)
        object.__setattr__(self, "forbid_command_like_strings", True)

    def evaluate(
        self, browser_input: BrowserHostInput | Mapping[str, Any]
    ) -> AuthorityDecision:
        """Reject browser content that attempts host path/command selection."""
        payload_input = self._coerce_input(browser_input)

        if not payload_input.fixture_only:
            return _decision(
                AuthorityVerdict.REJECT,
                AuthorityReasonCode.FIXTURE_ONLY_VIOLATION,
                interface=self.interface,
                schema=self.schema,
                message="browser optimizer inputs must be fixture-only",
            )

        production_flags = {
            "uses_production_credentials": payload_input.uses_production_credentials,
            "uses_production_services": payload_input.uses_production_services,
            "uses_production_mcp_tools": payload_input.uses_production_mcp_tools,
            "uses_user_or_legal_data": payload_input.uses_user_or_legal_data,
        }
        if any(production_flags.values()):
            return _decision(
                AuthorityVerdict.REJECT,
                AuthorityReasonCode.BROWSER_PRODUCTION_INPUT_FORBIDDEN,
                interface=self.interface,
                schema=self.schema,
                message=(
                    "production credentials, services, tools, or data are forbidden"
                ),
                details=production_flags,
            )

        if any(path for path in payload_input.selected_host_paths):
            return _decision(
                AuthorityVerdict.REJECT,
                AuthorityReasonCode.BROWSER_HOST_PATH_FORBIDDEN,
                interface=self.interface,
                schema=self.schema,
                message="browser content cannot select host paths",
                details={
                    "selected_host_paths": list(payload_input.selected_host_paths)
                },
            )

        if any(payload_input.selected_commands) or any(
            payload_input.selected_executables
        ):
            return _decision(
                AuthorityVerdict.REJECT,
                AuthorityReasonCode.BROWSER_COMMAND_FORBIDDEN,
                interface=self.interface,
                schema=self.schema,
                message="browser content cannot select host commands",
                details={
                    "selected_commands": list(payload_input.selected_commands),
                    "selected_executables": list(payload_input.selected_executables),
                },
            )

        forbidden_keys = _walk_forbidden_payload_keys(payload_input.payload)
        if forbidden_keys:
            classifications = {item[1] for item in forbidden_keys}
            if "credential" in classifications:
                reason = AuthorityReasonCode.BROWSER_CREDENTIAL_FORBIDDEN
            elif "command" in classifications:
                reason = AuthorityReasonCode.BROWSER_COMMAND_FORBIDDEN
            else:
                reason = AuthorityReasonCode.BROWSER_HOST_PATH_FORBIDDEN
            return _decision(
                AuthorityVerdict.REJECT,
                reason,
                interface=self.interface,
                schema=self.schema,
                message="browser payload contains forbidden host-boundary keys",
                details={
                    "forbidden_keys": [item[0] for item in forbidden_keys],
                    "classifications": sorted(classifications),
                },
            )

        # Path, command, and credential value inspection execute unconditionally.
        string_hits = self._scan_string_values(payload_input.payload)
        if string_hits:
            return string_hits

        return _decision(
            AuthorityVerdict.ALLOW,
            AuthorityReasonCode.ALLOWED,
            interface=self.interface,
            schema=self.schema,
            message="browser input respects the host boundary",
        )

    def _scan_string_values(
        self, value: Any, *, path: str = ""
    ) -> AuthorityDecision | None:
        # Operate only on the same recursively canonicalized retained tree.
        if type(value) is dict:
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else key
                hit = self._scan_string_values(child, path=child_path)
                if hit is not None:
                    return hit
            return None
        if type(value) is list:
            for index, child in enumerate(value):
                hit = self._scan_string_values(child, path=f"{path}[{index}]")
                if hit is not None:
                    return hit
            return None
        if type(value) is not str:
            return None
        if _looks_like_host_path(value):
            return _decision(
                AuthorityVerdict.REJECT,
                AuthorityReasonCode.BROWSER_HOST_PATH_FORBIDDEN,
                interface=self.interface,
                schema=self.schema,
                message="browser payload must not embed host filesystem paths",
                details={"path": path, "value": value},
            )
        if _looks_like_command(value):
            return _decision(
                AuthorityVerdict.REJECT,
                AuthorityReasonCode.BROWSER_COMMAND_FORBIDDEN,
                interface=self.interface,
                schema=self.schema,
                message="browser payload must not embed host process commands",
                details={"path": path, "value": value},
            )
        if _looks_like_credential(value):
            return _decision(
                AuthorityVerdict.REJECT,
                AuthorityReasonCode.BROWSER_CREDENTIAL_FORBIDDEN,
                interface=self.interface,
                schema=self.schema,
                message="browser payload must not embed credentials",
                details={"path": path, "value": value},
            )
        return None

    def _coerce_input(
        self, browser_input: BrowserHostInput | Mapping[str, Any]
    ) -> BrowserHostInput:
        if type(browser_input) is BrowserHostInput:
            return browser_input
        if type(browser_input) is not dict:
            raise GuiAuthorityError(
                "browser_input must be a BrowserHostInput or mapping",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
                details={"value_type": type(browser_input).__name__},
            )
        return BrowserHostInput.from_mapping(browser_input)


# ---------------------------------------------------------------------------
# GuiAcceptanceAuthority@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuthorityEvidence:
    """One piece of evidence offered to justify automatic acceptance.

    Identity is mandatory.  Evidence used to authorize an intended action must
    also be current and bound to that exact action and canonical argument digest.
    """

    kind: AuthorityEvidenceKind
    valid: bool
    evidence_id: str = ""
    binds_action_id: str = ""
    binds_argument_digest: str = ""
    policy_decision_id: str = ""
    policy_fresh: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        # Direct constructors may pass typed Enum members; wire path uses
        # from_mapping which forces exact-string kinds first.
        if type(self.kind) is AuthorityEvidenceKind:
            kind = self.kind
        else:
            kind = _as_evidence_kind(self.kind, wire=False)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "valid", _bool(self.valid, "valid"))
        if self.evidence_id is None:
            raise GuiAuthorityError(
                "authority evidence requires a nonempty identity",
                reason_code=AuthorityReasonCode.EVIDENCE_IDENTITY_REQUIRED.value,
                details={"field": "evidence_id"},
            )
        evidence_id = _identifier_field(
            self.evidence_id,
            "evidence_id",
            allow_empty=False,
        )
        object.__setattr__(self, "evidence_id", evidence_id)
        # Present null rejects for optional binding/identity scalars.
        for field_name, current in (
            ("binds_action_id", self.binds_action_id),
            ("binds_argument_digest", self.binds_argument_digest),
            ("policy_decision_id", self.policy_decision_id),
            ("notes", self.notes),
        ):
            if current is None:
                raise GuiAuthorityError(
                    f"{field_name} must not be null when present",
                    reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
                    details={"field": field_name, "value_type": "NoneType"},
                )
        object.__setattr__(
            self,
            "binds_action_id",
            _identifier_field(
                self.binds_action_id,
                "binds_action_id",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "binds_argument_digest",
            _digest_field(
                self.binds_argument_digest,
                "binds_argument_digest",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "policy_decision_id",
            _identifier_field(
                self.policy_decision_id,
                "policy_decision_id",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self, "policy_fresh", _bool(self.policy_fresh, "policy_fresh")
        )
        notes = _exact_str(self.notes, "notes")
        if "\x00" in notes:
            raise GuiAuthorityError(
                "notes must not contain NUL",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
                details={"field": "notes"},
            )
        object.__setattr__(self, "notes", notes)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, index: int = 0) -> "AuthorityEvidence":
        # Exact-dict check before any attribute introspection so subclass-
        # controlled RuntimeError is a test failure rather than accepted rejection.
        if type(raw) is not dict:
            raise GuiAuthorityError(
                f"evidence[{index}] must be a JSON object",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_EVIDENCE.value,
                details={"value_type": type(raw).__name__},
            )
        _reject_unknown(raw, _AUTHORITY_EVIDENCE_KEYS, f"evidence[{index}]")
        if "valid" not in raw:
            raise GuiAuthorityError(
                f"evidence[{index}].valid is required",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_EVIDENCE.value,
            )
        if raw["valid"] is None:
            raise GuiAuthorityError(
                f"evidence[{index}].valid must not be null when present",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
                details={"field": "valid", "value_type": "NoneType"},
            )
        if "kind" not in raw:
            raise GuiAuthorityError(
                f"evidence[{index}].kind is required",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_EVIDENCE.value,
            )
        if raw["kind"] is None:
            raise GuiAuthorityError(
                f"evidence[{index}].kind must not be null when present",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
                details={"field": "kind", "value_type": "NoneType"},
            )
        # Wire kinds are exact built-in strings only (Enums reject).
        kind = _as_evidence_kind(raw["kind"], wire=True)
        if "evidence_id" not in raw:
            raise GuiAuthorityError(
                "authority evidence requires a nonempty identity",
                reason_code=AuthorityReasonCode.EVIDENCE_IDENTITY_REQUIRED.value,
                details={"field": "evidence_id"},
            )
        if raw["evidence_id"] is None:
            raise GuiAuthorityError(
                "evidence_id must not be null when present",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
                details={"field": "evidence_id", "value_type": "NoneType"},
            )
        if "policy_fresh" in raw and raw["policy_fresh"] is None:
            raise GuiAuthorityError(
                "policy_fresh must not be null when present",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
                details={"field": "policy_fresh", "value_type": "NoneType"},
            )
        return cls(
            kind=kind,
            valid=_bool(raw["valid"], f"evidence[{index}].valid"),
            evidence_id=raw["evidence_id"],
            binds_action_id=_optional_identifier(raw, "binds_action_id"),
            binds_argument_digest=_optional_digest(raw, "binds_argument_digest"),
            policy_decision_id=_optional_identifier(raw, "policy_decision_id"),
            policy_fresh=_optional_mapping_bool(raw, "policy_fresh", False),
            notes=_optional_notes(raw, "notes"),
        )


@dataclass(frozen=True)
class AcceptanceAuthorityRequest:
    """Inputs for automatic-acceptance evaluation.

    UI visibility/enabled state and browser policy output are recorded only so
    the authority can refuse to treat them as authorization.  Caller-supplied
    ``policy_decision_id`` / ``policy_fresh`` are recorded for freshness checks
    but never grant authority without bound current evidence.
    """

    intended_action_id: str = ""
    intended_argument_digest: str = ""
    ui_visible: bool = False
    ui_enabled: bool = False
    browser_policy_outcome: str = ""
    browser_policy_authoritative_claim: bool = False
    policy_decision_id: str = ""
    policy_fresh: bool = False
    confirmation_required: bool = False
    confirmation_action_id: str = ""
    confirmation_argument_digest: str = ""
    confirmation_granted: bool = False
    change_kinds: tuple[ForbiddenChangeKind, ...] = ()
    evidence: tuple[AuthorityEvidence, ...] = ()
    accessibility_regression: bool = False
    security_regression: bool = False
    host_boundary_decision: AuthorityDecision | None = None
    patch_authority_decision: AuthorityDecision | None = None

    def __post_init__(self) -> None:
        # Present null rejects for optional identity/digest/outcome scalars.
        for field_name, current in (
            ("intended_action_id", self.intended_action_id),
            ("intended_argument_digest", self.intended_argument_digest),
            ("browser_policy_outcome", self.browser_policy_outcome),
            ("policy_decision_id", self.policy_decision_id),
            ("confirmation_action_id", self.confirmation_action_id),
            ("confirmation_argument_digest", self.confirmation_argument_digest),
        ):
            if current is None:
                raise GuiAuthorityError(
                    f"{field_name} must not be null when present",
                    reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
                    details={"field": field_name, "value_type": "NoneType"},
                )
        object.__setattr__(
            self,
            "intended_action_id",
            _identifier_field(
                self.intended_action_id,
                "intended_action_id",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "intended_argument_digest",
            _digest_field(
                self.intended_argument_digest,
                "intended_argument_digest",
                allow_empty=True,
            ),
        )
        object.__setattr__(self, "ui_visible", _bool(self.ui_visible, "ui_visible"))
        object.__setattr__(self, "ui_enabled", _bool(self.ui_enabled, "ui_enabled"))
        object.__setattr__(
            self,
            "browser_policy_outcome",
            _identifier_field(
                self.browser_policy_outcome,
                "browser_policy_outcome",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "browser_policy_authoritative_claim",
            _bool(
                self.browser_policy_authoritative_claim,
                "browser_policy_authoritative_claim",
            ),
        )
        object.__setattr__(
            self,
            "policy_decision_id",
            _identifier_field(
                self.policy_decision_id,
                "policy_decision_id",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self, "policy_fresh", _bool(self.policy_fresh, "policy_fresh")
        )
        object.__setattr__(
            self,
            "confirmation_required",
            _bool(self.confirmation_required, "confirmation_required"),
        )
        object.__setattr__(
            self,
            "confirmation_action_id",
            _identifier_field(
                self.confirmation_action_id,
                "confirmation_action_id",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "confirmation_argument_digest",
            _digest_field(
                self.confirmation_argument_digest,
                "confirmation_argument_digest",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "confirmation_granted",
            _bool(self.confirmation_granted, "confirmation_granted"),
        )
        if self.change_kinds is None:
            raise GuiAuthorityError(
                "change_kinds must be a JSON array; null is not a collection",
                reason_code=AuthorityReasonCode.INVALID_COLLECTION_TYPE.value,
                details={"field": "change_kinds"},
            )
        # Direct constructors may pass typed Enum members; wire uses from_mapping.
        kinds = tuple(
            _as_change_kind(kind, wire=False)
            for kind in _require_python_sequence(self.change_kinds, "change_kinds")
        )
        object.__setattr__(self, "change_kinds", kinds)
        if self.evidence is None:
            raise GuiAuthorityError(
                "evidence must be a JSON array; null is not a collection",
                reason_code=AuthorityReasonCode.INVALID_COLLECTION_TYPE.value,
                details={"field": "evidence"},
            )
        evidence_items: list[AuthorityEvidence] = []
        for index, item in enumerate(
            _require_python_sequence(self.evidence, "evidence")
        ):
            if type(item) is AuthorityEvidence:
                evidence_items.append(item)
            elif type(item) is dict:
                evidence_items.append(AuthorityEvidence.from_mapping(item, index=index))
            else:
                raise GuiAuthorityError(
                    "evidence items must be AuthorityEvidence or exact JSON objects",
                    reason_code=AuthorityReasonCode.INVALID_AUTHORITY_EVIDENCE.value,
                    details={"value_type": type(item).__name__},
                )
        object.__setattr__(self, "evidence", tuple(evidence_items))
        object.__setattr__(
            self,
            "accessibility_regression",
            _bool(self.accessibility_regression, "accessibility_regression"),
        )
        object.__setattr__(
            self,
            "security_regression",
            _bool(self.security_regression, "security_regression"),
        )
        if self.host_boundary_decision is not None and type(
            self.host_boundary_decision
        ) is not AuthorityDecision:
            raise GuiAuthorityError(
                "host_boundary_decision must be an AuthorityDecision",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            )
        if self.patch_authority_decision is not None and type(
            self.patch_authority_decision
        ) is not AuthorityDecision:
            raise GuiAuthorityError(
                "patch_authority_decision must be an AuthorityDecision",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AcceptanceAuthorityRequest":
        if type(raw) is not dict:
            raise GuiAuthorityError(
                "request must be a mapping",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
                details={"value_type": type(raw).__name__},
            )
        _reject_unknown(raw, _ACCEPTANCE_REQUEST_KEYS, "acceptance_request")
        for flag in (
            "ui_visible",
            "ui_enabled",
            "browser_policy_authoritative_claim",
            "policy_fresh",
            "confirmation_required",
            "confirmation_granted",
            "accessibility_regression",
            "security_regression",
        ):
            if flag in raw and raw[flag] is None:
                raise GuiAuthorityError(
                    f"{flag} must not be null when present",
                    reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
                    details={"field": flag, "value_type": "NoneType"},
                )
        evidence_raw = _optional_json_array(raw, "evidence")
        evidence_items: tuple[AuthorityEvidence, ...]
        if evidence_raw is None:
            evidence_items = ()
        else:
            parsed: list[AuthorityEvidence] = []
            for index, item in enumerate(evidence_raw):
                # Every evidence-array wire entry is an exact JSON object.
                # Model/dataclass instances and dict subclasses reject before
                # attribute introspection.
                if type(item) is not dict:
                    raise GuiAuthorityError(
                        f"evidence[{index}] must be a JSON object",
                        reason_code=AuthorityReasonCode.INVALID_AUTHORITY_EVIDENCE.value,
                        details={"value_type": type(item).__name__},
                    )
                parsed.append(AuthorityEvidence.from_mapping(item, index=index))
            evidence_items = tuple(parsed)
        return cls(
            intended_action_id=_optional_identifier(raw, "intended_action_id"),
            intended_argument_digest=_optional_digest(raw, "intended_argument_digest"),
            ui_visible=_optional_mapping_bool(raw, "ui_visible", False),
            ui_enabled=_optional_mapping_bool(raw, "ui_enabled", False),
            browser_policy_outcome=_optional_identifier(raw, "browser_policy_outcome"),
            browser_policy_authoritative_claim=_optional_mapping_bool(
                raw, "browser_policy_authoritative_claim", False
            ),
            policy_decision_id=_optional_identifier(raw, "policy_decision_id"),
            policy_fresh=_optional_mapping_bool(raw, "policy_fresh", False),
            confirmation_required=_optional_mapping_bool(
                raw, "confirmation_required", False
            ),
            confirmation_action_id=_optional_identifier(raw, "confirmation_action_id"),
            confirmation_argument_digest=_optional_digest(
                raw, "confirmation_argument_digest"
            ),
            confirmation_granted=_optional_mapping_bool(
                raw, "confirmation_granted", False
            ),
            change_kinds=_optional_change_kinds(raw),
            evidence=evidence_items,
            accessibility_regression=_optional_mapping_bool(
                raw, "accessibility_regression", False
            ),
            security_regression=_optional_mapping_bool(
                raw, "security_regression", False
            ),
            host_boundary_decision=raw.get("host_boundary_decision"),
            patch_authority_decision=raw.get("patch_authority_decision"),
        )


@dataclass(frozen=True)
class GuiAcceptanceAuthority:
    """Evidence doctrine for automatic acceptance.

    Interface: ``GuiAcceptanceAuthority@1``.

    UI state cannot synthesize authorization.  Browser policy output is never
    authoritative.  Sensitive changes require contract verification or human
    review.  Missing or invalid authority evidence rejects safely.  Scope
    declarations never grant host authority by themselves.  Caller-supplied
    policy fields never authorize without bound current evidence.
    """

    schema: str = GUI_ACCEPTANCE_AUTHORITY_SCHEMA
    interface: str = GUI_ACCEPTANCE_AUTHORITY_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        object.__setattr__(self, "interface", _text(self.interface, "interface"))

    def evaluate(
        self, request: AcceptanceAuthorityRequest | Mapping[str, Any]
    ) -> AuthorityDecision:
        """Decide whether automatic acceptance is permitted."""
        req = self._coerce_request(request)

        if req.host_boundary_decision is not None and not req.host_boundary_decision.allowed:
            return _decision(
                AuthorityVerdict.REJECT,
                *req.host_boundary_decision.reason_codes,
                interface=self.interface,
                schema=self.schema,
                message="host-boundary rejection blocks acceptance",
                details={"host_boundary": req.host_boundary_decision.to_dict()},
            )

        if (
            req.patch_authority_decision is not None
            and req.patch_authority_decision.rejected
        ):
            return _decision(
                AuthorityVerdict.REJECT,
                *req.patch_authority_decision.reason_codes,
                interface=self.interface,
                schema=self.schema,
                message="patch-authority rejection blocks acceptance",
                details={"patch_authority": req.patch_authority_decision.to_dict()},
            )

        if req.accessibility_regression:
            return _decision(
                AuthorityVerdict.REJECT,
                AuthorityReasonCode.ACCESSIBILITY_REGRESSION,
                interface=self.interface,
                schema=self.schema,
                message="accessibility regressions block automatic acceptance",
            )

        if req.security_regression:
            return _decision(
                AuthorityVerdict.REJECT,
                AuthorityReasonCode.SECURITY_REGRESSION,
                interface=self.interface,
                schema=self.schema,
                message="security regressions block automatic acceptance",
            )

        # UI visibility / enabled state never authorizes.
        if (req.ui_visible or req.ui_enabled) and not self._has_valid_host_authority(
            req
        ):
            if not req.evidence:
                return _decision(
                    AuthorityVerdict.REJECT,
                    AuthorityReasonCode.UI_STATE_NOT_AUTHORIZATION,
                    AuthorityReasonCode.MISSING_AUTHORITY_EVIDENCE,
                    interface=self.interface,
                    schema=self.schema,
                    message=(
                        "UI visibility/enabled state cannot synthesize authorization"
                    ),
                    details={
                        "ui_visible": req.ui_visible,
                        "ui_enabled": req.ui_enabled,
                    },
                )

        if req.browser_policy_authoritative_claim:
            return _decision(
                AuthorityVerdict.REJECT,
                AuthorityReasonCode.BROWSER_POLICY_NOT_AUTHORITATIVE,
                interface=self.interface,
                schema=self.schema,
                message="browser policy output is never authoritative",
                details={"browser_policy_outcome": req.browser_policy_outcome},
            )

        # Caller-supplied policy fields alone never authorize; a stale caller
        # policy id still rejects when present without fresh evidence.
        if req.policy_decision_id and not req.policy_fresh:
            return _decision(
                AuthorityVerdict.REJECT,
                AuthorityReasonCode.STALE_POLICY_DECISION,
                AuthorityReasonCode.CALLER_POLICY_NOT_AUTHORITY,
                interface=self.interface,
                schema=self.schema,
                message="a stale policy decision cannot authorize the current action",
                details={"policy_decision_id": req.policy_decision_id},
            )

        if req.confirmation_required:
            if not req.confirmation_granted:
                return _decision(
                    AuthorityVerdict.REJECT,
                    AuthorityReasonCode.CONFIRMATION_REQUIRED,
                    interface=self.interface,
                    schema=self.schema,
                    message="destructive/sensitive action requires exact confirmation",
                )
            if not self._exact_confirmation_binding(req):
                return _decision(
                    AuthorityVerdict.REJECT,
                    AuthorityReasonCode.CONFIRMATION_BINDING_MISMATCH,
                    interface=self.interface,
                    schema=self.schema,
                    message="confirmation must bind the exact action and arguments",
                    details={
                        "intended_action_id": req.intended_action_id,
                        "confirmation_action_id": req.confirmation_action_id,
                        "intended_argument_digest": req.intended_argument_digest,
                        "confirmation_argument_digest": (
                            req.confirmation_argument_digest
                        ),
                    },
                )

        invalid_evidence = [
            item.evidence_id or item.kind.value
            for item in req.evidence
            if not item.valid
        ]
        if invalid_evidence:
            return _decision(
                AuthorityVerdict.REJECT,
                AuthorityReasonCode.INVALID_AUTHORITY_EVIDENCE,
                interface=self.interface,
                schema=self.schema,
                message="invalid authority evidence rejects safely",
                details={"invalid_evidence": invalid_evidence},
            )

        # Reject valid-looking evidence that is stale or wrongly bound when it
        # claims to authorize the intended action.
        binding_failure = self._evidence_binding_failure(req)
        if binding_failure is not None:
            return binding_failure

        # Scope-only packages never satisfy host authority.
        if (
            req.intended_action_id
            and not self._has_valid_host_authority(req)
            and self._only_scope_evidence(req)
        ):
            return _decision(
                AuthorityVerdict.REJECT,
                AuthorityReasonCode.SCOPE_DECLARATION_NOT_AUTHORITY,
                AuthorityReasonCode.MISSING_AUTHORITY_EVIDENCE,
                interface=self.interface,
                schema=self.schema,
                message="a scope declaration alone is never host authority",
                details={"intended_action_id": req.intended_action_id},
            )

        sensitive = tuple(
            kind for kind in req.change_kinds if kind in SENSITIVE_CHANGE_KINDS
        )
        if sensitive:
            always_review = tuple(
                kind for kind in sensitive if kind in ALWAYS_HUMAN_REVIEW_KINDS
            )
            has_human = self._has_valid_evidence(
                req, AuthorityEvidenceKind.HUMAN_REVIEW
            )
            has_contract = self._has_valid_evidence(
                req, AuthorityEvidenceKind.CONTRACT_VERIFICATION
            )
            if always_review and not has_human:
                return _decision(
                    AuthorityVerdict.REQUIRE_HUMAN_REVIEW,
                    AuthorityReasonCode.SENSITIVE_CHANGE_REQUIRES_REVIEW,
                    *[kind.value for kind in always_review],
                    interface=self.interface,
                    schema=self.schema,
                    message=(
                        "sensitive change kinds require human review receipts"
                    ),
                    details={"change_kinds": [kind.value for kind in always_review]},
                )
            if not has_human and not has_contract:
                return _decision(
                    AuthorityVerdict.REQUIRE_HUMAN_REVIEW,
                    AuthorityReasonCode.SENSITIVE_CHANGE_REQUIRES_CONTRACT,
                    *[kind.value for kind in sensitive],
                    interface=self.interface,
                    schema=self.schema,
                    message=(
                        "sensitive changes require contract verification "
                        "or human review"
                    ),
                    details={"change_kinds": [kind.value for kind in sensitive]},
                )

        if (
            req.patch_authority_decision is not None
            and req.patch_authority_decision.requires_human_review
            and not self._has_valid_evidence(req, AuthorityEvidenceKind.HUMAN_REVIEW)
        ):
            return _decision(
                AuthorityVerdict.REQUIRE_HUMAN_REVIEW,
                *req.patch_authority_decision.reason_codes,
                interface=self.interface,
                schema=self.schema,
                message="patch authority requires human review before acceptance",
                details={"patch_authority": req.patch_authority_decision.to_dict()},
            )

        if req.intended_action_id and not self._has_valid_host_authority(req):
            # Surface the prior false-green: caller policy fields look fresh
            # but carry no bound evidence.
            if req.policy_decision_id or req.policy_fresh:
                return _decision(
                    AuthorityVerdict.REJECT,
                    AuthorityReasonCode.CALLER_POLICY_NOT_AUTHORITY,
                    AuthorityReasonCode.MISSING_AUTHORITY_EVIDENCE,
                    interface=self.interface,
                    schema=self.schema,
                    message=(
                        "caller-supplied policy_decision_id/policy_fresh have no "
                        "authority without current evidence bound to the exact "
                        "intended action and canonical nonempty argument digest"
                    ),
                    details={
                        "intended_action_id": req.intended_action_id,
                        "policy_decision_id": req.policy_decision_id,
                        "policy_fresh": req.policy_fresh,
                    },
                )
            return _decision(
                AuthorityVerdict.REJECT,
                AuthorityReasonCode.MISSING_AUTHORITY_EVIDENCE,
                interface=self.interface,
                schema=self.schema,
                message="missing authority evidence rejects safely",
                details={"intended_action_id": req.intended_action_id},
            )

        return _decision(
            AuthorityVerdict.ALLOW,
            AuthorityReasonCode.ALLOWED,
            interface=self.interface,
            schema=self.schema,
            message="acceptance authority requirements are satisfied",
        )

    def _only_scope_evidence(self, request: AcceptanceAuthorityRequest) -> bool:
        if not request.evidence:
            return False
        return all(
            item.kind is AuthorityEvidenceKind.SCOPE_DECLARATION
            for item in request.evidence
        )

    def _exact_confirmation_binding(
        self, request: AcceptanceAuthorityRequest
    ) -> bool:
        """Exact action + nonempty canonical argument digest confirmation."""
        if not request.intended_action_id or not request.intended_argument_digest:
            return False
        if request.confirmation_action_id != request.intended_action_id:
            return False
        if request.confirmation_argument_digest != request.intended_argument_digest:
            return False
        if not request.confirmation_argument_digest:
            return False
        return True

    def _evidence_is_current(
        self,
        item: AuthorityEvidence,
        request: AcceptanceAuthorityRequest,
    ) -> bool:
        if item.kind is AuthorityEvidenceKind.HOST_POLICY_REEVALUATION:
            if not item.policy_fresh:
                return False
            if (
                item.policy_decision_id
                and request.policy_decision_id
                and item.policy_decision_id != request.policy_decision_id
            ):
                return False
            if request.policy_decision_id and not request.policy_fresh:
                return False
        return True

    def _evidence_is_bound(
        self,
        item: AuthorityEvidence,
        request: AcceptanceAuthorityRequest,
    ) -> bool:
        """Evidence that authorizes must bind exact action + nonempty digest."""
        if not request.intended_action_id or not item.binds_action_id:
            return False
        if item.binds_action_id != request.intended_action_id:
            return False
        if not request.intended_argument_digest or not item.binds_argument_digest:
            return False
        if item.binds_argument_digest != request.intended_argument_digest:
            return False
        return True

    def _evidence_can_authorize(
        self,
        item: AuthorityEvidence,
        request: AcceptanceAuthorityRequest,
    ) -> bool:
        if not item.valid:
            return False
        if not item.evidence_id:
            return False
        if item.kind not in HOST_AUTHORIZING_EVIDENCE_KINDS:
            return False
        if not self._evidence_is_current(item, request):
            return False
        if not self._evidence_is_bound(item, request):
            return False
        return True

    def _evidence_binding_failure(
        self, request: AcceptanceAuthorityRequest
    ) -> AuthorityDecision | None:
        """Reject evidence that claims authorization but is stale or unbound."""
        for item in request.evidence:
            if not item.valid:
                continue
            if item.kind not in HOST_AUTHORIZING_EVIDENCE_KINDS:
                continue
            # Scrutinize evidence that attempts to bind or re-evaluate policy.
            claims_authorization = bool(
                item.binds_action_id
                or item.binds_argument_digest
                or item.kind is AuthorityEvidenceKind.HOST_POLICY_REEVALUATION
                or item.kind is AuthorityEvidenceKind.EXACT_CONFIRMATION_BINDING
            )
            if not claims_authorization:
                continue
            if not self._evidence_is_current(item, request):
                return _decision(
                    AuthorityVerdict.REJECT,
                    AuthorityReasonCode.EVIDENCE_NOT_CURRENT,
                    AuthorityReasonCode.STALE_POLICY_DECISION,
                    interface=self.interface,
                    schema=self.schema,
                    message=(
                        "authority evidence used to authorize must be current"
                    ),
                    details={
                        "evidence_id": item.evidence_id,
                        "kind": item.kind.value,
                    },
                )
            if request.intended_action_id and not self._evidence_is_bound(
                item, request
            ):
                empty_digest = (
                    not request.intended_argument_digest
                    or not item.binds_argument_digest
                )
                reason = (
                    AuthorityReasonCode.EMPTY_ARGUMENT_DIGEST
                    if empty_digest
                    else AuthorityReasonCode.EVIDENCE_BINDING_MISMATCH
                )
                return _decision(
                    AuthorityVerdict.REJECT,
                    reason,
                    AuthorityReasonCode.EVIDENCE_BINDING_MISMATCH,
                    interface=self.interface,
                    schema=self.schema,
                    message=(
                        "authority evidence must bind the exact action and "
                        "canonical nonempty argument digest"
                    ),
                    details={
                        "evidence_id": item.evidence_id,
                        "binds_action_id": item.binds_action_id,
                        "intended_action_id": request.intended_action_id,
                        "binds_argument_digest": item.binds_argument_digest,
                        "intended_argument_digest": request.intended_argument_digest,
                    },
                )
        return None

    def _has_valid_evidence(
        self,
        request: AcceptanceAuthorityRequest,
        kind: AuthorityEvidenceKind,
    ) -> bool:
        return any(
            item.kind is kind and self._evidence_can_authorize(item, request)
            for item in request.evidence
        )

    def _has_valid_host_authority(self, request: AcceptanceAuthorityRequest) -> bool:
        """Host re-evaluation, exact confirmation, contract, or human review.

        Caller-supplied ``policy_decision_id`` / ``policy_fresh`` never grant
        authority by themselves.  Scope declarations and fixture markers never
        do either.  Evidence only counts when current and bound to the intended
        action and a canonical nonempty argument digest.
        """
        for item in request.evidence:
            if self._evidence_can_authorize(item, request):
                return True
        if (
            request.confirmation_required
            and request.confirmation_granted
            and self._exact_confirmation_binding(request)
        ):
            return True
        return False

    def _coerce_request(
        self, request: AcceptanceAuthorityRequest | Mapping[str, Any]
    ) -> AcceptanceAuthorityRequest:
        if type(request) is AcceptanceAuthorityRequest:
            return request
        if type(request) is not dict:
            raise GuiAuthorityError(
                "request must be an AcceptanceAuthorityRequest or mapping",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
                details={"value_type": type(request).__name__},
            )
        return AcceptanceAuthorityRequest.from_mapping(request)


# ---------------------------------------------------------------------------
# Combined optimizer security authority facade
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuiOptimizerSecurityAuthority:
    """Combined fail-closed wrapper over the three VGO-009 interfaces."""

    patch: GuiPatchAuthority = field(default_factory=GuiPatchAuthority)
    host_boundary: GuiHostBoundaryPolicy = field(
        default_factory=GuiHostBoundaryPolicy
    )
    acceptance: GuiAcceptanceAuthority = field(
        default_factory=GuiAcceptanceAuthority
    )

    def evaluate_patch_claims(
        self, claims: Sequence[PatchPathClaim | Mapping[str, Any]]
    ) -> AuthorityDecision:
        return self.patch.evaluate_claims(claims)

    def evaluate_browser_input(
        self, browser_input: BrowserHostInput | Mapping[str, Any]
    ) -> AuthorityDecision:
        return self.host_boundary.evaluate(browser_input)

    def evaluate_acceptance(
        self, request: AcceptanceAuthorityRequest | Mapping[str, Any]
    ) -> AuthorityDecision:
        return self.acceptance.evaluate(request)

    def evaluate_proposal(
        self,
        *,
        claims: Sequence[PatchPathClaim | Mapping[str, Any]],
        browser_input: BrowserHostInput | Mapping[str, Any] | None = None,
        acceptance: AcceptanceAuthorityRequest | Mapping[str, Any] | None = None,
    ) -> AuthorityDecision:
        """Evaluate patch, optional host boundary, then acceptance in order.

        Claim-derived change kinds and computed patch/host decisions always
        override caller-supplied acceptance fields.
        """
        patch_decision = self.patch.evaluate_claims(claims)
        if patch_decision.rejected:
            return patch_decision

        host_decision: AuthorityDecision | None = None
        if browser_input is not None:
            host_decision = self.host_boundary.evaluate(browser_input)
            if host_decision.rejected:
                return host_decision

        if acceptance is None:
            if patch_decision.requires_human_review:
                return patch_decision
            return _decision(
                AuthorityVerdict.ALLOW,
                AuthorityReasonCode.ALLOWED,
                interface=GUI_PATCH_AUTHORITY_INTERFACE,
                message="patch and host-boundary checks passed",
                details={
                    "patch": patch_decision.to_dict(),
                    "host_boundary": (
                        host_decision.to_dict() if host_decision else None
                    ),
                },
            )

        claim_kinds = _claim_change_kinds(claims)

        if type(acceptance) is AcceptanceAuthorityRequest:
            # Claim-derived kinds cannot be stripped by acceptance input.
            merged_kinds = tuple(
                dict.fromkeys((*claim_kinds, *acceptance.change_kinds))
            )
            acceptance_request = AcceptanceAuthorityRequest(
                intended_action_id=acceptance.intended_action_id,
                intended_argument_digest=acceptance.intended_argument_digest,
                ui_visible=acceptance.ui_visible,
                ui_enabled=acceptance.ui_enabled,
                browser_policy_outcome=acceptance.browser_policy_outcome,
                browser_policy_authoritative_claim=(
                    acceptance.browser_policy_authoritative_claim
                ),
                policy_decision_id=acceptance.policy_decision_id,
                policy_fresh=acceptance.policy_fresh,
                confirmation_required=acceptance.confirmation_required,
                confirmation_action_id=acceptance.confirmation_action_id,
                confirmation_argument_digest=(
                    acceptance.confirmation_argument_digest
                ),
                confirmation_granted=acceptance.confirmation_granted,
                change_kinds=merged_kinds,
                evidence=acceptance.evidence,
                accessibility_regression=acceptance.accessibility_regression,
                security_regression=acceptance.security_regression,
                # Computed decisions always win over caller-supplied values.
                host_boundary_decision=host_decision,
                patch_authority_decision=patch_decision,
            )
        else:
            payload = dict(acceptance)
            # Force computed decisions; never setdefault (caller override vector).
            payload["host_boundary_decision"] = host_decision
            payload["patch_authority_decision"] = patch_decision
            if "change_kinds" in payload:
                acceptance_kinds = _coerce_change_kinds(
                    payload.get("change_kinds"), wire=True
                )
            else:
                acceptance_kinds = ()
            payload["change_kinds"] = [
                kind.value
                for kind in dict.fromkeys((*claim_kinds, *acceptance_kinds))
            ]
            acceptance_request = self.acceptance._coerce_request(payload)

        return self.acceptance.evaluate(acceptance_request)


def default_security_authority() -> GuiOptimizerSecurityAuthority:
    """Return the default fail-closed optimizer security authority."""
    return GuiOptimizerSecurityAuthority()


__all__ = (
    "ALWAYS_HUMAN_REVIEW_KINDS",
    "AcceptanceAuthorityRequest",
    "AuthorityDecision",
    "AuthorityEvidence",
    "AuthorityEvidenceKind",
    "AuthorityReasonCode",
    "AuthorityVerdict",
    "BrowserHostInput",
    "DEFAULT_ALLOWED_ROOTS",
    "DEFAULT_FORBIDDEN_PATH_PARTS",
    "FORBIDDEN_BROWSER_COMMAND_FIELDS",
    "FORBIDDEN_BROWSER_CREDENTIAL_FIELDS",
    "FORBIDDEN_BROWSER_PATH_FIELDS",
    "FORBIDDEN_BROWSER_PAYLOAD_KEYS",
    "ForbiddenChangeKind",
    "GUI_ACCEPTANCE_AUTHORITY_INTERFACE",
    "GUI_ACCEPTANCE_AUTHORITY_SCHEMA",
    "GUI_AUTHORITY_DECISION_SCHEMA",
    "GUI_HOST_BOUNDARY_POLICY_INTERFACE",
    "GUI_HOST_BOUNDARY_POLICY_SCHEMA",
    "GUI_PATCH_AUTHORITY_INTERFACE",
    "GUI_PATCH_AUTHORITY_SCHEMA",
    "HOST_AUTHORIZING_EVIDENCE_KINDS",
    "GuiAcceptanceAuthority",
    "GuiAuthorityError",
    "GuiHostBoundaryPolicy",
    "GuiOptimizerSecurityAuthority",
    "GuiPatchAuthority",
    "PatchPathClaim",
    "SENSITIVE_CHANGE_KINDS",
    "default_security_authority",
    "path_has_forbidden_segment",
    "path_under_allowed_roots",
)
