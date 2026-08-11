"""Fail-closed patch and browser-host security authority for VerifiedGuiOptimizer.

Interfaces owned by this module (board VGO-009):

* ``GuiPatchAuthority@1`` — allowed repository roots and forbidden change kinds
* ``GuiHostBoundaryPolicy@1`` — browser content cannot select host paths/commands
* ``GuiAcceptanceAuthority@1`` — evidence required for automatic acceptance

This is a pure, provider-free doctrine layer.  It does not alter backend
authorization, credentials, MCP execution, or the SwissKnife browser gateway.
Callers inject explicit path/change/evidence claims; the authority never
elevates UI state, browser policy output, or missing evidence into permission.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Final

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
# process/command selectors used by optimizer fixture doctrine.
FORBIDDEN_BROWSER_PAYLOAD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "authorization",
        "backend_credentials",
        "bearer_token",
        "api_key",
        "password",
        "secret",
        "host_path",
        "file_path",
        "filesystem_path",
        "python_process",
        "process_command",
        "stdio",
        "shell_command",
        "subprocess",
        "executable",
        "argv",
    }
)

# Host-side process/command selectors that browser content must not choose.
# Keep this aligned with the SwissKnife all-app tool gateway; do not forbid
# ordinary application intent fields such as a UI "command" name.
FORBIDDEN_BROWSER_COMMAND_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "process_command",
        "shell_command",
        "subprocess",
        "python_process",
        "stdio",
        "argv",
        "executable",
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


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise GuiAuthorityError(
            f"{name} must be a string",
            reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            details={"field": name},
        )
    if "\x00" in value:
        raise GuiAuthorityError(
            f"{name} must not contain NUL",
            reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            details={"field": name},
        )
    text = value.strip()
    if required and not text:
        raise GuiAuthorityError(
            f"{name} must not be empty",
            reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            details={"field": name},
        )
    return text


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise GuiAuthorityError(
            f"{name} must be a boolean",
            reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            details={"field": name},
        )
    return value


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


def _as_change_kind(value: Any) -> ForbiddenChangeKind:
    if isinstance(value, ForbiddenChangeKind):
        return value
    text = _text(value, "change_kind")
    try:
        return ForbiddenChangeKind(text)
    except ValueError as exc:
        raise GuiAuthorityError(
            f"unknown change kind: {text}",
            reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            details={"change_kind": text},
        ) from exc


def _as_evidence_kind(value: Any) -> AuthorityEvidenceKind:
    if isinstance(value, AuthorityEvidenceKind):
        return value
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
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise GuiAuthorityError(
            "details must be a string-keyed mapping",
            reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
        )
    return MappingProxyType(dict(value))


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
        kinds = tuple(_as_change_kind(kind) for kind in (self.change_kinds or ()))
        object.__setattr__(self, "change_kinds", kinds)


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
        roots = tuple(
            _text(root, "allowed_root")
            if str(root).endswith("/")
            else f"{_text(root, 'allowed_root')}/"
            for root in (self.allowed_roots or ())
        )
        if not roots:
            raise GuiAuthorityError(
                "allowed_roots must not be empty",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            )
        object.__setattr__(self, "allowed_roots", roots)
        parts = frozenset(
            _text(part, "forbidden_path_part")
            for part in (self.forbidden_path_parts or ())
        )
        object.__setattr__(self, "forbidden_path_parts", parts)
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        object.__setattr__(self, "interface", _text(self.interface, "interface"))

    def evaluate_path(self, path: str, *, declared: bool = True) -> AuthorityDecision:
        """Evaluate a single path against allowed roots and forbidden segments."""
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
        if not declared:
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
                details={"path": normalized, "allowed_roots": list(self.allowed_roots)},
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
        self, change_kinds: Sequence[ForbiddenChangeKind | str]
    ) -> AuthorityDecision:
        """Classify sensitive change kinds as reject / review / allow."""
        kinds = tuple(_as_change_kind(kind) for kind in (change_kinds or ()))
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
        if claims is None or (
            not isinstance(claims, Sequence)
            or isinstance(claims, (str, bytes, bytearray))
        ):
            raise GuiAuthorityError(
                "claims must be a sequence",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            )
        if not claims:
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
        if isinstance(raw, PatchPathClaim):
            return raw
        if not isinstance(raw, Mapping):
            raise GuiAuthorityError(
                f"claims[{index}] must be a PatchPathClaim or mapping",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            )
        kinds = raw.get("change_kinds") or ()
        if isinstance(kinds, (str, ForbiddenChangeKind)):
            kinds = (kinds,)
        return PatchPathClaim(
            path=raw.get("path", ""),
            declared=bool(raw.get("declared", True)),
            change_kinds=tuple(kinds),
        )


# ---------------------------------------------------------------------------
# GuiHostBoundaryPolicy@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrowserHostInput:
    """Browser-origin content presented to the host boundary.

    Fixture-only doctrine: production credentials, services, MCP tools, host
    paths, and process commands are forbidden.
    """

    payload: Mapping[str, Any] = field(default_factory=dict)
    fixture_only: bool = True
    uses_production_credentials: bool = False
    uses_production_services: bool = False
    uses_production_mcp_tools: bool = False
    uses_user_or_legal_data: bool = False
    selected_host_paths: tuple[str, ...] = ()
    selected_commands: tuple[str, ...] = ()
    selected_executables: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise GuiAuthorityError(
                "payload must be a mapping",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            )
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))
        object.__setattr__(
            self, "fixture_only", _bool(self.fixture_only, "fixture_only")
        )
        object.__setattr__(
            self,
            "uses_production_credentials",
            _bool(self.uses_production_credentials, "uses_production_credentials"),
        )
        object.__setattr__(
            self,
            "uses_production_services",
            _bool(self.uses_production_services, "uses_production_services"),
        )
        object.__setattr__(
            self,
            "uses_production_mcp_tools",
            _bool(self.uses_production_mcp_tools, "uses_production_mcp_tools"),
        )
        object.__setattr__(
            self,
            "uses_user_or_legal_data",
            _bool(self.uses_user_or_legal_data, "uses_user_or_legal_data"),
        )
        object.__setattr__(
            self,
            "selected_host_paths",
            tuple(
                _text(item, "selected_host_path", required=False)
                for item in (self.selected_host_paths or ())
            ),
        )
        object.__setattr__(
            self,
            "selected_commands",
            tuple(
                _text(item, "selected_command", required=False)
                for item in (self.selected_commands or ())
            ),
        )
        object.__setattr__(
            self,
            "selected_executables",
            tuple(
                _text(item, "selected_executable", required=False)
                for item in (self.selected_executables or ())
            ),
        )


def _walk_forbidden_payload_keys(
    value: Any,
    *,
    path: str = "",
    found: list[str] | None = None,
) -> list[str]:
    hits = found if found is not None else []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            lowered = key_text.lower()
            if lowered in FORBIDDEN_BROWSER_PAYLOAD_KEYS:
                hits.append(child_path)
            if lowered in FORBIDDEN_BROWSER_COMMAND_FIELDS:
                hits.append(child_path)
            _walk_forbidden_payload_keys(child, path=child_path, found=hits)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            _walk_forbidden_payload_keys(child, path=child_path, found=hits)
    return hits


def _looks_like_host_path(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if text.startswith("/") or text.startswith("~"):
        return True
    if len(text) >= 3 and text[1] == ":" and text[0].isalpha():
        return True
    if text.startswith("file:"):
        return True
    return False


def _looks_like_command(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    lowered = text.lower()
    markers = (
        "&&",
        "||",
        ";",
        "`",
        "$(",
        "\n",
        "sudo ",
        "rm -",
        "curl ",
        "wget ",
        "python ",
        "python3 ",
        "bash ",
        "sh ",
        "powershell ",
        "cmd.exe",
    )
    return any(marker in lowered for marker in markers)


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
        object.__setattr__(
            self,
            "forbid_absolute_path_strings",
            _bool(self.forbid_absolute_path_strings, "forbid_absolute_path_strings"),
        )
        object.__setattr__(
            self,
            "forbid_command_like_strings",
            _bool(self.forbid_command_like_strings, "forbid_command_like_strings"),
        )

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
                message="production credentials, services, tools, or data are forbidden",
                details=production_flags,
            )

        if payload_input.selected_host_paths:
            return _decision(
                AuthorityVerdict.REJECT,
                AuthorityReasonCode.BROWSER_HOST_PATH_FORBIDDEN,
                interface=self.interface,
                schema=self.schema,
                message="browser content cannot select host paths",
                details={"selected_host_paths": list(payload_input.selected_host_paths)},
            )

        if payload_input.selected_commands or payload_input.selected_executables:
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
            credential_hit = any(
                key.lower().rsplit(".", 1)[-1]
                in {
                    "authorization",
                    "backend_credentials",
                    "bearer_token",
                    "api_key",
                    "password",
                    "secret",
                }
                for key in forbidden_keys
            )
            command_hit = any(
                key.lower().rsplit(".", 1)[-1] in FORBIDDEN_BROWSER_COMMAND_FIELDS
                or key.lower().rsplit(".", 1)[-1]
                in {"python_process", "process_command", "stdio", "subprocess"}
                for key in forbidden_keys
            )
            path_hit = any(
                key.lower().rsplit(".", 1)[-1]
                in {"host_path", "file_path", "filesystem_path"}
                for key in forbidden_keys
            )
            if credential_hit:
                reason = AuthorityReasonCode.BROWSER_CREDENTIAL_FORBIDDEN
            elif command_hit:
                reason = AuthorityReasonCode.BROWSER_COMMAND_FORBIDDEN
            elif path_hit:
                reason = AuthorityReasonCode.BROWSER_HOST_PATH_FORBIDDEN
            else:
                reason = AuthorityReasonCode.BROWSER_HOST_PATH_FORBIDDEN
            return _decision(
                AuthorityVerdict.REJECT,
                reason,
                interface=self.interface,
                schema=self.schema,
                message="browser payload contains forbidden host-boundary keys",
                details={"forbidden_keys": forbidden_keys},
            )

        if self.forbid_absolute_path_strings or self.forbid_command_like_strings:
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
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                hit = self._scan_string_values(child, path=child_path)
                if hit is not None:
                    return hit
            return None
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, child in enumerate(value):
                hit = self._scan_string_values(child, path=f"{path}[{index}]")
                if hit is not None:
                    return hit
            return None
        if not isinstance(value, str):
            return None
        if self.forbid_absolute_path_strings and _looks_like_host_path(value):
            return _decision(
                AuthorityVerdict.REJECT,
                AuthorityReasonCode.BROWSER_HOST_PATH_FORBIDDEN,
                interface=self.interface,
                schema=self.schema,
                message="browser payload must not embed host filesystem paths",
                details={"path": path, "value": value},
            )
        if self.forbid_command_like_strings and _looks_like_command(value):
            return _decision(
                AuthorityVerdict.REJECT,
                AuthorityReasonCode.BROWSER_COMMAND_FORBIDDEN,
                interface=self.interface,
                schema=self.schema,
                message="browser payload must not embed host process commands",
                details={"path": path, "value": value},
            )
        return None

    def _coerce_input(
        self, browser_input: BrowserHostInput | Mapping[str, Any]
    ) -> BrowserHostInput:
        if isinstance(browser_input, BrowserHostInput):
            return browser_input
        if not isinstance(browser_input, Mapping):
            raise GuiAuthorityError(
                "browser_input must be a BrowserHostInput or mapping",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            )
        return BrowserHostInput(
            payload=browser_input.get("payload") or {},
            fixture_only=bool(browser_input.get("fixture_only", True)),
            uses_production_credentials=bool(
                browser_input.get("uses_production_credentials", False)
            ),
            uses_production_services=bool(
                browser_input.get("uses_production_services", False)
            ),
            uses_production_mcp_tools=bool(
                browser_input.get("uses_production_mcp_tools", False)
            ),
            uses_user_or_legal_data=bool(
                browser_input.get("uses_user_or_legal_data", False)
            ),
            selected_host_paths=tuple(
                browser_input.get("selected_host_paths") or ()
            ),
            selected_commands=tuple(browser_input.get("selected_commands") or ()),
            selected_executables=tuple(
                browser_input.get("selected_executables") or ()
            ),
        )


# ---------------------------------------------------------------------------
# GuiAcceptanceAuthority@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuthorityEvidence:
    """One piece of evidence offered to justify automatic acceptance."""

    kind: AuthorityEvidenceKind
    valid: bool
    evidence_id: str = ""
    binds_action_id: str = ""
    binds_argument_digest: str = ""
    policy_decision_id: str = ""
    policy_fresh: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _as_evidence_kind(self.kind))
        object.__setattr__(self, "valid", _bool(self.valid, "valid"))
        object.__setattr__(
            self,
            "evidence_id",
            _text(self.evidence_id, "evidence_id", required=False),
        )
        object.__setattr__(
            self,
            "binds_action_id",
            _text(self.binds_action_id, "binds_action_id", required=False),
        )
        object.__setattr__(
            self,
            "binds_argument_digest",
            _text(
                self.binds_argument_digest, "binds_argument_digest", required=False
            ),
        )
        object.__setattr__(
            self,
            "policy_decision_id",
            _text(self.policy_decision_id, "policy_decision_id", required=False),
        )
        object.__setattr__(
            self, "policy_fresh", _bool(self.policy_fresh, "policy_fresh")
        )
        object.__setattr__(self, "notes", str(self.notes or ""))


@dataclass(frozen=True)
class AcceptanceAuthorityRequest:
    """Inputs for automatic-acceptance evaluation.

    UI visibility/enabled state and browser policy output are recorded only so
    the authority can refuse to treat them as authorization.
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
        object.__setattr__(
            self,
            "intended_action_id",
            _text(self.intended_action_id, "intended_action_id", required=False),
        )
        object.__setattr__(
            self,
            "intended_argument_digest",
            _text(
                self.intended_argument_digest,
                "intended_argument_digest",
                required=False,
            ),
        )
        object.__setattr__(self, "ui_visible", _bool(self.ui_visible, "ui_visible"))
        object.__setattr__(self, "ui_enabled", _bool(self.ui_enabled, "ui_enabled"))
        object.__setattr__(
            self,
            "browser_policy_outcome",
            _text(
                self.browser_policy_outcome,
                "browser_policy_outcome",
                required=False,
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
            _text(self.policy_decision_id, "policy_decision_id", required=False),
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
            _text(
                self.confirmation_action_id, "confirmation_action_id", required=False
            ),
        )
        object.__setattr__(
            self,
            "confirmation_argument_digest",
            _text(
                self.confirmation_argument_digest,
                "confirmation_argument_digest",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "confirmation_granted",
            _bool(self.confirmation_granted, "confirmation_granted"),
        )
        kinds = tuple(_as_change_kind(kind) for kind in (self.change_kinds or ()))
        object.__setattr__(self, "change_kinds", kinds)
        evidence_items: list[AuthorityEvidence] = []
        for item in self.evidence or ():
            if isinstance(item, AuthorityEvidence):
                evidence_items.append(item)
            elif isinstance(item, Mapping):
                evidence_items.append(
                    AuthorityEvidence(
                        kind=item.get("kind", ""),
                        valid=bool(item.get("valid", False)),
                        evidence_id=str(item.get("evidence_id") or ""),
                        binds_action_id=str(item.get("binds_action_id") or ""),
                        binds_argument_digest=str(
                            item.get("binds_argument_digest") or ""
                        ),
                        policy_decision_id=str(item.get("policy_decision_id") or ""),
                        policy_fresh=bool(item.get("policy_fresh", False)),
                        notes=str(item.get("notes") or ""),
                    )
                )
            else:
                raise GuiAuthorityError(
                    "evidence items must be AuthorityEvidence or mappings",
                    reason_code=AuthorityReasonCode.INVALID_AUTHORITY_EVIDENCE.value,
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
        if self.host_boundary_decision is not None and not isinstance(
            self.host_boundary_decision, AuthorityDecision
        ):
            raise GuiAuthorityError(
                "host_boundary_decision must be an AuthorityDecision",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            )
        if self.patch_authority_decision is not None and not isinstance(
            self.patch_authority_decision, AuthorityDecision
        ):
            raise GuiAuthorityError(
                "patch_authority_decision must be an AuthorityDecision",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            )


@dataclass(frozen=True)
class GuiAcceptanceAuthority:
    """Evidence doctrine for automatic acceptance.

    Interface: ``GuiAcceptanceAuthority@1``.

    UI state cannot synthesize authorization.  Browser policy output is never
    authoritative.  Sensitive changes require contract verification or human
    review.  Missing or invalid authority evidence rejects safely.
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
            # Presence of UI state alone is fine; using it as the sole authority is not.
            if not req.evidence and not req.policy_decision_id:
                return _decision(
                    AuthorityVerdict.REJECT,
                    AuthorityReasonCode.UI_STATE_NOT_AUTHORIZATION,
                    AuthorityReasonCode.MISSING_AUTHORITY_EVIDENCE,
                    interface=self.interface,
                    schema=self.schema,
                    message="UI visibility/enabled state cannot synthesize authorization",
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

        if req.policy_decision_id and not req.policy_fresh:
            return _decision(
                AuthorityVerdict.REJECT,
                AuthorityReasonCode.STALE_POLICY_DECISION,
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
            if (
                not req.intended_action_id
                or req.confirmation_action_id != req.intended_action_id
                or req.confirmation_argument_digest != req.intended_argument_digest
            ):
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

        # Automatic acceptance still needs at least one valid host-side evidence
        # class when an action is declared, or an explicit empty-action allow for
        # pure observation receipts.
        if req.intended_action_id and not self._has_valid_host_authority(req):
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

    def _has_valid_evidence(
        self,
        request: AcceptanceAuthorityRequest,
        kind: AuthorityEvidenceKind,
    ) -> bool:
        return any(
            item.kind is kind and item.valid for item in request.evidence
        )

    def _has_valid_host_authority(self, request: AcceptanceAuthorityRequest) -> bool:
        """Host re-evaluation, exact confirmation, contract, or human review."""
        if request.policy_decision_id and request.policy_fresh:
            return True
        for item in request.evidence:
            if not item.valid:
                continue
            if item.kind in {
                AuthorityEvidenceKind.HOST_POLICY_REEVALUATION,
                AuthorityEvidenceKind.CONTRACT_VERIFICATION,
                AuthorityEvidenceKind.HUMAN_REVIEW,
                AuthorityEvidenceKind.EXACT_CONFIRMATION_BINDING,
                AuthorityEvidenceKind.SCOPE_DECLARATION,
            }:
                return True
        if (
            request.confirmation_required
            and request.confirmation_granted
            and request.confirmation_action_id == request.intended_action_id
            and request.confirmation_argument_digest
            == request.intended_argument_digest
            and request.intended_action_id
        ):
            return True
        return False

    def _coerce_request(
        self, request: AcceptanceAuthorityRequest | Mapping[str, Any]
    ) -> AcceptanceAuthorityRequest:
        if isinstance(request, AcceptanceAuthorityRequest):
            return request
        if not isinstance(request, Mapping):
            raise GuiAuthorityError(
                "request must be an AcceptanceAuthorityRequest or mapping",
                reason_code=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            )
        kinds = request.get("change_kinds") or ()
        if isinstance(kinds, (str, ForbiddenChangeKind)):
            kinds = (kinds,)
        return AcceptanceAuthorityRequest(
            intended_action_id=str(request.get("intended_action_id") or ""),
            intended_argument_digest=str(
                request.get("intended_argument_digest") or ""
            ),
            ui_visible=bool(request.get("ui_visible", False)),
            ui_enabled=bool(request.get("ui_enabled", False)),
            browser_policy_outcome=str(
                request.get("browser_policy_outcome") or ""
            ),
            browser_policy_authoritative_claim=bool(
                request.get("browser_policy_authoritative_claim", False)
            ),
            policy_decision_id=str(request.get("policy_decision_id") or ""),
            policy_fresh=bool(request.get("policy_fresh", False)),
            confirmation_required=bool(
                request.get("confirmation_required", False)
            ),
            confirmation_action_id=str(
                request.get("confirmation_action_id") or ""
            ),
            confirmation_argument_digest=str(
                request.get("confirmation_argument_digest") or ""
            ),
            confirmation_granted=bool(request.get("confirmation_granted", False)),
            change_kinds=tuple(kinds),
            evidence=tuple(request.get("evidence") or ()),
            accessibility_regression=bool(
                request.get("accessibility_regression", False)
            ),
            security_regression=bool(request.get("security_regression", False)),
            host_boundary_decision=request.get("host_boundary_decision"),
            patch_authority_decision=request.get("patch_authority_decision"),
        )


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
        """Evaluate patch, optional host boundary, then acceptance in order."""
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

        if isinstance(acceptance, AcceptanceAuthorityRequest):
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
                change_kinds=acceptance.change_kinds,
                evidence=acceptance.evidence,
                accessibility_regression=acceptance.accessibility_regression,
                security_regression=acceptance.security_regression,
                host_boundary_decision=host_decision
                or acceptance.host_boundary_decision,
                patch_authority_decision=patch_decision,
            )
        else:
            payload = dict(acceptance)
            payload.setdefault("host_boundary_decision", host_decision)
            payload.setdefault("patch_authority_decision", patch_decision)
            # Merge claim change kinds when the caller omitted them.
            if not payload.get("change_kinds"):
                merged: list[ForbiddenChangeKind] = []
                for claim in claims:
                    if isinstance(claim, PatchPathClaim):
                        merged.extend(claim.change_kinds)
                    elif isinstance(claim, Mapping):
                        kinds = claim.get("change_kinds") or ()
                        if isinstance(kinds, (str, ForbiddenChangeKind)):
                            kinds = (kinds,)
                        merged.extend(_as_change_kind(kind) for kind in kinds)
                payload["change_kinds"] = tuple(merged)
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
    "FORBIDDEN_BROWSER_PAYLOAD_KEYS",
    "ForbiddenChangeKind",
    "GUI_ACCEPTANCE_AUTHORITY_INTERFACE",
    "GUI_ACCEPTANCE_AUTHORITY_SCHEMA",
    "GUI_AUTHORITY_DECISION_SCHEMA",
    "GUI_HOST_BOUNDARY_POLICY_INTERFACE",
    "GUI_HOST_BOUNDARY_POLICY_SCHEMA",
    "GUI_PATCH_AUTHORITY_INTERFACE",
    "GUI_PATCH_AUTHORITY_SCHEMA",
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
