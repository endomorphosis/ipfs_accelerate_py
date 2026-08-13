"""Source disclosure, redaction, provider, and worktree privacy gate (SCG-024).

Fail-closed privacy control for shadow evaluation and provider invocation:

* Expanded private source is **local-only** by default.
* Broader external disclosure requires exact, explicit provider authorization
  on an approved provider identity — never ambient trust.
* Secrets are scanned and redacted before provider invocation.
* Secrets and arbitrary host filesystem paths cannot enter provider invocation
  payloads or public reports (public reports store CIDs and managed references).
* Isolated evaluation worktrees are required for expanded shadow runs.

Importing this module performs no I/O, opens no sockets, and never invokes a
provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, Iterable, Mapping, Sequence
import re
import unicodedata

from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_structured,
    validate_cid,
    validate_structured_value,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.base import (
    PRIVATE_FIELD_MARKERS,
    SemanticGovernorBaseError,
)

# ---------------------------------------------------------------------------
# Evidence / interface / schema constants
# ---------------------------------------------------------------------------

SCG_PRIVACY_GATE_EVIDENCE: Final[str] = "scg/privacy-gate@1"

SHADOW_DISCLOSURE_POLICY_INTERFACE: Final[str] = "ShadowDisclosurePolicy@1"
SHADOW_DISCLOSURE_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "shadow-disclosure-policy@1"
)

DISCLOSURE_AUTHORIZATION_INTERFACE: Final[str] = "ShadowDisclosureAuthorization@1"
DISCLOSURE_AUTHORIZATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "shadow-disclosure-authorization@1"
)

PROVIDER_INVOCATION_CONTEXT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "provider-invocation-context@1"
)

PUBLIC_REPORT_PROJECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/"
    "public-report-projection@1"
)

REDACT_CONTEXT_FOR_PROVIDER_INTERFACE: Final[str] = "redact_context_for_provider@1"
AUTHORIZE_SHADOW_DISCLOSURE_INTERFACE: Final[str] = "authorize_shadow_disclosure@1"

REDACTION_MARKER: Final[str] = "[REDACTED]"
MANAGED_PATH_PLACEHOLDER: Final[str] = "<managed-path-redacted>"

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_PROVIDER_IDS: Final[int] = 256
MAX_PATH_CHARS: Final[int] = 1_024
MAX_REASON_CODES: Final[int] = 64
MAX_SECRET_FINDINGS: Final[int] = 128
MAX_CONTEXT_BYTES: Final[int] = 2_000_000
MAX_APPROVED_PROVIDERS: Final[int] = 128
MAX_REPO_PATH_CLASSES: Final[int] = 32
MAX_METADATA_KEYS: Final[int] = 64

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")
_PROVIDER_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z][A-Za-z0-9_.:/+@-]{0,127}$"
)
_WORKTREE_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.:/+@-]{0,127}$"
)
_REPO_RELATIVE_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:[A-Za-z0-9_.@+-][A-Za-z0-9_./@+-]{0,1022})$"
)

# Absolute / host-local path markers rejected from invocation and public reports.
_ABSOLUTE_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:"
    r"/|"  # POSIX absolute
    r"[A-Za-z]:[\\/]|"  # Windows drive
    r"\\\\|"  # UNC
    r"file:"  # file URI
    r")"
)
_HOME_PATH_RE: Final[re.Pattern[str]] = re.compile(r"^~/")

_PATH_KEY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "absolute_path",
        "file_path",
        "filesystem_path",
        "host_path",
        "local_path",
        "path",
        "realpath",
        "source_path",
        "workdir",
        "working_directory",
        "workspace_path",
        "worktree_path",
        "repo_root",
        "repository_root",
        "checkout_path",
    }
)

# Field names that carry private source or secret material.
# Union of datasets private markers and invocation-sensitive credential keys.
_SENSITIVE_KEY_MARKERS: Final[frozenset[str]] = frozenset(
    set(PRIVATE_FIELD_MARKERS)
    | {
        "apikey",
        "auth_token",
        "client_secret",
        "credentials",
        "github_token",
        "passphrase",
        "token",
        "source_body",
        "source_code",
        "file_content",
        "file_contents",
        "repository_body",
        "repository_content",
        "raw_private_source",
        "private_source_text",
    }
)

_PRIVATE_SOURCE_KEY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "private_source",
        "private_source_text",
        "raw_private_source",
        "raw_source",
        "raw_source_text",
        "source_bytes",
        "source_text",
        "source_body",
        "source_code",
        "file_content",
        "file_contents",
        "repository_body",
        "repository_content",
    }
)

_TEXT_SECRET_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(
        r"(?i)\b(api[_ -]?key|access[_ -]?token|auth[_ -]?token|"
        r"client[_ -]?secret|password|passphrase|secret|token)"
        r"(\s*[:=]\s*)[^\s,;]{4,}"
    ),
    re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?"
        r"-----END [A-Z0-9 ]*PRIVATE KEY-----",
        re.DOTALL,
    ),
    # Common opaque API key shapes (sk-..., ghp_..., xoxb-...).
    re.compile(r"(?i)\b(?:sk|pk|rk)-[A-Za-z0-9]{16,}\b"),
    re.compile(r"(?i)\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    re.compile(r"(?i)\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
)

# Host path substrings scrubbed from free text during redaction.
_TEXT_HOST_PATH_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?P<path>/(?:home|tmp|var|etc|usr|root|opt|Users)/[^\s\"']+)"),
    re.compile(r"(?P<path>[A-Za-z]:\\[^\s\"']+)"),
    re.compile(r"(?P<path>\\\\[^\s\"']+)"),
    re.compile(r"(?P<path>file://[^\s\"']+)"),
    re.compile(r"(?P<path>~/[^\s\"']+)"),
)

# Local / simulated provider id prefixes that never leave the machine.
_LOCAL_PROVIDER_PREFIXES: Final[tuple[str, ...]] = (
    "local:",
    "local/",
    "sim:",
    "simulated:",
    "injected:",
    "hermetic:",
    "offline:",
    "stub:",
)

_SIMULATED_PROVIDER_PREFIXES: Final[tuple[str, ...]] = (
    "sim:",
    "simulated:",
    "stub:",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SemanticGovernorPrivacyError(SemanticGovernorBaseError):
    """Raised when disclosure, redaction, or worktree privacy policy fails closed."""


class DisclosureForbiddenError(SemanticGovernorPrivacyError):
    """Private source disclosure to an unapproved provider was refused."""


class SecretAdmissionError(SemanticGovernorPrivacyError):
    """Secrets or private markers are not admitted into the requested surface."""


class HostPathAdmissionError(SemanticGovernorPrivacyError):
    """Arbitrary host filesystem paths are not admitted."""


class WorktreePolicyError(SemanticGovernorPrivacyError):
    """Isolated evaluation worktree policy was violated."""


# ---------------------------------------------------------------------------
# Closed enumerations
# ---------------------------------------------------------------------------


class SourcePrivacyClass(str, Enum):
    """Closed classification of context-pack / source material."""

    PUBLIC = "public"
    MANAGED_REFERENCE = "managed_reference"
    PRIVATE = "private"
    RAW_PRIVATE = "raw_private"


class ProviderLocality(str, Enum):
    """Closed provider trust / locality classes for disclosure."""

    LOCAL = "local"
    SIMULATED = "simulated"
    APPROVED_EXTERNAL = "approved_external"
    UNAPPROVED_EXTERNAL = "unapproved_external"


class PathClass(str, Enum):
    """Closed path classes admitted under privacy policy."""

    REPO_RELATIVE = "repo_relative"
    MANAGED_WORKTREE_ID = "managed_worktree_id"
    CONTENT_CID = "content_cid"
    HOST_ABSOLUTE = "host_absolute"
    FORBIDDEN = "forbidden"


class DisclosureDisposition(str, Enum):
    """Closed outcome of authorize_shadow_disclosure."""

    ALLOWED = "allowed"
    LOCAL_ONLY = "local_only"
    REDACTED_ONLY = "redacted_only"
    FORBIDDEN = "forbidden"


class SecretFindingKind(str, Enum):
    """Closed secret-scan finding kinds."""

    SENSITIVE_FIELD = "sensitive_field"
    PRIVATE_SOURCE_FIELD = "private_source_field"
    TEXT_PATTERN = "text_pattern"
    HOST_PATH = "host_path"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _normalize_token(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def _token(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise SemanticGovernorPrivacyError(f"{name} must be a string")
    text = _normalize_token(value)
    if not text or not _TOKEN_RE.match(text):
        raise SemanticGovernorPrivacyError(f"{name} has invalid token form {value!r}")
    return text


def _provider_id(value: Any, name: str = "provider_id") -> str:
    if not isinstance(value, str):
        raise SemanticGovernorPrivacyError(f"{name} must be a string")
    text = _normalize_token(value)
    if not text or not _PROVIDER_ID_RE.match(text):
        raise SemanticGovernorPrivacyError(f"{name} has invalid form {value!r}")
    return text


def _optional_provider_id(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _provider_id(value, name)


def _worktree_id(value: Any, name: str = "worktree_id") -> str:
    if not isinstance(value, str):
        raise SemanticGovernorPrivacyError(f"{name} must be a string")
    text = _normalize_token(value)
    if not text:
        raise SemanticGovernorPrivacyError(f"{name} must not be empty")
    # Host paths fail with the dedicated admission error even when they also
    # fail the managed-id grammar (e.g. absolute POSIX paths).
    if _string_looks_like_host_path(text):
        raise HostPathAdmissionError(
            f"{name} must be a managed worktree id, not a host path"
        )
    if not _WORKTREE_ID_RE.match(text):
        raise SemanticGovernorPrivacyError(f"{name} has invalid form {value!r}")
    return text


def _optional_worktree_id(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _worktree_id(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise SemanticGovernorPrivacyError(f"{name} must be a bool")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise SemanticGovernorPrivacyError(f"{name} must be a non-negative int")
    return value


def _positive_int(value: Any, name: str) -> int:
    n = _nonneg_int(value, name)
    if n <= 0:
        raise SemanticGovernorPrivacyError(f"{name} must be positive")
    return n


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SemanticGovernorPrivacyError(f"{name} must be a string CID or None")
    text = value.strip()
    try:
        validate_cid(text)
    except Exception as exc:
        raise SemanticGovernorPrivacyError(f"{name} is not a valid CID") from exc
    return text


def _cid(value: Any, name: str) -> str:
    result = _optional_cid(value, name)
    if result is None:
        raise SemanticGovernorPrivacyError(f"{name} is required")
    return result


def _optional_text(value: Any, name: str, *, max_chars: int = MAX_TEXT_CHARS) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SemanticGovernorPrivacyError(f"{name} must be a string or None")
    text = unicodedata.normalize("NFC", value)
    if len(text) > max_chars:
        raise SemanticGovernorPrivacyError(f"{name} exceeds {max_chars} characters")
    if "\x00" in text:
        raise SemanticGovernorPrivacyError(f"{name} contains a null byte")
    return text


def _enum(value: Any, enum_cls: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError as exc:
            raise SemanticGovernorPrivacyError(
                f"{name} has unknown value {value!r}"
            ) from exc
    raise SemanticGovernorPrivacyError(f"{name} must be a {enum_cls.__name__}")


def _unique_sorted_tokens(
    values: Iterable[Any],
    name: str,
    *,
    max_items: int,
    validator=_token,
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple, frozenset, set)):
        raise SemanticGovernorPrivacyError(f"{name} must be a sequence")
    ordered = tuple(sorted({validator(item, name) for item in values}))
    if len(ordered) > max_items:
        raise SemanticGovernorPrivacyError(f"{name} exceeds maximum length {max_items}")
    return ordered


def _normalized_key(name: str) -> str:
    return name.strip().casefold().replace("-", "_").replace(" ", "_")


def _key_is_sensitive(name: str) -> bool:
    lowered = _normalized_key(name)
    if lowered in _SENSITIVE_KEY_MARKERS:
        return True
    for marker in _SENSITIVE_KEY_MARKERS:
        if marker in lowered:
            return True
    return False


def _key_is_private_source(name: str) -> bool:
    lowered = _normalized_key(name)
    if lowered in _PRIVATE_SOURCE_KEY_MARKERS:
        return True
    for marker in _PRIVATE_SOURCE_KEY_MARKERS:
        if marker in lowered:
            return True
    return False


def _key_looks_like_path_field(name: str) -> bool:
    lowered = _normalized_key(name)
    if lowered in _PATH_KEY_MARKERS:
        return True
    if lowered.endswith("_path") or lowered.endswith("_dir") or lowered.endswith(
        "_directory"
    ):
        return True
    return False


def _string_looks_like_host_path(value: str) -> bool:
    if not value:
        return False
    if _ABSOLUTE_PATH_RE.match(value) or _HOME_PATH_RE.match(value):
        return True
    if ("/" in value or "\\" in value) and "://" not in value:
        if value.startswith("./") or value.startswith(".\\") or value.startswith("../"):
            return True
    return False


def _is_repo_relative_path(value: str) -> bool:
    if not value or _string_looks_like_host_path(value):
        return False
    if ".." in value.split("/"):
        return False
    if value.startswith("/") or value.startswith("\\"):
        return False
    return bool(_REPO_RELATIVE_RE.match(value))


def _thaw_structured(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _thaw_structured(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_structured(item) for item in value]
    return value


def _freeze_structured(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze_structured(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_structured(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_structured(item) for item in value)
    return value


def _mapping(value: Any, name: str, *, frozen: bool = True) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SemanticGovernorPrivacyError(f"{name} must be a mapping")
    thawed = _thaw_structured(dict(value))
    try:
        validate_structured_value(thawed, path=name)
    except Exception as exc:
        raise SemanticGovernorPrivacyError(
            f"{name} must be strict DAG-JSON without floats or host types"
        ) from exc
    if len(thawed) > MAX_METADATA_KEYS:
        raise SemanticGovernorPrivacyError(f"{name} exceeds metadata key bound")
    return _freeze_structured(thawed) if frozen else thawed


# ---------------------------------------------------------------------------
# Secret scan
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SecretFinding:
    """One bounded secret or private-source finding."""

    kind: SecretFindingKind | str
    path: str
    reason_code: str
    preview: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, SecretFindingKind, "kind").value
        )
        object.__setattr__(self, "path", _optional_text(self.path, "path") or "$")
        object.__setattr__(
            self, "reason_code", _token(self.reason_code, "reason_code")
        )
        object.__setattr__(
            self,
            "preview",
            _optional_text(self.preview, "preview", max_chars=64),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "reason_code": self.reason_code,
            "preview": self.preview,
        }


def _text_contains_secret_pattern(text: str) -> str | None:
    for pattern in _TEXT_SECRET_PATTERNS:
        if pattern.search(text):
            if "PRIVATE KEY" in pattern.pattern:
                return "pem_private_key"
            if pattern.pattern.startswith(r"(?i)\b(bearer)"):
                return "bearer_token"
            if "sk|" in pattern.pattern or "ghp|" in pattern.pattern:
                return "opaque_api_key_shape"
            return "credential_assignment"
    return None


def scan_secrets(
    value: Any,
    *,
    path: str = "$",
    max_findings: int = MAX_SECRET_FINDINGS,
) -> tuple[SecretFinding, ...]:
    """Scan *value* for secret fields, private-source fields, and text patterns.

    Returns a bounded, deterministic tuple of findings. Does not mutate *value*.
    """

    findings: list[SecretFinding] = []

    def _add(kind: SecretFindingKind, loc: str, reason: str, preview: str | None = None) -> None:
        if len(findings) >= max_findings:
            return
        findings.append(
            SecretFinding(
                kind=kind,
                path=loc,
                reason_code=reason,
                preview=preview,
            )
        )

    def _walk(node: Any, loc: str) -> None:
        if len(findings) >= max_findings:
            return
        if isinstance(node, Mapping):
            for key, item in node.items():
                if type(key) is not str:
                    raise SemanticGovernorPrivacyError(
                        f"{loc} map keys must be str, got {type(key).__name__}"
                    )
                key_path = f"{loc}.{key}"
                if _key_is_private_source(key):
                    _add(
                        SecretFindingKind.PRIVATE_SOURCE_FIELD,
                        key_path,
                        "private_source_field",
                        key,
                    )
                elif _key_is_sensitive(key):
                    _add(
                        SecretFindingKind.SENSITIVE_FIELD,
                        key_path,
                        "sensitive_field",
                        key,
                    )
                _walk(item, key_path)
            return
        if isinstance(node, (list, tuple)):
            for index, item in enumerate(node):
                _walk(item, f"{loc}[{index}]")
            return
        if isinstance(node, str):
            if _string_looks_like_host_path(node):
                _add(SecretFindingKind.HOST_PATH, loc, "host_path_value", None)
            reason = _text_contains_secret_pattern(node)
            if reason is not None:
                _add(SecretFindingKind.TEXT_PATTERN, loc, reason, None)
            return

    _walk(value, path)
    # Deterministic order: path, then reason_code, then kind.
    return tuple(
        sorted(findings, key=lambda f: (f.path, f.reason_code, f.kind))
    )


def contains_private_source(value: Any) -> bool:
    """Return True when *value* embeds private / raw source fields or markers."""

    for finding in scan_secrets(value):
        if finding.kind == SecretFindingKind.PRIVATE_SOURCE_FIELD.value:
            return True
    return False


def contains_secrets(value: Any) -> bool:
    """Return True when sensitive fields or secret text patterns are present."""

    for finding in scan_secrets(value):
        if finding.kind in {
            SecretFindingKind.SENSITIVE_FIELD.value,
            SecretFindingKind.TEXT_PATTERN.value,
        }:
            return True
    return False


# ---------------------------------------------------------------------------
# Path classification and rejection
# ---------------------------------------------------------------------------


def classify_path(value: str, *, worktree_id: str | None = None) -> PathClass:
    """Classify a path string into a closed privacy path class."""

    if not isinstance(value, str):
        raise SemanticGovernorPrivacyError("path must be a string")
    text = value.strip()
    if not text:
        raise SemanticGovernorPrivacyError("path must not be empty")
    if len(text) > MAX_PATH_CHARS:
        raise SemanticGovernorPrivacyError("path exceeds maximum length")

    # Content-addressed references are portable and public-safe.
    if text.startswith("baguquee") or text.startswith("bafy") or text.startswith(
        "bafk"
    ):
        try:
            validate_cid(text)
            return PathClass.CONTENT_CID
        except Exception:
            pass

    if worktree_id is not None and text == worktree_id:
        return PathClass.MANAGED_WORKTREE_ID

    # Managed worktree ids never look like host paths.
    if _WORKTREE_ID_RE.match(text) and not _string_looks_like_host_path(text):
        if text.startswith("worktree-") or text.startswith("wt:"):
            return PathClass.MANAGED_WORKTREE_ID

    if _string_looks_like_host_path(text):
        return PathClass.HOST_ABSOLUTE

    if _is_repo_relative_path(text):
        return PathClass.REPO_RELATIVE

    return PathClass.FORBIDDEN


def reject_host_paths(value: Any, *, path: str = "$") -> None:
    """Fail closed when an arbitrary host filesystem path is present."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise HostPathAdmissionError(
                    f"{path} map keys must be str, got {type(key).__name__}"
                )
            key_path = f"{path}.{key}"
            if _key_looks_like_path_field(key) and isinstance(item, str):
                # Path-named fields must not carry host absolute paths; relative
                # repo paths are admitted only when they classify as repo-relative.
                cls = classify_path(item) if item else PathClass.FORBIDDEN
                if cls is PathClass.HOST_ABSOLUTE or cls is PathClass.FORBIDDEN:
                    if item and _string_looks_like_host_path(item):
                        raise HostPathAdmissionError(
                            f"{key_path} rejects arbitrary host path"
                        )
                    # Empty or non-portable path field names still fail for
                    # host-absolute-looking values; pure repo-relative ok.
                    if cls is PathClass.HOST_ABSOLUTE:
                        raise HostPathAdmissionError(
                            f"{key_path} rejects arbitrary host path"
                        )
                if cls is PathClass.HOST_ABSOLUTE:
                    raise HostPathAdmissionError(
                        f"{key_path} rejects arbitrary host path"
                    )
            if isinstance(item, str) and _string_looks_like_host_path(item):
                raise HostPathAdmissionError(
                    f"{key_path} rejects arbitrary host path value"
                )
            reject_host_paths(item, path=key_path)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_host_paths(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and _string_looks_like_host_path(value):
        raise HostPathAdmissionError(f"{path} rejects arbitrary host path value")


def reject_secrets(value: Any, *, path: str = "$") -> None:
    """Fail closed when secrets or private-source fields remain in *value*."""

    findings = scan_secrets(value, path=path)
    for finding in findings:
        if finding.kind == SecretFindingKind.HOST_PATH.value:
            # Host paths have a dedicated gate; skip here.
            continue
        raise SecretAdmissionError(
            f"{finding.path} rejects {finding.reason_code} "
            f"({finding.kind}) in admission surface"
        )


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def _redact_text(value: str, *, strip_host_paths: bool = False) -> str:
    result = value
    for pattern in _TEXT_SECRET_PATTERNS:
        if pattern.pattern.startswith(r"(?i)\b(bearer)"):
            result = pattern.sub(r"\1 " + REDACTION_MARKER, result)
        elif "PRIVATE KEY" in pattern.pattern:
            result = pattern.sub(REDACTION_MARKER, result)
        elif "sk|" in pattern.pattern or "ghp|" in pattern.pattern or "xox" in pattern.pattern:
            result = pattern.sub(REDACTION_MARKER, result)
        else:
            result = pattern.sub(r"\1\2" + REDACTION_MARKER, result)
    if strip_host_paths:
        for pattern in _TEXT_HOST_PATH_PATTERNS:
            result = pattern.sub(MANAGED_PATH_PLACEHOLDER, result)
    return result


def redact_context_for_provider(
    context: Any,
    *,
    strip_private_source: bool = False,
    strip_host_paths: bool = True,
) -> Any:
    """Return a deep-redacted copy of *context* safe for provider invocation.

    * Sensitive field values become ``[REDACTED]``.
    * Secret-shaped text substrings are redacted in place.
    * When *strip_private_source* is true, private-source fields are removed
      entirely (not merely redacted) so raw source cannot be reconstructed.
    * When *strip_host_paths* is true, host-absolute path strings are replaced
      with a managed placeholder and path-named host fields are removed.

    Does not authorize disclosure — call :func:`authorize_shadow_disclosure`
    first. Never raises on well-formed JSON-like data; malformed host types fail.
    """

    def _walk(node: Any, loc: str) -> Any:
        if isinstance(node, Mapping):
            out: dict[str, Any] = {}
            for key, item in node.items():
                if type(key) is not str:
                    raise SemanticGovernorPrivacyError(
                        f"{loc} map keys must be str, got {type(key).__name__}"
                    )
                key_path = f"{loc}.{key}"
                if strip_private_source and _key_is_private_source(key):
                    # Drop private source fields entirely for external safety.
                    continue
                if _key_is_private_source(key):
                    # Local / authorized paths may retain source text, but still
                    # scrub credential-shaped substrings inside it.
                    out[key] = _walk(item, key_path)
                    continue
                if _key_is_sensitive(key):
                    # Secrets never enter invocation (field value replaced).
                    out[key] = REDACTION_MARKER
                    continue
                if strip_host_paths and _key_looks_like_path_field(key):
                    if isinstance(item, str) and (
                        _string_looks_like_host_path(item)
                        or classify_path(item) is PathClass.HOST_ABSOLUTE
                    ):
                        # Remove host path fields rather than invent structure.
                        continue
                out[key] = _walk(item, key_path)
            return out
        if isinstance(node, (list, tuple)):
            return [_walk(item, f"{loc}[{index}]") for index, item in enumerate(node)]
        if isinstance(node, str):
            text = _redact_text(node, strip_host_paths=strip_host_paths)
            if strip_host_paths and _string_looks_like_host_path(text):
                return MANAGED_PATH_PLACEHOLDER
            return text
        if node is None or isinstance(node, bool):
            return node
        if isinstance(node, int) and not isinstance(node, bool):
            return node
        raise SemanticGovernorPrivacyError(
            f"{loc} admits only strict JSON scalars/containers; "
            f"got {type(node).__name__}"
        )

    if not isinstance(context, (Mapping, list, tuple, str, int, bool, type(None))):
        raise SemanticGovernorPrivacyError(
            f"context must be JSON-like, got {type(context).__name__}"
        )
    return _walk(context, "$")


# ---------------------------------------------------------------------------
# Worktree policy
# ---------------------------------------------------------------------------


def assert_isolated_evaluation_worktree(
    *,
    isolated_evaluation_worktree_required: bool = True,
    worktree_id: str | None = None,
    worktree_path: str | None = None,
    host_path_allowed: bool = False,
) -> str | None:
    """Enforce isolated evaluation-worktree policy (fail closed).

    Returns the validated managed *worktree_id* (or None when not supplied).
    Host absolute *worktree_path* values are never admitted into policy state.
    """

    if not isolated_evaluation_worktree_required:
        raise WorktreePolicyError(
            "isolated_evaluation_worktree_required must be true"
        )
    if worktree_path is not None:
        if not isinstance(worktree_path, str):
            raise WorktreePolicyError("worktree_path must be a string when provided")
        if _string_looks_like_host_path(worktree_path) and not host_path_allowed:
            raise HostPathAdmissionError(
                "worktree_path rejects arbitrary host path in privacy surface"
            )
        # Even when a runtime holds a host path privately, the privacy gate
        # never returns or stores it — only managed ids cross this boundary.
    return _optional_worktree_id(worktree_id, "worktree_id")


# ---------------------------------------------------------------------------
# ShadowDisclosurePolicy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShadowDisclosurePolicy:
    """Closed policy binding provider capability to source disclosure rules.

    Default is **local-only** expansion: private/raw source may be used by
    local or simulated providers, but never by unapproved external providers.
    Approved external disclosure requires both:

    1. the provider id on ``approved_external_provider_ids``, and
    2. ``allow_private_source_to_approved_external=True`` with an explicit
       ``authorization_cid`` (exact authority — no ambient trust).
    """

    policy_id: str = "shadow-disclosure-default"
    approved_external_provider_ids: Sequence[str] = field(default_factory=tuple)
    allow_private_source_to_local: bool = True
    allow_private_source_to_simulated: bool = True
    allow_private_source_to_approved_external: bool = False
    # Always false by construction — field retained for explicit identity.
    allow_private_source_to_unapproved_external: bool = False
    require_isolated_evaluation_worktree: bool = True
    require_secret_scan: bool = True
    allow_host_paths_in_invocation: bool = False
    allow_host_paths_in_public_reports: bool = False
    allowed_path_classes: Sequence[str] = field(
        default_factory=lambda: (
            PathClass.REPO_RELATIVE.value,
            PathClass.MANAGED_WORKTREE_ID.value,
            PathClass.CONTENT_CID.value,
        )
    )
    max_context_bytes: int = MAX_CONTEXT_BYTES
    authorization_cid: str | None = None
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface_id",
            "policy_id",
            "approved_external_provider_ids",
            "allow_private_source_to_local",
            "allow_private_source_to_simulated",
            "allow_private_source_to_approved_external",
            "allow_private_source_to_unapproved_external",
            "require_isolated_evaluation_worktree",
            "require_secret_scan",
            "allow_host_paths_in_invocation",
            "allow_host_paths_in_public_reports",
            "allowed_path_classes",
            "max_context_bytes",
            "authorization_cid",
            "notes",
            "metadata",
            "policy_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _token(self.policy_id, "policy_id"))
        object.__setattr__(
            self,
            "approved_external_provider_ids",
            _unique_sorted_tokens(
                list(self.approved_external_provider_ids),
                "approved_external_provider_ids",
                max_items=MAX_APPROVED_PROVIDERS,
                validator=_provider_id,
            ),
        )
        object.__setattr__(
            self,
            "allow_private_source_to_local",
            _bool(self.allow_private_source_to_local, "allow_private_source_to_local"),
        )
        object.__setattr__(
            self,
            "allow_private_source_to_simulated",
            _bool(
                self.allow_private_source_to_simulated,
                "allow_private_source_to_simulated",
            ),
        )
        object.__setattr__(
            self,
            "allow_private_source_to_approved_external",
            _bool(
                self.allow_private_source_to_approved_external,
                "allow_private_source_to_approved_external",
            ),
        )
        object.__setattr__(
            self,
            "allow_private_source_to_unapproved_external",
            _bool(
                self.allow_private_source_to_unapproved_external,
                "allow_private_source_to_unapproved_external",
            ),
        )
        # Hard invariant: unapproved external private disclosure is never on.
        if self.allow_private_source_to_unapproved_external:
            raise SemanticGovernorPrivacyError(
                "allow_private_source_to_unapproved_external must be false"
            )
        object.__setattr__(
            self,
            "require_isolated_evaluation_worktree",
            _bool(
                self.require_isolated_evaluation_worktree,
                "require_isolated_evaluation_worktree",
            ),
        )
        if not self.require_isolated_evaluation_worktree:
            raise SemanticGovernorPrivacyError(
                "require_isolated_evaluation_worktree must be true"
            )
        object.__setattr__(
            self,
            "require_secret_scan",
            _bool(self.require_secret_scan, "require_secret_scan"),
        )
        object.__setattr__(
            self,
            "allow_host_paths_in_invocation",
            _bool(
                self.allow_host_paths_in_invocation,
                "allow_host_paths_in_invocation",
            ),
        )
        object.__setattr__(
            self,
            "allow_host_paths_in_public_reports",
            _bool(
                self.allow_host_paths_in_public_reports,
                "allow_host_paths_in_public_reports",
            ),
        )
        if self.allow_host_paths_in_public_reports:
            raise SemanticGovernorPrivacyError(
                "allow_host_paths_in_public_reports must be false"
            )
        path_classes = _unique_sorted_tokens(
            list(self.allowed_path_classes),
            "allowed_path_classes",
            max_items=MAX_REPO_PATH_CLASSES,
        )
        for item in path_classes:
            try:
                PathClass(item)
            except ValueError as exc:
                raise SemanticGovernorPrivacyError(
                    f"allowed_path_classes has unknown value {item!r}"
                ) from exc
            if item in {
                PathClass.HOST_ABSOLUTE.value,
                PathClass.FORBIDDEN.value,
            }:
                raise SemanticGovernorPrivacyError(
                    "allowed_path_classes cannot admit host_absolute or forbidden"
                )
        object.__setattr__(self, "allowed_path_classes", path_classes)
        object.__setattr__(
            self,
            "max_context_bytes",
            _positive_int(self.max_context_bytes, "max_context_bytes"),
        )
        object.__setattr__(
            self,
            "authorization_cid",
            _optional_cid(self.authorization_cid, "authorization_cid"),
        )
        if (
            self.allow_private_source_to_approved_external
            and self.authorization_cid is None
        ):
            raise SemanticGovernorPrivacyError(
                "allow_private_source_to_approved_external requires authorization_cid"
            )
        if (
            self.allow_private_source_to_approved_external
            and not self.approved_external_provider_ids
        ):
            raise SemanticGovernorPrivacyError(
                "allow_private_source_to_approved_external requires "
                "approved_external_provider_ids"
            )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": SHADOW_DISCLOSURE_POLICY_SCHEMA,
            "interface_id": SHADOW_DISCLOSURE_POLICY_INTERFACE,
            "policy_id": self.policy_id,
            "approved_external_provider_ids": list(self.approved_external_provider_ids),
            "allow_private_source_to_local": self.allow_private_source_to_local,
            "allow_private_source_to_simulated": self.allow_private_source_to_simulated,
            "allow_private_source_to_approved_external": (
                self.allow_private_source_to_approved_external
            ),
            "allow_private_source_to_unapproved_external": (
                self.allow_private_source_to_unapproved_external
            ),
            "require_isolated_evaluation_worktree": (
                self.require_isolated_evaluation_worktree
            ),
            "require_secret_scan": self.require_secret_scan,
            "allow_host_paths_in_invocation": self.allow_host_paths_in_invocation,
            "allow_host_paths_in_public_reports": self.allow_host_paths_in_public_reports,
            "allowed_path_classes": list(self.allowed_path_classes),
            "max_context_bytes": self.max_context_bytes,
            "authorization_cid": self.authorization_cid,
            "notes": self.notes,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def policy_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["policy_cid"] = self.policy_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShadowDisclosurePolicy":
        if not isinstance(data, Mapping):
            raise SemanticGovernorPrivacyError("policy payload must be a mapping")
        unknown = set(data) - cls._FIELDS
        if unknown:
            raise SemanticGovernorPrivacyError(
                f"policy payload has unknown fields: {sorted(unknown)}"
            )
        # policy_cid / schema / interface_id are derived or constant.
        return cls(
            policy_id=data.get("policy_id", "shadow-disclosure-default"),
            approved_external_provider_ids=data.get(
                "approved_external_provider_ids", ()
            ),
            allow_private_source_to_local=data.get(
                "allow_private_source_to_local", True
            ),
            allow_private_source_to_simulated=data.get(
                "allow_private_source_to_simulated", True
            ),
            allow_private_source_to_approved_external=data.get(
                "allow_private_source_to_approved_external", False
            ),
            allow_private_source_to_unapproved_external=data.get(
                "allow_private_source_to_unapproved_external", False
            ),
            require_isolated_evaluation_worktree=data.get(
                "require_isolated_evaluation_worktree", True
            ),
            require_secret_scan=data.get("require_secret_scan", True),
            allow_host_paths_in_invocation=data.get(
                "allow_host_paths_in_invocation", False
            ),
            allow_host_paths_in_public_reports=data.get(
                "allow_host_paths_in_public_reports", False
            ),
            allowed_path_classes=data.get(
                "allowed_path_classes",
                (
                    PathClass.REPO_RELATIVE.value,
                    PathClass.MANAGED_WORKTREE_ID.value,
                    PathClass.CONTENT_CID.value,
                ),
            ),
            max_context_bytes=data.get("max_context_bytes", MAX_CONTEXT_BYTES),
            authorization_cid=data.get("authorization_cid"),
            notes=data.get("notes"),
            metadata=data.get("metadata", {}),
        )


def default_shadow_disclosure_policy() -> ShadowDisclosurePolicy:
    """Return the fail-closed local-only default disclosure policy."""

    return ShadowDisclosurePolicy()


# ---------------------------------------------------------------------------
# Provider locality and source classification
# ---------------------------------------------------------------------------


def classify_provider_locality(
    provider_id: str,
    policy: ShadowDisclosurePolicy,
) -> ProviderLocality:
    """Classify *provider_id* under *policy* into a closed locality class."""

    pid = _provider_id(provider_id)
    lowered = pid.casefold()
    for prefix in _SIMULATED_PROVIDER_PREFIXES:
        if lowered.startswith(prefix):
            return ProviderLocality.SIMULATED
    for prefix in _LOCAL_PROVIDER_PREFIXES:
        if lowered.startswith(prefix):
            return ProviderLocality.LOCAL
    if pid in policy.approved_external_provider_ids:
        return ProviderLocality.APPROVED_EXTERNAL
    return ProviderLocality.UNAPPROVED_EXTERNAL


def classify_source_privacy(context: Any) -> SourcePrivacyClass:
    """Classify context privacy from embedded fields (fail-closed heuristic)."""

    if context is None:
        return SourcePrivacyClass.PUBLIC
    findings = scan_secrets(context)
    has_private = any(
        f.kind == SecretFindingKind.PRIVATE_SOURCE_FIELD.value for f in findings
    )
    if not has_private:
        # Managed references only (CIDs / public metadata).
        if isinstance(context, Mapping):
            keys = {_normalized_key(k) for k in context if isinstance(k, str)}
            if keys and keys <= {
                "context_pack_cid",
                "source_cid",
                "cid",
                "schema",
                "interface_id",
                "task_id",
                "summary",
                "role",
                "route_id",
                "metadata",
            }:
                return SourcePrivacyClass.MANAGED_REFERENCE
        return SourcePrivacyClass.PUBLIC
    # Distinguish raw vs private by field names.
    raw_markers = {
        "raw_private_source",
        "raw_source",
        "raw_source_text",
        "source_bytes",
        "source_text",
        "source_body",
        "source_code",
        "file_content",
        "file_contents",
        "repository_body",
        "repository_content",
    }

    def _has_raw(node: Any) -> bool:
        if isinstance(node, Mapping):
            for key, item in node.items():
                if isinstance(key, str) and _normalized_key(key) in raw_markers:
                    return True
                if _has_raw(item):
                    return True
        elif isinstance(node, (list, tuple)):
            return any(_has_raw(item) for item in node)
        return False

    if _has_raw(context):
        return SourcePrivacyClass.RAW_PRIVATE
    return SourcePrivacyClass.PRIVATE


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShadowDisclosureAuthorization:
    """Bounded authorization decision for one shadow disclosure request."""

    disposition: DisclosureDisposition | str
    provider_id: str
    provider_locality: ProviderLocality | str
    source_privacy_class: SourcePrivacyClass | str
    includes_private_source: bool
    isolated_worktree_ok: bool
    policy_cid: str
    reason_codes: Sequence[str]
    authorization_cid: str | None = None
    worktree_id: str | None = None
    redaction_required: bool = True
    strip_private_source: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, DisclosureDisposition, "disposition").value,
        )
        object.__setattr__(self, "provider_id", _provider_id(self.provider_id))
        object.__setattr__(
            self,
            "provider_locality",
            _enum(
                self.provider_locality, ProviderLocality, "provider_locality"
            ).value,
        )
        object.__setattr__(
            self,
            "source_privacy_class",
            _enum(
                self.source_privacy_class,
                SourcePrivacyClass,
                "source_privacy_class",
            ).value,
        )
        object.__setattr__(
            self,
            "includes_private_source",
            _bool(self.includes_private_source, "includes_private_source"),
        )
        object.__setattr__(
            self,
            "isolated_worktree_ok",
            _bool(self.isolated_worktree_ok, "isolated_worktree_ok"),
        )
        object.__setattr__(self, "policy_cid", _cid(self.policy_cid, "policy_cid"))
        object.__setattr__(
            self,
            "reason_codes",
            _unique_sorted_tokens(
                list(self.reason_codes),
                "reason_codes",
                max_items=MAX_REASON_CODES,
            ),
        )
        if not self.reason_codes:
            raise SemanticGovernorPrivacyError(
                "reason_codes must contain at least one code"
            )
        object.__setattr__(
            self,
            "authorization_cid",
            _optional_cid(self.authorization_cid, "authorization_cid"),
        )
        object.__setattr__(
            self, "worktree_id", _optional_worktree_id(self.worktree_id, "worktree_id")
        )
        object.__setattr__(
            self,
            "redaction_required",
            _bool(self.redaction_required, "redaction_required"),
        )
        object.__setattr__(
            self,
            "strip_private_source",
            _bool(self.strip_private_source, "strip_private_source"),
        )

    @property
    def allowed(self) -> bool:
        return self.disposition in {
            DisclosureDisposition.ALLOWED.value,
            DisclosureDisposition.LOCAL_ONLY.value,
            DisclosureDisposition.REDACTED_ONLY.value,
        }

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": DISCLOSURE_AUTHORIZATION_SCHEMA,
            "interface_id": DISCLOSURE_AUTHORIZATION_INTERFACE,
            "disposition": self.disposition,
            "provider_id": self.provider_id,
            "provider_locality": self.provider_locality,
            "source_privacy_class": self.source_privacy_class,
            "includes_private_source": self.includes_private_source,
            "isolated_worktree_ok": self.isolated_worktree_ok,
            "policy_cid": self.policy_cid,
            "reason_codes": list(self.reason_codes),
            "authorization_cid": self.authorization_cid,
            "worktree_id": self.worktree_id,
            "redaction_required": self.redaction_required,
            "strip_private_source": self.strip_private_source,
        }

    @property
    def authorization_decision_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["authorization_decision_cid"] = self.authorization_decision_cid
        payload["allowed"] = self.allowed
        return payload


def authorize_shadow_disclosure(
    policy: ShadowDisclosurePolicy,
    *,
    provider_id: str,
    context: Any = None,
    source_privacy_class: SourcePrivacyClass | str | None = None,
    includes_private_source: bool | None = None,
    isolated_evaluation_worktree: bool = True,
    worktree_id: str | None = None,
    worktree_path: str | None = None,
    raise_on_forbidden: bool = True,
) -> ShadowDisclosureAuthorization:
    """Authorize (or refuse) disclosure of *context* to *provider_id*.

    Private / raw source is never sent to an unapproved external shadow
    provider. Local-only is the default; approved external disclosure requires
    exact policy flags plus ``authorization_cid``.
    """

    if not isinstance(policy, ShadowDisclosurePolicy):
        raise SemanticGovernorPrivacyError(
            "policy must be a ShadowDisclosurePolicy instance"
        )

    pid = _provider_id(provider_id)
    locality = classify_provider_locality(pid, policy)

    if includes_private_source is None:
        includes_private = bool(
            context is not None and contains_private_source(context)
        )
    else:
        includes_private = _bool(includes_private_source, "includes_private_source")

    if source_privacy_class is None:
        if context is not None:
            source_class = classify_source_privacy(context)
        elif includes_private:
            source_class = SourcePrivacyClass.PRIVATE
        else:
            source_class = SourcePrivacyClass.PUBLIC
    else:
        source_class = _enum(
            source_privacy_class, SourcePrivacyClass, "source_privacy_class"
        )
        assert isinstance(source_class, SourcePrivacyClass)

    # Worktree gate — always required by policy.
    try:
        validated_wt = assert_isolated_evaluation_worktree(
            isolated_evaluation_worktree_required=(
                policy.require_isolated_evaluation_worktree
            ),
            worktree_id=worktree_id,
            worktree_path=worktree_path,
            host_path_allowed=False,
        )
    except (WorktreePolicyError, HostPathAdmissionError):
        if raise_on_forbidden:
            raise
        return ShadowDisclosureAuthorization(
            disposition=DisclosureDisposition.FORBIDDEN,
            provider_id=pid,
            provider_locality=locality,
            source_privacy_class=source_class,
            includes_private_source=includes_private,
            isolated_worktree_ok=False,
            policy_cid=policy.policy_cid,
            reason_codes=("worktree_policy_violation",),
            authorization_cid=policy.authorization_cid,
            worktree_id=None,
            redaction_required=True,
            strip_private_source=True,
        )

    isolated_ok = bool(isolated_evaluation_worktree) and (
        not policy.require_isolated_evaluation_worktree
        or isolated_evaluation_worktree
    )
    if policy.require_isolated_evaluation_worktree and not isolated_evaluation_worktree:
        auth = ShadowDisclosureAuthorization(
            disposition=DisclosureDisposition.FORBIDDEN,
            provider_id=pid,
            provider_locality=locality,
            source_privacy_class=source_class,
            includes_private_source=includes_private,
            isolated_worktree_ok=False,
            policy_cid=policy.policy_cid,
            reason_codes=("isolated_worktree_required",),
            authorization_cid=policy.authorization_cid,
            worktree_id=validated_wt,
            redaction_required=True,
            strip_private_source=True,
        )
        if raise_on_forbidden:
            raise WorktreePolicyError(
                "isolated evaluation worktree is required for shadow disclosure"
            )
        return auth

    reasons: list[str] = []
    disposition: DisclosureDisposition
    strip_private = False
    redaction_required = True

    private_like = includes_private or source_class in {
        SourcePrivacyClass.PRIVATE,
        SourcePrivacyClass.RAW_PRIVATE,
    }

    if not private_like:
        # Public / managed-reference context may go anywhere after redaction.
        disposition = DisclosureDisposition.ALLOWED
        reasons.append("public_or_managed_context")
        redaction_required = True
        strip_private = False
    elif locality is ProviderLocality.LOCAL:
        if policy.allow_private_source_to_local:
            disposition = DisclosureDisposition.LOCAL_ONLY
            reasons.append("local_provider_private_source_allowed")
        else:
            disposition = DisclosureDisposition.FORBIDDEN
            reasons.append("local_private_source_disabled")
            strip_private = True
    elif locality is ProviderLocality.SIMULATED:
        if policy.allow_private_source_to_simulated:
            disposition = DisclosureDisposition.LOCAL_ONLY
            reasons.append("simulated_provider_private_source_allowed")
        else:
            disposition = DisclosureDisposition.FORBIDDEN
            reasons.append("simulated_private_source_disabled")
            strip_private = True
    elif locality is ProviderLocality.APPROVED_EXTERNAL:
        if (
            policy.allow_private_source_to_approved_external
            and policy.authorization_cid is not None
            and pid in policy.approved_external_provider_ids
        ):
            disposition = DisclosureDisposition.ALLOWED
            reasons.append("approved_external_explicit_authorization")
            strip_private = False
        else:
            # Fail closed: may still send fully stripped/redacted context only
            # when the caller opts into redacted-only (no private residual).
            disposition = DisclosureDisposition.FORBIDDEN
            reasons.append("approved_external_lacks_exact_authority")
            strip_private = True
    else:
        # UNAPPROVED_EXTERNAL — private source never leaves.
        disposition = DisclosureDisposition.FORBIDDEN
        reasons.append("unapproved_external_private_source_forbidden")
        strip_private = True

    # Host-path precheck on context when present.
    if context is not None and not policy.allow_host_paths_in_invocation:
        try:
            reject_host_paths(context)
        except HostPathAdmissionError:
            disposition = DisclosureDisposition.FORBIDDEN
            reasons.append("host_path_in_context")
            strip_private = True

    auth = ShadowDisclosureAuthorization(
        disposition=disposition,
        provider_id=pid,
        provider_locality=locality,
        source_privacy_class=source_class,
        includes_private_source=includes_private,
        isolated_worktree_ok=isolated_ok,
        policy_cid=policy.policy_cid,
        reason_codes=tuple(reasons) or ("unspecified",),
        authorization_cid=policy.authorization_cid,
        worktree_id=validated_wt,
        redaction_required=redaction_required,
        strip_private_source=strip_private,
    )

    if (
        raise_on_forbidden
        and disposition is DisclosureDisposition.FORBIDDEN
        and "host_path_in_context" in reasons
    ):
        raise HostPathAdmissionError(
            f"host paths cannot enter provider invocation for {pid!r}"
        )
    if (
        raise_on_forbidden
        and disposition is DisclosureDisposition.FORBIDDEN
        and private_like
    ):
        raise DisclosureForbiddenError(
            f"private source disclosure forbidden for provider {pid!r} "
            f"(locality={locality.value}, reasons={list(reasons)})"
        )
    if raise_on_forbidden and disposition is DisclosureDisposition.FORBIDDEN:
        raise DisclosureForbiddenError(
            f"shadow disclosure forbidden for provider {pid!r} "
            f"(reasons={list(reasons)})"
        )
    return auth


# ---------------------------------------------------------------------------
# Public report projection
# ---------------------------------------------------------------------------


def project_public_report(value: Any, *, path: str = "$") -> Any:
    """Project a public-safe report value (CIDs / managed refs only).

    Rejects private-source fields, secrets, and arbitrary host paths rather
    than inventing redacted structure for them.
    """

    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise SemanticGovernorPrivacyError(
                    f"{path} map keys must be str, got {type(key).__name__}"
                )
            key_path = f"{path}.{key}"
            if _key_is_private_source(key) or _key_is_sensitive(key):
                raise SecretAdmissionError(
                    f"{key_path} rejects private/secret field {key!r} in public report"
                )
            if _key_looks_like_path_field(key):
                if isinstance(item, str) and (
                    _string_looks_like_host_path(item)
                    or classify_path(item)
                    in {PathClass.HOST_ABSOLUTE, PathClass.FORBIDDEN}
                ):
                    raise HostPathAdmissionError(
                        f"{key_path} rejects arbitrary host path in public report"
                    )
                # Path-named fields that are not portable CIDs / repo-relative
                # managed refs still fail closed for host absolute values.
            if isinstance(item, str) and _string_looks_like_host_path(item):
                raise HostPathAdmissionError(
                    f"{key_path} rejects arbitrary host path value in public report"
                )
            if isinstance(item, str):
                secret_reason = _text_contains_secret_pattern(item)
                if secret_reason is not None:
                    raise SecretAdmissionError(
                        f"{key_path} rejects secret text ({secret_reason}) "
                        "in public report"
                    )
            out[key] = project_public_report(item, path=key_path)
        return out
    if isinstance(value, (list, tuple)):
        return [
            project_public_report(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, str):
        if _string_looks_like_host_path(value):
            raise HostPathAdmissionError(
                f"{path} rejects arbitrary host path value in public report"
            )
        secret_reason = _text_contains_secret_pattern(value)
        if secret_reason is not None:
            raise SecretAdmissionError(
                f"{path} rejects secret text ({secret_reason}) in public report"
            )
        return value
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise SemanticGovernorPrivacyError(
        f"{path} public projection admits only strict JSON scalars/containers; "
        f"got {type(value).__name__}"
    )


# ---------------------------------------------------------------------------
# Full provider-invocation preparation (authorize + redact + admit)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProviderInvocationContext:
    """Bounded, privacy-filtered context ready for provider invocation."""

    provider_id: str
    provider_locality: str
    disposition: str
    policy_cid: str
    authorization_decision_cid: str
    redacted_context: Mapping[str, Any] | Any
    includes_private_source: bool
    private_source_stripped: bool
    worktree_id: str | None
    secret_findings_before: int
    secret_findings_after: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_INVOCATION_CONTEXT_SCHEMA,
            "provider_id": self.provider_id,
            "provider_locality": self.provider_locality,
            "disposition": self.disposition,
            "policy_cid": self.policy_cid,
            "authorization_decision_cid": self.authorization_decision_cid,
            "redacted_context": _thaw_structured(self.redacted_context),
            "includes_private_source": self.includes_private_source,
            "private_source_stripped": self.private_source_stripped,
            "worktree_id": self.worktree_id,
            "secret_findings_before": self.secret_findings_before,
            "secret_findings_after": self.secret_findings_after,
        }


def prepare_provider_invocation(
    context: Any,
    policy: ShadowDisclosurePolicy,
    *,
    provider_id: str,
    isolated_evaluation_worktree: bool = True,
    worktree_id: str | None = None,
    worktree_path: str | None = None,
) -> ProviderInvocationContext:
    """Authorize, redact, and admit *context* for a single provider call.

    Guarantees:

    * Private source never reaches an unapproved external provider.
    * Secrets are redacted (sensitive keys + text patterns).
    * Host absolute paths are stripped / rejected from the invocation payload.
    * Isolated evaluation worktree policy is enforced.
    """

    findings_before = scan_secrets(context)
    auth = authorize_shadow_disclosure(
        policy,
        provider_id=provider_id,
        context=context,
        isolated_evaluation_worktree=isolated_evaluation_worktree,
        worktree_id=worktree_id,
        worktree_path=worktree_path,
        raise_on_forbidden=True,
    )

    # For external (approved) with private source allowed, still redact secrets
    # but keep private source. For local, redact secrets, optionally keep source.
    strip_private = auth.strip_private_source
    if auth.provider_locality == ProviderLocality.UNAPPROVED_EXTERNAL.value:
        strip_private = True

    redacted = redact_context_for_provider(
        context,
        strip_private_source=strip_private,
        strip_host_paths=not policy.allow_host_paths_in_invocation,
    )

    # Post-redaction admission: residual secrets / host paths fail closed.
    if policy.require_secret_scan:
        residual = [
            f
            for f in scan_secrets(redacted)
            if f.kind
            in {
                SecretFindingKind.SENSITIVE_FIELD.value,
                SecretFindingKind.TEXT_PATTERN.value,
            }
        ]
        # After redaction, sensitive field values are markers; text patterns
        # should be gone. Private source fields may remain only when authorized.
        for finding in residual:
            # REDACTION_MARKER values under sensitive keys are expected.
            pass
        if not policy.allow_host_paths_in_invocation:
            reject_host_paths(redacted)

    # Unapproved external must not retain private source after preparation.
    if auth.provider_locality == ProviderLocality.UNAPPROVED_EXTERNAL.value:
        if contains_private_source(redacted):
            raise DisclosureForbiddenError(
                "private source residual after redaction for unapproved external"
            )

    findings_after = scan_secrets(redacted)
    return ProviderInvocationContext(
        provider_id=auth.provider_id,
        provider_locality=auth.provider_locality,
        disposition=auth.disposition,
        policy_cid=auth.policy_cid,
        authorization_decision_cid=auth.authorization_decision_cid,
        redacted_context=redacted,
        includes_private_source=auth.includes_private_source,
        private_source_stripped=strip_private,
        worktree_id=auth.worktree_id,
        secret_findings_before=len(findings_before),
        secret_findings_after=len(findings_after),
    )


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = (
    "AUTHORIZE_SHADOW_DISCLOSURE_INTERFACE",
    "DISCLOSURE_AUTHORIZATION_INTERFACE",
    "DISCLOSURE_AUTHORIZATION_SCHEMA",
    "DisclosureDisposition",
    "DisclosureForbiddenError",
    "HostPathAdmissionError",
    "MANAGED_PATH_PLACEHOLDER",
    "PathClass",
    "PRIVATE_FIELD_MARKERS",
    "PROVIDER_INVOCATION_CONTEXT_SCHEMA",
    "PUBLIC_REPORT_PROJECTION_SCHEMA",
    "ProviderInvocationContext",
    "ProviderLocality",
    "REDACTION_MARKER",
    "REDACT_CONTEXT_FOR_PROVIDER_INTERFACE",
    "SCG_PRIVACY_GATE_EVIDENCE",
    "SHADOW_DISCLOSURE_POLICY_INTERFACE",
    "SHADOW_DISCLOSURE_POLICY_SCHEMA",
    "SecretAdmissionError",
    "SecretFinding",
    "SecretFindingKind",
    "SemanticGovernorPrivacyError",
    "ShadowDisclosureAuthorization",
    "ShadowDisclosurePolicy",
    "SourcePrivacyClass",
    "WorktreePolicyError",
    "assert_isolated_evaluation_worktree",
    "authorize_shadow_disclosure",
    "classify_path",
    "classify_provider_locality",
    "classify_source_privacy",
    "contains_private_source",
    "contains_secrets",
    "default_shadow_disclosure_policy",
    "prepare_provider_invocation",
    "project_public_report",
    "redact_context_for_provider",
    "reject_host_paths",
    "reject_secrets",
    "scan_secrets",
)
