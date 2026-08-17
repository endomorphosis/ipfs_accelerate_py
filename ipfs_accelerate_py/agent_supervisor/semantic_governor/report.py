"""Privacy-filtered governor report and dashboard-data projections (SCG-039).

Produces machine-readable **summary** and **detail** projections over governor
histories and metrics. Conflict policy: projection only — no graphical
dashboard, no public server, and no ambient provider/network I/O on import.

Normative fail-closed rules:

* **Privacy** — raw private source, secrets, and arbitrary host filesystem
  paths are rejected from every public projection. Outputs store CIDs and
  managed references only (via :func:`project_public_report`).
* **Authority** — human/model free-form authority claims are rejected; models
  and free-form narrative cannot authorize promotion, trusted keys, or
  proof scope.
* **Unavailable is explicit** — missing measurements are ``None`` or listed in
  ``unavailable_fields``, never fabricated success or zero-as-success.
* **Cohort honesty** — live and simulated evidence stay labeled and separate.
* **Bounded claims** — seal/proof scope fields are closed tokens; overclaim
  kinds are rejected.
* **Determinism** — sealed payloads recompute content identity; identical
  inputs yield identical CIDs.

Interfaces: :func:`build_governor_report`, :func:`build_dashboard_data`.

Importing this module performs no I/O and never invokes a provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Iterable, Mapping, Sequence
import re
import unicodedata

from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_structured,
    validate_cid,
    validate_structured_value,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.base import (
    MODEL_AUTHORITY_FORBIDDEN_KEYS,
    PRIVATE_FIELD_MARKERS,
    SemanticGovernorBaseError,
    reject_private_and_model_authority,
)

from ipfs_accelerate_py.agent_supervisor.semantic_governor.privacy import (
    HostPathAdmissionError,
    SecretAdmissionError,
    SemanticGovernorPrivacyError,
    classify_path,
    contains_private_source,
    reject_host_paths,
)

# Optional metrics surface (SCG-038). Imported lazily in helpers so this
# module remains importable even when only projection types are needed.
try:  # pragma: no cover - import side is deterministic
    from ipfs_accelerate_py.agent_supervisor.semantic_governor.metrics import (
        GovernorMetricReport,
    )
except Exception:  # pragma: no cover
    GovernorMetricReport = None  # type: ignore[misc, assignment]


# ---------------------------------------------------------------------------
# Evidence / interface / schema constants
# ---------------------------------------------------------------------------

SCG_DASHBOARD_DATA_EVIDENCE: Final[str] = "scg/dashboard-data@1"
SCG_FINAL_REPORT_EVIDENCE: Final[str] = "scg/final-report@1"

BUILD_GOVERNOR_REPORT_INTERFACE: Final[str] = "build_governor_report@1"
BUILD_DASHBOARD_DATA_INTERFACE: Final[str] = "build_dashboard_data@1"
GOVERNOR_REPORT_INTERFACE: Final[str] = "GovernorReport@1"
DASHBOARD_DATA_INTERFACE: Final[str] = "DashboardData@1"

GOVERNOR_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/governor-report@1"
)
DASHBOARD_DATA_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-governor/dashboard-data@1"
)

GENERATOR_ID: Final[str] = "semantic_governor_report"
GENERATOR_VERSION: Final[str] = "1.0.0"
PRODUCER_ID: Final[str] = "semantic_governor"
PRODUCER_VERSION: Final[str] = "1"
TOOL_ID: Final[str] = "report.v1"

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_METADATA_KEYS: Final[int] = 64
MAX_CID_LIST: Final[int] = 4_096
MAX_TOKEN_LIST: Final[int] = 1_024
MAX_UNAVAILABLE_FIELDS: Final[int] = 512
MAX_RISKS: Final[int] = 256
MAX_INTERFACES: Final[int] = 512
MAX_COMMITS: Final[int] = 512
MAX_COUNTER: Final[int] = 2**63 - 1

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")
_COMMIT_SHA_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{7,64}$")
_INTERFACE_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z][A-Za-z0-9_./:@+-]{0,255}$"
)

# Plan §16 final-report sections that must always be representable.
REQUIRED_FINAL_REPORT_FIELDS: Final[tuple[str, ...]] = (
    "inspected_commits",
    "implemented_commits",
    "consumed_interfaces",
    "audit_population",
    "differential_outcomes",
    "omission_detection",
    "expansion",
    "final_context_reduction",
    "route_distribution",
    "quality",
    "overhead_and_cost",
    "rules",
    "rollback",
    "seal_scope",
    "proof_scope",
    "heuristics",
    "remaining_production_risks",
    "unavailable_fields",
    "live_metrics_cid",
    "simulated_metrics_present",
    "metric_report_cid",
    "evidence_mode",
)

# Free-form human/model authority markers rejected from public projections.
FREE_FORM_AUTHORITY_FORBIDDEN_KEYS: Final[frozenset[str]] = frozenset(
    set(MODEL_AUTHORITY_FORBIDDEN_KEYS)
    | {
        "ad_hoc_authority",
        "free_form_authority",
        "free_form_authorization",
        "freeform_authority",
        "freeform_authorization",
        "human_authority",
        "human_free_form",
        "human_free_form_authority",
        "human_override_authority",
        "llm_free_form_authority",
        "model_free_form",
        "model_free_form_authority",
        "narrative_authority",
        "unstructured_authority",
        "verbal_authority",
    }
)

# Closed seal/proof scope status vocabulary.
class EvidenceMode(str, Enum):
    """How evidence in the report was produced."""

    LIVE = "live"
    SIMULATED = "simulated"
    MIXED = "mixed"
    UNAVAILABLE = "unavailable"


class SealScopeStatus(str, Enum):
    """Closed seal status tokens for report projection."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    BOUND = "bound"
    BLOCKED = "blocked"
    SIMULATED = "simulated"


class ProofScopeKind(str, Enum):
    """Closed proof-scope kinds (bounded claim surface, plan §14)."""

    BOUNDED_ARTIFACT_EVALUATION = "bounded_artifact_evaluation"
    STRUCTURAL_NON_ZK = "structural_non_zk"
    UNAVAILABLE = "unavailable"
    HEURISTIC_ONLY = "heuristic_only"
    NONE = "none"


class HeuristicClass(str, Enum):
    """Closed heuristic classification labels."""

    NONE = "none"
    PRESENT = "present"
    EXCLUDED_FROM_EXACT = "excluded_from_exact"
    UNAVAILABLE = "unavailable"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ReportError(SemanticGovernorBaseError):
    """Raised when report/dashboard inputs fail closed."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "report_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})


class FreeFormAuthorityError(ReportError):
    """Human/model free-form authority is not admitted into public reports."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            reason_code=kwargs.pop("reason_code", "free_form_authority_rejected"),
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        raise ReportError(f"{name} must be a nonempty string")
    text = _normalize_text(value)
    if not empty and not text:
        raise ReportError(f"{name} must be a nonempty string")
    if value != text:
        raise ReportError(f"{name} must be trimmed NFC text")
    if len(text) > MAX_TEXT_CHARS or any(not char.isprintable() for char in text):
        raise ReportError(f"{name} contains invalid text")
    return text


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if _TOKEN_RE.fullmatch(text) is None:
        raise ReportError(
            f"{name} must be a lowercase token matching {_TOKEN_RE.pattern}"
        )
    return text


def _optional_token(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _token(value, name)


def _cid(value: Any, name: str) -> str:
    try:
        return validate_cid(value)
    except Exception as exc:
        raise ReportError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ReportError(f"{name} must be a boolean")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise ReportError(f"{name} must be a nonnegative integer")
    if value > MAX_COUNTER:
        raise ReportError(f"{name} exceeds maximum")
    return value


def _optional_nonneg_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _nonneg_int(value, name)


def _enum_value(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    if type(value) is str:
        try:
            return enum_type(value).value
        except ValueError as exc:
            raise ReportError(f"{name} has unsupported value {value!r}") from exc
    raise ReportError(f"{name} must be a {enum_type.__name__} or string")


def _freeze_structured(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_structured(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_structured(item) for item in value)
    return value


def _thaw_structured(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_structured(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_structured(item) for item in value]
    return value


def _key_is_free_form_authority(name: str) -> bool:
    lowered = name.lower()
    if lowered in FREE_FORM_AUTHORITY_FORBIDDEN_KEYS:
        return True
    for marker in FREE_FORM_AUTHORITY_FORBIDDEN_KEYS:
        if marker in lowered:
            return True
    return False


def reject_free_form_authority(value: Any, *, path: str = "$") -> None:
    """Fail closed when free-form human/model authority fields are present."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise FreeFormAuthorityError(
                    f"{path} map keys must be str, got {type(key).__name__}"
                )
            key_path = f"{path}.{key}"
            if _key_is_free_form_authority(key):
                raise FreeFormAuthorityError(
                    f"{key_path} rejects free-form human/model authority field "
                    f"{key!r}"
                )
            reject_free_form_authority(item, path=key_path)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_free_form_authority(item, path=f"{path}[{index}]")


# Auth/credential markers for report projection. Metric unit fields that use
# the word "tokens" (e.g. raw_tokens_total) are admitted; bare auth "token"
# fields are not. project_public_report cannot make that distinction.
_AUTH_SECRET_KEY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth_token",
        "authorization",
        "bearer_token",
        "client_secret",
        "cookie",
        "credential",
        "credentials",
        "github_token",
        "id_token",
        "password",
        "passphrase",
        "private_key",
        "refresh_token",
        "secret",
        "session_token",
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
    re.compile(r"(?i)\b(?:sk|pk|rk)-[A-Za-z0-9]{16,}\b"),
    re.compile(r"(?i)\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    re.compile(r"(?i)\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
)

_ABSOLUTE_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:/|[A-Za-z]:[\\/]|\\\\|file:)"
)


def _normalized_key(name: str) -> str:
    return name.strip().casefold().replace("-", "_").replace(" ", "_")


def _key_is_metric_token_count(name: str) -> bool:
    """True for compression/metric unit fields that use the word tokens."""

    lowered = _normalized_key(name)
    if "tokens" in lowered:
        return True
    if lowered in {"token_count", "token_total"}:
        return True
    if lowered.endswith("_token_count") or lowered.endswith("_token_total"):
        return True
    return False


def _key_is_report_secret(name: str) -> bool:
    lowered = _normalized_key(name)
    if lowered in PRIVATE_FIELD_MARKERS:
        return True
    for marker in PRIVATE_FIELD_MARKERS:
        if marker in lowered:
            return True
    if lowered in _AUTH_SECRET_KEY_MARKERS:
        return True
    for marker in _AUTH_SECRET_KEY_MARKERS:
        if marker in lowered:
            return True
    # Bare "token" auth fields — not metric *tokens* unit counts.
    if "token" in lowered and not _key_is_metric_token_count(name):
        if (
            lowered == "token"
            or lowered.endswith("_token")
            or lowered.startswith("token_")
            or "_token_" in lowered
        ):
            return True
    return False


def _key_is_private_source_field(name: str) -> bool:
    lowered = _normalized_key(name)
    markers = frozenset(PRIVATE_FIELD_MARKERS) | {
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
    }
    if lowered in markers:
        return True
    for marker in markers:
        if marker in lowered:
            return True
    return False


def _string_looks_like_host_path(value: str) -> bool:
    if not value:
        return False
    if _ABSOLUTE_PATH_RE.match(value) or value.startswith("~/"):
        return True
    return False


def _text_contains_secret_pattern(text: str) -> str | None:
    for pattern in _TEXT_SECRET_PATTERNS:
        if pattern.search(text):
            return "secret_text_pattern"
    return None


def project_governor_public(value: Any, *, path: str = "$") -> Any:
    """Project a public-safe governor report / dashboard payload.

    Rejects raw private source, secrets, arbitrary host paths, and free-form
    human/model authority. Metric unit fields named ``*tokens*`` are admitted
    (they are counts, not auth tokens).
    """

    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ReportError(
                    f"{path} map keys must be str, got {type(key).__name__}"
                )
            key_path = f"{path}.{key}"
            if _key_is_free_form_authority(key):
                raise FreeFormAuthorityError(
                    f"{key_path} rejects free-form human/model authority field "
                    f"{key!r}"
                )
            if _key_is_private_source_field(key):
                raise ReportError(
                    f"{key_path} rejects private source field {key!r}",
                    reason_code="private_source_rejected",
                )
            if _key_is_report_secret(key):
                raise ReportError(
                    f"{key_path} rejects secret field {key!r}",
                    reason_code="secret_field_rejected",
                )
            if isinstance(item, str) and _string_looks_like_host_path(item):
                raise ReportError(
                    f"{key_path} rejects arbitrary host path value",
                    reason_code="host_path_rejected",
                )
            if isinstance(item, str):
                secret_reason = _text_contains_secret_pattern(item)
                if secret_reason is not None:
                    raise ReportError(
                        f"{key_path} rejects secret text in public report",
                        reason_code="secret_text_rejected",
                    )
                # Path-named fields must not carry host absolute paths.
                lowered = _normalized_key(key)
                if (
                    lowered.endswith("_path")
                    or lowered.endswith("_dir")
                    or lowered in {
                        "path",
                        "workdir",
                        "workspace_path",
                        "worktree_path",
                        "repo_root",
                    }
                ):
                    try:
                        path_class = classify_path(item)
                    except Exception:
                        path_class = None
                    if path_class is not None and path_class.value in {
                        "host_absolute",
                        "forbidden",
                    }:
                        raise ReportError(
                            f"{key_path} rejects arbitrary host path",
                            reason_code="host_path_rejected",
                        )
            out[key] = project_governor_public(item, path=key_path)
        return out
    if isinstance(value, (list, tuple)):
        return [
            project_governor_public(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, str):
        if _string_looks_like_host_path(value):
            raise ReportError(
                f"{path} rejects arbitrary host path value",
                reason_code="host_path_rejected",
            )
        if _text_contains_secret_pattern(value) is not None:
            raise ReportError(
                f"{path} rejects secret text in public report",
                reason_code="secret_text_rejected",
            )
        return value
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ReportError(
        f"{path} public projection admits only strict JSON scalars/containers; "
        f"got {type(value).__name__}"
    )


def _admit_public_structured(value: Any, name: str) -> Any:
    """Validate strict DAG-JSON, reject private/authority, privacy-project."""

    thawed = _thaw_structured(value)
    try:
        validate_structured_value(thawed, path=name)
    except Exception as exc:
        raise ReportError(
            f"{name} must be strict DAG-JSON without floats or host types"
        ) from exc
    try:
        reject_private_and_model_authority(thawed, path=name)
    except SemanticGovernorBaseError as exc:
        raise ReportError(str(exc), reason_code="private_or_model_authority") from exc
    try:
        reject_free_form_authority(thawed, path=name)
    except FreeFormAuthorityError:
        raise
    if contains_private_source(thawed):
        raise ReportError(
            f"{name} rejects embedded private source",
            reason_code="private_source_rejected",
        )
    try:
        reject_host_paths(thawed, path=name)
    except (HostPathAdmissionError, SemanticGovernorPrivacyError) as exc:
        raise ReportError(
            str(exc), reason_code="host_path_rejected"
        ) from exc
    try:
        projected = project_governor_public(thawed, path=name)
    except FreeFormAuthorityError:
        raise
    except ReportError:
        raise
    except (SecretAdmissionError, HostPathAdmissionError, SemanticGovernorPrivacyError) as exc:
        raise ReportError(
            str(exc),
            reason_code="privacy_projection_rejected",
        ) from exc
    return projected


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ReportError(f"{name} must be a mapping")
    if len(value) > MAX_METADATA_KEYS:
        raise ReportError(f"{name} exceeds maximum key count")
    projected = _admit_public_structured(dict(value), name)
    if not isinstance(projected, dict):
        raise ReportError(f"{name} projection must remain a mapping")
    return _freeze_structured(projected)


def _unique_sorted_cids(values: Any, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ReportError(f"{name} must be a list or tuple")
    if len(values) > MAX_CID_LIST:
        raise ReportError(f"{name} exceeds maximum length")
    ordered = tuple(sorted({_cid(item, name) for item in values}))
    return ordered


def _unique_sorted_tokens(values: Any, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ReportError(f"{name} must be a list or tuple")
    if len(values) > MAX_TOKEN_LIST:
        raise ReportError(f"{name} exceeds maximum length")
    ordered = tuple(sorted({_token(item, name) for item in values}))
    return ordered


def _managed_commit_ref(value: Any, name: str) -> str:
    """Admit a git SHA or content CID as a managed commit reference."""

    text = _text(value, name)
    # Prefer CID validation when the value is a multibase CID.
    if text.startswith("baf") or text.startswith("bag") or text.startswith("Qm"):
        return _cid(text, name)
    if _COMMIT_SHA_RE.fullmatch(text) is None:
        raise ReportError(
            f"{name} must be a managed git SHA or content CID, not a host path"
        )
    # Reject path-like SHAs that slipped past (defensive).
    if "/" in text or "\\" in text or text.startswith("~"):
        raise ReportError(f"{name} rejects path-like commit references")
    return text


def _unique_sorted_commits(values: Any, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ReportError(f"{name} must be a list or tuple")
    if len(values) > MAX_COMMITS:
        raise ReportError(f"{name} exceeds maximum length")
    ordered = tuple(sorted({_managed_commit_ref(item, name) for item in values}))
    return ordered


def _interface_id(value: Any, name: str) -> str:
    text = _text(value, name)
    if _INTERFACE_RE.fullmatch(text) is None:
        raise ReportError(f"{name} has invalid interface id form")
    return text


def _unique_sorted_interfaces(values: Any, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ReportError(f"{name} must be a list or tuple")
    if len(values) > MAX_INTERFACES:
        raise ReportError(f"{name} exceeds maximum length")
    ordered = tuple(sorted({_interface_id(item, name) for item in values}))
    return ordered


def _risk_tokens(values: Any, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ReportError(f"{name} must be a list or tuple")
    if len(values) > MAX_RISKS:
        raise ReportError(f"{name} exceeds maximum length")
    # Risks are closed tokens, not free-form prose.
    ordered = tuple(sorted({_token(item, name) for item in values}))
    return ordered


def _unavailable_field_names(values: Any, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ReportError(f"{name} must be a list or tuple")
    if len(values) > MAX_UNAVAILABLE_FIELDS:
        raise ReportError(f"{name} exceeds maximum length")
    ordered = tuple(sorted({_token(item, name) for item in values}))
    return ordered


def _count_map(values: Any, name: str) -> Mapping[str, int]:
    if values is None:
        return MappingProxyType({})
    if not isinstance(values, Mapping):
        raise ReportError(f"{name} must be a mapping")
    if len(values) > MAX_METADATA_KEYS:
        raise ReportError(f"{name} exceeds maximum key count")
    out: dict[str, int] = {}
    for key, item in values.items():
        token = _token(key, f"{name}.key")
        out[token] = _nonneg_int(item, f"{name}.{token}")
    return MappingProxyType(dict(sorted(out.items())))


# ---------------------------------------------------------------------------
# Nested report sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuditPopulationSection:
    """Audit population counts with live/simulated separation."""

    total_audits: int | None = None
    live_audits: int | None = None
    simulated_audits: int | None = None
    source_receipt_cids: Sequence[str] = ()
    history_cids: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "total_audits",
            _optional_nonneg_int(self.total_audits, "total_audits"),
        )
        object.__setattr__(
            self, "live_audits", _optional_nonneg_int(self.live_audits, "live_audits")
        )
        object.__setattr__(
            self,
            "simulated_audits",
            _optional_nonneg_int(self.simulated_audits, "simulated_audits"),
        )
        object.__setattr__(
            self,
            "source_receipt_cids",
            _unique_sorted_cids(self.source_receipt_cids, "source_receipt_cids"),
        )
        object.__setattr__(
            self, "history_cids", _unique_sorted_cids(self.history_cids, "history_cids")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_audits": self.total_audits,
            "live_audits": self.live_audits,
            "simulated_audits": self.simulated_audits,
            "source_receipt_cids": list(self.source_receipt_cids),
            "history_cids": list(self.history_cids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "AuditPopulationSection":
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ReportError("audit_population must be a mapping")
        return cls(
            total_audits=data.get("total_audits"),
            live_audits=data.get("live_audits"),
            simulated_audits=data.get("simulated_audits"),
            source_receipt_cids=tuple(data.get("source_receipt_cids") or ()),
            history_cids=tuple(data.get("history_cids") or ()),
        )

    @classmethod
    def unavailable(cls) -> "AuditPopulationSection":
        return cls()


@dataclass(frozen=True, slots=True)
class DifferentialOutcomesSection:
    """Comparative differential outcome distribution."""

    outcome_counts: Mapping[str, int] = field(default_factory=dict)
    equivalent_success_count: int | None = None
    compressed_failed_expanded_succeeded_count: int | None = None
    both_failed_count: int | None = None
    verification_inconclusive_count: int | None = None
    unavailable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "outcome_counts", _count_map(self.outcome_counts, "outcome_counts")
        )
        object.__setattr__(
            self,
            "equivalent_success_count",
            _optional_nonneg_int(
                self.equivalent_success_count, "equivalent_success_count"
            ),
        )
        object.__setattr__(
            self,
            "compressed_failed_expanded_succeeded_count",
            _optional_nonneg_int(
                self.compressed_failed_expanded_succeeded_count,
                "compressed_failed_expanded_succeeded_count",
            ),
        )
        object.__setattr__(
            self,
            "both_failed_count",
            _optional_nonneg_int(self.both_failed_count, "both_failed_count"),
        )
        object.__setattr__(
            self,
            "verification_inconclusive_count",
            _optional_nonneg_int(
                self.verification_inconclusive_count,
                "verification_inconclusive_count",
            ),
        )
        object.__setattr__(self, "unavailable", _bool(self.unavailable, "unavailable"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome_counts": dict(self.outcome_counts),
            "equivalent_success_count": self.equivalent_success_count,
            "compressed_failed_expanded_succeeded_count": (
                self.compressed_failed_expanded_succeeded_count
            ),
            "both_failed_count": self.both_failed_count,
            "verification_inconclusive_count": self.verification_inconclusive_count,
            "unavailable": self.unavailable,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "DifferentialOutcomesSection":
        if data is None:
            return cls(unavailable=True)
        if not isinstance(data, Mapping):
            raise ReportError("differential_outcomes must be a mapping")
        return cls(
            outcome_counts=dict(data.get("outcome_counts") or {}),
            equivalent_success_count=data.get("equivalent_success_count"),
            compressed_failed_expanded_succeeded_count=data.get(
                "compressed_failed_expanded_succeeded_count"
            ),
            both_failed_count=data.get("both_failed_count"),
            verification_inconclusive_count=data.get(
                "verification_inconclusive_count"
            ),
            unavailable=bool(data.get("unavailable", False)),
        )

    @classmethod
    def unavailable_section(cls) -> "DifferentialOutcomesSection":
        return cls(unavailable=True)


@dataclass(frozen=True, slots=True)
class OmissionDetectionSection:
    """Omission detection and critical-acceptance projection."""

    intentional_omission_count: int | None = None
    detected_before_execution_count: int | None = None
    detected_after_execution_count: int | None = None
    critical_omission_count: int | None = None
    critical_omissions_accepted_count: int | None = None
    detection_before_rate_bp: int | None = None
    critical_acceptance_rate_bp: int | None = None
    false_alarm_count: int | None = None
    unavailable: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "intentional_omission_count",
            "detected_before_execution_count",
            "detected_after_execution_count",
            "critical_omission_count",
            "critical_omissions_accepted_count",
            "detection_before_rate_bp",
            "critical_acceptance_rate_bp",
            "false_alarm_count",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_nonneg_int(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "unavailable", _bool(self.unavailable, "unavailable"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "intentional_omission_count": self.intentional_omission_count,
            "detected_before_execution_count": self.detected_before_execution_count,
            "detected_after_execution_count": self.detected_after_execution_count,
            "critical_omission_count": self.critical_omission_count,
            "critical_omissions_accepted_count": (
                self.critical_omissions_accepted_count
            ),
            "detection_before_rate_bp": self.detection_before_rate_bp,
            "critical_acceptance_rate_bp": self.critical_acceptance_rate_bp,
            "false_alarm_count": self.false_alarm_count,
            "unavailable": self.unavailable,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "OmissionDetectionSection":
        if data is None:
            return cls(unavailable=True)
        if not isinstance(data, Mapping):
            raise ReportError("omission_detection must be a mapping")
        return cls(
            intentional_omission_count=data.get("intentional_omission_count"),
            detected_before_execution_count=data.get(
                "detected_before_execution_count"
            ),
            detected_after_execution_count=data.get("detected_after_execution_count"),
            critical_omission_count=data.get("critical_omission_count"),
            critical_omissions_accepted_count=data.get(
                "critical_omissions_accepted_count"
            ),
            detection_before_rate_bp=data.get("detection_before_rate_bp"),
            critical_acceptance_rate_bp=data.get("critical_acceptance_rate_bp"),
            false_alarm_count=data.get("false_alarm_count"),
            unavailable=bool(data.get("unavailable", False)),
        )

    @classmethod
    def unavailable_section(cls) -> "OmissionDetectionSection":
        return cls(unavailable=True)


@dataclass(frozen=True, slots=True)
class ExpansionSection:
    """Expansion success and size projection."""

    expansion_count: int | None = None
    expansion_rate_bp: int | None = None
    expansion_true_positive_count: int | None = None
    expansion_false_positive_count: int | None = None
    expansion_false_negative_count: int | None = None
    expansion_precision_bp: int | None = None
    expansion_recall_bp: int | None = None
    expanded_tokens_total: int | None = None
    unavailable: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "expansion_count",
            "expansion_rate_bp",
            "expansion_true_positive_count",
            "expansion_false_positive_count",
            "expansion_false_negative_count",
            "expansion_precision_bp",
            "expansion_recall_bp",
            "expanded_tokens_total",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_nonneg_int(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "unavailable", _bool(self.unavailable, "unavailable"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "expansion_count": self.expansion_count,
            "expansion_rate_bp": self.expansion_rate_bp,
            "expansion_true_positive_count": self.expansion_true_positive_count,
            "expansion_false_positive_count": self.expansion_false_positive_count,
            "expansion_false_negative_count": self.expansion_false_negative_count,
            "expansion_precision_bp": self.expansion_precision_bp,
            "expansion_recall_bp": self.expansion_recall_bp,
            "expanded_tokens_total": self.expanded_tokens_total,
            "unavailable": self.unavailable,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ExpansionSection":
        if data is None:
            return cls(unavailable=True)
        if not isinstance(data, Mapping):
            raise ReportError("expansion must be a mapping")
        return cls(
            expansion_count=data.get("expansion_count"),
            expansion_rate_bp=data.get("expansion_rate_bp"),
            expansion_true_positive_count=data.get("expansion_true_positive_count"),
            expansion_false_positive_count=data.get("expansion_false_positive_count"),
            expansion_false_negative_count=data.get("expansion_false_negative_count"),
            expansion_precision_bp=data.get("expansion_precision_bp"),
            expansion_recall_bp=data.get("expansion_recall_bp"),
            expanded_tokens_total=data.get("expanded_tokens_total"),
            unavailable=bool(data.get("unavailable", False)),
        )

    @classmethod
    def unavailable_section(cls) -> "ExpansionSection":
        return cls(unavailable=True)


@dataclass(frozen=True, slots=True)
class ContextReductionSection:
    """Final context reduction projection (basis points)."""

    median_context_reduction_bp: int | None = None
    mean_context_reduction_bp: int | None = None
    raw_tokens_total: int | None = None
    compressed_tokens_total: int | None = None
    unavailable: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "median_context_reduction_bp",
            "mean_context_reduction_bp",
            "raw_tokens_total",
            "compressed_tokens_total",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_nonneg_int(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "unavailable", _bool(self.unavailable, "unavailable"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "median_context_reduction_bp": self.median_context_reduction_bp,
            "mean_context_reduction_bp": self.mean_context_reduction_bp,
            "raw_tokens_total": self.raw_tokens_total,
            "compressed_tokens_total": self.compressed_tokens_total,
            "unavailable": self.unavailable,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ContextReductionSection":
        if data is None:
            return cls(unavailable=True)
        if not isinstance(data, Mapping):
            raise ReportError("final_context_reduction must be a mapping")
        return cls(
            median_context_reduction_bp=data.get("median_context_reduction_bp"),
            mean_context_reduction_bp=data.get("mean_context_reduction_bp"),
            raw_tokens_total=data.get("raw_tokens_total"),
            compressed_tokens_total=data.get("compressed_tokens_total"),
            unavailable=bool(data.get("unavailable", False)),
        )

    @classmethod
    def unavailable_section(cls) -> "ContextReductionSection":
        return cls(unavailable=True)


@dataclass(frozen=True, slots=True)
class RouteDistributionSection:
    """Route share and escalation projection."""

    route_share_counts: Mapping[str, int] = field(default_factory=dict)
    route_share_bp: Mapping[str, int | None] = field(default_factory=dict)
    escalation_count: int | None = None
    retry_count: int | None = None
    escalation_rate_bp: int | None = None
    unavailable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "route_share_counts",
            _count_map(self.route_share_counts, "route_share_counts"),
        )
        if self.route_share_bp is None:
            bp: Mapping[str, int | None] = MappingProxyType({})
        elif not isinstance(self.route_share_bp, Mapping):
            raise ReportError("route_share_bp must be a mapping")
        else:
            cleaned: dict[str, int | None] = {}
            for key, item in self.route_share_bp.items():
                token = _token(key, "route_share_bp.key")
                cleaned[token] = _optional_nonneg_int(item, f"route_share_bp.{token}")
            bp = MappingProxyType(dict(sorted(cleaned.items())))
        object.__setattr__(self, "route_share_bp", bp)
        object.__setattr__(
            self,
            "escalation_count",
            _optional_nonneg_int(self.escalation_count, "escalation_count"),
        )
        object.__setattr__(
            self, "retry_count", _optional_nonneg_int(self.retry_count, "retry_count")
        )
        object.__setattr__(
            self,
            "escalation_rate_bp",
            _optional_nonneg_int(self.escalation_rate_bp, "escalation_rate_bp"),
        )
        object.__setattr__(self, "unavailable", _bool(self.unavailable, "unavailable"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_share_counts": dict(self.route_share_counts),
            "route_share_bp": dict(self.route_share_bp),
            "escalation_count": self.escalation_count,
            "retry_count": self.retry_count,
            "escalation_rate_bp": self.escalation_rate_bp,
            "unavailable": self.unavailable,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RouteDistributionSection":
        if data is None:
            return cls(unavailable=True)
        if not isinstance(data, Mapping):
            raise ReportError("route_distribution must be a mapping")
        return cls(
            route_share_counts=dict(data.get("route_share_counts") or {}),
            route_share_bp=dict(data.get("route_share_bp") or {}),
            escalation_count=data.get("escalation_count"),
            retry_count=data.get("retry_count"),
            escalation_rate_bp=data.get("escalation_rate_bp"),
            unavailable=bool(data.get("unavailable", False)),
        )

    @classmethod
    def unavailable_section(cls) -> "RouteDistributionSection":
        return cls(unavailable=True)


@dataclass(frozen=True, slots=True)
class QualitySection:
    """Quality / regression comparison projection."""

    accepted_patch_count: int | None = None
    regression_count: int | None = None
    selected_test_false_negative_count: int | None = None
    proof_failure_count: int | None = None
    review_disagreement_count: int | None = None
    accepted_rate_bp: int | None = None
    regression_rate_bp: int | None = None
    unavailable: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "accepted_patch_count",
            "regression_count",
            "selected_test_false_negative_count",
            "proof_failure_count",
            "review_disagreement_count",
            "accepted_rate_bp",
            "regression_rate_bp",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_nonneg_int(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "unavailable", _bool(self.unavailable, "unavailable"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_patch_count": self.accepted_patch_count,
            "regression_count": self.regression_count,
            "selected_test_false_negative_count": (
                self.selected_test_false_negative_count
            ),
            "proof_failure_count": self.proof_failure_count,
            "review_disagreement_count": self.review_disagreement_count,
            "accepted_rate_bp": self.accepted_rate_bp,
            "regression_rate_bp": self.regression_rate_bp,
            "unavailable": self.unavailable,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "QualitySection":
        if data is None:
            return cls(unavailable=True)
        if not isinstance(data, Mapping):
            raise ReportError("quality must be a mapping")
        return cls(
            accepted_patch_count=data.get("accepted_patch_count"),
            regression_count=data.get("regression_count"),
            selected_test_false_negative_count=data.get(
                "selected_test_false_negative_count"
            ),
            proof_failure_count=data.get("proof_failure_count"),
            review_disagreement_count=data.get("review_disagreement_count"),
            accepted_rate_bp=data.get("accepted_rate_bp"),
            regression_rate_bp=data.get("regression_rate_bp"),
            unavailable=bool(data.get("unavailable", False)),
        )

    @classmethod
    def unavailable_section(cls) -> "QualitySection":
        return cls(unavailable=True)


@dataclass(frozen=True, slots=True)
class OverheadCostSection:
    """Overhead and economic cost projection (integer micros)."""

    model_spend_micros_total: int | None = None
    verification_compute_micros_total: int | None = None
    shadow_compute_micros_total: int | None = None
    audit_overhead_micros_total: int | None = None
    gross_savings_micros: int | None = None
    net_savings_micros: int | None = None
    cost_per_accepted_patch_micros: int | None = None
    unavailable: bool = False

    def __post_init__(self) -> None:
        # Savings may be negative in theory; store as optional int without
        # fabricating zeros. Nonnegativity is only enforced for cost totals.
        for field_name in (
            "model_spend_micros_total",
            "verification_compute_micros_total",
            "shadow_compute_micros_total",
            "audit_overhead_micros_total",
            "cost_per_accepted_patch_micros",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_nonneg_int(getattr(self, field_name), field_name),
            )
        for field_name in ("gross_savings_micros", "net_savings_micros"):
            value = getattr(self, field_name)
            if value is not None:
                if type(value) is not int or isinstance(value, bool):
                    raise ReportError(f"{field_name} must be an integer or null")
                if abs(value) > MAX_COUNTER:
                    raise ReportError(f"{field_name} exceeds maximum")
        object.__setattr__(self, "unavailable", _bool(self.unavailable, "unavailable"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_spend_micros_total": self.model_spend_micros_total,
            "verification_compute_micros_total": (
                self.verification_compute_micros_total
            ),
            "shadow_compute_micros_total": self.shadow_compute_micros_total,
            "audit_overhead_micros_total": self.audit_overhead_micros_total,
            "gross_savings_micros": self.gross_savings_micros,
            "net_savings_micros": self.net_savings_micros,
            "cost_per_accepted_patch_micros": self.cost_per_accepted_patch_micros,
            "unavailable": self.unavailable,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "OverheadCostSection":
        if data is None:
            return cls(unavailable=True)
        if not isinstance(data, Mapping):
            raise ReportError("overhead_and_cost must be a mapping")
        return cls(
            model_spend_micros_total=data.get("model_spend_micros_total"),
            verification_compute_micros_total=data.get(
                "verification_compute_micros_total"
            ),
            shadow_compute_micros_total=data.get("shadow_compute_micros_total"),
            audit_overhead_micros_total=data.get("audit_overhead_micros_total"),
            gross_savings_micros=data.get("gross_savings_micros"),
            net_savings_micros=data.get("net_savings_micros"),
            cost_per_accepted_patch_micros=data.get("cost_per_accepted_patch_micros"),
            unavailable=bool(data.get("unavailable", False)),
        )

    @classmethod
    def unavailable_section(cls) -> "OverheadCostSection":
        return cls(unavailable=True)


@dataclass(frozen=True, slots=True)
class RulesSection:
    """Rule proposal / rejection / promotion counts."""

    proposed_count: int | None = None
    rejected_count: int | None = None
    promoted_count: int | None = None
    candidate_cids: Sequence[str] = ()
    evaluation_report_cids: Sequence[str] = ()
    unavailable: bool = False

    def __post_init__(self) -> None:
        for field_name in ("proposed_count", "rejected_count", "promoted_count"):
            object.__setattr__(
                self,
                field_name,
                _optional_nonneg_int(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "candidate_cids",
            _unique_sorted_cids(self.candidate_cids, "candidate_cids"),
        )
        object.__setattr__(
            self,
            "evaluation_report_cids",
            _unique_sorted_cids(self.evaluation_report_cids, "evaluation_report_cids"),
        )
        object.__setattr__(self, "unavailable", _bool(self.unavailable, "unavailable"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposed_count": self.proposed_count,
            "rejected_count": self.rejected_count,
            "promoted_count": self.promoted_count,
            "candidate_cids": list(self.candidate_cids),
            "evaluation_report_cids": list(self.evaluation_report_cids),
            "unavailable": self.unavailable,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RulesSection":
        if data is None:
            return cls(unavailable=True)
        if not isinstance(data, Mapping):
            raise ReportError("rules must be a mapping")
        return cls(
            proposed_count=data.get("proposed_count"),
            rejected_count=data.get("rejected_count"),
            promoted_count=data.get("promoted_count"),
            candidate_cids=tuple(data.get("candidate_cids") or ()),
            evaluation_report_cids=tuple(data.get("evaluation_report_cids") or ()),
            unavailable=bool(data.get("unavailable", False)),
        )

    @classmethod
    def unavailable_section(cls) -> "RulesSection":
        return cls(unavailable=True)


@dataclass(frozen=True, slots=True)
class RollbackSection:
    """Rollback event projection (counts + decision CIDs)."""

    rollback_count: int | None = None
    rollback_decision_cids: Sequence[str] = ()
    last_rollback_decision_cid: str | None = None
    unavailable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rollback_count",
            _optional_nonneg_int(self.rollback_count, "rollback_count"),
        )
        object.__setattr__(
            self,
            "rollback_decision_cids",
            _unique_sorted_cids(
                self.rollback_decision_cids, "rollback_decision_cids"
            ),
        )
        object.__setattr__(
            self,
            "last_rollback_decision_cid",
            _optional_cid(
                self.last_rollback_decision_cid, "last_rollback_decision_cid"
            ),
        )
        object.__setattr__(self, "unavailable", _bool(self.unavailable, "unavailable"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollback_count": self.rollback_count,
            "rollback_decision_cids": list(self.rollback_decision_cids),
            "last_rollback_decision_cid": self.last_rollback_decision_cid,
            "unavailable": self.unavailable,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RollbackSection":
        if data is None:
            return cls(unavailable=True)
        if not isinstance(data, Mapping):
            raise ReportError("rollback must be a mapping")
        return cls(
            rollback_count=data.get("rollback_count"),
            rollback_decision_cids=tuple(data.get("rollback_decision_cids") or ()),
            last_rollback_decision_cid=data.get("last_rollback_decision_cid"),
            unavailable=bool(data.get("unavailable", False)),
        )

    @classmethod
    def unavailable_section(cls) -> "RollbackSection":
        return cls(unavailable=True)


@dataclass(frozen=True, slots=True)
class SealScopeSection:
    """Seal status and bound artifact projection."""

    status: str = SealScopeStatus.UNAVAILABLE.value
    seal_cid: str | None = None
    sealer_interface_id: str | None = None
    bound_artifact_cids: Sequence[str] = ()
    qualification_path: str | None = None
    unavailable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "status", _enum_value(self.status, SealScopeStatus, "status")
        )
        object.__setattr__(self, "seal_cid", _optional_cid(self.seal_cid, "seal_cid"))
        object.__setattr__(
            self,
            "sealer_interface_id",
            _optional_text(self.sealer_interface_id, "sealer_interface_id"),
        )
        if self.sealer_interface_id is not None:
            object.__setattr__(
                self,
                "sealer_interface_id",
                _interface_id(self.sealer_interface_id, "sealer_interface_id"),
            )
        object.__setattr__(
            self,
            "bound_artifact_cids",
            _unique_sorted_cids(self.bound_artifact_cids, "bound_artifact_cids"),
        )
        object.__setattr__(
            self,
            "qualification_path",
            _optional_token(self.qualification_path, "qualification_path"),
        )
        object.__setattr__(self, "unavailable", _bool(self.unavailable, "unavailable"))
        if self.status == SealScopeStatus.UNAVAILABLE.value:
            object.__setattr__(self, "unavailable", True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "seal_cid": self.seal_cid,
            "sealer_interface_id": self.sealer_interface_id,
            "bound_artifact_cids": list(self.bound_artifact_cids),
            "qualification_path": self.qualification_path,
            "unavailable": self.unavailable,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SealScopeSection":
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ReportError("seal_scope must be a mapping")
        return cls(
            status=data.get("status", SealScopeStatus.UNAVAILABLE.value),
            seal_cid=data.get("seal_cid"),
            sealer_interface_id=data.get("sealer_interface_id"),
            bound_artifact_cids=tuple(data.get("bound_artifact_cids") or ()),
            qualification_path=data.get("qualification_path"),
            unavailable=bool(
                data.get(
                    "unavailable",
                    data.get("status", SealScopeStatus.UNAVAILABLE.value)
                    == SealScopeStatus.UNAVAILABLE.value,
                )
            ),
        )

    @classmethod
    def unavailable_section(cls) -> "SealScopeSection":
        return cls()


@dataclass(frozen=True, slots=True)
class ProofScopeSection:
    """Bounded proof-scope claim surface (plan §14)."""

    kind: str = ProofScopeKind.UNAVAILABLE.value
    claim_kinds: Sequence[str] = ()
    claims_semantic_sufficiency: bool = False
    is_zero_knowledge: bool = False
    commitment_cid: str | None = None
    unavailable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum_value(self.kind, ProofScopeKind, "kind")
        )
        object.__setattr__(
            self, "claim_kinds", _unique_sorted_tokens(self.claim_kinds, "claim_kinds")
        )
        object.__setattr__(
            self,
            "claims_semantic_sufficiency",
            _bool(
                self.claims_semantic_sufficiency, "claims_semantic_sufficiency"
            ),
        )
        object.__setattr__(
            self, "is_zero_knowledge", _bool(self.is_zero_knowledge, "is_zero_knowledge")
        )
        object.__setattr__(
            self, "commitment_cid", _optional_cid(self.commitment_cid, "commitment_cid")
        )
        object.__setattr__(self, "unavailable", _bool(self.unavailable, "unavailable"))
        # Fail closed: never allow overclaim of semantic sufficiency or ZK.
        if self.claims_semantic_sufficiency:
            raise ReportError(
                "proof_scope must not claim semantic sufficiency",
                reason_code="proof_scope_overclaim",
            )
        if self.is_zero_knowledge:
            raise ReportError(
                "proof_scope must not claim zero-knowledge proof",
                reason_code="proof_scope_overclaim",
            )
        if self.kind == ProofScopeKind.UNAVAILABLE.value:
            object.__setattr__(self, "unavailable", True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "claim_kinds": list(self.claim_kinds),
            "claims_semantic_sufficiency": self.claims_semantic_sufficiency,
            "is_zero_knowledge": self.is_zero_knowledge,
            "commitment_cid": self.commitment_cid,
            "unavailable": self.unavailable,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ProofScopeSection":
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ReportError("proof_scope must be a mapping")
        return cls(
            kind=data.get("kind", ProofScopeKind.UNAVAILABLE.value),
            claim_kinds=tuple(data.get("claim_kinds") or ()),
            claims_semantic_sufficiency=bool(
                data.get("claims_semantic_sufficiency", False)
            ),
            is_zero_knowledge=bool(data.get("is_zero_knowledge", False)),
            commitment_cid=data.get("commitment_cid"),
            unavailable=bool(
                data.get(
                    "unavailable",
                    data.get("kind", ProofScopeKind.UNAVAILABLE.value)
                    == ProofScopeKind.UNAVAILABLE.value,
                )
            ),
        )

    @classmethod
    def unavailable_section(cls) -> "ProofScopeSection":
        return cls()


@dataclass(frozen=True, slots=True)
class HeuristicsSection:
    """Heuristic presence without elevating heuristics to exact authority."""

    classification: str = HeuristicClass.UNAVAILABLE.value
    heuristic_labels: Sequence[str] = ()
    treated_as_exact: bool = False
    unavailable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "classification",
            _enum_value(self.classification, HeuristicClass, "classification"),
        )
        object.__setattr__(
            self,
            "heuristic_labels",
            _unique_sorted_tokens(self.heuristic_labels, "heuristic_labels"),
        )
        object.__setattr__(
            self, "treated_as_exact", _bool(self.treated_as_exact, "treated_as_exact")
        )
        object.__setattr__(self, "unavailable", _bool(self.unavailable, "unavailable"))
        if self.treated_as_exact:
            raise ReportError(
                "heuristics must never be treated as exact",
                reason_code="heuristic_as_exact_rejected",
            )
        if self.classification == HeuristicClass.UNAVAILABLE.value:
            object.__setattr__(self, "unavailable", True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "heuristic_labels": list(self.heuristic_labels),
            "treated_as_exact": self.treated_as_exact,
            "unavailable": self.unavailable,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "HeuristicsSection":
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise ReportError("heuristics must be a mapping")
        return cls(
            classification=data.get(
                "classification", HeuristicClass.UNAVAILABLE.value
            ),
            heuristic_labels=tuple(data.get("heuristic_labels") or ()),
            treated_as_exact=bool(data.get("treated_as_exact", False)),
            unavailable=bool(
                data.get(
                    "unavailable",
                    data.get("classification", HeuristicClass.UNAVAILABLE.value)
                    == HeuristicClass.UNAVAILABLE.value,
                )
            ),
        )

    @classmethod
    def unavailable_section(cls) -> "HeuristicsSection":
        return cls()


# ---------------------------------------------------------------------------
# Sealed report and dashboard projections
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GovernorReport:
    """Sealed privacy-filtered final-report projection (detail view).

    Every plan-required final-report field is always present and representable.
    Missing measurements use ``None`` / ``unavailable=True`` rather than
    fabricated success.
    """

    inspected_commits: Sequence[str] = ()
    implemented_commits: Sequence[str] = ()
    consumed_interfaces: Sequence[str] = ()
    audit_population: AuditPopulationSection = field(
        default_factory=AuditPopulationSection.unavailable
    )
    differential_outcomes: DifferentialOutcomesSection = field(
        default_factory=DifferentialOutcomesSection.unavailable_section
    )
    omission_detection: OmissionDetectionSection = field(
        default_factory=OmissionDetectionSection.unavailable_section
    )
    expansion: ExpansionSection = field(
        default_factory=ExpansionSection.unavailable_section
    )
    final_context_reduction: ContextReductionSection = field(
        default_factory=ContextReductionSection.unavailable_section
    )
    route_distribution: RouteDistributionSection = field(
        default_factory=RouteDistributionSection.unavailable_section
    )
    quality: QualitySection = field(default_factory=QualitySection.unavailable_section)
    overhead_and_cost: OverheadCostSection = field(
        default_factory=OverheadCostSection.unavailable_section
    )
    rules: RulesSection = field(default_factory=RulesSection.unavailable_section)
    rollback: RollbackSection = field(
        default_factory=RollbackSection.unavailable_section
    )
    seal_scope: SealScopeSection = field(
        default_factory=SealScopeSection.unavailable_section
    )
    proof_scope: ProofScopeSection = field(
        default_factory=ProofScopeSection.unavailable_section
    )
    heuristics: HeuristicsSection = field(
        default_factory=HeuristicsSection.unavailable_section
    )
    remaining_production_risks: Sequence[str] = ()
    unavailable_fields: Sequence[str] = ()
    live_metrics_cid: str | None = None
    simulated_metrics_present: bool = False
    metric_report_cid: str | None = None
    evidence_mode: str = EvidenceMode.UNAVAILABLE.value
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "inspected_commits",
            _unique_sorted_commits(self.inspected_commits, "inspected_commits"),
        )
        object.__setattr__(
            self,
            "implemented_commits",
            _unique_sorted_commits(self.implemented_commits, "implemented_commits"),
        )
        object.__setattr__(
            self,
            "consumed_interfaces",
            _unique_sorted_interfaces(
                self.consumed_interfaces, "consumed_interfaces"
            ),
        )
        if not isinstance(self.audit_population, AuditPopulationSection):
            object.__setattr__(
                self,
                "audit_population",
                AuditPopulationSection.from_dict(self.audit_population),  # type: ignore[arg-type]
            )
        if not isinstance(self.differential_outcomes, DifferentialOutcomesSection):
            object.__setattr__(
                self,
                "differential_outcomes",
                DifferentialOutcomesSection.from_dict(self.differential_outcomes),  # type: ignore[arg-type]
            )
        if not isinstance(self.omission_detection, OmissionDetectionSection):
            object.__setattr__(
                self,
                "omission_detection",
                OmissionDetectionSection.from_dict(self.omission_detection),  # type: ignore[arg-type]
            )
        if not isinstance(self.expansion, ExpansionSection):
            object.__setattr__(
                self, "expansion", ExpansionSection.from_dict(self.expansion)  # type: ignore[arg-type]
            )
        if not isinstance(self.final_context_reduction, ContextReductionSection):
            object.__setattr__(
                self,
                "final_context_reduction",
                ContextReductionSection.from_dict(self.final_context_reduction),  # type: ignore[arg-type]
            )
        if not isinstance(self.route_distribution, RouteDistributionSection):
            object.__setattr__(
                self,
                "route_distribution",
                RouteDistributionSection.from_dict(self.route_distribution),  # type: ignore[arg-type]
            )
        if not isinstance(self.quality, QualitySection):
            object.__setattr__(
                self, "quality", QualitySection.from_dict(self.quality)  # type: ignore[arg-type]
            )
        if not isinstance(self.overhead_and_cost, OverheadCostSection):
            object.__setattr__(
                self,
                "overhead_and_cost",
                OverheadCostSection.from_dict(self.overhead_and_cost),  # type: ignore[arg-type]
            )
        if not isinstance(self.rules, RulesSection):
            object.__setattr__(self, "rules", RulesSection.from_dict(self.rules))  # type: ignore[arg-type]
        if not isinstance(self.rollback, RollbackSection):
            object.__setattr__(
                self, "rollback", RollbackSection.from_dict(self.rollback)  # type: ignore[arg-type]
            )
        if not isinstance(self.seal_scope, SealScopeSection):
            object.__setattr__(
                self, "seal_scope", SealScopeSection.from_dict(self.seal_scope)  # type: ignore[arg-type]
            )
        if not isinstance(self.proof_scope, ProofScopeSection):
            object.__setattr__(
                self, "proof_scope", ProofScopeSection.from_dict(self.proof_scope)  # type: ignore[arg-type]
            )
        if not isinstance(self.heuristics, HeuristicsSection):
            object.__setattr__(
                self, "heuristics", HeuristicsSection.from_dict(self.heuristics)  # type: ignore[arg-type]
            )
        object.__setattr__(
            self,
            "remaining_production_risks",
            _risk_tokens(
                self.remaining_production_risks, "remaining_production_risks"
            ),
        )
        object.__setattr__(
            self,
            "unavailable_fields",
            _unavailable_field_names(self.unavailable_fields, "unavailable_fields"),
        )
        object.__setattr__(
            self, "live_metrics_cid", _optional_cid(self.live_metrics_cid, "live_metrics_cid")
        )
        object.__setattr__(
            self,
            "simulated_metrics_present",
            _bool(self.simulated_metrics_present, "simulated_metrics_present"),
        )
        object.__setattr__(
            self,
            "metric_report_cid",
            _optional_cid(self.metric_report_cid, "metric_report_cid"),
        )
        object.__setattr__(
            self,
            "evidence_mode",
            _enum_value(self.evidence_mode, EvidenceMode, "evidence_mode"),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": GOVERNOR_REPORT_SCHEMA,
            "evidence": SCG_FINAL_REPORT_EVIDENCE,
            "interface_id": GOVERNOR_REPORT_INTERFACE,
            "generator_id": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "producer_id": PRODUCER_ID,
            "producer_version": PRODUCER_VERSION,
            "tool_id": TOOL_ID,
            "inspected_commits": list(self.inspected_commits),
            "implemented_commits": list(self.implemented_commits),
            "consumed_interfaces": list(self.consumed_interfaces),
            "audit_population": self.audit_population.to_dict(),
            "differential_outcomes": self.differential_outcomes.to_dict(),
            "omission_detection": self.omission_detection.to_dict(),
            "expansion": self.expansion.to_dict(),
            "final_context_reduction": self.final_context_reduction.to_dict(),
            "route_distribution": self.route_distribution.to_dict(),
            "quality": self.quality.to_dict(),
            "overhead_and_cost": self.overhead_and_cost.to_dict(),
            "rules": self.rules.to_dict(),
            "rollback": self.rollback.to_dict(),
            "seal_scope": self.seal_scope.to_dict(),
            "proof_scope": self.proof_scope.to_dict(),
            "heuristics": self.heuristics.to_dict(),
            "remaining_production_risks": list(self.remaining_production_risks),
            "unavailable_fields": list(self.unavailable_fields),
            "live_metrics_cid": self.live_metrics_cid,
            "simulated_metrics_present": self.simulated_metrics_present,
            "metric_report_cid": self.metric_report_cid,
            "evidence_mode": self.evidence_mode,
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def report_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["report_cid"] = self.report_cid
        # Final privacy projection — fail closed if anything leaked in.
        return _admit_public_structured(payload, "GovernorReport")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GovernorReport":
        if not isinstance(data, Mapping):
            raise ReportError("GovernorReport payload must be a mapping")
        payload = dict(data)
        claimed = payload.pop("report_cid", None)
        schema = payload.get("schema", GOVERNOR_REPORT_SCHEMA)
        if schema != GOVERNOR_REPORT_SCHEMA:
            raise ReportError("unsupported GovernorReport schema version")
        report = cls(
            inspected_commits=tuple(payload.get("inspected_commits") or ()),
            implemented_commits=tuple(payload.get("implemented_commits") or ()),
            consumed_interfaces=tuple(payload.get("consumed_interfaces") or ()),
            audit_population=AuditPopulationSection.from_dict(
                payload.get("audit_population")
            ),
            differential_outcomes=DifferentialOutcomesSection.from_dict(
                payload.get("differential_outcomes")
            ),
            omission_detection=OmissionDetectionSection.from_dict(
                payload.get("omission_detection")
            ),
            expansion=ExpansionSection.from_dict(payload.get("expansion")),
            final_context_reduction=ContextReductionSection.from_dict(
                payload.get("final_context_reduction")
            ),
            route_distribution=RouteDistributionSection.from_dict(
                payload.get("route_distribution")
            ),
            quality=QualitySection.from_dict(payload.get("quality")),
            overhead_and_cost=OverheadCostSection.from_dict(
                payload.get("overhead_and_cost")
            ),
            rules=RulesSection.from_dict(payload.get("rules")),
            rollback=RollbackSection.from_dict(payload.get("rollback")),
            seal_scope=SealScopeSection.from_dict(payload.get("seal_scope")),
            proof_scope=ProofScopeSection.from_dict(payload.get("proof_scope")),
            heuristics=HeuristicsSection.from_dict(payload.get("heuristics")),
            remaining_production_risks=tuple(
                payload.get("remaining_production_risks") or ()
            ),
            unavailable_fields=tuple(payload.get("unavailable_fields") or ()),
            live_metrics_cid=payload.get("live_metrics_cid"),
            simulated_metrics_present=bool(
                payload.get("simulated_metrics_present", False)
            ),
            metric_report_cid=payload.get("metric_report_cid"),
            evidence_mode=payload.get(
                "evidence_mode", EvidenceMode.UNAVAILABLE.value
            ),
            metadata=dict(payload.get("metadata") or {}),
        )
        if claimed is not None and claimed != report.report_cid:
            raise ReportError("GovernorReport report_cid does not verify")
        return report


@dataclass(frozen=True, slots=True)
class DashboardData:
    """Bounded machine-readable dashboard summary (no GUI / no server)."""

    report_cid: str
    evidence: str = SCG_DASHBOARD_DATA_EVIDENCE
    evidence_mode: str = EvidenceMode.UNAVAILABLE.value
    live_observation_count: int | None = None
    simulated_observation_count: int | None = None
    median_context_reduction_bp: int | None = None
    omission_detection_before_rate_bp: int | None = None
    critical_omissions_accepted_count: int | None = None
    net_savings_micros: int | None = None
    escalation_rate_bp: int | None = None
    seal_status: str = SealScopeStatus.UNAVAILABLE.value
    proof_scope_kind: str = ProofScopeKind.UNAVAILABLE.value
    heuristic_classification: str = HeuristicClass.UNAVAILABLE.value
    simulated_metrics_present: bool = False
    unavailable_fields: Sequence[str] = ()
    remaining_production_risks: Sequence[str] = ()
    metric_report_cid: str | None = None
    summary_tokens: Mapping[str, int | None] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "report_cid", _cid(self.report_cid, "report_cid"))
        object.__setattr__(self, "evidence", _text(self.evidence, "evidence"))
        if self.evidence != SCG_DASHBOARD_DATA_EVIDENCE:
            raise ReportError(
                f"dashboard evidence must be {SCG_DASHBOARD_DATA_EVIDENCE}"
            )
        object.__setattr__(
            self,
            "evidence_mode",
            _enum_value(self.evidence_mode, EvidenceMode, "evidence_mode"),
        )
        object.__setattr__(
            self,
            "live_observation_count",
            _optional_nonneg_int(self.live_observation_count, "live_observation_count"),
        )
        object.__setattr__(
            self,
            "simulated_observation_count",
            _optional_nonneg_int(
                self.simulated_observation_count, "simulated_observation_count"
            ),
        )
        for field_name in (
            "median_context_reduction_bp",
            "omission_detection_before_rate_bp",
            "critical_omissions_accepted_count",
            "escalation_rate_bp",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_nonneg_int(getattr(self, field_name), field_name),
            )
        if self.net_savings_micros is not None:
            if type(self.net_savings_micros) is not int or isinstance(
                self.net_savings_micros, bool
            ):
                raise ReportError("net_savings_micros must be an integer or null")
        object.__setattr__(
            self, "seal_status", _enum_value(self.seal_status, SealScopeStatus, "seal_status")
        )
        object.__setattr__(
            self,
            "proof_scope_kind",
            _enum_value(self.proof_scope_kind, ProofScopeKind, "proof_scope_kind"),
        )
        object.__setattr__(
            self,
            "heuristic_classification",
            _enum_value(
                self.heuristic_classification,
                HeuristicClass,
                "heuristic_classification",
            ),
        )
        object.__setattr__(
            self,
            "simulated_metrics_present",
            _bool(self.simulated_metrics_present, "simulated_metrics_present"),
        )
        object.__setattr__(
            self,
            "unavailable_fields",
            _unavailable_field_names(self.unavailable_fields, "unavailable_fields"),
        )
        object.__setattr__(
            self,
            "remaining_production_risks",
            _risk_tokens(
                self.remaining_production_risks, "remaining_production_risks"
            ),
        )
        object.__setattr__(
            self,
            "metric_report_cid",
            _optional_cid(self.metric_report_cid, "metric_report_cid"),
        )
        # summary_tokens: closed int|None values only
        if not isinstance(self.summary_tokens, Mapping):
            raise ReportError("summary_tokens must be a mapping")
        tokens: dict[str, int | None] = {}
        for key, item in self.summary_tokens.items():
            token = _token(key, "summary_tokens.key")
            if item is None:
                tokens[token] = None
            else:
                if type(item) is not int or isinstance(item, bool):
                    raise ReportError(f"summary_tokens.{token} must be int or null")
                tokens[token] = item
        object.__setattr__(
            self, "summary_tokens", MappingProxyType(dict(sorted(tokens.items())))
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": DASHBOARD_DATA_SCHEMA,
            "evidence": self.evidence,
            "interface_id": DASHBOARD_DATA_INTERFACE,
            "generator_id": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "producer_id": PRODUCER_ID,
            "producer_version": PRODUCER_VERSION,
            "tool_id": TOOL_ID,
            "report_cid": self.report_cid,
            "evidence_mode": self.evidence_mode,
            "live_observation_count": self.live_observation_count,
            "simulated_observation_count": self.simulated_observation_count,
            "median_context_reduction_bp": self.median_context_reduction_bp,
            "omission_detection_before_rate_bp": (
                self.omission_detection_before_rate_bp
            ),
            "critical_omissions_accepted_count": (
                self.critical_omissions_accepted_count
            ),
            "net_savings_micros": self.net_savings_micros,
            "escalation_rate_bp": self.escalation_rate_bp,
            "seal_status": self.seal_status,
            "proof_scope_kind": self.proof_scope_kind,
            "heuristic_classification": self.heuristic_classification,
            "simulated_metrics_present": self.simulated_metrics_present,
            "unavailable_fields": list(self.unavailable_fields),
            "remaining_production_risks": list(self.remaining_production_risks),
            "metric_report_cid": self.metric_report_cid,
            "summary_tokens": dict(self.summary_tokens),
            "metadata": _thaw_structured(self.metadata),
        }

    @property
    def dashboard_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["dashboard_cid"] = self.dashboard_cid
        return _admit_public_structured(payload, "DashboardData")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DashboardData":
        if not isinstance(data, Mapping):
            raise ReportError("DashboardData payload must be a mapping")
        payload = dict(data)
        claimed = payload.pop("dashboard_cid", None)
        schema = payload.get("schema", DASHBOARD_DATA_SCHEMA)
        if schema != DASHBOARD_DATA_SCHEMA:
            raise ReportError("unsupported DashboardData schema version")
        dashboard = cls(
            report_cid=payload["report_cid"],
            evidence=payload.get("evidence", SCG_DASHBOARD_DATA_EVIDENCE),
            evidence_mode=payload.get(
                "evidence_mode", EvidenceMode.UNAVAILABLE.value
            ),
            live_observation_count=payload.get("live_observation_count"),
            simulated_observation_count=payload.get("simulated_observation_count"),
            median_context_reduction_bp=payload.get("median_context_reduction_bp"),
            omission_detection_before_rate_bp=payload.get(
                "omission_detection_before_rate_bp"
            ),
            critical_omissions_accepted_count=payload.get(
                "critical_omissions_accepted_count"
            ),
            net_savings_micros=payload.get("net_savings_micros"),
            escalation_rate_bp=payload.get("escalation_rate_bp"),
            seal_status=payload.get(
                "seal_status", SealScopeStatus.UNAVAILABLE.value
            ),
            proof_scope_kind=payload.get(
                "proof_scope_kind", ProofScopeKind.UNAVAILABLE.value
            ),
            heuristic_classification=payload.get(
                "heuristic_classification", HeuristicClass.UNAVAILABLE.value
            ),
            simulated_metrics_present=bool(
                payload.get("simulated_metrics_present", False)
            ),
            unavailable_fields=tuple(payload.get("unavailable_fields") or ()),
            remaining_production_risks=tuple(
                payload.get("remaining_production_risks") or ()
            ),
            metric_report_cid=payload.get("metric_report_cid"),
            summary_tokens=dict(payload.get("summary_tokens") or {}),
            metadata=dict(payload.get("metadata") or {}),
        )
        if claimed is not None and claimed != dashboard.dashboard_cid:
            raise ReportError("DashboardData dashboard_cid does not verify")
        return dashboard


# ---------------------------------------------------------------------------
# Metrics → section projection helpers
# ---------------------------------------------------------------------------


def _metric_report_dict(
    metrics: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    """Normalize a GovernorMetricReport or mapping into a dict + report_cid."""

    if metrics is None:
        return None, None
    if GovernorMetricReport is not None and isinstance(metrics, GovernorMetricReport):
        payload = metrics.to_dict()
        return payload, payload.get("report_cid")
    if isinstance(metrics, Mapping):
        payload = dict(metrics)
        cid = payload.get("report_cid")
        if cid is not None:
            cid = _cid(cid, "metric_report_cid")
        return payload, cid
    raise ReportError("metrics must be GovernorMetricReport, mapping, or null")


def _sections_from_metrics(
    metrics_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Project live cohort metrics into report sections; mark unavailable."""

    if metrics_payload is None:
        return {
            "audit_population": AuditPopulationSection.unavailable(),
            "differential_outcomes": DifferentialOutcomesSection.unavailable_section(),
            "omission_detection": OmissionDetectionSection.unavailable_section(),
            "expansion": ExpansionSection.unavailable_section(),
            "final_context_reduction": ContextReductionSection.unavailable_section(),
            "route_distribution": RouteDistributionSection.unavailable_section(),
            "quality": QualitySection.unavailable_section(),
            "overhead_and_cost": OverheadCostSection.unavailable_section(),
            "live_metrics_cid": None,
            "simulated_metrics_present": False,
            "evidence_mode": EvidenceMode.UNAVAILABLE.value,
            "unavailable_fields": list(REQUIRED_FINAL_REPORT_FIELDS),
        }

    live = dict(metrics_payload.get("live") or {})
    simulated = dict(metrics_payload.get("simulated") or {})
    live_count = int(live.get("observation_count") or 0)
    sim_count = int(simulated.get("observation_count") or 0)

    compression = dict(live.get("compression") or {})
    quality = dict(live.get("quality") or {})
    omission = dict(live.get("omission") or {})
    routing = dict(live.get("routing") or {})
    economic = dict(live.get("economic") or {})

    outcome_counts = dict(quality.get("outcome_counts") or {})
    unavailable: list[str] = []

    if live_count == 0 and sim_count == 0:
        unavailable.extend(
            [
                "audit_population",
                "differential_outcomes",
                "omission_detection",
                "expansion",
                "final_context_reduction",
                "route_distribution",
                "quality",
                "overhead_and_cost",
            ]
        )
        mode = EvidenceMode.UNAVAILABLE.value
    elif live_count > 0 and sim_count > 0:
        mode = EvidenceMode.MIXED.value
    elif sim_count > 0:
        mode = EvidenceMode.SIMULATED.value
    else:
        mode = EvidenceMode.LIVE.value

    empty_live = live_count == 0

    audit = AuditPopulationSection(
        total_audits=live_count + sim_count if (live_count + sim_count) else None,
        live_audits=live_count if live_count or sim_count else None,
        simulated_audits=sim_count if live_count or sim_count else None,
        source_receipt_cids=tuple(metrics_payload.get("source_receipt_cids") or ()),
        history_cids=(),
    )

    differential = DifferentialOutcomesSection(
        outcome_counts=outcome_counts,
        equivalent_success_count=outcome_counts.get("equivalent_success"),
        compressed_failed_expanded_succeeded_count=outcome_counts.get(
            "compressed_failed_expanded_succeeded"
        ),
        both_failed_count=outcome_counts.get("both_failed"),
        verification_inconclusive_count=outcome_counts.get(
            "verification_inconclusive"
        ),
        unavailable=empty_live,
    )

    omission_section = OmissionDetectionSection(
        intentional_omission_count=omission.get("intentional_omission_count"),
        detected_before_execution_count=omission.get(
            "detected_before_execution_count"
        ),
        detected_after_execution_count=omission.get("detected_after_execution_count"),
        critical_omission_count=omission.get("critical_omission_count"),
        critical_omissions_accepted_count=omission.get(
            "critical_omissions_accepted_count"
        ),
        detection_before_rate_bp=omission.get("detection_before_rate_bp"),
        critical_acceptance_rate_bp=omission.get("critical_acceptance_rate_bp"),
        false_alarm_count=omission.get("false_alarm_count"),
        unavailable=empty_live,
    )

    expansion = ExpansionSection(
        expansion_count=compression.get("expansion_count"),
        expansion_rate_bp=compression.get("expansion_rate_bp"),
        expansion_true_positive_count=omission.get("expansion_true_positive_count"),
        expansion_false_positive_count=omission.get("expansion_false_positive_count"),
        expansion_false_negative_count=omission.get("expansion_false_negative_count"),
        expansion_precision_bp=omission.get("expansion_precision_bp"),
        expansion_recall_bp=omission.get("expansion_recall_bp"),
        expanded_tokens_total=compression.get("expanded_tokens_total"),
        unavailable=empty_live,
    )

    reduction = ContextReductionSection(
        median_context_reduction_bp=compression.get("median_context_reduction_bp"),
        mean_context_reduction_bp=compression.get("mean_context_reduction_bp"),
        raw_tokens_total=compression.get("raw_tokens_total"),
        compressed_tokens_total=compression.get("compressed_tokens_total"),
        unavailable=empty_live
        or compression.get("median_context_reduction_bp") is None,
    )

    route = RouteDistributionSection(
        route_share_counts=dict(routing.get("route_share_counts") or {}),
        route_share_bp=dict(routing.get("route_share_bp") or {}),
        escalation_count=routing.get("escalation_count"),
        retry_count=routing.get("retry_count"),
        escalation_rate_bp=routing.get("escalation_rate_bp"),
        unavailable=empty_live,
    )

    quality_section = QualitySection(
        accepted_patch_count=quality.get("accepted_patch_count"),
        regression_count=quality.get("regression_count"),
        selected_test_false_negative_count=quality.get(
            "selected_test_false_negative_count"
        ),
        proof_failure_count=quality.get("proof_failure_count"),
        review_disagreement_count=quality.get("review_disagreement_count"),
        accepted_rate_bp=quality.get("accepted_rate_bp"),
        regression_rate_bp=quality.get("regression_rate_bp"),
        unavailable=empty_live,
    )

    cost = OverheadCostSection(
        model_spend_micros_total=economic.get("model_spend_micros_total"),
        verification_compute_micros_total=economic.get(
            "verification_compute_micros_total"
        ),
        shadow_compute_micros_total=economic.get("shadow_compute_micros_total"),
        audit_overhead_micros_total=economic.get("audit_overhead_micros_total"),
        gross_savings_micros=economic.get("gross_savings_micros"),
        net_savings_micros=economic.get("net_savings_micros"),
        cost_per_accepted_patch_micros=economic.get(
            "cost_per_accepted_patch_micros"
        ),
        unavailable=empty_live
        or economic.get("net_savings_micros") is None,
    )

    return {
        "audit_population": audit,
        "differential_outcomes": differential,
        "omission_detection": omission_section,
        "expansion": expansion,
        "final_context_reduction": reduction,
        "route_distribution": route,
        "quality": quality_section,
        "overhead_and_cost": cost,
        "live_metrics_cid": None,
        "simulated_metrics_present": sim_count > 0,
        "evidence_mode": mode,
        "unavailable_fields": sorted(set(unavailable)),
        "live_observation_count": live_count if (live_count or sim_count) else None,
        "simulated_observation_count": sim_count if (live_count or sim_count) else None,
    }




def _collect_unavailable_fields(
    base: Iterable[str],
    *,
    differential_outcomes: DifferentialOutcomesSection,
    omission_detection: OmissionDetectionSection,
    expansion: ExpansionSection,
    final_context_reduction: ContextReductionSection,
    route_distribution: RouteDistributionSection,
    quality: QualitySection,
    overhead_and_cost: OverheadCostSection,
    rules: RulesSection,
    rollback: RollbackSection,
    seal_scope: SealScopeSection,
    proof_scope: ProofScopeSection,
    heuristics: HeuristicsSection,
    metric_report_cid: str | None,
) -> tuple[str, ...]:
    names = {str(item) for item in base}
    checks: tuple[tuple[str, bool], ...] = (
        ("differential_outcomes", differential_outcomes.unavailable),
        ("omission_detection", omission_detection.unavailable),
        ("expansion", expansion.unavailable),
        ("final_context_reduction", final_context_reduction.unavailable),
        ("route_distribution", route_distribution.unavailable),
        ("quality", quality.unavailable),
        ("overhead_and_cost", overhead_and_cost.unavailable),
        ("rules", rules.unavailable),
        ("rollback", rollback.unavailable),
        ("seal_scope", seal_scope.unavailable),
        ("proof_scope", proof_scope.unavailable),
        ("heuristics", heuristics.unavailable),
        ("metric_report_cid", metric_report_cid is None),
    )
    for field_name, is_unavailable in checks:
        if is_unavailable:
            names.add(field_name)
    return tuple(sorted(names))


def _resolve_section(override: Any, default: Any, factory: Any, name: str) -> Any:
    if override is None:
        return default
    if isinstance(override, factory):
        return override
    if isinstance(override, Mapping):
        return factory.from_dict(override)
    raise ReportError(f"{name} must be {factory.__name__}, mapping, or null")


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_governor_report(
    *,
    metrics: Any = None,
    histories: Sequence[str] | None = None,
    inspected_commits: Sequence[str] | None = None,
    implemented_commits: Sequence[str] | None = None,
    consumed_interfaces: Sequence[str] | None = None,
    rules: Mapping[str, Any] | RulesSection | None = None,
    rollback: Mapping[str, Any] | RollbackSection | None = None,
    seal_scope: Mapping[str, Any] | SealScopeSection | None = None,
    proof_scope: Mapping[str, Any] | ProofScopeSection | None = None,
    heuristics: Mapping[str, Any] | HeuristicsSection | None = None,
    remaining_production_risks: Sequence[str] | None = None,
    audit_population: Mapping[str, Any] | AuditPopulationSection | None = None,
    differential_outcomes: Mapping[str, Any]
    | DifferentialOutcomesSection
    | None = None,
    omission_detection: Mapping[str, Any]
    | OmissionDetectionSection
    | None = None,
    expansion: Mapping[str, Any] | ExpansionSection | None = None,
    final_context_reduction: Mapping[str, Any]
    | ContextReductionSection
    | None = None,
    route_distribution: Mapping[str, Any]
    | RouteDistributionSection
    | None = None,
    quality: Mapping[str, Any] | QualitySection | None = None,
    overhead_and_cost: Mapping[str, Any] | OverheadCostSection | None = None,
    unavailable_fields: Sequence[str] | None = None,
    evidence_mode: str | EvidenceMode | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> GovernorReport:
    """Build a sealed privacy-filtered final-report projection.

    Inputs are optional. Missing evidence becomes explicit ``unavailable``
    rather than fabricated success. When *metrics* is supplied, live-cohort
    counters populate the corresponding sections; simulated cohort presence is
    labeled but never mixed into live quality claims.
    """

    metrics_payload, metric_report_cid = _metric_report_dict(metrics)
    projected = _sections_from_metrics(metrics_payload)

    history_cids = _unique_sorted_cids(histories, "histories")

    audit = _resolve_section(
        audit_population,
        projected["audit_population"],
        AuditPopulationSection,
        "audit_population",
    )
    if history_cids:
        audit = AuditPopulationSection(
            total_audits=audit.total_audits,
            live_audits=audit.live_audits,
            simulated_audits=audit.simulated_audits,
            source_receipt_cids=audit.source_receipt_cids,
            history_cids=history_cids,
        )

    differential = _resolve_section(
        differential_outcomes,
        projected["differential_outcomes"],
        DifferentialOutcomesSection,
        "differential_outcomes",
    )
    omission = _resolve_section(
        omission_detection,
        projected["omission_detection"],
        OmissionDetectionSection,
        "omission_detection",
    )
    expansion_section = _resolve_section(
        expansion, projected["expansion"], ExpansionSection, "expansion"
    )
    reduction = _resolve_section(
        final_context_reduction,
        projected["final_context_reduction"],
        ContextReductionSection,
        "final_context_reduction",
    )
    route = _resolve_section(
        route_distribution,
        projected["route_distribution"],
        RouteDistributionSection,
        "route_distribution",
    )
    quality_section = _resolve_section(
        quality, projected["quality"], QualitySection, "quality"
    )
    cost = _resolve_section(
        overhead_and_cost,
        projected["overhead_and_cost"],
        OverheadCostSection,
        "overhead_and_cost",
    )
    rules_section = _resolve_section(
        rules, RulesSection.unavailable_section(), RulesSection, "rules"
    )
    rollback_section = _resolve_section(
        rollback, RollbackSection.unavailable_section(), RollbackSection, "rollback"
    )
    seal = _resolve_section(
        seal_scope,
        SealScopeSection.unavailable_section(),
        SealScopeSection,
        "seal_scope",
    )
    proof = _resolve_section(
        proof_scope,
        ProofScopeSection.unavailable_section(),
        ProofScopeSection,
        "proof_scope",
    )
    heuristics_section = _resolve_section(
        heuristics,
        HeuristicsSection.unavailable_section(),
        HeuristicsSection,
        "heuristics",
    )

    mode = (
        _enum_value(evidence_mode, EvidenceMode, "evidence_mode")
        if evidence_mode is not None
        else projected["evidence_mode"]
    )

    base_unavailable = list(projected.get("unavailable_fields") or ())
    if unavailable_fields is not None:
        base_unavailable.extend(list(unavailable_fields))

    merged_unavailable = _collect_unavailable_fields(
        base_unavailable,
        differential_outcomes=differential,
        omission_detection=omission,
        expansion=expansion_section,
        final_context_reduction=reduction,
        route_distribution=route,
        quality=quality_section,
        overhead_and_cost=cost,
        rules=rules_section,
        rollback=rollback_section,
        seal_scope=seal,
        proof_scope=proof,
        heuristics=heuristics_section,
        metric_report_cid=metric_report_cid,
    )

    meta = dict(metadata or {})
    meta.setdefault("evidence", SCG_FINAL_REPORT_EVIDENCE)
    meta.setdefault("track", "reports")

    report = GovernorReport(
        inspected_commits=tuple(inspected_commits or ()),
        implemented_commits=tuple(implemented_commits or ()),
        consumed_interfaces=tuple(consumed_interfaces or ()),
        audit_population=audit,
        differential_outcomes=differential,
        omission_detection=omission,
        expansion=expansion_section,
        final_context_reduction=reduction,
        route_distribution=route,
        quality=quality_section,
        overhead_and_cost=cost,
        rules=rules_section,
        rollback=rollback_section,
        seal_scope=seal,
        proof_scope=proof,
        heuristics=heuristics_section,
        remaining_production_risks=tuple(remaining_production_risks or ()),
        unavailable_fields=merged_unavailable,
        live_metrics_cid=projected.get("live_metrics_cid"),
        simulated_metrics_present=bool(projected.get("simulated_metrics_present")),
        metric_report_cid=metric_report_cid,
        evidence_mode=mode,
        metadata=meta,
    )
    sealed = report.to_dict()
    return GovernorReport.from_dict(sealed)


def build_dashboard_data(
    report: GovernorReport | Mapping[str, Any] | None = None,
    *,
    metrics: Any = None,
    metadata: Mapping[str, Any] | None = None,
    **report_kwargs: Any,
) -> DashboardData:
    """Build a bounded machine-readable dashboard-data summary projection.

    Accepts an existing :class:`GovernorReport` (or mapping), or the same
    keyword inputs as :func:`build_governor_report` when *report* is omitted.
    Never starts a server or GUI.
    """

    if report is None:
        if report_kwargs or metrics is not None:
            governor = build_governor_report(metrics=metrics, **report_kwargs)
        else:
            governor = build_governor_report(metrics=metrics)
    elif isinstance(report, GovernorReport):
        governor = report
    elif isinstance(report, Mapping):
        governor = GovernorReport.from_dict(report)
    else:
        raise ReportError(
            "report must be GovernorReport, mapping, or null",
            reason_code="invalid_report_input",
        )

    metrics_payload, metric_report_cid = _metric_report_dict(metrics)
    if metric_report_cid is None:
        metric_report_cid = governor.metric_report_cid

    live_count: int | None = governor.audit_population.live_audits
    sim_count: int | None = governor.audit_population.simulated_audits
    if metrics_payload is not None:
        live = dict(metrics_payload.get("live") or {})
        simulated = dict(metrics_payload.get("simulated") or {})
        live_count = int(live.get("observation_count") or 0)
        sim_count = int(simulated.get("observation_count") or 0)

    meta = dict(metadata or {})
    meta.setdefault("evidence", SCG_DASHBOARD_DATA_EVIDENCE)
    meta.setdefault("track", "reports")
    meta.setdefault("source_report_cid", governor.report_cid)

    summary_tokens: dict[str, int | None] = {
        "live_observation_count": live_count,
        "simulated_observation_count": sim_count,
        "median_context_reduction_bp": (
            governor.final_context_reduction.median_context_reduction_bp
        ),
        "net_savings_micros": governor.overhead_and_cost.net_savings_micros,
        "escalation_rate_bp": governor.route_distribution.escalation_rate_bp,
        "critical_omissions_accepted_count": (
            governor.omission_detection.critical_omissions_accepted_count
        ),
    }

    dashboard = DashboardData(
        report_cid=governor.report_cid,
        evidence=SCG_DASHBOARD_DATA_EVIDENCE,
        evidence_mode=governor.evidence_mode,
        live_observation_count=live_count,
        simulated_observation_count=sim_count,
        median_context_reduction_bp=(
            governor.final_context_reduction.median_context_reduction_bp
        ),
        omission_detection_before_rate_bp=(
            governor.omission_detection.detection_before_rate_bp
        ),
        critical_omissions_accepted_count=(
            governor.omission_detection.critical_omissions_accepted_count
        ),
        net_savings_micros=governor.overhead_and_cost.net_savings_micros,
        escalation_rate_bp=governor.route_distribution.escalation_rate_bp,
        seal_status=governor.seal_scope.status,
        proof_scope_kind=governor.proof_scope.kind,
        heuristic_classification=governor.heuristics.classification,
        simulated_metrics_present=governor.simulated_metrics_present,
        unavailable_fields=governor.unavailable_fields,
        remaining_production_risks=governor.remaining_production_risks,
        metric_report_cid=metric_report_cid,
        summary_tokens=summary_tokens,
        metadata=meta,
    )
    sealed = dashboard.to_dict()
    return DashboardData.from_dict(sealed)


def required_final_report_fields() -> tuple[str, ...]:
    return REQUIRED_FINAL_REPORT_FIELDS


def build_governor_report_interface_id() -> str:
    return BUILD_GOVERNOR_REPORT_INTERFACE


def build_dashboard_data_interface_id() -> str:
    return BUILD_DASHBOARD_DATA_INTERFACE


def governor_report_interface_id() -> str:
    return GOVERNOR_REPORT_INTERFACE


def dashboard_data_interface_id() -> str:
    return DASHBOARD_DATA_INTERFACE


def dashboard_data_evidence_id() -> str:
    return SCG_DASHBOARD_DATA_EVIDENCE


def final_report_evidence_id() -> str:
    return SCG_FINAL_REPORT_EVIDENCE


def evidence_modes() -> tuple[str, ...]:
    return tuple(item.value for item in EvidenceMode)


def seal_scope_statuses() -> tuple[str, ...]:
    return tuple(item.value for item in SealScopeStatus)


def proof_scope_kinds() -> tuple[str, ...]:
    return tuple(item.value for item in ProofScopeKind)


def heuristic_classes() -> tuple[str, ...]:
    return tuple(item.value for item in HeuristicClass)


__all__ = [
    "BUILD_DASHBOARD_DATA_INTERFACE",
    "BUILD_GOVERNOR_REPORT_INTERFACE",
    "DASHBOARD_DATA_INTERFACE",
    "DASHBOARD_DATA_SCHEMA",
    "FREE_FORM_AUTHORITY_FORBIDDEN_KEYS",
    "GENERATOR_ID",
    "GENERATOR_VERSION",
    "GOVERNOR_REPORT_INTERFACE",
    "GOVERNOR_REPORT_SCHEMA",
    "PRIVATE_FIELD_MARKERS",
    "PRODUCER_ID",
    "PRODUCER_VERSION",
    "REQUIRED_FINAL_REPORT_FIELDS",
    "SCG_DASHBOARD_DATA_EVIDENCE",
    "SCG_FINAL_REPORT_EVIDENCE",
    "TOOL_ID",
    "AuditPopulationSection",
    "ContextReductionSection",
    "DashboardData",
    "DifferentialOutcomesSection",
    "EvidenceMode",
    "ExpansionSection",
    "FreeFormAuthorityError",
    "HeuristicClass",
    "HeuristicsSection",
    "OmissionDetectionSection",
    "OverheadCostSection",
    "ProofScopeKind",
    "ProofScopeSection",
    "QualitySection",
    "ReportError",
    "RollbackSection",
    "RouteDistributionSection",
    "RulesSection",
    "SealScopeSection",
    "SealScopeStatus",
    "GovernorReport",
    "build_dashboard_data",
    "build_dashboard_data_interface_id",
    "build_governor_report",
    "build_governor_report_interface_id",
    "dashboard_data_evidence_id",
    "dashboard_data_interface_id",
    "evidence_modes",
    "final_report_evidence_id",
    "governor_report_interface_id",
    "heuristic_classes",
    "proof_scope_kinds",
    "reject_free_form_authority",
    "required_final_report_fields",
    "seal_scope_statuses",
]
