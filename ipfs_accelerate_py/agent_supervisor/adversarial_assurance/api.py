"""Compose the complete public Python campaign API (AAE-048).

This module is the stable accelerate-side facade over the twelve plan-required
public entry points:

* ``create_assurance_manifest``
* ``generate_mutation_candidates``
* ``predict_detection_set``
* ``execute_mutation``
* ``classify_mutation_outcome``
* ``diagnose_surviving_mutant``
* ``analyze_vacuity``
* ``propose_gap_remediation``
* ``evaluate_remediation``
* ``promote_assurance_policy``
* ``plan_mutation_campaign``
* ``execute_mutation_campaign``

Ten APIs re-export exact leaf callables so signatures, return types, and
identity gates cannot be bypassed through the facade. The two campaign-level
composers that have no prior leaf owner — ``analyze_vacuity`` and
``execute_mutation_campaign`` — live here.

Composition is lazy and dependency-injectable. Importing this module performs
no I/O, starts no processes or network activity, and does not load optional
providers. Unknown commands and unknown mapping/parameter fields fail closed.
Public inputs reject host-path and absolute-filesystem leakage.
"""

from __future__ import annotations

import importlib
import inspect
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.common import (
    ArtifactProvenance,
    AssuranceArtifactHeader,
    AssuranceTerminalStatus,
    AuthoritySource,
    ExecutionMode,
    GeneratorIdentity,
    VersionBinding,
    reject_private_model_authority_and_host_fallbacks,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.analysis_contracts import (
    VacuityFamily,
    VacuityFinding,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.execution_contracts import (
    ExpectedDetectionSet,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.mutation_contracts import (
    MutationCampaignPlan,
    MutationCandidate,
    MutationContractError,
)

# ---------------------------------------------------------------------------
# Evidence / interface / schema pins
# ---------------------------------------------------------------------------

AAE_PUBLIC_API_EVIDENCE: Final[str] = "aae/public-api@1"
ADVERSARIAL_ASSURANCE_PUBLIC_API_INTERFACE: Final[str] = (
    "AdversarialAssurancePublicApi@1"
)
ADVERSARIAL_ASSURANCE_PUBLIC_API_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-public-api@1"
)
ASSURANCE_CAMPAIGN_API_INTERFACE: Final[str] = "AssuranceCampaignApi@1"

ANALYZE_VACUITY_INTERFACE: Final[str] = "analyze_vacuity@1"
EXECUTE_MUTATION_CAMPAIGN_INTERFACE: Final[str] = "execute_mutation_campaign@1"

VACUITY_CAMPAIGN_RESULT_INTERFACE: Final[str] = "VacuityCampaignAnalysisResult@1"
VACUITY_CAMPAIGN_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-vacuity-campaign-result@1"
)
MUTATION_CAMPAIGN_EXECUTION_RESULT_INTERFACE: Final[str] = (
    "MutationCampaignExecutionResult@1"
)
MUTATION_CAMPAIGN_EXECUTION_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "adversarial-assurance-mutation-campaign-execution-result@1"
)

GENERATOR_ID: Final[str] = "aae_public_api"
GENERATOR_VERSION: Final[str] = "1.0.0"

# Twelve plan-required module-level APIs (order is stable for evidence dumps).
REQUIRED_PUBLIC_APIS: Final[tuple[str, ...]] = (
    "create_assurance_manifest",
    "generate_mutation_candidates",
    "predict_detection_set",
    "execute_mutation",
    "classify_mutation_outcome",
    "diagnose_surviving_mutant",
    "analyze_vacuity",
    "propose_gap_remediation",
    "evaluate_remediation",
    "promote_assurance_policy",
    "plan_mutation_campaign",
    "execute_mutation_campaign",
)

REQUIRED_COMMANDS: Final[tuple[str, ...]] = REQUIRED_PUBLIC_APIS

# Closed top-level fields for mapping-form invoke envelopes.
_INVOKE_ENVELOPE_FIELDS: Final[frozenset[str]] = frozenset(
    {"command", "args", "kwargs", "arguments"}
)

# name -> (import module path, attribute)
# Local composers resolve to this module; leaf APIs resolve to owners.
_API_OWNERS: Final[dict[str, tuple[str, str]]] = {
    "create_assurance_manifest": (
        "ipfs_accelerate_py.agent_supervisor.adversarial_assurance.manifest",
        "create_assurance_manifest",
    ),
    "generate_mutation_candidates": (
        "ipfs_accelerate_py.agent_supervisor.adversarial_assurance.planning",
        "generate_mutation_candidates",
    ),
    "predict_detection_set": (
        "ipfs_accelerate_py.agent_supervisor.adversarial_assurance.planning",
        "predict_detection_set",
    ),
    "execute_mutation": (
        "ipfs_accelerate_py.agent_supervisor.adversarial_assurance.execution",
        "execute_mutation",
    ),
    "classify_mutation_outcome": (
        "ipfs_accelerate_py.agent_supervisor.adversarial_assurance.execution",
        "classify_mutation_outcome",
    ),
    "diagnose_surviving_mutant": (
        "ipfs_accelerate_py.agent_supervisor.adversarial_assurance.diagnosis",
        "diagnose_surviving_mutant",
    ),
    "analyze_vacuity": (
        "ipfs_accelerate_py.agent_supervisor.adversarial_assurance.api",
        "analyze_vacuity",
    ),
    "propose_gap_remediation": (
        "ipfs_accelerate_py.agent_supervisor.adversarial_assurance.remediation",
        "propose_gap_remediation",
    ),
    "evaluate_remediation": (
        "ipfs_accelerate_py.agent_supervisor.adversarial_assurance.remediation",
        "evaluate_remediation",
    ),
    "promote_assurance_policy": (
        "ipfs_accelerate_py.agent_supervisor.adversarial_assurance.promotion",
        "promote_assurance_policy",
    ),
    "plan_mutation_campaign": (
        "ipfs_accelerate_py.agent_supervisor.adversarial_assurance.planning",
        "plan_mutation_campaign",
    ),
    "execute_mutation_campaign": (
        "ipfs_accelerate_py.agent_supervisor.adversarial_assurance.api",
        "execute_mutation_campaign",
    ),
}

# Per-API interface pins (stable evidence labels).
_API_INTERFACE_IDS: Final[dict[str, str]] = {
    "create_assurance_manifest": "create_assurance_manifest@1",
    "generate_mutation_candidates": "generate_mutation_candidates@1",
    "predict_detection_set": "predict_detection_set@1",
    "execute_mutation": "execute_mutation@1",
    "classify_mutation_outcome": "classify_mutation_outcome@1",
    "diagnose_surviving_mutant": "diagnose_surviving_mutant@1",
    "analyze_vacuity": ANALYZE_VACUITY_INTERFACE,
    "propose_gap_remediation": "propose_gap_remediation@1",
    "evaluate_remediation": "evaluate_remediation@1",
    "promote_assurance_policy": "promote_assurance_policy@1",
    "plan_mutation_campaign": "plan_mutation_campaign@1",
    "execute_mutation_campaign": EXECUTE_MUTATION_CAMPAIGN_INTERFACE,
}

if frozenset(_API_OWNERS) != frozenset(REQUIRED_PUBLIC_APIS):
    raise RuntimeError("REQUIRED_PUBLIC_APIS and _API_OWNERS must match exactly")
if frozenset(_API_INTERFACE_IDS) != frozenset(REQUIRED_PUBLIC_APIS):
    raise RuntimeError("REQUIRED_PUBLIC_APIS and _API_INTERFACE_IDS must match exactly")

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_DIAGNOSTIC: Final[int] = 1_024
MAX_CANDIDATES: Final[int] = 4_096
MAX_FINDINGS: Final[int] = 1_024
MAX_REASON_CODES: Final[int] = 256

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")
_ABSOLUTE_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:[A-Za-z]:[\\/]|\\\\|/)"
)
_HOME_PATH_MARKERS: Final[tuple[str, ...]] = (
    "/home/",
    "/Users/",
    "\\Users\\",
    "/tmp/",
    "\\Temp\\",
    "C:\\",
    "c:\\",
)

REASON_NO_PRODUCTION_POLICY_CHANGE: Final[str] = "no_production_policy_change"
REASON_NO_ARBITRARY_PATH_EXPOSURE: Final[str] = "no_arbitrary_path_exposure"
REASON_DISPOSABLE_WORKTREE_REQUIRED: Final[str] = "disposable_worktree_required"
REASON_NETWORK_DISABLED: Final[str] = "network_disabled"
REASON_CANONICAL_BINDINGS: Final[str] = "canonical_bindings"
REASON_VACUITY_ANALYZED: Final[str] = "vacuity_analyzed"
REASON_CAMPAIGN_EXECUTED: Final[str] = "campaign_executed"
REASON_PRECOMPUTED_REPORTS: Final[str] = "precomputed_reports"
REASON_INJECTED_EXECUTOR: Final[str] = "injected_candidate_executor"

CandidateExecutor = Callable[..., Any]

# ---------------------------------------------------------------------------
# Errors / typed unavailable
# ---------------------------------------------------------------------------


class AssurancePublicApiError(ValueError):
    """Base error for the public adversarial-assurance campaign facade."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "public_api_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "public_api_error")
        self.details = dict(details or {})


class UnknownCommandError(AssurancePublicApiError):
    """Raised when an invoke command is outside the closed vocabulary."""

    def __init__(self, command: str) -> None:
        super().__init__(
            f"unknown command: {command!r}; allowed={list(REQUIRED_COMMANDS)}",
            reason_code="unknown_command",
        )
        self.command = command


class UnknownFieldError(AssurancePublicApiError):
    """Raised when a closed mapping or parameter set contains unknown fields."""

    def __init__(
        self,
        fields: Sequence[str],
        *,
        context: str = "payload",
    ) -> None:
        ordered = tuple(sorted(str(item) for item in fields))
        super().__init__(
            f"{context} has unknown fields: {list(ordered)}",
            reason_code="unknown_field",
        )
        self.fields = ordered
        self.context = context


class AssuranceApiUnavailableError(AssurancePublicApiError):
    """Raised when a required API surface is typed-unavailable."""

    def __init__(
        self,
        command: str,
        *,
        reason_code: str = "api_unavailable",
        diagnostic: str | None = None,
        status: str = "unavailable",
    ) -> None:
        message = diagnostic or f"required API {command!r} is unavailable"
        super().__init__(message, reason_code=reason_code)
        self.command = command
        self.status = status
        self.diagnostic = message


class PathExposureError(AssurancePublicApiError):
    """Raised when a public API input would expose an arbitrary host path."""

    def __init__(self, message: str, *, field: str = "path") -> None:
        super().__init__(
            message,
            reason_code="path_exposure",
            details={"field": field},
        )
        self.field = field


class ApiAvailability(str, Enum):
    """Closed availability vocabulary for typed public-API probes."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    MISSING = "missing"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True)
class AssuranceApiUnavailableResult:
    """Typed unavailable result for a required public API."""

    command: str
    status: str = ApiAvailability.UNAVAILABLE.value
    reason_code: str = "api_unavailable"
    diagnostic: str | None = None
    interface_id: str = ASSURANCE_CAMPAIGN_API_INTERFACE
    evidence_id: str = AAE_PUBLIC_API_EVIDENCE
    api_interface_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "command", str(self.command))
        object.__setattr__(
            self,
            "status",
            str(self.status or ApiAvailability.UNAVAILABLE.value),
        )
        object.__setattr__(self, "reason_code", str(self.reason_code))
        if self.diagnostic is not None:
            object.__setattr__(self, "diagnostic", str(self.diagnostic))
        object.__setattr__(self, "interface_id", str(self.interface_id))
        object.__setattr__(self, "evidence_id", str(self.evidence_id))
        if self.api_interface_id is not None:
            object.__setattr__(self, "api_interface_id", str(self.api_interface_id))
        meta = self.metadata if isinstance(self.metadata, Mapping) else {}
        object.__setattr__(self, "metadata", MappingProxyType(dict(meta)))

    @property
    def available(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "status": self.status,
            "reason_code": self.reason_code,
            "diagnostic": self.diagnostic,
            "interface_id": self.interface_id,
            "evidence_id": self.evidence_id,
            "api_interface_id": self.api_interface_id,
            "available": False,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clip(text: str, *, limit: int = MAX_DIAGNOSTIC) -> str:
    raw = str(text or "")
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 3)] + "..."


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str):
        raise AssurancePublicApiError(
            f"{name} must be a string",
            reason_code="invalid_type",
        )
    text = unicodedata.normalize("NFC", value)
    if not empty and not text.strip():
        raise AssurancePublicApiError(
            f"{name} must not be empty",
            reason_code="empty_value",
        )
    if len(text) > MAX_TEXT_CHARS:
        raise AssurancePublicApiError(
            f"{name} exceeds maximum length",
            reason_code="bounds",
        )
    return text


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name, empty=True) or None


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if not _TOKEN_RE.match(text):
        raise AssurancePublicApiError(
            f"{name} is not a valid token",
            reason_code="invalid_token",
        )
    return text


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise AssurancePublicApiError(
            f"{name} must be a bool",
            reason_code="invalid_type",
        )
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AssurancePublicApiError(
            f"{name} must be a mapping",
            reason_code="invalid_type",
        )
    return dict(value)


def _looks_like_host_path(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if _ABSOLUTE_PATH_RE.match(text):
        return True
    for marker in _HOME_PATH_MARKERS:
        if marker in text:
            return True
    # Windows drive-relative style without leading slash still absolute-ish.
    if len(text) >= 3 and text[1] == ":" and text[2] in {"\\", "/"}:
        return True
    return False


def _reject_path_exposure(value: Any, *, path: str = "$") -> None:
    """Reject absolute host paths and host-fallback field names on public inputs."""

    try:
        reject_private_model_authority_and_host_fallbacks(value, path=path)
    except Exception as exc:
        # Surface host/private leakage through the public path-exposure gate.
        message = str(exc)
        if any(
            token in message
            for token in (
                "host fallback",
                "private data",
                "absolute host path",
            )
        ):
            raise PathExposureError(message, field=path) from exc
        raise
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_str = str(key)
            lowered = key_str.lower()
            if lowered.endswith("_path") or lowered in {
                "path",
                "filepath",
                "file_path",
                "workdir",
                "worktree",
                "worktree_path",
                "repo_root",
                "repository_path",
                "absolute_path",
                "local_path",
                "host_path",
            }:
                if isinstance(item, str) and _looks_like_host_path(item):
                    raise PathExposureError(
                        f"{path}.{key_str} exposes an absolute host path",
                        field=f"{path}.{key_str}",
                    )
            _reject_path_exposure(item, path=f"{path}.{key_str}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_path_exposure(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and _looks_like_host_path(value):
        # Bare absolute strings in free-form public kwargs are rejected.
        raise PathExposureError(
            f"{path} exposes an absolute host path",
            field=path,
        )


def _repo_relative_path(value: Any, name: str) -> str:
    """Normalize a repository-relative path; reject absolute host paths."""

    text = _text(value, name)
    if _looks_like_host_path(text):
        raise PathExposureError(
            f"{name} must be a repository-relative path, not an absolute host path",
            field=name,
        )
    normalized = str(PurePosixPath(text))
    if normalized.startswith("..") or "/../" in f"/{normalized}/":
        raise PathExposureError(
            f"{name} must not escape the repository via '..'",
            field=name,
        )
    if normalized.startswith("/"):
        raise PathExposureError(
            f"{name} must be repository-relative",
            field=name,
        )
    return normalized


def _thaw_structured(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _thaw_structured(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_structured(item) for item in value]
    return value


def _identity_cid(payload: Mapping[str, Any]) -> str:
    return cid_for_structured(_thaw_structured(dict(payload)))


def _normalize_plan(
    value: Any,
) -> MutationCampaignPlan:
    if isinstance(value, MutationCampaignPlan):
        return value
    if isinstance(value, Mapping):
        try:
            return MutationCampaignPlan.from_dict(value)
        except (MutationContractError, TypeError, ValueError, KeyError) as exc:
            raise AssurancePublicApiError(
                f"plan is not a sealed MutationCampaignPlan: {exc}",
                reason_code="invalid_plan",
            ) from exc
    # MutationCampaignPlanResult-like
    plan_attr = getattr(value, "plan", None)
    if isinstance(plan_attr, MutationCampaignPlan):
        return plan_attr
    if isinstance(plan_attr, Mapping):
        return _normalize_plan(plan_attr)
    raise AssurancePublicApiError(
        "plan must be MutationCampaignPlan, plan result, or mapping",
        reason_code="invalid_plan",
    )


def _normalize_verification_policy(value: Any) -> dict[str, Any]:
    if value is None:
        raise AssurancePublicApiError(
            "verification_policy is required",
            reason_code="missing_verification_policy",
        )
    if hasattr(value, "to_dict") and callable(value.to_dict):
        payload = value.to_dict()
        if not isinstance(payload, Mapping):
            raise AssurancePublicApiError(
                "verification_policy.to_dict() must return a mapping",
                reason_code="invalid_verification_policy",
            )
        data = dict(payload)
    elif isinstance(value, Mapping):
        data = dict(value)
    elif isinstance(value, str):
        # Allow bare policy CID with explicit binding.
        data = {"policy_cid": value}
    else:
        raise AssurancePublicApiError(
            "verification_policy must be a mapping, policy binding, or CID",
            reason_code="invalid_verification_policy",
        )
    _reject_path_exposure(data, path="verification_policy")
    return data


def _normalize_repository_state(value: Any) -> dict[str, Any]:
    if value is None:
        raise AssurancePublicApiError(
            "repository_state is required",
            reason_code="missing_repository_state",
        )
    if hasattr(value, "to_dict") and callable(value.to_dict):
        payload = value.to_dict()
        if not isinstance(payload, Mapping):
            raise AssurancePublicApiError(
                "repository_state.to_dict() must return a mapping",
                reason_code="invalid_repository_state",
            )
        data = dict(payload)
    elif isinstance(value, Mapping):
        data = dict(value)
    elif isinstance(value, str):
        data = {"repository_state_cid": value}
    else:
        raise AssurancePublicApiError(
            "repository_state must be a mapping, binding, or state CID",
            reason_code="invalid_repository_state",
        )
    _reject_path_exposure(data, path="repository_state")
    return data


def _normalize_manifest(value: Any) -> dict[str, Any]:
    if value is None:
        raise AssurancePublicApiError(
            "assurance_manifest is required",
            reason_code="missing_assurance_manifest",
        )
    if hasattr(value, "to_dict") and callable(value.to_dict):
        payload = value.to_dict()
        if not isinstance(payload, Mapping):
            raise AssurancePublicApiError(
                "assurance_manifest.to_dict() must return a mapping",
                reason_code="invalid_assurance_manifest",
            )
        data = dict(payload)
    elif isinstance(value, Mapping):
        data = dict(value)
    else:
        raise AssurancePublicApiError(
            "assurance_manifest must be a mapping or sealed manifest",
            reason_code="invalid_assurance_manifest",
        )
    _reject_path_exposure(data, path="assurance_manifest")
    return data


def _finding_to_dict(finding: Any) -> dict[str, Any]:
    if isinstance(finding, VacuityFinding):
        return finding.to_dict()
    if isinstance(finding, Mapping):
        return dict(finding)
    if hasattr(finding, "to_dict") and callable(finding.to_dict):
        payload = finding.to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    raise AssurancePublicApiError(
        "finding must be VacuityFinding or mapping",
        reason_code="invalid_finding",
    )


def _report_to_dict(report: Any) -> dict[str, Any]:
    if isinstance(report, Mapping):
        return dict(report)
    if hasattr(report, "to_dict") and callable(report.to_dict):
        payload = report.to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    # Best-effort attribute projection for lightweight test doubles.
    fields = (
        "candidate_id",
        "candidate_cid",
        "mutant_identity_cid",
        "terminal_status",
        "outcome_status",
        "disposition",
        "report_cid",
        "outcome_cid",
        "reason_codes",
    )
    projected: dict[str, Any] = {}
    for name in fields:
        if hasattr(report, name):
            projected[name] = getattr(report, name)
    if not projected:
        raise AssurancePublicApiError(
            "candidate report must be a mapping or object with to_dict()",
            reason_code="invalid_report",
        )
    return projected


# ---------------------------------------------------------------------------
# Result types for local composers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VacuityCampaignAnalysisResult:
    """Aggregate vacuity analysis across the four required families.

    Interface: ``VacuityCampaignAnalysisResult@1`` / ``analyze_vacuity@1``
    """

    interface_id: str
    repository_state_cid: str
    assurance_manifest_cid: str | None
    family_results: tuple[Mapping[str, Any], ...]
    findings: tuple[Mapping[str, Any], ...]
    finding_cids: tuple[str, ...]
    residual_properties: tuple[str, ...]
    precise_nonclaims: tuple[str, ...]
    families_analyzed: tuple[str, ...]
    reason_codes: tuple[str, ...]
    terminal_status: str
    production_policy_changed: bool = False
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "interface_id", _text(self.interface_id, "interface_id")
        )
        if self.interface_id != ANALYZE_VACUITY_INTERFACE:
            raise AssurancePublicApiError(
                "interface_id must be analyze_vacuity@1",
                reason_code="invalid_interface",
            )
        object.__setattr__(
            self,
            "repository_state_cid",
            _text(self.repository_state_cid, "repository_state_cid"),
        )
        if self.assurance_manifest_cid is not None:
            object.__setattr__(
                self,
                "assurance_manifest_cid",
                _text(self.assurance_manifest_cid, "assurance_manifest_cid"),
            )
        object.__setattr__(
            self,
            "family_results",
            tuple(MappingProxyType(dict(item)) for item in self.family_results),
        )
        object.__setattr__(
            self,
            "findings",
            tuple(MappingProxyType(dict(item)) for item in self.findings),
        )
        if len(self.findings) > MAX_FINDINGS:
            raise AssurancePublicApiError(
                "findings exceed maximum",
                reason_code="bounds",
            )
        object.__setattr__(
            self,
            "finding_cids",
            tuple(str(item) for item in self.finding_cids),
        )
        object.__setattr__(
            self,
            "residual_properties",
            tuple(str(item) for item in self.residual_properties),
        )
        object.__setattr__(
            self,
            "precise_nonclaims",
            tuple(str(item) for item in self.precise_nonclaims),
        )
        object.__setattr__(
            self,
            "families_analyzed",
            tuple(str(item) for item in self.families_analyzed),
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(str(item) for item in self.reason_codes),
        )
        object.__setattr__(
            self,
            "terminal_status",
            _text(self.terminal_status, "terminal_status"),
        )
        object.__setattr__(
            self,
            "production_policy_changed",
            _bool(self.production_policy_changed, "production_policy_changed"),
        )
        if self.production_policy_changed:
            raise AssurancePublicApiError(
                "analyze_vacuity must not change production policy",
                reason_code="production_policy_change",
            )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        meta = self.metadata if isinstance(self.metadata, Mapping) else {}
        _reject_path_exposure(dict(meta), path="metadata")
        object.__setattr__(self, "metadata", MappingProxyType(dict(meta)))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": VACUITY_CAMPAIGN_RESULT_SCHEMA,
            "interface_id": self.interface_id,
            "repository_state_cid": self.repository_state_cid,
            "assurance_manifest_cid": self.assurance_manifest_cid,
            "family_results": [dict(item) for item in self.family_results],
            "findings": [dict(item) for item in self.findings],
            "finding_cids": list(self.finding_cids),
            "residual_properties": list(self.residual_properties),
            "precise_nonclaims": list(self.precise_nonclaims),
            "families_analyzed": list(self.families_analyzed),
            "reason_codes": list(self.reason_codes),
            "terminal_status": self.terminal_status,
            "production_policy_changed": False,
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }

    @property
    def result_cid(self) -> str:
        return _identity_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["result_cid"] = self.result_cid
        return payload


@dataclass(frozen=True)
class MutationCampaignExecutionResult:
    """Bounded campaign execution result for ``execute_mutation_campaign@1``.

    Interface: ``MutationCampaignExecutionResult@1``
    """

    interface_id: str
    plan_id: str
    plan_cid: str
    repository_state_cid: str
    verification_policy_cid: str
    candidate_reports: tuple[Mapping[str, Any], ...]
    candidate_cids: tuple[str, ...]
    outcome_cids: tuple[str, ...]
    killed_count: int
    survivor_count: int
    invalid_count: int
    inconclusive_count: int
    reason_codes: tuple[str, ...]
    terminal_status: str
    require_sandbox: bool = True
    network_disabled: bool = True
    production_policy_changed: bool = False
    notes: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "interface_id", _text(self.interface_id, "interface_id")
        )
        if self.interface_id != EXECUTE_MUTATION_CAMPAIGN_INTERFACE:
            raise AssurancePublicApiError(
                "interface_id must be execute_mutation_campaign@1",
                reason_code="invalid_interface",
            )
        object.__setattr__(self, "plan_id", _token(self.plan_id, "plan_id"))
        object.__setattr__(self, "plan_cid", _text(self.plan_cid, "plan_cid"))
        object.__setattr__(
            self,
            "repository_state_cid",
            _text(self.repository_state_cid, "repository_state_cid"),
        )
        object.__setattr__(
            self,
            "verification_policy_cid",
            _text(self.verification_policy_cid, "verification_policy_cid"),
        )
        reports = tuple(MappingProxyType(dict(item)) for item in self.candidate_reports)
        if len(reports) > MAX_CANDIDATES:
            raise AssurancePublicApiError(
                "candidate_reports exceed maximum",
                reason_code="bounds",
            )
        object.__setattr__(self, "candidate_reports", reports)
        object.__setattr__(
            self, "candidate_cids", tuple(str(item) for item in self.candidate_cids)
        )
        object.__setattr__(
            self, "outcome_cids", tuple(str(item) for item in self.outcome_cids)
        )
        for name in (
            "killed_count",
            "survivor_count",
            "invalid_count",
            "inconclusive_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise AssurancePublicApiError(
                    f"{name} must be a non-negative int",
                    reason_code="invalid_type",
                )
        object.__setattr__(
            self, "reason_codes", tuple(str(item) for item in self.reason_codes)
        )
        object.__setattr__(
            self, "terminal_status", _text(self.terminal_status, "terminal_status")
        )
        object.__setattr__(
            self, "require_sandbox", _bool(self.require_sandbox, "require_sandbox")
        )
        object.__setattr__(
            self, "network_disabled", _bool(self.network_disabled, "network_disabled")
        )
        object.__setattr__(
            self,
            "production_policy_changed",
            _bool(self.production_policy_changed, "production_policy_changed"),
        )
        if not self.require_sandbox:
            raise AssurancePublicApiError(
                "require_sandbox must be true",
                reason_code="sandbox_required",
            )
        if not self.network_disabled:
            raise AssurancePublicApiError(
                "network_disabled must be true",
                reason_code="network_required_disabled",
            )
        if self.production_policy_changed:
            raise AssurancePublicApiError(
                "execute_mutation_campaign must not change production policy",
                reason_code="production_policy_change",
            )
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes"))
        meta = self.metadata if isinstance(self.metadata, Mapping) else {}
        _reject_path_exposure(dict(meta), path="metadata")
        object.__setattr__(self, "metadata", MappingProxyType(dict(meta)))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": MUTATION_CAMPAIGN_EXECUTION_RESULT_SCHEMA,
            "interface_id": self.interface_id,
            "plan_id": self.plan_id,
            "plan_cid": self.plan_cid,
            "repository_state_cid": self.repository_state_cid,
            "verification_policy_cid": self.verification_policy_cid,
            "candidate_reports": [dict(item) for item in self.candidate_reports],
            "candidate_cids": list(self.candidate_cids),
            "outcome_cids": list(self.outcome_cids),
            "killed_count": self.killed_count,
            "survivor_count": self.survivor_count,
            "invalid_count": self.invalid_count,
            "inconclusive_count": self.inconclusive_count,
            "reason_codes": list(self.reason_codes),
            "terminal_status": self.terminal_status,
            "require_sandbox": True,
            "network_disabled": True,
            "production_policy_changed": False,
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }

    @property
    def result_cid(self) -> str:
        return _identity_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["result_cid"] = self.result_cid
        return payload


# ---------------------------------------------------------------------------
# Lazy API resolution
# ---------------------------------------------------------------------------


def _load_api(name: str) -> Callable[..., Any]:
    """Import and return the owning leaf implementation of a required API."""

    owner = _API_OWNERS.get(name)
    if owner is None:
        raise UnknownCommandError(name)
    module_path, attr = owner
    # Local composers: bind without circular re-import of the same symbol mid-load.
    if module_path.endswith(".api") and attr in {
        "analyze_vacuity",
        "execute_mutation_campaign",
    }:
        value = globals().get(attr)
        if value is None or not callable(value):
            raise AssuranceApiUnavailableError(
                name,
                reason_code="missing_export",
                diagnostic=f"local composer {attr!r} is not defined",
                status=ApiAvailability.MISSING.value,
            )
        return value
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise AssuranceApiUnavailableError(
            name,
            reason_code="import_failed",
            diagnostic=f"failed to import {module_path!r}: {exc}",
            status=ApiAvailability.MISSING.value,
        ) from exc
    try:
        value = getattr(module, attr)
    except AttributeError as exc:
        raise AssuranceApiUnavailableError(
            name,
            reason_code="missing_export",
            diagnostic=f"{module_path!r} has no attribute {attr!r}",
            status=ApiAvailability.MISSING.value,
        ) from exc
    if not callable(value):
        raise AssuranceApiUnavailableError(
            name,
            reason_code="not_callable",
            diagnostic=f"{module_path}.{attr} is not callable",
            status=ApiAvailability.INCOMPATIBLE.value,
        )
    return value


def _reject_unknown_params(
    fn: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Reject kwargs that are not parameters of ``fn`` (closed field set)."""

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return

    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return

    try:
        signature.bind_partial(*args, **dict(kwargs))
    except TypeError as exc:
        message = str(exc)
        if "unexpected keyword argument" in message:
            unknown: list[str] = []
            for key in kwargs:
                try:
                    signature.bind_partial(*args, **{key: kwargs[key]})
                except TypeError:
                    unknown.append(key)
            if unknown:
                raise UnknownFieldError(unknown, context=context) from exc
        raise AssurancePublicApiError(
            f"{context} rejected parameters: {message}",
            reason_code="invalid_parameters",
        ) from exc


def _closed_mapping(
    value: Mapping[str, Any] | None,
    allowed: frozenset[str],
    *,
    name: str,
) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise AssurancePublicApiError(
            f"{name} must be a mapping",
            reason_code="invalid_mapping",
        )
    data = dict(value)
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise UnknownFieldError(unknown, context=name)
    return data


# ---------------------------------------------------------------------------
# Module-level helpers / pins
# ---------------------------------------------------------------------------


def public_api_evidence_id() -> str:
    """Return the public-API evidence pin."""

    return AAE_PUBLIC_API_EVIDENCE


def public_api_interface_id() -> str:
    """Return the versioned package public-API interface pin."""

    return ADVERSARIAL_ASSURANCE_PUBLIC_API_INTERFACE


def public_api_schema() -> str:
    """Return the public API schema identifier."""

    return ADVERSARIAL_ASSURANCE_PUBLIC_API_SCHEMA


def campaign_api_interface_id() -> str:
    """Return the AssuranceCampaignApi class interface pin."""

    return ASSURANCE_CAMPAIGN_API_INTERFACE


def required_public_apis() -> tuple[str, ...]:
    """Return the closed primary public entry-point names."""

    return REQUIRED_PUBLIC_APIS


def required_commands() -> tuple[str, ...]:
    """Return the closed invoke-command vocabulary."""

    return REQUIRED_COMMANDS


def api_interface_id(name: str) -> str:
    """Return the stable interface id for one required public API."""

    if name not in _API_INTERFACE_IDS:
        raise UnknownCommandError(name)
    return _API_INTERFACE_IDS[name]


def api_interface_ids() -> Mapping[str, str]:
    """Return the closed mapping of required API name → interface id."""

    return MappingProxyType(dict(_API_INTERFACE_IDS))


def resolve_public_api(name: str) -> Callable[..., Any]:
    """Resolve a required public API callable (lazy leaf import).

    Returns the exact leaf callable so signatures, return types, and identity
    gates match the owning implementation.
    """

    if name not in REQUIRED_PUBLIC_APIS:
        raise UnknownCommandError(name)
    cached = globals().get(name)
    if cached is not None and callable(cached):
        owner = _API_OWNERS[name]
        if getattr(cached, "__name__", None) == owner[1]:
            return cached  # type: ignore[return-value]
    value = _load_api(name)
    globals()[name] = value
    return value


# ---------------------------------------------------------------------------
# Local composers: analyze_vacuity / execute_mutation_campaign
# ---------------------------------------------------------------------------


def _build_default_header(
    *,
    repository_state: Mapping[str, Any],
    interface_id: str,
    artifact_kind: str,
) -> AssuranceArtifactHeader:
    repo_id = str(
        repository_state.get("repository_id")
        or repository_state.get("repository")
        or "repository:sha256:adversarial-assurance-public-api"
    )
    state_cid = str(
        repository_state.get("repository_state_cid")
        or repository_state.get("state_cid")
        or ""
    )
    if not state_cid:
        state_cid = _identity_cid(
            {"repository_id": repo_id, "kind": "public_api_state_binding"}
        )
    environment_cid = str(repository_state.get("environment_cid") or "") or _identity_cid(
        {"kind": "public_api_environment", "repository_id": repo_id}
    )
    dependency_lock_cid = (
        str(repository_state.get("dependency_lock_cid") or "")
        or _identity_cid({"kind": "public_api_dependency_lock", "repository_id": repo_id})
    )
    policy_cid = str(repository_state.get("policy_cid") or "") or None
    generator = GeneratorIdentity(
        generator_id=GENERATOR_ID,
        generator_version=GENERATOR_VERSION,
        interface_id=interface_id,
    )
    versions = VersionBinding(
        operator_id="public_api",
        operator_version=GENERATOR_VERSION,
        campaign_policy_id="public_api_campaign",
        campaign_policy_version=GENERATOR_VERSION,
        generator=generator,
    )
    provenance = ArtifactProvenance(
        producer_id=GENERATOR_ID,
        producer_version=GENERATOR_VERSION,
        execution_mode=ExecutionMode.LIVE,
        authority_source=AuthoritySource.DETERMINISTIC,
        input_cids=(state_cid,),
        tool_ids=("aae.public_api.v1",),
        policy_cid=policy_cid,
        notes=None,
    )
    return AssuranceArtifactHeader(
        artifact_kind=artifact_kind,
        repository_id=repo_id,
        repository_state_cid=state_cid,
        target_symbol_ids=("aae.public_api",),
        target_artifact_cids=(state_cid,),
        capsule_cids=(state_cid,),
        proof_unit_cids=(state_cid,),
        environment_cid=environment_cid,
        dependency_lock_cid=dependency_lock_cid,
        versions=versions,
        provenance=provenance,
        terminal_status=AssuranceTerminalStatus.COMPLETE,
        receipt_cids=(),
        proof_cids=(),
        metadata={"track": "public-api", "interface_id": interface_id},
    )


def _dispatch_vacuity_family(
    family: str,
    subject: Any,
    header: AssuranceArtifactHeader | Mapping[str, Any],
    *,
    notes: str | None,
    metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    family_key = str(family)
    if family_key == VacuityFamily.FORMAL_PROOF.value:
        from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.vacuity_formal_policy import (
            analyze_formal_vacuity,
        )

        result = analyze_formal_vacuity(
            subject, header, notes=notes, metadata=metadata
        )
    elif family_key == VacuityFamily.POLICY.value:
        from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.vacuity_formal_policy import (
            analyze_policy_vacuity,
        )

        result = analyze_policy_vacuity(
            subject, header, notes=notes, metadata=metadata
        )
    elif family_key == VacuityFamily.TEST.value:
        from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.vacuity_test_zk import (
            analyze_test_vacuity,
        )

        result = analyze_test_vacuity(
            subject, header, notes=notes, metadata=metadata
        )
    elif family_key == VacuityFamily.ZK_RECEIPT.value:
        from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.vacuity_test_zk import (
            analyze_zk_receipt_vacuity,
        )

        result = analyze_zk_receipt_vacuity(
            subject, header, notes=notes, metadata=metadata
        )
    else:
        raise AssurancePublicApiError(
            f"unknown vacuity family: {family_key!r}",
            reason_code="unknown_vacuity_family",
        )
    if hasattr(result, "to_dict") and callable(result.to_dict):
        payload = result.to_dict()
        if not isinstance(payload, Mapping):
            raise AssurancePublicApiError(
                "vacuity family result.to_dict() must return a mapping",
                reason_code="invalid_vacuity_result",
            )
        return dict(payload)
    if isinstance(result, Mapping):
        return dict(result)
    raise AssurancePublicApiError(
        "vacuity family analyzer returned unsupported type",
        reason_code="invalid_vacuity_result",
    )


def analyze_vacuity(
    assurance_manifest: Any,
    repository_state: Any,
    *,
    formal_subject: Any | None = None,
    policy_subject: Any | None = None,
    test_subject: Any | None = None,
    zk_receipt_subject: Any | None = None,
    subjects: Sequence[Mapping[str, Any]] | None = None,
    header: AssuranceArtifactHeader | Mapping[str, Any] | None = None,
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> VacuityCampaignAnalysisResult:
    """Analyze vacuity across formal, policy, test, and ZK/receipt families.

    Interface: ``analyze_vacuity@1``

    Plan signature: ``analyze_vacuity(assurance_manifest, repository_state)``.

    Family subjects are supplied as explicit kwargs or a ``subjects`` sequence
    of mappings with a ``vacuity_family`` field. At least one subject is
    required (fail closed). Production policy is never changed. Absolute host
    paths are rejected on public inputs.
    """

    manifest = _normalize_manifest(assurance_manifest)
    repo_state = _normalize_repository_state(repository_state)
    meta = dict(metadata or {})
    _reject_path_exposure(meta, path="metadata")
    note_text = _optional_text(notes, "notes")

    subject_jobs: list[tuple[str, Any]] = []
    if formal_subject is not None:
        subject_jobs.append((VacuityFamily.FORMAL_PROOF.value, formal_subject))
    if policy_subject is not None:
        subject_jobs.append((VacuityFamily.POLICY.value, policy_subject))
    if test_subject is not None:
        subject_jobs.append((VacuityFamily.TEST.value, test_subject))
    if zk_receipt_subject is not None:
        subject_jobs.append((VacuityFamily.ZK_RECEIPT.value, zk_receipt_subject))
    if subjects is not None:
        if not isinstance(subjects, Sequence) or isinstance(subjects, (str, bytes)):
            raise AssurancePublicApiError(
                "subjects must be a sequence of mappings",
                reason_code="invalid_subjects",
            )
        for index, item in enumerate(subjects):
            if not isinstance(item, Mapping):
                raise AssurancePublicApiError(
                    f"subjects[{index}] must be a mapping",
                    reason_code="invalid_subjects",
                )
            data = dict(item)
            _reject_path_exposure(data, path=f"subjects[{index}]")
            family = data.get("vacuity_family") or data.get("family")
            if family is None:
                raise AssurancePublicApiError(
                    f"subjects[{index}] requires vacuity_family",
                    reason_code="missing_vacuity_family",
                )
            subject_payload = data.get("subject", data)
            subject_jobs.append((str(family), subject_payload))

    if not subject_jobs:
        raise AssurancePublicApiError(
            "analyze_vacuity requires at least one vacuity subject",
            reason_code="missing_vacuity_subjects",
        )

    if header is None:
        sealed_header: AssuranceArtifactHeader | Mapping[str, Any] = (
            _build_default_header(
                repository_state=repo_state,
                interface_id=ANALYZE_VACUITY_INTERFACE,
                artifact_kind="vacuity_finding",
            )
        )
    else:
        sealed_header = header

    family_results: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    finding_cids: list[str] = []
    residual: list[str] = []
    nonclaims: list[str] = []
    families: list[str] = []

    for family, subject in subject_jobs:
        if isinstance(subject, Mapping):
            _reject_path_exposure(dict(subject), path=f"subject.{family}")
        result = _dispatch_vacuity_family(
            family,
            subject,
            sealed_header,
            notes=note_text,
            metadata=meta or None,
        )
        family_results.append(result)
        families.append(str(result.get("vacuity_family") or family))
        for finding in result.get("findings") or ():
            findings.append(_finding_to_dict(finding))
        for cid in result.get("finding_cids") or ():
            finding_cids.append(str(cid))
        for item in result.get("residual_properties") or ():
            residual.append(str(item))
        for item in result.get("precise_nonclaims") or ():
            nonclaims.append(str(item))

    # Deterministic ordering of aggregate collections.
    finding_cids = sorted(set(finding_cids))
    residual = tuple(sorted(set(residual)))
    nonclaims = tuple(sorted(set(nonclaims)))
    families_unique = tuple(sorted(set(families)))

    state_cid = str(
        repo_state.get("repository_state_cid")
        or repo_state.get("state_cid")
        or _identity_cid(repo_state)
    )
    manifest_cid = None
    for key in ("manifest_cid", "assurance_manifest_cid", "result_cid"):
        if manifest.get(key):
            manifest_cid = str(manifest[key])
            break
    if manifest_cid is None:
        manifest_cid = _identity_cid(manifest)

    terminal = (
        AssuranceTerminalStatus.COMPLETE.value
        if findings or family_results
        else AssuranceTerminalStatus.INCONCLUSIVE.value
    )
    reasons = (
        REASON_VACUITY_ANALYZED,
        REASON_NO_PRODUCTION_POLICY_CHANGE,
        REASON_NO_ARBITRARY_PATH_EXPOSURE,
        REASON_CANONICAL_BINDINGS,
    )

    return VacuityCampaignAnalysisResult(
        interface_id=ANALYZE_VACUITY_INTERFACE,
        repository_state_cid=state_cid,
        assurance_manifest_cid=manifest_cid,
        family_results=tuple(family_results),
        findings=tuple(findings),
        finding_cids=tuple(finding_cids),
        residual_properties=residual,
        precise_nonclaims=nonclaims,
        families_analyzed=families_unique,
        reason_codes=reasons,
        terminal_status=terminal,
        production_policy_changed=False,
        notes=note_text,
        metadata=meta,
    )


def _classify_report_bucket(report: Mapping[str, Any]) -> str:
    status = str(
        report.get("terminal_status")
        or report.get("outcome_status")
        or report.get("disposition")
        or ""
    ).lower()
    if any(token in status for token in ("kill", "killed")):
        return "killed"
    if any(token in status for token in ("invalid", "uncompilable")):
        return "invalid"
    if any(token in status for token in ("surviv", "equivalent")):
        return "survivor"
    if status in {"", "inconclusive", "human_review_required", "baseline_blocked"}:
        return "inconclusive"
    if "fail" in status or "error" in status:
        return "invalid"
    return "inconclusive"


def execute_mutation_campaign(
    plan: Any,
    verification_policy: Any,
    *,
    candidates: Sequence[MutationCandidate | Mapping[str, Any]] | None = None,
    expected_detections: Sequence[ExpectedDetectionSet | Mapping[str, Any]] | None = None,
    precomputed_reports: Sequence[Any] | None = None,
    candidate_executor: CandidateExecutor | None = None,
    notes: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> MutationCampaignExecutionResult:
    """Execute a planned mutation campaign under disposable isolation policy.

    Interface: ``execute_mutation_campaign@1``

    Plan signature: ``execute_mutation_campaign(plan, verification_policy)``.

    Execution is dependency-injected for hermetic contract tests:

    * ``precomputed_reports`` — already-sealed per-candidate reports
    * ``candidate_executor`` — callable invoked once per candidate

    At least one of those surfaces is required (fail closed). The campaign
    never mutates production policy, always requires disposable sandbox
    isolation, and never exposes absolute host paths through public inputs or
    result metadata.
    """

    sealed_plan = _normalize_plan(plan)
    policy = _normalize_verification_policy(verification_policy)
    meta = dict(metadata or {})
    _reject_path_exposure(meta, path="metadata")
    note_text = _optional_text(notes, "notes")

    if not sealed_plan.require_sandbox:
        raise AssurancePublicApiError(
            "campaign plan require_sandbox must be true",
            reason_code="sandbox_required",
        )
    if not sealed_plan.require_rollback:
        raise AssurancePublicApiError(
            "campaign plan require_rollback must be true",
            reason_code="rollback_required",
        )

    policy_cid = str(
        policy.get("policy_cid")
        or policy.get("verification_policy_cid")
        or policy.get("cid")
        or ""
    )
    if not policy_cid:
        policy_cid = _identity_cid(policy)

    reports: list[dict[str, Any]] = []
    reasons: list[str] = [
        REASON_CAMPAIGN_EXECUTED,
        REASON_NO_PRODUCTION_POLICY_CHANGE,
        REASON_NO_ARBITRARY_PATH_EXPOSURE,
        REASON_DISPOSABLE_WORKTREE_REQUIRED,
        REASON_NETWORK_DISABLED,
        REASON_CANONICAL_BINDINGS,
    ]

    if precomputed_reports is not None:
        if not isinstance(precomputed_reports, Sequence) or isinstance(
            precomputed_reports, (str, bytes)
        ):
            raise AssurancePublicApiError(
                "precomputed_reports must be a sequence",
                reason_code="invalid_reports",
            )
        if len(precomputed_reports) > MAX_CANDIDATES:
            raise AssurancePublicApiError(
                "precomputed_reports exceed maximum",
                reason_code="bounds",
            )
        for index, item in enumerate(precomputed_reports):
            report = _report_to_dict(item)
            _reject_path_exposure(report, path=f"precomputed_reports[{index}]")
            reports.append(report)
        reasons.append(REASON_PRECOMPUTED_REPORTS)
    elif candidate_executor is not None:
        if not callable(candidate_executor):
            raise AssurancePublicApiError(
                "candidate_executor must be callable",
                reason_code="invalid_executor",
            )
        reasons.append(REASON_INJECTED_EXECUTOR)
        candidate_list: list[Any]
        if candidates is not None:
            if not isinstance(candidates, Sequence) or isinstance(
                candidates, (str, bytes)
            ):
                raise AssurancePublicApiError(
                    "candidates must be a sequence",
                    reason_code="invalid_candidates",
                )
            candidate_list = list(candidates)
        else:
            # Plan only binds candidate CIDs; executor receives CID bindings.
            candidate_list = [
                {"candidate_cid": cid, "candidate_id": f"cand_{index:04d}"}
                for index, cid in enumerate(sealed_plan.candidate_cids)
            ]
        if len(candidate_list) > MAX_CANDIDATES:
            raise AssurancePublicApiError(
                "candidates exceed maximum",
                reason_code="bounds",
            )
        detection_list: list[Any] | None
        if expected_detections is not None:
            if not isinstance(expected_detections, Sequence) or isinstance(
                expected_detections, (str, bytes)
            ):
                raise AssurancePublicApiError(
                    "expected_detections must be a sequence",
                    reason_code="invalid_detections",
                )
            detection_list = list(expected_detections)
            if len(detection_list) != len(candidate_list):
                raise AssurancePublicApiError(
                    "expected_detections must align 1:1 with candidates",
                    reason_code="detection_alignment",
                )
        else:
            detection_list = None

        for index, candidate in enumerate(candidate_list):
            if isinstance(candidate, Mapping):
                _reject_path_exposure(dict(candidate), path=f"candidates[{index}]")
            detection = None if detection_list is None else detection_list[index]
            if isinstance(detection, Mapping):
                _reject_path_exposure(
                    dict(detection), path=f"expected_detections[{index}]"
                )
            try:
                raw = candidate_executor(
                    candidate=candidate,
                    expected_detection=detection,
                    plan=sealed_plan,
                    verification_policy=policy,
                    index=index,
                )
            except AssurancePublicApiError:
                raise
            except Exception as exc:  # noqa: BLE001 — surface as typed failure
                raise AssurancePublicApiError(
                    f"candidate_executor failed for index {index}: {_clip(str(exc))}",
                    reason_code="executor_failed",
                    details={"index": index},
                ) from exc
            report = _report_to_dict(raw)
            _reject_path_exposure(report, path=f"candidate_reports[{index}]")
            reports.append(report)
    else:
        raise AssurancePublicApiError(
            "execute_mutation_campaign requires precomputed_reports or "
            "candidate_executor (fail closed without execution surface)",
            reason_code="missing_execution_surface",
        )

    killed = survivor = invalid = inconclusive = 0
    candidate_cids: list[str] = []
    outcome_cids: list[str] = []
    for report in reports:
        bucket = _classify_report_bucket(report)
        if bucket == "killed":
            killed += 1
        elif bucket == "survivor":
            survivor += 1
        elif bucket == "invalid":
            invalid += 1
        else:
            inconclusive += 1
        for key in ("candidate_cid", "mutant_identity_cid"):
            if report.get(key):
                candidate_cids.append(str(report[key]))
                break
        for key in ("outcome_cid", "report_cid", "result_cid"):
            if report.get(key):
                outcome_cids.append(str(report[key]))
                break

    if reports and killed + survivor + invalid == len(reports):
        terminal = AssuranceTerminalStatus.COMPLETE.value
    elif reports:
        terminal = AssuranceTerminalStatus.COMPLETE.value
    else:
        terminal = AssuranceTerminalStatus.INCONCLUSIVE.value

    # Stable unique ordering for identity.
    candidate_cids_u = tuple(sorted(set(candidate_cids)))
    outcome_cids_u = tuple(sorted(set(outcome_cids)))
    # Keep report order as executed (not sorted) for honest sequence evidence.

    return MutationCampaignExecutionResult(
        interface_id=EXECUTE_MUTATION_CAMPAIGN_INTERFACE,
        plan_id=sealed_plan.plan_id,
        plan_cid=sealed_plan.plan_cid,
        repository_state_cid=sealed_plan.repository_state_cid,
        verification_policy_cid=policy_cid,
        candidate_reports=tuple(reports),
        candidate_cids=candidate_cids_u,
        outcome_cids=outcome_cids_u,
        killed_count=killed,
        survivor_count=survivor,
        invalid_count=invalid,
        inconclusive_count=inconclusive,
        reason_codes=tuple(dict.fromkeys(reasons)),
        terminal_status=terminal,
        require_sandbox=True,
        network_disabled=True,
        production_policy_changed=False,
        notes=note_text,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Module-level required APIs (exact leaf identities via __getattr__)
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any:
    """Resolve required public API callables from their owning modules."""

    if name not in _API_OWNERS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Local composers are defined above; return them directly.
    if name in {"analyze_vacuity", "execute_mutation_campaign"}:
        value = globals()[name]
        return value
    value = _load_api(name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


# ---------------------------------------------------------------------------
# AssuranceCampaignApi composition
# ---------------------------------------------------------------------------


@dataclass
class AssuranceCampaignApi:
    """Injectable composition facade for the twelve required public APIs."""

    create_assurance_manifest_fn: Callable[..., Any] | None = None
    generate_mutation_candidates_fn: Callable[..., Any] | None = None
    predict_detection_set_fn: Callable[..., Any] | None = None
    execute_mutation_fn: Callable[..., Any] | None = None
    classify_mutation_outcome_fn: Callable[..., Any] | None = None
    diagnose_surviving_mutant_fn: Callable[..., Any] | None = None
    analyze_vacuity_fn: Callable[..., Any] | None = None
    propose_gap_remediation_fn: Callable[..., Any] | None = None
    evaluate_remediation_fn: Callable[..., Any] | None = None
    promote_assurance_policy_fn: Callable[..., Any] | None = None
    plan_mutation_campaign_fn: Callable[..., Any] | None = None
    execute_mutation_campaign_fn: Callable[..., Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        meta = self.metadata if isinstance(self.metadata, Mapping) else {}
        _reject_path_exposure(dict(meta), path="metadata")
        object.__setattr__(self, "metadata", MappingProxyType(dict(meta)))

    @property
    def interface_id(self) -> str:
        return ASSURANCE_CAMPAIGN_API_INTERFACE

    @property
    def evidence_id(self) -> str:
        return AAE_PUBLIC_API_EVIDENCE

    @property
    def required_public_apis(self) -> tuple[str, ...]:
        return REQUIRED_PUBLIC_APIS

    @property
    def required_commands(self) -> tuple[str, ...]:
        return REQUIRED_COMMANDS

    def descriptor(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "interface_id": ASSURANCE_CAMPAIGN_API_INTERFACE,
                "package_interface_id": ADVERSARIAL_ASSURANCE_PUBLIC_API_INTERFACE,
                "schema": ADVERSARIAL_ASSURANCE_PUBLIC_API_SCHEMA,
                "evidence_id": AAE_PUBLIC_API_EVIDENCE,
                "required_public_apis": list(REQUIRED_PUBLIC_APIS),
                "api_interface_ids": dict(_API_INTERFACE_IDS),
                "generator_id": GENERATOR_ID,
                "generator_version": GENERATOR_VERSION,
                "production_policy_change": False,
                "path_exposure": False,
            }
        )

    def _override_for(self, name: str) -> Callable[..., Any] | None:
        mapping = {
            "create_assurance_manifest": self.create_assurance_manifest_fn,
            "generate_mutation_candidates": self.generate_mutation_candidates_fn,
            "predict_detection_set": self.predict_detection_set_fn,
            "execute_mutation": self.execute_mutation_fn,
            "classify_mutation_outcome": self.classify_mutation_outcome_fn,
            "diagnose_surviving_mutant": self.diagnose_surviving_mutant_fn,
            "analyze_vacuity": self.analyze_vacuity_fn,
            "propose_gap_remediation": self.propose_gap_remediation_fn,
            "evaluate_remediation": self.evaluate_remediation_fn,
            "promote_assurance_policy": self.promote_assurance_policy_fn,
            "plan_mutation_campaign": self.plan_mutation_campaign_fn,
            "execute_mutation_campaign": self.execute_mutation_campaign_fn,
        }
        return mapping.get(name)

    def resolve(self, name: str) -> Callable[..., Any]:
        """Resolve one required API (injected override or lazy leaf)."""

        if name not in REQUIRED_PUBLIC_APIS:
            raise UnknownCommandError(name)
        override = self._override_for(name)
        if override is not None:
            if not callable(override):
                raise AssuranceApiUnavailableError(
                    name,
                    reason_code="not_callable",
                    diagnostic=f"injected {name!r} is not callable",
                    status=ApiAvailability.INCOMPATIBLE.value,
                )
            return override
        return resolve_public_api(name)

    def probe_api(self, name: str) -> Mapping[str, Any]:
        """Probe availability of one required API without invoking it."""

        if name not in REQUIRED_PUBLIC_APIS:
            raise UnknownCommandError(name)
        try:
            fn = self.resolve(name)
        except AssuranceApiUnavailableError as exc:
            return MappingProxyType(
                AssuranceApiUnavailableResult(
                    command=name,
                    status=exc.status,
                    reason_code=exc.reason_code,
                    diagnostic=exc.diagnostic,
                    api_interface_id=_API_INTERFACE_IDS[name],
                ).to_dict()
            )
        return MappingProxyType(
            {
                "command": name,
                "status": ApiAvailability.AVAILABLE.value,
                "available": True,
                "reason_code": None,
                "diagnostic": None,
                "interface_id": ASSURANCE_CAMPAIGN_API_INTERFACE,
                "evidence_id": AAE_PUBLIC_API_EVIDENCE,
                "api_interface_id": _API_INTERFACE_IDS[name],
                "callable": True,
                "module": getattr(fn, "__module__", None),
                "qualname": getattr(fn, "__qualname__", getattr(fn, "__name__", None)),
            }
        )

    # -- required API methods -----------------------------------------------

    def create_assurance_manifest(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("create_assurance_manifest", args, kwargs)

    def generate_mutation_candidates(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("generate_mutation_candidates", args, kwargs)

    def predict_detection_set(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("predict_detection_set", args, kwargs)

    def execute_mutation(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("execute_mutation", args, kwargs)

    def classify_mutation_outcome(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("classify_mutation_outcome", args, kwargs)

    def diagnose_surviving_mutant(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("diagnose_surviving_mutant", args, kwargs)

    def analyze_vacuity(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("analyze_vacuity", args, kwargs)

    def propose_gap_remediation(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("propose_gap_remediation", args, kwargs)

    def evaluate_remediation(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("evaluate_remediation", args, kwargs)

    def promote_assurance_policy(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("promote_assurance_policy", args, kwargs)

    def plan_mutation_campaign(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("plan_mutation_campaign", args, kwargs)

    def execute_mutation_campaign(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("execute_mutation_campaign", args, kwargs)

    def _call(
        self,
        name: str,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        # Path exposure gate on free-form kwargs for public safety.
        for key, value in kwargs.items():
            if isinstance(value, (Mapping, list, tuple, str)):
                try:
                    _reject_path_exposure(
                        value if not isinstance(value, str) else {key: value},
                        path=f"{name}.{key}",
                    )
                except PathExposureError:
                    raise
                except Exception:
                    # Non-structured values are left to the leaf implementation.
                    pass
        fn = self.resolve(name)
        _reject_unknown_params(fn, args, kwargs, context=name)
        return fn(*args, **dict(kwargs))

    # -- closed command dispatch --------------------------------------------

    def invoke(self, command: str, *args: Any, **kwargs: Any) -> Any:
        """Dispatch a closed command to the matching required public API."""

        if command not in REQUIRED_COMMANDS:
            raise UnknownCommandError(command)
        return self._call(command, args, kwargs)

    def invoke_envelope(self, payload: Mapping[str, Any]) -> Any:
        """Dispatch from a closed mapping envelope.

        Allowed fields: ``command`` (required), ``args``, ``kwargs`` /
        ``arguments`` (optional). Any other top-level field is rejected.
        """

        data = _closed_mapping(payload, _INVOKE_ENVELOPE_FIELDS, name="invoke_envelope")
        if "command" not in data:
            raise AssurancePublicApiError(
                "invoke_envelope requires 'command'",
                reason_code="missing_command",
            )
        command = data["command"]
        if not isinstance(command, str):
            raise AssurancePublicApiError(
                "command must be a string",
                reason_code="invalid_command",
            )
        if command not in REQUIRED_COMMANDS:
            raise UnknownCommandError(command)

        raw_args = data.get("args", ())
        if raw_args is None:
            raw_args = ()
        if not isinstance(raw_args, Sequence) or isinstance(raw_args, (str, bytes)):
            raise AssurancePublicApiError(
                "args must be a sequence",
                reason_code="invalid_args",
            )

        if "kwargs" in data and "arguments" in data:
            raise AssurancePublicApiError(
                "invoke_envelope accepts only one of 'kwargs' or 'arguments'",
                reason_code="conflicting_fields",
            )
        raw_kwargs = data.get("kwargs", data.get("arguments", {}))
        if raw_kwargs is None:
            raw_kwargs = {}
        if not isinstance(raw_kwargs, Mapping):
            raise AssurancePublicApiError(
                "kwargs/arguments must be a mapping",
                reason_code="invalid_kwargs",
            )
        return self.invoke(command, *tuple(raw_args), **dict(raw_kwargs))


def create_assurance_campaign_api(**dependencies: Any) -> AssuranceCampaignApi:
    """Construct an :class:`AssuranceCampaignApi` with optional DI.

    Unknown dependency field names are rejected (closed constructor surface).
    """

    allowed = frozenset(
        {
            "create_assurance_manifest_fn",
            "generate_mutation_candidates_fn",
            "predict_detection_set_fn",
            "execute_mutation_fn",
            "classify_mutation_outcome_fn",
            "diagnose_surviving_mutant_fn",
            "analyze_vacuity_fn",
            "propose_gap_remediation_fn",
            "evaluate_remediation_fn",
            "promote_assurance_policy_fn",
            "plan_mutation_campaign_fn",
            "execute_mutation_campaign_fn",
            "metadata",
        }
    )
    unknown = sorted(set(dependencies) - allowed)
    if unknown:
        raise UnknownFieldError(unknown, context="create_assurance_campaign_api")
    return AssuranceCampaignApi(**dependencies)


# ---------------------------------------------------------------------------
# Module-level invoke helpers (stateless default campaign API)
# ---------------------------------------------------------------------------


def invoke(command: str, *args: Any, **kwargs: Any) -> Any:
    """Dispatch a closed command through a default campaign API instance."""

    return AssuranceCampaignApi().invoke(command, *args, **kwargs)


def invoke_envelope(payload: Mapping[str, Any]) -> Any:
    """Dispatch a closed mapping envelope through a default campaign API."""

    return AssuranceCampaignApi().invoke_envelope(payload)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "AAE_PUBLIC_API_EVIDENCE",
    "ADVERSARIAL_ASSURANCE_PUBLIC_API_INTERFACE",
    "ADVERSARIAL_ASSURANCE_PUBLIC_API_SCHEMA",
    "ANALYZE_VACUITY_INTERFACE",
    "ASSURANCE_CAMPAIGN_API_INTERFACE",
    "ApiAvailability",
    "AssuranceApiUnavailableError",
    "AssuranceApiUnavailableResult",
    "AssuranceCampaignApi",
    "AssurancePublicApiError",
    "EXECUTE_MUTATION_CAMPAIGN_INTERFACE",
    "GENERATOR_ID",
    "GENERATOR_VERSION",
    "MUTATION_CAMPAIGN_EXECUTION_RESULT_INTERFACE",
    "MUTATION_CAMPAIGN_EXECUTION_RESULT_SCHEMA",
    "MutationCampaignExecutionResult",
    "PathExposureError",
    "REQUIRED_COMMANDS",
    "REQUIRED_PUBLIC_APIS",
    "UnknownCommandError",
    "UnknownFieldError",
    "VACUITY_CAMPAIGN_RESULT_INTERFACE",
    "VACUITY_CAMPAIGN_RESULT_SCHEMA",
    "VacuityCampaignAnalysisResult",
    "analyze_vacuity",
    "api_interface_id",
    "api_interface_ids",
    "campaign_api_interface_id",
    "classify_mutation_outcome",
    "create_assurance_campaign_api",
    "create_assurance_manifest",
    "diagnose_surviving_mutant",
    "evaluate_remediation",
    "execute_mutation",
    "execute_mutation_campaign",
    "generate_mutation_candidates",
    "invoke",
    "invoke_envelope",
    "plan_mutation_campaign",
    "predict_detection_set",
    "promote_assurance_policy",
    "propose_gap_remediation",
    "public_api_evidence_id",
    "public_api_interface_id",
    "public_api_schema",
    "required_commands",
    "required_public_apis",
    "resolve_public_api",
]
