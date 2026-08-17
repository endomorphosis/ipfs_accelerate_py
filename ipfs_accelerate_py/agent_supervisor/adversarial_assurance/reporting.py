"""Bounded assurance campaign report builders (AAE-058).

Interface surface:

* ``build_assurance_report`` — compose a sealed ``AssuranceReport@1`` from a
  campaign execution result (and optional plan / metrics inputs).
* Prefer explicit outcomes/gaps/remediations/economics when supplied; otherwise
  project metrics from candidate reports embedded in the campaign result.

Normative properties:

* Deterministic, content-addressed identity (``software_contracts.content``).
* Metrics populations remain the five disjoint AAE-058 families.
* Success **targets** are reported as goals only — never fabricated as results.
* No production policy change; cold import is side-effect free.
* Path-like absolute repo roots in report payloads are rejected.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_structured,
    validate_cid,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.common import (
    AssuranceBaseError,
)

from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.metrics import (
    AAE_METRICS_EVIDENCE,
    ASSURANCE_METRICS_INTERFACE,
    AssuranceMetrics,
    METRICS_POPULATION_KINDS,
    MetricsError,
    compute_assurance_metrics,
    verify_assurance_metrics_identity,
)

# ---------------------------------------------------------------------------
# Schema / interface / evidence
# ---------------------------------------------------------------------------

BUILD_ASSURANCE_REPORT_INTERFACE: Final[str] = "build_assurance_report@1"
ASSURANCE_REPORT_INTERFACE: Final[str] = "AssuranceReport@1"
ASSURANCE_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-report@1"
)
ASSURANCE_SUCCESS_TARGETS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "success-targets@1"
)

AAE_REPORT_EVIDENCE: Final[str] = "aae/metrics@1"
ADAPTER_ID: Final[str] = "aae-reporting"
BOARD_NAMESPACE: Final[str] = "adversarial-assurance-engine-v1"
GENERATOR_ID: Final[str] = "assurance_report"
GENERATOR_VERSION: Final[str] = "1.0.0"

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_CANDIDATES: Final[int] = 4_096
MAX_NOTES: Final[int] = 4_096
MAX_REASON_CODES: Final[int] = 128

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")

# Plan §15 success targets — goals, never fabricated results.
DEFAULT_SUCCESS_TARGETS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "schema": ASSURANCE_SUCCESS_TARGETS_SCHEMA,
        "targets_are_goals_not_results": True,
        "zero_controlled_critical_security_survivors_after_remediation": True,
        "zero_accepted_stale_proof_or_seal_integrity_mutants": True,
        "high_risk_semantic_detection_min_bp": 9_000,
        "explicit_gap_for_every_high_risk_survivor": True,
        "held_out_evaluation_for_every_promotion": True,
        "no_meaningful_claim_for_vacuous_proof": True,
        "compute_savings_min_bp": 5_000,
        "deterministic_campaign_ids": True,
        "no_worktree_escape": True,
        "no_unauthorized_production_policy_change": True,
    }
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ReportingError(AssuranceBaseError):
    """Raised when report inputs are malformed or unsafe."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "reporting_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        raise ReportingError(
            f"{name} must be a nonempty string",
            reason_code="invalid_type",
        )
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise ReportingError(
            f"{name} must be trimmed NFC text",
            reason_code="invalid_text",
        )
    if len(value) > MAX_TEXT_CHARS or any(not char.isprintable() for char in value):
        raise ReportingError(
            f"{name} contains invalid text",
            reason_code="invalid_text",
        )
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if value == "":
        return ""
    return _text(value, name)


def _token(value: Any, name: str) -> str:
    text = _text(value, name)
    if not _TOKEN_RE.match(text):
        raise ReportingError(
            f"{name} is not a valid token",
            reason_code="invalid_token",
        )
    return text


def _optional_token(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _token(value, name)


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        validate_cid(text)
    except Exception as exc:  # noqa: BLE001
        raise ReportingError(
            f"{name} is not a valid CID",
            reason_code="invalid_cid",
            details={"value": text},
        ) from exc
    return text


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ReportingError(f"{name} must be a bool", reason_code="invalid_type")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise ReportingError(
            f"{name} must be a non-negative int",
            reason_code="invalid_type",
        )
    return value


def _optional_nonneg_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _nonneg_int(value, name)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or isinstance(value, (str, bytes)):
        raise ReportingError(f"{name} must be a mapping", reason_code="invalid_type")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ReportingError(
            f"{name} must be a sequence",
            reason_code="invalid_type",
        )
    return value


def _reject_path_exposure(value: Any, *, path: str) -> None:
    """Fail closed on absolute filesystem / URI path leakage."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_s = str(key)
            lower = key_s.lower()
            if lower in {
                "repo_root",
                "repository_root",
                "absolute_path",
                "filesystem_path",
                "local_path",
            }:
                if isinstance(item, str) and item.startswith(("/", "file:", "\\")):
                    raise ReportingError(
                        f"absolute path exposure at {path}.{key_s}",
                        reason_code="path_exposure",
                        details={"path": f"{path}.{key_s}", "value": item},
                    )
            _reject_path_exposure(item, path=f"{path}.{key_s}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_path_exposure(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        # Bare absolute POSIX paths that look like host paths.
        if value.startswith("/") and len(value) > 1:
            try:
                pure = PurePosixPath(value)
            except Exception:  # noqa: BLE001
                return
            if pure.is_absolute() and any(
                part in {"home", "var", "tmp", "Users", "etc"}
                for part in pure.parts
            ):
                raise ReportingError(
                    f"absolute path exposure at {path}",
                    reason_code="path_exposure",
                    details={"path": path, "value": value},
                )


# ---------------------------------------------------------------------------
# Report model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AssuranceReport:
    """Sealed campaign report binding optional disjoint metrics.

    Interface: ``AssuranceReport@1``
    """

    interface_id: str
    plan_id: str | None
    plan_cid: str | None
    result_cid: str | None
    repository_state_cid: str | None
    verification_policy_cid: str | None
    terminal_status: str | None
    killed_count: int | None
    survivor_count: int | None
    invalid_count: int | None
    inconclusive_count: int | None
    candidate_report_count: int
    candidate_reports: tuple[Mapping[str, Any], ...]
    reason_codes: tuple[str, ...]
    summary: str
    notes: str | None
    metrics_available: bool
    metrics: Mapping[str, Any] | None
    metrics_cid: str | None
    success_targets: Mapping[str, Any]
    require_sandbox: bool
    network_disabled: bool
    production_policy_changed: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface_id",
            "plan_id",
            "plan_cid",
            "result_cid",
            "repository_state_cid",
            "verification_policy_cid",
            "terminal_status",
            "killed_count",
            "survivor_count",
            "invalid_count",
            "inconclusive_count",
            "candidate_report_count",
            "candidate_reports",
            "reason_codes",
            "summary",
            "notes",
            "metrics_available",
            "metrics",
            "metrics_cid",
            "success_targets",
            "require_sandbox",
            "network_disabled",
            "production_policy_changed",
            "metadata",
            "report_cid",
            "evidence",
            "build_interface",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "interface_id", _text(self.interface_id, "interface_id")
        )
        if self.interface_id != ASSURANCE_REPORT_INTERFACE:
            raise ReportingError(
                "interface_id must be AssuranceReport@1",
                reason_code="invalid_interface",
            )
        object.__setattr__(self, "plan_id", _optional_token(self.plan_id, "plan_id"))
        object.__setattr__(self, "plan_cid", _optional_cid(self.plan_cid, "plan_cid"))
        object.__setattr__(
            self, "result_cid", _optional_cid(self.result_cid, "result_cid")
        )
        object.__setattr__(
            self,
            "repository_state_cid",
            _optional_cid(self.repository_state_cid, "repository_state_cid"),
        )
        object.__setattr__(
            self,
            "verification_policy_cid",
            _optional_cid(self.verification_policy_cid, "verification_policy_cid"),
        )
        object.__setattr__(
            self,
            "terminal_status",
            None
            if self.terminal_status is None
            else _token(self.terminal_status, "terminal_status"),
        )
        for name in (
            "killed_count",
            "survivor_count",
            "invalid_count",
            "inconclusive_count",
        ):
            object.__setattr__(
                self, name, _optional_nonneg_int(getattr(self, name), name)
            )
        count = _nonneg_int(self.candidate_report_count, "candidate_report_count")
        reports = tuple(
            MappingProxyType(dict(_mapping(item, "candidate_reports")))
            for item in self.candidate_reports
        )
        if len(reports) > MAX_CANDIDATES:
            raise ReportingError(
                "candidate_reports exceed maximum",
                reason_code="bounds",
            )
        if count != len(reports):
            raise ReportingError(
                "candidate_report_count must match candidate_reports length",
                reason_code="count_mismatch",
            )
        object.__setattr__(self, "candidate_report_count", count)
        object.__setattr__(self, "candidate_reports", reports)

        codes = tuple(
            _token(item, "reason_codes") for item in (self.reason_codes or ())
        )
        if len(codes) > MAX_REASON_CODES:
            raise ReportingError("too many reason_codes", reason_code="bounds")
        object.__setattr__(self, "reason_codes", codes)
        object.__setattr__(self, "summary", _text(self.summary, "summary", empty=True))
        notes = _optional_text(self.notes, "notes")
        if notes is not None and len(notes) > MAX_NOTES:
            raise ReportingError("notes exceed maximum", reason_code="bounds")
        object.__setattr__(self, "notes", notes)

        available = _bool(self.metrics_available, "metrics_available")
        object.__setattr__(self, "metrics_available", available)
        if available:
            if not isinstance(self.metrics, Mapping):
                raise ReportingError(
                    "metrics_available requires metrics mapping",
                    reason_code="metrics_required",
                )
            metrics_map = dict(self.metrics)
            object.__setattr__(self, "metrics", MappingProxyType(metrics_map))
            object.__setattr__(
                self, "metrics_cid", _cid(self.metrics_cid, "metrics_cid")
            )
        else:
            object.__setattr__(self, "metrics", None)
            object.__setattr__(self, "metrics_cid", None)

        targets = dict(_mapping(self.success_targets, "success_targets"))
        if not targets.get("targets_are_goals_not_results", False):
            raise ReportingError(
                "success_targets must declare targets_are_goals_not_results",
                reason_code="targets_not_goals",
            )
        object.__setattr__(self, "success_targets", MappingProxyType(targets))

        sandbox = _bool(self.require_sandbox, "require_sandbox")
        network = _bool(self.network_disabled, "network_disabled")
        changed = _bool(self.production_policy_changed, "production_policy_changed")
        if not sandbox:
            raise ReportingError(
                "require_sandbox must be true",
                reason_code="sandbox_required",
            )
        if not network:
            raise ReportingError(
                "network_disabled must be true",
                reason_code="network_required_disabled",
            )
        if changed:
            raise ReportingError(
                "report must not claim production policy change",
                reason_code="production_policy_change",
            )
        object.__setattr__(self, "require_sandbox", True)
        object.__setattr__(self, "network_disabled", True)
        object.__setattr__(self, "production_policy_changed", False)
        meta = self.metadata if isinstance(self.metadata, Mapping) else {}
        object.__setattr__(self, "metadata", MappingProxyType(dict(meta)))

        # Path exposure check on projected payload fields.
        _reject_path_exposure(self.identity_payload(), path="assurance_report")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": ASSURANCE_REPORT_SCHEMA,
            "interface_id": self.interface_id,
            "build_interface": BUILD_ASSURANCE_REPORT_INTERFACE,
            "plan_id": self.plan_id,
            "plan_cid": self.plan_cid,
            "result_cid": self.result_cid,
            "repository_state_cid": self.repository_state_cid,
            "verification_policy_cid": self.verification_policy_cid,
            "terminal_status": self.terminal_status,
            "killed_count": self.killed_count,
            "survivor_count": self.survivor_count,
            "invalid_count": self.invalid_count,
            "inconclusive_count": self.inconclusive_count,
            "candidate_report_count": self.candidate_report_count,
            "candidate_reports": [dict(item) for item in self.candidate_reports],
            "reason_codes": list(self.reason_codes),
            "summary": self.summary,
            "notes": self.notes,
            "metrics_available": self.metrics_available,
            "metrics": None if self.metrics is None else dict(self.metrics),
            "metrics_cid": self.metrics_cid,
            "success_targets": dict(self.success_targets),
            "require_sandbox": True,
            "network_disabled": True,
            "production_policy_changed": False,
            "metadata": dict(self.metadata),
            "evidence": AAE_REPORT_EVIDENCE,
        }

    @property
    def report_cid(self) -> str:
        return cid_for_structured(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["report_cid"] = self.report_cid
        return payload


def verify_assurance_report_identity(
    report: AssuranceReport | Mapping[str, Any],
) -> str:
    """Recompute and return report_cid; raise on forged input."""

    declared_cid: str | None = None
    if isinstance(report, AssuranceReport):
        sealed = report
        declared_cid = report.report_cid
    elif isinstance(report, Mapping):
        raw = dict(report)
        declared = raw.get("report_cid")
        if declared is not None:
            declared_cid = _cid(declared, "report_cid")
        sealed = assurance_report_from_dict(raw)
    else:
        raise ReportingError(
            "report must be AssuranceReport or mapping",
            reason_code="invalid_type",
        )
    recomputed = cid_for_structured(sealed.identity_payload())
    if declared_cid is not None and recomputed != declared_cid:
        raise ReportingError(
            "report_cid identity mismatch with recomputed identity",
            reason_code="identity_mismatch",
            details={"declared": declared_cid, "recomputed": recomputed},
        )
    if recomputed != sealed.report_cid:
        raise ReportingError(
            "report_cid identity mismatch with recomputed identity",
            reason_code="identity_mismatch",
        )
    return recomputed


def assurance_report_from_dict(data: Mapping[str, Any]) -> AssuranceReport:
    payload = dict(_mapping(data, "report"))
    payload.pop("report_cid", None)
    payload.pop("evidence", None)
    payload.pop("build_interface", None)
    schema = payload.pop("schema", ASSURANCE_REPORT_SCHEMA)
    if schema != ASSURANCE_REPORT_SCHEMA:
        raise ReportingError(
            f"unexpected report schema {schema!r}",
            reason_code="invalid_schema",
        )
    reports = tuple(payload.get("candidate_reports") or ())
    return AssuranceReport(
        interface_id=payload.get("interface_id", ASSURANCE_REPORT_INTERFACE),
        plan_id=payload.get("plan_id"),
        plan_cid=payload.get("plan_cid"),
        result_cid=payload.get("result_cid"),
        repository_state_cid=payload.get("repository_state_cid"),
        verification_policy_cid=payload.get("verification_policy_cid"),
        terminal_status=payload.get("terminal_status"),
        killed_count=payload.get("killed_count"),
        survivor_count=payload.get("survivor_count"),
        invalid_count=payload.get("invalid_count"),
        inconclusive_count=payload.get("inconclusive_count"),
        candidate_report_count=int(
            payload.get("candidate_report_count", len(reports))
        ),
        candidate_reports=reports,
        reason_codes=tuple(payload.get("reason_codes") or ()),
        summary=str(payload.get("summary") or ""),
        notes=payload.get("notes"),
        metrics_available=bool(payload.get("metrics_available", False)),
        metrics=payload.get("metrics"),
        metrics_cid=payload.get("metrics_cid"),
        success_targets=payload.get("success_targets") or dict(DEFAULT_SUCCESS_TARGETS),
        require_sandbox=bool(payload.get("require_sandbox", True)),
        network_disabled=bool(payload.get("network_disabled", True)),
        production_policy_changed=bool(
            payload.get("production_policy_changed", False)
        ),
        metadata=payload.get("metadata") or {},
    )


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------


def _project_candidate_report(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": item.get("candidate_id"),
        "candidate_cid": item.get("candidate_cid") or item.get("mutant_identity_cid"),
        "terminal_status": (
            item.get("terminal_status")
            or item.get("outcome_status")
            or item.get("disposition")
        ),
        "outcome_cid": item.get("outcome_cid"),
        "report_cid": item.get("report_cid") or item.get("result_cid"),
    }


def _campaign_id_of(
    campaign_result: Mapping[str, Any],
    plan: Mapping[str, Any] | None,
) -> str:
    raw = (
        campaign_result.get("campaign_id")
        or campaign_result.get("plan_id")
        or (plan or {}).get("plan_id")
        or (plan or {}).get("campaign_id")
        or "campaign"
    )
    return _token(str(raw), "campaign_id")


def _extract_outcomes(
    campaign_result: Mapping[str, Any],
    outcomes: Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    if outcomes is not None:
        return [dict(_mapping(item, "outcomes")) for item in _sequence(outcomes, "outcomes")]
    # Prefer explicit outcome lists when present.
    for key in ("outcomes", "mutation_outcomes", "classified_outcomes"):
        raw = campaign_result.get(key)
        if raw is not None:
            return [
                dict(_mapping(item, f"{key}[{index}]"))
                for index, item in enumerate(_sequence(raw, key))
            ]
    reports = campaign_result.get("candidate_reports") or ()
    projected: list[Mapping[str, Any]] = []
    for index, item in enumerate(_sequence(reports, "candidate_reports")):
        if not isinstance(item, Mapping):
            continue
        entry = dict(item)
        # Promote coarse terminal_status into outcome_status for metrics.
        if "outcome_status" not in entry and "terminal_status" in entry:
            entry["outcome_status"] = entry["terminal_status"]
        if "candidate_id" not in entry:
            entry["candidate_id"] = f"candidate_{index}"
        projected.append(entry)
    return projected


def _extract_optional_sequence(
    campaign_result: Mapping[str, Any],
    explicit: Sequence[Mapping[str, Any]] | None,
    *keys: str,
) -> list[Mapping[str, Any]]:
    if explicit is not None:
        return [
            dict(_mapping(item, "explicit"))
            for item in _sequence(explicit, "explicit")
        ]
    for key in keys:
        raw = campaign_result.get(key)
        if raw is not None:
            return [
                dict(_mapping(item, f"{key}[{index}]"))
                for index, item in enumerate(_sequence(raw, key))
            ]
    return []


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_assurance_report(
    campaign_result: Mapping[str, Any] | Any,
    *,
    plan: Mapping[str, Any] | None = None,
    notes: str | None = None,
    outcomes: Sequence[Mapping[str, Any]] | None = None,
    gaps: Sequence[Mapping[str, Any]] | None = None,
    remediations: Sequence[Mapping[str, Any]] | None = None,
    economics_records: Sequence[Mapping[str, Any]] | None = None,
    metrics: AssuranceMetrics | Mapping[str, Any] | None = None,
    success_targets: Mapping[str, Any] | None = None,
    generated_count: int | None = None,
    admitted_count: int | None = None,
    include_metrics: bool = True,
) -> AssuranceReport:
    """Build a sealed assurance report, optionally binding AAE-058 metrics.

    Compatible with the campaign CLI ``assurance report`` entry point:
    ``build_assurance_report(campaign_result, plan=plan, notes=notes)``.
    """

    if hasattr(campaign_result, "to_dict") and callable(campaign_result.to_dict):
        campaign_result = campaign_result.to_dict()
    campaign = dict(_mapping(campaign_result, "campaign_result"))
    plan_map = None if plan is None else dict(_mapping(plan, "plan"))

    _reject_path_exposure(campaign, path="campaign_result")
    if plan_map is not None:
        _reject_path_exposure(plan_map, path="plan")

    reports_raw = campaign.get("candidate_reports") or ()
    if not isinstance(reports_raw, Sequence) or isinstance(reports_raw, (str, bytes)):
        reports_raw = ()
    if len(reports_raw) > MAX_CANDIDATES:
        raise ReportingError(
            "candidate_reports exceed maximum",
            reason_code="bounds",
            details={"count": len(reports_raw), "max": MAX_CANDIDATES},
        )
    projected_reports = tuple(
        _project_candidate_report(item)
        for item in reports_raw
        if isinstance(item, Mapping)
    )

    plan_id = campaign.get("plan_id")
    plan_cid = campaign.get("plan_cid")
    if plan_map is not None:
        plan_id = plan_id or plan_map.get("plan_id")
        plan_cid = plan_cid or plan_map.get("plan_cid")

    killed = campaign.get("killed_count")
    survivor = campaign.get("survivor_count")
    invalid = campaign.get("invalid_count")
    inconclusive = campaign.get("inconclusive_count")
    terminal = campaign.get("terminal_status")

    metrics_payload: dict[str, Any] | None = None
    metrics_cid: str | None = None
    metrics_available = False
    reason_codes: list[str] = ["report_built", "no_production_policy_change"]

    if include_metrics:
        try:
            if metrics is None:
                sealed = compute_assurance_metrics(
                    campaign_id=_campaign_id_of(campaign, plan_map),
                    outcomes=_extract_outcomes(campaign, outcomes),
                    gaps=_extract_optional_sequence(
                        campaign, gaps, "gaps", "assurance_gaps"
                    ),
                    remediations=_extract_optional_sequence(
                        campaign,
                        remediations,
                        "remediations",
                        "remediation_candidates",
                    ),
                    economics_records=_extract_optional_sequence(
                        campaign,
                        economics_records,
                        "economics_records",
                        "cost_records",
                        "mutation_costs",
                    ),
                    plan_id=None if plan_id is None else str(plan_id),
                    plan_cid=None if plan_cid is None else str(plan_cid),
                    result_cid=(
                        None
                        if campaign.get("result_cid") is None
                        else str(campaign.get("result_cid"))
                    ),
                    repository_state_cid=(
                        None
                        if campaign.get("repository_state_cid") is None
                        else str(campaign.get("repository_state_cid"))
                    ),
                    generated_count=generated_count,
                    admitted_count=admitted_count,
                    notes=notes,
                )
            elif isinstance(metrics, AssuranceMetrics):
                sealed = metrics
            else:
                from ipfs_accelerate_py.agent_supervisor.adversarial_assurance.metrics import (
                    assurance_metrics_from_dict,
                )

                sealed = assurance_metrics_from_dict(metrics)

            metrics_cid = verify_assurance_metrics_identity(sealed)
            metrics_payload = sealed.to_dict()
            metrics_available = True
            reason_codes.extend(
                [
                    "metrics_bound",
                    "populations_disjoint",
                    "denominators_exclude_invalid_equivalent_infrastructure",
                ]
            )
            # Prefer metrics-derived counts when campaign omitted them.
            cov = sealed.mutation_coverage
            if killed is None:
                killed = cov.killed_count
            if survivor is None:
                survivor = cov.selected_survivor_count + cov.full_survivor_count
            if invalid is None:
                invalid = cov.invalid_count
            if inconclusive is None:
                inconclusive = cov.inconclusive_count
        except (MetricsError, ReportingError) as exc:
            reason_codes.append("metrics_unavailable")
            reason_codes.append(getattr(exc, "reason_code", "metrics_error"))
            metrics_available = False
            metrics_payload = None
            metrics_cid = None

    summary_parts = [
        f"terminal={terminal}",
        f"killed={killed}",
        f"survivor={survivor}",
        f"invalid={invalid}",
        f"inconclusive={inconclusive}",
        f"metrics_available={metrics_available}",
    ]
    summary_text = " ".join(str(part) for part in summary_parts if part is not None)

    targets = dict(DEFAULT_SUCCESS_TARGETS)
    if success_targets is not None:
        extra = dict(_mapping(success_targets, "success_targets"))
        targets.update(extra)
        targets["targets_are_goals_not_results"] = True

    # Merge campaign reason codes without claiming authority they lack.
    for code in campaign.get("reason_codes") or ():
        if isinstance(code, str) and code and code not in reason_codes:
            reason_codes.append(_token(code, "reason_codes"))

    report = AssuranceReport(
        interface_id=ASSURANCE_REPORT_INTERFACE,
        plan_id=None if plan_id is None else str(plan_id),
        plan_cid=None if plan_cid is None else str(plan_cid),
        result_cid=(
            None
            if campaign.get("result_cid") is None
            else str(campaign.get("result_cid"))
        ),
        repository_state_cid=(
            None
            if campaign.get("repository_state_cid") is None
            else str(campaign.get("repository_state_cid"))
        ),
        verification_policy_cid=(
            None
            if campaign.get("verification_policy_cid") is None
            else str(campaign.get("verification_policy_cid"))
        ),
        terminal_status=None if terminal is None else str(terminal),
        killed_count=None if killed is None else int(killed),
        survivor_count=None if survivor is None else int(survivor),
        invalid_count=None if invalid is None else int(invalid),
        inconclusive_count=None if inconclusive is None else int(inconclusive),
        candidate_report_count=len(projected_reports),
        candidate_reports=projected_reports,
        reason_codes=tuple(reason_codes),
        summary=summary_text,
        notes=notes,
        metrics_available=metrics_available,
        metrics=metrics_payload,
        metrics_cid=metrics_cid,
        success_targets=targets,
        require_sandbox=bool(campaign.get("require_sandbox", True)),
        network_disabled=bool(campaign.get("network_disabled", True)),
        production_policy_changed=False,
        metadata={
            "metrics_interface": ASSURANCE_METRICS_INTERFACE,
            "metrics_evidence": AAE_METRICS_EVIDENCE,
            "population_kinds": list(METRICS_POPULATION_KINDS),
        },
    )
    return report


def reporting_descriptor() -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "interface": ASSURANCE_REPORT_INTERFACE,
            "build_interface": BUILD_ASSURANCE_REPORT_INTERFACE,
            "schema": ASSURANCE_REPORT_SCHEMA,
            "evidence": AAE_REPORT_EVIDENCE,
            "adapter_id": ADAPTER_ID,
            "generator_id": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "api": "build_assurance_report",
            "metrics_interface": ASSURANCE_METRICS_INTERFACE,
            "production_policy_change": False,
            "targets_are_goals_not_results": True,
        }
    )


__all__ = [
    "AAE_REPORT_EVIDENCE",
    "ADAPTER_ID",
    "ASSURANCE_REPORT_INTERFACE",
    "ASSURANCE_REPORT_SCHEMA",
    "ASSURANCE_SUCCESS_TARGETS_SCHEMA",
    "BOARD_NAMESPACE",
    "BUILD_ASSURANCE_REPORT_INTERFACE",
    "DEFAULT_SUCCESS_TARGETS",
    "GENERATOR_ID",
    "GENERATOR_VERSION",
    "AssuranceReport",
    "ReportingError",
    "assurance_report_from_dict",
    "build_assurance_report",
    "reporting_descriptor",
    "verify_assurance_report_identity",
]
