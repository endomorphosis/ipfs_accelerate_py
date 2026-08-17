"""Bounded, resumable GUI improvement loop (VGO-053).

Interfaces owned by this module:

* ``VerifiedGuiOptimizer@1`` — journaled orchestrator for one explicit
  objective (or a few) through the closed improvement phases
* ``GuiImprovementRun@1`` — resumable run record keyed by a stable run ID
* ``GuiImprovementDecision@1`` — accept, reject, or human-review outcome

The loop never chooses a model vendor, never merges into the canonical
branch, and never treats process exit as completion.  Acceptance requires
a measurable target-metric improvement plus every hard gate.  Missing
evidence rejects or reviews.  Every terminal decision writes a receipt.

Fail-closed invariants:

* one or a few explicit objectives per bounded iteration;
* no whole-app aesthetic rewrite;
* no automatic canonical merge;
* opaque, ambiguous, policy-bound, or repeatedly failed proposals
  escalate to human review without fabricating a patch;
* accessibility, security, and confirmation regressions block accept;
* interrupted runs resume from the journal after identity revalidation.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_datasets_py.logic.gui_optimizer.schema import (
    GUI_IMPROVEMENT_RECEIPT_INTERFACE,
    GUI_IMPROVEMENT_RECEIPT_SCHEMA,
    ProposalDecision,
)

from .authority import AuthorityReasonCode, GuiAuthorityError
from .check_plan import GuiAffectedCheckPlanner, default_affected_check_planner
from .patch_scope import DEFAULT_MAX_FILES
from .proposal import (
    GuiPatchProposer,
    GuiProposalError,
    GuiProposalResult,
    ProposalDisposition,
    ProposalReasonCode,
    default_gui_patch_proposer,
)
from .run_journal import (
    GuiRunCheckpoint,
    GuiRunJournal,
    JournalPhase,
    PhaseRecordStatus,
    ResumeAction,
    RunStatus,
    default_run_journal,
)
from .worktree_executor import GuiIsolatedWorktreeExecutor

# ---------------------------------------------------------------------------
# Interface / schema identity
# ---------------------------------------------------------------------------

VERIFIED_GUI_OPTIMIZER_INTERFACE: Final[str] = "VerifiedGuiOptimizer@1"
GUI_IMPROVEMENT_RUN_INTERFACE: Final[str] = "GuiImprovementRun@1"
GUI_IMPROVEMENT_DECISION_INTERFACE: Final[str] = "GuiImprovementDecision@1"

VERIFIED_GUI_OPTIMIZER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/verified-optimizer@1"
)
GUI_IMPROVEMENT_RUN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/improvement-run@1"
)
GUI_IMPROVEMENT_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/gui-optimizer/improvement-decision@1"
)

DEFAULT_MAX_ATTEMPTS: Final[int] = 3
DEFAULT_MAX_OBJECTIVES: Final[int] = 2
DEFAULT_OBJECTIVE_METRIC: Final[str] = "accessible_name_coverage"

PHASE_ORDER: Final[tuple[JournalPhase, ...]] = (
    JournalPhase.BASELINE,
    JournalPhase.SELECT_OBJECTIVE,
    JournalPhase.IMPACT,
    JournalPhase.CONTEXT_PACK,
    JournalPhase.PROPOSAL,
    JournalPhase.ISOLATED_WORKTREE,
    JournalPhase.RESCAN,
    JournalPhase.INVALIDATION,
    JournalPhase.AFFECTED_CHECKS,
    JournalPhase.FALLBACK,
    JournalPhase.COMPARE,
    JournalPhase.DECISION,
    JournalPhase.RECEIPT,
)

HIGHER_IS_BETTER: Final[frozenset[str]] = frozenset(
    {
        "accessible_name_coverage",
        "keyboard_access_coverage",
        "objective_score",
    }
)
LOWER_IS_BETTER: Final[frozenset[str]] = frozenset(
    {
        "confirmation_bypass_count",
        "critical_accessibility_violations",
        "interaction_step_count",
        "pixel_diff_percent",
        "security_violations",
    }
)
HARD_GATE_METRICS: Final[frozenset[str]] = frozenset(
    {
        "confirmation_bypass_count",
        "critical_accessibility_violations",
        "security_violations",
    }
)
_REWRITE_RE: Final = re.compile(
    r"(?i)\b(whole[- ]app|entire application|redesign the (?:app|application)|"
    r"aesthetic rewrite|rewrite everything)\b"
)
_IDENTIFIER_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/#@-]{0,255}$")
_FULL_SHA_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")

_REQUEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "acceptance_criteria",
        "affected_screenshot_ids",
        "ambiguous",
        "analysis_classification",
        "application",
        "application_id",
        "attempt",
        "baseline",
        "baseline_metrics",
        "candidate_metrics",
        "canonical_branch",
        "canonical_porcelain",
        "canonical_revision",
        "change_kinds",
        "check_execution",
        "constraint_conflict",
        "context_pack",
        "declared_method",
        "declared_tier",
        "escalation_conditions",
        "evaluator_risk",
        "evidence",
        "expected_screenshot_ids",
        "expected_test_ids",
        "halt_after_phase",
        "hard_gates",
        "heuristic_scores",
        "impact",
        "intended_component_ids",
        "intended_file_paths",
        "invalidation",
        "known_screenshot_ids",
        "max_attempts",
        "objective",
        "objective_id",
        "objective_ids",
        "objective_metric_id",
        "opaque",
        "pixel_change_only",
        "policy_bound",
        "prior_failure_count",
        "process_alive",
        "proposal_id",
        "proposal_request",
        "request_id",
        "resume",
        "route_kind",
        "run_id",
        "screen_id",
        "security_sensitive",
        "source_revision",
        "state_effect_ids",
        "transformations",
        "unrelated_screenshot_ids",
        "verification_status",
        "visual_effect_summary",
    }
)
_FORBIDDEN_PATH_KEYS: Final[frozenset[str]] = frozenset(
    {
        "browser_input",
        "command",
        "commands",
        "cwd",
        "file_path",
        "filesystem_path",
        "host_path",
        "path",
        "selected_host_paths",
        "working_directory",
    }
)


class ImprovementDecisionKind(str, Enum):
    """Closed ``GuiImprovementDecision@1`` outcomes."""

    ACCEPT = "accept"
    REJECT = "reject"
    HUMAN_REVIEW = "human_review"
    PENDING = "pending"


class ImprovementReasonCode(str, Enum):
    """Stable reason codes for loop decisions."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    MEASURABLE_IMPROVEMENT = "measurable_improvement"
    NO_MEASURABLE_IMPROVEMENT = "no_measurable_improvement"
    HARD_GATE_REGRESSION = "hard_gate_regression"
    ACCESSIBILITY_REGRESSION = "accessibility_regression"
    SECURITY_REGRESSION = "security_regression"
    CONFIRMATION_REGRESSION = "confirmation_regression"
    POLICY_REGRESSION = "policy_regression"
    FUNCTIONAL_REGRESSION = "functional_regression"
    PIXEL_CHANGE_ONLY = "pixel_change_only"
    MISSING_EVIDENCE = "missing_evidence"
    UNKNOWN_CRITICAL_EVIDENCE = "unknown_critical_evidence"
    REQUIRED_CHECK_FAILED = "required_check_failed"
    PROPOSAL_ESCALATED = "proposal_escalated"
    PROPOSAL_REJECTED = "proposal_rejected"
    ATTEMPT_BUDGET_EXHAUSTED = "attempt_budget_exhausted"
    TOO_MANY_OBJECTIVES = "too_many_objectives"
    WHOLE_APP_REWRITE = "whole_app_rewrite"
    CANONICAL_MERGE_FORBIDDEN = "canonical_merge_forbidden"
    ISOLATED_APPLY_MISSING = "isolated_apply_missing"
    ISOLATED_APPLY_REJECTED = "isolated_apply_rejected"
    HEURISTIC_CANNOT_OVERRIDE = "heuristic_cannot_override"
    RESUMED = "resumed"
    INTERRUPTED = "interrupted"
    PHASE_RECORDED = "phase_recorded"
    INVALID_IMPROVEMENT_INPUT = "invalid_improvement_input"
    UNKNOWN_FIELD = AuthorityReasonCode.UNKNOWN_FIELD.value
    INVALID_COLLECTION_TYPE = AuthorityReasonCode.INVALID_COLLECTION_TYPE.value


class GuiImprovementLoopError(GuiAuthorityError):
    """Malformed loop input.  Never grants acceptance."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = ImprovementReasonCode.INVALID_IMPROVEMENT_INPUT.value,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, reason_code=reason_code, details=details)


# ---------------------------------------------------------------------------
# Closed input helpers
# ---------------------------------------------------------------------------


def _exact_str(value: Any, name: str) -> str:
    if type(value) is not str:
        raise GuiImprovementLoopError(
            f"{name} must be a string",
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text_value = _exact_str(value, name)
    if "\x00" in text_value:
        raise GuiImprovementLoopError(
            f"{name} must not contain NUL",
            details={"field": name},
        )
    text = text_value.strip()
    if required and not text:
        raise GuiImprovementLoopError(
            f"{name} must not be empty",
            details={"field": name},
        )
    return text


def _identifier(value: Any, name: str) -> str:
    text_value = _exact_str(value, name)
    if "\x00" in text_value:
        raise GuiImprovementLoopError(
            f"{name} must not contain NUL",
            details={"field": name},
        )
    if text_value == "" or text_value != text_value.strip():
        raise GuiImprovementLoopError(
            f"{name} must be a canonical nonempty string identifier",
            details={"field": name},
        )
    if not _IDENTIFIER_RE.fullmatch(text_value):
        raise GuiImprovementLoopError(
            f"{name} is not a stable identifier",
            details={"field": name},
        )
    return text_value


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise GuiImprovementLoopError(
            f"{name} must be a boolean",
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or type(value) is bool:
        raise GuiImprovementLoopError(
            f"{name} must be an integer",
            details={"field": name, "value_type": type(value).__name__},
        )
    if value < 1:
        raise GuiImprovementLoopError(
            f"{name} must be a positive integer",
            details={"field": name, "value": value},
        )
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or type(value) is bool:
        raise GuiImprovementLoopError(
            f"{name} must be an integer",
            details={"field": name, "value_type": type(value).__name__},
        )
    if value < 0:
        raise GuiImprovementLoopError(
            f"{name} must be a non-negative integer",
            details={"field": name, "value": value},
        )
    return value


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise GuiImprovementLoopError(
            f"{name} must be a JSON object",
            reason_code=ImprovementReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    for key in value:
        if type(key) is not str:
            raise GuiImprovementLoopError(
                f"{name} keys must be strings",
                reason_code=ImprovementReasonCode.INVALID_COLLECTION_TYPE.value,
                details={"field": name, "key_type": type(key).__name__},
            )
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if type(value) is not list:
        raise GuiImprovementLoopError(
            f"{name} must be a JSON array",
            reason_code=ImprovementReasonCode.INVALID_COLLECTION_TYPE.value,
            details={"field": name, "value_type": type(value).__name__},
        )
    return value


def _reject_unknown(
    payload: Mapping[str, Any], allowed: frozenset[str], noun: str
) -> None:
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise GuiImprovementLoopError(
            f"{noun} contains unknown fields: {unknown}",
            reason_code=ImprovementReasonCode.UNKNOWN_FIELD.value,
            details={"noun": noun, "unknown_fields": unknown},
        )


def _reject_present_null(payload: Mapping[str, Any], key: str) -> None:
    if key in payload and payload[key] is None:
        raise GuiImprovementLoopError(
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
    text = _exact_str(payload[key], key)
    if text == "":
        return ""
    return _identifier(text, key)


def _optional_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    if key not in payload:
        return {}
    _reject_present_null(payload, key)
    return dict(_require_mapping(payload[key], key))


def _optional_string_list(
    payload: Mapping[str, Any],
    key: str,
    *,
    as_text: bool = False,
) -> tuple[str, ...]:
    if key not in payload:
        return ()
    _reject_present_null(payload, key)
    items = _require_list(payload[key], key)
    out: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        token = (
            _text(item, f"{key}[{index}]")
            if as_text
            else _identifier(item, f"{key}[{index}]")
        )
        if token not in seen:
            seen.add(token)
            out.append(token)
    return tuple(out)


def _require_revision(value: Any, name: str) -> str:
    text = _identifier(value, name)
    if not (_FULL_SHA_RE.fullmatch(text) or _DIGEST_RE.fullmatch(text)):
        raise GuiImprovementLoopError(
            f"{name} must be a 40-character SHA-1 or sha256 digest",
            details={"field": name},
        )
    return text


def _as_phase(value: Any) -> JournalPhase:
    if type(value) is JournalPhase:
        return value
    text = _text(value, "phase")
    try:
        return JournalPhase(text)
    except ValueError as exc:
        raise GuiImprovementLoopError(
            f"unknown journal phase: {text}",
            details={"phase": text},
        ) from exc


def _as_kind(value: Any) -> ImprovementDecisionKind:
    if type(value) is ImprovementDecisionKind:
        return value
    text = _text(value, "decision")
    try:
        return ImprovementDecisionKind(text)
    except ValueError as exc:
        raise GuiImprovementLoopError(
            f"unknown improvement decision: {text}",
            details={"decision": text},
        ) from exc


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    return MappingProxyType(dict(_require_mapping(value, "details")))


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


def _sha256_digest(payload: Mapping[str, Any] | str | bytes) -> str:
    if type(payload) is bytes:
        raw = payload
    elif type(payload) is str:
        raw = payload.encode("utf-8")
    else:
        raw = _canonical_bytes(payload)
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _sanitize_payload(value: Any) -> Any:
    if type(value) is dict:
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or key in _FORBIDDEN_PATH_KEYS:
                continue
            cleaned[key] = _sanitize_payload(item)
        return cleaned
    if type(value) is list:
        return [_sanitize_payload(item) for item in value]
    return value


def _metric_improved(metric_id: str, before: float, after: float) -> bool:
    if metric_id in LOWER_IS_BETTER:
        return after < before
    return after > before


def _metric_regressed(metric_id: str, before: float, after: float) -> bool:
    if metric_id in LOWER_IS_BETTER:
        return after > before
    if metric_id in HIGHER_IS_BETTER:
        return after < before
    return False


# ---------------------------------------------------------------------------
# Typed records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuiImprovementDecision:
    """Typed accept / reject / human-review outcome.

    Interface: ``GuiImprovementDecision@1``.
    """

    kind: ImprovementDecisionKind
    reason_codes: tuple[str, ...]
    measurable_improvement: bool = False
    hard_gates_passed: bool = False
    missing_evidence: bool = False
    invariants_preserved: bool = False
    pixel_change_only: bool = False
    promoted: bool = False
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)
    interface: str = GUI_IMPROVEMENT_DECISION_INTERFACE
    schema_version: str = GUI_IMPROVEMENT_DECISION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _as_kind(self.kind))
        codes = tuple(
            sorted({_text(code, "reason_code") for code in self.reason_codes})
        )
        if not codes:
            codes = (ImprovementReasonCode.INVALID_IMPROVEMENT_INPUT.value,)
        object.__setattr__(self, "reason_codes", codes)
        object.__setattr__(self, "promoted", False)
        object.__setattr__(
            self, "details", _freeze_mapping(dict(self.details) if self.details else {})
        )
        object.__setattr__(self, "message", str(self.message or ""))

    @property
    def accepted(self) -> bool:
        return self.kind is ImprovementDecisionKind.ACCEPT

    @property
    def rejected(self) -> bool:
        return self.kind is ImprovementDecisionKind.REJECT

    @property
    def requires_human_review(self) -> bool:
        return self.kind is ImprovementDecisionKind.HUMAN_REVIEW

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "details": dict(self.details),
            "hard_gates_passed": self.hard_gates_passed,
            "interface": self.interface,
            "invariants_preserved": self.invariants_preserved,
            "kind": self.kind.value,
            "measurable_improvement": self.measurable_improvement,
            "message": self.message,
            "missing_evidence": self.missing_evidence,
            "pixel_change_only": self.pixel_change_only,
            "promoted": False,
            "reason_codes": list(self.reason_codes),
            "rejected": self.rejected,
            "requires_human_review": self.requires_human_review,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class GuiImprovementRun:
    """Resumable loop record.  Interface: ``GuiImprovementRun@1``."""

    run_id: str
    attempt: int
    status: RunStatus
    phases: tuple[str, ...]
    decision: GuiImprovementDecision
    application_id: str
    screen_id: str
    objective_id: str
    source_revision: str
    canonical_revision: str
    canonical_branch: str
    checkpoint_cid: str = ""
    terminal_receipt_cid: str = ""
    receipt: Mapping[str, Any] | None = None
    proposal_id: str = ""
    declared_method: str = ""
    declared_tier: str = ""
    canonical_mutated: bool = False
    promoted: bool = False
    interface: str = GUI_IMPROVEMENT_RUN_INTERFACE
    schema_version: str = GUI_IMPROVEMENT_RUN_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id"))
        object.__setattr__(self, "attempt", _positive_int(self.attempt, "attempt"))
        if type(self.status) is not RunStatus:
            object.__setattr__(self, "status", RunStatus(str(self.status)))
        if type(self.decision) is not GuiImprovementDecision:
            raise GuiImprovementLoopError(
                "decision must be a GuiImprovementDecision",
                reason_code=ImprovementReasonCode.INVALID_COLLECTION_TYPE.value,
            )
        object.__setattr__(self, "canonical_mutated", False)
        object.__setattr__(self, "promoted", False)
        if self.receipt is not None:
            object.__setattr__(self, "receipt", MappingProxyType(dict(self.receipt)))

    @property
    def terminal(self) -> bool:
        return self.status is RunStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        return {
            "application_id": self.application_id,
            "attempt": self.attempt,
            "canonical_branch": self.canonical_branch,
            "canonical_mutated": False,
            "canonical_revision": self.canonical_revision,
            "checkpoint_cid": self.checkpoint_cid,
            "decision": self.decision.to_dict(),
            "declared_method": self.declared_method,
            "declared_tier": self.declared_tier,
            "interface": self.interface,
            "objective_id": self.objective_id,
            "phases": list(self.phases),
            "promoted": False,
            "proposal_id": self.proposal_id,
            "receipt": None if self.receipt is None else dict(self.receipt),
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "screen_id": self.screen_id,
            "source_revision": self.source_revision,
            "status": self.status.value,
            "terminal": self.terminal,
            "terminal_receipt_cid": self.terminal_receipt_cid,
        }


# ---------------------------------------------------------------------------
# VerifiedGuiOptimizer@1
# ---------------------------------------------------------------------------


class VerifiedGuiOptimizer:
    """``VerifiedGuiOptimizer@1`` — bounded, journaled improvement loop."""

    interface: str = VERIFIED_GUI_OPTIMIZER_INTERFACE
    schema_version: str = VERIFIED_GUI_OPTIMIZER_SCHEMA

    def __init__(
        self,
        *,
        journal: GuiRunJournal,
        proposer: GuiPatchProposer | None = None,
        planner: GuiAffectedCheckPlanner | None = None,
        executor: GuiIsolatedWorktreeExecutor | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        max_objectives: int = DEFAULT_MAX_OBJECTIVES,
        max_files: int = DEFAULT_MAX_FILES,
    ) -> None:
        if type(journal) is not GuiRunJournal:
            raise GuiImprovementLoopError(
                "journal must be a GuiRunJournal",
                reason_code=ImprovementReasonCode.INVALID_COLLECTION_TYPE.value,
                details={"value_type": type(journal).__name__},
            )
        self.journal = journal
        self.proposer = proposer if proposer is not None else default_gui_patch_proposer()
        self.planner = (
            planner if planner is not None else default_affected_check_planner()
        )
        self.executor = executor
        self.max_attempts = _positive_int(max_attempts, "max_attempts")
        self.max_objectives = _positive_int(max_objectives, "max_objectives")
        self.max_files = _positive_int(max_files, "max_files")
        if self.max_files > DEFAULT_MAX_FILES:
            raise GuiImprovementLoopError(
                "max_files cannot exceed the patch-scope file cap",
                details={"max_files": self.max_files},
            )

    def improve(self, request: Mapping[str, Any]) -> GuiImprovementRun:
        """Run or resume one bounded improvement iteration."""

        parsed = self._parse_request(request)
        if parsed["resume"]:
            return self._resume(parsed)
        return self._run(parsed)

    def resume(self, request: Mapping[str, Any]) -> GuiImprovementRun:
        """Resume an interrupted run by stable run ID."""

        parsed = self._parse_request(request)
        parsed["resume"] = True
        return self._resume(parsed)

    def _parse_request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = _require_mapping(request, "request")
        _reject_unknown(payload, _REQUEST_KEYS, "request")
        required = (
            "run_id",
            "application_id",
            "screen_id",
            "objective_id",
            "source_revision",
            "canonical_branch",
            "canonical_revision",
        )
        for key in required:
            if key not in payload:
                raise GuiImprovementLoopError(
                    f"request.{key} is required",
                    details={"field": key},
                )
            _reject_present_null(payload, key)
        attempt = payload["attempt"] if "attempt" in payload else 1
        if "attempt" in payload:
            _reject_present_null(payload, "attempt")
            attempt = _positive_int(attempt, "attempt")
        max_attempts = (
            payload["max_attempts"] if "max_attempts" in payload else self.max_attempts
        )
        if "max_attempts" in payload:
            _reject_present_null(payload, "max_attempts")
            max_attempts = _positive_int(max_attempts, "max_attempts")
        objective_ids = _optional_string_list(payload, "objective_ids")
        if not objective_ids:
            objective_ids = (_identifier(payload["objective_id"], "objective_id"),)
        halt = ""
        if "halt_after_phase" in payload:
            _reject_present_null(payload, "halt_after_phase")
            halt = _as_phase(payload["halt_after_phase"]).value
        parsed = {
            "run_id": _identifier(payload["run_id"], "run_id"),
            "application_id": _identifier(payload["application_id"], "application_id"),
            "screen_id": _identifier(payload["screen_id"], "screen_id"),
            "objective_id": _identifier(payload["objective_id"], "objective_id"),
            "objective_ids": objective_ids,
            "objective": _optional_text(payload, "objective")
            or "Repair the declared accessible-name defect.",
            "source_revision": _require_revision(
                payload["source_revision"], "source_revision"
            ),
            "canonical_branch": _identifier(
                payload["canonical_branch"], "canonical_branch"
            ),
            "canonical_revision": _require_revision(
                payload["canonical_revision"], "canonical_revision"
            ),
            "canonical_porcelain": _optional_text(payload, "canonical_porcelain"),
            "attempt": attempt,
            "max_attempts": max_attempts,
            "proposal_id": _optional_identifier(payload, "proposal_id"),
            "request_id": _optional_identifier(payload, "request_id")
            or f"req:{payload['run_id']}",
            "route_kind": _optional_text(payload, "route_kind")
            or "deterministic_transform",
            "context_pack": _optional_mapping(payload, "context_pack"),
            "proposal_request": _optional_mapping(payload, "proposal_request"),
            "transformations": payload["transformations"]
            if "transformations" in payload
            else [],
            "intended_file_paths": list(
                _optional_string_list(payload, "intended_file_paths")
            ),
            "intended_component_ids": list(
                _optional_string_list(payload, "intended_component_ids")
            ),
            "acceptance_criteria": list(
                _optional_string_list(payload, "acceptance_criteria", as_text=True)
            ),
            "expected_test_ids": list(
                _optional_string_list(payload, "expected_test_ids")
            ),
            "expected_screenshot_ids": list(
                _optional_string_list(payload, "expected_screenshot_ids")
            ),
            "state_effect_ids": list(
                _optional_string_list(payload, "state_effect_ids")
            ),
            "visual_effect_summary": _optional_text(payload, "visual_effect_summary"),
            "analysis_classification": _optional_text(
                payload, "analysis_classification"
            )
            or "exact",
            "verification_status": _optional_text(payload, "verification_status")
            or "unverified",
            "prior_failure_count": (
                _nonneg_int(payload["prior_failure_count"], "prior_failure_count")
                if "prior_failure_count" in payload
                else 0
            ),
            "escalation_conditions": list(
                _optional_string_list(payload, "escalation_conditions")
            ),
            "policy_bound": _optional_bool(payload, "policy_bound", False),
            "security_sensitive": _optional_bool(payload, "security_sensitive", False),
            "opaque": _optional_bool(payload, "opaque", False),
            "ambiguous": _optional_bool(payload, "ambiguous", False),
            "constraint_conflict": _optional_bool(payload, "constraint_conflict", False),
            "declared_method": _optional_text(payload, "declared_method"),
            "declared_tier": _optional_text(payload, "declared_tier"),
            "baseline": _optional_mapping(payload, "baseline"),
            "baseline_metrics": _optional_mapping(payload, "baseline_metrics"),
            "candidate_metrics": _optional_mapping(payload, "candidate_metrics"),
            "objective_metric_id": _optional_identifier(payload, "objective_metric_id")
            or DEFAULT_OBJECTIVE_METRIC,
            "impact": _optional_mapping(payload, "impact"),
            "invalidation": _optional_mapping(payload, "invalidation"),
            "evaluator_risk": _optional_mapping(payload, "evaluator_risk"),
            "change_kinds": list(_optional_string_list(payload, "change_kinds")),
            "affected_screenshot_ids": list(
                _optional_string_list(payload, "affected_screenshot_ids")
            ),
            "known_screenshot_ids": list(
                _optional_string_list(payload, "known_screenshot_ids")
            ),
            "unrelated_screenshot_ids": list(
                _optional_string_list(payload, "unrelated_screenshot_ids")
            ),
            "application": _optional_mapping(payload, "application"),
            "check_execution": _optional_mapping(payload, "check_execution"),
            "evidence": _optional_mapping(payload, "evidence"),
            "hard_gates": _optional_mapping(payload, "hard_gates"),
            "heuristic_scores": payload["heuristic_scores"]
            if "heuristic_scores" in payload
            else [],
            "pixel_change_only": _optional_bool(payload, "pixel_change_only", False),
            "process_alive": payload.get("process_alive"),
            "resume": _optional_bool(payload, "resume", False),
            "halt_after_phase": halt,
        }
        pack = parsed["context_pack"]
        if pack:
            if (
                parsed["analysis_classification"] == "exact"
                and "analysis_classification" not in payload
                and type(pack.get("analysis_classification")) is str
            ):
                parsed["analysis_classification"] = pack["analysis_classification"]
            if (
                parsed["verification_status"] == "unverified"
                and "verification_status" not in payload
                and type(pack.get("verification_status")) is str
            ):
                parsed["verification_status"] = pack["verification_status"]
            if (
                parsed["objective"] == "Repair the declared accessible-name defect."
                and "objective" not in payload
                and type(pack.get("objective")) is str
            ):
                parsed["objective"] = pack["objective"]
            if not parsed["acceptance_criteria"] and type(
                pack.get("acceptance_criteria")
            ) is list:
                parsed["acceptance_criteria"] = [
                    item
                    for item in pack["acceptance_criteria"]
                    if type(item) is str and item.strip()
                ]
        return parsed

    def _resume(self, parsed: dict[str, Any]) -> GuiImprovementRun:
        decision = self.journal.decide_resume(
            run_id=parsed["run_id"],
            source_revision=parsed["source_revision"],
            canonical_branch=parsed["canonical_branch"],
            canonical_revision=parsed["canonical_revision"],
            canonical_porcelain=parsed["canonical_porcelain"],
            process_alive=parsed["process_alive"]
            if type(parsed["process_alive"]) is bool
            else False,
        )
        if decision.action is ResumeAction.RETURN_COMPLETED:
            checkpoint = decision.checkpoint
            if checkpoint is None:
                raise GuiImprovementLoopError(
                    "completed resume is missing a checkpoint",
                    details={"run_id": parsed["run_id"]},
                )
            receipt = self._load_terminal_receipt(checkpoint)
            return self._run_from_checkpoint(
                parsed,
                checkpoint,
                self._decision_from_receipt(receipt),
                receipt,
            )
        if decision.action is ResumeAction.REJECT:
            return self._terminal(
                parsed,
                GuiImprovementDecision(
                    kind=ImprovementDecisionKind.REJECT,
                    reason_codes=decision.reason_codes,
                    message=decision.message,
                ),
                checkpoint=decision.checkpoint,
            )
        if decision.action is ResumeAction.RESTART:
            return self._run(parsed)
        return self._run(parsed)

    def _run(self, parsed: dict[str, Any]) -> GuiImprovementRun:
        early = self._early_decision(parsed)
        checkpoint = self.journal.open_run(
            run_id=parsed["run_id"],
            application_id=parsed["application_id"],
            screen_id=parsed["screen_id"],
            objective_id=parsed["objective_id"],
            source_revision=parsed["source_revision"],
            canonical_branch=parsed["canonical_branch"],
            canonical_revision=parsed["canonical_revision"],
            canonical_porcelain=parsed["canonical_porcelain"],
            proposal_id=parsed["proposal_id"],
            attempt=parsed["attempt"],
        )
        if checkpoint.status is RunStatus.COMPLETED:
            receipt = self._load_terminal_receipt(checkpoint)
            return self._run_from_checkpoint(
                parsed,
                checkpoint,
                self._decision_from_receipt(receipt),
                receipt,
            )
        state: dict[str, Any] = {
            "proposal_result": None,
            "proposal": None,
            "patch_digest": "",
            "application": dict(parsed["application"]),
            "check_execution": dict(parsed["check_execution"]),
            "comparison": {},
            "early": early,
        }
        for phase in PHASE_ORDER:
            if phase in {JournalPhase.DECISION, JournalPhase.RECEIPT}:
                continue
            if early is not None and phase is not JournalPhase.BASELINE:
                if phase is JournalPhase.SELECT_OBJECTIVE:
                    self._record_phase(
                        parsed,
                        JournalPhase.SELECT_OBJECTIVE,
                        {
                            "objective_id": parsed["objective_id"],
                            "objective_ids": list(parsed["objective_ids"]),
                        },
                    )
                continue
            self._execute_phase(parsed, phase, state)
            if parsed["halt_after_phase"] == phase.value:
                self.journal.mark_interrupted(parsed["run_id"])
                checkpoint = self.journal.require_checkpoint(parsed["run_id"])
                pending = GuiImprovementDecision(
                    kind=ImprovementDecisionKind.PENDING,
                    reason_codes=(ImprovementReasonCode.INTERRUPTED.value,),
                    message="run halted after the requested phase",
                )
                return self._run_from_checkpoint(parsed, checkpoint, pending, None)
            latest = self.journal.require_checkpoint(parsed["run_id"])
            if latest.status is RunStatus.REJECTED and early is None:
                break
        decision = early if early is not None else self._decide(parsed, state)
        return self._terminal(parsed, decision, state=state)

    def _early_decision(
        self, parsed: Mapping[str, Any]
    ) -> GuiImprovementDecision | None:
        if parsed["attempt"] > parsed["max_attempts"]:
            return GuiImprovementDecision(
                kind=ImprovementDecisionKind.REJECT,
                reason_codes=(
                    ImprovementReasonCode.REJECTED.value,
                    ImprovementReasonCode.ATTEMPT_BUDGET_EXHAUSTED.value,
                ),
                message="attempt budget exhausted",
            )
        if len(parsed["objective_ids"]) > self.max_objectives:
            return GuiImprovementDecision(
                kind=ImprovementDecisionKind.REJECT,
                reason_codes=(
                    ImprovementReasonCode.REJECTED.value,
                    ImprovementReasonCode.TOO_MANY_OBJECTIVES.value,
                ),
                message="too many objectives for one bounded iteration",
            )
        files = parsed["intended_file_paths"]
        if len(files) > self.max_files or _REWRITE_RE.search(parsed["objective"]):
            return GuiImprovementDecision(
                kind=ImprovementDecisionKind.REJECT,
                reason_codes=(
                    ImprovementReasonCode.REJECTED.value,
                    ImprovementReasonCode.WHOLE_APP_REWRITE.value,
                ),
                message="whole-app aesthetic rewrite is forbidden",
            )
        application = parsed["application"]
        if application.get("promoted") is True:
            return GuiImprovementDecision(
                kind=ImprovementDecisionKind.REJECT,
                reason_codes=(
                    ImprovementReasonCode.REJECTED.value,
                    ImprovementReasonCode.CANONICAL_MERGE_FORBIDDEN.value,
                ),
                message="automatic canonical merge is forbidden",
            )
        return None

    def _execute_phase(
        self,
        parsed: dict[str, Any],
        phase: JournalPhase,
        state: dict[str, Any],
    ) -> None:
        if phase is JournalPhase.BASELINE:
            payload = dict(parsed["baseline"]) or {
                "violations": [],
                "metric_ids": list(parsed["baseline_metrics"]),
            }
            self._record_phase(parsed, phase, payload)
            return
        if phase is JournalPhase.SELECT_OBJECTIVE:
            self._record_phase(
                parsed,
                phase,
                {
                    "objective_id": parsed["objective_id"],
                    "objective_ids": list(parsed["objective_ids"]),
                    "objective_metric_id": parsed["objective_metric_id"],
                },
            )
            return
        if phase is JournalPhase.IMPACT:
            payload = dict(parsed["impact"]) or {
                "affected_component_ids": list(parsed["intended_component_ids"]),
            }
            self._record_phase(parsed, phase, payload)
            return
        if phase is JournalPhase.CONTEXT_PACK:
            pack = parsed["context_pack"]
            self._record_phase(
                parsed,
                phase,
                {
                    "pack_id": pack.get("pack_id", ""),
                    "analysis_classification": parsed["analysis_classification"],
                    "present": bool(pack),
                },
            )
            return
        if phase is JournalPhase.PROPOSAL:
            self._run_proposal(parsed, state)
            return
        if phase is JournalPhase.ISOLATED_WORKTREE:
            self._run_apply(parsed, state)
            return
        if phase is JournalPhase.RESCAN:
            self._record_phase(
                parsed,
                phase,
                {
                    "changed_component_ids": list(
                        parsed["impact"].get("affected_component_ids", [])
                        or parsed["intended_component_ids"]
                    )
                },
            )
            return
        if phase is JournalPhase.INVALIDATION:
            invalidation = dict(parsed["invalidation"]) or {
                "plan_id": parsed["evidence"].get(
                    "invalidation_plan_id", "invalidate:unspecified"
                ),
                "fallback_triggered": bool(
                    parsed["evaluator_risk"].get("graph_confidence")
                    in {"heuristic", "opaque", "stale"}
                ),
            }
            self._record_phase(parsed, phase, invalidation)
            return
        if phase is JournalPhase.AFFECTED_CHECKS:
            execution = dict(parsed["check_execution"])
            state["check_execution"] = execution
            self._record_phase(
                parsed,
                phase,
                {
                    "acceptance_blocked": bool(execution.get("acceptance_blocked")),
                    "executed_check_ids": list(execution.get("executed_check_ids", [])),
                    "failed_required_check_ids": list(
                        execution.get("failed_required_check_ids", [])
                    ),
                    "present": bool(execution),
                },
            )
            return
        if phase is JournalPhase.FALLBACK:
            execution = state.get("check_execution") or {}
            triggered = bool(
                execution.get("fallback_applied")
                or parsed["invalidation"].get("fallback_triggered")
                or parsed["evaluator_risk"].get("graph_confidence")
                in {"heuristic", "opaque", "stale"}
            )
            self._record_phase(
                parsed,
                phase,
                {
                    "fallback_applied": triggered,
                    "explanation": (
                        "uncertainty or required-check failure expanded fallback"
                        if triggered
                        else "no uncertainty requires broad fallback"
                    ),
                },
            )
            return
        if phase is JournalPhase.COMPARE:
            comparison = self._compare(parsed)
            state["comparison"] = comparison
            self._record_phase(parsed, phase, comparison)
            return

    def _run_proposal(self, parsed: dict[str, Any], state: dict[str, Any]) -> None:
        request = parsed["proposal_request"] or self._build_proposal_request(parsed)
        if not request:
            self._record_phase(
                parsed,
                JournalPhase.PROPOSAL,
                {
                    "disposition": ProposalDisposition.REJECT.value,
                    "reason_codes": [
                        ImprovementReasonCode.MISSING_EVIDENCE.value,
                    ],
                },
                status=PhaseRecordStatus.REJECTED,
            )
            state["proposal_result"] = None
            return
        try:
            result = self.proposer.propose(request)
        except GuiProposalError as exc:
            self._record_phase(
                parsed,
                JournalPhase.PROPOSAL,
                {
                    "disposition": ProposalDisposition.REJECT.value,
                    "reason_codes": [exc.reason_code],
                    "message": str(exc),
                },
                status=PhaseRecordStatus.REJECTED,
            )
            state["proposal_result"] = None
            state["proposal_error"] = exc.reason_code
            return
        state["proposal_result"] = result
        if result.proposed and result.proposal is not None:
            state["proposal"] = dict(result.proposal)
            parsed["proposal_id"] = str(result.proposal.get("proposal_id", ""))
            if result.patch_text:
                state["patch_digest"] = _sha256_digest(result.patch_text)
        self._record_phase(
            parsed,
            JournalPhase.PROPOSAL,
            {
                "disposition": result.disposition.value,
                "reason_codes": list(result.reason_codes),
                "declared_method": result.declared_method,
                "declared_tier": result.declared_tier,
                "proposal_id": parsed["proposal_id"],
                "escalated": result.escalated,
                "review_id": (
                    ""
                    if result.review_request is None
                    else result.review_request.review_id
                ),
            },
            status=(
                PhaseRecordStatus.COMPLETED
                if result.proposed
                else PhaseRecordStatus.REJECTED
                if result.disposition is ProposalDisposition.REJECT
                else PhaseRecordStatus.COMPLETED
            ),
        )

    def _build_proposal_request(
        self, parsed: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        pack = parsed["context_pack"]
        if not pack:
            return None
        files = list(parsed["intended_file_paths"])
        sources = pack.get("raw_sources") if type(pack.get("raw_sources")) is list else []
        if not files:
            for item in sources:
                if type(item) is dict and type(item.get("path")) is str:
                    files.append(item["path"])
        components = list(parsed["intended_component_ids"])
        if not components:
            for item in sources:
                if type(item) is dict and type(item.get("component_id")) is str:
                    components.append(item["component_id"])
        if not components:
            components = ["comp:unspecified"]
        criteria = list(parsed["acceptance_criteria"]) or ["crit:unspecified"]
        request: dict[str, Any] = {
            "request_id": parsed["request_id"],
            "route_kind": parsed["route_kind"],
            "context_pack": dict(pack),
            "intended_file_paths": files,
            "intended_component_ids": components,
            "acceptance_criteria": criteria,
            "objective": parsed["objective"],
            "application_id": parsed["application_id"],
            "screen_id": parsed["screen_id"],
            "analysis_classification": parsed["analysis_classification"],
            "verification_status": parsed["verification_status"],
            "prior_failure_count": parsed["prior_failure_count"],
        }
        if parsed["transformations"]:
            request["transformations"] = parsed["transformations"]
        if parsed["expected_test_ids"]:
            request["expected_test_ids"] = list(parsed["expected_test_ids"])
        if parsed["expected_screenshot_ids"]:
            request["expected_screenshot_ids"] = list(parsed["expected_screenshot_ids"])
        if parsed["state_effect_ids"]:
            request["state_effect_ids"] = list(parsed["state_effect_ids"])
        if parsed["escalation_conditions"]:
            request["escalation_conditions"] = list(parsed["escalation_conditions"])
        if parsed["visual_effect_summary"]:
            request["visual_effect_summary"] = parsed["visual_effect_summary"]
        if parsed["declared_method"]:
            request["declared_method"] = parsed["declared_method"]
        if parsed["declared_tier"]:
            request["declared_tier"] = parsed["declared_tier"]
        if parsed["policy_bound"]:
            request["policy_bound"] = True
        if parsed["security_sensitive"]:
            request["security_sensitive"] = True
        if parsed["opaque"]:
            request["opaque"] = True
        if parsed["ambiguous"]:
            request["ambiguous"] = True
        if parsed["constraint_conflict"]:
            request["constraint_conflict"] = True
        return request

    def _run_apply(self, parsed: dict[str, Any], state: dict[str, Any]) -> None:
        result: GuiProposalResult | None = state.get("proposal_result")
        application = dict(parsed["application"])
        if result is None or not result.proposed:
            self._record_phase(
                parsed,
                JournalPhase.ISOLATED_WORKTREE,
                {
                    "applied": False,
                    "promoted": False,
                    "skipped": True,
                    "reason": "no admitted proposal",
                },
            )
            state["application"] = {
                "applied": False,
                "promoted": False,
                "skipped": True,
            }
            return
        if application.get("promoted") is True:
            self._record_phase(
                parsed,
                JournalPhase.ISOLATED_WORKTREE,
                {
                    "applied": False,
                    "promoted": False,
                    "reason_codes": [
                        ImprovementReasonCode.CANONICAL_MERGE_FORBIDDEN.value
                    ],
                },
                status=PhaseRecordStatus.REJECTED,
            )
            state["application"] = {"applied": False, "promoted": False}
            return
        applied = application.get("applied") is True
        self._record_phase(
            parsed,
            JournalPhase.ISOLATED_WORKTREE,
            {
                "applied": applied,
                "promoted": False,
                "disposition": application.get(
                    "disposition", "applied" if applied else "missing"
                ),
                "reason_codes": list(application.get("reason_codes", [])),
            },
            status=(
                PhaseRecordStatus.COMPLETED if applied else PhaseRecordStatus.REJECTED
            ),
        )
        state["application"] = {
            "applied": applied,
            "promoted": False,
            "present": bool(application),
        }

    def _compare(self, parsed: Mapping[str, Any]) -> dict[str, Any]:
        metric_id = parsed["objective_metric_id"]
        before_map = parsed["baseline_metrics"]
        after_map = parsed["candidate_metrics"]
        before = before_map.get(metric_id)
        after = after_map.get(metric_id)
        improved = False
        if type(before) in {int, float} and type(after) in {int, float}:
            improved = _metric_improved(metric_id, float(before), float(after))
        regressions: list[str] = []
        gates = parsed["hard_gates"]
        if gates.get("accessibility_regression") is True:
            regressions.append(ImprovementReasonCode.ACCESSIBILITY_REGRESSION.value)
        if gates.get("security_regression") is True:
            regressions.append(ImprovementReasonCode.SECURITY_REGRESSION.value)
        if gates.get("confirmation_regression") is True:
            regressions.append(ImprovementReasonCode.CONFIRMATION_REGRESSION.value)
        if gates.get("policy_regression") is True:
            regressions.append(ImprovementReasonCode.POLICY_REGRESSION.value)
        if gates.get("functional_regression") is True:
            regressions.append(ImprovementReasonCode.FUNCTIONAL_REGRESSION.value)
        for key, value in after_map.items():
            if key not in HARD_GATE_METRICS:
                continue
            prior = before_map.get(key)
            if (
                type(prior) in {int, float}
                and type(value) in {int, float}
                and _metric_regressed(str(key), float(prior), float(value))
            ):
                if key == "critical_accessibility_violations":
                    regressions.append(
                        ImprovementReasonCode.ACCESSIBILITY_REGRESSION.value
                    )
                elif key == "security_violations":
                    regressions.append(ImprovementReasonCode.SECURITY_REGRESSION.value)
                elif key == "confirmation_bypass_count":
                    regressions.append(
                        ImprovementReasonCode.CONFIRMATION_REGRESSION.value
                    )
                else:
                    regressions.append(ImprovementReasonCode.HARD_GATE_REGRESSION.value)
        unique_reg = list(dict.fromkeys(regressions))
        pixel_only = parsed["pixel_change_only"] is True
        if not pixel_only and after_map and before_map:
            other_improved = False
            pixel_improved = False
            for key, value in after_map.items():
                prior = before_map.get(key)
                if type(prior) not in {int, float} or type(value) not in {int, float}:
                    continue
                if not _metric_improved(str(key), float(prior), float(value)):
                    continue
                if key == "pixel_diff_percent":
                    pixel_improved = True
                elif key != metric_id or key == "pixel_diff_percent":
                    other_improved = True
                else:
                    other_improved = True
            if pixel_improved and not other_improved and metric_id != "pixel_diff_percent":
                pixel_only = True
        return {
            "objective_metric_id": metric_id,
            "before": before,
            "after": after,
            "measurable_improvement": improved,
            "hard_gate_regressions": unique_reg,
            "pixel_change_only": pixel_only,
            "heuristic_scores_present": bool(parsed["heuristic_scores"]),
        }

    def _decide(
        self, parsed: Mapping[str, Any], state: Mapping[str, Any]
    ) -> GuiImprovementDecision:
        result: GuiProposalResult | None = state.get("proposal_result")
        comparison: Mapping[str, Any] = state.get("comparison") or {}
        application: Mapping[str, Any] = state.get("application") or {}
        checks: Mapping[str, Any] = state.get("check_execution") or {}
        evidence = parsed["evidence"]
        codes: list[str] = []

        if state.get("proposal_error"):
            return GuiImprovementDecision(
                kind=ImprovementDecisionKind.REJECT,
                reason_codes=(
                    ImprovementReasonCode.REJECTED.value,
                    ImprovementReasonCode.PROPOSAL_REJECTED.value,
                    str(state["proposal_error"]),
                ),
                message="proposal input was rejected",
            )
        if result is not None and result.escalated:
            codes = [
                ImprovementReasonCode.HUMAN_REVIEW_REQUIRED.value,
                ImprovementReasonCode.PROPOSAL_ESCALATED.value,
                *result.reason_codes,
            ]
            return GuiImprovementDecision(
                kind=ImprovementDecisionKind.HUMAN_REVIEW,
                reason_codes=tuple(dict.fromkeys(codes)),
                missing_evidence=False,
                message="proposal escalated without a fabricated patch",
                details={
                    "declared_method": result.declared_method,
                    "declared_tier": result.declared_tier,
                },
            )
        if result is None or not result.proposed:
            return GuiImprovementDecision(
                kind=ImprovementDecisionKind.REJECT,
                reason_codes=(
                    ImprovementReasonCode.REJECTED.value,
                    ImprovementReasonCode.PROPOSAL_REJECTED.value,
                    ImprovementReasonCode.MISSING_EVIDENCE.value,
                ),
                missing_evidence=True,
                message="no admitted proposal",
            )

        regressions = list(comparison.get("hard_gate_regressions") or [])
        if regressions:
            codes = [
                ImprovementReasonCode.REJECTED.value,
                ImprovementReasonCode.HARD_GATE_REGRESSION.value,
                *regressions,
            ]
            return GuiImprovementDecision(
                kind=ImprovementDecisionKind.REJECT,
                reason_codes=tuple(dict.fromkeys(codes)),
                measurable_improvement=bool(comparison.get("measurable_improvement")),
                hard_gates_passed=False,
                invariants_preserved=False,
                message="hard-gate regression blocks acceptance",
            )

        if checks.get("acceptance_blocked") or checks.get("failed_required_check_ids"):
            return GuiImprovementDecision(
                kind=ImprovementDecisionKind.REJECT,
                reason_codes=(
                    ImprovementReasonCode.REJECTED.value,
                    ImprovementReasonCode.REQUIRED_CHECK_FAILED.value,
                ),
                measurable_improvement=bool(comparison.get("measurable_improvement")),
                hard_gates_passed=True,
                message="a required check blocked acceptance",
            )

        if application.get("promoted") is True:
            return GuiImprovementDecision(
                kind=ImprovementDecisionKind.REJECT,
                reason_codes=(
                    ImprovementReasonCode.REJECTED.value,
                    ImprovementReasonCode.CANONICAL_MERGE_FORBIDDEN.value,
                ),
                message="automatic canonical merge is forbidden",
            )
        if not application.get("applied"):
            code = (
                ImprovementReasonCode.ISOLATED_APPLY_REJECTED.value
                if application.get("present")
                else ImprovementReasonCode.ISOLATED_APPLY_MISSING.value
            )
            return GuiImprovementDecision(
                kind=ImprovementDecisionKind.REJECT,
                reason_codes=(ImprovementReasonCode.REJECTED.value, code),
                missing_evidence=not application.get("present"),
                message="isolated apply did not admit the patch",
            )

        missing = self._missing_evidence(evidence)
        classification = parsed["analysis_classification"]
        if missing:
            kind = (
                ImprovementDecisionKind.HUMAN_REVIEW
                if classification in {"opaque", "heuristic"}
                else ImprovementDecisionKind.REJECT
            )
            return GuiImprovementDecision(
                kind=kind,
                reason_codes=(
                    ImprovementReasonCode.HUMAN_REVIEW_REQUIRED.value
                    if kind is ImprovementDecisionKind.HUMAN_REVIEW
                    else ImprovementReasonCode.REJECTED.value,
                    ImprovementReasonCode.MISSING_EVIDENCE.value,
                    ImprovementReasonCode.UNKNOWN_CRITICAL_EVIDENCE.value,
                ),
                measurable_improvement=bool(comparison.get("measurable_improvement")),
                hard_gates_passed=True,
                missing_evidence=True,
                message="missing critical evidence cannot auto-accept",
            )

        if comparison.get("pixel_change_only"):
            return GuiImprovementDecision(
                kind=ImprovementDecisionKind.REJECT,
                reason_codes=(
                    ImprovementReasonCode.REJECTED.value,
                    ImprovementReasonCode.PIXEL_CHANGE_ONLY.value,
                    ImprovementReasonCode.NO_MEASURABLE_IMPROVEMENT.value,
                ),
                pixel_change_only=True,
                hard_gates_passed=True,
                message="pixel change alone is not an objective improvement",
            )

        if parsed["heuristic_scores"] and not comparison.get("measurable_improvement"):
            return GuiImprovementDecision(
                kind=ImprovementDecisionKind.HUMAN_REVIEW,
                reason_codes=(
                    ImprovementReasonCode.HUMAN_REVIEW_REQUIRED.value,
                    ImprovementReasonCode.HEURISTIC_CANNOT_OVERRIDE.value,
                ),
                hard_gates_passed=True,
                message="heuristic scores cannot override missing objective gain",
            )

        if not comparison.get("measurable_improvement"):
            return GuiImprovementDecision(
                kind=ImprovementDecisionKind.REJECT,
                reason_codes=(
                    ImprovementReasonCode.REJECTED.value,
                    ImprovementReasonCode.NO_MEASURABLE_IMPROVEMENT.value,
                ),
                hard_gates_passed=True,
                invariants_preserved=True,
                message="acceptance requires a measurable target-metric gain",
            )

        return GuiImprovementDecision(
            kind=ImprovementDecisionKind.ACCEPT,
            reason_codes=(
                ImprovementReasonCode.ACCEPTED.value,
                ImprovementReasonCode.MEASURABLE_IMPROVEMENT.value,
            ),
            measurable_improvement=True,
            hard_gates_passed=True,
            invariants_preserved=True,
            message="target metric improved and hard gates held",
        )

    def _missing_evidence(self, evidence: Mapping[str, Any]) -> bool:
        required = (
            "visual_receipt_ids",
            "accessibility_receipt_ids",
            "interaction_receipt_ids",
            "constraint_receipt_ids",
        )
        for key in required:
            value = evidence.get(key)
            if type(value) is not list or not value:
                return True
        if not evidence.get("invalidation_plan_id") or not evidence.get("context_pack_id"):
            return True
        return False

    def _terminal(
        self,
        parsed: dict[str, Any],
        decision: GuiImprovementDecision,
        *,
        checkpoint: GuiRunCheckpoint | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> GuiImprovementRun:
        state = state or {}
        result: GuiProposalResult | None = state.get("proposal_result")
        receipt = self._build_receipt(parsed, decision, state)
        self._record_phase(
            parsed,
            JournalPhase.DECISION,
            {
                "kind": decision.kind.value,
                "reason_codes": list(decision.reason_codes),
                "measurable_improvement": decision.measurable_improvement,
                "hard_gates_passed": decision.hard_gates_passed,
                "missing_evidence": decision.missing_evidence,
                "promoted": False,
            },
            status=(
                PhaseRecordStatus.COMPLETED
                if decision.kind is not ImprovementDecisionKind.REJECT
                else PhaseRecordStatus.REJECTED
            ),
        )
        self._record_phase(
            parsed,
            JournalPhase.RECEIPT,
            {
                "receipt_id": receipt["receipt_id"],
                "decision": receipt["decision"],
            },
        )
        sealed = self.journal.commit_terminal_receipt(parsed["run_id"], receipt)
        return GuiImprovementRun(
            run_id=parsed["run_id"],
            attempt=parsed["attempt"],
            status=sealed.status,
            phases=tuple(record.phase.value for record in sealed.phase_records),
            decision=decision,
            application_id=parsed["application_id"],
            screen_id=parsed["screen_id"],
            objective_id=parsed["objective_id"],
            source_revision=parsed["source_revision"],
            canonical_revision=parsed["canonical_revision"],
            canonical_branch=parsed["canonical_branch"],
            checkpoint_cid=sealed.cid,
            terminal_receipt_cid=sealed.terminal_receipt_cid,
            receipt=receipt,
            proposal_id=parsed["proposal_id"]
            or ("" if result is None or result.proposal is None else str(
                result.proposal.get("proposal_id", "")
            )),
            declared_method="" if result is None else result.declared_method,
            declared_tier="" if result is None else result.declared_tier,
        )

    def _build_receipt(
        self,
        parsed: Mapping[str, Any],
        decision: GuiImprovementDecision,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        evidence = parsed["evidence"]
        result: GuiProposalResult | None = state.get("proposal_result")
        proposal_id = parsed["proposal_id"]
        if not proposal_id and result is not None and result.proposal is not None:
            proposal_id = str(result.proposal.get("proposal_id", ""))
        if not proposal_id and result is not None and result.review_request is not None:
            proposal_id = result.review_request.review_id
        if not proposal_id:
            proposal_id = "proposal:none"
        patch_digest = str(state.get("patch_digest") or "")
        if decision.kind is ImprovementDecisionKind.ACCEPT and not patch_digest:
            patch_digest = _sha256_digest(
                {
                    "run_id": parsed["run_id"],
                    "proposal_id": proposal_id,
                }
            )
        classification = parsed["analysis_classification"]
        if classification not in {"exact", "conservative", "heuristic", "opaque"}:
            classification = "conservative"
        if decision.kind is ImprovementDecisionKind.ACCEPT:
            verification = "verified"
            reasons: list[str] = []
            visual = list(evidence.get("visual_receipt_ids") or [])
            a11y = list(evidence.get("accessibility_receipt_ids") or [])
            interaction = list(evidence.get("interaction_receipt_ids") or [])
            constraint = list(evidence.get("constraint_receipt_ids") or [])
            invalidation_id = str(evidence.get("invalidation_plan_id") or "")
            pack_id = str(evidence.get("context_pack_id") or "")
        else:
            verification = "unverified"
            reasons = list(decision.reason_codes)
            visual = list(evidence.get("visual_receipt_ids") or [])
            a11y = list(evidence.get("accessibility_receipt_ids") or [])
            interaction = list(evidence.get("interaction_receipt_ids") or [])
            constraint = list(evidence.get("constraint_receipt_ids") or [])
            invalidation_id = str(evidence.get("invalidation_plan_id") or "")
            pack_id = str(
                evidence.get("context_pack_id")
                or parsed["context_pack"].get("pack_id")
                or ""
            )
            if decision.kind is ImprovementDecisionKind.REJECT and not reasons:
                reasons = [ImprovementReasonCode.REJECTED.value]
        receipt_decision = (
            ProposalDecision.ACCEPT.value
            if decision.kind is ImprovementDecisionKind.ACCEPT
            else ProposalDecision.HUMAN_REVIEW.value
            if decision.kind is ImprovementDecisionKind.HUMAN_REVIEW
            else ProposalDecision.REJECT.value
        )
        body = {
            "accessibility_receipt_ids": a11y,
            "analysis_classification": classification,
            "application_id": parsed["application_id"],
            "constraint_receipt_ids": constraint,
            "context_pack_id": pack_id,
            "decision": receipt_decision,
            "interaction_receipt_ids": interaction,
            "interface": GUI_IMPROVEMENT_RECEIPT_INTERFACE,
            "invalidation_plan_id": invalidation_id,
            "patch_digest": patch_digest
            if decision.kind is ImprovementDecisionKind.ACCEPT
            else "",
            "proposal_id": proposal_id,
            "rejection_reasons": reasons,
            "repository_revision": parsed["source_revision"],
            "schema_version": GUI_IMPROVEMENT_RECEIPT_SCHEMA,
            "screen_id": parsed["screen_id"],
            "verification_status": verification,
            "visual_receipt_ids": visual,
        }
        body["receipt_id"] = _digest_id("receipt", body)
        return body

    def _record_phase(
        self,
        parsed: Mapping[str, Any],
        phase: JournalPhase,
        payload: Mapping[str, Any],
        *,
        status: PhaseRecordStatus = PhaseRecordStatus.COMPLETED,
    ) -> GuiRunCheckpoint:
        effect_id = f"effect:{parsed['run_id']}:{phase.value}:{parsed['attempt']}"
        return self.journal.append_phase(
            run_id=parsed["run_id"],
            phase=phase,
            effect_id=effect_id,
            payload=_sanitize_payload(dict(payload)),
            status=status,
        )

    def _load_terminal_receipt(
        self, checkpoint: GuiRunCheckpoint
    ) -> dict[str, Any] | None:
        if not checkpoint.terminal_receipt_cid:
            return None
        body, _record = self.journal.store.get(checkpoint.terminal_receipt_cid)
        payload = json.loads(body.decode("utf-8"))
        if type(payload) is dict:
            return dict(payload)
        return None

    def _decision_from_receipt(
        self, receipt: Mapping[str, Any] | None
    ) -> GuiImprovementDecision:
        if receipt is None:
            return GuiImprovementDecision(
                kind=ImprovementDecisionKind.PENDING,
                reason_codes=(ImprovementReasonCode.INTERRUPTED.value,),
            )
        kind = receipt.get("decision", ImprovementDecisionKind.REJECT.value)
        if kind == ProposalDecision.ACCEPT.value:
            mapped = ImprovementDecisionKind.ACCEPT
        elif kind == ProposalDecision.HUMAN_REVIEW.value:
            mapped = ImprovementDecisionKind.HUMAN_REVIEW
        else:
            mapped = ImprovementDecisionKind.REJECT
        reasons = receipt.get("rejection_reasons") or (
            [ImprovementReasonCode.ACCEPTED.value]
            if mapped is ImprovementDecisionKind.ACCEPT
            else [ImprovementReasonCode.REJECTED.value]
        )
        return GuiImprovementDecision(
            kind=mapped,
            reason_codes=tuple(str(item) for item in reasons),
            measurable_improvement=mapped is ImprovementDecisionKind.ACCEPT,
            hard_gates_passed=mapped is ImprovementDecisionKind.ACCEPT,
            invariants_preserved=mapped is ImprovementDecisionKind.ACCEPT,
        )

    def _run_from_checkpoint(
        self,
        parsed: Mapping[str, Any],
        checkpoint: GuiRunCheckpoint,
        decision: GuiImprovementDecision,
        receipt: Mapping[str, Any] | None,
    ) -> GuiImprovementRun:
        return GuiImprovementRun(
            run_id=checkpoint.run_id,
            attempt=checkpoint.attempt,
            status=checkpoint.status,
            phases=tuple(record.phase.value for record in checkpoint.phase_records),
            decision=decision,
            application_id=checkpoint.application_id,
            screen_id=checkpoint.screen_id,
            objective_id=checkpoint.objective_id,
            source_revision=checkpoint.source_revision,
            canonical_revision=checkpoint.canonical_revision,
            canonical_branch=checkpoint.canonical_branch,
            checkpoint_cid=checkpoint.cid,
            terminal_receipt_cid=checkpoint.terminal_receipt_cid,
            receipt=receipt,
            proposal_id=checkpoint.proposal_id,
        )


def default_verified_gui_optimizer(host_root: Path | str) -> VerifiedGuiOptimizer:
    """Construct the standalone optimizer bound to a host-owned journal."""

    return VerifiedGuiOptimizer(journal=default_run_journal(host_root))


__all__ = (
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_OBJECTIVES",
    "DEFAULT_OBJECTIVE_METRIC",
    "GUI_IMPROVEMENT_DECISION_INTERFACE",
    "GUI_IMPROVEMENT_DECISION_SCHEMA",
    "GUI_IMPROVEMENT_RUN_INTERFACE",
    "GUI_IMPROVEMENT_RUN_SCHEMA",
    "PHASE_ORDER",
    "VERIFIED_GUI_OPTIMIZER_INTERFACE",
    "VERIFIED_GUI_OPTIMIZER_SCHEMA",
    "GuiImprovementDecision",
    "GuiImprovementLoopError",
    "GuiImprovementRun",
    "ImprovementDecisionKind",
    "ImprovementReasonCode",
    "VerifiedGuiOptimizer",
    "default_verified_gui_optimizer",
)
