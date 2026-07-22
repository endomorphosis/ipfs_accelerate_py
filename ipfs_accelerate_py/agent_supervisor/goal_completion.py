"""Evidence-backed objective lifecycle and completion decisions.

Goal completion is intentionally a two-phase operation.  Finishing the tasks
associated with a goal only makes the goal *provisionally* complete.  A later
evaluation may verify it after every acceptance criterion has fresh,
tree-bound, passing evidence with content-addressed provenance.

The types in this module are independent of the markdown objective tracker so
they can also be used by daemons, APIs, and persisted graph consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


GOAL_COMPLETION_SCHEMA_VERSION = 1
DEFAULT_EVIDENCE_FRESHNESS_SECONDS = 3600.0
DEFAULT_CLOCK_SKEW_SECONDS = 300.0


class GoalState(str, Enum):
    """Canonical states in the goal lifecycle."""

    ACTIVE = "active"
    PROVISIONALLY_COMPLETE = "provisionally_complete"
    VERIFIED_COMPLETE = "verified_complete"
    ANALYSIS_INCONCLUSIVE = "analysis_inconclusive"
    BLOCKED = "blocked"
    REOPENED = "reopened"


_GOAL_STATE_ALIASES = {
    "active": GoalState.ACTIVE,
    "todo": GoalState.ACTIVE,
    "open": GoalState.ACTIVE,
    "in_progress": GoalState.ACTIVE,
    "provisional": GoalState.PROVISIONALLY_COMPLETE,
    "provisionally_complete": GoalState.PROVISIONALLY_COMPLETE,
    "provisionally_completed": GoalState.PROVISIONALLY_COMPLETE,
    "verified": GoalState.VERIFIED_COMPLETE,
    "verified_complete": GoalState.VERIFIED_COMPLETE,
    # Compatibility is deliberately confined to normalization.  New writers
    # always persist the more precise verified_complete spelling.
    "complete": GoalState.VERIFIED_COMPLETE,
    "completed": GoalState.VERIFIED_COMPLETE,
    "analysis_inconclusive": GoalState.ANALYSIS_INCONCLUSIVE,
    "inconclusive": GoalState.ANALYSIS_INCONCLUSIVE,
    "blocked": GoalState.BLOCKED,
    "reopened": GoalState.REOPENED,
}


def normalize_goal_state(value: GoalState | str | None) -> GoalState:
    """Return a canonical state, accepting documented legacy spellings."""

    if isinstance(value, GoalState):
        return value
    normalized = str(value or "active").strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _GOAL_STATE_ALIASES[normalized]
    except KeyError as exc:
        choices = ", ".join(state.value for state in GoalState)
        raise ValueError(f"unknown goal state {value!r}; expected one of: {choices}") from exc


LEGAL_GOAL_TRANSITIONS: Mapping[GoalState, frozenset[GoalState]] = {
    GoalState.ACTIVE: frozenset(
        {
            GoalState.PROVISIONALLY_COMPLETE,
            GoalState.ANALYSIS_INCONCLUSIVE,
            GoalState.BLOCKED,
        }
    ),
    GoalState.PROVISIONALLY_COMPLETE: frozenset(
        {
            GoalState.VERIFIED_COMPLETE,
            GoalState.REOPENED,
            GoalState.ANALYSIS_INCONCLUSIVE,
            GoalState.BLOCKED,
        }
    ),
    GoalState.VERIFIED_COMPLETE: frozenset({GoalState.REOPENED}),
    GoalState.ANALYSIS_INCONCLUSIVE: frozenset(
        {GoalState.ACTIVE, GoalState.REOPENED, GoalState.BLOCKED, GoalState.PROVISIONALLY_COMPLETE}
    ),
    GoalState.BLOCKED: frozenset({GoalState.REOPENED}),
    GoalState.REOPENED: frozenset(
        {
            GoalState.ACTIVE,
            GoalState.PROVISIONALLY_COMPLETE,
            GoalState.ANALYSIS_INCONCLUSIVE,
            GoalState.BLOCKED,
        }
    ),
}


def legal_goal_transitions(state: GoalState | str) -> frozenset[GoalState]:
    return LEGAL_GOAL_TRANSITIONS[normalize_goal_state(state)]


def is_terminal_goal_state(state: GoalState | str) -> bool:
    """Whether no work should be scheduled until an explicit transition."""

    return normalize_goal_state(state) in {GoalState.VERIFIED_COMPLETE, GoalState.BLOCKED}


def is_schedulable_goal_state(state: GoalState | str) -> bool:
    """Whether backlog generation may schedule implementation for the goal."""

    return normalize_goal_state(state) in {GoalState.ACTIVE, GoalState.REOPENED}


class IllegalGoalTransitionError(ValueError):
    """Raised when a caller attempts a transition outside the state machine."""


# Concise public spelling retained for callers which do not use the Error
# suffix convention.
IllegalGoalTransition = IllegalGoalTransitionError


def _utc_datetime(value: datetime | str | None, *, field_name: str) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            value = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime or ISO-8601 string")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _now(value: datetime | str | None) -> datetime:
    return _utc_datetime(value, field_name="now") or datetime.now(timezone.utc)


def _criterion_key(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class CompletionEvidence:
    """Proof offered for one acceptance criterion.

    Fields are permissive at construction time so incomplete external records
    can be evaluated and rejected with actionable reasons instead of failing
    deserialization.  ``producer_id``/``tree_id`` are compatibility aliases for
    the more descriptive canonical fields.
    """

    acceptance_criterion: str
    producing_task_or_scan: str = ""
    validation_receipt: Mapping[str, Any] | str | None = None
    repository_tree: str = ""
    freshness: Mapping[str, Any] | str | bool | None = None
    provenance_cid: str = ""
    producer_id: str = ""
    producer_kind: str = "task"
    repository_id: str = ""
    tree_id: str = ""
    observed_at: datetime | str | None = None
    fresh_until: datetime | str | None = None
    validation_passed: bool | None = None
    contradictory: bool = False
    contradiction: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = GOAL_COMPLETION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "acceptance_criterion", str(self.acceptance_criterion or "").strip())
        producer = str(self.producing_task_or_scan or self.producer_id or "").strip()
        object.__setattr__(self, "producing_task_or_scan", producer)
        object.__setattr__(self, "producer_id", str(self.producer_id or producer).strip())
        object.__setattr__(self, "producer_kind", str(self.producer_kind or "task").strip().lower())
        tree = str(self.repository_tree or self.tree_id or "").strip()
        object.__setattr__(self, "repository_tree", tree)
        object.__setattr__(self, "tree_id", str(self.tree_id or tree).strip())
        object.__setattr__(self, "repository_id", str(self.repository_id or "").strip())
        object.__setattr__(self, "provenance_cid", str(self.provenance_cid or "").strip())
        object.__setattr__(self, "observed_at", _utc_datetime(self.observed_at, field_name="observed_at"))
        object.__setattr__(self, "fresh_until", _utc_datetime(self.fresh_until, field_name="fresh_until"))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    @property
    def evidence_cid(self) -> str:
        return self.provenance_cid

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "acceptance_criterion": self.acceptance_criterion,
            "producing_task_or_scan": self.producing_task_or_scan,
            "producer_id": self.producer_id,
            "producer_kind": self.producer_kind,
            "validation_receipt": _json_value(self.validation_receipt),
            "repository_id": self.repository_id,
            "repository_tree": self.repository_tree,
            "tree_id": self.tree_id,
            "freshness": _json_value(self.freshness),
            "observed_at": _json_value(self.observed_at),
            "fresh_until": _json_value(self.fresh_until),
            "provenance_cid": self.provenance_cid,
            "validation_passed": self.validation_passed,
            "contradictory": self.contradictory,
            "contradiction": self.contradiction,
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompletionEvidence":
        values = dict(payload)
        # Common persisted receipt spellings.
        values.setdefault("producing_task_or_scan", values.get("producer_id", values.get("producer", "")))
        values.setdefault("repository_tree", values.get("tree_id", values.get("tree_identity", "")))
        values.setdefault("provenance_cid", values.get("receipt_cid", values.get("cid", "")))
        values.setdefault("validation_receipt", values.get("validation", values.get("receipt", None)))
        values.setdefault("freshness", values.get("fresh", values.get("freshness_status", None)))
        values.setdefault(
            "observed_at",
            values.get("finished_at", values.get("generated_at", values.get("created_at", None))),
        )
        allowed = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in allowed})

    @classmethod
    def from_scan_receipt(
        cls,
        *,
        acceptance_criterion: str,
        receipt: Any,
        validation_receipt: Mapping[str, Any] | str,
        validation_passed: bool = True,
    ) -> "CompletionEvidence":
        """Build evidence from a typed REF-200 scan receipt."""

        safe = bool(getattr(receipt, "safe_for_completion_reasoning", False))
        return cls(
            acceptance_criterion=acceptance_criterion,
            producing_task_or_scan=f"scan:{getattr(receipt, 'analyzer_version', '')}",
            producer_kind="scan",
            validation_receipt=validation_receipt,
            validation_passed=validation_passed,
            repository_id=str(getattr(receipt, "repository_id", "")),
            repository_tree=str(getattr(receipt, "tree_id", "")),
            observed_at=getattr(receipt, "finished_at", None),
            freshness={"fresh": True},
            provenance_cid=str(getattr(receipt, "receipt_cid", "")),
            metadata={
                "safe_for_completion_reasoning": safe,
                "terminal_reason": str(getattr(getattr(receipt, "terminal_reason", ""), "value", getattr(receipt, "terminal_reason", ""))),
            },
        )


@dataclass(frozen=True)
class EvidenceValidationResult:
    evidence: CompletionEvidence
    valid: bool
    reason_codes: tuple[str, ...] = ()
    actionable_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "reason_codes": list(self.reason_codes),
            "actionable_reasons": list(self.actionable_reasons),
            "evidence": self.evidence.to_dict(),
        }


def _validation_passed(evidence: CompletionEvidence) -> bool | None:
    declared = (
        _bool_value(evidence.validation_passed)
        if evidence.validation_passed is not None
        else None
    )
    receipt = evidence.validation_receipt
    if not isinstance(receipt, Mapping):
        return declared
    # A positive convenience flag must never override a terminal failure in
    # the receipt it summarizes.  Persisted records occasionally contain both
    # during interrupted status updates, and completion must fail closed.
    attempted = _bool_value(receipt.get("attempted")) if "attempted" in receipt else None
    if attempted is False:
        return False
    for flag in ("timed_out", "cancelled", "canceled", "skipped", "partial"):
        if _bool_value(receipt.get(flag)) is True:
            return False
    receipt_result: bool | None = None
    if "passed" in receipt:
        receipt_result = _bool_value(receipt["passed"])
    elif "returncode" in receipt:
        try:
            receipt_result = int(receipt["returncode"]) == 0 and attempted is not False
        except (TypeError, ValueError):
            receipt_result = False
    status = str(receipt.get("status", receipt.get("terminal_reason", ""))).strip().lower()
    if status in {"passed", "pass", "succeeded", "success", "verified", "ok"}:
        status_result: bool | None = True
    elif status in {
        "failed", "failure", "error", "timed_out", "timeout", "cancelled",
        "canceled", "skipped", "unattempted", "partial", "duplicate_only",
        "duplicate-only", "unsupported", "disabled", "cooldown",
    }:
        status_result = False
    else:
        status_result = None
    outcomes = [item for item in (declared, receipt_result, status_result) if item is not None]
    if False in outcomes:
        return False
    return True if True in outcomes else None


def _bool_value(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "yes", "1", "passed", "fresh"}:
        return True
    if normalized in {"false", "no", "0", "failed", "stale"}:
        return False
    return None


def _freshness_claim(evidence: CompletionEvidence) -> bool | None:
    value = evidence.freshness
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        for key in ("fresh", "is_fresh"):
            if key in value:
                return _bool_value(value[key])
        status = str(value.get("status", "")).strip().lower()
        if status:
            return status == "fresh"
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"fresh", "current", "true", "yes"}:
            return True
        if normalized in {"stale", "expired", "false", "no"}:
            return False
    return None


def validate_completion_evidence(
    evidence: CompletionEvidence | Mapping[str, Any],
    *,
    repository_tree: str = "",
    repository_id: str = "",
    now: datetime | str | None = None,
    freshness_seconds: float = DEFAULT_EVIDENCE_FRESHNESS_SECONDS,
    clock_skew_seconds: float = DEFAULT_CLOCK_SKEW_SECONDS,
) -> EvidenceValidationResult:
    """Validate one evidence record, collecting every actionable defect."""

    if not isinstance(evidence, CompletionEvidence):
        evidence = CompletionEvidence.from_dict(evidence)
    reason_codes: list[str] = []
    reasons: list[str] = []

    def reject(code: str, reason: str) -> None:
        if code not in reason_codes:
            reason_codes.append(code)
            reasons.append(reason)

    if not evidence.acceptance_criterion:
        reject("missing_acceptance_criterion", "Name the acceptance criterion this evidence proves.")
    if evidence.schema_version != GOAL_COMPLETION_SCHEMA_VERSION:
        reject("unsupported_schema_version", "Migrate the evidence record to the current completion schema.")
    if not evidence.producing_task_or_scan:
        reject("missing_producer", "Record the producing task or scan for this evidence.")
    if evidence.producer_kind not in {"task", "scan"}:
        reject("invalid_producer_kind", "Set producer_kind to 'task' or 'scan'.")
    if not evidence.validation_receipt:
        reject("missing_validation_receipt", "Attach the validation receipt for this criterion.")
    passed = _validation_passed(evidence)
    if passed is not True:
        code = "failed_validation" if passed is False else "unknown_validation_result"
        reject(code, "Run the criterion validation successfully and attach a passing receipt.")
    if not evidence.repository_tree:
        reject("missing_repository_tree", "Bind the evidence to the repository tree that was validated.")
    elif repository_tree and evidence.repository_tree != str(repository_tree):
        reject("repository_tree_mismatch", "Regenerate evidence against the current repository tree.")
    if repository_id and evidence.repository_id and evidence.repository_id != str(repository_id):
        reject("repository_mismatch", "Use evidence produced by the current repository.")
    if not evidence.provenance_cid:
        reject("missing_provenance_cid", "Persist the evidence and attach its provenance CID.")

    receipt = evidence.validation_receipt
    if isinstance(receipt, Mapping):
        receipt_tree = str(receipt.get("tree_id", receipt.get("tree_identity", "")) or "")
        if receipt_tree and evidence.repository_tree and receipt_tree != evidence.repository_tree:
            reject("validation_tree_mismatch", "Use a validation receipt for the same repository tree as the evidence.")

    current = _now(now)
    freshness_claim = _freshness_claim(evidence)
    if freshness_claim is not True:
        code = "stale_evidence" if freshness_claim is False else "missing_freshness"
        detail = "Evidence is stale; regenerate it and record a fresh status." if freshness_claim is False else "Refresh the evidence and record an explicit fresh status."
        reject(code, detail)
    observed = evidence.observed_at
    if observed is None:
        reject("missing_observed_at", "Record observed_at so evidence freshness can be checked.")
    else:
        if observed > current + timedelta(seconds=max(0.0, float(clock_skew_seconds))):
            reject("future_evidence", "Correct the evidence timestamp; it is in the future.")
        if current - observed > timedelta(seconds=max(0.0, float(freshness_seconds))):
            reject("stale_evidence", "Regenerate evidence because its freshness window has expired.")
    if evidence.fresh_until is not None and current > evidence.fresh_until:
        reject("stale_evidence", "Regenerate evidence because its declared freshness deadline passed.")
    if evidence.producer_kind == "scan":
        safe = evidence.metadata.get("safe_for_completion_reasoning")
        if safe is not True:
            reject("unsafe_scan_receipt", "Use an exhaustive scan receipt marked safe for completion reasoning.")
    if evidence.contradictory or evidence.contradiction.strip():
        detail = evidence.contradiction.strip()
        reject(
            "contradictory_evidence",
            "Resolve contradictory evidence before verification" + (f": {detail}" if detail else "."),
        )
    return EvidenceValidationResult(
        evidence=evidence,
        valid=not reason_codes,
        reason_codes=tuple(reason_codes),
        actionable_reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class GoalTransition:
    previous_state: GoalState
    state: GoalState
    reason: str
    transitioned_at: datetime
    evidence_cids: tuple[str, ...] = ()
    evidence: tuple[CompletionEvidence, ...] = ()

    @property
    def from_state(self) -> GoalState:
        return self.previous_state

    @property
    def to_state(self) -> GoalState:
        return self.state

    @property
    def next_state(self) -> GoalState:
        return self.state

    def to_dict(self) -> dict[str, Any]:
        return {
            "previous_state": self.previous_state.value,
            "from_state": self.previous_state.value,
            "state": self.state.value,
            "next_state": self.state.value,
            "to_state": self.state.value,
            "reason": self.reason,
            "transitioned_at": self.transitioned_at.isoformat(),
            "evidence_cids": list(self.evidence_cids),
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalTransition":
        return cls(
            previous_state=normalize_goal_state(payload.get("previous_state", payload.get("from_state"))),
            state=normalize_goal_state(payload.get("state", payload.get("to_state"))),
            reason=str(payload.get("reason", "")),
            transitioned_at=_utc_datetime(payload.get("transitioned_at"), field_name="transitioned_at") or datetime.now(timezone.utc),
            evidence_cids=tuple(str(item) for item in payload.get("evidence_cids", ()) if str(item)),
            evidence=tuple(CompletionEvidence.from_dict(item) for item in payload.get("evidence", ()) if isinstance(item, Mapping)),
        )


@dataclass
class GoalLifecycle:
    """Mutable state holder which enforces and audits lifecycle transitions."""

    goal_id: str = ""
    state: GoalState | str = GoalState.ACTIVE
    history: list[GoalTransition] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.state = normalize_goal_state(self.state)
        self.goal_id = str(self.goal_id or "").strip()

    def can_transition(self, target: GoalState | str) -> bool:
        return normalize_goal_state(target) in legal_goal_transitions(self.state)

    def transition(
        self,
        target: GoalState | str,
        *,
        reason: str,
        evidence: Iterable[CompletionEvidence | str] = (),
        now: datetime | str | None = None,
    ) -> GoalState:
        target_state = normalize_goal_state(target)
        if not self.can_transition(target_state):
            legal = ", ".join(item.value for item in sorted(legal_goal_transitions(self.state), key=lambda item: item.value))
            raise IllegalGoalTransitionError(
                f"illegal goal transition for {self.goal_id or '<unnamed goal>'}: "
                f"{self.state.value} -> {target_state.value}; legal targets: {legal or 'none'}"
            )
        explanation = str(reason or "").strip()
        if not explanation:
            raise ValueError("goal transitions require a non-empty reason")
        cids: list[str] = []
        records: list[CompletionEvidence] = []
        for item in evidence:
            cid = item.provenance_cid if isinstance(item, CompletionEvidence) else str(item or "").strip()
            if isinstance(item, CompletionEvidence):
                records.append(item)
            if cid and cid not in cids:
                cids.append(cid)
        transition = GoalTransition(
            previous_state=self.state,
            state=target_state,
            reason=explanation,
            transitioned_at=_now(now),
            evidence_cids=tuple(cids),
            evidence=tuple(records),
        )
        self.state = target_state
        self.history.append(transition)
        return self.state

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GOAL_COMPLETION_SCHEMA_VERSION,
            "goal_id": self.goal_id,
            "state": self.state.value,
            "history": [item.to_dict() for item in self.history],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GoalLifecycle":
        return cls(
            goal_id=str(payload.get("goal_id", "")),
            state=normalize_goal_state(payload.get("state")),
            history=[
                GoalTransition.from_dict(item)
                for item in payload.get("history", ())
                if isinstance(item, Mapping)
            ],
        )


@dataclass(frozen=True)
class GoalCompletionDecision:
    """Fail-closed result of evaluating all completion evidence for a goal."""

    previous_state: GoalState
    state: GoalState
    tasks_complete: bool
    verified: bool
    acceptance_criteria: tuple[str, ...]
    missing_criteria: tuple[str, ...]
    invalid_criteria: tuple[str, ...]
    evidence_results: tuple[EvidenceValidationResult, ...]
    reason_codes: tuple[str, ...]
    actionable_reasons: tuple[str, ...]
    gate: "CompletionGateResult | None" = None
    schema_version: int = GOAL_COMPLETION_SCHEMA_VERSION

    @property
    def next_state(self) -> GoalState:
        return self.state

    @property
    def provisional(self) -> bool:
        return self.state is GoalState.PROVISIONALLY_COMPLETE

    @property
    def can_verify(self) -> bool:
        return self.verified

    @property
    def evidence(self) -> list[CompletionEvidence]:
        return [result.evidence for result in self.evidence_results]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "previous_state": self.previous_state.value,
            "state": self.state.value,
            "next_state": self.state.value,
            "tasks_complete": self.tasks_complete,
            "verified": self.verified,
            "provisional": self.provisional,
            "acceptance_criteria": list(self.acceptance_criteria),
            "missing_criteria": list(self.missing_criteria),
            "invalid_criteria": list(self.invalid_criteria),
            "reason_codes": list(self.reason_codes),
            "actionable_reasons": list(self.actionable_reasons),
            "evidence": [item.to_dict() for item in self.evidence],
            "evidence_results": [result.to_dict() for result in self.evidence_results],
            "completion_gate": self.gate.to_dict() if self.gate is not None else None,
        }


def _mapping_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return {str(key): _json_value(item) for key, item in payload.items()}
    return {"value": _json_value(value)}


def _gate_datetime(value: Any) -> datetime | None:
    """Best-effort timestamp parsing for untrusted persisted gate proof."""

    try:
        return _utc_datetime(value, field_name="completion gate timestamp")
    except (TypeError, ValueError):
        return None


def _gate_count(value: Any) -> int | None:
    """Return a non-negative integer without accepting booleans or fractions."""

    if isinstance(value, bool):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        if float(value) != count:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return count if count >= 0 else None


@dataclass(frozen=True)
class CompletionGateCheck:
    """One independently inspectable fail-closed completion requirement."""

    name: str
    passed: bool
    reason_code: str = ""
    reason: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)
    pass_code: str = ""
    pass_reason: str = ""

    @property
    def outcome_code(self) -> str:
        return self.pass_code if self.passed else self.reason_code

    @property
    def outcome_reason(self) -> str:
        return self.pass_reason if self.passed else self.reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "outcome_code": self.outcome_code,
            "outcome_reason": self.outcome_reason,
            "evidence": _json_value(self.evidence),
        }


@dataclass(frozen=True)
class CompletionGateResult:
    """Machine-readable verdict and the exact evidence evaluated."""

    passed: bool
    checks: tuple[CompletionGateCheck, ...]
    evaluated_evidence: Mapping[str, Any]
    schema_version: int = GOAL_COMPLETION_SCHEMA_VERSION

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(check.reason_code for check in self.checks if check.reason_code))

    @property
    def actionable_reasons(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(check.reason for check in self.checks if check.reason))

    @property
    def pass_reason_codes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(check.pass_code for check in self.checks if check.passed and check.pass_code))

    @property
    def fail_reason_codes(self) -> tuple[str, ...]:
        return self.reason_codes

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ipfs_accelerate_py.agent_supervisor.completion_gate.v1",
            "schema_version": self.schema_version,
            "passed": self.passed,
            "reason_codes": list(self.reason_codes),
            "pass_reason_codes": list(self.pass_reason_codes),
            "fail_reason_codes": list(self.fail_reason_codes),
            "actionable_reasons": list(self.actionable_reasons),
            "checks": [check.to_dict() for check in self.checks],
            "evaluated_evidence": _json_value(self.evaluated_evidence),
        }


def evaluate_completion_gate(
    *,
    acceptance_criteria: Sequence[str] | str | None,
    evidence_results: Sequence[EvidenceValidationResult | Mapping[str, Any]] = (),
    coverage: Any = None,
    analyzer_health: Any = None,
    exhaustion_quorum: Any = None,
    child_goals: Sequence[Any] = (),
    analysis_result: Any = None,
    repository_tree: str = "",
    repository_id: str = "",
    now: datetime | str | None = None,
    freshness_seconds: float = DEFAULT_EVIDENCE_FRESHNESS_SECONDS,
    clock_skew_seconds: float = DEFAULT_CLOCK_SKEW_SECONDS,
) -> CompletionGateResult:
    """Evaluate coverage, validation, health, quorum, and descendants.

    Inputs accept typed domain records or their persisted mapping forms.  Any
    absent, unsupported, partial, stale, skipped, failed, timed-out, or
    duplicate-only proof fails closed.
    """

    criteria = _acceptance_criteria(acceptance_criteria)
    current = _now(now)
    max_age = timedelta(seconds=max(0.0, float(freshness_seconds)))
    clock_skew = timedelta(seconds=max(0.0, float(clock_skew_seconds)))
    coverage_payload = _mapping_value(coverage)
    health_payload = _mapping_value(analyzer_health)
    quorum_payload = _mapping_value(exhaustion_quorum)
    analysis_payload = _mapping_value(analysis_result)
    child_payloads = [_mapping_value(item) for item in child_goals]
    raw_result_payloads = [_mapping_value(item) for item in evidence_results]
    normalized_results: list[EvidenceValidationResult] = []
    for item in evidence_results:
        if isinstance(item, EvidenceValidationResult):
            normalized_results.append(item)
            continue
        source = item.get("evidence", item) if isinstance(item, Mapping) else {}
        try:
            record = CompletionEvidence.from_dict(source if isinstance(source, Mapping) else {})
            normalized_results.append(
                validate_completion_evidence(
                    record,
                    repository_tree=repository_tree,
                    repository_id=repository_id,
                    now=current,
                    freshness_seconds=freshness_seconds,
                    clock_skew_seconds=clock_skew_seconds,
                )
            )
        except (TypeError, ValueError) as exc:
            placeholder = CompletionEvidence(
                acceptance_criterion="",
                metadata={"malformed_evidence": _json_value(source), "error": str(exc)},
            )
            normalized_results.append(
                EvidenceValidationResult(
                    placeholder,
                    False,
                    ("malformed_evidence",),
                    ("Repair or regenerate the malformed persisted completion evidence.",),
                )
            )
    evidence_results = tuple(normalized_results)
    result_payloads = [item.to_dict() for item in evidence_results]
    checks: list[CompletionGateCheck] = []

    def add_check(
        name: str,
        passed: bool,
        fail_code: str,
        fail_reason: str,
        check_evidence: Mapping[str, Any],
        pass_code: str,
        pass_reason: str,
    ) -> None:
        checks.append(
            CompletionGateCheck(
                name=name,
                passed=passed,
                reason_code="" if passed else fail_code,
                reason="" if passed else fail_reason,
                evidence=check_evidence,
                pass_code=pass_code,
                pass_reason=pass_reason,
            )
        )

    coverage_rows = coverage_payload.get("criteria")
    if not isinstance(coverage_rows, list):
        coverage_rows = []
    by_criterion: dict[str, list[Mapping[str, Any]]] = {}
    for row in coverage_rows:
        if not isinstance(row, Mapping):
            continue
        # AcceptanceCoverage.to_dict() uses ``criterion``.  The aliases make
        # the gate tolerant of earlier graph projections without weakening
        # the required exact normalized-text match.
        criterion_text = row.get(
            "criterion",
            row.get("acceptance_criterion", row.get("acceptance", "")),
        )
        key = _criterion_key(criterion_text)
        if key:
            by_criterion.setdefault(key, []).append(row)
    missing_coverage = [criterion for criterion in criteria if _criterion_key(criterion) not in by_criterion]
    unverified_coverage = [
        criterion
        for criterion in criteria
        if _criterion_key(criterion) in by_criterion
        and not all(
            str(row.get("status") or "").strip().lower() == "verified"
            and ("verified" not in row or _bool_value(row.get("verified")) is True)
            for row in by_criterion[_criterion_key(criterion)]
        )
    ]
    coverage_tree = str(coverage_payload.get("repository_tree") or coverage_payload.get("tree_id") or "")
    coverage_tree_mismatch = bool(repository_tree and coverage_tree != repository_tree)
    coverage_declared_false = (
        "verified" in coverage_payload
        and _bool_value(coverage_payload.get("verified")) is not True
    )
    coverage_declared_reasons = coverage_payload.get("reason_codes")
    coverage_declared_reasons = (
        list(coverage_declared_reasons)
        if isinstance(coverage_declared_reasons, (list, tuple))
        else []
    )
    coverage_evaluated_at_value = coverage_payload.get("evaluated_at")
    coverage_evaluated_at = _gate_datetime(coverage_evaluated_at_value)
    has_coverage_time = coverage_evaluated_at_value is not None and coverage_evaluated_at_value != ""
    coverage_fresh = has_coverage_time
    coverage_freshness_error = ""
    if not has_coverage_time:
        coverage_freshness_error = "missing"
    else:
        if coverage_evaluated_at is None:
            coverage_fresh = False
            coverage_freshness_error = "malformed"
        elif coverage_evaluated_at > current + clock_skew:
            coverage_fresh = False
            coverage_freshness_error = "future"
        elif current - coverage_evaluated_at > max_age:
            coverage_fresh = False
            coverage_freshness_error = "stale"
    coverage_ok = bool(criteria) and not missing_coverage and not unverified_coverage
    coverage_ok = bool(coverage_ok and not coverage_declared_false and not coverage_declared_reasons)
    coverage_ok = bool(coverage_ok and not coverage_tree_mismatch and coverage_fresh)
    if not coverage_rows or missing_coverage:
        coverage_code = "coverage_missing"
    elif unverified_coverage or coverage_declared_false or coverage_declared_reasons:
        coverage_code = "coverage_unverified"
    elif coverage_tree_mismatch:
        coverage_code = "coverage_tree_mismatch"
    else:
        coverage_code = "coverage_stale"
    add_check(
        "mandatory_coverage",
        coverage_ok,
        coverage_code,
        "Map every mandatory acceptance criterion to fresh, verified implementation and validation proof bound to the current tree.",
        {
            "missing_criteria": missing_coverage,
            "unverified_criteria": unverified_coverage,
            "tree_mismatch": coverage_tree_mismatch,
            "freshness_error": coverage_freshness_error,
            "declared_verified": coverage_payload.get("verified"),
            "declared_reason_codes": coverage_declared_reasons,
            "coverage": coverage_payload,
        },
        "coverage_verified",
        "Every mandatory acceptance criterion has fresh, verified coverage.",
    )

    each_criterion_valid = bool(criteria) and all(
        any(
            item.valid and _criterion_key(item.evidence.acceptance_criterion) == _criterion_key(criterion)
            for item in evidence_results
        )
        for criterion in criteria
    )
    invalid_results = [item.to_dict() for item in evidence_results if not item.valid]
    # The exact submitted evidence set is authoritative.  A fresh receipt
    # cannot mask a stale, failed, or contradictory receipt submitted beside
    # it for the same decision.
    validation_ok = bool(each_criterion_valid and evidence_results and not invalid_results)
    add_check(
        "required_validations",
        validation_ok,
        "validation_evidence_incomplete",
        "Every submitted validation proof must be fresh and passing, and every mandatory criterion must have one.",
        {"results": result_payloads, "invalid_results": invalid_results},
        "validations_verified",
        "Every mandatory criterion has fresh, passing validation evidence.",
    )

    health_status = str(
        health_payload.get("status")
        or health_payload.get("health")
        or ("healthy" if health_payload.get("healthy") is True else "")
    ).strip().lower()
    health_declared = _bool_value(health_payload.get("healthy")) if "healthy" in health_payload else None
    health_safe = (
        _bool_value(health_payload.get("safe_for_completion_reasoning"))
        if "safe_for_completion_reasoning" in health_payload
        else None
    )
    health_ok = (
        health_status == "healthy"
        and ("healthy" not in health_payload or health_declared is True)
        and (
            "safe_for_completion_reasoning" not in health_payload
            or health_safe is True
        )
    )
    if not health_status:
        health_code = "analyzer_health_missing"
    elif health_status != "healthy" or health_declared is False:
        health_code = "analyzer_unhealthy"
    else:
        health_code = "analyzer_completion_unsafe"
    add_check(
        "analyzer_health",
        health_ok,
        health_code,
        "Require an explicitly healthy analyzer that is safe for completion reasoning.",
        health_payload,
        "analyzer_healthy",
        "Analyzer health is explicitly healthy and completion-safe.",
    )

    declared_values = [
        _bool_value(quorum_payload[key])
        for key in ("satisfied", "quorum_met")
        if key in quorum_payload
    ]
    declared_quorum = bool(declared_values) and all(item is True for item in declared_values)
    binding = quorum_payload.get("binding") if isinstance(quorum_payload.get("binding"), Mapping) else {}
    binding_mismatch: list[str] = []
    if quorum_payload and repository_tree and str(binding.get("tree_id") or "") != repository_tree:
        binding_mismatch.append("tree_id")
    if quorum_payload and repository_id and str(binding.get("repository_id") or "") != repository_id:
        binding_mismatch.append("repository_id")
    required_members = _gate_count(quorum_payload.get("required_members", quorum_payload.get("required")))
    members_present = "members" in quorum_payload or "eligible_members" in quorum_payload
    raw_members = quorum_payload.get("members", quorum_payload.get("eligible_members", ()))
    members = list(raw_members) if isinstance(raw_members, (list, tuple)) else []
    declared_member_count = _gate_count(quorum_payload.get("member_count", quorum_payload.get("count")))
    member_count = len(members) if declared_member_count is None and members_present else declared_member_count
    quorum_inconsistencies: list[str] = []
    if required_members is None or required_members < 1:
        quorum_inconsistencies.append("invalid_required_members")
    if member_count is None:
        quorum_inconsistencies.append("missing_member_count")
    if members_present and not isinstance(raw_members, (list, tuple)):
        quorum_inconsistencies.append("invalid_members")
    if not members:
        quorum_inconsistencies.append("missing_quorum_members")
    if members_present and declared_member_count is not None and declared_member_count != len(members):
        quorum_inconsistencies.append("member_count_mismatch")
    if required_members is not None and member_count is not None and member_count < required_members:
        quorum_inconsistencies.append("insufficient_members")
    member_ids: list[str] = []
    channels: list[str] = []
    stale_members: list[str] = []
    for index, member in enumerate(members):
        if not isinstance(member, Mapping):
            quorum_inconsistencies.append(f"invalid_member:{index}")
            continue
        member_id = str(member.get("member_id") or "").strip()
        channel = str(member.get("evidence_channel") or member.get("independence_key") or "").strip()
        receipt_cid = str(member.get("receipt_cid") or "").strip()
        if not member_id or not channel or not receipt_cid:
            quorum_inconsistencies.append(f"incomplete_member:{index}")
        member_ids.append(member_id)
        channels.append(channel)
        member_binding = member.get("binding") if isinstance(member.get("binding"), Mapping) else {}
        if not member_binding:
            quorum_inconsistencies.append(f"missing_member_binding:{index}")
        elif binding and any(
            str(member_binding.get(key) or "") != str(binding.get(key) or "")
            for key in ("repository_id", "tree_id", "analyzer_version", "configuration_revision", "objective_revision")
            if binding.get(key)
        ):
            quorum_inconsistencies.append(f"member_binding_mismatch:{index}")
        finished_value = member.get("finished_at")
        finished_at = _gate_datetime(finished_value)
        if finished_value is None or finished_value == "" or finished_at is None:
            stale_members.append(member_id or str(index))
        elif finished_at > current + clock_skew or current - finished_at > max_age:
            stale_members.append(member_id or str(index))
    if len(set(member_ids)) != len(member_ids):
        quorum_inconsistencies.append("duplicate_member_id")
    if len(set(channels)) != len(channels):
        quorum_inconsistencies.append("duplicate_evidence_channel")
    if stale_members:
        quorum_inconsistencies.append("stale_quorum_members")
    quorum_ok = bool(
        quorum_payload
        and declared_quorum
        and not binding_mismatch
        and not quorum_inconsistencies
    )
    if not quorum_payload:
        quorum_code = "exhaustion_quorum_missing"
    elif declared_quorum and binding_mismatch:
        quorum_code = "exhaustion_quorum_binding_mismatch"
    elif declared_quorum and quorum_inconsistencies:
        quorum_code = "exhaustion_quorum_inconsistent"
    else:
        quorum_code = "exhaustion_quorum_unsatisfied"
    add_check(
        "exhaustion_quorum",
        quorum_ok,
        quorum_code,
        "Require the configured number of independent, fresh, healthy exhaustive receipts bound to the current repository tree.",
        {
            "binding_mismatch": binding_mismatch,
            "inconsistencies": quorum_inconsistencies,
            "stale_members": stale_members,
            "quorum": quorum_payload,
        },
        "exhaustion_quorum_satisfied",
        "The configured exhaustion quorum is satisfied by independent proof.",
    )

    unsafe_analysis = False
    effective_analysis = analysis_payload
    if isinstance(analysis_payload.get("receipt"), Mapping):
        effective_analysis = dict(analysis_payload["receipt"])
    terminal = ""
    safe: bool | None = None
    if effective_analysis:
        terminal = str(effective_analysis.get("terminal_reason") or effective_analysis.get("status") or "").strip().lower().replace("-", "_")
        safe = _bool_value(effective_analysis.get("safe_for_completion_reasoning"))
        unsafe_analysis = safe is not True or terminal not in {"exhausted", "healthy_exhausted"}
    add_check(
        "analysis_terminal_state",
        not unsafe_analysis,
        "analysis_not_completion_safe",
        "Analysis was partial, skipped, failed, timed out, duplicate-only, unsupported, or not explicitly completion-safe.",
        {"analysis": analysis_payload, "effective_terminal_reason": terminal, "effective_safe": safe},
        "analysis_completion_safe" if effective_analysis else "analysis_not_supplied",
        "Analysis proof is exhausted and completion-safe." if effective_analysis else "No optional standalone analysis result was supplied; quorum proof remains authoritative.",
    )

    bad_children: list[dict[str, Any]] = []
    descendants: list[dict[str, Any]] = []

    def collect_descendants(values: Sequence[dict[str, Any]]) -> None:
        for child in values:
            descendants.append(child)
            gate_value = child.get("completion_gate", child.get("gate"))
            gate_value = gate_value if isinstance(gate_value, Mapping) else {}
            evaluated_value = gate_value.get("evaluated_evidence")
            evaluated_value = evaluated_value if isinstance(evaluated_value, Mapping) else {}
            nested = evaluated_value.get("child_goals", child.get("child_goals", ()))
            if isinstance(nested, (list, tuple)):
                collect_descendants([_mapping_value(item) for item in nested])

    collect_descendants(child_payloads)
    for child in descendants:
        state = str(child.get("state") or child.get("next_state") or "").lower()
        gate_value = child.get("completion_gate", child.get("gate"))
        gate_value = gate_value if isinstance(gate_value, Mapping) else {}
        verified = child.get("verified") is True and state == GoalState.VERIFIED_COMPLETE.value
        if not verified or gate_value.get("passed") is False:
            bad_children.append(child)
    child_ok = not bad_children
    child_code = ""
    if bad_children:
        states = {str(item.get("state") or item.get("next_state") or "").lower() for item in bad_children}
        child_code = "child_analysis_inconclusive" if GoalState.ANALYSIS_INCONCLUSIVE.value in states else (
            "child_reopened" if GoalState.REOPENED.value in states else "child_unverified"
        )
    add_check(
        "child_goals",
        child_ok,
        child_code,
        "Every descendant must remain verified; parent aggregation cannot hide an inconclusive or reopened child.",
        {"children": child_payloads, "descendants": descendants, "unverified_children": bad_children},
        "descendants_verified",
        "Every supplied child and descendant remains verified.",
    )

    evaluated = {
        "acceptance_criteria": list(criteria),
        "coverage": coverage_payload,
        "validation_evidence": raw_result_payloads,
        "analyzer_health": health_payload,
        "exhaustion_quorum": quorum_payload,
        "analysis_result": analysis_payload,
        "child_goals": child_payloads,
        "repository_tree": repository_tree,
        "repository_id": repository_id,
        "evaluated_at": current.isoformat(),
        "freshness_seconds": max(0.0, float(freshness_seconds)),
    }
    return CompletionGateResult(all(check.passed for check in checks), tuple(checks), evaluated)


def _acceptance_criteria(values: Sequence[str] | str | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        # Acceptance fields conventionally delimit independent criteria with
        # semicolons.  Do not split commas: prose commonly contains them.
        source = values.split(";")
    else:
        source = values
    result: list[str] = []
    seen: set[str] = set()
    for value in source:
        criterion = str(value or "").strip()
        key = _criterion_key(criterion)
        if criterion and key not in seen:
            seen.add(key)
            result.append(criterion)
    return tuple(result)


def evaluate_goal_completion(
    *,
    current_state: GoalState | str = GoalState.ACTIVE,
    acceptance_criteria: Sequence[str] | str | None = None,
    evidence: Sequence[CompletionEvidence | Mapping[str, Any]] = (),
    tasks_complete: bool = False,
    repository_tree: str = "",
    repository_id: str = "",
    now: datetime | str | None = None,
    freshness_seconds: float = DEFAULT_EVIDENCE_FRESHNESS_SECONDS,
    clock_skew_seconds: float = DEFAULT_CLOCK_SKEW_SECONDS,
    analysis_inconclusive: bool = False,
    blocked_reason: str = "",
    coverage: Any = None,
    analyzer_health: Any = None,
    exhaustion_quorum: Any = None,
    child_goals: Sequence[Any] = (),
    analysis_result: Any = None,
    require_completion_gate: bool = True,
) -> GoalCompletionDecision:
    """Evaluate a goal without ever inferring verification from task status.

    Verification is only reachable from ``provisionally_complete``.  This
    makes the two-phase contract observable and prevents a single task status
    update from atomically manufacturing verified completion.
    """

    previous = normalize_goal_state(current_state)
    records = [item if isinstance(item, CompletionEvidence) else CompletionEvidence.from_dict(item) for item in evidence]
    criteria = _acceptance_criteria(acceptance_criteria)
    if acceptance_criteria is None and records:
        criteria = _acceptance_criteria([item.acceptance_criterion for item in records])
    results = tuple(
        validate_completion_evidence(
            item,
            repository_tree=repository_tree,
            repository_id=repository_id,
            now=now,
            freshness_seconds=freshness_seconds,
            clock_skew_seconds=clock_skew_seconds,
        )
        for item in records
    )
    by_criterion: dict[str, list[EvidenceValidationResult]] = {}
    for result in results:
        by_criterion.setdefault(_criterion_key(result.evidence.acceptance_criterion), []).append(result)

    missing: list[str] = []
    invalid: list[str] = []
    reason_codes: list[str] = []
    reasons: list[str] = []

    def add_reason(code: str, reason: str) -> None:
        if code not in reason_codes:
            reason_codes.append(code)
            reasons.append(reason)

    # Preserve defects on unbound records too (for example a record missing
    # its acceptance-criterion name).  Otherwise the caller would only see a
    # generic missing-criterion message and could not repair the source data.
    for result in results:
        for code, reason in zip(result.reason_codes, result.actionable_reasons):
            add_reason(code, reason)

    if not criteria:
        add_reason("missing_acceptance_criteria", "Define explicit acceptance criteria before verifying the goal.")
    for criterion in criteria:
        candidates = by_criterion.get(_criterion_key(criterion), [])
        if not candidates:
            missing.append(criterion)
            add_reason("missing_criterion_evidence", f"Produce completion evidence for: {criterion}")
            continue
        # Contradiction always fails closed, even if another record is valid.
        contradicted = any("contradictory_evidence" in item.reason_codes for item in candidates)
        if contradicted or not any(item.valid for item in candidates):
            invalid.append(criterion)
            for item in candidates:
                for code, reason in zip(item.reason_codes, item.actionable_reasons):
                    add_reason(code, f"{criterion}: {reason}")

    gate = evaluate_completion_gate(
        acceptance_criteria=criteria,
        evidence_results=results,
        coverage=coverage,
        analyzer_health=analyzer_health,
        exhaustion_quorum=exhaustion_quorum,
        child_goals=child_goals,
        analysis_result=analysis_result,
        repository_tree=repository_tree,
        repository_id=repository_id,
        now=now,
        freshness_seconds=freshness_seconds,
        clock_skew_seconds=clock_skew_seconds,
    ) if require_completion_gate else None
    if gate is not None:
        for check in gate.checks:
            if not check.passed and check.reason_code:
                add_reason(check.reason_code, check.reason)
    all_valid = bool(criteria) and not missing and not invalid and (gate is None or gate.passed)
    verified = bool(tasks_complete and all_valid and previous is GoalState.PROVISIONALLY_COMPLETE)

    if blocked_reason:
        next_state = GoalState.REOPENED if previous is GoalState.VERIFIED_COMPLETE else GoalState.BLOCKED
        add_reason("goal_blocked", str(blocked_reason).strip())
        if previous is GoalState.VERIFIED_COMPLETE:
            add_reason("verification_invalidated", "Reopen the verified goal before recording its blocker.")
    elif analysis_inconclusive:
        if previous is GoalState.VERIFIED_COMPLETE:
            next_state = GoalState.REOPENED
            add_reason("verification_invalidated", "Reopen the verified goal because completion analysis became inconclusive.")
        elif previous is GoalState.BLOCKED:
            next_state = GoalState.BLOCKED
        else:
            next_state = GoalState.ANALYSIS_INCONCLUSIVE
        add_reason("analysis_inconclusive", "Analysis was inconclusive; run the indicated fallback before claiming completion.")
    elif previous is GoalState.VERIFIED_COMPLETE and (not tasks_complete or not all_valid):
        next_state = GoalState.REOPENED
        add_reason("verification_invalidated", "Reopen the goal because its completion evidence is no longer valid.")
    elif verified:
        next_state = GoalState.VERIFIED_COMPLETE
    elif tasks_complete and previous in {
        GoalState.ACTIVE,
        GoalState.REOPENED,
        GoalState.ANALYSIS_INCONCLUSIVE,
        GoalState.PROVISIONALLY_COMPLETE,
    }:
        next_state = GoalState.PROVISIONALLY_COMPLETE
        if all_valid and previous is not GoalState.PROVISIONALLY_COMPLETE:
            add_reason("provisional_transition_required", "Record provisional completion, then verify in a separate lifecycle transition.")
        elif not all_valid:
            add_reason("verification_evidence_incomplete", "Task completion is provisional until every criterion has valid evidence.")
    elif previous in {GoalState.PROVISIONALLY_COMPLETE, GoalState.REOPENED} and not tasks_complete:
        next_state = GoalState.REOPENED
        add_reason("tasks_incomplete", "Complete the remaining producing tasks before requesting verification.")
    else:
        next_state = previous
        if not tasks_complete:
            add_reason("tasks_incomplete", "Complete the goal's producing tasks before requesting completion.")

    return GoalCompletionDecision(
        previous_state=previous,
        state=next_state,
        tasks_complete=bool(tasks_complete),
        verified=verified,
        acceptance_criteria=criteria,
        missing_criteria=tuple(missing),
        invalid_criteria=tuple(invalid),
        evidence_results=results,
        reason_codes=tuple(reason_codes),
        actionable_reasons=tuple(reasons),
        gate=gate,
    )


# Names used by older prototypes and external consumers.
GoalLifecycleState = GoalState
CompletionDecision = GoalCompletionDecision
assess_goal_completion = evaluate_goal_completion


__all__ = [
    "CompletionDecision",
    "CompletionGateCheck",
    "CompletionGateResult",
    "CompletionEvidence",
    "DEFAULT_CLOCK_SKEW_SECONDS",
    "DEFAULT_EVIDENCE_FRESHNESS_SECONDS",
    "EvidenceValidationResult",
    "GOAL_COMPLETION_SCHEMA_VERSION",
    "GoalCompletionDecision",
    "GoalLifecycle",
    "GoalLifecycleState",
    "GoalState",
    "GoalTransition",
    "IllegalGoalTransition",
    "IllegalGoalTransitionError",
    "LEGAL_GOAL_TRANSITIONS",
    "assess_goal_completion",
    "evaluate_goal_completion",
    "evaluate_completion_gate",
    "is_schedulable_goal_state",
    "is_terminal_goal_state",
    "legal_goal_transitions",
    "normalize_goal_state",
    "validate_completion_evidence",
]
