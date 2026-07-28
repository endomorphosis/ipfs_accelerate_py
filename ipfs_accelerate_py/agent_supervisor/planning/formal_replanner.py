"""Bounded counterexample-guided repairs for canonical formal work plans.

The replanner is intentionally a deterministic transformation layer.  It does
not ask a language model to rediscover repository context.  A minimized
``FormalCounterexample`` selects one of a small set of reviewed repair rules,
the rule edits only bound source records, and the resulting source is compiled
and checked by the normal formal-plan compiler and validator before it can be
offered to a taskboard admission callback.

Only :class:`CodexRepairPacket` is model-facing.  It contains the selected
transition and the already redacted, byte-bounded counterexample capsule; it
never contains the source snapshot, rejected candidates, compiler diagnostics,
or validator traces.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, ClassVar, Final

from ..proof.formal_counterexamples import (
    CounterexampleContextCapsule,
    CounterexampleLimits,
    CounterexampleValidationError,
    FormalCounterexample,
    RepairClass,
    build_counterexample_context_capsule,
)
from .formal_plan_compiler import (
    CompilationStatus,
    FormalPlanCompiler,
    PlanCompilationResult,
)
from .formal_plan_validator import (
    FormalPlanValidator,
    PlanValidationResult,
    PlanValidationStatus,
)
from ..proof.formal_verification_contracts import (
    CanonicalContract,
    canonical_json,
    content_identity,
)
from plan_failure_memory import (
    DELTA_REPLAN_REQUIREMENT_ID,
    BranchFailureObservation,
    FailureMemoryDecision,
    FailureMemoryDisposition,
    FailureMemoryScope,
    PlanFailureMemory,
    PlanFailureMemoryError,
)


FORMAL_REPLANNER_VERSION: Final = 1
REPAIR_TRANSITION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-repair-transition@1"
)
REPAIR_CANDIDATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-repair-candidate@1"
)
REPLAN_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-replan-result@1"
)
CODEX_REPAIR_PACKET_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/codex-repair-packet@1"
)
RESPONSIVE_REPLAN_DECISION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/responsive-replan-decision@1"
)
DIAGNOSTIC_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/retry-diagnostic-receipt@1"
)
DELTA_PLAN_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/delta-plan-snapshot@1"
)
DELTA_REPLAN_DECISION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/delta-replan-decision@1"
)
# Objective-heap evidence identity for one bounded changed-evidence refinement.
BOUNDED_REFINEMENT_EVIDENCE_ID: Final = (
    "003778425160038348524906247302938706902"
)
UNCHANGED_FAILURE_BACKOFF_EVIDENCE_ID: Final = (
    "312819945606360295782005228058369235550"
)
# These are deliberately absent from ResponsiveReplanDecision.  Naming the
# roles makes the trust boundary machine-readable: deterministic routing
# metadata cannot be submitted as completion analyzer health, completion
# criterion coverage, or completion exhaustion quorum evidence.
OBJECTIVE_COMPLETION_EVIDENCE_ROLES: Final[tuple[str, ...]] = (
    "completion_analyzer_health",
    "completion_criterion_coverage",
    "completion_exhaustion_quorum",
)
RESPONSIVE_REPLAN_SIGNAL_KINDS: Final[frozenset[str]] = frozenset(
    {
        "counterexample",
        "stale_evidence",
        "repeated_failure",
        "capability_change",
        "interface_change",
        "scope_change",
        "scope_conflict",
        "resource_change",
        "resource_infeasible",
    }
)


class ReplannerValidationError(ValueError):
    """Raised when a repair request violates the replanner contract."""


class ReplanCancelled(ReplannerValidationError):
    """A cooperative cancellation stopped repair before admission."""


def _cancelled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if callable(value):
        return bool(value())
    checker = getattr(value, "is_set", None)
    if callable(checker):
        return bool(checker())
    raise ReplannerValidationError(
        "cancelled must be a boolean, predicate, event, or None"
    )


class RepairRuleKind(str, Enum):
    """Reviewed source transformations corresponding to repair classes."""

    ADD_DEPENDENCY = RepairClass.ADD_DEPENDENCY.value
    SPLIT_EFFECTS = RepairClass.SPLIT_TASK.value
    TIGHTEN_AUTHORITY = RepairClass.TIGHTEN_AUTHORITY.value
    ADD_EVIDENCE = RepairClass.ADD_OBLIGATION.value
    CONSTRAIN_SCOPE = RepairClass.CONSTRAIN_SCOPE.value
    ADD_PREMISE = RepairClass.ADD_PREMISE.value
    CHANGE_RESOURCE_BOUNDS = RepairClass.ADJUST_RESOURCES.value
    HUMAN_REVIEW = RepairClass.HUMAN_REVIEW.value


RepairKind = RepairRuleKind


class RepairCandidateStatus(str, Enum):
    GENERATED = "generated"
    DUPLICATE = "duplicate"
    COMPILE_REJECTED = "compile_rejected"
    GOAL_REJECTED = "goal_rejected"
    CHECK_REJECTED = "check_rejected"
    COUNTEREXAMPLE_REJECTED = "counterexample_rejected"
    NO_PROGRESS = "no_progress"
    ADMISSIBLE = "admissible"
    ADMITTED = "admitted"
    ADMISSION_REJECTED = "admission_rejected"


class ReplanStopReason(str, Enum):
    ADMITTED = "admitted"
    NO_ADMISSIBLE_REPAIR = "no_admissible_repair"
    RETRY_BUDGET_EXHAUSTED = "retry_budget_exhausted"
    REFINEMENT_DEPTH_EXHAUSTED = "refinement_depth_exhausted"
    CANDIDATE_BUDGET_EXHAUSTED = "candidate_budget_exhausted"
    COUNTEREXAMPLE_PLAN_MISMATCH = "counterexample_plan_mismatch"
    ORIGINAL_PLAN_INVALID = "original_plan_invalid"
    UNCHANGED_COUNTEREXAMPLE_BACKOFF = "unchanged_counterexample_backoff"
    IDENTICAL_FAILURE_ESCALATED = "identical_failure_escalated"
    CANCELLED = "cancelled"


def _positive(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReplannerValidationError(f"{name} must be an integer of at least {minimum}")
    return value


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    values: Iterable[Any]
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        values = (value,)
    return tuple(sorted({str(item).strip() for item in values if str(item).strip()}))


def _public_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Defensively copy a canonical-JSON mapping."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ReplannerValidationError("details must be an object")
    try:
        decoded = copy.deepcopy(dict(value))
        # canonical_json performs the repository's strict JSON validation.
        canonical_json(decoded)
    except (TypeError, ValueError) as exc:
        raise ReplannerValidationError(f"details are not canonical JSON: {exc}") from exc
    return decoded


@dataclass(frozen=True)
class ReplanLimits:
    """Finite generation, refinement, retry, and prompt budgets."""

    max_candidates: int = 16
    max_candidates_per_rule: int = 3
    max_retry_attempts: int = 3
    max_refinement_depth: int = 3
    max_changed_records: int = 8
    max_generated_tasks: int = 2
    max_capsule_bytes: int = 16_384
    max_prompt_bytes: int = 24_576
    max_prompt_tokens: int = 6_144

    def __post_init__(self) -> None:
        for name in (
            "max_candidates",
            "max_candidates_per_rule",
            "max_retry_attempts",
            "max_refinement_depth",
            "max_changed_records",
            "max_generated_tasks",
        ):
            _positive(getattr(self, name), name)
        _positive(self.max_capsule_bytes, "max_capsule_bytes", minimum=1024)
        _positive(self.max_prompt_bytes, "max_prompt_bytes", minimum=1024)
        _positive(self.max_prompt_tokens, "max_prompt_tokens", minimum=256)
        if self.max_capsule_bytes > self.max_prompt_bytes:
            raise ReplannerValidationError(
                "max_capsule_bytes cannot exceed max_prompt_bytes"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


ReplanBudget = ReplanLimits


@dataclass(frozen=True)
class RepairOperation:
    """One typed, canonical repair rule invocation.

    ``parameters`` is not an open-ended patch language.  The implementation
    below consumes a fixed field set for each enum member and rejects missing
    required fields.  Keeping one canonical envelope makes identities and
    persistence stable while retaining typed rule dispatch.
    """

    kind: RepairRuleKind
    target_task_id: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    counterexample_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", RepairRuleKind(self.kind))
        target = str(self.target_task_id or "").strip()
        if not target:
            raise ReplannerValidationError("target_task_id is required")
        object.__setattr__(self, "target_task_id", target)
        params = _public_mapping(self.parameters)
        object.__setattr__(self, "parameters", params)
        object.__setattr__(
            self, "counterexample_id", str(self.counterexample_id or "").strip()
        )
        required = {
            RepairRuleKind.ADD_DEPENDENCY: ("dependency_task_id",),
            RepairRuleKind.SPLIT_EFFECTS: ("split_index", "generated_task_id"),
            RepairRuleKind.TIGHTEN_AUTHORITY: ("actor_ids", "fencing_token"),
            RepairRuleKind.ADD_EVIDENCE: ("evidence_kind", "check_ids"),
            RepairRuleKind.CONSTRAIN_SCOPE: ("scope_ids",),
            RepairRuleKind.ADD_PREMISE: ("premise_ids",),
            RepairRuleKind.CHANGE_RESOURCE_BOUNDS: ("resource_bounds",),
            RepairRuleKind.HUMAN_REVIEW: ("reviewer_actor_id", "scope_ids"),
        }[self.kind]
        missing = [name for name in required if params.get(name) in (None, "", [], {})]
        if missing:
            raise ReplannerValidationError(
                f"{self.kind.value} requires {', '.join(missing)}"
            )

    @property
    def semantic_id(self) -> str:
        return content_identity(
            {
                "kind": self.kind.value,
                "target_task_id": self.target_task_id,
                "parameters": self.parameters,
                "counterexample_id": self.counterexample_id,
            }
        )

    @property
    def repair_id(self) -> str:
        return self.semantic_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "repair_id": self.semantic_id,
            "kind": self.kind.value,
            "target_task_id": self.target_task_id,
            "parameters": dict(self.parameters),
            "counterexample_id": self.counterexample_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepairOperation":
        result = cls(
            kind=payload.get("kind", RepairRuleKind.HUMAN_REVIEW),
            target_task_id=str(payload.get("target_task_id") or ""),
            parameters=payload.get("parameters") or {},
            counterexample_id=str(payload.get("counterexample_id") or ""),
        )
        claimed = payload.get("repair_id") or payload.get("semantic_id")
        if claimed and claimed != result.semantic_id:
            raise ReplannerValidationError("repair semantic identity does not match")
        return result


RepairRule = RepairOperation


@dataclass(frozen=True)
class RepairProgress:
    """Lexicographic measure proving that a repair step moves forward."""

    before_open_counterexamples: int
    after_open_counterexamples: int
    before_validation_findings: int
    after_validation_findings: int
    changed_records: int
    generated_tasks: int = 0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ReplannerValidationError(f"{name} must be non-negative")

    @property
    def before(self) -> tuple[int, int]:
        return (
            self.before_open_counterexamples,
            self.before_validation_findings,
        )

    @property
    def after(self) -> tuple[int, int]:
        return (
            self.after_open_counterexamples,
            self.after_validation_findings,
        )

    @property
    def improved(self) -> bool:
        return self.after < self.before

    def to_dict(self) -> dict[str, Any]:
        return {
            "before": {
                "open_counterexamples": self.before_open_counterexamples,
                "validation_findings": self.before_validation_findings,
            },
            "after": {
                "open_counterexamples": self.after_open_counterexamples,
                "validation_findings": self.after_validation_findings,
            },
            "changed_records": self.changed_records,
            "generated_tasks": self.generated_tasks,
            "improved": self.improved,
        }


@dataclass(frozen=True)
class RepairTransition:
    """The compact transition admitted to a taskboard and shown to Codex."""

    original_plan_id: str
    repaired_plan_id: str
    counterexample_id: str
    repair: RepairOperation
    goal_ids: tuple[str, ...]
    taskboard_records: tuple[Mapping[str, Any], ...]
    refinement_depth: int
    progress: RepairProgress

    def __post_init__(self) -> None:
        for name in ("original_plan_id", "repaired_plan_id", "counterexample_id"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ReplannerValidationError(f"{name} is required")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "goal_ids", _strings(self.goal_ids))
        _positive(self.refinement_depth, "refinement_depth")
        records = tuple(_public_mapping(item) for item in self.taskboard_records)
        if not records:
            raise ReplannerValidationError("taskboard_records must not be empty")
        object.__setattr__(self, "taskboard_records", records)
        if not isinstance(self.progress, RepairProgress):
            raise ReplannerValidationError("progress must be RepairProgress")

    @property
    def transition_id(self) -> str:
        return content_identity(self.to_dict(include_schema=False))

    @property
    def semantic_id(self) -> str:
        return self.transition_id

    def to_dict(self, *, include_schema: bool = True) -> dict[str, Any]:
        """Serialize versioned transition provenance for snapshot binding.

        An adaptive-plan candidate incorporates this complete value into its
        canonical snapshot, so changing a formal plan or repair transition
        invalidates every prior hard-gate receipt. The transition is still
        runtime provenance rather than criterion validation, analyzer health,
        or exhaustion-quorum evidence for objective completion.
        """

        value = {
            "replanner_version": FORMAL_REPLANNER_VERSION,
            "transition_id": self.transition_id if include_schema else "",
            "original_plan_id": self.original_plan_id,
            "repaired_plan_id": self.repaired_plan_id,
            "counterexample_id": self.counterexample_id,
            "repair": self.repair.to_dict(),
            "goal_ids": list(self.goal_ids),
            "taskboard_records": [dict(item) for item in self.taskboard_records],
            "refinement_depth": self.refinement_depth,
            "progress": self.progress.to_dict(),
        }
        if include_schema:
            value["schema"] = REPAIR_TRANSITION_SCHEMA
        else:
            value.pop("transition_id")
        return value

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepairTransition":
        allowed = {
            "schema",
            "replanner_version",
            "transition_id",
            "semantic_id",
            "original_plan_id",
            "repaired_plan_id",
            "counterexample_id",
            "repair",
            "goal_ids",
            "taskboard_records",
            "refinement_depth",
            "progress",
        }
        unknown = sorted(str(key) for key in payload if key not in allowed)
        if unknown:
            raise ReplannerValidationError(
                "unknown repair transition fields: " + ", ".join(unknown)
            )
        if payload.get("schema") != REPAIR_TRANSITION_SCHEMA:
            raise ReplannerValidationError(
                "unsupported repair transition schema"
            )
        if payload.get("replanner_version") != FORMAL_REPLANNER_VERSION:
            raise ReplannerValidationError(
                "unsupported formal replanner version"
            )
        progress = payload.get("progress") or {}
        before = progress.get("before") or {}
        after = progress.get("after") or {}
        progress_unknown = sorted(
            str(key)
            for key in progress
            if key
            not in {
                "before",
                "after",
                "changed_records",
                "generated_tasks",
                "improved",
            }
        )
        before_unknown = sorted(
            str(key)
            for key in before
            if key not in {"open_counterexamples", "validation_findings"}
        )
        after_unknown = sorted(
            str(key)
            for key in after
            if key not in {"open_counterexamples", "validation_findings"}
        )
        if progress_unknown or before_unknown or after_unknown:
            raise ReplannerValidationError(
                "unknown repair transition progress fields"
            )
        result = cls(
            original_plan_id=str(payload.get("original_plan_id") or ""),
            repaired_plan_id=str(payload.get("repaired_plan_id") or ""),
            counterexample_id=str(payload.get("counterexample_id") or ""),
            repair=RepairOperation.from_dict(payload.get("repair") or {}),
            goal_ids=tuple(payload.get("goal_ids") or ()),
            taskboard_records=tuple(payload.get("taskboard_records") or ()),
            refinement_depth=payload.get("refinement_depth", 0),
            progress=RepairProgress(
                before_open_counterexamples=before.get("open_counterexamples", 0),
                after_open_counterexamples=after.get("open_counterexamples", 0),
                before_validation_findings=before.get("validation_findings", 0),
                after_validation_findings=after.get("validation_findings", 0),
                changed_records=progress.get("changed_records", 0),
                generated_tasks=progress.get("generated_tasks", 0),
            ),
        )
        if progress.get("improved") is not result.progress.improved:
            raise ReplannerValidationError(
                "repair transition progress projection is inconsistent"
            )
        claimed = payload.get("transition_id") or payload.get("semantic_id")
        if not claimed:
            raise ReplannerValidationError(
                "repair transition identity is required"
            )
        if claimed != result.transition_id:
            raise ReplannerValidationError("repair transition identity does not match")
        return result


@dataclass(frozen=True)
class RepairCandidate:
    repair: RepairOperation
    status: RepairCandidateStatus
    compilation: PlanCompilationResult | None = None
    validation: PlanValidationResult | None = None
    transition: RepairTransition | None = None
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", RepairCandidateStatus(self.status))
        object.__setattr__(
            self,
            "rejection_reasons",
            tuple(str(item).strip() for item in self.rejection_reasons if str(item).strip()),
        )
        if self.status in {
            RepairCandidateStatus.ADMISSIBLE,
            RepairCandidateStatus.ADMITTED,
        } and (
            self.compilation is None
            or self.validation is None
            or self.transition is None
            or not self.validation.consistent
        ):
            raise ReplannerValidationError(
                "admissible repairs require a compiled, consistent transition"
            )

    @property
    def candidate_id(self) -> str:
        return content_identity(
            {
                "repair_id": self.repair.semantic_id,
                "compilation_source_identity": (
                    self.compilation.source_identity if self.compilation else ""
                ),
                "counterexample_id": self.repair.counterexample_id,
            }
        )

    @property
    def admissible(self) -> bool:
        return self.status in {
            RepairCandidateStatus.ADMISSIBLE,
            RepairCandidateStatus.ADMITTED,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPAIR_CANDIDATE_SCHEMA,
            "candidate_id": self.candidate_id,
            "repair": self.repair.to_dict(),
            "status": self.status.value,
            "compilation_status": (
                self.compilation.status.value if self.compilation else None
            ),
            "validation_status": (
                self.validation.status.value if self.validation else None
            ),
            "transition_id": (
                self.transition.transition_id if self.transition else None
            ),
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True)
class CodexRepairPacket:
    """The sole bounded projection intended for a Codex repair prompt."""

    transition: RepairTransition
    counterexample_capsule: CounterexampleContextCapsule
    max_bytes: int
    max_tokens: int

    def __post_init__(self) -> None:
        _positive(self.max_bytes, "max_bytes", minimum=1024)
        _positive(self.max_tokens, "max_tokens", minimum=256)
        ids = {
            str(item.get("counterexample_id") or "")
            for item in self.counterexample_capsule.counterexamples
        }
        if ids != {self.transition.counterexample_id}:
            raise ReplannerValidationError(
                "Codex packet must contain exactly the selected counterexample"
            )
        if self.byte_size > self.max_bytes:
            raise ReplannerValidationError("Codex repair packet exceeds max_bytes")
        if self.estimated_tokens > self.max_tokens:
            raise ReplannerValidationError("Codex repair packet exceeds max_tokens")

    @property
    def byte_size(self) -> int:
        return len(canonical_json(self.to_dict()).encode("utf-8"))

    @property
    def estimated_tokens(self) -> int:
        # A conservative deterministic bound; no tokenizer dependency is needed.
        return (self.byte_size + 2) // 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CODEX_REPAIR_PACKET_SCHEMA,
            "replanner_version": FORMAL_REPLANNER_VERSION,
            "transition": self.transition.to_dict(),
            "counterexample_capsule": self.counterexample_capsule.to_dict(),
            "limits": {
                "max_bytes": self.max_bytes,
                "max_tokens": self.max_tokens,
            },
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


@dataclass(frozen=True)
class ReplanResult:
    original_compilation: PlanCompilationResult
    original_validation: PlanValidationResult | None
    counterexample_id: str
    candidates: tuple[RepairCandidate, ...]
    selected: RepairCandidate | None
    codex_packet: CodexRepairPacket | None
    stop_reason: ReplanStopReason
    retry_attempt: int
    refinement_depth: int

    @property
    def admitted(self) -> bool:
        return (
            self.selected is not None
            and self.selected.status is RepairCandidateStatus.ADMITTED
        )

    @property
    def selected_transition(self) -> RepairTransition | None:
        return self.selected.transition if self.selected else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPLAN_RESULT_SCHEMA,
            "replanner_version": FORMAL_REPLANNER_VERSION,
            "counterexample_id": self.counterexample_id,
            "original_plan_id": self.original_compilation.plan_id,
            "original_validation_status": (
                self.original_validation.status.value
                if self.original_validation
                else None
            ),
            "candidates": [item.to_dict() for item in self.candidates],
            "selected_candidate_id": (
                self.selected.candidate_id if self.selected else None
            ),
            "codex_packet": self.codex_packet.to_dict() if self.codex_packet else None,
            "stop_reason": self.stop_reason.value,
            "retry_attempt": self.retry_attempt,
            "refinement_depth": self.refinement_depth,
        }


@dataclass(frozen=True)
class DiagnosticReceipt(CanonicalContract):
    """Stable identity for one semantic failure diagnosis.

    Volatile timestamps, log paths, and retry counters are intentionally
    absent, so an identical counterexample/trigger pair reuses this receipt
    across repair rounds and process restarts.
    """

    SCHEMA: ClassVar[str] = DIAGNOSTIC_RECEIPT_SCHEMA

    prior_decision_id: str
    counterexample_id: str
    trigger_evidence_id: str
    trigger_signal_kind: str
    repository_tree_id: str = "tree:unspecified"

    def __post_init__(self) -> None:
        for name in (
            "prior_decision_id",
            "counterexample_id",
            "trigger_evidence_id",
            "repository_tree_id",
        ):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ReplannerValidationError(f"{name} is required")
            object.__setattr__(self, name, value)
        kind = str(self.trigger_signal_kind or "").strip()
        if kind not in RESPONSIVE_REPLAN_SIGNAL_KINDS:
            raise ReplannerValidationError("trigger_signal_kind is unsupported")
        object.__setattr__(self, "trigger_signal_kind", kind)

    @property
    def receipt_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "replanner_version": FORMAL_REPLANNER_VERSION,
            "prior_decision_id": self.prior_decision_id,
            "counterexample_id": self.counterexample_id,
            "trigger_evidence_id": self.trigger_evidence_id,
            "trigger_signal_kind": self.trigger_signal_kind,
            "repository_tree_id": self.repository_tree_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DiagnosticReceipt":
        allowed = {
            "schema",
            "content_id",
            "replanner_version",
            "prior_decision_id",
            "counterexample_id",
            "trigger_evidence_id",
            "trigger_signal_kind",
            "repository_tree_id",
        }
        if not isinstance(payload, Mapping) or set(payload).difference(allowed):
            raise ReplannerValidationError(
                "diagnostic receipt contains unsupported fields"
            )
        if payload.get("schema") not in (None, "", cls.SCHEMA):
            raise ReplannerValidationError("diagnostic receipt schema is unsupported")
        if payload.get("replanner_version") not in (
            None,
            FORMAL_REPLANNER_VERSION,
        ):
            raise ReplannerValidationError(
                "diagnostic receipt replanner version is unsupported"
            )
        result = cls(
            prior_decision_id=str(payload.get("prior_decision_id") or ""),
            counterexample_id=str(payload.get("counterexample_id") or ""),
            trigger_evidence_id=str(payload.get("trigger_evidence_id") or ""),
            trigger_signal_kind=str(payload.get("trigger_signal_kind") or ""),
            repository_tree_id=str(payload.get("repository_tree_id") or ""),
        )
        claimed = payload.get("content_id")
        if claimed not in (None, "", result.content_id):
            raise ReplannerValidationError(
                "diagnostic receipt identity does not match payload"
            )
        return result


@dataclass(frozen=True)
class ResponsiveReplanDecision:
    """Typed boundary decision for evidence-responsive refinement.

    ``result`` is intentionally absent for an unchanged trigger/counterexample
    pair: no source compilation, repair generation, validation, taskboard
    admission, or model-facing packet construction occurred.  Changed evidence
    delegates to exactly one normal :meth:`FormalReplanner.replan` pass and
    retains that complete verified result.
    """

    counterexample_id: str
    previous_counterexample_id: str
    changed: bool
    stop_reason: ReplanStopReason
    result: ReplanResult | None
    backoff_attempt: int
    backoff_seconds: int
    trigger_evidence_id: str = ""
    previous_trigger_evidence_id: str = ""
    trigger_signal_kind: str = "counterexample"
    diagnostic_receipt: DiagnosticReceipt | None = None
    diagnostic_reused: bool = False

    def __post_init__(self) -> None:
        current = str(self.counterexample_id or "").strip()
        previous = str(self.previous_counterexample_id or "").strip()
        if not current:
            raise ReplannerValidationError("counterexample_id is required")
        object.__setattr__(self, "counterexample_id", current)
        object.__setattr__(self, "previous_counterexample_id", previous)
        trigger = str(self.trigger_evidence_id or current).strip()
        previous_trigger = str(
            self.previous_trigger_evidence_id or previous
        ).strip()
        if not trigger:
            raise ReplannerValidationError("trigger_evidence_id is required")
        object.__setattr__(self, "trigger_evidence_id", trigger)
        object.__setattr__(
            self, "previous_trigger_evidence_id", previous_trigger
        )
        trigger_kind = str(self.trigger_signal_kind or "").strip()
        if trigger_kind not in RESPONSIVE_REPLAN_SIGNAL_KINDS:
            raise ReplannerValidationError("trigger_signal_kind is unsupported")
        object.__setattr__(self, "trigger_signal_kind", trigger_kind)
        receipt = self.diagnostic_receipt
        if receipt is not None and not isinstance(receipt, DiagnosticReceipt):
            if not isinstance(receipt, Mapping):
                raise ReplannerValidationError(
                    "diagnostic_receipt must be a DiagnosticReceipt"
                )
            receipt = DiagnosticReceipt.from_dict(receipt)
        if receipt is not None and (
            receipt.counterexample_id != current
            or receipt.trigger_evidence_id != trigger
            or receipt.trigger_signal_kind != trigger_kind
        ):
            raise ReplannerValidationError(
                "diagnostic receipt is not bound to the responsive trigger"
            )
        object.__setattr__(self, "diagnostic_receipt", receipt)
        if not isinstance(self.diagnostic_reused, bool):
            raise ReplannerValidationError("diagnostic_reused must be boolean")
        if self.diagnostic_reused and receipt is None:
            raise ReplannerValidationError(
                "diagnostic reuse requires the reused receipt"
            )
        if not isinstance(self.changed, bool):
            raise ReplannerValidationError("changed must be boolean")
        object.__setattr__(self, "stop_reason", ReplanStopReason(self.stop_reason))
        if (
            isinstance(self.backoff_attempt, bool)
            or not isinstance(self.backoff_attempt, int)
            or self.backoff_attempt < 0
        ):
            raise ReplannerValidationError("backoff_attempt must be non-negative")
        if (
            isinstance(self.backoff_seconds, bool)
            or not isinstance(self.backoff_seconds, int)
            or self.backoff_seconds < 0
        ):
            raise ReplannerValidationError("backoff_seconds must be non-negative")
        if self.stop_reason is ReplanStopReason.CANCELLED:
            if self.result is not None or self.backoff_seconds:
                raise ReplannerValidationError(
                    "cancelled evidence cannot carry a result or backoff"
                )
        elif self.stop_reason is ReplanStopReason.IDENTICAL_FAILURE_ESCALATED:
            if self.changed or self.result is not None or self.backoff_seconds:
                raise ReplannerValidationError(
                    "identical failure escalation cannot replan or back off"
                )
        elif self.changed:
            if self.result is None:
                raise ReplannerValidationError(
                    "changed evidence requires one replanning result"
                )
            if self.stop_reason is ReplanStopReason.UNCHANGED_COUNTEREXAMPLE_BACKOFF:
                raise ReplannerValidationError(
                    "changed evidence cannot return unchanged backoff"
                )
            if self.backoff_seconds:
                raise ReplannerValidationError(
                    "changed evidence cannot request backoff"
                )
        elif (
            self.result is not None
            or self.stop_reason
            is not ReplanStopReason.UNCHANGED_COUNTEREXAMPLE_BACKOFF
            or self.backoff_seconds <= 0
        ):
            raise ReplannerValidationError(
                "unchanged evidence requires a positive backoff and no replan result"
            )

    @property
    def refined(self) -> bool:
        return self.changed and self.result is not None and self.result.admitted

    @property
    def model_call_required(self) -> bool:
        """Whether the admitted transition has a bounded Codex repair packet."""

        return bool(
            self.result is not None
            and self.result.admitted
            and self.result.codex_packet is not None
        )

    @property
    def cancelled(self) -> bool:
        return self.stop_reason is ReplanStopReason.CANCELLED

    @property
    def escalated(self) -> bool:
        return self.stop_reason is ReplanStopReason.IDENTICAL_FAILURE_ESCALATED

    @property
    def diagnostic_receipt_id(self) -> str:
        return (
            self.diagnostic_receipt.receipt_id
            if self.diagnostic_receipt is not None
            else ""
        )

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        """Return no objective evidence outside the bound receipt producer.

        This decision is useful deterministic routing metadata, but it lacks
        the durable request/policy/tree/verifier witness emitted by
        :class:`AdaptiveRefinementReceipt`.  ``requirement_ids`` identifies
        the downstream objective without allowing this wrapper to satisfy it.
        In particular, it must never be supplied as criterion validation,
        analyzer-health, or exhaustion-quorum evidence to
        ``AdaptiveRefinementResult.evaluate_objective_completion``.
        """

        return ()

    @property
    def requirement_ids(self) -> tuple[str, ...]:
        """Route to the bound producer without claiming evidence authority."""

        if self.cancelled:
            return ()
        if self.changed and self.trigger_signal_kind == "counterexample":
            return (BOUNDED_REFINEMENT_EVIDENCE_ID,)
        if not self.changed and self.trigger_signal_kind in {
            "counterexample",
            "repeated_failure",
        }:
            return (UNCHANGED_FAILURE_BACKOFF_EVIDENCE_ID,)
        return ()

    @property
    def completion_evidence_roles(self) -> tuple[str, ...]:
        """Return no completion-proof authority for routing metadata.

        The responsive decision has neither a completion-analyzer execution
        nor a criterion coverage join nor independent exhaustion scans.  A
        caller must obtain all three from their canonical producers and pass
        them separately to the adaptive result's completion gate.
        """

        return ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RESPONSIVE_REPLAN_DECISION_SCHEMA,
            "replanner_version": FORMAL_REPLANNER_VERSION,
            "requirement_ids": list(self.requirement_ids),
            "evidence_ids": list(self.evidence_ids),
            "completion_evidence_roles": list(
                self.completion_evidence_roles
            ),
            "completion_authority": False,
            "safe_for_completion_reasoning": False,
            "counterexample_id": self.counterexample_id,
            "previous_counterexample_id": self.previous_counterexample_id,
            "trigger_evidence_id": self.trigger_evidence_id,
            "previous_trigger_evidence_id": self.previous_trigger_evidence_id,
            "trigger_signal_kind": self.trigger_signal_kind,
            "diagnostic_receipt": (
                self.diagnostic_receipt.to_record()
                if self.diagnostic_receipt is not None
                else None
            ),
            "diagnostic_receipt_id": self.diagnostic_receipt_id,
            "diagnostic_reused": self.diagnostic_reused,
            "changed": self.changed,
            "refined": self.refined,
            "cancelled": self.cancelled,
            "escalated": self.escalated,
            "model_call_required": self.model_call_required,
            "stop_reason": self.stop_reason.value,
            "backoff_attempt": self.backoff_attempt,
            "backoff_seconds": self.backoff_seconds,
            "result": self.result.to_dict() if self.result is not None else None,
        }


_DELTA_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@=-]{0,255}$")


def _delta_identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _DELTA_IDENTIFIER.fullmatch(
        value.strip()
    ):
        raise ReplannerValidationError(
            f"{name} must be a bounded typed identifier"
        )
    return value.strip()


def _delta_identifiers(
    value: Iterable[Any], name: str
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ReplannerValidationError(f"{name} must be an array")
    return tuple(
        sorted({_delta_identifier(item, name) for item in value})
    )


@dataclass(frozen=True)
class DeltaPlanStep:
    """One dependency-addressable unit in an accepted plan."""

    step_id: str
    branch_id: str
    dependency_ids: tuple[str, ...] = ()
    accepted: bool = True
    evidence_ids: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    alternative_ids: tuple[str, ...] = ()
    constraint_ids: tuple[str, ...] = ()
    validation_signature_ids: tuple[str, ...] = ()
    capability_ids: tuple[str, ...] = ()
    conflict_scope_ids: tuple[str, ...] = ()
    resource_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "step_id", _delta_identifier(self.step_id, "step_id")
        )
        object.__setattr__(
            self, "branch_id", _delta_identifier(self.branch_id, "branch_id")
        )
        if not isinstance(self.accepted, bool):
            raise ReplannerValidationError("accepted must be boolean")
        for name in (
            "dependency_ids",
            "evidence_ids",
            "obligation_ids",
            "alternative_ids",
            "constraint_ids",
            "validation_signature_ids",
            "capability_ids",
            "conflict_scope_ids",
            "resource_ids",
        ):
            object.__setattr__(
                self,
                name,
                _delta_identifiers(getattr(self, name), name),
            )

    def invalidate(self) -> "DeltaPlanStep":
        """Clear acceptance and evidence while retaining reviewed structure."""

        return replace(self, accepted=False, evidence_ids=())

    def to_dict(self) -> dict[str, Any]:
        return {
            name: (
                list(getattr(self, name))
                if isinstance(getattr(self, name), tuple)
                else getattr(self, name)
            )
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeltaPlanStep":
        if not isinstance(payload, Mapping) or set(payload) != set(
            cls.__dataclass_fields__
        ):
            raise ReplannerValidationError(
                "delta plan step must use the closed schema"
            )
        return cls(**dict(payload))


@dataclass(frozen=True)
class DeltaPlan:
    """A frozen plan projection sufficient for dependency delta analysis."""

    scope: FailureMemoryScope
    steps: tuple[DeltaPlanStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, FailureMemoryScope):
            if not isinstance(self.scope, Mapping):
                raise ReplannerValidationError(
                    "delta plan scope must be FailureMemoryScope"
                )
            try:
                object.__setattr__(
                    self,
                    "scope",
                    FailureMemoryScope.from_dict(self.scope),
                )
            except PlanFailureMemoryError as exc:
                raise ReplannerValidationError(str(exc)) from exc
        steps = tuple(
            item
            if isinstance(item, DeltaPlanStep)
            else DeltaPlanStep.from_dict(item)
            for item in self.steps
        )
        if not steps:
            raise ReplannerValidationError(
                "delta plan requires at least one step"
            )
        ids = [item.step_id for item in steps]
        if len(ids) != len(set(ids)):
            raise ReplannerValidationError(
                "delta plan step identities must be unique"
            )
        known = set(ids)
        if any(
            set(item.dependency_ids).difference(known) for item in steps
        ):
            raise ReplannerValidationError(
                "delta plan contains a dangling dependency"
            )
        by_id = {item.step_id: item for item in steps}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise ReplannerValidationError(
                    "delta plan dependency graph contains a cycle"
                )
            if step_id in visited:
                return
            visiting.add(step_id)
            for dependency_id in by_id[step_id].dependency_ids:
                visit(dependency_id)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in sorted(known):
            visit(step_id)
        object.__setattr__(
            self, "steps", tuple(sorted(steps, key=lambda item: item.step_id))
        )

    @property
    def plan_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": DELTA_PLAN_SCHEMA,
            "replanner_version": FORMAL_REPLANNER_VERSION,
            "scope": self.scope.to_dict(),
            "steps": [item.to_dict() for item in self.steps],
        }
        if include_identity:
            payload["plan_id"] = self.plan_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeltaPlan":
        expected = {
            "schema",
            "replanner_version",
            "plan_id",
            "scope",
            "steps",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ReplannerValidationError(
                "delta plan must use the closed schema"
            )
        if (
            payload.get("schema") != DELTA_PLAN_SCHEMA
            or payload.get("replanner_version") != FORMAL_REPLANNER_VERSION
        ):
            raise ReplannerValidationError(
                "delta plan version is unsupported"
            )
        result = cls(
            scope=payload.get("scope") or {},
            steps=tuple(
                DeltaPlanStep.from_dict(item)
                for item in payload.get("steps") or ()
            ),
        )
        if payload.get("plan_id") != result.plan_id:
            raise ReplannerValidationError(
                "delta plan identity does not match content"
            )
        return result


PlanStep = DeltaPlanStep
PlanSnapshot = DeltaPlan


@dataclass(frozen=True)
class DeltaReplanLimits:
    """Hard bounds for one dependency-suffix repair decision."""

    max_invalidated_steps: int = 64
    max_reopened_branches: int = 16
    max_repair_attempts: int = 1
    max_repair_milliseconds: int = 30_000

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _positive(getattr(self, name), name)

    def to_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeltaReplanLimits":
        if not isinstance(payload, Mapping) or set(payload) != set(
            cls.__dataclass_fields__
        ):
            raise ReplannerValidationError(
                "delta replan limits must use the closed schema"
            )
        return cls(**dict(payload))


class DeltaReplanStopReason(str, Enum):
    REPLAN_REQUIRED = "replan_required"
    UNCHANGED_FAILURE_BACKOFF = "unchanged_failure_backoff"
    IDENTICAL_FAILURE_EXHAUSTED = "identical_failure_exhausted"
    FAILURE_MEMORY_BOUND_REACHED = "failure_memory_bound_reached"
    UNBOUND_FAILURE = "unbound_failure"
    REPAIR_BOUND_EXCEEDED = "repair_bound_exceeded"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class DeltaReplanDecision:
    """Tamper-evident result of one smallest-suffix invalidation."""

    original_plan_id: str
    resulting_plan: DeltaPlan
    failure_event_id: str
    diagnostic_id: str
    stop_reason: DeltaReplanStopReason
    direct_failure_step_ids: tuple[str, ...]
    invalidated_step_ids: tuple[str, ...]
    stale_dependency_step_ids: tuple[str, ...]
    preserved_step_ids: tuple[str, ...]
    reopened_branch_ids: tuple[str, ...]
    preserved_branch_ids: tuple[str, ...]
    diagnostic_reused: bool
    backoff_attempt: int
    backoff_milliseconds: int
    repair_attempts: int
    limits: DeltaReplanLimits

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "original_plan_id",
            _delta_identifier(self.original_plan_id, "original_plan_id"),
        )
        if not isinstance(self.resulting_plan, DeltaPlan):
            if not isinstance(self.resulting_plan, Mapping):
                raise ReplannerValidationError(
                    "resulting_plan must be DeltaPlan"
                )
            object.__setattr__(
                self,
                "resulting_plan",
                DeltaPlan.from_dict(self.resulting_plan),
            )
        object.__setattr__(
            self, "stop_reason", DeltaReplanStopReason(self.stop_reason)
        )
        for name in ("failure_event_id", "diagnostic_id"):
            object.__setattr__(
                self, name, _delta_identifier(getattr(self, name), name)
            )
        for name in (
            "direct_failure_step_ids",
            "invalidated_step_ids",
            "stale_dependency_step_ids",
            "preserved_step_ids",
            "reopened_branch_ids",
            "preserved_branch_ids",
        ):
            object.__setattr__(
                self,
                name,
                _delta_identifiers(getattr(self, name), name),
            )
        if not isinstance(self.diagnostic_reused, bool):
            raise ReplannerValidationError(
                "diagnostic_reused must be boolean"
            )
        for name in (
            "backoff_attempt",
            "backoff_milliseconds",
            "repair_attempts",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ReplannerValidationError(
                    f"{name} must be a non-negative integer"
                )
        if not isinstance(self.limits, DeltaReplanLimits):
            if not isinstance(self.limits, Mapping):
                raise ReplannerValidationError(
                    "limits must be DeltaReplanLimits"
                )
            object.__setattr__(
                self, "limits", DeltaReplanLimits.from_dict(self.limits)
            )
        known = {item.step_id for item in self.resulting_plan.steps}
        if (
            set(self.direct_failure_step_ids).difference(known)
            or set(self.invalidated_step_ids).difference(known)
            or set(self.preserved_step_ids).difference(known)
        ):
            raise ReplannerValidationError(
                "delta decision names a step outside the resulting plan"
            )
        if set(self.invalidated_step_ids) & set(self.preserved_step_ids):
            raise ReplannerValidationError(
                "invalidated and preserved step sets must be disjoint"
            )
        by_id = {item.step_id: item for item in self.resulting_plan.steps}
        expected_preserved = tuple(
            sorted(
                item.step_id
                for item in self.resulting_plan.steps
                if item.accepted
                and item.step_id not in self.invalidated_step_ids
            )
        )
        if self.preserved_step_ids != expected_preserved:
            raise ReplannerValidationError(
                "preserved step projection is inconsistent"
            )
        by_branch: dict[str, list[DeltaPlanStep]] = {}
        for item in self.resulting_plan.steps:
            by_branch.setdefault(item.branch_id, []).append(item)
        expected_reopened = tuple(
            sorted(
                branch_id
                for branch_id, items in by_branch.items()
                if any(
                    item.step_id in self.invalidated_step_ids
                    for item in items
                )
            )
        )
        expected_preserved_branches = tuple(
            sorted(
                branch_id
                for branch_id, items in by_branch.items()
                if all(
                    item.accepted
                    and item.step_id not in self.invalidated_step_ids
                    for item in items
                )
            )
        )
        if (
            self.reopened_branch_ids != expected_reopened
            or self.preserved_branch_ids != expected_preserved_branches
        ):
            raise ReplannerValidationError(
                "branch preservation projection is inconsistent"
            )
        if self.stale_dependency_step_ids != tuple(
            sorted(
                set(self.invalidated_step_ids).difference(
                    self.direct_failure_step_ids
                )
            )
        ):
            raise ReplannerValidationError(
                "stale dependency projection is inconsistent"
            )
        if self.stop_reason is DeltaReplanStopReason.REPLAN_REQUIRED:
            if (
                not self.invalidated_step_ids
                or self.repair_attempts != 1
                or self.backoff_milliseconds
            ):
                raise ReplannerValidationError(
                    "active delta repair projection is inconsistent"
                )
            if any(
                by_id[item].accepted or by_id[item].evidence_ids
                for item in self.invalidated_step_ids
            ):
                raise ReplannerValidationError(
                    "invalidated steps must be reopened without stale evidence"
                )
            expected_suffix: set[str] = set(self.direct_failure_step_ids)
            changed = True
            while changed:
                changed = False
                for item in self.resulting_plan.steps:
                    if (
                        item.step_id not in expected_suffix
                        and set(item.dependency_ids).intersection(expected_suffix)
                    ):
                        expected_suffix.add(item.step_id)
                        changed = True
            if self.invalidated_step_ids != tuple(sorted(expected_suffix)):
                raise ReplannerValidationError(
                    "invalidated suffix is not dependency-minimal"
                )
        elif self.repair_attempts:
            raise ReplannerValidationError(
                "non-repair decisions cannot consume repair attempts"
            )
        if self.repair_attempts > self.limits.max_repair_attempts:
            raise ReplannerValidationError(
                "delta decision exceeds its repair-attempt bound"
            )

    @property
    def changed(self) -> bool:
        return self.stop_reason is DeltaReplanStopReason.REPLAN_REQUIRED

    @property
    def should_replan(self) -> bool:
        return self.changed

    @property
    def plan(self) -> DeltaPlan:
        return self.resulting_plan

    @property
    def invalidated_dependency_ids(self) -> tuple[str, ...]:
        return self.stale_dependency_step_ids

    @property
    def requirement_ids(self) -> tuple[str, ...]:
        return (DELTA_REPLAN_REQUIREMENT_ID,) if self.changed else ()

    @property
    def decision_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    @property
    def receipt_id(self) -> str:
        return self.decision_id

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": DELTA_REPLAN_DECISION_SCHEMA,
            "replanner_version": FORMAL_REPLANNER_VERSION,
            "original_plan_id": self.original_plan_id,
            "resulting_plan": self.resulting_plan.to_dict(),
            "failure_event_id": self.failure_event_id,
            "diagnostic_id": self.diagnostic_id,
            "stop_reason": self.stop_reason.value,
            "direct_failure_step_ids": list(self.direct_failure_step_ids),
            "invalidated_step_ids": list(self.invalidated_step_ids),
            "stale_dependency_step_ids": list(
                self.stale_dependency_step_ids
            ),
            "preserved_step_ids": list(self.preserved_step_ids),
            "reopened_branch_ids": list(self.reopened_branch_ids),
            "preserved_branch_ids": list(self.preserved_branch_ids),
            "diagnostic_reused": self.diagnostic_reused,
            "backoff_attempt": self.backoff_attempt,
            "backoff_milliseconds": self.backoff_milliseconds,
            "repair_attempts": self.repair_attempts,
            "limits": self.limits.to_dict(),
            "requirement_ids": list(self.requirement_ids),
        }
        if include_identity:
            payload["decision_id"] = self.decision_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeltaReplanDecision":
        expected = {
            "schema",
            "replanner_version",
            "decision_id",
            "original_plan_id",
            "resulting_plan",
            "failure_event_id",
            "diagnostic_id",
            "stop_reason",
            "direct_failure_step_ids",
            "invalidated_step_ids",
            "stale_dependency_step_ids",
            "preserved_step_ids",
            "reopened_branch_ids",
            "preserved_branch_ids",
            "diagnostic_reused",
            "backoff_attempt",
            "backoff_milliseconds",
            "repair_attempts",
            "limits",
            "requirement_ids",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ReplannerValidationError(
                "delta replan decision must use the closed schema"
            )
        if (
            payload.get("schema") != DELTA_REPLAN_DECISION_SCHEMA
            or payload.get("replanner_version") != FORMAL_REPLANNER_VERSION
        ):
            raise ReplannerValidationError(
                "delta replan decision version is unsupported"
            )
        result = cls(
            original_plan_id=payload.get("original_plan_id", ""),
            resulting_plan=DeltaPlan.from_dict(
                payload.get("resulting_plan") or {}
            ),
            failure_event_id=payload.get("failure_event_id", ""),
            diagnostic_id=payload.get("diagnostic_id", ""),
            stop_reason=payload.get("stop_reason", ""),
            direct_failure_step_ids=tuple(
                payload.get("direct_failure_step_ids") or ()
            ),
            invalidated_step_ids=tuple(
                payload.get("invalidated_step_ids") or ()
            ),
            stale_dependency_step_ids=tuple(
                payload.get("stale_dependency_step_ids") or ()
            ),
            preserved_step_ids=tuple(
                payload.get("preserved_step_ids") or ()
            ),
            reopened_branch_ids=tuple(
                payload.get("reopened_branch_ids") or ()
            ),
            preserved_branch_ids=tuple(
                payload.get("preserved_branch_ids") or ()
            ),
            diagnostic_reused=payload.get("diagnostic_reused"),
            backoff_attempt=payload.get("backoff_attempt", -1),
            backoff_milliseconds=payload.get(
                "backoff_milliseconds", -1
            ),
            repair_attempts=payload.get("repair_attempts", -1),
            limits=DeltaReplanLimits.from_dict(payload.get("limits") or {}),
        )
        if payload.get("requirement_ids") != list(result.requirement_ids):
            raise ReplannerValidationError(
                "delta replan requirement projection is inconsistent"
            )
        if payload.get("decision_id") != result.decision_id:
            raise ReplannerValidationError(
                "delta replan decision identity does not match content"
            )
        return result


_SECTION_ALIASES: Final[Mapping[str, str]] = {
    "objective": "objectives",
    "objective_record": "objectives",
    "objective_records": "objectives",
    "goals": "objectives",
    "objectives": "objectives",
    "task": "tasks",
    "task_record": "tasks",
    "task_records": "tasks",
    "taskboard": "tasks",
    "taskboard_records": "tasks",
    "tasks": "tasks",
    "ast": "ast",
    "ast_record": "ast",
    "ast_records": "ast",
    "ast_scopes": "ast",
    "symbols": "ast",
    "policy": "policies",
    "policies": "policies",
    "policy_record": "policies",
    "policy_records": "policies",
    "proof_policy": "policies",
    "proof_policies": "policies",
    "lease": "leases",
    "leases": "leases",
    "lease_record": "leases",
    "lease_records": "leases",
    "evidence": "evidence",
    "evidence_record": "evidence",
    "evidence_records": "evidence",
}


def _record_values(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [_public_mapping(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_public_mapping(item) for item in value if isinstance(item, Mapping)]
    raise ReplannerValidationError("formal-plan source sections must contain records")


def _source_bundle(source: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise ReplannerValidationError("source must be a formal-plan input object")
    bundle: dict[str, Any] = {
        "objectives": [],
        "tasks": [],
        "ast": [],
        "policies": [],
        "leases": [],
        "evidence": [],
        "repository_tree_id": str(
            source.get("repository_tree_id")
            or source.get("tree_cid")
            or source.get("tree_id")
            or ""
        ).strip(),
    }
    for key, value in source.items():
        section = _SECTION_ALIASES.get(str(key).lower())
        if section:
            bundle[section].extend(_record_values(value))
    records = source.get("records")
    if isinstance(records, Sequence) and not isinstance(
        records, (str, bytes, bytearray)
    ):
        for wrapped in records:
            if not isinstance(wrapped, Mapping):
                continue
            section = _SECTION_ALIASES.get(
                str(
                    wrapped.get("record_type")
                    or wrapped.get("section")
                    or wrapped.get("kind")
                    or ""
                ).lower()
            )
            record = wrapped.get("record", wrapped.get("payload", wrapped))
            if section and isinstance(record, Mapping):
                bundle[section].append(_public_mapping(record))
    # Canonicalize source order and collapse duplicate aliases.
    for section in ("objectives", "tasks", "ast", "policies", "leases", "evidence"):
        unique = {canonical_json(item): item for item in bundle[section]}
        bundle[section] = [unique[key] for key in sorted(unique)]
    return bundle


def _identity(record: Mapping[str, Any], kind: str) -> str:
    names = {
        "task": (
            "task_cid", "canonical_task_cid", "content_id", "cid",
            "canonical_task_id", "task_id", "id",
        ),
        "goal": (
            "goal_cid", "content_id", "cid", "canonical_goal_id", "goal_id", "id",
        ),
    }[kind]
    return next(
        (str(record.get(name)).strip() for name in names if record.get(name)),
        "",
    )


def _aliases(record: Mapping[str, Any], kind: str) -> set[str]:
    names = (
        ("task_cid", "canonical_task_cid", "content_id", "cid", "canonical_task_id", "task_id", "id")
        if kind == "task"
        else ("goal_cid", "content_id", "cid", "canonical_goal_id", "goal_id", "id")
    )
    return {str(record.get(name)).strip() for name in names if record.get(name)}


def _task(bundle: Mapping[str, Any], task_id: str) -> dict[str, Any] | None:
    for record in bundle["tasks"]:
        if task_id in _aliases(record, "task"):
            return record
    return None


def _payload_values(value: Any, *names: str) -> list[Any]:
    """Find named values in a bounded, already sanitized counterexample payload."""

    result: list[Any] = []
    queue: list[tuple[Any, int]] = [(value, 0)]
    wanted = set(names)
    while queue:
        current, depth = queue.pop(0)
        if depth > 4:
            continue
        if isinstance(current, Mapping):
            for key in sorted(current):
                item = current[key]
                if str(key) in wanted:
                    result.append(item)
                if isinstance(item, (Mapping, list, tuple)):
                    queue.append((item, depth + 1))
        elif isinstance(current, (list, tuple)):
            queue.extend((item, depth + 1) for item in current[:32])
    return result


def _first_string(value: Any, *names: str) -> str:
    for found in _payload_values(value, *names):
        values = _strings(found)
        if values:
            return values[0]
    return ""


def _counterexample(
    value: FormalCounterexample | Mapping[str, Any],
) -> FormalCounterexample:
    if isinstance(value, FormalCounterexample):
        return value
    if not isinstance(value, Mapping):
        raise ReplannerValidationError(
            "counterexample must be FormalCounterexample or canonical object"
        )
    return FormalCounterexample.from_dict(value)


class FormalReplanner:
    """Generate, compile, check, rank, and admit one bounded repair transition."""

    def __init__(
        self,
        *,
        compiler: FormalPlanCompiler | None = None,
        validator: FormalPlanValidator | None = None,
        limits: ReplanLimits | Mapping[str, Any] | None = None,
        admission_callback: Callable[[RepairTransition], bool | None] | None = None,
    ) -> None:
        self.compiler = compiler or FormalPlanCompiler()
        self.validator = validator or FormalPlanValidator()
        if limits is None:
            limits = ReplanLimits()
        elif isinstance(limits, Mapping):
            limits = ReplanLimits(
                **{
                    name: limits[name]
                    for name in ReplanLimits.__dataclass_fields__
                    if name in limits
                }
            )
        if not isinstance(limits, ReplanLimits):
            raise ReplannerValidationError("limits must be ReplanLimits or an object")
        self.limits = limits
        self.admission_callback = admission_callback
        self._seen_semantic_ids: set[str] = set()
        self._attempts: dict[str, int] = {}

    @property
    def seen_semantic_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._seen_semantic_ids))

    def reset_history(self, counterexample_id: str | None = None) -> None:
        """Clear bounded retry history, normally after external state changes."""

        if counterexample_id is None:
            self._seen_semantic_ids.clear()
            self._attempts.clear()
            return
        self._attempts.pop(str(counterexample_id), None)

    def replan_if_changed(
        self,
        source: Mapping[str, Any],
        counterexample: FormalCounterexample | Mapping[str, Any],
        *,
        previous_counterexample_id: str | None,
        candidate_repairs: Iterable[RepairOperation | Mapping[str, Any]]
        | None = None,
        prior_semantic_ids: Iterable[str] = (),
        retry_attempt: int | None = None,
        refinement_depth: int = 0,
        backoff_attempt: int = 0,
        base_backoff_seconds: int = 1,
        max_backoff_seconds: int = 300,
        trigger_evidence_id: str | None = None,
        previous_trigger_evidence_id: str | None = None,
        trigger_signal_kind: str = "counterexample",
        prior_decision_id: str | None = None,
        repository_tree_id: str | None = None,
        previous_diagnostic_receipt_id: str | None = None,
        max_identical_failures: int = 8,
        cancelled: Any = None,
    ) -> ResponsiveReplanDecision:
        """Replan changed evidence once; back off unchanged evidence pre-compile.

        The caller persists the previous semantic trigger and counterexample
        identities.  A matching pair is content-level unchanged regardless of
        incidental payload ordering and therefore needs neither a compiler
        pass nor another model-facing request.
        """

        value = _counterexample(counterexample)
        previous = str(previous_counterexample_id or "").strip()
        trigger = str(trigger_evidence_id or value.semantic_id).strip()
        previous_trigger = str(
            previous_trigger_evidence_id
            if previous_trigger_evidence_id is not None
            else previous
        ).strip()
        trigger_kind = str(trigger_signal_kind or "").strip()
        if not trigger:
            raise ReplannerValidationError("trigger_evidence_id is required")
        if trigger_kind not in RESPONSIVE_REPLAN_SIGNAL_KINDS:
            raise ReplannerValidationError("trigger_signal_kind is unsupported")
        if (
            isinstance(max_identical_failures, bool)
            or not isinstance(max_identical_failures, int)
            or max_identical_failures < 1
        ):
            raise ReplannerValidationError(
                "max_identical_failures must be a positive integer"
            )
        for name, item, minimum in (
            ("backoff_attempt", backoff_attempt, 0),
            ("base_backoff_seconds", base_backoff_seconds, 1),
            ("max_backoff_seconds", max_backoff_seconds, 1),
        ):
            if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
                raise ReplannerValidationError(
                    f"{name} must be an integer of at least {minimum}"
                )
        if base_backoff_seconds > max_backoff_seconds:
            raise ReplannerValidationError(
                "base_backoff_seconds cannot exceed max_backoff_seconds"
            )
        default_decision_id = (
            value.bindings.plan_ids[0]
            if value.bindings.plan_ids
            else value.semantic_id
        )
        diagnostic = DiagnosticReceipt(
            prior_decision_id=str(prior_decision_id or default_decision_id),
            counterexample_id=value.semantic_id,
            trigger_evidence_id=trigger,
            trigger_signal_kind=trigger_kind,
            repository_tree_id=str(
                repository_tree_id
                or source.get("repository_tree_id")
                or source.get("tree_id")
                or "tree:unspecified"
            ),
        )
        same_failure = (
            previous
            and previous == value.semantic_id
            and previous_trigger
            and previous_trigger == trigger
        )
        supplied_diagnostic_id = str(
            previous_diagnostic_receipt_id or ""
        ).strip()
        if (
            same_failure
            and supplied_diagnostic_id
            and supplied_diagnostic_id != diagnostic.receipt_id
        ):
            raise ReplannerValidationError(
                "previous diagnostic receipt does not match identical failure"
            )
        diagnostic_reused = bool(
            same_failure
            and supplied_diagnostic_id in ("", diagnostic.receipt_id)
        )
        if _cancelled(cancelled):
            return ResponsiveReplanDecision(
                counterexample_id=value.semantic_id,
                previous_counterexample_id=previous,
                changed=not bool(same_failure),
                stop_reason=ReplanStopReason.CANCELLED,
                result=None,
                backoff_attempt=backoff_attempt,
                backoff_seconds=0,
                trigger_evidence_id=trigger,
                previous_trigger_evidence_id=previous_trigger,
                trigger_signal_kind=trigger_kind,
                diagnostic_receipt=diagnostic,
                diagnostic_reused=diagnostic_reused,
            )
        if same_failure:
            next_attempt = backoff_attempt + 1
            if next_attempt >= max_identical_failures:
                return ResponsiveReplanDecision(
                    counterexample_id=value.semantic_id,
                    previous_counterexample_id=previous,
                    changed=False,
                    stop_reason=ReplanStopReason.IDENTICAL_FAILURE_ESCALATED,
                    result=None,
                    backoff_attempt=next_attempt,
                    backoff_seconds=0,
                    trigger_evidence_id=trigger,
                    previous_trigger_evidence_id=previous_trigger,
                    trigger_signal_kind=trigger_kind,
                    diagnostic_receipt=diagnostic,
                    diagnostic_reused=diagnostic_reused,
                )
            exponent = min(backoff_attempt, 30)
            seconds = min(
                max_backoff_seconds,
                base_backoff_seconds * (2 ** exponent),
            )
            return ResponsiveReplanDecision(
                counterexample_id=value.semantic_id,
                previous_counterexample_id=previous,
                changed=False,
                stop_reason=ReplanStopReason.UNCHANGED_COUNTEREXAMPLE_BACKOFF,
                result=None,
                backoff_attempt=backoff_attempt + 1,
                backoff_seconds=seconds,
                trigger_evidence_id=trigger,
                previous_trigger_evidence_id=previous_trigger,
                trigger_signal_kind=trigger_kind,
                diagnostic_receipt=diagnostic,
                diagnostic_reused=diagnostic_reused,
            )

        try:
            result = self.replan(
                source,
                value,
                candidate_repairs=candidate_repairs,
                prior_semantic_ids=prior_semantic_ids,
                retry_attempt=retry_attempt,
                refinement_depth=refinement_depth,
                cancelled=cancelled,
            )
        except ReplanCancelled:
            return ResponsiveReplanDecision(
                counterexample_id=value.semantic_id,
                previous_counterexample_id=previous,
                changed=True,
                stop_reason=ReplanStopReason.CANCELLED,
                result=None,
                backoff_attempt=backoff_attempt,
                backoff_seconds=0,
                trigger_evidence_id=trigger,
                previous_trigger_evidence_id=previous_trigger,
                trigger_signal_kind=trigger_kind,
                diagnostic_receipt=diagnostic,
                diagnostic_reused=False,
            )
        return ResponsiveReplanDecision(
            counterexample_id=value.semantic_id,
            previous_counterexample_id=previous,
            changed=True,
            stop_reason=result.stop_reason,
            result=result,
            backoff_attempt=0,
            backoff_seconds=0,
            trigger_evidence_id=trigger,
            previous_trigger_evidence_id=previous_trigger,
            trigger_signal_kind=trigger_kind,
            diagnostic_receipt=diagnostic,
            diagnostic_reused=False,
        )

    def replan_for_signal(
        self,
        source: Mapping[str, Any],
        counterexample: FormalCounterexample | Mapping[str, Any],
        signal: Any,
        *,
        previous_signal_id: str | None,
        previous_counterexample_id: str | None = None,
        candidate_repairs: Iterable[RepairOperation | Mapping[str, Any]]
        | None = None,
        prior_semantic_ids: Iterable[str] = (),
        retry_attempt: int | None = None,
        refinement_depth: int = 0,
        backoff_attempt: int = 0,
        base_backoff_seconds: int = 1,
        max_backoff_seconds: int = 300,
        prior_decision_id: str | None = None,
        repository_tree_id: str | None = None,
        previous_diagnostic_receipt_id: str | None = None,
        max_identical_failures: int = 8,
        cancelled: Any = None,
    ) -> ResponsiveReplanDecision:
        """Route one reviewed runtime signal into bounded formal replanning.

        Importing the adaptive type locally keeps the deterministic replanner
        usable on its own while ensuring arbitrary mappings cannot masquerade
        as reviewed runtime evidence.  A changed signal identity forces one
        bounded pass even if its minimized formal counterexample is unchanged;
        an unchanged signal/counterexample pair backs off before compilation.
        """

        from ..objectives.adaptive_goal_refiner import RefinementSignal

        if not isinstance(signal, RefinementSignal):
            raise ReplannerValidationError(
                "signal must be a typed RefinementSignal"
            )
        return self.replan_if_changed(
            source,
            counterexample,
            previous_counterexample_id=previous_counterexample_id,
            candidate_repairs=candidate_repairs,
            prior_semantic_ids=prior_semantic_ids,
            retry_attempt=retry_attempt,
            refinement_depth=refinement_depth,
            backoff_attempt=backoff_attempt,
            base_backoff_seconds=base_backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
            trigger_evidence_id=signal.evidence_id,
            previous_trigger_evidence_id=previous_signal_id,
            trigger_signal_kind=signal.kind.value,
            prior_decision_id=prior_decision_id,
            repository_tree_id=repository_tree_id,
            previous_diagnostic_receipt_id=previous_diagnostic_receipt_id,
            max_identical_failures=max_identical_failures,
            cancelled=cancelled,
        )

    def replan_delta(
        self,
        plan: DeltaPlan | Mapping[str, Any],
        observation: BranchFailureObservation | Mapping[str, Any],
        *,
        failure_memory: PlanFailureMemory | None = None,
        limits: DeltaReplanLimits | Mapping[str, Any] | None = None,
        observed_at_milliseconds: int = 1,
        now_milliseconds: int | None = None,
        deadline_milliseconds: int | None = None,
        cancelled: Any = None,
    ) -> DeltaReplanDecision:
        """Invalidate only the failed step and its transitive dependants."""

        return FormalDeltaReplanner(
            failure_memory=failure_memory,
            limits=limits,
        ).replan(
            plan,
            observation,
            observed_at_milliseconds=observed_at_milliseconds,
            now_milliseconds=now_milliseconds,
            deadline_milliseconds=deadline_milliseconds,
            cancelled=cancelled,
        )

    def generate_repairs(
        self,
        source: Mapping[str, Any],
        counterexample: FormalCounterexample | Mapping[str, Any],
    ) -> tuple[RepairOperation, ...]:
        """Generate deterministic typed operations without compiling them."""

        bundle = _source_bundle(source)
        value = _counterexample(counterexample)
        bound_ids = [
            item for item in value.bindings.task_ids if _task(bundle, item) is not None
        ]
        target = bound_ids[0] if bound_ids else ""
        if not target:
            target = _first_string(
                value.payload, "task_id", "target_task_id", "successor_task_id"
            )
        if not target or _task(bundle, target) is None:
            return ()

        repair_classes = value.repair_classes or (RepairClass.HUMAN_REVIEW,)
        operations: list[RepairOperation] = []
        for repair_class in repair_classes:
            generated = self._generate_for_class(bundle, value, target, repair_class)
            operations.extend(generated[: self.limits.max_candidates_per_rule])
            if len(operations) >= self.limits.max_candidates:
                break
        unique = {item.semantic_id: item for item in operations}
        return tuple(unique[key] for key in sorted(unique))[: self.limits.max_candidates]

    def replan(
        self,
        source: Mapping[str, Any],
        counterexample: FormalCounterexample | Mapping[str, Any],
        *,
        candidate_repairs: Iterable[RepairOperation | Mapping[str, Any]] | None = None,
        prior_semantic_ids: Iterable[str] = (),
        retry_attempt: int | None = None,
        refinement_depth: int = 0,
        cancelled: Any = None,
    ) -> ReplanResult:
        """Run one bounded refinement and admit at most one selected transition."""

        if _cancelled(cancelled):
            raise ReplanCancelled("formal replanning was cancelled")
        value = _counterexample(counterexample)
        bundle = _source_bundle(source)
        if _cancelled(cancelled):
            raise ReplanCancelled("formal replanning was cancelled")
        compilation = self.compiler.compile(bundle)
        if compilation.status is not CompilationStatus.COMPILED or compilation.plan is None:
            return ReplanResult(
                compilation, None, value.semantic_id, (), None, None,
                ReplanStopReason.ORIGINAL_PLAN_INVALID,
                retry_attempt or 0, refinement_depth,
            )

        original_validation = self.validator.validate(
            compilation.plan, compilation.formulas
        )
        if _cancelled(cancelled):
            raise ReplanCancelled("formal replanning was cancelled")
        if retry_attempt is None:
            retry_attempt = self._attempts.get(value.semantic_id, 0)
        if isinstance(retry_attempt, bool) or not isinstance(retry_attempt, int) or retry_attempt < 0:
            raise ReplannerValidationError("retry_attempt must be non-negative")
        if isinstance(refinement_depth, bool) or not isinstance(refinement_depth, int) or refinement_depth < 0:
            raise ReplannerValidationError("refinement_depth must be non-negative")
        if retry_attempt >= self.limits.max_retry_attempts:
            return ReplanResult(
                compilation, original_validation, value.semantic_id, (), None, None,
                ReplanStopReason.RETRY_BUDGET_EXHAUSTED,
                retry_attempt, refinement_depth,
            )
        if refinement_depth >= self.limits.max_refinement_depth:
            return ReplanResult(
                compilation, original_validation, value.semantic_id, (), None, None,
                ReplanStopReason.REFINEMENT_DEPTH_EXHAUSTED,
                retry_attempt, refinement_depth,
            )
        bound_plan_ids = set(value.bindings.plan_ids)
        accepted_plan_ids = {compilation.plan_id, compilation.source_identity}
        if bound_plan_ids and not (bound_plan_ids & accepted_plan_ids):
            return ReplanResult(
                compilation, original_validation, value.semantic_id, (), None, None,
                ReplanStopReason.COUNTEREXAMPLE_PLAN_MISMATCH,
                retry_attempt, refinement_depth,
            )

        self._attempts[value.semantic_id] = retry_attempt + 1
        if candidate_repairs is None:
            operations = self.generate_repairs(bundle, value)
        else:
            parsed_operations: list[RepairOperation] = []
            for item in candidate_repairs:
                operation = (
                    item
                    if isinstance(item, RepairOperation)
                    else RepairOperation.from_dict(item)
                )
                if not operation.counterexample_id:
                    operation = replace(
                        operation, counterexample_id=value.semantic_id
                    )
                parsed_operations.append(operation)
            operations = tuple(parsed_operations)
        operations = operations[: self.limits.max_candidates]
        known = {
            str(item).strip()
            for item in prior_semantic_ids
            if str(item).strip()
        } | self._seen_semantic_ids
        candidates: list[RepairCandidate] = []
        for operation in operations:
            if _cancelled(cancelled):
                raise ReplanCancelled("formal replanning was cancelled")
            candidate = self._evaluate(
                bundle,
                compilation,
                original_validation,
                value,
                operation,
                known,
                refinement_depth + 1,
            )
            candidates.append(candidate)
            known.add(operation.semantic_id)
            self._seen_semantic_ids.add(operation.semantic_id)
        admissible = [item for item in candidates if item.admissible]
        selected = min(admissible, key=self._rank) if admissible else None
        packet: CodexRepairPacket | None = None
        stop = ReplanStopReason.NO_ADMISSIBLE_REPAIR
        if selected is not None and selected.transition is not None:
            # Prompt construction is part of admission: a transition which
            # cannot fit the configured capsule/token limits must never be
            # written to the taskboard and leave Codex without its bounded
            # context.
            try:
                if _cancelled(cancelled):
                    raise ReplanCancelled("formal replanning was cancelled")
                prospective_packet = self._codex_packet(
                    selected.transition, value
                )
            except ReplanCancelled:
                raise
            except (ReplannerValidationError, CounterexampleValidationError) as exc:
                selected = replace(
                    selected,
                    status=RepairCandidateStatus.ADMISSION_REJECTED,
                    rejection_reasons=(
                        *selected.rejection_reasons,
                        f"model-facing repair packet rejected: {exc}",
                    ),
                )
                candidates = [
                    selected if item.candidate_id == selected.candidate_id else item
                    for item in candidates
                ]
                prospective_packet = None
            admitted = prospective_packet is not None
            if admitted and self.admission_callback is not None:
                if _cancelled(cancelled):
                    raise ReplanCancelled("formal replanning was cancelled")
                try:
                    response = self.admission_callback(selected.transition)
                    admitted = response is not False
                except Exception:
                    admitted = False
            if _cancelled(cancelled):
                raise ReplanCancelled("formal replanning was cancelled")
            selected_status = (
                RepairCandidateStatus.ADMITTED
                if admitted
                else RepairCandidateStatus.ADMISSION_REJECTED
            )
            selected = replace(selected, status=selected_status)
            candidates = [
                selected if item.candidate_id == selected.candidate_id else item
                for item in candidates
            ]
            if admitted:
                packet = prospective_packet
                stop = ReplanStopReason.ADMITTED
        return ReplanResult(
            original_compilation=compilation,
            original_validation=original_validation,
            counterexample_id=value.semantic_id,
            candidates=tuple(candidates),
            selected=selected,
            codex_packet=packet,
            stop_reason=stop,
            retry_attempt=retry_attempt + 1,
            refinement_depth=refinement_depth + 1,
        )

    def _generate_for_class(
        self,
        bundle: Mapping[str, Any],
        counterexample: FormalCounterexample,
        target: str,
        repair_class: RepairClass,
    ) -> list[RepairOperation]:
        task = _task(bundle, target)
        assert task is not None
        payload = counterexample.payload
        common = {"counterexample_id": counterexample.semantic_id}
        if repair_class is RepairClass.ADD_DEPENDENCY:
            dependency = _first_string(
                payload,
                "dependency_task_id",
                "missing_dependency",
                "predecessor_task_id",
                "required_before_task_id",
            )
            alternatives = [dependency] if dependency else []
            alternatives.extend(
                item
                for item in counterexample.bindings.task_ids
                if item != target and _task(bundle, item) is not None
            )
            existing = set(_strings(task.get("depends_on") or task.get("dependencies")))
            return [
                RepairOperation(
                    RepairRuleKind.ADD_DEPENDENCY,
                    target,
                    {"dependency_task_id": item},
                    **common,
                )
                for item in _strings(alternatives)
                if item not in existing and item != target
            ]
        if repair_class is RepairClass.SPLIT_TASK:
            effects = task.get("effects")
            if not isinstance(effects, Sequence) or isinstance(effects, (str, bytes)):
                return []
            if len(effects) < 2:
                return []
            indices = sorted({len(effects) // 2, 1})
            result = []
            for index in indices:
                suffix = content_identity(
                    {
                        "counterexample_id": counterexample.semantic_id,
                        "target": target,
                        "split_index": index,
                    }
                ).split(":")[-1][:12]
                result.append(
                    RepairOperation(
                        RepairRuleKind.SPLIT_EFFECTS,
                        target,
                        {
                            "split_index": index,
                            "generated_task_id": f"{_identity(task, 'task')}:repair:{suffix}",
                        },
                        **common,
                    )
                )
            return result
        if repair_class is RepairClass.TIGHTEN_AUTHORITY:
            actor = _first_string(
                payload, "authorized_actor_id", "required_actor_id", "actor_id"
            )
            actors = _strings((actor,) if actor else task.get("actor_ids") or task.get("actor_id"))
            if not actors:
                actors = ("supervisor",)
            token_value = next(
                iter(_payload_values(payload, "fencing_token", "required_fencing_token")),
                1,
            )
            token = token_value if isinstance(token_value, int) and not isinstance(token_value, bool) and token_value >= 0 else 1
            return [
                RepairOperation(
                    RepairRuleKind.TIGHTEN_AUTHORITY,
                    target,
                    {"actor_ids": list(actors[:1]), "fencing_token": token},
                    **common,
                )
            ]
        if repair_class is RepairClass.ADD_OBLIGATION:
            proof_template = _first_string(
                payload, "proof_template_id", "obligation_template_id"
            )
            command = _first_string(
                payload, "test_command", "validation_command", "fallback_check_id"
            )
            if proof_template:
                kind, checks = "code_proof", (proof_template,)
            else:
                kind = "test"
                checks = (
                    command
                    or f"counterexample-regression:{counterexample.semantic_id}",
                )
            return [
                RepairOperation(
                    RepairRuleKind.ADD_EVIDENCE,
                    target,
                    {"evidence_kind": kind, "check_ids": list(checks)},
                    **common,
                )
            ]
        if repair_class is RepairClass.CONSTRAIN_SCOPE:
            scopes = counterexample.bindings.ast_scope_ids or _strings(
                _payload_values(payload, "scope_ids", "ast_scope_ids")
            )
            if not scopes:
                return []
            return [
                RepairOperation(
                    RepairRuleKind.CONSTRAIN_SCOPE,
                    target,
                    {"scope_ids": list(scopes)},
                    **common,
                )
            ]
        if repair_class is RepairClass.ADD_PREMISE:
            premises = (
                counterexample.assumption_ids
                or counterexample.bindings.assumption_ids
                or _strings(_payload_values(payload, "premise_ids", "assumption_ids"))
            )
            if not premises:
                premises = (f"reviewed-premise:{counterexample.semantic_id}",)
            return [
                RepairOperation(
                    RepairRuleKind.ADD_PREMISE,
                    target,
                    {"premise_ids": list(premises)},
                    **common,
                )
            ]
        if repair_class is RepairClass.ADJUST_RESOURCES:
            raw_bounds = next(
                (
                    item
                    for item in _payload_values(
                        payload, "resource_bounds", "required_resources", "bounds"
                    )
                    if isinstance(item, Mapping)
                ),
                None,
            )
            bounds = dict(raw_bounds or counterexample.finite_bounds)
            allowed = {
                str(key): value
                for key, value in bounds.items()
                if str(key) in {
                    "cpu", "memory_mb", "timeout_ms", "portfolio_width",
                    "trace_bound", "deadline",
                }
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
            }
            if not allowed:
                allowed = {"portfolio_width": 1}
            return [
                RepairOperation(
                    RepairRuleKind.CHANGE_RESOURCE_BOUNDS,
                    target,
                    {"resource_bounds": allowed},
                    **common,
                )
            ]
        if repair_class is RepairClass.HUMAN_REVIEW:
            scopes = (
                counterexample.bindings.ast_scope_ids
                or _strings(task.get("ast_scope_ids") or task.get("changed_ast_scopes"))
                or (target,)
            )
            reviewer = _first_string(payload, "reviewer_actor_id") or "human:semantic-reviewer"
            return [
                RepairOperation(
                    RepairRuleKind.HUMAN_REVIEW,
                    target,
                    {
                        "reviewer_actor_id": reviewer,
                        "scope_ids": list(scopes),
                        "question": counterexample.summary,
                    },
                    **common,
                )
            ]
        return []

    def _evaluate(
        self,
        bundle: Mapping[str, Any],
        original: PlanCompilationResult,
        original_validation: PlanValidationResult,
        counterexample: FormalCounterexample,
        operation: RepairOperation,
        known: set[str],
        refinement_depth: int,
    ) -> RepairCandidate:
        if operation.semantic_id in known:
            return RepairCandidate(
                operation, RepairCandidateStatus.DUPLICATE,
                rejection_reasons=("semantic repair identity was already attempted",),
            )
        if operation.counterexample_id and operation.counterexample_id != counterexample.semantic_id:
            return RepairCandidate(
                operation, RepairCandidateStatus.COUNTEREXAMPLE_REJECTED,
                rejection_reasons=("repair is bound to a different counterexample",),
            )
        try:
            repaired, changed, generated, taskboard = self._apply(bundle, operation)
        except ReplannerValidationError as exc:
            return RepairCandidate(
                operation, RepairCandidateStatus.COUNTEREXAMPLE_REJECTED,
                rejection_reasons=(str(exc),),
            )
        if changed > self.limits.max_changed_records or generated > self.limits.max_generated_tasks:
            return RepairCandidate(
                operation, RepairCandidateStatus.NO_PROGRESS,
                rejection_reasons=("repair exceeds changed-record or generated-task bound",),
            )
        compilation = self.compiler.compile(repaired)
        if compilation.status is not CompilationStatus.COMPILED or compilation.plan is None:
            reasons = tuple(item.message for item in compilation.issues) or (
                "candidate did not compile",
            )
            return RepairCandidate(
                operation, RepairCandidateStatus.COMPILE_REJECTED,
                compilation=compilation, rejection_reasons=reasons[:8],
            )
        if not self._same_goals(original, compilation, bundle, repaired):
            return RepairCandidate(
                operation, RepairCandidateStatus.GOAL_REJECTED,
                compilation=compilation,
                rejection_reasons=("candidate changed or removed the original goal",),
            )
        if compilation.plan_id == original.plan_id:
            return RepairCandidate(
                operation,
                RepairCandidateStatus.NO_PROGRESS,
                compilation=compilation,
                rejection_reasons=("repair did not change the formal plan identity",),
            )
        validation = self.validator.validate(compilation.plan, compilation.formulas)
        if validation.status is not PlanValidationStatus.CONSISTENT:
            return RepairCandidate(
                operation, RepairCandidateStatus.CHECK_REJECTED,
                compilation=compilation, validation=validation,
                rejection_reasons=tuple(item.message for item in validation.findings)[:8]
                or ("candidate plan is not bounded-consistent",),
            )
        addressed = self._addresses_counterexample(repaired, operation, counterexample)
        if not addressed:
            return RepairCandidate(
                operation, RepairCandidateStatus.COUNTEREXAMPLE_REJECTED,
                compilation=compilation, validation=validation,
                rejection_reasons=("typed postcondition did not address the counterexample",),
            )
        progress = RepairProgress(
            before_open_counterexamples=1,
            after_open_counterexamples=0,
            before_validation_findings=len(original_validation.findings),
            after_validation_findings=len(validation.findings),
            changed_records=changed,
            generated_tasks=generated,
        )
        if not progress.improved:
            return RepairCandidate(
                operation, RepairCandidateStatus.NO_PROGRESS,
                compilation=compilation, validation=validation,
                rejection_reasons=("explicit progress measure did not decrease",),
            )
        assert original.plan is not None
        transition = RepairTransition(
            original_plan_id=original.plan_id,
            repaired_plan_id=compilation.plan_id,
            counterexample_id=counterexample.semantic_id,
            repair=operation,
            goal_ids=tuple(item.goal_id for item in original.plan.goals),
            taskboard_records=taskboard,
            refinement_depth=refinement_depth,
            progress=progress,
        )
        return RepairCandidate(
            operation,
            RepairCandidateStatus.ADMISSIBLE,
            compilation=compilation,
            validation=validation,
            transition=transition,
        )

    def _apply(
        self,
        source: Mapping[str, Any],
        operation: RepairOperation,
    ) -> tuple[dict[str, Any], int, int, tuple[Mapping[str, Any], ...]]:
        bundle = copy.deepcopy(dict(source))
        task = _task(bundle, operation.target_task_id)
        if task is None:
            raise ReplannerValidationError("repair target is not in the source plan")
        kind = operation.kind
        params = operation.parameters
        changed = 1
        generated = 0
        taskboard: list[Mapping[str, Any]] = []
        if kind is RepairRuleKind.ADD_DEPENDENCY:
            dependency = str(params["dependency_task_id"]).strip()
            dependency_task = _task(bundle, dependency)
            if dependency_task is None:
                raise ReplannerValidationError("dependency target is not in the source plan")
            canonical = _identity(dependency_task, "task")
            current = set(_strings(task.get("depends_on") or task.get("dependencies")))
            if canonical in current or dependency in current:
                raise ReplannerValidationError("dependency is already present")
            task["depends_on"] = sorted(current | {canonical})
        elif kind is RepairRuleKind.SPLIT_EFFECTS:
            effects = task.get("effects")
            index = params["split_index"]
            if (
                not isinstance(effects, Sequence)
                or isinstance(effects, (str, bytes))
                or isinstance(index, bool)
                or not isinstance(index, int)
                or index <= 0
                or index >= len(effects)
            ):
                raise ReplannerValidationError("split index must divide explicit effects")
            generated_id = str(params["generated_task_id"]).strip()
            if _task(bundle, generated_id) is not None:
                raise ReplannerValidationError("generated split task already exists")
            original_id = _identity(task, "task")
            second = copy.deepcopy(task)
            for field_name in (
                "id", "task_id", "content_id", "cid", "canonical_task_id",
                "canonical_task_cid",
            ):
                second.pop(field_name, None)
            second["task_cid"] = generated_id
            second["effects"] = list(effects[index:])
            second["depends_on"] = [original_id]
            second["title"] = f"Continuation of {task.get('title') or original_id}"
            task["effects"] = list(effects[:index])
            for downstream in bundle["tasks"]:
                if downstream is task:
                    continue
                deps = list(_strings(downstream.get("depends_on") or downstream.get("dependencies")))
                if original_id in deps or any(
                    alias in deps for alias in _aliases(task, "task")
                ):
                    downstream["depends_on"] = sorted(
                        generated_id if item in _aliases(task, "task") else item
                        for item in deps
                    )
                    changed += 1
            bundle["tasks"].append(second)
            generated = 1
            changed += 1
        elif kind is RepairRuleKind.TIGHTEN_AUTHORITY:
            actors = _strings(params["actor_ids"])
            if len(actors) != 1:
                raise ReplannerValidationError(
                    "authority repair must select exactly one actor"
                )
            task["actor_ids"] = list(actors)
            for field_name in ("actor_id", "assigned_to", "assignee"):
                task.pop(field_name, None)
            task["lease"] = {
                "lease_cid": content_identity(
                    {
                        "repair_id": operation.semantic_id,
                        "task_id": _identity(task, "task"),
                        "actor_id": actors[0],
                    }
                ),
                "holder_id": actors[0],
                "fencing_token": params["fencing_token"],
            }
        elif kind is RepairRuleKind.ADD_EVIDENCE:
            criteria = list(task.get("acceptance_criteria") or ())
            criterion = {
                "kind": str(params["evidence_kind"]),
                "check_ids": list(_strings(params["check_ids"])),
            }
            if canonical_json(criterion) in {canonical_json(item) for item in criteria}:
                raise ReplannerValidationError("evidence requirement is already present")
            criteria.append(criterion)
            task["acceptance_criteria"] = criteria
        elif kind is RepairRuleKind.CONSTRAIN_SCOPE:
            scopes = _strings(params["scope_ids"])
            task["ast_scope_ids"] = list(scopes)
            for field_name in ("changed_ast_scopes", "symbol_cids"):
                task.pop(field_name, None)
        elif kind is RepairRuleKind.ADD_PREMISE:
            premises = _strings(params["premise_ids"])
            criteria = list(task.get("acceptance_criteria") or ())
            criterion = {
                "kind": "plan_check",
                "check_ids": list(premises),
            }
            if canonical_json(criterion) in {canonical_json(item) for item in criteria}:
                raise ReplannerValidationError("premise dependency is already present")
            criteria.append(criterion)
            task["acceptance_criteria"] = criteria
        elif kind is RepairRuleKind.CHANGE_RESOURCE_BOUNDS:
            bounds = dict(params["resource_bounds"])
            resources = {
                str(key): value
                for key, value in bounds.items()
                if key not in {"trace_bound", "deadline"}
            }
            if resources:
                task["resource_needs"] = resources
                task.pop("resources", None)
                task.pop("required_resources", None)
            if "deadline" in bounds:
                task["deadline"] = bounds["deadline"]
            if "trace_bound" in bounds:
                for policy in bundle["policies"]:
                    policy["trace_bound"] = bounds["trace_bound"]
                    changed += 1
        elif kind is RepairRuleKind.HUMAN_REVIEW:
            generated_id = f"{_identity(task, 'task')}:human-review:{operation.semantic_id.split(':')[-1][:12]}"
            if _task(bundle, generated_id) is not None:
                raise ReplannerValidationError("human review request already exists")
            goals = _aliases(task, "goal")
            review = {
                "task_cid": generated_id,
                "goal_id": str(task.get("goal_cid") or task.get("goal_id") or next(iter(goals), "")),
                "actor_id": str(params["reviewer_actor_id"]),
                "depends_on": [_identity(task, "task")],
                "ast_scope_ids": list(_strings(params["scope_ids"])),
                "acceptance_criteria": [
                    {
                        "kind": "review",
                        "check_ids": [f"review:{operation.counterexample_id}"],
                    }
                ],
                "title": "Scoped semantic review",
                "description": str(params.get("question") or "Review formal counterexample"),
            }
            bundle["tasks"].append(review)
            generated = 1
            changed += 1
        else:  # pragma: no cover - enum dispatch is exhaustive
            raise ReplannerValidationError(f"unsupported repair kind {kind.value}")

        touched = [task]
        if generated:
            touched.append(bundle["tasks"][-1])
        for record in touched:
            taskboard.append(
                {
                    "record_type": "formal_repair",
                    "task_id": _identity(record, "task"),
                    "goal_id": str(record.get("goal_cid") or record.get("goal_id") or ""),
                    "repair_id": operation.semantic_id,
                    "repair_class": operation.kind.value,
                    "counterexample_id": operation.counterexample_id,
                    "depends_on": list(
                        _strings(record.get("depends_on") or record.get("dependencies"))
                    ),
                    "ast_scope_ids": list(
                        _strings(record.get("ast_scope_ids") or record.get("changed_ast_scopes"))
                    ),
                }
            )
        return bundle, changed, generated, tuple(taskboard)

    @staticmethod
    def _same_goals(
        original: PlanCompilationResult,
        repaired: PlanCompilationResult,
        before_source: Mapping[str, Any],
        after_source: Mapping[str, Any],
    ) -> bool:
        if original.plan is None or repaired.plan is None:
            return False
        before_records = {
            canonical_json(item) for item in before_source.get("objectives", ())
        }
        after_records = {
            canonical_json(item) for item in after_source.get("objectives", ())
        }
        return (
            before_records == after_records
            and {item.goal_id for item in original.plan.goals}
            == {item.goal_id for item in repaired.plan.goals}
        )

    @staticmethod
    def _addresses_counterexample(
        source: Mapping[str, Any],
        operation: RepairOperation,
        counterexample: FormalCounterexample,
    ) -> bool:
        task = _task(source, operation.target_task_id)
        if task is None:
            return False
        params = operation.parameters
        if operation.kind is RepairRuleKind.ADD_DEPENDENCY:
            dependency = str(params["dependency_task_id"])
            dep_task = _task(source, dependency)
            if dep_task is None:
                return False
            values = set(_strings(task.get("depends_on") or task.get("dependencies")))
            return bool(values & _aliases(dep_task, "task"))
        if operation.kind is RepairRuleKind.SPLIT_EFFECTS:
            generated = _task(source, str(params["generated_task_id"]))
            return generated is not None and _identity(task, "task") in set(
                _strings(generated.get("depends_on"))
            )
        if operation.kind is RepairRuleKind.TIGHTEN_AUTHORITY:
            lease = task.get("lease")
            return (
                isinstance(lease, Mapping)
                and _strings(task.get("actor_ids")) == _strings(params["actor_ids"])
                and lease.get("fencing_token") == params["fencing_token"]
            )
        if operation.kind in {RepairRuleKind.ADD_EVIDENCE, RepairRuleKind.ADD_PREMISE}:
            serialized = canonical_json(task.get("acceptance_criteria") or [])
            expected = (
                _strings(params["check_ids"])
                if operation.kind is RepairRuleKind.ADD_EVIDENCE
                else _strings(params["premise_ids"])
            )
            return all(item in serialized for item in expected)
        if operation.kind is RepairRuleKind.CONSTRAIN_SCOPE:
            return _strings(task.get("ast_scope_ids")) == _strings(params["scope_ids"])
        if operation.kind is RepairRuleKind.CHANGE_RESOURCE_BOUNDS:
            bounds = params["resource_bounds"]
            return bool(bounds) and (
                bool(task.get("resource_needs"))
                or "deadline" in bounds
                or "trace_bound" in bounds
            )
        if operation.kind is RepairRuleKind.HUMAN_REVIEW:
            return any(
                operation.counterexample_id
                in canonical_json(item.get("acceptance_criteria") or ())
                for item in source["tasks"]
                if item is not task
            )
        return False

    @staticmethod
    def _rank(candidate: RepairCandidate) -> tuple[Any, ...]:
        assert candidate.transition is not None
        progress = candidate.transition.progress
        # Greatest decrease, then least scope growth, then canonical identity.
        return (
            progress.after,
            progress.changed_records,
            progress.generated_tasks,
            candidate.repair.kind.value,
            candidate.repair.semantic_id,
        )

    def _codex_packet(
        self,
        transition: RepairTransition,
        counterexample: FormalCounterexample,
    ) -> CodexRepairPacket:
        capsule_limit = min(
            self.limits.max_capsule_bytes,
            max(1024, self.limits.max_prompt_bytes // 2),
        )
        capsule = build_counterexample_context_capsule(
            (counterexample,),
            # The source is already the single selected counterexample.  An
            # empty target filter retains it; filtering by its own node id
            # would ask the graph for counterexamples *adjacent* to itself.
            target_ids=(),
            limits=CounterexampleLimits(max_capsule_bytes=capsule_limit),
        )
        return CodexRepairPacket(
            transition=transition,
            counterexample_capsule=capsule,
            max_bytes=self.limits.max_prompt_bytes,
            max_tokens=self.limits.max_prompt_tokens,
        )


class FormalDeltaReplanner:
    """Bind a typed failure to the minimal dependent plan suffix.

    The operation is a single deterministic repair round.  It never regenerates
    an unaffected step and never promotes a pending step to accepted.  Durable
    retry state is delegated to :class:`PlanFailureMemory`.
    """

    def __init__(
        self,
        *,
        failure_memory: PlanFailureMemory | None = None,
        limits: DeltaReplanLimits | Mapping[str, Any] | None = None,
    ) -> None:
        self.failure_memory = failure_memory or PlanFailureMemory()
        if limits is None:
            limits = DeltaReplanLimits()
        elif isinstance(limits, Mapping):
            limits = DeltaReplanLimits.from_dict(limits)
        if not isinstance(limits, DeltaReplanLimits):
            raise ReplannerValidationError(
                "limits must be DeltaReplanLimits or an object"
            )
        self.limits = limits

    @staticmethod
    def _anchors(
        plan: DeltaPlan, observation: BranchFailureObservation
    ) -> tuple[str, ...]:
        features = observation.features
        by_id = {item.step_id: item for item in plan.steps}
        explicit = tuple(
            sorted(set(features.step_ids).intersection(by_id))
        )
        if explicit:
            return explicit
        candidates = [
            item for item in plan.steps if item.branch_id == features.branch_id
        ]
        if not candidates:
            return ()
        bindings = (
            ("obligation_ids", features.obligation_ids),
            ("alternative_ids", features.alternative_ids),
            ("constraint_ids", features.constraint_ids),
            (
                "validation_signature_ids",
                features.validation_signature_ids,
            ),
            ("capability_ids", features.capability_ids),
            ("conflict_scope_ids", features.conflict_scope_ids),
            ("resource_ids", features.resource_ids),
        )
        matched_sets: list[set[str]] = []
        for field_name, expected in bindings:
            if not expected:
                continue
            matched = {
                item.step_id
                for item in candidates
                if set(getattr(item, field_name)).intersection(expected)
            }
            if matched:
                matched_sets.append(matched)
        if matched_sets:
            intersection = set.intersection(*matched_sets)
            selected = intersection or set.union(*matched_sets)
            return tuple(sorted(selected))
        # The branch itself is still a typed binding.  If its producer did not
        # expose finer feature bindings, fail safely within that branch only.
        return tuple(sorted(item.step_id for item in candidates))

    @staticmethod
    def _dependent_suffix(
        plan: DeltaPlan, anchors: Iterable[str]
    ) -> tuple[str, ...]:
        reverse: dict[str, set[str]] = {
            item.step_id: set() for item in plan.steps
        }
        for item in plan.steps:
            for dependency_id in item.dependency_ids:
                reverse[dependency_id].add(item.step_id)
        affected = set(anchors)
        frontier = list(sorted(affected))
        while frontier:
            current = frontier.pop()
            for dependent in sorted(reverse[current]):
                if dependent not in affected:
                    affected.add(dependent)
                    frontier.append(dependent)
        return tuple(sorted(affected))

    @staticmethod
    def _branch_projection(
        plan: DeltaPlan, invalidated: set[str]
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        accepted_preserved = tuple(
            sorted(
                item.step_id
                for item in plan.steps
                if item.accepted and item.step_id not in invalidated
            )
        )
        by_branch: dict[str, list[DeltaPlanStep]] = {}
        for item in plan.steps:
            by_branch.setdefault(item.branch_id, []).append(item)
        reopened = tuple(
            sorted(
                branch_id
                for branch_id, items in by_branch.items()
                if any(item.step_id in invalidated for item in items)
            )
        )
        preserved = tuple(
            sorted(
                branch_id
                for branch_id, items in by_branch.items()
                if all(
                    item.accepted and item.step_id not in invalidated
                    for item in items
                )
            )
        )
        return accepted_preserved, reopened, preserved

    def _unchanged_decision(
        self,
        plan: DeltaPlan,
        observation: BranchFailureObservation,
        reason: DeltaReplanStopReason,
        *,
        memory: FailureMemoryDecision | None = None,
        anchors: tuple[str, ...] = (),
    ) -> DeltaReplanDecision:
        preserved, reopened, preserved_branches = self._branch_projection(
            plan, set()
        )
        return DeltaReplanDecision(
            original_plan_id=plan.plan_id,
            resulting_plan=plan,
            failure_event_id=observation.event_id,
            diagnostic_id=observation.diagnostic_id,
            stop_reason=reason,
            direct_failure_step_ids=anchors,
            invalidated_step_ids=(),
            stale_dependency_step_ids=(),
            preserved_step_ids=preserved,
            reopened_branch_ids=(),
            preserved_branch_ids=preserved_branches,
            diagnostic_reused=(
                memory.diagnostic_reused if memory is not None else False
            ),
            backoff_attempt=(
                memory.backoff_attempt if memory is not None else 0
            ),
            backoff_milliseconds=(
                memory.backoff_milliseconds if memory is not None else 0
            ),
            repair_attempts=0,
            limits=self.limits,
        )

    def replan(
        self,
        plan: DeltaPlan | Mapping[str, Any],
        observation: BranchFailureObservation | Mapping[str, Any],
        *,
        observed_at_milliseconds: int = 1,
        now_milliseconds: int | None = None,
        deadline_milliseconds: int | None = None,
        cancelled: Any = None,
    ) -> DeltaReplanDecision:
        value = plan if isinstance(plan, DeltaPlan) else DeltaPlan.from_dict(plan)
        failure = (
            observation
            if isinstance(observation, BranchFailureObservation)
            else BranchFailureObservation.from_dict(observation)
        )
        if value.scope != failure.features.scope:
            raise ReplannerValidationError(
                "failure scope does not match the delta plan"
            )
        observed = (
            observed_at_milliseconds
            if isinstance(observed_at_milliseconds, int)
            and not isinstance(observed_at_milliseconds, bool)
            and observed_at_milliseconds >= 1
            else None
        )
        if observed is None:
            raise ReplannerValidationError(
                "observed_at_milliseconds must be a positive integer"
            )
        current = observed if now_milliseconds is None else now_milliseconds
        if (
            isinstance(current, bool)
            or not isinstance(current, int)
            or current < observed
        ):
            raise ReplannerValidationError(
                "now_milliseconds must not precede the observation"
            )
        if deadline_milliseconds is not None and (
            isinstance(deadline_milliseconds, bool)
            or not isinstance(deadline_milliseconds, int)
            or deadline_milliseconds < 1
        ):
            raise ReplannerValidationError(
                "deadline_milliseconds must be a positive integer"
            )
        if _cancelled(cancelled):
            return self._unchanged_decision(
                value, failure, DeltaReplanStopReason.CANCELLED
            )
        effective_deadline = observed + self.limits.max_repair_milliseconds
        if deadline_milliseconds is not None:
            effective_deadline = min(
                effective_deadline, deadline_milliseconds
            )
        if current >= effective_deadline:
            return self._unchanged_decision(
                value, failure, DeltaReplanStopReason.DEADLINE_EXCEEDED
            )
        anchors = self._anchors(value, failure)
        if not anchors:
            return self._unchanged_decision(
                value, failure, DeltaReplanStopReason.UNBOUND_FAILURE
            )
        suffix = self._dependent_suffix(value, anchors)
        _, reopened, _ = self._branch_projection(value, set(suffix))
        if (
            len(suffix) > self.limits.max_invalidated_steps
            or len(reopened) > self.limits.max_reopened_branches
        ):
            return self._unchanged_decision(
                value,
                failure,
                DeltaReplanStopReason.REPAIR_BOUND_EXCEEDED,
                anchors=anchors,
            )
        try:
            memory = self.failure_memory.observe(
                failure, observed_at_milliseconds=observed
            )
        except PlanFailureMemoryError as exc:
            raise ReplannerValidationError(str(exc)) from exc
        reason_by_disposition = {
            FailureMemoryDisposition.UNCHANGED_BACKOFF: (
                DeltaReplanStopReason.UNCHANGED_FAILURE_BACKOFF
            ),
            FailureMemoryDisposition.IDENTICAL_FAILURE_EXHAUSTED: (
                DeltaReplanStopReason.IDENTICAL_FAILURE_EXHAUSTED
            ),
            FailureMemoryDisposition.MEMORY_BOUND_REACHED: (
                DeltaReplanStopReason.FAILURE_MEMORY_BOUND_REACHED
            ),
        }
        if not memory.should_replan:
            return self._unchanged_decision(
                value,
                failure,
                reason_by_disposition[memory.disposition],
                memory=memory,
                anchors=anchors,
            )
        if _cancelled(cancelled):
            return self._unchanged_decision(
                value,
                failure,
                DeltaReplanStopReason.CANCELLED,
                memory=memory,
                anchors=anchors,
            )
        invalidated = set(suffix)
        resulting = DeltaPlan(
            scope=value.scope,
            steps=tuple(
                item.invalidate()
                if item.step_id in invalidated
                else item
                for item in value.steps
            ),
        )
        preserved, reopened, preserved_branches = self._branch_projection(
            value, invalidated
        )
        return DeltaReplanDecision(
            original_plan_id=value.plan_id,
            resulting_plan=resulting,
            failure_event_id=failure.event_id,
            diagnostic_id=failure.diagnostic_id,
            stop_reason=DeltaReplanStopReason.REPLAN_REQUIRED,
            direct_failure_step_ids=anchors,
            invalidated_step_ids=suffix,
            stale_dependency_step_ids=tuple(
                sorted(invalidated.difference(anchors))
            ),
            preserved_step_ids=preserved,
            reopened_branch_ids=reopened,
            preserved_branch_ids=preserved_branches,
            diagnostic_reused=memory.diagnostic_reused,
            backoff_attempt=memory.backoff_attempt,
            backoff_milliseconds=0,
            repair_attempts=1,
            limits=self.limits,
        )


CounterexampleDeltaReplanner = FormalDeltaReplanner
DeltaReplanner = FormalDeltaReplanner
DeltaReplanResult = DeltaReplanDecision
DeltaReplanBudget = DeltaReplanLimits
DeltaPlanNode = DeltaPlanStep
FormalPlanReplanner = FormalReplanner


def replan_plan_delta(
    plan: DeltaPlan | Mapping[str, Any],
    observation: BranchFailureObservation | Mapping[str, Any],
    *,
    failure_memory: PlanFailureMemory | None = None,
    limits: DeltaReplanLimits | Mapping[str, Any] | None = None,
    observed_at_milliseconds: int = 1,
    now_milliseconds: int | None = None,
    deadline_milliseconds: int | None = None,
    cancelled: Any = None,
) -> DeltaReplanDecision:
    """Functional entry point for one bounded dependency-suffix repair."""

    return FormalDeltaReplanner(
        failure_memory=failure_memory,
        limits=limits,
    ).replan(
        plan,
        observation,
        observed_at_milliseconds=observed_at_milliseconds,
        now_milliseconds=now_milliseconds,
        deadline_milliseconds=deadline_milliseconds,
        cancelled=cancelled,
    )


delta_replan = replan_plan_delta


def generate_plan_repairs(
    source: Mapping[str, Any],
    counterexample: FormalCounterexample | Mapping[str, Any],
    *,
    limits: ReplanLimits | Mapping[str, Any] | None = None,
    admission_callback: Callable[[RepairTransition], bool | None] | None = None,
    candidate_repairs: Iterable[RepairOperation | Mapping[str, Any]] | None = None,
    prior_semantic_ids: Iterable[str] = (),
    retry_attempt: int | None = None,
    refinement_depth: int = 0,
    cancelled: Any = None,
) -> ReplanResult:
    """Convenience entry point for one complete bounded replanning pass."""

    return FormalReplanner(
        limits=limits,
        admission_callback=admission_callback,
    ).replan(
        source,
        counterexample,
        candidate_repairs=candidate_repairs,
        prior_semantic_ids=prior_semantic_ids,
        retry_attempt=retry_attempt,
        refinement_depth=refinement_depth,
        cancelled=cancelled,
    )


replan_from_counterexample = generate_plan_repairs


def replan_if_changed(
    source: Mapping[str, Any],
    counterexample: FormalCounterexample | Mapping[str, Any],
    *,
    previous_counterexample_id: str | None,
    limits: ReplanLimits | Mapping[str, Any] | None = None,
    admission_callback: Callable[[RepairTransition], bool | None] | None = None,
    candidate_repairs: Iterable[RepairOperation | Mapping[str, Any]] | None = None,
    prior_semantic_ids: Iterable[str] = (),
    retry_attempt: int | None = None,
    refinement_depth: int = 0,
    backoff_attempt: int = 0,
    base_backoff_seconds: int = 1,
    max_backoff_seconds: int = 300,
    trigger_evidence_id: str | None = None,
    previous_trigger_evidence_id: str | None = None,
    trigger_signal_kind: str = "counterexample",
    prior_decision_id: str | None = None,
    repository_tree_id: str | None = None,
    previous_diagnostic_receipt_id: str | None = None,
    max_identical_failures: int = 8,
    cancelled: Any = None,
) -> ResponsiveReplanDecision:
    """Stateless convenience entry point for responsive bounded replanning."""

    return FormalReplanner(
        limits=limits,
        admission_callback=admission_callback,
    ).replan_if_changed(
        source,
        counterexample,
        previous_counterexample_id=previous_counterexample_id,
        candidate_repairs=candidate_repairs,
        prior_semantic_ids=prior_semantic_ids,
        retry_attempt=retry_attempt,
        refinement_depth=refinement_depth,
        backoff_attempt=backoff_attempt,
        base_backoff_seconds=base_backoff_seconds,
        max_backoff_seconds=max_backoff_seconds,
        trigger_evidence_id=trigger_evidence_id,
        previous_trigger_evidence_id=previous_trigger_evidence_id,
        trigger_signal_kind=trigger_signal_kind,
        prior_decision_id=prior_decision_id,
        repository_tree_id=repository_tree_id,
        previous_diagnostic_receipt_id=previous_diagnostic_receipt_id,
        max_identical_failures=max_identical_failures,
        cancelled=cancelled,
    )


def replan_for_signal(
    source: Mapping[str, Any],
    counterexample: FormalCounterexample | Mapping[str, Any],
    signal: Any,
    *,
    previous_signal_id: str | None,
    previous_counterexample_id: str | None = None,
    limits: ReplanLimits | Mapping[str, Any] | None = None,
    admission_callback: Callable[[RepairTransition], bool | None] | None = None,
    candidate_repairs: Iterable[RepairOperation | Mapping[str, Any]]
    | None = None,
    prior_semantic_ids: Iterable[str] = (),
    retry_attempt: int | None = None,
    refinement_depth: int = 0,
    backoff_attempt: int = 0,
    base_backoff_seconds: int = 1,
    max_backoff_seconds: int = 300,
    prior_decision_id: str | None = None,
    repository_tree_id: str | None = None,
    previous_diagnostic_receipt_id: str | None = None,
    max_identical_failures: int = 8,
    cancelled: Any = None,
) -> ResponsiveReplanDecision:
    """Stateless typed-signal entry point for bounded formal replanning."""

    return FormalReplanner(
        limits=limits,
        admission_callback=admission_callback,
    ).replan_for_signal(
        source,
        counterexample,
        signal,
        previous_signal_id=previous_signal_id,
        previous_counterexample_id=previous_counterexample_id,
        candidate_repairs=candidate_repairs,
        prior_semantic_ids=prior_semantic_ids,
        retry_attempt=retry_attempt,
        refinement_depth=refinement_depth,
        backoff_attempt=backoff_attempt,
        base_backoff_seconds=base_backoff_seconds,
        max_backoff_seconds=max_backoff_seconds,
        prior_decision_id=prior_decision_id,
        repository_tree_id=repository_tree_id,
        previous_diagnostic_receipt_id=previous_diagnostic_receipt_id,
        max_identical_failures=max_identical_failures,
        cancelled=cancelled,
    )


__all__ = [
    "BOUNDED_REFINEMENT_EVIDENCE_ID",
    "UNCHANGED_FAILURE_BACKOFF_EVIDENCE_ID",
    "CODEX_REPAIR_PACKET_SCHEMA",
    "DELTA_PLAN_SCHEMA",
    "DELTA_REPLAN_DECISION_SCHEMA",
    "DELTA_REPLAN_REQUIREMENT_ID",
    "DIAGNOSTIC_RECEIPT_SCHEMA",
    "FORMAL_REPLANNER_VERSION",
    "OBJECTIVE_COMPLETION_EVIDENCE_ROLES",
    "REPAIR_CANDIDATE_SCHEMA",
    "REPAIR_TRANSITION_SCHEMA",
    "REPLAN_RESULT_SCHEMA",
    "RESPONSIVE_REPLAN_DECISION_SCHEMA",
    "RESPONSIVE_REPLAN_SIGNAL_KINDS",
    "CodexRepairPacket",
    "CounterexampleDeltaReplanner",
    "DeltaPlan",
    "DeltaPlanNode",
    "DeltaPlanStep",
    "DeltaReplanBudget",
    "DeltaReplanDecision",
    "DeltaReplanLimits",
    "DeltaReplanner",
    "DeltaReplanResult",
    "DeltaReplanStopReason",
    "DiagnosticReceipt",
    "FormalDeltaReplanner",
    "FormalPlanReplanner",
    "FormalReplanner",
    "RepairCandidate",
    "RepairCandidateStatus",
    "RepairKind",
    "RepairOperation",
    "RepairProgress",
    "RepairRule",
    "RepairRuleKind",
    "RepairTransition",
    "ReplanBudget",
    "ReplanCancelled",
    "ReplanLimits",
    "ReplanResult",
    "ReplanStopReason",
    "ReplannerValidationError",
    "ResponsiveReplanDecision",
    "PlanSnapshot",
    "PlanStep",
    "delta_replan",
    "generate_plan_repairs",
    "replan_plan_delta",
    "replan_if_changed",
    "replan_for_signal",
    "replan_from_counterexample",
]
