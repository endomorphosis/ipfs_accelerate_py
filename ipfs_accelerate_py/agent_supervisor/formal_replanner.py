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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Final

from .formal_counterexamples import (
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
from .formal_verification_contracts import canonical_json, content_identity


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


class ReplannerValidationError(ValueError):
    """Raised when a repair request violates the replanner contract."""


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
        progress = payload.get("progress") or {}
        before = progress.get("before") or {}
        after = progress.get("after") or {}
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
        claimed = payload.get("transition_id") or payload.get("semantic_id")
        if claimed and claimed != result.transition_id:
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
    ) -> ReplanResult:
        """Run one bounded refinement and admit at most one selected transition."""

        value = _counterexample(counterexample)
        bundle = _source_bundle(source)
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
                prospective_packet = self._codex_packet(
                    selected.transition, value
                )
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
                try:
                    response = self.admission_callback(selected.transition)
                    admitted = response is not False
                except Exception:
                    admitted = False
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


FormalPlanReplanner = FormalReplanner


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
    )


replan_from_counterexample = generate_plan_repairs


__all__ = [
    "CODEX_REPAIR_PACKET_SCHEMA",
    "FORMAL_REPLANNER_VERSION",
    "REPAIR_CANDIDATE_SCHEMA",
    "REPAIR_TRANSITION_SCHEMA",
    "REPLAN_RESULT_SCHEMA",
    "CodexRepairPacket",
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
    "ReplanLimits",
    "ReplanResult",
    "ReplanStopReason",
    "ReplannerValidationError",
    "generate_plan_repairs",
    "replan_from_counterexample",
]
