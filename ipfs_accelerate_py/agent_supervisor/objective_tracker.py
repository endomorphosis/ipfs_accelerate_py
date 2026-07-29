"""Objective tracking document helpers for autonomous agent supervisors."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from hashlib import sha1, sha256
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .adaptive_goal_refiner import (
    AdaptiveGoalRefiner,
    AdaptiveGoalRefinementError,
    AdaptiveRefinementReceipt,
    AdaptiveRefinementRequest,
    AdaptiveRefinementResult,
    GoalDebtRecord,
    GoalQualityRecord,
    RefinementSignal,
)
from .formal_planning_contracts import FormalWorkPlan
from .goal_completion import (
    DEFAULT_CLOCK_SKEW_SECONDS,
    DEFAULT_EVIDENCE_FRESHNESS_SECONDS,
    GOAL_COMPLETION_SCHEMA_VERSION,
    GOAL_COMPLETION_MIGRATION_SCHEMA_VERSION,
    CompletionEvidence,
    GoalCompletionDecision,
    GoalState,
    is_legacy_completed_goal_state,
    migrate_legacy_goal_completion,
    evaluate_goal_completion,
    normalize_goal_state,
)
from .external_completion import (
    EXTERNAL_COMPLETION_EVIDENCE_SCHEMA,
    EXTERNAL_COMPLETION_VALIDATION_SCHEMA,
    ExternalCompletionAuthority,
    ExternalCompletionEvaluation,
    evaluate_external_completion_authority,
)
from .objective_graph import (
    DEFAULT_EMBEDDING_MIN_SCORE,
    OPAQUE_EVIDENCE_REQUIREMENT_PATTERN,
    ObjectiveFinding,
    ObjectiveGoal,
    ObjectiveGoalMaterializationPreview,
    canonical_interoperability_component,
    completion_evidence_source_decision,
    evidence_index,
    external_authority_goal_fence,
    goal_graph,
    normalize_field_key,
    objective_goal_content_id,
    objective_heap_content_id,
    objective_heap_schedule,
    parse_goal_heap,
    safe_bundle_key,
    resolve_scan_exclude_paths,
    split_terms,
    utc_now,
)
from .validation_commands import split_validation_commands
from .formal_verification_contracts import content_identity
from .goal_quality import (
    DebtSeverity,
    ObjectiveTypedGoals,
    lint_objective_markdown,
    lint_objective_typed_goals,
    migrate_objective_markdown,
)
from .validation_runtime import (
    build_validation_environment,
    validation_shell_command,
)
from .scan_receipts import RepositoryTreeIdentity, scan_identity
from .task_identity import canonical_content_cid, normalize_identity_text


DEFAULT_ULTIMATE_GOAL = (
    "Make this repository satisfy its stated objective with verifiable code, tests, docs, and runtime evidence."
)
DEFAULT_ROOT_EVIDENCE = (
    "objective goal graph",
    "bundle-local todo shards",
    "AST evidence dataset",
    "embedding evidence scan",
    "LLM merge conflict resolver",
)
DEFAULT_GOAL_PREFIX = os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_GOAL_PREFIX", "OBJ-G")
DEFAULT_TRACKING_DOCUMENT_TITLE = os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_DOCUMENT_TITLE", "Objective Heap")
DEFAULT_ROOT_GOAL_TITLE = os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_ROOT_TITLE", "Objective outcome")
OPEN_TASK_STATUSES_FOR_GOAL_COMPLETION = {"todo", "ready", "in_progress"}
TASK_GOAL_METADATA_KEYS = (
    "goal id",
    "goal ids",
    "goal packet goals",
    "graph parents",
)
OBJECTIVE_GOAL_QUALITY_REPORT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/objective-goal-quality-report@1"
)
OBJECTIVE_LAUNCH_QUALITY_SUMMARY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/objective-launch-quality-summary@1"
)
_COMPLETION_GATE_REQUIRED_CHECK_NAMES = frozenset(
    {
        "mandatory_coverage",
        "required_validations",
        "analyzer_health",
        "exhaustion_quorum",
        "analysis_terminal_state",
        "child_goals",
    }
)


def _completion_gate_projection_is_current(
    payload: Mapping[str, Any],
    *,
    repository_id: str = "",
    repository_tree: str = "",
    now: datetime | str | None = None,
    freshness_seconds: float = DEFAULT_EVIDENCE_FRESHNESS_SECONDS,
    clock_skew_seconds: float = DEFAULT_CLOCK_SKEW_SECONDS,
) -> bool:
    """Validate a serialized completion decision before backlog suppression.

    A durable summary is less trusted than the evaluator object that produced
    it.  Requiring the complete canonical projection prevents a skeletal
    ``verified`` mapping, an old passing decision, or a foreign-tree decision
    from silently removing its parent from supervisor refill.
    """

    required_fields = {
        "schema_version",
        "state",
        "verified",
        "tasks_complete",
        "acceptance_criteria",
        "missing_criteria",
        "invalid_criteria",
        "reason_codes",
        "actionable_reasons",
        "evidence_results",
        "completion_gate",
    }
    if not required_fields.issubset(payload):
        return False
    if payload.get("schema_version") != GOAL_COMPLETION_SCHEMA_VERSION:
        return False
    if payload.get("tasks_complete") is not True:
        return False

    def sequence(value: Any) -> list[Any] | None:
        if not isinstance(value, (list, tuple)):
            return None
        return list(value)

    criteria = sequence(payload.get("acceptance_criteria"))
    missing = sequence(payload.get("missing_criteria"))
    invalid = sequence(payload.get("invalid_criteria"))
    reason_codes = sequence(payload.get("reason_codes"))
    actionable = sequence(payload.get("actionable_reasons"))
    results = sequence(payload.get("evidence_results"))
    if any(
        item is None
        for item in (
            criteria,
            missing,
            invalid,
            reason_codes,
            actionable,
            results,
        )
    ):
        return False
    assert criteria is not None
    assert missing is not None
    assert invalid is not None
    assert reason_codes is not None
    assert actionable is not None
    assert results is not None
    criterion_keys = [
        " ".join(str(item or "").strip().lower().split())
        for item in criteria
    ]
    if (
        not criterion_keys
        or any(not item for item in criterion_keys)
        or len(criterion_keys) != len(set(criterion_keys))
        or missing
        or invalid
        or reason_codes
        or actionable
    ):
        return False

    result_keys: list[str] = []
    for result in results:
        if not isinstance(result, Mapping) or result.get("valid") is not True:
            return False
        evidence = result.get("evidence")
        if not isinstance(evidence, Mapping):
            return False
        criterion = " ".join(
            str(evidence.get("acceptance_criterion") or "")
            .strip()
            .lower()
            .split()
        )
        if not criterion:
            return False
        result_keys.append(criterion)
    if (
        len(result_keys) != len(criterion_keys)
        or len(result_keys) != len(set(result_keys))
        or set(result_keys) != set(criterion_keys)
    ):
        return False

    gate_value = payload.get("completion_gate")
    if not isinstance(gate_value, Mapping):
        return False
    gate = dict(gate_value)
    if (
        gate.get("schema_version") != GOAL_COMPLETION_SCHEMA_VERSION
        or gate.get("passed") is not True
        or sequence(gate.get("reason_codes")) != []
        or sequence(gate.get("fail_reason_codes")) != []
        or sequence(gate.get("actionable_reasons")) != []
    ):
        return False
    checks = sequence(gate.get("checks"))
    if checks is None:
        return False
    check_names = [
        str(check.get("name") or "").strip()
        for check in checks
        if isinstance(check, Mapping)
    ]
    if (
        len(check_names) != len(checks)
        or len(check_names) != len(set(check_names))
        or not _COMPLETION_GATE_REQUIRED_CHECK_NAMES.issubset(check_names)
        or any(check.get("passed") is not True for check in checks)
    ):
        return False

    evaluated_value = gate.get("evaluated_evidence")
    if not isinstance(evaluated_value, Mapping):
        return False
    evaluated = dict(evaluated_value)
    evaluated_criteria = sequence(evaluated.get("acceptance_criteria"))
    evaluated_results = sequence(evaluated.get("validation_evidence"))
    if evaluated_criteria is None or evaluated_results is None:
        return False
    evaluated_keys = [
        " ".join(str(item or "").strip().lower().split())
        for item in evaluated_criteria
    ]
    if evaluated_keys != criterion_keys or evaluated_results != results:
        return False
    for required_payload in (
        "coverage",
        "analyzer_health",
        "exhaustion_quorum",
    ):
        if (
            not isinstance(evaluated.get(required_payload), Mapping)
            or not evaluated[required_payload]
        ):
            return False

    evaluated_repository_id = str(evaluated.get("repository_id") or "").strip()
    evaluated_tree = str(evaluated.get("repository_tree") or "").strip()
    if not evaluated_repository_id or not evaluated_tree:
        return False
    if repository_id and evaluated_repository_id != str(repository_id):
        return False
    if repository_tree and evaluated_tree != str(repository_tree):
        return False

    def timestamp(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            parsed = value
        elif isinstance(value, str) and value.strip():
            try:
                parsed = datetime.fromisoformat(
                    value.strip().replace("Z", "+00:00")
                )
            except ValueError:
                return None
        else:
            return None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed.astimezone(timezone.utc)

    current = timestamp(now) if now is not None else datetime.now(timezone.utc)
    evaluated_at = timestamp(evaluated.get("evaluated_at"))
    if current is None or evaluated_at is None:
        return False
    if (
        isinstance(freshness_seconds, bool)
        or not isinstance(freshness_seconds, (int, float))
        or float(freshness_seconds) < 0
        or isinstance(clock_skew_seconds, bool)
        or not isinstance(clock_skew_seconds, (int, float))
        or float(clock_skew_seconds) < 0
    ):
        return False
    declared_freshness = evaluated.get("freshness_seconds")
    if (
        isinstance(declared_freshness, bool)
        or not isinstance(declared_freshness, (int, float))
        or float(declared_freshness) < 0
    ):
        return False
    max_age = timedelta(
        seconds=min(float(freshness_seconds), float(declared_freshness))
    )
    skew = timedelta(seconds=float(clock_skew_seconds))
    return bool(
        evaluated_at <= current + skew
        and current - evaluated_at <= max_age
    )


def completion_gate_actionable_goal_ids(
    goal_id: str,
    decision: GoalCompletionDecision | Mapping[str, Any] | None,
    *,
    repository_id: str = "",
    repository_tree: str = "",
    now: datetime | str | None = None,
    freshness_seconds: float = DEFAULT_EVIDENCE_FRESHNESS_SECONDS,
    clock_skew_seconds: float = DEFAULT_CLOCK_SKEW_SECONDS,
) -> tuple[str, ...]:
    """Project an incomplete completion gate back into objective scheduling.

    Completion-gate tasks are proof-producing projections, not replacements
    for their parent goal.  The parent therefore remains eligible for forced
    objective refill until a canonical decision says that it is verified,
    its completion gate passed, and no actionable reason remains.  Mapping
    inputs are accepted for durable supervisor records, but fail closed when
    any of those fields is absent.
    """

    normalized_goal_id = str(goal_id or "").strip()
    if not normalized_goal_id:
        raise ValueError("goal_id is required")
    if decision is None:
        return (normalized_goal_id,)
    if isinstance(decision, GoalCompletionDecision):
        payload = decision.to_dict()
    elif isinstance(decision, Mapping):
        payload = dict(decision)
    else:
        raise TypeError("decision must be a GoalCompletionDecision or mapping")
    gate_value = payload.get("completion_gate", payload.get("gate"))
    gate = dict(gate_value) if isinstance(gate_value, Mapping) else {}
    state = str(
        payload.get("state", payload.get("next_state", ""))
        or ""
    ).strip().lower()
    verified = bool(
        state == GoalState.VERIFIED_COMPLETE.value
        and payload.get("verified") is True
        and gate.get("passed") is True
        and _completion_gate_projection_is_current(
            payload,
            repository_id=repository_id,
            repository_tree=repository_tree,
            now=now,
            freshness_seconds=freshness_seconds,
            clock_skew_seconds=clock_skew_seconds,
        )
    )
    return () if verified else (normalized_goal_id,)


def _quality_terms(goal: ObjectiveGoal, *field_names: str) -> tuple[str, ...]:
    """Read one canonical JSON list or legacy delimited objective field."""

    for name in field_names:
        raw = str(goal.fields.get(name) or "").strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, list) and all(
            isinstance(item, str) for item in payload
        ):
            return tuple(
                sorted({item.strip() for item in payload if item.strip()})
            )
        return tuple(sorted(set(split_terms(raw))))
    return ()


def _quality_mapping(goal: ObjectiveGoal, *field_names: str) -> dict[str, Any]:
    for name in field_names:
        raw = str(goal.fields.get(name) or "").strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        if isinstance(payload, Mapping) and all(
            isinstance(key, str) for key in payload
        ):
            # The quality type applies canonical-JSON validation and copying.
            return dict(payload)
        return {}
    return {}


def _quality_nonnegative_integer(
    goal: ObjectiveGoal,
    *field_names: str,
    default: int = 0,
) -> int:
    for name in field_names:
        raw = str(goal.fields.get(name) or "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            return default
        return value if value >= 0 else default
    return default


@dataclass(frozen=True)
class ObjectiveTrackingResult:
    """Summary of objective tracking document mutations."""

    objective_path: Path
    created: bool
    appended_goal_ids: list[str]
    graph_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["objective_path"] = str(self.objective_path)
        if self.graph_path is not None:
            payload["graph_path"] = str(self.graph_path)
        return payload


@dataclass(frozen=True)
class ObjectiveGoalQualityReport:
    """Restart-safe quality/debt projection of one exact objective heap."""

    objective_heap_id: str
    quality_records: tuple[GoalQualityRecord, ...]

    def __post_init__(self) -> None:
        heap_id = str(self.objective_heap_id or "").strip()
        if not heap_id:
            raise ValueError("objective_heap_id is required")
        object.__setattr__(self, "objective_heap_id", heap_id)
        records = tuple(self.quality_records)
        if any(not isinstance(item, GoalQualityRecord) for item in records):
            raise TypeError(
                "quality_records must contain GoalQualityRecord values"
            )
        if len({item.goal_id for item in records}) != len(records):
            raise ValueError("quality_records contain duplicate goal IDs")
        object.__setattr__(
            self,
            "quality_records",
            tuple(sorted(records, key=lambda item: item.goal_id)),
        )

    @property
    def debt_records(self) -> tuple[GoalDebtRecord, ...]:
        return tuple(
            debt
            for quality in self.quality_records
            for debt in quality.debt_records
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": OBJECTIVE_GOAL_QUALITY_REPORT_SCHEMA,
            "version": 1,
            "objective_heap_id": self.objective_heap_id,
            "quality_records": tuple(
                item.to_dict() for item in self.quality_records
            ),
            "debt_records": tuple(item.to_dict() for item in self.debt_records),
        }

    @property
    def content_id(self) -> str:
        return content_identity(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "content_id": self.content_id}

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ObjectiveGoalQualityReport":
        if not isinstance(payload, Mapping):
            raise TypeError("objective goal-quality report must be an object")
        allowed = {
            "schema",
            "version",
            "content_id",
            "objective_heap_id",
            "quality_records",
            "debt_records",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(
                "unknown objective goal-quality report fields: "
                + ", ".join(unknown)
            )
        if payload.get("schema") != OBJECTIVE_GOAL_QUALITY_REPORT_SCHEMA:
            raise ValueError("unsupported objective goal-quality report schema")
        if payload.get("version") != 1:
            raise ValueError("unsupported objective goal-quality report version")
        quality_values = payload.get("quality_records")
        debt_values = payload.get("debt_records")
        if not isinstance(quality_values, Sequence) or isinstance(
            quality_values, (str, bytes, bytearray)
        ):
            raise ValueError("quality_records must be a sequence")
        if not isinstance(debt_values, Sequence) or isinstance(
            debt_values, (str, bytes, bytearray)
        ):
            raise ValueError("debt_records must be a sequence")
        quality_records = tuple(
            GoalQualityRecord.from_dict(item)
            if isinstance(item, Mapping)
            else (_raise_quality_report_value("quality_records"))
            for item in quality_values
        )
        result = cls(
            objective_heap_id=str(payload.get("objective_heap_id") or ""),
            quality_records=quality_records,
        )
        restored_debt = tuple(
            GoalDebtRecord.from_dict(item)
            if isinstance(item, Mapping)
            else (_raise_quality_report_value("debt_records"))
            for item in debt_values
        )
        if restored_debt != result.debt_records:
            raise ValueError(
                "objective goal-quality debt records do not match quality records"
            )
        identity = payload.get("content_id")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("objective goal-quality report identity is required")
        if identity != result.content_id:
            raise ValueError(
                "objective goal-quality report content identity does not match"
            )
        return result


def _raise_quality_report_value(field_name: str) -> Any:
    raise ValueError(f"{field_name} must contain objects")


def objective_goal_quality_record(
    goal: ObjectiveGoal,
    *,
    breadth: int = 1,
    default_max_breadth: int = 8,
) -> GoalQualityRecord:
    """Project one markdown goal into the reviewed adaptive quality schema."""

    if not isinstance(goal, ObjectiveGoal):
        raise TypeError("goal must be an ObjectiveGoal")
    outcome = str(
        goal.fields.get("outcome")
        or goal.fields.get("goal")
        or goal.fields.get("objective")
        or goal.title
        or ""
    ).strip()
    scope_ids = tuple(
        sorted(
            set(
                _quality_terms(
                    goal,
                    "scope_ids_json",
                    "scope_ids",
                    "scope",
                )
                + tuple(goal.predicted_files)
                + tuple(goal.predicted_symbols)
            )
        )
    )
    acceptance = _quality_terms(
        goal,
        "acceptance_criteria_json",
        "acceptance_criteria",
        "acceptance",
    )
    # ``Evidence`` is the objective heap's canonical declaration of the
    # artifacts or child goals that produce evidence for a goal.  Preserve
    # those bindings in the compatibility quality report in addition to the
    # newer explicit producer fields; otherwise a supervisor can report a
    # goal while silently dropping the evidence obligation that caused it to
    # be scheduled.
    producers = set(goal.required_evidence)
    producers.update(
        _quality_terms(
            goal,
            "evidence_producer_ids_json",
            "evidence_producer_ids",
            "evidence_producers",
            "producing_task_or_scan",
            "produced_by",
        )
    )
    metadata_producer = str(
        goal.completion_evidence_metadata.get("producer") or ""
    ).strip()
    if metadata_producer:
        producers.add(metadata_producer)
    validation = set(goal.validation_commands)
    validation.update(
        _quality_terms(
            goal,
            "validation_policy_json",
            "validation_policy",
            "validation_ids",
        )
    )
    resource_envelope = _quality_mapping(
        goal, "resource_envelope_json", "resource_envelope"
    )
    if not resource_envelope:
        resource_envelope = {
            key: value
            for key, value in {
                "resource_class": str(
                    goal.fields.get("resource_class") or ""
                ).strip(),
                "estimated_tokens": str(
                    goal.fields.get("estimated_tokens") or ""
                ).strip(),
                "estimated_runtime": str(
                    goal.fields.get("estimated_runtime") or ""
                ).strip(),
                "estimated_memory": str(
                    goal.fields.get("estimated_memory") or ""
                ).strip(),
                "artifact_budget": str(
                    goal.fields.get("artifact_budget") or ""
                ).strip(),
            }.items()
            if value
        }
    refinement_budget = _quality_mapping(
        goal, "refinement_budget_json", "refinement_budget"
    )
    if not refinement_budget:
        refinement_budget = {
            key: value
            for key, value in {
                "max_depth": str(
                    goal.fields.get("max_refinement_depth")
                    or goal.fields.get("refinement_depth_limit")
                    or ""
                ).strip(),
                "max_children": str(
                    goal.fields.get("max_refinement_children")
                    or goal.fields.get("refinement_breadth_limit")
                    or ""
                ).strip(),
            }.items()
            if value
        }
    explicit_breadth = _quality_nonnegative_integer(
        goal, "breadth", default=max(1, breadth)
    )
    max_breadth = _quality_nonnegative_integer(
        goal, "max_breadth", "refinement_breadth_limit",
        default=default_max_breadth,
    )
    return GoalQualityRecord(
        goal_id=goal.goal_id,
        outcome=outcome,
        scope_ids=scope_ids,
        assumption_ids=_quality_terms(
            goal,
            "assumption_ids_json",
            "assumptions_json",
            "assumption_ids",
            "assumptions",
        ),
        non_goals=_quality_terms(
            goal, "non_goals_json", "non_goals", "non_goal"
        ),
        acceptance_criteria=acceptance,
        evidence_producer_ids=tuple(sorted(producers)),
        validation_ids=tuple(sorted(validation)),
        freshness_horizon_seconds=_quality_nonnegative_integer(
            goal,
            "freshness_horizon_seconds",
            "evidence_freshness_seconds",
        ),
        resource_envelope=resource_envelope,
        refinement_budget=refinement_budget,
        ambiguities=_quality_terms(
            goal, "ambiguities_json", "ambiguities", "ambiguity"
        ),
        stale_evidence_ids=_quality_terms(
            goal, "stale_evidence_ids_json", "stale_evidence", "stale_receipts"
        ),
        uncovered_acceptance_criteria=_quality_terms(
            goal,
            "uncovered_acceptance_criteria_json",
            "uncovered_acceptance_criteria",
            "uncovered_criteria",
        ),
        unsupported_semantics=_quality_terms(
            goal,
            "unsupported_semantics_json",
            "unsupported_semantics",
        ),
        breadth=max(1, explicit_breadth),
        max_breadth=max(1, max_breadth),
    )


def build_objective_goal_quality_report(
    objective_text: str,
    *,
    default_max_breadth: int = 8,
) -> ObjectiveGoalQualityReport:
    """Build a deterministic report without mutating the objective heap."""

    if not isinstance(objective_text, str):
        raise TypeError("objective_text must be a string")
    goals = parse_goal_heap(objective_text)
    child_counts: dict[str, int] = {}
    for goal in goals:
        for parent_id in goal.parent_goal_ids:
            child_counts[parent_id] = child_counts.get(parent_id, 0) + 1
    return ObjectiveGoalQualityReport(
        objective_heap_id=objective_heap_content_id(objective_text),
        quality_records=tuple(
            objective_goal_quality_record(
                goal,
                breadth=max(1, child_counts.get(goal.goal_id, 0)),
                default_max_breadth=default_max_breadth,
            )
            for goal in goals
        ),
    )


def write_objective_goal_quality_report(
    objective_path: Path,
    report_path: Path,
    *,
    default_max_breadth: int = 8,
) -> ObjectiveGoalQualityReport:
    """Atomically persist an exact-heap quality snapshot for restart reuse."""

    if objective_path.resolve() == report_path.resolve():
        raise ValueError(
            "goal-quality report path must not overwrite the objective heap"
        )
    text = objective_path.read_text(encoding="utf-8")
    report = build_objective_goal_quality_report(
        text, default_max_breadth=default_max_breadth
    )
    _atomic_write_json(report_path, report.to_dict())
    return report


def load_objective_goal_quality_report(
    report_path: Path,
    *,
    objective_path: Path | None = None,
) -> ObjectiveGoalQualityReport:
    """Restore a report fail-closed and optionally reject a stale heap."""

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid objective goal-quality report: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("objective goal-quality report must contain an object")
    report = ObjectiveGoalQualityReport.from_dict(payload)
    if objective_path is not None:
        objective_text = objective_path.read_text(encoding="utf-8")
        current_id = objective_heap_content_id(objective_text)
        if report.objective_heap_id != current_id:
            raise ValueError(
                "objective goal-quality report is stale for the current heap"
            )
        expected_goal_ids = {
            goal.goal_id for goal in parse_goal_heap(objective_text)
        }
        report_goal_ids = {
            record.goal_id for record in report.quality_records
        }
        if report_goal_ids != expected_goal_ids:
            missing = sorted(expected_goal_ids - report_goal_ids)
            unexpected = sorted(report_goal_ids - expected_goal_ids)
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if unexpected:
                details.append("unexpected: " + ", ".join(unexpected))
            raise ValueError(
                "objective goal-quality report goal coverage does not match "
                f"the current heap ({'; '.join(details)})"
            )
    return report


def build_objective_typed_goals(
    objective_text: str,
    *,
    lossless: bool = True,
) -> ObjectiveTypedGoals:
    """Build the versioned typed sidecar for one exact objective heap."""

    if not isinstance(objective_text, str):
        raise TypeError("objective_text must be a string")
    if lossless:
        return migrate_objective_markdown(objective_text)
    from .goal_quality import project_objective_markdown

    return ObjectiveTypedGoals(
        objective_heap_id=objective_heap_content_id(objective_text),
        goals=project_objective_markdown(objective_text, lossless=False),
    )


def write_objective_typed_goals(
    objective_path: Path,
    sidecar_path: Path,
    *,
    lossless: bool = True,
) -> ObjectiveTypedGoals:
    """Atomically persist a heap-bound typed goal sidecar."""

    if objective_path.resolve() == sidecar_path.resolve():
        raise ValueError(
            "typed goal sidecar path must not overwrite the objective heap"
        )
    text = objective_path.read_text(encoding="utf-8")
    document = build_objective_typed_goals(text, lossless=lossless)
    _atomic_write_json(sidecar_path, document.to_dict())
    return document


def load_objective_typed_goals(
    sidecar_path: Path,
    *,
    objective_path: Path | None = None,
) -> ObjectiveTypedGoals:
    """Restore a typed goal sidecar and optionally reject a stale heap."""

    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid objective typed goals sidecar: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("objective typed goals sidecar must contain an object")
    document = ObjectiveTypedGoals.from_dict(payload)
    if objective_path is not None:
        current_id = objective_heap_content_id(
            objective_path.read_text(encoding="utf-8")
        )
        if document.objective_heap_id != current_id:
            raise ValueError(
                "objective typed goals sidecar is stale for the current heap"
            )
    return document


@dataclass(frozen=True)
class ObjectiveLaunchQualitySummary:
    """Launcher-facing structural gate plus explicit typed quality debt.

    This summary never grants mutation or completion authority.  Typed
    admission is reported only when every migrated goal has zero
    error-severity debt; otherwise the launcher must stay on the structural
    legacy path and surface debt counts.
    """

    objective_heap_id: str
    goal_count: int
    legacy_structure_accepted: bool
    compatibility_report_id: str
    strict_typed_accepted: int
    strict_typed_rejected: int
    strict_typed_debt: Mapping[str, int]
    strict_typed_error_debt: Mapping[str, int]
    strict_typed_required: bool
    typed_admission_claimed: bool
    typed_sidecar_content_id: str
    admission_path: str

    def __post_init__(self) -> None:
        heap_id = str(self.objective_heap_id or "").strip()
        if not heap_id:
            raise ValueError("objective_heap_id is required")
        object.__setattr__(self, "objective_heap_id", heap_id)
        for name in (
            "goal_count",
            "strict_typed_accepted",
            "strict_typed_rejected",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in ("strict_typed_debt", "strict_typed_error_debt"):
            raw = getattr(self, name)
            if not isinstance(raw, Mapping):
                raise TypeError(f"{name} must be a mapping")
            normalized = {
                str(key): int(value)
                for key, value in sorted(raw.items(), key=lambda item: str(item[0]))
            }
            object.__setattr__(self, name, dict(normalized))
        for name in (
            "compatibility_report_id",
            "typed_sidecar_content_id",
            "admission_path",
        ):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ValueError(f"{name} is required")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self, "legacy_structure_accepted", bool(self.legacy_structure_accepted)
        )
        object.__setattr__(
            self, "strict_typed_required", bool(self.strict_typed_required)
        )
        claimed = bool(self.typed_admission_claimed)
        if claimed and (
            self.strict_typed_rejected
            or not self.legacy_structure_accepted
            or self.admission_path != "typed_sidecar"
        ):
            raise ValueError(
                "typed admission cannot be claimed while structural or typed debt remains"
            )
        if not claimed and self.admission_path == "typed_sidecar":
            raise ValueError(
                "typed_sidecar admission path requires typed_admission_claimed"
            )
        object.__setattr__(self, "typed_admission_claimed", claimed)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": OBJECTIVE_LAUNCH_QUALITY_SUMMARY_SCHEMA,
            "version": 1,
            "objective_heap_id": self.objective_heap_id,
            "goal_count": self.goal_count,
            "legacy_structure_accepted": self.legacy_structure_accepted,
            "compatibility_report_id": self.compatibility_report_id,
            "strict_typed_accepted": self.strict_typed_accepted,
            "strict_typed_rejected": self.strict_typed_rejected,
            "strict_typed_debt": dict(self.strict_typed_debt),
            "strict_typed_error_debt": dict(self.strict_typed_error_debt),
            "strict_typed_required": self.strict_typed_required,
            "typed_admission_claimed": self.typed_admission_claimed,
            "typed_sidecar_content_id": self.typed_sidecar_content_id,
            "admission_path": self.admission_path,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "content_id": self.content_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ObjectiveLaunchQualitySummary":
        if not isinstance(payload, Mapping):
            raise TypeError("objective launch quality summary must be an object")
        allowed = {
            "schema",
            "version",
            "content_id",
            "objective_heap_id",
            "goal_count",
            "legacy_structure_accepted",
            "compatibility_report_id",
            "strict_typed_accepted",
            "strict_typed_rejected",
            "strict_typed_debt",
            "strict_typed_error_debt",
            "strict_typed_required",
            "typed_admission_claimed",
            "typed_sidecar_content_id",
            "admission_path",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(
                "unknown objective launch quality summary fields: "
                + ", ".join(unknown)
            )
        if payload.get("schema") != OBJECTIVE_LAUNCH_QUALITY_SUMMARY_SCHEMA:
            raise ValueError("unsupported objective launch quality summary schema")
        if payload.get("version") != 1:
            raise ValueError("unsupported objective launch quality summary version")
        result = cls(
            objective_heap_id=str(payload.get("objective_heap_id") or ""),
            goal_count=int(payload.get("goal_count") or 0),
            legacy_structure_accepted=bool(payload.get("legacy_structure_accepted")),
            compatibility_report_id=str(payload.get("compatibility_report_id") or ""),
            strict_typed_accepted=int(payload.get("strict_typed_accepted") or 0),
            strict_typed_rejected=int(payload.get("strict_typed_rejected") or 0),
            strict_typed_debt=payload.get("strict_typed_debt") or {},
            strict_typed_error_debt=payload.get("strict_typed_error_debt") or {},
            strict_typed_required=bool(payload.get("strict_typed_required")),
            typed_admission_claimed=bool(payload.get("typed_admission_claimed")),
            typed_sidecar_content_id=str(payload.get("typed_sidecar_content_id") or ""),
            admission_path=str(payload.get("admission_path") or ""),
        )
        identity = payload.get("content_id")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("objective launch quality summary identity is required")
        if identity != result.content_id:
            raise ValueError(
                "objective launch quality summary content identity does not match"
            )
        return result


def build_objective_launch_quality_summary(
    objective_text: str,
    *,
    require_typed: bool = False,
    claim_typed_admission: bool = False,
) -> ObjectiveLaunchQualitySummary:
    """Report legacy structural readiness and typed debt for one heap.

    Until callers explicitly claim typed admission, the summary stays on the
    documented structural legacy path and reports quality debt from the
    conservative Markdown projection rather than claiming typed admission.
    """

    if not isinstance(objective_text, str):
        raise TypeError("objective_text must be a string")
    compatibility = build_objective_goal_quality_report(objective_text)
    legacy_reports = lint_objective_markdown(objective_text, lossless=False)
    typed_document = migrate_objective_markdown(objective_text)
    if typed_document.objective_heap_id != compatibility.objective_heap_id:
        raise ValueError(
            "typed sidecar heap identity diverged from compatibility report"
        )
    sidecar_reports = lint_objective_typed_goals(typed_document)
    sidecar_accepted = sum(1 for report in sidecar_reports if report.accepted)
    sidecar_rejected = len(sidecar_reports) - sidecar_accepted

    # Legacy projection debt is the diagnostic signal for the structural path.
    legacy_debt: dict[str, int] = {}
    legacy_error_debt: dict[str, int] = {}
    for report in legacy_reports:
        for debt in report.debt:
            legacy_debt[debt.code.value] = legacy_debt.get(debt.code.value, 0) + 1
            if debt.severity is DebtSeverity.ERROR:
                legacy_error_debt[debt.code.value] = (
                    legacy_error_debt.get(debt.code.value, 0) + 1
                )
    legacy_accepted = sum(1 for report in legacy_reports if report.accepted)
    legacy_rejected = len(legacy_reports) - legacy_accepted

    sidecar_debt: dict[str, int] = {}
    sidecar_error_debt: dict[str, int] = {}
    for report in sidecar_reports:
        for debt in report.debt:
            sidecar_debt[debt.code.value] = sidecar_debt.get(debt.code.value, 0) + 1
            if debt.severity is DebtSeverity.ERROR:
                sidecar_error_debt[debt.code.value] = (
                    sidecar_error_debt.get(debt.code.value, 0) + 1
                )

    # Structural acceptance is owned by hierarchy checks outside this builder.
    legacy_structure_accepted = True
    can_claim = (
        claim_typed_admission
        and sidecar_rejected == 0
        and legacy_structure_accepted
        and bool(typed_document.goals)
    )
    if require_typed and sidecar_rejected:
        can_claim = False
    if can_claim:
        accepted, rejected = sidecar_accepted, sidecar_rejected
        debt_counts, error_counts = sidecar_debt, sidecar_error_debt
        admission_path = "typed_sidecar"
    else:
        accepted, rejected = legacy_accepted, legacy_rejected
        debt_counts, error_counts = legacy_debt, legacy_error_debt
        admission_path = "structural_legacy"
    return ObjectiveLaunchQualitySummary(
        objective_heap_id=compatibility.objective_heap_id,
        goal_count=len(typed_document.goals),
        legacy_structure_accepted=legacy_structure_accepted,
        compatibility_report_id=compatibility.content_id,
        strict_typed_accepted=accepted,
        strict_typed_rejected=rejected,
        strict_typed_debt=debt_counts,
        strict_typed_error_debt=error_counts,
        strict_typed_required=require_typed,
        typed_admission_claimed=can_claim,
        typed_sidecar_content_id=typed_document.content_id,
        admission_path=admission_path,
    )


OBJECTIVE_REFINEMENT_EVENT_STATE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/objective-refinement-event-state@1"
)


class ObjectiveRefinementPollDecision(str, Enum):
    """Closed outcomes for an event-driven objective poll."""

    NO_SEMANTIC_CHANGE = "no_semantic_change"
    REFINEMENT_EVALUATED = "refinement_evaluated"
    DELTA_COMMITTED = "delta_committed"


@dataclass(frozen=True)
class ObjectiveRefinementEventState:
    """Restart-safe semantic cursor bound to one immutable root context."""

    root_goal_id: str
    root_goal_content_id: str
    assumption_ids: tuple[str, ...]
    policy_id: str
    semantic_event_ids: Mapping[str, str] = field(default_factory=dict)
    last_receipt_id: str = ""
    retry_after: int = 0

    def __post_init__(self) -> None:
        for name in (
            "root_goal_id",
            "root_goal_content_id",
            "policy_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
            value = value.strip()
            if not value:
                raise ValueError(f"{name} is required")
            object.__setattr__(self, name, value)
        if isinstance(
            self.assumption_ids, (str, bytes, bytearray, memoryview)
        ) or any(not isinstance(item, str) for item in self.assumption_ids):
            raise ValueError("assumption_ids must be a sequence of strings")
        assumptions = tuple(
            sorted(
                {
                    item.strip()
                    for item in self.assumption_ids
                    if item.strip()
                }
            )
        )
        object.__setattr__(self, "assumption_ids", assumptions)
        if not isinstance(self.semantic_event_ids, Mapping) or any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or not value.strip()
            for key, value in self.semantic_event_ids.items()
        ):
            raise ValueError(
                "semantic_event_ids must be a string-to-string mapping"
            )
        object.__setattr__(
            self,
            "semantic_event_ids",
            {
                key: self.semantic_event_ids[key]
                for key in sorted(self.semantic_event_ids)
            },
        )
        if not isinstance(self.last_receipt_id, str):
            raise ValueError("last_receipt_id must be a string")
        object.__setattr__(
            self, "last_receipt_id", self.last_receipt_id.strip()
        )
        if (
            isinstance(self.retry_after, bool)
            or not isinstance(self.retry_after, int)
            or self.retry_after < 0
        ):
            raise ValueError("retry_after must be a non-negative integer")

    @property
    def content_id(self) -> str:
        return content_identity(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": OBJECTIVE_REFINEMENT_EVENT_STATE_SCHEMA,
            "root_goal_id": self.root_goal_id,
            "root_goal_content_id": self.root_goal_content_id,
            "assumption_ids": self.assumption_ids,
            "policy_id": self.policy_id,
            "semantic_event_ids": dict(self.semantic_event_ids),
            "last_receipt_id": self.last_receipt_id,
            "retry_after": self.retry_after,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "content_id": self.content_id}

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ObjectiveRefinementEventState":
        expected = {
            "schema",
            "root_goal_id",
            "root_goal_content_id",
            "assumption_ids",
            "policy_id",
            "semantic_event_ids",
            "last_receipt_id",
            "retry_after",
            "content_id",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError(
                "objective refinement event state must use the closed schema"
            )
        if payload.get("schema") != OBJECTIVE_REFINEMENT_EVENT_STATE_SCHEMA:
            raise ValueError("unsupported objective refinement event state schema")
        result = cls(
            root_goal_id=payload.get("root_goal_id", ""),
            root_goal_content_id=payload.get("root_goal_content_id", ""),
            assumption_ids=tuple(payload.get("assumption_ids") or ()),
            policy_id=payload.get("policy_id", ""),
            semantic_event_ids=payload.get("semantic_event_ids") or {},
            last_receipt_id=payload.get("last_receipt_id", ""),
            retry_after=payload.get("retry_after", 0),
        )
        if payload.get("content_id") != result.content_id:
            raise ValueError(
                "objective refinement event state identity does not match"
            )
        return result


@dataclass(frozen=True)
class ObjectiveRefinementPollResult:
    decision: ObjectiveRefinementPollDecision
    changed_signal_ids: tuple[str, ...]
    state_content_id: str
    refinement_result: AdaptiveRefinementResult | None = None
    objective_written: bool = False

    @property
    def model_called(self) -> bool:
        return bool(
            self.refinement_result is not None
            and self.refinement_result.model_called
        )


ObjectiveDeltaCommitter = Callable[
    [FormalWorkPlan, AdaptiveRefinementReceipt], None
]


class ObjectiveRefinementEventTracker:
    """Gate adaptive refinement on persisted semantic event changes.

    Merely polling this object is side-effect free when every event cursor is
    unchanged.  Changed delivery timestamps and occurrence counts are already
    excluded from :attr:`RefinementSignal.evidence_id`, so delivery churn
    cannot reach candidate generation.
    """

    def __init__(
        self,
        refiner: AdaptiveGoalRefiner,
        state_path: Path,
        *,
        objective_committer: ObjectiveDeltaCommitter | None = None,
    ) -> None:
        if not isinstance(refiner, AdaptiveGoalRefiner):
            raise TypeError("refiner must be an AdaptiveGoalRefiner")
        if objective_committer is not None and not callable(objective_committer):
            raise TypeError("objective_committer must be callable")
        self.refiner = refiner
        self.state_path = Path(state_path)
        self.objective_committer = objective_committer
        self._lock = threading.RLock()

    @contextmanager
    def _transaction(self):
        lock_path = self.state_path.with_suffix(self.state_path.suffix + ".lock")
        with self._lock:
            try:
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                handle = lock_path.open("a+", encoding="utf-8")
            except OSError as exc:
                raise ValueError(
                    f"could not lock objective refinement event state: {exc}"
                ) from exc
            with handle:
                try:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _event_slot(signal: RefinementSignal) -> str:
        return content_identity(
            {
                "kind": signal.kind.value,
                "subject_id": signal.subject_id,
            }
        )

    def _load(self) -> ObjectiveRefinementEventState | None:
        if not self.state_path.exists():
            return None
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid objective refinement event state: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ValueError(
                "objective refinement event state must contain an object"
            )
        return ObjectiveRefinementEventState.from_dict(payload)

    def poll(
        self, request: AdaptiveRefinementRequest
    ) -> ObjectiveRefinementPollResult:
        """Evaluate only semantically changed signals from one frozen request."""

        if not isinstance(request, AdaptiveRefinementRequest):
            raise TypeError("request must be an AdaptiveRefinementRequest")
        with self._transaction():
            state = self._load()
            if state is not None:
                frozen_mismatch = (
                    state.root_goal_id != request.root_goal_id
                    or state.root_goal_content_id != request.root_goal_content_id
                    or state.assumption_ids != request.assumption_ids
                    or state.policy_id != self.refiner.policy.content_id
                )
                if frozen_mismatch:
                    raise AdaptiveGoalRefinementError(
                        "event state does not match the frozen root, assumptions, "
                        "or refinement policy"
                    )
            prior_events = dict(
                state.semantic_event_ids if state is not None else {}
            )
            current_slots: dict[str, list[str]] = {}
            for signal in request.signals:
                current_slots.setdefault(
                    self._event_slot(signal), []
                ).append(signal.evidence_id)
            current_event_ids = {
                slot: content_identity(
                    {
                        "slot": slot,
                        "event_ids": tuple(sorted(event_ids)),
                    }
                )
                for slot, event_ids in current_slots.items()
            }
            changed_slots = {
                slot
                for slot, event_id in current_event_ids.items()
                if prior_events.get(slot) != event_id
            }
            changed = tuple(
                signal
                for signal in request.signals
                if self._event_slot(signal) in changed_slots
            )
            if not changed:
                return ObjectiveRefinementPollResult(
                    decision=ObjectiveRefinementPollDecision.NO_SEMANTIC_CHANGE,
                    changed_signal_ids=(),
                    state_content_id=state.content_id if state is not None else "",
                )

            changed_request = replace(request, signals=changed)
            result = self.refiner.refine(changed_request)
            objective_written = False
            if result.admitted_plan is not None and self.objective_committer is not None:
                quality_report = result.receipt.quality_lint_report
                if quality_report is None or not quality_report.accepted:
                    raise AdaptiveGoalRefinementError(
                        "admitted delta is missing an accepted quality lint"
                    )
                self.objective_committer(result.admitted_plan, result.receipt)
                objective_written = True

            for slot in changed_slots:
                prior_events[slot] = current_event_ids[slot]
            next_state = ObjectiveRefinementEventState(
                root_goal_id=request.root_goal_id,
                root_goal_content_id=request.root_goal_content_id,
                assumption_ids=request.assumption_ids,
                policy_id=self.refiner.policy.content_id,
                semantic_event_ids=prior_events,
                last_receipt_id=result.receipt.receipt_id,
                retry_after=result.receipt.retry_after,
            )
            _atomic_write_json(self.state_path, next_state.to_dict())
            return ObjectiveRefinementPollResult(
                decision=(
                    ObjectiveRefinementPollDecision.DELTA_COMMITTED
                    if objective_written
                    else ObjectiveRefinementPollDecision.REFINEMENT_EVALUATED
                ),
                changed_signal_ids=tuple(
                    signal.evidence_id for signal in changed
                ),
                state_content_id=next_state.content_id,
                refinement_result=result,
                objective_written=objective_written,
            )


@dataclass(frozen=True)
class ObjectiveCompletionResult:
    """Summary of objective goals reconciled from repository evidence."""

    objective_path: Path
    completed_goal_ids: list[str]
    active_goal_count: int
    completed_goal_count: int
    completion_evidence: dict[str, dict[str, list[str]]]
    validation_results: dict[str, dict[str, Any]]
    provisional_goal_ids: list[str] = field(default_factory=list)
    verified_goal_ids: list[str] = field(default_factory=list)
    reopened_goal_ids: list[str] = field(default_factory=list)
    analysis_inconclusive_goal_ids: list[str] = field(default_factory=list)
    blocked_goal_ids: list[str] = field(default_factory=list)
    state_counts: dict[str, int] = field(default_factory=dict)
    decisions: dict[str, dict[str, Any]] = field(default_factory=dict)
    migration: dict[str, Any] = field(default_factory=dict)
    external_completion: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["objective_path"] = str(self.objective_path)
        return payload


@dataclass(frozen=True)
class ObjectiveGoalMigrationResult:
    """Previewable and resumable migration of legacy completion claims."""

    objective_path: Path
    preview: bool
    scanned_goal_count: int
    candidate_goal_ids: list[str]
    migrated_goal_ids: list[str]
    provisional_goal_ids: list[str]
    verified_goal_ids: list[str]
    remaining_goal_ids: list[str]
    records: list[dict[str, Any]]
    schema_version: int = GOAL_COMPLETION_MIGRATION_SCHEMA_VERSION

    @property
    def changed(self) -> bool:
        return bool(self.migrated_goal_ids) and not self.preview

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update({
            "schema": "ipfs_accelerate_py.agent_supervisor.objective_goal_migration@1",
            "objective_path": str(self.objective_path),
            "changed": self.changed,
        })
        return payload


class ObjectiveMaterializationTransactionState(str, Enum):
    """Durable state of an objective-heap materialization transaction."""

    PREPARED = "prepared"
    COMMITTED = "committed"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ObjectiveMaterializationTransactionResult:
    """Outcome of applying one immutable objective materialization preview."""

    objective_path: Path
    journal_path: Path
    transaction_id: str
    state: ObjectiveMaterializationTransactionState
    admitted_proposal_ids: tuple[str, ...] = ()
    changed: bool = False
    resumed: bool = False
    reason_codes: tuple[str, ...] = ()
    base_heap_content_id: str = ""
    candidate_heap_content_id: str = ""
    repository_tree_id: str = ""
    root_goal_id: str = ""
    root_content_id: str = ""
    epoch_id: str = ""
    mapped_goal_ids: tuple[str, ...] = ()

    @property
    def committed(self) -> bool:
        return self.state is ObjectiveMaterializationTransactionState.COMMITTED

    @property
    def resumable(self) -> bool:
        return self.state is not ObjectiveMaterializationTransactionState.COMMITTED

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "schema": (
                    "ipfs_accelerate_py.agent_supervisor."
                    "objective_materialization_transaction_result@1"
                ),
                "objective_path": str(self.objective_path),
                "journal_path": str(self.journal_path),
                "state": self.state.value,
                "admitted_proposal_ids": list(self.admitted_proposal_ids),
                "mapped_goal_ids": list(self.mapped_goal_ids),
                "reason_codes": list(self.reason_codes),
                "committed": self.committed,
                "resumable": self.resumable,
            }
        )
        return payload


OBJECTIVE_EVIDENCE_PROJECTION_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.objective_evidence_projection.v1"
)


@dataclass(frozen=True)
class ObjectiveEvidenceProjection:
    """Stable owner of one evidence requirement in the objective heap.

    Supervisor backlog records can outlive objective refinement.  Consumers
    must therefore resolve an old aggregate goal to the one current child that
    owns the requirement instead of recreating the aggregate or appending
    another child.  This projection is intentionally read-only and
    content-addressed to the exact heap.
    """

    requirement_id: str
    goal_id: str
    parent_goal_id: str
    objective_heap_id: str
    goal_content_id: str

    def __post_init__(self) -> None:
        for name in (
            "requirement_id",
            "goal_id",
            "parent_goal_id",
            "objective_heap_id",
            "goal_content_id",
        ):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ValueError(f"{name} is required")
            object.__setattr__(self, name, value)

    @property
    def projection_id(self) -> str:
        return content_identity(
            {
                "schema": OBJECTIVE_EVIDENCE_PROJECTION_SCHEMA,
                "version": 1,
                "requirement_id": self.requirement_id,
                "goal_id": self.goal_id,
                "parent_goal_id": self.parent_goal_id,
                "objective_heap_id": self.objective_heap_id,
                "goal_content_id": self.goal_content_id,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": OBJECTIVE_EVIDENCE_PROJECTION_SCHEMA,
            "version": 1,
            "requirement_id": self.requirement_id,
            "goal_id": self.goal_id,
            "parent_goal_id": self.parent_goal_id,
            "objective_heap_id": self.objective_heap_id,
            "goal_content_id": self.goal_content_id,
            "projection_id": self.projection_id,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ObjectiveEvidenceProjection":
        allowed = {
            "schema",
            "version",
            "requirement_id",
            "goal_id",
            "parent_goal_id",
            "objective_heap_id",
            "goal_content_id",
            "projection_id",
        }
        unknown = sorted(str(key) for key in payload if str(key) not in allowed)
        if unknown:
            raise ValueError(
                "objective evidence projection contains unknown fields: "
                + ", ".join(unknown)
            )
        if (
            payload.get("schema") != OBJECTIVE_EVIDENCE_PROJECTION_SCHEMA
            or payload.get("version") != 1
        ):
            raise ValueError("unsupported objective evidence projection schema")
        result = cls(
            requirement_id=str(payload.get("requirement_id") or ""),
            goal_id=str(payload.get("goal_id") or ""),
            parent_goal_id=str(payload.get("parent_goal_id") or ""),
            objective_heap_id=str(payload.get("objective_heap_id") or ""),
            goal_content_id=str(payload.get("goal_content_id") or ""),
        )
        if payload.get("projection_id") != result.projection_id:
            raise ValueError("objective evidence projection identity does not match")
        return result


def resolve_objective_evidence_projection(
    objective_text: str,
    *,
    requirement_id: str,
    expected_parent_goal_id: str = "",
    expected_goal_id: str = "",
) -> ObjectiveEvidenceProjection:
    """Resolve exactly one current heap owner for an evidence requirement.

    Ambiguous, detached, or unexpectedly renamed owners fail closed.  The
    caller may omit either expected ID for general use, while objective-gap
    repairs should bind both IDs so stale supervisor metadata cannot silently
    redirect completion evidence.
    """

    requirement = str(requirement_id or "").strip()
    if not requirement:
        raise ValueError("requirement_id is required")
    goals = parse_goal_heap(objective_text)
    owners = [
        goal
        for goal in goals
        if requirement in {str(item).strip() for item in goal.required_evidence}
    ]
    if not owners:
        raise ValueError(
            f"objective heap has no owner for evidence requirement {requirement}"
        )
    goals_by_id = {goal.goal_id: goal for goal in goals}

    def ancestor_ids(goal: ObjectiveGoal) -> set[str]:
        pending = list(goal.parent_goal_ids)
        result: set[str] = set()
        while pending:
            goal_id = str(pending.pop()).strip()
            if not goal_id or goal_id in result:
                continue
            result.add(goal_id)
            parent = goals_by_id.get(goal_id)
            if parent is not None:
                pending.extend(parent.parent_goal_ids)
        return result

    expected_goal = str(expected_goal_id or "").strip()
    if expected_goal:
        owner = next(
            (goal for goal in owners if goal.goal_id == expected_goal),
            None,
        )
        if owner is None:
            owner_ids = ", ".join(sorted(goal.goal_id for goal in owners))
            raise ValueError(
                f"evidence requirement {requirement} is owned by "
                f"{owner_ids}, expected {expected_goal}"
            )
        ancestors = ancestor_ids(owner)
        incomparable = sorted(
            goal.goal_id
            for goal in owners
            if goal.goal_id != owner.goal_id and goal.goal_id not in ancestors
        )
        if incomparable:
            raise ValueError(
                f"objective heap has ambiguous owners for evidence requirement "
                f"{requirement}: {', '.join(incomparable + [owner.goal_id])}"
            )
    else:
        maximal = [
            goal
            for goal in owners
            if all(
                other.goal_id == goal.goal_id
                or other.goal_id in ancestor_ids(goal)
                for other in owners
            )
        ]
        if len(maximal) != 1:
            raise ValueError(
                f"objective heap has multiple owners for evidence requirement "
                f"{requirement}"
            )
        owner = maximal[0]
    parents = tuple(str(item).strip() for item in owner.parent_goal_ids if str(item).strip())
    expected_parent = str(expected_parent_goal_id or "").strip()
    if expected_parent and expected_parent not in parents:
        raise ValueError(
            f"evidence owner {owner.goal_id} is not a child of {expected_parent}"
        )
    if not parents:
        raise ValueError(f"evidence owner {owner.goal_id} has no parent goal")
    parent = expected_parent or parents[0]
    return ObjectiveEvidenceProjection(
        requirement_id=requirement,
        goal_id=owner.goal_id,
        parent_goal_id=parent,
        objective_heap_id=objective_heap_content_id(objective_text),
        goal_content_id=objective_goal_content_id(owner),
    )


SELF_IMPROVEMENT_GOAL_EVIDENCE_BINDING_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "self_improvement_goal_evidence_binding.v1"
)
SELF_IMPROVEMENT_GOAL_EVIDENCE_RECONCILIATION_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "self_improvement_goal_evidence_reconciliation.v1"
)


def _strict_record_keys(
    payload: Mapping[str, Any],
    allowed: set[str],
    *,
    record_name: str,
) -> None:
    unknown = sorted(str(key) for key in payload if str(key) not in allowed)
    if unknown:
        raise ValueError(
            f"{record_name} contains unknown fields: {', '.join(unknown)}"
        )


def _canonical_receipt_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    converter = getattr(value, "to_dict", None)
    projected = converter() if callable(converter) else None
    if not isinstance(projected, Mapping):
        raise TypeError("typed evidence receipt must be a mapping or expose to_dict()")
    return {str(key): item for key, item in projected.items()}


def _receipt_string_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        values = value
    else:
        return ()
    return tuple(
        dict.fromkeys(
            compact
            for item in values
            if (compact := " ".join(str(item or "").strip().split()))
        )
    )


def _receipt_requirement_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    result: list[str] = []
    for name in (
        "requirement_id",
        "requirement_ids",
        "proved_requirement_ids",
        "evidence_claim_references",
        "authoritative_evidence_claim_references",
    ):
        result.extend(_receipt_string_values(payload.get(name)))
    criterion = str(
        payload.get("acceptance_criterion")
        or payload.get("criterion")
        or ""
    ).strip()
    if criterion:
        result.append(criterion)
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        result.extend(_receipt_string_values(metadata.get("requirement_id")))
        result.extend(_receipt_string_values(metadata.get("requirement_ids")))
    return tuple(dict.fromkeys(result))


def _receipt_content_identity(payload: Mapping[str, Any]) -> str:
    """Hash finite measurement JSON without the task-ID float restriction."""

    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("typed evidence receipt is not finite canonical JSON") from exc
    return f"sha256:{sha256(encoded).hexdigest()}"


def _receipt_nonempty_text(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return bool(value) and all(
            isinstance(item, str) and bool(item.strip()) for item in value
        )
    return False


def _receipt_digest_valid(value: str) -> bool:
    """Accept canonical SHA-256 digests and CID-like content identifiers."""

    normalized = str(value or "").strip()
    if re.fullmatch(r"sha256:[0-9a-fA-F]{64}", normalized):
        return True
    # Repository content identities are CIDv1 base32 strings (normally
    # ``baguq...``).  Requiring the multibase prefix and a meaningful length
    # rejects labels such as ``artifact-1`` without coupling the tracker to a
    # single multicodec.
    return bool(re.fullmatch(r"b[a-z2-7]{31,}", normalized.casefold()))


def _receipt_identifier(payload: Mapping[str, Any]) -> str:
    for name in (
        "receipt_id",
        "evidence_id",
        "witness_id",
        "provenance_cid",
    ):
        value = str(payload.get(name) or "").strip()
        if value:
            return value
    return _receipt_content_identity(payload)


def _receipt_timestamp(
    payload: Mapping[str, Any], *names: str
) -> datetime | None:
    for name in names:
        raw = payload.get(name)
        if raw in (None, ""):
            continue
        if isinstance(raw, datetime):
            value = raw
        else:
            text = str(raw).strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                value = datetime.fromisoformat(text)
            except ValueError:
                return None
        if value.tzinfo is None or value.utcoffset() is None:
            return None
        return value.astimezone(timezone.utc)
    return None


def _reconciliation_now(value: datetime | str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    payload = {"value": value}
    parsed = _receipt_timestamp(payload, "value")
    if parsed is None:
        raise ValueError("now must be a timezone-aware datetime or ISO-8601 value")
    return parsed


def _embedded_self_improvement_receipt(
    payload: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Return the producer-owned witness inside CompletionEvidence, if any."""

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    for name in (
        "healthy_exhaustion_evidence",
        "successor_refill_evidence",
        "epoch_replay_evidence",
    ):
        candidate = metadata.get(name)
        if isinstance(candidate, Mapping):
            return candidate
    return None


def _restore_self_improvement_receipt(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Strictly restore a known producer receipt without a module cycle.

    ``self_improvement`` imports this module for objective projections, so its
    receipt types are imported only when this public reconciliation API is
    invoked.  Unknown receipt schemas remain eligible for the general source
    policy; known self-improvement schemas must pass their producer-owned
    deserializer and reproduce byte-equivalent canonical JSON.
    """

    candidate = _embedded_self_improvement_receipt(payload) or payload
    schema = str(candidate.get("schema") or "")
    if schema not in {
        "ipfs_accelerate_py.agent_supervisor.healthy_exhaustion_evidence.v1",
        "ipfs_accelerate_py.agent_supervisor.successor_refill_evidence.v1",
        "ipfs_accelerate_py.agent_supervisor.self_improvement_epoch_replay.v1",
    }:
        return dict(candidate), ()
    try:
        from .self_improvement import (
            EpochReplayEvidence,
            HealthyExhaustionEvidence,
            SuccessorRefillEvidence,
        )

        receipt_types = {
            "ipfs_accelerate_py.agent_supervisor.healthy_exhaustion_evidence.v1": (
                HealthyExhaustionEvidence
            ),
            "ipfs_accelerate_py.agent_supervisor.successor_refill_evidence.v1": (
                SuccessorRefillEvidence
            ),
            "ipfs_accelerate_py.agent_supervisor.self_improvement_epoch_replay.v1": (
                EpochReplayEvidence
            ),
        }
        restored = receipt_types[schema].from_dict(candidate)
        reproduced = restored.to_dict()
        if _receipt_content_identity(reproduced) != _receipt_content_identity(
            candidate
        ):
            return dict(candidate), ("receipt_canonical_projection_mismatch",)
        return reproduced, ()
    except (KeyError, TypeError, ValueError):
        return dict(candidate), ("receipt_integrity_invalid",)


@dataclass(frozen=True)
class SelfImprovementGoalEvidenceBinding:
    """One opaque requirement's exact leaf owner and receipt decision."""

    requirement_id: str
    goal_projection: ObjectiveEvidenceProjection | None
    receipt_id: str
    receipt_content_id: str
    producer_kind: str
    source_tier: str
    repository_tree: str
    policy_id: str
    artifact_digest: str
    observed_at: str
    fresh_until: str
    authoritative: bool
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        requirement = str(self.requirement_id or "").strip()
        if not requirement:
            raise ValueError("requirement_id is required")
        object.__setattr__(self, "requirement_id", requirement)
        projection = self.goal_projection
        if projection is not None and not isinstance(
            projection, ObjectiveEvidenceProjection
        ):
            if not isinstance(projection, Mapping):
                raise TypeError("goal_projection must be an evidence projection")
            projection = ObjectiveEvidenceProjection.from_dict(projection)
        if projection is not None and projection.requirement_id != requirement:
            raise ValueError("goal projection binds a different requirement")
        object.__setattr__(self, "goal_projection", projection)
        for name in (
            "receipt_id",
            "receipt_content_id",
            "producer_kind",
            "source_tier",
            "repository_tree",
            "policy_id",
            "artifact_digest",
            "observed_at",
            "fresh_until",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
        object.__setattr__(
            self,
            "reason_codes",
            tuple(
                dict.fromkeys(
                    str(item).strip()
                    for item in self.reason_codes
                    if str(item).strip()
                )
            ),
        )
        if self.authoritative and (
            self.goal_projection is None or self.reason_codes
        ):
            raise ValueError(
                "authoritative evidence requires a projection and no rejection"
            )

    @property
    def binding_id(self) -> str:
        return content_identity(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": SELF_IMPROVEMENT_GOAL_EVIDENCE_BINDING_SCHEMA,
            "version": 1,
            "requirement_id": self.requirement_id,
            "goal_projection": (
                self.goal_projection.to_dict()
                if self.goal_projection is not None
                else None
            ),
            "receipt_id": self.receipt_id,
            "receipt_content_id": self.receipt_content_id,
            "producer_kind": self.producer_kind,
            "source_tier": self.source_tier,
            "repository_tree": self.repository_tree,
            "policy_id": self.policy_id,
            "artifact_digest": self.artifact_digest,
            "observed_at": self.observed_at,
            "fresh_until": self.fresh_until,
            "authoritative": self.authoritative,
            "reason_codes": list(self.reason_codes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "binding_id": self.binding_id}

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "SelfImprovementGoalEvidenceBinding":
        allowed = {
            "schema",
            "version",
            "binding_id",
            "requirement_id",
            "goal_projection",
            "receipt_id",
            "receipt_content_id",
            "producer_kind",
            "source_tier",
            "repository_tree",
            "policy_id",
            "artifact_digest",
            "observed_at",
            "fresh_until",
            "authoritative",
            "reason_codes",
        }
        _strict_record_keys(
            payload, allowed, record_name="self-improvement evidence binding"
        )
        if (
            payload.get("schema")
            != SELF_IMPROVEMENT_GOAL_EVIDENCE_BINDING_SCHEMA
            or payload.get("version") != 1
        ):
            raise ValueError("unsupported self-improvement evidence binding schema")
        result = cls(
            requirement_id=str(payload.get("requirement_id") or ""),
            goal_projection=payload.get("goal_projection"),
            receipt_id=str(payload.get("receipt_id") or ""),
            receipt_content_id=str(payload.get("receipt_content_id") or ""),
            producer_kind=str(payload.get("producer_kind") or ""),
            source_tier=str(payload.get("source_tier") or ""),
            repository_tree=str(payload.get("repository_tree") or ""),
            policy_id=str(payload.get("policy_id") or ""),
            artifact_digest=str(payload.get("artifact_digest") or ""),
            observed_at=str(payload.get("observed_at") or ""),
            fresh_until=str(payload.get("fresh_until") or ""),
            authoritative=payload.get("authoritative") is True,
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )
        if payload.get("binding_id") != result.binding_id:
            raise ValueError("self-improvement evidence binding identity mismatch")
        return result


@dataclass(frozen=True)
class SelfImprovementGoalEvidenceReconciliation:
    """Content-addressed, mutation-free typed evidence batch decision."""

    objective_heap_id: str
    repository_tree: str
    policy_id: str
    evaluated_at: str
    requested_requirement_ids: tuple[str, ...]
    bindings: tuple[SelfImprovementGoalEvidenceBinding, ...]
    proposal_evidence: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "objective_heap_id",
            "repository_tree",
            "policy_id",
            "evaluated_at",
        ):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ValueError(f"{name} is required")
            object.__setattr__(self, name, value)
        requested = tuple(
            sorted(
                {
                    str(item).strip()
                    for item in self.requested_requirement_ids
                    if str(item).strip()
                }
            )
        )
        object.__setattr__(self, "requested_requirement_ids", requested)
        normalized_bindings = tuple(
            sorted(
                (
                    item
                    if isinstance(item, SelfImprovementGoalEvidenceBinding)
                    else SelfImprovementGoalEvidenceBinding.from_dict(item)
                    for item in self.bindings
                ),
                key=lambda item: (
                    item.requirement_id,
                    item.receipt_id,
                    item.receipt_content_id,
                ),
            )
        )
        object.__setattr__(self, "bindings", normalized_bindings)
        proposals = {
            str(requirement).strip(): tuple(
                sorted(
                    {
                        str(reference).strip()
                        for reference in references
                        if str(reference).strip()
                    }
                )
            )
            for requirement, references in dict(
                self.proposal_evidence or {}
            ).items()
            if str(requirement).strip()
        }
        object.__setattr__(
            self,
            "proposal_evidence",
            {key: proposals[key] for key in sorted(proposals)},
        )

    @property
    def authoritative_requirement_ids(self) -> tuple[str, ...]:
        return tuple(
            requirement
            for requirement in self.requested_requirement_ids
            if any(
                item.requirement_id == requirement and item.authoritative
                for item in self.bindings
            )
        )

    @property
    def rejected_requirement_ids(self) -> tuple[str, ...]:
        authoritative = set(self.authoritative_requirement_ids)
        return tuple(
            requirement
            for requirement in self.requested_requirement_ids
            if requirement not in authoritative
            and any(
                item.requirement_id == requirement for item in self.bindings
            )
        )

    @property
    def proposal_only_requirement_ids(self) -> tuple[str, ...]:
        authoritative = set(self.authoritative_requirement_ids)
        return tuple(
            requirement
            for requirement in self.requested_requirement_ids
            if requirement not in authoritative
            and bool(self.proposal_evidence.get(requirement))
        )

    @property
    def missing_requirement_ids(self) -> tuple[str, ...]:
        covered = set(self.authoritative_requirement_ids)
        proposed = set(self.proposal_only_requirement_ids)
        rejected = set(self.rejected_requirement_ids)
        return tuple(
            requirement
            for requirement in self.requested_requirement_ids
            if requirement not in covered | proposed | rejected
        )

    @property
    def satisfied(self) -> bool:
        return bool(self.requested_requirement_ids) and (
            self.authoritative_requirement_ids
            == self.requested_requirement_ids
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": SELF_IMPROVEMENT_GOAL_EVIDENCE_RECONCILIATION_SCHEMA,
            "version": 1,
            "objective_heap_id": self.objective_heap_id,
            "repository_tree": self.repository_tree,
            "policy_id": self.policy_id,
            "evaluated_at": self.evaluated_at,
            "requested_requirement_ids": list(
                self.requested_requirement_ids
            ),
            "bindings": [item.to_dict() for item in self.bindings],
            "proposal_evidence": {
                key: list(value)
                for key, value in self.proposal_evidence.items()
            },
            "authoritative_requirement_ids": list(
                self.authoritative_requirement_ids
            ),
            "rejected_requirement_ids": list(
                self.rejected_requirement_ids
            ),
            "proposal_only_requirement_ids": list(
                self.proposal_only_requirement_ids
            ),
            "missing_requirement_ids": list(self.missing_requirement_ids),
            "satisfied": self.satisfied,
        }

    @property
    def reconciliation_id(self) -> str:
        return content_identity(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._payload(),
            "reconciliation_id": self.reconciliation_id,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "SelfImprovementGoalEvidenceReconciliation":
        allowed = {
            "schema",
            "version",
            "reconciliation_id",
            "objective_heap_id",
            "repository_tree",
            "policy_id",
            "evaluated_at",
            "requested_requirement_ids",
            "bindings",
            "proposal_evidence",
            "authoritative_requirement_ids",
            "rejected_requirement_ids",
            "proposal_only_requirement_ids",
            "missing_requirement_ids",
            "satisfied",
        }
        _strict_record_keys(
            payload,
            allowed,
            record_name="self-improvement goal evidence reconciliation",
        )
        if (
            payload.get("schema")
            != SELF_IMPROVEMENT_GOAL_EVIDENCE_RECONCILIATION_SCHEMA
            or payload.get("version") != 1
        ):
            raise ValueError(
                "unsupported self-improvement goal evidence reconciliation schema"
            )
        bindings = payload.get("bindings")
        proposals = payload.get("proposal_evidence")
        if not isinstance(bindings, Sequence) or isinstance(
            bindings, (str, bytes, bytearray)
        ):
            raise ValueError("bindings must be a sequence")
        if not isinstance(proposals, Mapping):
            raise ValueError("proposal_evidence must be an object")
        result = cls(
            objective_heap_id=str(payload.get("objective_heap_id") or ""),
            repository_tree=str(payload.get("repository_tree") or ""),
            policy_id=str(payload.get("policy_id") or ""),
            evaluated_at=str(payload.get("evaluated_at") or ""),
            requested_requirement_ids=tuple(
                payload.get("requested_requirement_ids") or ()
            ),
            bindings=tuple(
                SelfImprovementGoalEvidenceBinding.from_dict(item)
                if isinstance(item, Mapping)
                else (_raise_quality_report_value("bindings"))
                for item in bindings
            ),
            proposal_evidence={
                str(key): tuple(value)
                for key, value in proposals.items()
                if isinstance(value, Sequence)
                and not isinstance(value, (str, bytes, bytearray))
            },
        )
        projected = result.to_dict()
        for name in (
            "authoritative_requirement_ids",
            "rejected_requirement_ids",
            "proposal_only_requirement_ids",
            "missing_requirement_ids",
            "satisfied",
            "reconciliation_id",
        ):
            if payload.get(name) != projected[name]:
                raise ValueError(
                    f"self-improvement reconciliation {name} mismatch"
                )
        return result


def reconcile_self_improvement_goal_evidence(
    objective_text: str,
    *,
    typed_receipts: Sequence[Mapping[str, Any] | Any] = (),
    requirement_ids: Iterable[str] = (),
    proposal_evidence: Mapping[str, Sequence[str]] | None = None,
    repository_tree: str,
    policy_id: str,
    now: datetime | str | None = None,
    freshness_seconds: float = DEFAULT_EVIDENCE_FRESHNESS_SECONDS,
) -> SelfImprovementGoalEvidenceReconciliation:
    """Reconcile an immutable batch of self-improvement goal evidence.

    Opaque IDs are resolved to their unique most-refined heap owner.  Text and
    retrieval references are retained as proposal evidence but never become
    completion authority.  Typed receipts must be current-tree, exact-policy,
    fresh, terminally successful producer records.  Known self-improvement
    receipts additionally pass their strict producer deserializer and content
    identity check.  A receipt spanning different leaf owners and multiple
    distinct receipts claiming the same requirement both fail closed.
    """

    if not isinstance(objective_text, str):
        raise TypeError("objective_text must be a string")
    tree = str(repository_tree or "").strip()
    policy = str(policy_id or "").strip()
    if not tree:
        raise ValueError("repository_tree is required")
    if not policy:
        raise ValueError("policy_id is required")
    if (
        isinstance(freshness_seconds, bool)
        or not isinstance(freshness_seconds, (int, float))
        or freshness_seconds < 0
    ):
        raise ValueError("freshness_seconds must be a non-negative number")
    evaluated = _reconciliation_now(now)
    explicit_requirements = {
        str(item).strip()
        for item in requirement_ids
        if str(item).strip()
    }
    if explicit_requirements:
        requested = tuple(sorted(explicit_requirements))
    else:
        requested = tuple(
            sorted(
                {
                    str(requirement).strip()
                    for goal in parse_goal_heap(objective_text)
                    for requirement in goal.required_evidence
                    if OPAQUE_EVIDENCE_REQUIREMENT_PATTERN.fullmatch(
                        str(requirement).strip()
                    )
                }
            )
        )
    nonopaque = tuple(
        item
        for item in requested
        if not OPAQUE_EVIDENCE_REQUIREMENT_PATTERN.fullmatch(item)
    )
    if nonopaque:
        raise ValueError(
            "self-improvement reconciliation accepts only opaque requirement "
            f"IDs: {', '.join(nonopaque)}"
        )

    projections: dict[str, ObjectiveEvidenceProjection] = {}
    projection_errors: dict[str, str] = {}
    for requirement in requested:
        try:
            projections[requirement] = resolve_objective_evidence_projection(
                objective_text, requirement_id=requirement
            )
        except ValueError as exc:
            projection_errors[requirement] = str(exc)

    normalized_proposals = {
        requirement: tuple(
            sorted(
                {
                    str(reference).strip()
                    for reference in (proposal_evidence or {}).get(
                        requirement, ()
                    )
                    if str(reference).strip()
                }
            )
        )
        for requirement in requested
        if (proposal_evidence or {}).get(requirement)
    }
    provisional: list[SelfImprovementGoalEvidenceBinding] = []
    for receipt_value in typed_receipts:
        outer = _canonical_receipt_payload(receipt_value)
        receipt, integrity_reasons = _restore_self_improvement_receipt(outer)
        claims = tuple(
            requirement
            for requirement in _receipt_requirement_ids(receipt)
            if requirement in requested
        )
        if not claims:
            continue
        owner_ids = {
            projections[requirement].goal_id
            for requirement in claims
            if requirement in projections
        }
        shared_reasons: list[str] = list(integrity_reasons)
        if len(owner_ids) > 1:
            shared_reasons.append("receipt_claims_multiple_goal_owners")
        receipt_id = _receipt_identifier(receipt)
        receipt_content_id = _receipt_content_identity(receipt)
        receipt_tree = str(
            receipt.get("repository_tree")
            or receipt.get("repository_tree_id")
            or receipt.get("tree_id")
            or ""
        ).strip()
        receipt_policy = str(
            receipt.get("policy_id") or receipt.get("policy_digest") or ""
        ).strip()
        producer_kind = str(receipt.get("producer_kind") or "").strip()
        source_tier = str(
            receipt.get("source_tier")
            or receipt.get("receipt_kind")
            or producer_kind
        ).strip()
        artifact_digest = str(
            receipt.get("artifact_digest")
            or receipt.get("provenance_cid")
            or ""
        ).strip()
        command = receipt.get("command", receipt.get("commands"))
        toolchain = receipt.get("toolchain", receipt.get("toolchains"))
        scope = receipt.get("scope")
        result = receipt.get("result")
        if not _receipt_nonempty_text(command):
            shared_reasons.append("receipt_command_missing_or_invalid")
        if not _receipt_nonempty_text(toolchain):
            shared_reasons.append("receipt_toolchain_missing_or_invalid")
        if (
            not isinstance(scope, Sequence)
            or isinstance(scope, (str, bytes, bytearray))
            or not scope
            or any(not str(item).strip() for item in scope)
        ):
            shared_reasons.append("receipt_scope_missing_or_invalid")
        if not isinstance(result, Mapping) or not result:
            shared_reasons.append("receipt_result_missing_or_invalid")
        observed = _receipt_timestamp(
            receipt,
            "observed_at",
            "replayed_at",
            "finished_at",
            "generated_at",
            "created_at",
        )
        fresh_until = _receipt_timestamp(receipt, "fresh_until", "expires_at")
        if observed is None:
            shared_reasons.append("receipt_observed_at_missing_or_invalid")
        elif observed > evaluated + timedelta(seconds=300):
            shared_reasons.append("receipt_observed_in_future")
        elif fresh_until is None and (
            evaluated - observed
        ).total_seconds() > float(freshness_seconds):
            shared_reasons.append("receipt_stale")
        if fresh_until is not None and evaluated > fresh_until:
            shared_reasons.append("receipt_stale")
        if not artifact_digest:
            shared_reasons.append("receipt_artifact_digest_missing")
        elif not _receipt_digest_valid(artifact_digest):
            shared_reasons.append("receipt_artifact_digest_invalid")

        embedded_projection = receipt.get("goal_projection")
        parsed_embedded: ObjectiveEvidenceProjection | None = None
        if embedded_projection is not None:
            try:
                if not isinstance(embedded_projection, Mapping):
                    raise TypeError
                parsed_embedded = ObjectiveEvidenceProjection.from_dict(
                    embedded_projection
                )
            except (TypeError, ValueError):
                shared_reasons.append("receipt_goal_projection_invalid")

        for requirement in claims:
            reasons = list(shared_reasons)
            projection = projections.get(requirement)
            if projection is None:
                reasons.append("requirement_owner_missing_or_ambiguous")
            if (
                parsed_embedded is not None
                and projection is not None
                and parsed_embedded != projection
            ):
                reasons.append("receipt_goal_projection_mismatch")
            try:
                source_decision = completion_evidence_source_decision(
                    receipt,
                    requirement=requirement,
                    repository_tree=tree,
                    policy_id=policy,
                )
                reasons.extend(source_decision.reason_codes)
                resolved_source_tier = source_decision.source_tier.value
            except (TypeError, ValueError):
                reasons.append("source_policy_evaluation_failed")
                resolved_source_tier = source_tier
            unique_reasons = tuple(dict.fromkeys(reasons))
            provisional.append(
                SelfImprovementGoalEvidenceBinding(
                    requirement_id=requirement,
                    goal_projection=projection,
                    receipt_id=receipt_id,
                    receipt_content_id=receipt_content_id,
                    producer_kind=producer_kind,
                    source_tier=resolved_source_tier,
                    repository_tree=receipt_tree,
                    policy_id=receipt_policy,
                    artifact_digest=artifact_digest,
                    observed_at=observed.isoformat() if observed else "",
                    fresh_until=(
                        fresh_until.isoformat() if fresh_until else ""
                    ),
                    authoritative=not unique_reasons and projection is not None,
                    reason_codes=unique_reasons,
                )
            )

    distinct_by_requirement: dict[str, set[str]] = {}
    for item in provisional:
        distinct_by_requirement.setdefault(item.requirement_id, set()).add(
            item.receipt_id or item.receipt_content_id
        )
    bindings = tuple(
        SelfImprovementGoalEvidenceBinding(
            requirement_id=item.requirement_id,
            goal_projection=item.goal_projection,
            receipt_id=item.receipt_id,
            receipt_content_id=item.receipt_content_id,
            producer_kind=item.producer_kind,
            source_tier=item.source_tier,
            repository_tree=item.repository_tree,
            policy_id=item.policy_id,
            artifact_digest=item.artifact_digest,
            observed_at=item.observed_at,
            fresh_until=item.fresh_until,
            authoritative=(
                item.authoritative
                and len(distinct_by_requirement[item.requirement_id]) == 1
            ),
            reason_codes=(
                item.reason_codes
                if len(distinct_by_requirement[item.requirement_id]) == 1
                else tuple(
                    dict.fromkeys(
                        (
                            *item.reason_codes,
                            "duplicate_requirement_receipts",
                        )
                    )
                )
            ),
        )
        for item in provisional
    )
    return SelfImprovementGoalEvidenceReconciliation(
        objective_heap_id=objective_heap_content_id(objective_text),
        repository_tree=tree,
        policy_id=policy,
        evaluated_at=evaluated.isoformat(),
        requested_requirement_ids=requested,
        bindings=bindings,
        proposal_evidence=normalized_proposals,
    )


@dataclass(frozen=True)
class RepositoryComponent:
    """A repository component that can participate in interoperability goals."""

    path: str
    sources: list[str] = field(default_factory=list)
    exists: bool = False
    is_gitlink: bool = False
    is_gitmodule: bool = False
    manifests: list[str] = field(default_factory=list)
    interface_descriptors: list[str] = field(default_factory=list)
    mcp_descriptors: list[str] = field(default_factory=list)
    python_import_roots: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fibonacci_number(index: int) -> int:
    """Return a one-indexed Fibonacci-ish priority bucket."""

    if index <= 1:
        return 1
    left, right = 1, 1
    for _ in range(2, index + 1):
        left, right = right, left + right
    return right


def fibonacci_priority(depth: int, sibling_index: int = 0) -> int:
    """Return a stable integer priority for a goal depth and sibling order."""

    return fibonacci_number(max(1, depth + 2)) * 1000 + max(0, sibling_index)


def infer_goal_prefix(goals: Sequence[ObjectiveGoal], *, fallback: str = DEFAULT_GOAL_PREFIX) -> str:
    """Infer a numeric goal-id prefix from existing goals."""

    prefixes: dict[str, int] = {}
    for goal in goals:
        match = re.match(r"^(.+?)(\d+)$", goal.goal_id)
        if not match:
            continue
        prefixes[match.group(1)] = prefixes.get(match.group(1), 0) + 1
    if not prefixes:
        return fallback
    return sorted(prefixes.items(), key=lambda item: (-item[1], item[0]))[0][0]


def next_goal_id(goals: Sequence[ObjectiveGoal], *, prefix: str | None = None) -> str:
    prefix = prefix or infer_goal_prefix(goals)
    highest = -1
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for goal in goals:
        match = pattern.match(goal.goal_id)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"{prefix}{highest + 1:03d}"


def render_goal_block(*, goal_id: str, title: str, fields: dict[str, str]) -> str:
    rows = [f"## {goal_id} {title.strip()}", ""]
    for key, value in fields.items():
        rows.append(f"- {key}: {value}")
    return "\n".join(rows).rstrip() + "\n"


def rewrite_goal_fields(text: str, updates: Mapping[str, Mapping[str, str]]) -> str:
    """Rewrite selected markdown goal fields without reparsing the whole file."""

    if not updates:
        return text

    lines = text.splitlines()
    rewritten: list[str] = []
    block: list[str] = []
    current_goal_id = ""
    header_pattern = re.compile(r"^##\s+(\S+)\s+.+?\s*$")

    def flush() -> None:
        nonlocal block, current_goal_id
        if not block:
            return
        goal_updates = updates.get(current_goal_id)
        if not goal_updates:
            rewritten.extend(block)
            block = []
            return

        normalized_updates = {normalize_field_key(key): (key, value) for key, value in goal_updates.items()}
        seen_keys: set[str] = set()
        output: list[str] = []
        last_field_index = 0
        replaced_wrapped_field = False
        for line in block:
            if line.startswith("- ") and ":" in line:
                key, _value = line[2:].split(":", 1)
                normalized = normalize_field_key(key)
                if normalized in normalized_updates:
                    output.append(f"- {normalized_updates[normalized][0]}: {normalized_updates[normalized][1]}")
                    seen_keys.add(normalized)
                    replaced_wrapped_field = True
                else:
                    output.append(line)
                    replaced_wrapped_field = False
                last_field_index = len(output)
            elif replaced_wrapped_field and line[:1].isspace() and line.strip():
                # The parser treats this as part of the replaced field. Keeping
                # it would append stale prose to the new value on reparse.
                continue
            else:
                output.append(line)
                if not line.strip() or not line[:1].isspace():
                    replaced_wrapped_field = False

        missing_lines = [
            f"- {key}: {value}"
            for normalized, (key, value) in normalized_updates.items()
            if normalized not in seen_keys
        ]
        if missing_lines:
            insert_at = max(1, last_field_index)
            output[insert_at:insert_at] = missing_lines
        rewritten.extend(output)
        block = []

    for line in lines:
        header = header_pattern.match(line)
        if header:
            flush()
            current_goal_id = header.group(1)
            block = [line]
            continue
        if block:
            block.append(line)
        else:
            rewritten.append(line)
    flush()
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(rewritten) + suffix


def completion_evidence_summary(evidence: Mapping[str, Sequence[str]]) -> str:
    parts: list[str] = []
    for term, paths in evidence.items():
        compact_paths = ", ".join(str(path) for path in list(paths)[:3])
        if compact_paths:
            parts.append(f"{term} => {compact_paths}")
    return "; ".join(parts)


def open_goal_ids_from_todo_board(todo_path: Path, task_header_prefix: str = "") -> set[str]:
    """Return objective goal ids with open tasks in the current task board."""

    if not todo_path.exists():
        return set()

    from .todo_daemon.implementation_daemon import TASK_HEADER_PREFIX, parse_task_file

    open_goal_ids: set[str] = set()
    for task in parse_task_file(todo_path, task_header_prefix or TASK_HEADER_PREFIX):
        if task.status not in OPEN_TASK_STATUSES_FOR_GOAL_COMPLETION:
            continue
        for key in TASK_GOAL_METADATA_KEYS:
            open_goal_ids.update(split_terms(task.metadata.get(key, "")))
    return {goal_id for goal_id in open_goal_ids if goal_id}


def directly_open_goal_ids_from_todo_board(
    todo_path: Path,
    task_header_prefix: str = "",
) -> set[str]:
    """Return goals directly bound to open tasks, excluding ancestor lineage."""

    if not todo_path.exists():
        return set()

    from .todo_daemon.implementation_daemon import TASK_HEADER_PREFIX, parse_task_file

    goal_ids: set[str] = set()
    for task in parse_task_file(todo_path, task_header_prefix or TASK_HEADER_PREFIX):
        if task.status not in OPEN_TASK_STATUSES_FOR_GOAL_COMPLETION:
            continue
        for key in ("goal id", "goal ids", "goal packet goals"):
            goal_ids.update(split_terms(task.metadata.get(key, "")))
    return {goal_id for goal_id in goal_ids if goal_id}


def open_implementation_goal_ids_from_todo_board(
    todo_path: Path,
    task_header_prefix: str = "",
) -> set[str]:
    """Return goals whose implementation work, rather than proof gate, is open."""

    if not todo_path.exists():
        return set()

    from .todo_daemon.implementation_daemon import TASK_HEADER_PREFIX, parse_task_file

    goal_ids: set[str] = set()
    for task in parse_task_file(todo_path, task_header_prefix or TASK_HEADER_PREFIX):
        if task.status not in OPEN_TASK_STATUSES_FOR_GOAL_COMPLETION:
            continue
        candidate_kind = task.metadata.get("candidate kind", "").strip().casefold()
        merge_role = task.metadata.get("merge role", "").strip().casefold()
        if (
            candidate_kind == "validation_gate"
            or merge_role in {"validation_gate", "completion_gate"}
        ):
            continue
        for key in TASK_GOAL_METADATA_KEYS:
            goal_ids.update(split_terms(task.metadata.get(key, "")))
    return {goal_id for goal_id in goal_ids if goal_id}


def referenced_goal_ids_from_todo_board(
    todo_path: Path,
    task_header_prefix: str = "",
) -> set[str]:
    """Return every objective goal id referenced by a task of any status."""

    if not todo_path.exists():
        return set()

    from .todo_daemon.implementation_daemon import TASK_HEADER_PREFIX, parse_task_file

    goal_ids: set[str] = set()
    for task in parse_task_file(todo_path, task_header_prefix or TASK_HEADER_PREFIX):
        for key in TASK_GOAL_METADATA_KEYS:
            goal_ids.update(split_terms(task.metadata.get(key, "")))
    return {goal_id for goal_id in goal_ids if goal_id}


def referenced_goal_ids_from_todo_boards(
    todo_boards: Sequence[tuple[Path, str]],
) -> dict[str, list[str]]:
    """Map every task-linked objective goal to the boards that reference it."""

    goal_ids: dict[str, list[str]] = {}
    seen_boards: set[tuple[str, str]] = set()
    for todo_path, task_header_prefix in todo_boards:
        board_key = (str(todo_path), str(task_header_prefix))
        if board_key in seen_boards:
            continue
        seen_boards.add(board_key)
        for goal_id in referenced_goal_ids_from_todo_board(
            todo_path,
            task_header_prefix,
        ):
            goal_ids.setdefault(goal_id, []).append(str(todo_path))
    return goal_ids


def open_goal_ids_from_todo_boards(
    todo_boards: Sequence[tuple[Path, str]],
) -> dict[str, list[str]]:
    """Return open objective goal ids mapped to the board paths that still reference them."""

    open_goal_ids: dict[str, list[str]] = {}
    seen_boards: set[tuple[str, str]] = set()
    for todo_path, task_header_prefix in todo_boards:
        board_key = (str(todo_path), str(task_header_prefix))
        if board_key in seen_boards:
            continue
        seen_boards.add(board_key)
        for goal_id in open_goal_ids_from_todo_board(todo_path, task_header_prefix):
            open_goal_ids.setdefault(goal_id, []).append(str(todo_path))
    return open_goal_ids


def open_implementation_goal_ids_from_todo_boards(
    todo_boards: Sequence[tuple[Path, str]],
) -> dict[str, list[str]]:
    """Map goals to boards containing unfinished implementation-stage work."""

    open_goal_ids: dict[str, list[str]] = {}
    seen_boards: set[tuple[str, str]] = set()
    for todo_path, task_header_prefix in todo_boards:
        board_key = (str(todo_path), str(task_header_prefix))
        if board_key in seen_boards:
            continue
        seen_boards.add(board_key)
        for goal_id in open_implementation_goal_ids_from_todo_board(
            todo_path,
            task_header_prefix,
        ):
            open_goal_ids.setdefault(goal_id, []).append(str(todo_path))
    return open_goal_ids


def run_goal_validation(
    *,
    repo_root: Path,
    goal: ObjectiveGoal,
    timeout_seconds: float = 300.0,
    repository_identity: RepositoryTreeIdentity | None = None,
) -> dict[str, Any]:
    """Run goal validation and return a tree-bound, content-addressed receipt."""

    commands = split_validation_commands(str(goal.fields.get("validation") or ""))
    started_at = utc_now()
    if not commands:
        payload = {
            "schema": "ipfs_accelerate_py.agent_supervisor.goal-validation@1",
            "goal_id": goal.goal_id,
            "attempted": False,
            "passed": False,
            "returncode": 1,
            "results": [],
            "reason": "missing_validation_commands",
            "started_at": started_at,
            "finished_at": utc_now(),
        }
        identity = repository_identity or scan_identity(repo_root)
        payload.update(identity.to_dict())
        payload["receipt_cid"] = canonical_content_cid(payload)
        return payload
    results: list[dict[str, Any]] = []
    failure: dict[str, Any] = {}
    validation_environment = build_validation_environment()
    for command in commands:
        command_started_at = utc_now()
        try:
            completed = subprocess.run(
                validation_shell_command(command),
                cwd=repo_root,
                text=True,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
                env=validation_environment,
            )
        except subprocess.TimeoutExpired as exc:
            result = {
                "command": command,
                "started_at": command_started_at,
                "finished_at": utc_now(),
                "returncode": 124,
                "timed_out": True,
                "stdout": str(exc.stdout or "")[-4000:],
                "stderr": str(exc.stderr or "")[-4000:],
            }
            results.append(result)
            failure = {"returncode": 124, "failed_command": command, "error": "timeout"}
            break
        result = {
            "command": command,
            "started_at": command_started_at,
            "finished_at": utc_now(),
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
        results.append(result)
        if completed.returncode != 0:
            failure = {"returncode": completed.returncode, "failed_command": command}
            break
    identity = repository_identity or scan_identity(repo_root)
    payload = {
        "schema": "ipfs_accelerate_py.agent_supervisor.goal-validation@1",
        "goal_id": goal.goal_id,
        "attempted": True,
        "passed": not failure,
        "returncode": int(failure.get("returncode", 0)),
        "results": results,
        "started_at": started_at,
        "finished_at": utc_now(),
        **identity.to_dict(),
    }
    payload.update({key: value for key, value in failure.items() if key != "returncode"})
    payload["receipt_cid"] = canonical_content_cid(payload)
    return payload


def _git_output(repo_root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    """Best-effort bounded git query used for completion tree identity."""

    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            text=not binary,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return b"" if binary else ""
    if completed.returncode != 0:
        return b"" if binary else ""
    return completed.stdout if binary else str(completed.stdout).strip()


def _gitlink_paths(repo_root: Path) -> tuple[str, ...]:
    """Return index-backed submodule paths without trusting ``.gitmodules``."""

    raw_entries = _git_output(
        repo_root,
        "ls-files",
        "--stage",
        "-z",
        binary=True,
    )
    if not isinstance(raw_entries, bytes):
        return ()
    paths: list[str] = []
    for raw_entry in raw_entries.split(b"\0"):
        if not raw_entry:
            continue
        metadata, separator, raw_path = raw_entry.partition(b"\t")
        if not separator or not metadata.startswith(b"160000 "):
            continue
        path = raw_path.decode("utf-8", errors="surrogateescape")
        candidate = Path(path)
        if (
            not path
            or candidate.is_absolute()
            or ".." in candidate.parts
            or path in paths
        ):
            continue
        paths.append(path)
    return tuple(sorted(paths))


def _bind_submodule_worktree_identities(
    repo_root: Path,
    identity: RepositoryTreeIdentity,
    *,
    excluded_paths: Sequence[Path],
    visited_repositories: frozenset[Path],
    status_snapshot: bytes | None = None,
) -> RepositoryTreeIdentity:
    """Fold initialized submodule worktree bytes into a parent identity.

    Git's parent status records only that a submodule is dirty.  Two different
    dirty byte states can therefore have identical parent ``status`` and
    ``diff`` output.  Completion evidence needs the recursively computed child
    identity as well as the gitlink commit already present in the parent tree.
    """

    root = repo_root.resolve()
    gitlinks = _gitlink_paths(root)
    if not gitlinks:
        return identity
    if status_snapshot is None:
        raw_status = _git_output(
            root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=none",
            binary=True,
        )
        status_snapshot = raw_status if isinstance(raw_status, bytes) else b""
    dirty_paths = {
        entry[3:]
        for entry in status_snapshot.split(b"\0")
        if len(entry) >= 4
    }
    dirty_gitlinks = tuple(
        relative
        for relative in gitlinks
        if relative.encode("utf-8", errors="surrogateescape") in dirty_paths
    )
    if not dirty_gitlinks:
        # Clean submodule bytes are already content-addressed by the gitlink in
        # the parent manifest.  Inspect only dirty/mismatched worktrees; this
        # keeps completion identity bounded even in repositories with many
        # initialized submodules.
        return identity

    digest = sha256()
    digest.update(b"completion-source-tree-with-submodules-v1\0")
    digest.update(identity.tree_id.encode("utf-8", errors="surrogateescape"))
    next_visited = frozenset((*visited_repositories, root))
    for relative in dirty_gitlinks:
        raw_relative = relative.encode("utf-8", errors="surrogateescape")
        child = (root / relative).resolve()
        digest.update(b"\0submodule\0")
        digest.update(raw_relative)
        if child in next_visited:
            digest.update(b"\0cycle\0")
            continue
        child_top = str(_git_output(child, "rev-parse", "--show-toplevel") or "")
        if not child_top or Path(child_top).resolve() != child:
            # The gitlink commit remains bound by the parent manifest.  This
            # marker additionally distinguishes an absent/uninitialized child
            # from the initialized worktree that validators inspect.
            digest.update(b"\0uninitialized\0")
            continue
        child_controls: list[Path] = []
        for control in excluded_paths:
            try:
                control.resolve().relative_to(child)
            except ValueError:
                continue
            child_controls.append(control)
        child_identity = _control_tree_identity(
            child,
            excluded_paths=child_controls,
            _visited_repositories=next_visited,
        )
        digest.update(b"\0tree\0")
        digest.update(
            child_identity.tree_id.encode("utf-8", errors="surrogateescape")
        )
    return RepositoryTreeIdentity(
        repository_id=identity.repository_id,
        tree_id=f"sha256:{digest.hexdigest()}",
    )


def _control_tree_identity(
    repo_root: Path,
    *,
    excluded_paths: Sequence[Path],
    _visited_repositories: frozenset[Path] = frozenset(),
) -> RepositoryTreeIdentity:
    """Return source-tree identity while excluding supervisor control files."""

    root = repo_root.resolve()
    top_text = str(_git_output(root, "rev-parse", "--show-toplevel") or "")
    if not top_text:
        excluded_files = {
            path.resolve()
            for path in excluded_paths
            if not path.exists() or not path.is_dir()
        }
        excluded_directories = {
            path.resolve()
            for path in excluded_paths
            if path.exists() and path.is_dir()
        }

        def excluded(candidate: Path) -> bool:
            resolved = candidate.resolve()
            if resolved in excluded_files:
                return True
            return any(
                resolved == directory or directory in resolved.parents
                for directory in excluded_directories
            )

        # Git-less workspaces still need a real compare-and-swap fence.  The
        # generic scan identity is path-derived for such directories, so hash
        # names and bytes here while omitting only supervisor-owned controls.
        digest = sha256()
        for candidate in sorted(
            root.rglob("*"),
            key=lambda path: path.relative_to(root).as_posix(),
        ):
            if excluded(candidate):
                continue
            relative = candidate.relative_to(root).as_posix()
            try:
                if candidate.is_symlink():
                    digest.update(b"\0symlink\0")
                    digest.update(relative.encode("utf-8"))
                    digest.update(candidate.readlink().as_posix().encode("utf-8"))
                elif candidate.is_file():
                    digest.update(b"\0file\0")
                    digest.update(relative.encode("utf-8"))
                    digest.update(
                        str(candidate.stat().st_mode & 0o777).encode("ascii")
                    )
                    with candidate.open("rb") as stream:
                        for chunk in iter(
                            lambda: stream.read(1024 * 1024), b""
                        ):
                            digest.update(chunk)
            except OSError as exc:
                # A concurrently removed or unreadable source is itself a
                # distinct, fail-closed identity rather than an ignored file.
                digest.update(b"\0unreadable\0")
                digest.update(relative.encode("utf-8"))
                digest.update(type(exc).__name__.encode("ascii"))
        return RepositoryTreeIdentity(
            repository_id=str(root),
            tree_id=f"sha256:{digest.hexdigest()}",
        )
    top = Path(top_text).resolve()
    relatives: list[str] = []
    for path in excluded_paths:
        try:
            relative = path.resolve().relative_to(top).as_posix()
        except ValueError:
            continue
        if relative and relative not in relatives:
            relatives.append(relative)
    if not relatives:
        return _bind_submodule_worktree_identities(
            root,
            scan_identity(root),
            excluded_paths=excluded_paths,
            visited_repositories=_visited_repositories,
        )
    common_dir_text = str(
        _git_output(root, "rev-parse", "--git-common-dir") or ""
    )
    if common_dir_text:
        common_dir = Path(common_dir_text)
        if not common_dir.is_absolute():
            common_dir = root / common_dir
        repository_id = str(common_dir.resolve())
    else:
        repository_id = str(top)
    head_tree = str(_git_output(root, "rev-parse", "HEAD^{tree}") or "")
    if not head_tree:
        return RepositoryTreeIdentity(
            repository_id=repository_id,
            tree_id=(
                "unversioned:"
                + sha256(str(root).encode("utf-8")).hexdigest()
            ),
        )
    # Hash the tracked manifest after removing control paths.  Starting from
    # ``HEAD^{tree}`` would still include the last committed bytes of an
    # excluded artifact, so committing a regenerated proof would make that
    # proof part of the source identity it is trying to attest to.
    raw_head_entries = _git_output(
        top,
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        "HEAD",
        binary=True,
    )
    assert isinstance(raw_head_entries, bytes)
    excluded_bytes = tuple(relative.encode("utf-8") for relative in relatives)
    digest = sha256()
    digest.update(b"completion-source-tree-v1\0")
    for entry in raw_head_entries.split(b"\0"):
        if not entry:
            continue
        separator = entry.find(b"\t")
        entry_path = entry[separator + 1 :] if separator >= 0 else b""
        if entry_path and any(
            entry_path == excluded
            or entry_path.startswith(excluded + b"/")
            for excluded in excluded_bytes
        ):
            continue
        digest.update(entry)
        digest.update(b"\0")
    pathspec = ("--", ".", *(f":(exclude){relative}" for relative in relatives))
    status = _git_output(
        top,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
        *pathspec,
        binary=True,
    )
    assert isinstance(status, bytes)
    if not status:
        return _bind_submodule_worktree_identities(
            root,
            RepositoryTreeIdentity(
                repository_id=repository_id,
                tree_id=f"sha256:{digest.hexdigest()}",
            ),
            excluded_paths=excluded_paths,
            visited_repositories=_visited_repositories,
            status_snapshot=status,
        )
    digest.update(b"\0status\0")
    digest.update(status)
    digest.update(b"\0diff\0")
    diff = _git_output(
        top,
        "diff",
        "--binary",
        "--no-ext-diff",
        "HEAD",
        *pathspec,
        binary=True,
    )
    assert isinstance(diff, bytes)
    digest.update(diff)
    untracked = _git_output(
        top,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        *pathspec,
        binary=True,
    )
    assert isinstance(untracked, bytes)
    for raw_relative in sorted(path for path in untracked.split(b"\0") if path):
        digest.update(b"\0untracked\0")
        digest.update(raw_relative)
        try:
            candidate = top / raw_relative.decode("utf-8", errors="surrogateescape")
            if candidate.is_symlink():
                digest.update(candidate.readlink().as_posix().encode("utf-8"))
            elif candidate.is_file():
                with candidate.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
        except OSError:
            continue
    return _bind_submodule_worktree_identities(
        root,
        RepositoryTreeIdentity(
            repository_id=repository_id,
            tree_id=f"sha256:{digest.hexdigest()}",
        ),
        excluded_paths=excluded_paths,
        visited_repositories=_visited_repositories,
        status_snapshot=status,
    )


def _objective_goal_completion_policy(goal: ObjectiveGoal) -> dict[str, Any]:
    fields = goal.fields
    acceptance = str(
        fields.get("acceptance_criteria")
        or fields.get("acceptance")
        or fields.get("acceptance_criterion")
        or ""
    )
    return {
        "goal_id": str(goal.goal_id),
        "title": normalize_identity_text(goal.title),
        "goal": normalize_identity_text(fields.get("goal", "")),
        "conflict_policy": normalize_identity_text(
            fields.get("conflict_policy", "")
        ),
        "parents": sorted(str(item) for item in goal.parent_goal_ids),
        "acceptance": normalize_identity_text(acceptance),
        "required_evidence": sorted(
            normalize_identity_text(item) for item in goal.required_evidence
        ),
        "dependencies": sorted(str(item) for item in goal.dependencies),
        "outputs": sorted(str(item) for item in goal.predicted_files),
        "predicted_symbols": sorted(
            normalize_identity_text(item) for item in goal.predicted_symbols
        ),
        "validation": [
            normalize_identity_text(item) for item in goal.validation_commands
        ],
    }


def objective_goal_completion_revision(goal: ObjectiveGoal) -> str:
    """Return the canonical lifecycle-independent policy revision for one goal."""

    return canonical_content_cid(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/objective-goal-completion-policy@1",
            "goal": _objective_goal_completion_policy(goal),
        }
    )


def objective_completion_revision(
    objective_path: Path | None = None,
    *,
    goals: Sequence[ObjectiveGoal] | None = None,
) -> str:
    """Return a lifecycle-independent revision of the full completion policy.

    The objective markdown is mutable control state and is excluded from the
    repository-tree fence.  Completion proof therefore binds this separate
    semantic revision so changing a goal, its hierarchy, acceptance criteria,
    evidence surface, outputs, or validation policy invalidates prior proof,
    while status/diagnostic rewrites do not.
    """

    if goals is None:
        if objective_path is None or not Path(objective_path).exists():
            parsed_goals: Sequence[ObjectiveGoal] = ()
        else:
            parsed_goals = parse_goal_heap(
                Path(objective_path).read_text(encoding="utf-8", errors="replace")
            )
    else:
        parsed_goals = goals
    semantic_goals = [
        _objective_goal_completion_policy(goal)
        for goal in sorted(parsed_goals, key=lambda item: item.goal_id)
    ]
    return canonical_content_cid(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/objective-completion-policy@1",
            "goals": semantic_goals,
        }
    )


def completion_tree_identity(
    repo_root: Path,
    *,
    objective_path: Path,
    control_paths: Sequence[Path] = (),
    scan_exclude_paths: Iterable[str | Path] = (),
) -> RepositoryTreeIdentity:
    """Return source-tree identity without self-invalidating tracker writes.

    The objective document is mutable supervisor state.  Persisting a
    lifecycle transition must not immediately make otherwise-current evidence
    stale.  This calculation is byte-for-byte compatible with ``scan_identity``
    while excluding that document and explicit completion-control artifacts;
    every code, test, task-board, configuration, tracked, and untracked change
    remains in the digest.
    """

    resolved_scan_excludes = resolve_scan_exclude_paths(
        repo_root,
        scan_exclude_paths,
    )
    return _control_tree_identity(
        repo_root,
        excluded_paths=(
            objective_path,
            *(Path(path) for path in control_paths),
            *resolved_scan_excludes,
        ),
    )


def objective_materialization_tree_identity(
    repo_root: Path,
    *,
    objective_path: Path,
    journal_path: Path,
    control_paths: Sequence[Path] = (),
) -> RepositoryTreeIdentity:
    """Return a tree fence which ignores only this transaction's own files."""

    lock_path = objective_path.with_name(
        f".{objective_path.name}.admission.lock"
    )
    return _control_tree_identity(
        repo_root,
        excluded_paths=(
            objective_path,
            journal_path,
            lock_path,
            *(Path(path) for path in control_paths),
        ),
    )


def _goal_completion_records(
    goal: ObjectiveGoal,
    supplied_records: Mapping[str, Sequence[CompletionEvidence | Mapping[str, Any]]],
) -> list[CompletionEvidence]:
    """Read typed and legacy persisted evidence spellings without data loss."""

    records = [
        item if isinstance(item, CompletionEvidence) else CompletionEvidence.from_dict(item)
        for item in supplied_records.get(goal.goal_id, ())
    ]
    if goal.goal_id in supplied_records:
        return records
    raw_records = str(
        goal.fields.get("completion_evidence_records")
        or goal.fields.get("completion_evidence_json")
        or goal.fields.get("completion_receipts")
        or ""
    ).strip()
    if not raw_records:
        return []
    try:
        decoded = json.loads(raw_records)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if isinstance(decoded, Mapping):
        decoded = [decoded]
    if not isinstance(decoded, list):
        return []
    result: list[CompletionEvidence] = []
    for item in decoded:
        if not isinstance(item, Mapping):
            continue
        try:
            result.append(CompletionEvidence.from_dict(item))
        except (TypeError, ValueError):
            # A malformed historical record is not evidence for verification;
            # migration still proceeds provisionally instead of aborting the
            # rest of the board.
            continue
    return result


def _apply_completion_evidence_source_policy(
    records: Sequence[CompletionEvidence],
    *,
    repository_tree: str,
) -> list[CompletionEvidence]:
    """Attach deterministic source-policy decisions before completion gates."""

    evaluated: list[CompletionEvidence] = []
    for record in records:
        decision = completion_evidence_source_decision(
            record,
            repository_tree=repository_tree,
        )
        payload = record.to_dict()
        metadata = dict(payload.get("metadata") or {})
        metadata["evidence_source_policy"] = decision.to_dict()
        payload["metadata"] = metadata
        evaluated.append(CompletionEvidence.from_dict(payload))
    return evaluated


def _requires_external_completion(goal: ObjectiveGoal) -> bool:
    """Return whether a goal is explicitly or durably external-governed."""

    # The objective graph owns the canonical declaration and durable-history
    # aliases so reconciliation, generation, and task scheduling cannot drift.
    return goal.requires_external_completion


def _goal_completion_gate_record(
    goal: ObjectiveGoal,
    supplied_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    supplied = supplied_records.get(goal.goal_id)
    if isinstance(supplied, Mapping):
        return dict(supplied)
    raw = str(
        goal.fields.get("completion_gate_record")
        or goal.fields.get("completion_gate_json")
        or ""
    ).strip()
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(decoded) if isinstance(decoded, Mapping) else {}


def _atomic_rewrite(path: Path, text: str) -> None:
    """Replace a tracker document atomically, preserving its file mode."""

    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.migration-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        _fsync_parent_directory(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_parent_directory(path: Path) -> None:
    """Durably retain a preceding rename when the platform supports it."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path.parent, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically and durably replace a small tracker-owned JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_parent_directory(path)
    finally:
        if temporary.exists():
            temporary.unlink()


_OBJECTIVE_MATERIALIZATION_JOURNAL_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.objective_materialization_journal@1"
)


def _load_objective_materialization_journal(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema": _OBJECTIVE_MATERIALIZATION_JOURNAL_SCHEMA,
            "transactions": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("objective materialization journal must be a JSON object")
    if payload.get("schema") != _OBJECTIVE_MATERIALIZATION_JOURNAL_SCHEMA:
        raise ValueError("unsupported objective materialization journal schema")
    transactions = payload.get("transactions")
    if not isinstance(transactions, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, Mapping)
        for key, value in transactions.items()
    ):
        raise ValueError("objective materialization transactions must be an object")
    return {
        "schema": _OBJECTIVE_MATERIALIZATION_JOURNAL_SCHEMA,
        "transactions": {
            str(key): dict(value) for key, value in transactions.items()
        },
        **(
            {"latest_transaction_id": str(payload["latest_transaction_id"])}
            if payload.get("latest_transaction_id")
            else {}
        ),
    }


def _objective_materialization_transaction_id(
    preview: ObjectiveGoalMaterializationPreview,
    *,
    epoch_id: str = "",
    expected_goal_ids: Sequence[str] = (),
) -> str:
    material = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "objective_materialization_transaction@1"
        ),
        "base_heap_content_id": preview.base_heap_content_id,
        "candidate_heap_content_id": preview.candidate_heap_content_id,
        "root_goal_id": preview.root_goal_id,
        "root_content_id": preview.root_content_id,
        "proposal_ids": list(preview.admitted_proposal_ids),
        "lifecycle_owner": preview.policy.lifecycle_owner,
    }
    # Preserve the historical identity for legacy callers while making an
    # epoch-bound transaction impossible to alias to an unbound transaction.
    if epoch_id:
        material["epoch_id"] = epoch_id
    if expected_goal_ids:
        material["mapped_goal_ids"] = list(expected_goal_ids)
    digest = sha256(
        json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return f"objective-materialization:sha256:{digest}"


def _objective_transaction_result(
    *,
    objective_path: Path,
    journal_path: Path,
    preview: ObjectiveGoalMaterializationPreview,
    transaction_id: str,
    state: ObjectiveMaterializationTransactionState,
    repository_tree_id: str,
    epoch_id: str = "",
    expected_goal_ids: Sequence[str] = (),
    changed: bool = False,
    resumed: bool = False,
    reason_codes: Iterable[str] = (),
) -> ObjectiveMaterializationTransactionResult:
    return ObjectiveMaterializationTransactionResult(
        objective_path=objective_path,
        journal_path=journal_path,
        transaction_id=transaction_id,
        state=state,
        admitted_proposal_ids=(
            preview.admitted_proposal_ids
            if state is ObjectiveMaterializationTransactionState.COMMITTED
            else ()
        ),
        changed=bool(changed),
        resumed=bool(resumed),
        reason_codes=tuple(
            dict.fromkeys(
                str(item).strip() for item in reason_codes if str(item).strip()
            )
        ),
        base_heap_content_id=preview.base_heap_content_id,
        candidate_heap_content_id=preview.candidate_heap_content_id,
        repository_tree_id=repository_tree_id,
        root_goal_id=preview.root_goal_id,
        root_content_id=preview.root_content_id,
        epoch_id=epoch_id,
        mapped_goal_ids=(
            (
                tuple(expected_goal_ids)
                or tuple(item.goal.goal_id for item in preview.materialized)
            )
            if state is ObjectiveMaterializationTransactionState.COMMITTED
            else ()
        ),
    )


def _materialization_record_matches_preview(
    record: Mapping[str, Any],
    preview: ObjectiveGoalMaterializationPreview,
    *,
    epoch_id: str = "",
    expected_goal_ids: Sequence[str] = (),
) -> bool:
    matches = (
        str(record.get("base_heap_content_id") or "")
        == preview.base_heap_content_id
        and str(record.get("candidate_heap_content_id") or "")
        == preview.candidate_heap_content_id
        and str(record.get("root_goal_id") or "") == preview.root_goal_id
        and str(record.get("root_content_id") or "") == preview.root_content_id
        and tuple(record.get("proposal_ids") or ())
        == preview.admitted_proposal_ids
        and str(record.get("candidate_text") or "") == preview.candidate_text
    )
    if epoch_id:
        matches = matches and str(record.get("epoch_id") or "") == epoch_id
    if expected_goal_ids:
        matches = matches and tuple(record.get("mapped_goal_ids") or ()) == tuple(
            expected_goal_ids
        )
    return matches


def _normalize_objective_epoch_fences(
    *,
    preview: ObjectiveGoalMaterializationPreview,
    epoch_id: str,
    expected_objective_revision: str,
    expected_goal_ids: Sequence[str],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Normalize optional refill-epoch fences without weakening legacy calls."""

    if not isinstance(epoch_id, str):
        raise TypeError("epoch_id must be a string")
    normalized_epoch_id = epoch_id.strip()
    if (
        "\x00" in normalized_epoch_id
        or len(normalized_epoch_id.encode("utf-8")) > 512
    ):
        raise ValueError("epoch_id exceeds its safe text bound")
    if not isinstance(expected_objective_revision, str):
        raise TypeError("expected_objective_revision must be a string")
    normalized_revision = expected_objective_revision.strip()
    if (
        "\x00" in normalized_revision
        or len(normalized_revision.encode("utf-8")) > 512
    ):
        raise ValueError("expected_objective_revision exceeds its safe text bound")
    if isinstance(expected_goal_ids, (str, bytes)) or not isinstance(
        expected_goal_ids, Sequence
    ):
        raise TypeError("expected_goal_ids must be a sequence of goal IDs")
    normalized_goal_ids = tuple(
        str(item).strip() for item in expected_goal_ids
    )
    if any(
        not item or "\x00" in item or len(item.encode("utf-8")) > 512
        for item in normalized_goal_ids
    ):
        raise ValueError("expected_goal_ids must contain bounded non-empty text")
    if len(normalized_goal_ids) > 8:
        raise ValueError("a refill epoch may map at most 8 objective goals")
    if len(set(normalized_goal_ids)) != len(normalized_goal_ids):
        raise ValueError("expected_goal_ids must be unique")

    preview_goal_ids = tuple(
        item.goal.goal_id for item in preview.materialized
    )
    reason_codes: list[str] = []
    if (
        normalized_revision
        and normalized_revision != preview.base_heap_content_id
    ):
        reason_codes.append("objective_revision_conflict")
    if normalized_goal_ids and normalized_goal_ids != preview_goal_ids:
        reason_codes.append("goal_mapping_conflict")
    return normalized_epoch_id, normalized_goal_ids, tuple(reason_codes)


def _materialization_goal_errors(
    text: str,
    preview: ObjectiveGoalMaterializationPreview,
    *,
    require_all: bool,
) -> list[str]:
    """Reparse and compare every field whose preservation is contractual."""

    goals = parse_goal_heap(text)
    ids = [goal.goal_id for goal in goals]
    errors: list[str] = []
    duplicate_ids = sorted(
        goal_id for goal_id in set(ids) if ids.count(goal_id) > 1
    )
    if duplicate_ids:
        errors.append("duplicate_goal_id")
    by_id = {goal.goal_id: goal for goal in goals}
    known_ids = set(by_id)
    if any(
        parent not in known_ids
        for goal in goals
        for parent in goal.parent_goal_ids
    ):
        errors.append("unresolved_parent")

    state: dict[str, int] = {}

    def visit(goal_id: str) -> bool:
        marker = state.get(goal_id, 0)
        if marker == 1:
            return False
        if marker == 2:
            return True
        state[goal_id] = 1
        valid = all(
            visit(parent)
            for parent in by_id[goal_id].parent_goal_ids
            if parent in by_id
        )
        state[goal_id] = 2
        return valid

    if any(not visit(goal_id) for goal_id in sorted(by_id)):
        errors.append("parent_cycle")

    for materialized in preview.materialized:
        proposal = materialized.proposal
        goal = by_id.get(proposal.canonical_id)
        if goal is None:
            if require_all:
                errors.append(f"missing_goal:{proposal.canonical_id}")
            continue
        fields = goal.fields
        exact_values = {
            "canonical_proposal_id": proposal.canonical_id,
            "semantic_key": proposal.semantic_key,
            "proposal_kind": proposal.kind.value,
            "proposal_source": proposal.source,
            "proposal_source_id": proposal.source_id,
            "lifecycle_owner": preview.policy.lifecycle_owner,
            "goal": "; ".join(proposal.parent_objective_terms) or proposal.title,
            "evidence": ", ".join(proposal.expected_evidence_delta),
            "depends_on": ", ".join(proposal.dependencies),
            "outputs": ", ".join(proposal.predicted_files),
            "predicted_symbols": ", ".join(proposal.predicted_symbols),
            "validation": "; ".join(proposal.validation_commands),
            "graph_depth": str(materialized.graph_depth),
            "parent_goal_ids_json": json.dumps(
                list(materialized.parent_goal_ids),
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            "dependencies_json": json.dumps(
                list(proposal.dependencies),
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            "evidence_requirements_json": json.dumps(
                list(proposal.expected_evidence_delta),
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            "predicted_files_json": json.dumps(
                list(proposal.predicted_files),
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            "predicted_symbols_json": json.dumps(
                list(proposal.predicted_symbols),
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            "validation_commands_json": json.dumps(
                list(proposal.validation_commands),
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        }
        for key, expected in exact_values.items():
            if str(fields.get(key) or "") != expected:
                errors.append(f"metadata_mismatch:{proposal.canonical_id}:{key}")
        if tuple(goal.parent_goal_ids) != materialized.parent_goal_ids:
            errors.append(f"parent_mismatch:{proposal.canonical_id}")
        if materialized.graph_depth != proposal.depth:
            errors.append(f"depth_mismatch:{proposal.canonical_id}")
    return sorted(set(errors))


def _append_materialized_prefix(
    base_text: str,
    preview: ObjectiveGoalMaterializationPreview,
    count: int,
) -> str:
    selected = preview.materialized[: max(0, int(count))]
    if not selected:
        return base_text
    separator = "\n\n" if base_text.rstrip() else ""
    return (
        base_text.rstrip()
        + separator
        + "\n\n".join(item.rendered_block.strip() for item in selected)
        + "\n"
    )


def _validate_materialization_lease(
    lease_guard: Callable[..., Any] | None,
    expected_lease_token: str | int | None,
) -> str:
    """Run a caller-supplied fencing check immediately before heap mutation."""

    if lease_guard is None:
        return (
            "lease guard is required for the expected fencing token"
            if expected_lease_token is not None
            else ""
        )
    try:
        value = (
            lease_guard(expected_lease_token)
            if expected_lease_token is not None
            else lease_guard()
        )
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    if value is False or value is None:
        return "lease guard did not confirm ownership"
    if expected_lease_token is None:
        return ""
    if isinstance(value, Mapping):
        actual = value.get(
            "fencing_token",
            value.get("lease_token", value.get("token")),
        )
    else:
        actual = getattr(
            value,
            "fencing_token",
            getattr(value, "lease_token", value),
        )
    if str(actual) != str(expected_lease_token):
        return "lease fencing token changed"
    return ""


def commit_objective_goal_materialization(
    *,
    repo_root: Path,
    objective_path: Path,
    journal_path: Path,
    preview: ObjectiveGoalMaterializationPreview,
    expected_repository_tree_id: str = "",
    epoch_id: str = "",
    expected_objective_revision: str = "",
    expected_goal_ids: Sequence[str] = (),
    lease_guard: Callable[..., Any] | None = None,
    expected_lease_token: str | int | None = None,
    lock_timeout_seconds: float = 30.0,
    control_paths: Sequence[Path] = (),
) -> ObjectiveMaterializationTransactionResult:
    """Commit a preview with durable CAS fencing and idempotent recovery.

    The journal is written ``prepared`` before the heap replacement.  If the
    process stops after replacing the heap but before marking the journal
    committed, a later call recognizes either the exact candidate or an exact
    prefix of its rendered blocks and safely completes the same transaction.
    Stale source trees, roots, heap content, and lease fencing tokens leave the
    prepared record intact for a fresh preview or lease to resume.  Refill
    callers may additionally bind the transaction to an epoch, its frozen
    objective revision, and the exact (maximum eight) goal IDs expected from
    the immutable preview.
    """

    if not isinstance(preview, ObjectiveGoalMaterializationPreview):
        raise TypeError("preview must be ObjectiveGoalMaterializationPreview")
    normalized_epoch_id, normalized_goal_ids, fence_errors = (
        _normalize_objective_epoch_fences(
            preview=preview,
            epoch_id=epoch_id,
            expected_objective_revision=expected_objective_revision,
            expected_goal_ids=expected_goal_ids,
        )
    )
    transaction_id = _objective_materialization_transaction_id(
        preview,
        epoch_id=normalized_epoch_id,
        expected_goal_ids=normalized_goal_ids,
    )
    if fence_errors:
        return _objective_transaction_result(
            objective_path=objective_path,
            journal_path=journal_path,
            preview=preview,
            transaction_id=transaction_id,
            state=ObjectiveMaterializationTransactionState.BLOCKED,
            repository_tree_id=str(expected_repository_tree_id or ""),
            epoch_id=normalized_epoch_id,
            expected_goal_ids=normalized_goal_ids,
            reason_codes=fence_errors,
        )
    if not preview.ready or not preview.changed:
        reasons = [
            "preview_not_ready",
            *preview.fatal_reasons,
            *(item.reason for item in preview.rejected),
        ]
        return _objective_transaction_result(
            objective_path=objective_path,
            journal_path=journal_path,
            preview=preview,
            transaction_id=transaction_id,
            state=ObjectiveMaterializationTransactionState.BLOCKED,
            repository_tree_id=str(expected_repository_tree_id or ""),
            epoch_id=normalized_epoch_id,
            expected_goal_ids=normalized_goal_ids,
            reason_codes=reasons,
        )

    objective_path = objective_path.resolve()
    journal_path = journal_path.resolve()
    repo_root = repo_root.resolve()
    if objective_path == journal_path:
        raise ValueError("journal_path must be separate from objective_path")
    lock_path = objective_path.with_name(f".{objective_path.name}.admission.lock")

    from .duckdb_state import exclusive_file_lock

    try:
        lock = exclusive_file_lock(
            lock_path, timeout_seconds=max(0.0, float(lock_timeout_seconds))
        )
        with lock:
            journal = _load_objective_materialization_journal(journal_path)
            transactions = dict(journal["transactions"])
            prior = transactions.get(transaction_id)
            resumed = prior is not None
            if prior is not None and not _materialization_record_matches_preview(
                prior,
                preview,
                epoch_id=normalized_epoch_id,
                expected_goal_ids=normalized_goal_ids,
            ):
                return _objective_transaction_result(
                    objective_path=objective_path,
                    journal_path=journal_path,
                    preview=preview,
                    transaction_id=transaction_id,
                    state=ObjectiveMaterializationTransactionState.BLOCKED,
                    repository_tree_id=str(
                        prior.get("repository_tree_id")
                        or expected_repository_tree_id
                        or ""
                    ),
                    epoch_id=normalized_epoch_id,
                    expected_goal_ids=normalized_goal_ids,
                    resumed=True,
                    reason_codes=("transaction_identity_conflict",),
                )

            current_text = (
                objective_path.read_text(encoding="utf-8")
                if objective_path.exists()
                else ""
            )
            current_goals = parse_goal_heap(current_text)
            current_root = next(
                (
                    goal
                    for goal in current_goals
                    if goal.goal_id == preview.root_goal_id
                ),
                None,
            )
            if (
                preview.root_goal_id
                and (
                    current_root is None
                    or objective_goal_content_id(current_root)
                    != preview.root_content_id
                )
            ):
                return _objective_transaction_result(
                    objective_path=objective_path,
                    journal_path=journal_path,
                    preview=preview,
                    transaction_id=transaction_id,
                    state=ObjectiveMaterializationTransactionState.BLOCKED,
                    repository_tree_id=str(expected_repository_tree_id or ""),
                    epoch_id=normalized_epoch_id,
                    expected_goal_ids=normalized_goal_ids,
                    resumed=resumed,
                    reason_codes=("changed_root",),
                )

            current_content_id = objective_heap_content_id(current_text)
            metadata_errors = _materialization_goal_errors(
                current_text, preview, require_all=False
            )
            conflicting_metadata = [
                item for item in metadata_errors if item.startswith("metadata_mismatch:")
            ]
            if conflicting_metadata:
                return _objective_transaction_result(
                    objective_path=objective_path,
                    journal_path=journal_path,
                    preview=preview,
                    transaction_id=transaction_id,
                    state=ObjectiveMaterializationTransactionState.BLOCKED,
                    repository_tree_id=str(expected_repository_tree_id or ""),
                    epoch_id=normalized_epoch_id,
                    expected_goal_ids=normalized_goal_ids,
                    resumed=resumed,
                    reason_codes=("partial_write_conflict", *conflicting_metadata),
                )

            # A committed record remains idempotent after later transactions
            # append unrelated goals.  Revalidate its exact nodes and root
            # instead of requiring the entire historical heap digest.
            if prior is not None and prior.get("state") == "committed":
                complete_errors = _materialization_goal_errors(
                    current_text, preview, require_all=True
                )
                if not complete_errors:
                    return _objective_transaction_result(
                        objective_path=objective_path,
                        journal_path=journal_path,
                        preview=preview,
                        transaction_id=transaction_id,
                        state=ObjectiveMaterializationTransactionState.COMMITTED,
                        repository_tree_id=str(
                            prior.get("repository_tree_id")
                            or expected_repository_tree_id
                            or ""
                        ),
                        epoch_id=normalized_epoch_id,
                        expected_goal_ids=normalized_goal_ids,
                        resumed=True,
                    )
                return _objective_transaction_result(
                    objective_path=objective_path,
                    journal_path=journal_path,
                    preview=preview,
                    transaction_id=transaction_id,
                    state=ObjectiveMaterializationTransactionState.BLOCKED,
                    repository_tree_id=str(
                        prior.get("repository_tree_id")
                        or expected_repository_tree_id
                        or ""
                    ),
                    epoch_id=normalized_epoch_id,
                    expected_goal_ids=normalized_goal_ids,
                    resumed=True,
                    reason_codes=("committed_heap_conflict", *complete_errors),
                )

            identity = objective_materialization_tree_identity(
                repo_root,
                objective_path=objective_path,
                journal_path=journal_path,
                control_paths=control_paths,
            )
            frozen_tree_id = str(
                (prior or {}).get("repository_tree_id")
                or expected_repository_tree_id
                or identity.tree_id
            )
            if expected_repository_tree_id and (
                str(expected_repository_tree_id) != frozen_tree_id
            ):
                return _objective_transaction_result(
                    objective_path=objective_path,
                    journal_path=journal_path,
                    preview=preview,
                    transaction_id=transaction_id,
                    state=ObjectiveMaterializationTransactionState.BLOCKED,
                    repository_tree_id=frozen_tree_id,
                    epoch_id=normalized_epoch_id,
                    expected_goal_ids=normalized_goal_ids,
                    resumed=resumed,
                    reason_codes=("transaction_tree_conflict",),
                )
            if identity.tree_id != frozen_tree_id:
                return _objective_transaction_result(
                    objective_path=objective_path,
                    journal_path=journal_path,
                    preview=preview,
                    transaction_id=transaction_id,
                    state=ObjectiveMaterializationTransactionState.BLOCKED,
                    repository_tree_id=frozen_tree_id,
                    epoch_id=normalized_epoch_id,
                    expected_goal_ids=normalized_goal_ids,
                    resumed=resumed,
                    reason_codes=("stale_repository_tree",),
                )

            prefix_count = -1
            base_text = str((prior or {}).get("base_text") or "")
            if current_content_id == preview.base_heap_content_id:
                prefix_count = 0
                base_text = current_text
            elif prior is not None and base_text:
                for count in range(1, len(preview.materialized) + 1):
                    if objective_heap_content_id(
                        _append_materialized_prefix(base_text, preview, count)
                    ) == current_content_id:
                        prefix_count = count
                        break
            if (
                current_content_id == preview.candidate_heap_content_id
                and (prior is not None or not normalized_epoch_id)
            ):
                prefix_count = len(preview.materialized)
            if prefix_count < 0:
                return _objective_transaction_result(
                    objective_path=objective_path,
                    journal_path=journal_path,
                    preview=preview,
                    transaction_id=transaction_id,
                    state=ObjectiveMaterializationTransactionState.BLOCKED,
                    repository_tree_id=frozen_tree_id,
                    epoch_id=normalized_epoch_id,
                    expected_goal_ids=normalized_goal_ids,
                    resumed=resumed,
                    reason_codes=("stale_objective_heap",),
                )

            now = utc_now()
            prepared = dict(prior or {})
            prepared.update(
                {
                    "schema": (
                        "ipfs_accelerate_py.agent_supervisor."
                        "objective_materialization_transaction@1"
                    ),
                    "transaction_id": transaction_id,
                    "state": "prepared",
                    "prepared_at": str(prepared.get("prepared_at") or now),
                    "updated_at": now,
                    "objective_path": str(objective_path),
                    "base_heap_content_id": preview.base_heap_content_id,
                    "candidate_heap_content_id": preview.candidate_heap_content_id,
                    "root_goal_id": preview.root_goal_id,
                    "root_content_id": preview.root_content_id,
                    "repository_id": identity.repository_id,
                    "repository_tree_id": frozen_tree_id,
                    "proposal_ids": list(preview.admitted_proposal_ids),
                    "epoch_id": normalized_epoch_id,
                    "mapped_goal_ids": list(
                        normalized_goal_ids
                        or tuple(
                            item.goal.goal_id
                            for item in preview.materialized
                        )
                    ),
                    "candidate_text": preview.candidate_text,
                    "base_text": base_text,
                    "rendered_block_digests": {
                        item.proposal.canonical_id: sha256(
                            item.rendered_block.encode("utf-8")
                        ).hexdigest()
                        for item in preview.materialized
                    },
                    "lifecycle_owner": preview.policy.lifecycle_owner,
                    "last_error": "",
                }
            )
            transactions[transaction_id] = prepared
            journal.update(
                {
                    "transactions": transactions,
                    "latest_transaction_id": transaction_id,
                }
            )
            _atomic_write_json(journal_path, journal)

            lease_error = _validate_materialization_lease(
                lease_guard, expected_lease_token
            )
            if lease_error:
                prepared["updated_at"] = utc_now()
                prepared["last_error"] = lease_error
                prepared["reason_codes"] = ["lease_conflict"]
                transactions[transaction_id] = prepared
                journal["transactions"] = transactions
                _atomic_write_json(journal_path, journal)
                return _objective_transaction_result(
                    objective_path=objective_path,
                    journal_path=journal_path,
                    preview=preview,
                    transaction_id=transaction_id,
                    state=ObjectiveMaterializationTransactionState.PREPARED,
                    repository_tree_id=frozen_tree_id,
                    epoch_id=normalized_epoch_id,
                    expected_goal_ids=normalized_goal_ids,
                    resumed=resumed,
                    reason_codes=("lease_conflict",),
                )

            changed = prefix_count < len(preview.materialized)
            if changed:
                _atomic_rewrite(objective_path, preview.candidate_text)
            persisted_text = objective_path.read_text(encoding="utf-8")
            if (
                objective_heap_content_id(persisted_text)
                != preview.candidate_heap_content_id
            ):
                prepared["updated_at"] = utc_now()
                prepared["last_error"] = "objective heap replacement was incomplete"
                prepared["reason_codes"] = ["partial_write"]
                transactions[transaction_id] = prepared
                journal["transactions"] = transactions
                _atomic_write_json(journal_path, journal)
                return _objective_transaction_result(
                    objective_path=objective_path,
                    journal_path=journal_path,
                    preview=preview,
                    transaction_id=transaction_id,
                    state=ObjectiveMaterializationTransactionState.PREPARED,
                    repository_tree_id=frozen_tree_id,
                    epoch_id=normalized_epoch_id,
                    expected_goal_ids=normalized_goal_ids,
                    changed=changed,
                    resumed=resumed,
                    reason_codes=("partial_write",),
                )
            persisted_errors = _materialization_goal_errors(
                persisted_text, preview, require_all=True
            )
            if persisted_errors:
                prepared["updated_at"] = utc_now()
                prepared["last_error"] = "; ".join(persisted_errors)
                prepared["reason_codes"] = ["partial_write"]
                transactions[transaction_id] = prepared
                journal["transactions"] = transactions
                _atomic_write_json(journal_path, journal)
                return _objective_transaction_result(
                    objective_path=objective_path,
                    journal_path=journal_path,
                    preview=preview,
                    transaction_id=transaction_id,
                    state=ObjectiveMaterializationTransactionState.PREPARED,
                    repository_tree_id=frozen_tree_id,
                    epoch_id=normalized_epoch_id,
                    expected_goal_ids=normalized_goal_ids,
                    changed=changed,
                    resumed=resumed,
                    reason_codes=("partial_write", *persisted_errors),
                )

            prepared.update(
                {
                    "state": "committed",
                    "committed_at": utc_now(),
                    "updated_at": utc_now(),
                    "last_error": "",
                    "reason_codes": [],
                }
            )
            transactions[transaction_id] = prepared
            journal["transactions"] = transactions
            _atomic_write_json(journal_path, journal)
            return _objective_transaction_result(
                objective_path=objective_path,
                journal_path=journal_path,
                preview=preview,
                transaction_id=transaction_id,
                state=ObjectiveMaterializationTransactionState.COMMITTED,
                repository_tree_id=frozen_tree_id,
                epoch_id=normalized_epoch_id,
                expected_goal_ids=(
                    normalized_goal_ids
                    or tuple(
                        item.goal.goal_id for item in preview.materialized
                    )
                ),
                changed=changed,
                resumed=resumed or prefix_count > 0,
            )
    except TimeoutError:
        return _objective_transaction_result(
            objective_path=objective_path,
            journal_path=journal_path,
            preview=preview,
            transaction_id=transaction_id,
            state=ObjectiveMaterializationTransactionState.BLOCKED,
            repository_tree_id=str(expected_repository_tree_id or ""),
            epoch_id=normalized_epoch_id,
            expected_goal_ids=normalized_goal_ids,
            reason_codes=("lease_conflict",),
        )


# Compatibility spellings for generation-ledger callers.
commit_objective_work_materialization = commit_objective_goal_materialization
apply_objective_goal_materialization = commit_objective_goal_materialization


def migrate_legacy_objective_goals(
    *,
    repo_root: Path,
    objective_path: Path,
    todo_path: Path | None = None,
    task_header_prefix: str = "",
    todo_boards: Sequence[tuple[Path, str]] | None = None,
    completion_evidence_records: Mapping[
        str, Sequence[CompletionEvidence | Mapping[str, Any]]
    ] | None = None,
    completion_gate_records: Mapping[str, Mapping[str, Any]] | None = None,
    completion_control_paths: Sequence[Path] = (),
    require_artifact_binding: bool = False,
    scan_exclude_paths: Iterable[str | Path] = (),
    goal_ids: Iterable[str] | None = None,
    preview: bool = False,
    max_goals: int | None = None,
    now: str | None = None,
    evidence_freshness_seconds: float = DEFAULT_EVIDENCE_FRESHNESS_SECONDS,
) -> ObjectiveGoalMigrationResult:
    """Migrate ambiguous legacy completed goals in an atomic, bounded batch.

    The objective document is its own durable checkpoint.  Each migrated goal
    receives a canonical lifecycle state and deterministic migration ID, so a
    rerun naturally skips committed work and resumes the remaining legacy
    labels.  ``preview`` performs the same classification without writing.
    """

    if max_goals is not None and (
        isinstance(max_goals, bool) or not isinstance(max_goals, int) or max_goals < 0
    ):
        raise ValueError("max_goals must be a non-negative integer or None")
    if not objective_path.exists():
        return ObjectiveGoalMigrationResult(
            objective_path, bool(preview), 0, [], [], [], [], [], []
        )
    text = objective_path.read_text(encoding="utf-8")
    goals = parse_goal_heap(text)
    _external_goal_ids, external_blocked_goal_ids = (
        external_authority_goal_fence(goals)
    )
    supplied_records = completion_evidence_records or {}

    def supplied_external_records(goal_id: str) -> bool:
        raw_records = supplied_records.get(goal_id, ())
        if not raw_records:
            return False
        normalized: list[CompletionEvidence] = []
        for raw in raw_records:
            try:
                normalized.append(
                    raw
                    if isinstance(raw, CompletionEvidence)
                    else CompletionEvidence.from_dict(raw)
                )
            except (TypeError, ValueError):
                return False
        return bool(normalized) and all(
            record.metadata.get("external_operational_completion") is True
            for record in normalized
        )

    selected_ids = {str(item).strip() for item in goal_ids or () if str(item).strip()}
    candidates = [
        goal for goal in goals
        if is_legacy_completed_goal_state(goal.status)
        and (
            goal.goal_id not in external_blocked_goal_ids
            or (
                goal.goal_id in _external_goal_ids
                and supplied_external_records(goal.goal_id)
            )
        )
        and (not selected_ids or goal.goal_id in selected_ids)
    ]
    limit = len(candidates) if max_goals is None else max_goals
    batch = candidates[:limit]
    remaining = candidates[limit:]
    gate_records = completion_gate_records or {}
    boards: list[tuple[Path, str]] = []
    if todo_path is not None:
        boards.append((todo_path, task_header_prefix))
    boards.extend(todo_boards or ())
    open_goals = open_goal_ids_from_todo_boards(boards)
    identity = completion_tree_identity(
        repo_root,
        objective_path=objective_path,
        control_paths=completion_control_paths,
        scan_exclude_paths=scan_exclude_paths,
    )
    hierarchy = goal_graph(goals)
    goals_by_id = {goal.goal_id: goal for goal in goals}
    updates: dict[str, dict[str, str]] = {}
    records_out: list[dict[str, Any]] = []
    provisional: list[str] = []
    verified: list[str] = []
    migrated_at = now or utc_now()

    def descendants(goal_id: str) -> list[dict[str, Any]]:
        pending = list(hierarchy.get("children", {}).get(goal_id, ()))
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        while pending:
            child_id = str(pending.pop(0))
            if not child_id or child_id in seen:
                continue
            seen.add(child_id)
            child = goals_by_id.get(child_id)
            if child is not None:
                child_state = (
                    GoalState.PROVISIONALLY_COMPLETE
                    if is_legacy_completed_goal_state(child.status)
                    else normalize_goal_state(child.status)
                )
                result.append({
                    "goal_id": child_id,
                    "state": child_state.value,
                    "verified": child_state is GoalState.VERIFIED_COMPLETE,
                })
            pending.extend(hierarchy.get("children", {}).get(child_id, ()))
        return result

    for goal in batch:
        if require_artifact_binding:
            evidence = [
                item
                if isinstance(item, CompletionEvidence)
                else CompletionEvidence.from_dict(item)
                for item in supplied_records.get(goal.goal_id, ())
            ]
            supplied_gate = gate_records.get(goal.goal_id)
            gate = dict(supplied_gate) if isinstance(supplied_gate, Mapping) else {}
        else:
            evidence = _goal_completion_records(goal, supplied_records)
            gate = _goal_completion_gate_record(goal, gate_records)
        evidence = _apply_completion_evidence_source_policy(
            evidence,
            repository_tree=identity.tree_id,
        )
        criteria = str(
            goal.fields.get("acceptance_criteria")
            or goal.fields.get("acceptance")
            or ""
        ).strip() or goal.required_evidence
        migration = migrate_legacy_goal_completion(
            goal_id=goal.goal_id,
            legacy_state=goal.status,
            acceptance_criteria=criteria,
            evidence=evidence,
            tasks_complete=goal.goal_id not in open_goals,
            coverage=gate.get("coverage"),
            analyzer_health=gate.get("analyzer_health"),
            exhaustion_quorum=gate.get("exhaustion_quorum"),
            child_goals=[*descendants(goal.goal_id), *gate.get("child_goals", ())],
            analysis_result=gate.get("analysis_result"),
            analysis_inconclusive=bool(gate.get("analysis_inconclusive", False)),
            repository_tree=identity.tree_id,
            repository_id=identity.repository_id,
            objective_revision=objective_goal_completion_revision(goal),
            completion_binding=gate.get("binding"),
            require_artifact_binding=require_artifact_binding,
            now=now,
            freshness_seconds=evidence_freshness_seconds,
        )
        payload = migration.to_dict()
        diagnostics = payload["diagnostics"]
        records_out.append(payload)
        target = migration.state.value
        (verified if migration.verified else provisional).append(goal.goal_id)
        updates[goal.goal_id] = {
            "Status": target,
            "Goal completion schema version": str(GOAL_COMPLETION_SCHEMA_VERSION),
            "Legacy completion state": goal.status,
            "Completion migration id": migration.migration_id,
            "Completion migrated at": migrated_at,
            "Completion migration reason": "; ".join(migration.reason_codes),
            "Completion confidence": str(diagnostics["confidence"]),
            "Uncovered criteria": json.dumps(diagnostics["uncovered_criteria"], separators=(",", ":")),
            "Stale evidence": json.dumps(diagnostics["stale_evidence"], sort_keys=True, separators=(",", ":")),
            "Analyzer health": json.dumps(diagnostics["analyzer_health"], sort_keys=True, separators=(",", ":")),
            "Exhaustion quorum": json.dumps(diagnostics["exhaustion_quorum"], sort_keys=True, separators=(",", ":")),
            "Reopen reasons": json.dumps(diagnostics["reopen_reasons"], separators=(",", ":")),
        }
        if evidence:
            updates[goal.goal_id]["Completion evidence records"] = json.dumps(
                [record.to_dict() for record in evidence], sort_keys=True, separators=(",", ":")
            )
        if gate:
            updates[goal.goal_id]["Completion gate record"] = json.dumps(
                gate, sort_keys=True, separators=(",", ":"), default=str
            )

    if updates and not preview:
        _atomic_rewrite(objective_path, rewrite_goal_fields(text, updates))
    migrated_ids = [goal.goal_id for goal in batch]
    return ObjectiveGoalMigrationResult(
        objective_path=objective_path,
        preview=bool(preview),
        scanned_goal_count=len(goals),
        candidate_goal_ids=[goal.goal_id for goal in candidates],
        migrated_goal_ids=migrated_ids,
        provisional_goal_ids=provisional,
        verified_goal_ids=verified,
        remaining_goal_ids=[goal.goal_id for goal in remaining],
        records=records_out,
    )


def reconcile_objective_goal_completion(
    *,
    repo_root: Path,
    objective_path: Path,
    todo_path: Path | None = None,
    task_header_prefix: str = "",
    todo_boards: Sequence[tuple[Path, str]] | None = None,
    embedding_min_score: float = DEFAULT_EMBEDDING_MIN_SCORE,
    completion_evidence_records: Mapping[
        str, Sequence[CompletionEvidence | Mapping[str, Any]]
    ] | None = None,
    completion_gate_records: Mapping[str, Mapping[str, Any]] | None = None,
    completion_control_paths: Sequence[Path] = (),
    require_artifact_binding: bool = False,
    external_completion_authority: (
        ExternalCompletionAuthority | Mapping[str, Any] | None
    ) = None,
    scan_exclude_paths: Iterable[str | Path] = (),
    now: str | None = None,
    evidence_freshness_seconds: float = DEFAULT_EVIDENCE_FRESHNESS_SECONDS,
) -> ObjectiveCompletionResult:
    """Reconcile objective goals through the evidence-backed lifecycle.

    Closed tasks advance an active goal to ``provisionally_complete``.  A
    separate reconciliation can advance that provisional goal to
    ``verified_complete`` only when every acceptance criterion has a fresh,
    tree-bound evidence record and the goal's validation command passes.
    """

    if not objective_path.exists():
        return ObjectiveCompletionResult(
            objective_path=objective_path,
            completed_goal_ids=[],
            active_goal_count=0,
            completed_goal_count=0,
            completion_evidence={},
            validation_results={},
        )

    supplied_records = dict(completion_evidence_records or {})
    supplied_gate_records = completion_gate_records or {}
    initial_text = objective_path.read_text(encoding="utf-8")
    initial_goals = parse_goal_heap(initial_text)
    external_completion: ExternalCompletionEvaluation | None = None
    current_authority_goal_ids: set[str] = set()
    externally_governed_goal_ids = {
        goal.goal_id
        for goal in initial_goals
        if goal.goal_id and _requires_external_completion(goal)
    }
    for goal_id in externally_governed_goal_ids:
        # Once an operational goal has external provenance, omitting the
        # explicit authority cannot downgrade it into ordinary local evidence.
        supplied_records[goal_id] = []
    if external_completion_authority is not None:
        external_completion = evaluate_external_completion_authority(
            external_completion_authority,
            repo_root=repo_root,
            objective_path=objective_path,
            goal_evidence_terms={
                goal.goal_id: tuple(goal.required_evidence)
                for goal in initial_goals
                if goal.goal_id
            },
            now=now,
            freshness_seconds=evidence_freshness_seconds,
        )
        current_authority_goal_ids = set(
            external_completion.governed_goal_ids
        )
        externally_governed_goal_ids.update(current_authority_goal_ids)
        for goal_id in current_authority_goal_ids:
            # External authority replaces, rather than augments, locally
            # persisted/task-produced evidence for governed operational goals.
            supplied_records[goal_id] = list(
                external_completion.evidence_records.get(goal_id, ())
            )
    migration_result = migrate_legacy_objective_goals(
        repo_root=repo_root,
        objective_path=objective_path,
        todo_path=todo_path,
        task_header_prefix=task_header_prefix,
        todo_boards=todo_boards,
        completion_evidence_records=supplied_records,
        completion_gate_records=supplied_gate_records,
        completion_control_paths=completion_control_paths,
        require_artifact_binding=require_artifact_binding,
        scan_exclude_paths=scan_exclude_paths,
        now=now,
        evidence_freshness_seconds=evidence_freshness_seconds,
    )
    text = objective_path.read_text(encoding="utf-8")
    goals = parse_goal_heap(text)
    if require_artifact_binding:
        goal_ids = [goal.goal_id for goal in goals]
        duplicate_goal_ids = sorted(
            {
                goal_id
                for goal_id in goal_ids
                if goal_id and goal_ids.count(goal_id) > 1
            }
        )
        if duplicate_goal_ids:
            raise ValueError(
                "objective completion graph contains duplicate goal ids: "
                + ", ".join(duplicate_goal_ids)
            )
        known_goal_ids = {goal_id for goal_id in goal_ids if goal_id}
        unknown_parents = sorted(
            {
                parent
                for goal in goals
                for parent in goal.parent_goal_ids
                if parent and parent not in known_goal_ids
            }
        )
        if unknown_parents:
            raise ValueError(
                "objective completion graph contains unknown parent ids: "
                + ", ".join(unknown_parents)
            )
        parents_by_goal = {
            goal.goal_id: tuple(
                parent for parent in goal.parent_goal_ids if parent
            )
            for goal in goals
            if goal.goal_id
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def verify_acyclic(goal_id: str) -> None:
            if goal_id in visited:
                return
            if goal_id in visiting:
                raise ValueError(
                    "objective completion graph contains a parent cycle at "
                    f"{goal_id}"
                )
            visiting.add(goal_id)
            for parent_id in parents_by_goal.get(goal_id, ()):
                verify_acyclic(parent_id)
            visiting.remove(goal_id)
            visited.add(goal_id)

        for goal_id in sorted(parents_by_goal):
            verify_acyclic(goal_id)
    externally_governed_goal_ids.update(
        goal.goal_id
        for goal in goals
        if goal.goal_id and _requires_external_completion(goal)
    )
    if external_completion_authority is not None:
        # Migration may rewrite the tracked objective heap. Reinspect after
        # that phase so no pre-migration clean snapshot can authorize the
        # resulting source state.
        external_completion = evaluate_external_completion_authority(
            external_completion_authority,
            repo_root=repo_root,
            objective_path=objective_path,
            goal_evidence_terms={
                goal.goal_id: tuple(goal.required_evidence)
                for goal in goals
                if goal.goal_id
            },
            now=now,
            freshness_seconds=evidence_freshness_seconds,
        )
        current_authority_goal_ids = set(
            external_completion.governed_goal_ids
        )
        externally_governed_goal_ids.update(current_authority_goal_ids)
        for goal_id in externally_governed_goal_ids:
            supplied_records[goal_id] = list(
                external_completion.evidence_records.get(goal_id, ())
            )
    else:
        for goal_id in externally_governed_goal_ids:
            supplied_records[goal_id] = []
    repository_identity = completion_tree_identity(
        repo_root,
        objective_path=objective_path,
        control_paths=completion_control_paths,
        scan_exclude_paths=scan_exclude_paths,
    )
    candidate_goals = []
    persisted_records: dict[str, list[CompletionEvidence]] = {}
    for goal in goals:
        if require_artifact_binding:
            records = [
                item
                if isinstance(item, CompletionEvidence)
                else CompletionEvidence.from_dict(item)
                for item in supplied_records.get(goal.goal_id, ())
            ]
        else:
            records = _goal_completion_records(goal, supplied_records)
        if (
            external_completion is not None
            and goal.goal_id in current_authority_goal_ids
        ):
            rebound_records: list[CompletionEvidence] = []
            for record in records:
                payload = record.to_dict()
                metadata = dict(payload.get("metadata") or {})
                metadata.setdefault(
                    "external_source_repository_tree",
                    str(payload.get("repository_tree") or ""),
                )
                payload.update(
                    {
                        "repository_id": repository_identity.repository_id,
                        "repository_tree": repository_identity.tree_id,
                        "tree_id": repository_identity.tree_id,
                        "metadata": metadata,
                    }
                )
                rebound_records.append(CompletionEvidence.from_dict(payload))
            records = rebound_records
        records = _apply_completion_evidence_source_policy(
            records,
            repository_tree=repository_identity.tree_id,
        )
        persisted_records[goal.goal_id] = records
        state = normalize_goal_state(goal.status)
        if state in {
            GoalState.ACTIVE,
            GoalState.REOPENED,
            GoalState.PROVISIONALLY_COMPLETE,
            GoalState.ANALYSIS_INCONCLUSIVE,
            # A status label is not completion authority. Re-evaluate every
            # verified goal so a manually asserted label without durable,
            # current evidence fails closed into the reopened lifecycle.
            GoalState.VERIFIED_COMPLETE,
        }:
            candidate_goals.append(goal)

    terms: list[str] = []
    for goal in candidate_goals:
        if goal.goal_id not in externally_governed_goal_ids:
            terms.extend(goal.required_evidence)
    discovered_evidence = evidence_index(
        repo_root,
        objective_path=objective_path,
        terms=terms,
        embedding_min_score=embedding_min_score,
        scan_exclude_paths=scan_exclude_paths,
    )

    updates: dict[str, dict[str, str]] = {}
    completed_goal_ids: list[str] = []
    provisional_goal_ids: list[str] = []
    reopened_goal_ids: list[str] = []
    analysis_inconclusive_goal_ids: list[str] = []
    blocked_goal_ids: list[str] = []
    completion_evidence: dict[str, dict[str, list[str]]] = {}
    validation_results: dict[str, dict[str, Any]] = {}
    decisions: dict[str, dict[str, Any]] = {}
    transitioned_at = now or utc_now()
    completion_boards: list[tuple[Path, str]] = []
    if todo_path is not None:
        completion_boards.append((todo_path, task_header_prefix))
    completion_boards.extend(todo_boards or ())
    open_goal_ids = open_implementation_goal_ids_from_todo_boards(
        completion_boards
    )
    referenced_goal_ids = referenced_goal_ids_from_todo_boards(completion_boards)
    hierarchy = goal_graph(goals)
    goals_by_id = {item.goal_id: item for item in goals if item.goal_id}
    effective_states = {
        goal_id: normalize_goal_state(goal.status)
        for goal_id, goal in goals_by_id.items()
    }
    external_final_states = {
        goal_id: effective_states[goal_id]
        for goal_id in externally_governed_goal_ids
        if goal_id in effective_states
    }

    def governing_external_ancestors(goal_id: str) -> set[str]:
        """Return every transitive external gate governing ``goal_id``."""

        goal = goals_by_id.get(goal_id)
        pending = list(goal.parent_goal_ids if goal is not None else ())
        seen: set[str] = set()
        external_ancestors: set[str] = set()
        while pending:
            parent_id = str(pending.pop(0))
            if not parent_id or parent_id in seen:
                continue
            seen.add(parent_id)
            if parent_id in externally_governed_goal_ids:
                external_ancestors.add(parent_id)
            parent_goal = goals_by_id.get(parent_id)
            if parent_goal is not None:
                pending.extend(parent_goal.parent_goal_ids)
        return external_ancestors

    candidate_by_id = {goal.goal_id: goal for goal in candidate_goals}
    evaluation_goal_ids: list[str] = []
    external_evaluation_goal_ids = [
        goal.goal_id
        for goal in sorted(
            candidate_goals,
            key=lambda item: (
                int(hierarchy.get("depths", {}).get(item.goal_id, 0)),
                item.goal_id,
            ),
        )
        if goal.goal_id in externally_governed_goal_ids
    ]
    # External authority gates must be decided before locally executable
    # descendants. All remaining goals retain the descendants-first order
    # needed for truthful parent aggregation.
    visited_goal_ids: set[str] = set(external_evaluation_goal_ids)
    visiting_goal_ids: set[str] = set()

    def visit_descendants_first(goal_id: str) -> None:
        if goal_id in visited_goal_ids:
            return
        if goal_id in visiting_goal_ids:
            return
        visiting_goal_ids.add(goal_id)
        for child_id in hierarchy.get("children", {}).get(goal_id, ()):
            normalized_child_id = str(child_id)
            if normalized_child_id in candidate_by_id:
                visit_descendants_first(normalized_child_id)
        visiting_goal_ids.remove(goal_id)
        visited_goal_ids.add(goal_id)
        if goal_id in candidate_by_id:
            evaluation_goal_ids.append(goal_id)

    for candidate in candidate_goals:
        if candidate.goal_id not in externally_governed_goal_ids:
            visit_descendants_first(candidate.goal_id)
    evaluation_goals = [
        candidate_by_id[goal_id]
        for goal_id in (*external_evaluation_goal_ids, *evaluation_goal_ids)
    ]

    def descendant_states(goal_id: str) -> list[dict[str, Any]]:
        pending = list(hierarchy.get("children", {}).get(goal_id, ()))
        seen: set[str] = set()
        descendants: list[dict[str, Any]] = []
        while pending:
            child_id = str(pending.pop(0))
            if not child_id or child_id in seen:
                continue
            seen.add(child_id)
            child = goals_by_id.get(child_id)
            if child is not None:
                state = effective_states[child_id].value
                descendants.append({
                    "goal_id": child_id,
                    "state": state,
                    "verified": state == GoalState.VERIFIED_COMPLETE.value,
                })
            pending.extend(hierarchy.get("children", {}).get(child_id, ()))
        return descendants

    for goal in evaluation_goals:
        if goal.goal_id not in externally_governed_goal_ids:
            external_ancestors = governing_external_ancestors(goal.goal_id)
            if any(
                external_final_states.get(goal_id)
                is not GoalState.VERIFIED_COMPLETE
                for goal_id in external_ancestors
            ):
                # The governing external decision either has not been
                # evaluated or failed current reconciliation.  Do not let
                # locally discoverable evidence advance the descendant in the
                # same cycle that reopens its authorization gate.
                continue
        current_state = normalize_goal_state(goal.status)
        records = persisted_records.get(goal.goal_id, [])
        source_evidence_complete = bool(goal.required_evidence) and all(
            discovered_evidence.get(term) for term in goal.required_evidence
        )
        tasks_complete = (
            goal.goal_id not in open_goal_ids
            and (
                not completion_boards
                or goal.goal_id in referenced_goal_ids
                or bool(records)
                or source_evidence_complete
                or goal.goal_id in externally_governed_goal_ids
                or is_legacy_completed_goal_state(
                    str(goal.fields.get("legacy_completion_state") or "")
                )
            )
        )
        if require_artifact_binding:
            supplied_gate = supplied_gate_records.get(goal.goal_id)
            gate_record = (
                dict(supplied_gate)
                if isinstance(supplied_gate, Mapping)
                else {}
            )
        else:
            gate_record = _goal_completion_gate_record(
                goal,
                supplied_gate_records,
            )
        criteria_text = str(
            goal.fields.get("acceptance_criteria")
            or goal.fields.get("acceptance")
            or ""
        ).strip()
        criteria: Sequence[str] | str = (
            (
                goal.required_evidence
                or ("external_operational_completion_receipt",)
            )
            if goal.goal_id in externally_governed_goal_ids
            else criteria_text or goal.required_evidence
        )

        if not tasks_complete:
            referenced_boards = referenced_goal_ids.get(goal.goal_id, [])
            validation_results[goal.goal_id] = {
                "attempted": False,
                "passed": False,
                "returncode": 1,
                "reason": (
                    "open_todo_tasks"
                    if goal.goal_id in open_goal_ids
                    else "no_producing_tasks"
                ),
                "todo_boards": open_goal_ids.get(
                    goal.goal_id,
                    referenced_boards,
                ),
            }
        elif current_state in {GoalState.PROVISIONALLY_COMPLETE, GoalState.VERIFIED_COMPLETE} and records:
            # Receipts supplied by another process remain provenance inputs,
            # but verification also requires the repository's current
            # validation command to pass at reconciliation time.
            validation_results[goal.goal_id] = run_goal_validation(
                repo_root=repo_root,
                goal=goal,
                repository_identity=repository_identity,
            )
            reconciled_records: list[CompletionEvidence] = []
            for record in records:
                payload = record.to_dict()
                metadata = dict(payload.get("metadata") or {})
                # The producer receipt and its provenance CID are immutable.
                # A current validation rerun is a separate, content-addressed
                # reconciliation receipt; overwriting the producer receipt
                # would leave provenance_cid referring to bytes no longer
                # present in the evidence record.
                local_validation = validation_results[goal.goal_id]
                if record.metadata.get("external_operational_completion") is True:
                    reconciliation_validation_receipt: Mapping[str, Any] = {
                        "schema": EXTERNAL_COMPLETION_EVIDENCE_SCHEMA
                        + "/local-validation-join",
                        "attempted": bool(
                            local_validation.get("attempted", False)
                        ),
                        "passed": bool(local_validation.get("passed", False)),
                        "status": (
                            "verified"
                            if local_validation.get("passed") is True
                            else "failed"
                        ),
                        "tree_id": str(local_validation.get("tree_id") or ""),
                        "receipt_cid": str(
                            local_validation.get("receipt_cid") or ""
                        ),
                        "external_validator_receipt_cid": str(
                            record.metadata.get("validator_receipt_cid") or ""
                        ),
                        "external_operational_receipt_cid": (
                            record.provenance_cid
                        ),
                    }
                else:
                    reconciliation_validation_receipt = local_validation
                metadata["reconciliation_validation_receipt"] = (
                    reconciliation_validation_receipt
                )
                payload["metadata"] = metadata
                payload["validation_passed"] = bool(
                    local_validation.get("passed", False)
                )
                reconciled_records.append(CompletionEvidence.from_dict(payload))
            records = reconciled_records

        decision = evaluate_goal_completion(
            current_state=current_state,
            acceptance_criteria=criteria,
            evidence=records,
            tasks_complete=tasks_complete,
            repository_tree=repository_identity.tree_id,
            repository_id=repository_identity.repository_id,
            objective_revision=objective_goal_completion_revision(goal),
            completion_binding=gate_record.get("binding"),
            require_artifact_binding=require_artifact_binding,
            now=now,
            freshness_seconds=evidence_freshness_seconds,
            coverage=gate_record.get("coverage"),
            analyzer_health=gate_record.get("analyzer_health"),
            exhaustion_quorum=gate_record.get("exhaustion_quorum"),
            child_goals=[
                *(
                    ()
                    if goal.goal_id in externally_governed_goal_ids
                    else descendant_states(goal.goal_id)
                ),
                *gate_record.get("child_goals", ()),
            ],
            required_child_goal_ids=gate_record.get(
                "required_child_goal_ids", ()
            ),
            analysis_result=gate_record.get("analysis_result"),
            analysis_inconclusive=bool(
                gate_record.get("analysis_inconclusive", False)
            ),
        )
        if goal.goal_id in externally_governed_goal_ids:
            external_final_states[goal.goal_id] = decision.state
        decisions[goal.goal_id] = decision.to_dict()
        if (
            external_completion is not None
            and goal.goal_id in current_authority_goal_ids
        ):
            decisions[goal.goal_id]["external_completion"] = {
                "authority_cid": external_completion.authority_cid,
                "results": list(
                    external_completion.results_for_goal(goal.goal_id)
                ),
            }
        elif goal.goal_id in externally_governed_goal_ids:
            missing_authority_reason = (
                "external_authority_not_supplied"
                if external_completion_authority is None
                else "external_authority_binding_missing"
            )
            missing_authority_results = [
                {
                    "schema": EXTERNAL_COMPLETION_VALIDATION_SCHEMA,
                    "goal_id": goal.goal_id,
                    "evidence_term": term,
                    "valid": False,
                    "reason_codes": [missing_authority_reason],
                    "receipt_cid": "",
                    "requirement_cid": "",
                }
                for term in (
                    goal.required_evidence
                    or ("external_operational_completion_receipt",)
                )
            ]
            decisions[goal.goal_id]["external_completion"] = {
                "authority_cid": str(
                    goal.fields.get("external_completion_authority_cid")
                    or ""
                ).strip(),
                "results": missing_authority_results,
            }
        effective_states[goal.goal_id] = decision.state
        diagnostics = decisions[goal.goal_id]["diagnostics"]
        if goal.goal_id in externally_governed_goal_ids:
            goal_evidence = {
                term: [
                    (
                        f"{record.provenance_cid} "
                        "(external-operational-receipt)"
                    )
                    for record in records
                    if record.acceptance_criterion == term
                ]
                for term in goal.required_evidence
            }
        else:
            goal_evidence = {
                term: list(discovered_evidence.get(term, []))
                for term in goal.required_evidence
            }
        if any(goal_evidence.values()):
            completion_evidence[goal.goal_id] = goal_evidence

        goal_updates = {
            "Goal completion schema version": str(GOAL_COMPLETION_SCHEMA_VERSION),
            "Completion confidence": str(diagnostics["confidence"]),
            "Uncovered criteria": json.dumps(diagnostics["uncovered_criteria"], separators=(",", ":")),
            "Stale evidence": json.dumps(diagnostics["stale_evidence"], sort_keys=True, separators=(",", ":")),
            "Analyzer health": json.dumps(diagnostics["analyzer_health"], sort_keys=True, separators=(",", ":")),
            "Exhaustion quorum": json.dumps(diagnostics["exhaustion_quorum"], sort_keys=True, separators=(",", ":")),
            "Reopen reasons": json.dumps(diagnostics["reopen_reasons"], separators=(",", ":")),
        }
        if (
            external_completion is not None
            and goal.goal_id in current_authority_goal_ids
        ):
            external_results = external_completion.results_for_goal(goal.goal_id)
            goal_updates.update(
                {
                    "External completion authority CID": (
                        external_completion.authority_cid
                    ),
                    "External completion receipt CIDs": json.dumps(
                        sorted(
                            result["receipt_cid"]
                            for result in external_results
                            if result.get("valid") is True
                            and result.get("receipt_cid")
                        ),
                        separators=(",", ":"),
                    ),
                    "External completion validation": json.dumps(
                        list(external_results),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
        elif goal.goal_id in externally_governed_goal_ids:
            goal_updates.update(
                {
                    "External completion receipt CIDs": "[]",
                    "External completion validation": json.dumps(
                        decisions[goal.goal_id]["external_completion"][
                            "results"
                        ],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
        if decision.state is current_state:
            updates[goal.goal_id] = goal_updates
            continue
        reason = "; ".join(decision.actionable_reasons) or "all completion evidence gates passed"
        goal_updates.update({
            "Status": decision.state.value,
            "State transitioned at": transitioned_at,
            "State transition reason": reason,
        })
        if records or goal.goal_id in externally_governed_goal_ids:
            goal_updates["Completion evidence records"] = json.dumps(
                [record.to_dict() for record in records],
                sort_keys=True,
                separators=(",", ":"),
            )
        if decision.state is GoalState.PROVISIONALLY_COMPLETE:
            provisional_goal_ids.append(goal.goal_id)
            goal_updates["Provisional at"] = transitioned_at
        elif decision.state is GoalState.VERIFIED_COMPLETE:
            completed_goal_ids.append(goal.goal_id)
            goal_updates["Completed at"] = transitioned_at
            goal_updates["Completion evidence"] = completion_evidence_summary(goal_evidence)
            goal_updates["Completion validation"] = str(
                validation_results.get(goal.goal_id, {}).get("receipt_cid", "")
            )
        elif decision.state is GoalState.REOPENED:
            reopened_goal_ids.append(goal.goal_id)
        elif decision.state is GoalState.ANALYSIS_INCONCLUSIVE:
            analysis_inconclusive_goal_ids.append(goal.goal_id)
        elif decision.state is GoalState.BLOCKED:
            blocked_goal_ids.append(goal.goal_id)
        updates[goal.goal_id] = goal_updates

    goal_position = {goal.goal_id: index for index, goal in enumerate(goals)}
    for transitioned_goal_ids in (
        completed_goal_ids,
        provisional_goal_ids,
        reopened_goal_ids,
        analysis_inconclusive_goal_ids,
        blocked_goal_ids,
    ):
        transitioned_goal_ids.sort(
            key=lambda goal_id: goal_position.get(goal_id, len(goal_position))
        )

    final_repository_identity = completion_tree_identity(
        repo_root,
        objective_path=objective_path,
        control_paths=completion_control_paths,
        scan_exclude_paths=scan_exclude_paths,
    )
    if final_repository_identity != repository_identity:
        raise RuntimeError(
            "repository tree changed while live goal validation commands were "
            "running; completion reconciliation was aborted"
        )

    if updates:
        rewritten = rewrite_goal_fields(text, updates)
        if rewritten != text:
            _atomic_rewrite(objective_path, rewritten)
        goals = parse_goal_heap(objective_path.read_text(encoding="utf-8"))

    state_counts = {state.value: 0 for state in GoalState}
    for goal in goals:
        state_counts[normalize_goal_state(goal.status).value] += 1

    return ObjectiveCompletionResult(
        objective_path=objective_path,
        completed_goal_ids=completed_goal_ids,
        active_goal_count=sum(1 for goal in goals if goal.is_schedulable),
        completed_goal_count=state_counts[GoalState.VERIFIED_COMPLETE.value],
        completion_evidence=completion_evidence,
        validation_results=validation_results,
        provisional_goal_ids=provisional_goal_ids,
        verified_goal_ids=list(completed_goal_ids),
        reopened_goal_ids=reopened_goal_ids,
        analysis_inconclusive_goal_ids=analysis_inconclusive_goal_ids,
        blocked_goal_ids=blocked_goal_ids,
        state_counts=state_counts,
        decisions=decisions,
        migration=migration_result.to_dict(),
        external_completion=(
            external_completion.to_dict()
            if external_completion is not None
            else {
                "schema": (
                    EXTERNAL_COMPLETION_VALIDATION_SCHEMA + "/authority"
                ),
                "authority_cid": "",
                "governed_goal_ids": sorted(
                    externally_governed_goal_ids
                ),
                "valid_receipt_cids": [],
                "source_inspection": {},
                "results": [
                    result
                    for goal_id in sorted(externally_governed_goal_ids)
                    for result in decisions.get(goal_id, {})
                    .get("external_completion", {})
                    .get("results", [])
                ],
            }
        ),
    )


def ensure_objective_tracking_document(
    objective_path: Path,
    *,
    ultimate_goal: str = DEFAULT_ULTIMATE_GOAL,
    root_evidence: Sequence[str] = DEFAULT_ROOT_EVIDENCE,
    root_goal_id: str | None = None,
    goal_prefix: str = DEFAULT_GOAL_PREFIX,
    document_title: str = DEFAULT_TRACKING_DOCUMENT_TITLE,
    root_goal_title: str = DEFAULT_ROOT_GOAL_TITLE,
) -> ObjectiveTrackingResult:
    """Create the objective tracking document if it does not exist."""

    if objective_path.exists():
        return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=[])

    root_goal_id = root_goal_id or f"{goal_prefix}000"
    objective_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            f"# {document_title}",
            "",
            "This document is the supervisor planning state. It is intentionally separate from markdown todo",
            "boards: todos represent executable work, while this heap represents the objective graph used to",
            "decide what work should exist.",
            "",
            render_goal_block(
                goal_id=root_goal_id,
                title=root_goal_title,
                fields={
                    "Status": "active",
                    "Goal completion schema version": str(GOAL_COMPLETION_SCHEMA_VERSION),
                    "Parent": "",
                    "Fib priority": str(fibonacci_priority(0)),
                    "Track": "ops",
                    "Priority": "P0",
                    "Bundle": "objective/ops/root",
                    "Goal": ultimate_goal,
                    "Evidence": ", ".join(root_evidence),
                    "Outputs": "ipfs_accelerate_py/agent_supervisor, docs",
                    "Validation": f"test -f {objective_path.as_posix()}",
                    "Refinement depth": "0",
                    "Conflict policy": "prefer bundle-local changes; invoke the LLM merge resolver for semantic conflicts",
                    "Gap task": "Refine this root objective into concrete child goals with code, tests, docs, and runtime evidence.",
                },
            ).rstrip(),
            "",
        ]
    )
    objective_path.write_text(text, encoding="utf-8")
    return ObjectiveTrackingResult(objective_path=objective_path, created=True, appended_goal_ids=[root_goal_id])


COMPONENT_SCAN_SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}
COMPONENT_MANIFEST_NAMES = {
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
    "Cargo.toml",
    "go.mod",
}
INTERFACE_DESCRIPTOR_SUFFIXES = {".idl", ".proto", ".thrift", ".graphql", ".graphqls"}


def _unique_paths(paths: Iterable[str]) -> list[str]:
    unique: list[str] = []
    for raw in paths:
        path = str(raw).strip().strip("/")
        if not path or "\0" in path or ".." in Path(path).parts:
            continue
        if path not in unique:
            unique.append(path)
    return unique


def discover_gitmodule_paths(repo_root: Path) -> list[str]:
    """Return repo-relative Git submodule paths declared in .gitmodules."""

    gitmodules = repo_root / ".gitmodules"
    if not gitmodules.exists():
        return []
    paths: list[str] = []
    for line in gitmodules.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped.startswith("path") or "=" not in stripped:
            continue
        _key, value = stripped.split("=", 1)
        path = value.strip()
        if path and path not in paths:
            paths.append(path)
    return paths


def discover_gitlink_paths(repo_root: Path) -> list[str]:
    """Return repo-relative gitlink paths from the Git index.

    Gitlinks are the authoritative submodule entries in the index.  Some repos
    can have stale or incomplete .gitmodules mappings, so interoperability
    planning must not rely on .gitmodules alone.
    """

    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--stage"],
            text=True,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []

    paths: list[str] = []
    for line in completed.stdout.splitlines():
        parts = line.split(None, 3)
        if len(parts) != 4 or parts[0] != "160000":
            continue
        path = parts[3].strip()
        if path and path not in paths:
            paths.append(path)
    return paths


def discover_submodule_paths(repo_root: Path) -> list[str]:
    """Return repo-relative component paths from .gitmodules and Git gitlinks."""

    return _unique_paths([*discover_gitmodule_paths(repo_root), *discover_gitlink_paths(repo_root)])


def _component_relative_path(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except (OSError, RuntimeError, ValueError):
        try:
            return path.relative_to(repo_root).as_posix()
        except ValueError:
            return path.as_posix()


def _component_scan_filename_allowed(filename: str) -> bool:
    if filename in COMPONENT_SCAN_SKIP_DIRS:
        return False
    return True


def _component_scan_dirname_allowed(dirname: str) -> bool:
    if dirname in COMPONENT_SCAN_SKIP_DIRS:
        return False
    return True


def _component_scan_path_usable(path: Path) -> bool:
    try:
        path.resolve()
    except (OSError, RuntimeError):
        return False
    return True


def _component_walk_error(_error: OSError) -> None:
    return None


def _component_path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except (OSError, RuntimeError):
        return False


def _component_path_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except (OSError, RuntimeError):
        return False


def _component_path_is_symlink(path: Path) -> bool:
    try:
        return path.is_symlink()
    except (OSError, RuntimeError):
        return False


def _component_path_safe_for_file_scan(path: Path) -> bool:
    if _component_path_is_symlink(path) and not _component_scan_path_usable(path):
        return False
    return True


def _component_path_safe_for_component(path: Path) -> bool:
    if not _component_path_exists(path):
        return False
    if not _component_path_is_dir(path):
        return False
    if _component_path_is_symlink(path) and not _component_scan_path_usable(path):
        return False
    return True


def _scan_component_metadata(repo_root: Path, component_path: str, *, max_files: int = 256) -> dict[str, list[str]]:
    root = repo_root / component_path
    metadata = {
        "manifests": [],
        "interface_descriptors": [],
        "mcp_descriptors": [],
        "python_import_roots": [],
    }
    if not _component_path_safe_for_component(root):
        return metadata

    import_roots: set[str] = set()
    scanned = 0
    for current_root, dirnames, filenames in os.walk(root, onerror=_component_walk_error):
        dirnames[:] = [
            name
            for name in dirnames
            if _component_scan_dirname_allowed(name) and _component_scan_path_usable(Path(current_root) / name)
        ]
        current = Path(current_root)
        try:
            depth = len(current.relative_to(root).parts)
        except ValueError:
            depth = 0
        if depth > 4:
            dirnames[:] = []
            continue
        for filename in sorted(filenames):
            if not _component_scan_filename_allowed(filename):
                continue
            path = current / filename
            if not _component_path_safe_for_file_scan(path):
                continue
            relative = _component_relative_path(repo_root, path)
            lowered = filename.lower()
            suffix = path.suffix.lower()
            if filename in COMPONENT_MANIFEST_NAMES:
                metadata["manifests"].append(relative)
            if suffix in INTERFACE_DESCRIPTOR_SUFFIXES or any(
                token in lowered for token in ("interface", "descriptor", "contract", "schema")
            ):
                metadata["interface_descriptors"].append(relative)
            if "mcp" in lowered or "orb" in lowered:
                metadata["mcp_descriptors"].append(relative)
            if suffix == ".py" and scanned < max_files:
                scanned += 1
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for match in re.finditer(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_\.]*)", text, flags=re.MULTILINE):
                    import_roots.add(match.group(1).split(".", 1)[0])

    metadata["manifests"] = sorted(dict.fromkeys(metadata["manifests"]))[:40]
    metadata["interface_descriptors"] = sorted(dict.fromkeys(metadata["interface_descriptors"]))[:80]
    metadata["mcp_descriptors"] = sorted(dict.fromkeys(metadata["mcp_descriptors"]))[:80]
    metadata["python_import_roots"] = sorted(import_roots)[:80]
    return metadata


def discover_repository_components(
    repo_root: Path,
    *,
    component_paths: Sequence[str] = (),
) -> list[RepositoryComponent]:
    """Discover repository components for interoperability planning.

    Callers may provide explicit component paths, but the function also uses
    .gitmodules and Git gitlinks so it works in repos with incomplete
    .gitmodules metadata.
    """

    gitmodule_path_list = discover_gitmodule_paths(repo_root)
    gitlink_path_list = discover_gitlink_paths(repo_root)
    gitmodule_paths = set(gitmodule_path_list)
    gitlink_paths = set(gitlink_path_list)
    configured_paths = set(_unique_paths(component_paths))
    paths = _unique_paths([*component_paths, *gitmodule_path_list, *gitlink_path_list])
    components: list[RepositoryComponent] = []
    for path in paths:
        sources: list[str] = []
        if path in configured_paths:
            sources.append("configured")
        if path in gitmodule_paths:
            sources.append("gitmodules")
        if path in gitlink_paths:
            sources.append("gitlink")
        metadata = _scan_component_metadata(repo_root, path)
        components.append(
            RepositoryComponent(
                path=path,
                sources=sources,
                exists=_component_path_exists(repo_root / path),
                is_gitlink=path in gitlink_paths,
                is_gitmodule=path in gitmodule_paths,
                manifests=metadata["manifests"],
                interface_descriptors=metadata["interface_descriptors"],
                mcp_descriptors=metadata["mcp_descriptors"],
                python_import_roots=metadata["python_import_roots"],
            )
        )
    return components


def interoperability_pairs(submodules: Sequence[str], *, focus: Sequence[str] = ()) -> list[tuple[str, str]]:
    paths = [path for path in dict.fromkeys(str(item).strip() for item in submodules) if path]
    focus_paths = [path for path in dict.fromkeys(str(item).strip() for item in focus) if path]
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    if focus_paths:
        for left in focus_paths:
            if left not in paths:
                continue
            for right in paths:
                if right == left:
                    continue
                pair_key = tuple(sorted((left, right)))
                if pair_key in seen:
                    continue
                pairs.append((left, right))
                seen.add(pair_key)
        return pairs
    for left_index, left in enumerate(paths):
        for right in paths[left_index + 1 :]:
            pairs.append((left, right))
    return pairs


def interoperability_pair_key(value: str | Sequence[str]) -> str:
    """Return a stable key for an unordered interoperability component pair."""

    if isinstance(value, str):
        terms = split_terms(value)
    else:
        terms = [str(item).strip() for item in value if str(item).strip()]
    canonical_terms = [
        key
        for term in terms
        for key in [canonical_interoperability_component(term)]
        if key
    ]
    return "\0".join(sorted(canonical_terms))


def deduplicate_interoperability_goals(objective_path: Path) -> list[str]:
    """Remove duplicate interoperability goal blocks from an objective heap."""

    if not objective_path.exists():
        return []

    text = objective_path.read_text(encoding="utf-8")
    goals = parse_goal_heap(text)
    goal_by_id = {goal.goal_id: goal for goal in goals}
    header_matches = list(re.finditer(r"^##\s+(\S+)\s+.+$", text, flags=re.MULTILINE))
    if not header_matches:
        return []

    preamble = text[: header_matches[0].start()].rstrip()
    blocks: list[tuple[str, str]] = []
    for index, match in enumerate(header_matches):
        start = match.start()
        end = header_matches[index + 1].start() if index + 1 < len(header_matches) else len(text)
        blocks.append((match.group(1), text[start:end].strip()))

    winner_by_pair: dict[str, str] = {}
    duplicate_goal_ids: set[str] = set()
    for goal_id, _block in blocks:
        goal = goal_by_id.get(goal_id)
        if goal is None:
            continue
        pair_key = interoperability_pair_key(str(goal.fields.get("interoperability_pair") or ""))
        if not pair_key:
            continue
        if pair_key in winner_by_pair:
            duplicate_goal_ids.add(goal_id)
            continue
        winner_by_pair[pair_key] = goal_id

    if not duplicate_goal_ids:
        return []

    retained_blocks = [block for goal_id, block in blocks if goal_id not in duplicate_goal_ids]
    rewritten = "\n\n".join(part for part in [preamble, *retained_blocks] if part).rstrip() + "\n"
    objective_path.write_text(rewritten, encoding="utf-8")
    return sorted(duplicate_goal_ids)


def _component_pair_metadata(
    left: RepositoryComponent | None,
    right: RepositoryComponent | None,
) -> dict[str, Any]:
    components = [component for component in (left, right) if component is not None]
    manifests = sorted({path for component in components for path in component.manifests})
    interface_descriptors = sorted({path for component in components for path in component.interface_descriptors})
    mcp_descriptors = sorted({path for component in components for path in component.mcp_descriptors})
    python_import_roots = sorted({root for component in components for root in component.python_import_roots})
    sources = sorted({source for component in components for source in component.sources})
    score = 1
    score += len(components)
    score += min(6, len(manifests) * 2)
    score += min(9, len(interface_descriptors) * 3)
    score += min(9, len(mcp_descriptors) * 3)
    score += min(6, len(python_import_roots))
    score += 2 if any(component.is_gitlink for component in components) else 0
    return {
        "score": score,
        "manifests": manifests[:12],
        "interface_descriptors": interface_descriptors[:16],
        "mcp_descriptors": mcp_descriptors[:16],
        "python_import_roots": python_import_roots[:24],
        "sources": sources,
    }


def _join_or_none(values: Sequence[str]) -> str:
    return ", ".join(str(value) for value in values if str(value))


def append_interoperability_goals(
    objective_path: Path,
    *,
    repo_root: Path,
    focus: Sequence[str] = (),
    component_paths: Sequence[str] = (),
    max_goals: int = 12,
    goal_prefix: str | None = None,
) -> ObjectiveTrackingResult:
    """Seed graph goals for cross-submodule integration and interoperability tests."""

    if not objective_path.exists() or max_goals <= 0:
        return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=[])

    components = discover_repository_components(repo_root, component_paths=component_paths)
    component_by_path = {component.path: component for component in components}
    submodules = [component.path for component in components]
    pairs = interoperability_pairs(submodules, focus=focus)
    if not pairs:
        return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=[])

    text = objective_path.read_text(encoding="utf-8")
    goals = parse_goal_heap(text)
    existing_pairs = {
        interoperability_pair_key(str(goal.fields.get("interoperability_pair") or ""))
        for goal in goals
        if str(goal.fields.get("interoperability_pair") or "").strip()
    }
    graph = goal_graph(goals)
    root_goal_id = sorted(graph.get("roots") or [goals[0].goal_id if goals else ""])[0]
    goal_prefix = goal_prefix or infer_goal_prefix(goals)
    next_id = next_goal_id(goals, prefix=goal_prefix)
    appended_blocks: list[str] = []
    appended_goal_ids: list[str] = []

    def allocate_goal_id() -> str:
        nonlocal next_id
        current = next_id
        number = int(current[len(goal_prefix) :]) + 1
        next_id = f"{goal_prefix}{number:03d}"
        return current

    for left, right in pairs:
        pair_key = interoperability_pair_key((left, right))
        if pair_key in existing_pairs:
            continue
        goal_id = allocate_goal_id()
        metadata = _component_pair_metadata(component_by_path.get(left), component_by_path.get(right))
        safe_left = safe_bundle_key(left).replace("-", "_")
        safe_right = safe_bundle_key(right).replace("-", "_")
        test_path = f"tests/integration/test_{safe_left}_{safe_right}_interop.py"
        doc_path = f"docs/integration/{safe_left}-{safe_right}.md"
        descriptor_terms = [
            *metadata["interface_descriptors"],
            *metadata["mcp_descriptors"],
        ]
        evidence_terms = [
            test_path,
            doc_path,
            f"interface contract {left} {right}",
            *descriptor_terms[:6],
        ]
        fields = {
            "Status": "active",
            "Parent": root_goal_id,
            "Fib priority": str(fibonacci_priority(1, len(appended_goal_ids))),
            "Track": "interoperability",
            "Priority": "P1",
            "Bundle": f"objective/interoperability/{safe_left}-{safe_right}",
            "Goal kind": "interoperability",
            "Interoperability pair": f"{left}, {right}",
            "Submodules": f"{left}, {right}",
            "Interoperability score": str(metadata["score"]),
            "Discovery sources": _join_or_none(metadata["sources"]),
            "Package manifests": _join_or_none(metadata["manifests"]),
            "Interface descriptors": _join_or_none(metadata["interface_descriptors"]),
            "MCP descriptors": _join_or_none(metadata["mcp_descriptors"]),
            "Python import roots": _join_or_none(metadata["python_import_roots"]),
            "Goal": (
                f"Prove `{left}` interoperates with `{right}` through importable contracts, "
                "interface descriptors, runtime handoff behavior, and integration tests."
            ),
            "Evidence": ", ".join(evidence_terms),
            "Outputs": ", ".join([test_path, doc_path, left, right, *descriptor_terms[:4]]),
            "Validation": "python -m pytest tests/integration -q",
            "Refinement depth": "1",
            "Embedding query": (
                f"{left} {right} interoperability integration test interface descriptor "
                f"{' '.join(metadata['python_import_roots'][:12])}"
            ),
            "AST query": ", ".join(
                [
                    left,
                    right,
                    "interface contract",
                    "integration test",
                    *metadata["python_import_roots"][:12],
                ]
            ),
            "Parallel lane": f"objective/interoperability/{safe_left}-{safe_right}",
            "Conflict policy": "keep pair-specific integration edits isolated; use the LLM merge resolver for conflicts",
            "Gap task": (
                f"Create one larger integration work item proving `{left}` and `{right}` can be used together, "
                "including a test, a contract note, and any adapter code needed by the objective."
            ),
        }
        appended_blocks.append(render_goal_block(goal_id=goal_id, title=f"Interoperate {left} with {right}", fields=fields))
        appended_goal_ids.append(goal_id)
        existing_pairs.add(pair_key)
        if len(appended_goal_ids) >= max_goals:
            break

    if appended_blocks:
        objective_path.write_text(text.rstrip() + "\n\n" + "\n\n".join(block.strip() for block in appended_blocks) + "\n", encoding="utf-8")
    return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=appended_goal_ids)


LAUNCH_READINESS_GOAL_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "key": "hallucinate-mcp-dashboard-capability-catalog",
        "title": "Hallucinate App MCP dashboard capability catalog",
        "track": "launch",
        "priority": "P0",
        "submodules": "hallucinate_app, swissknife, external/ipfs_accelerate, external/ipfs_datasets, external/ipfs_kit",
        "goal": (
            "Hallucinate App menus and dashboards expose ipfs_accelerate_py, ipfs_datasets_py, "
            "and ipfs_kit_py MCP server dashboards, daemon health, tools/list, and tools/call "
            "so Swissknife can test backend interoperability from the UI."
        ),
        "evidence": (
            "hallucinate_app menus, Hallucinate App MCP dashboard, dashboard capability catalog, "
            "daemon health, tools/list, tools/call, ipfs_accelerate_py MCP server, "
            "ipfs_datasets_py MCP server, ipfs_kit_py MCP server, Swissknife applications, "
            "Playwright MCP dashboard interoperability, launch Playwright validation gate"
        ),
        "outputs": (
            "hallucinate_app, swissknife, external/ipfs_accelerate, external/ipfs_datasets, "
            "external/ipfs_kit, hallucinate_app/test/e2e/mcp-feature-exposure.spec.ts, "
            "hallucinate_app/test/e2e/mcp-dashboard-interoperability.spec.ts"
        ),
        "validation": (
            "npm --prefix hallucinate_app run test:e2e -- "
            "mcp-feature-exposure.spec.ts mcp-dashboard-interoperability.spec.ts"
        ),
        "embedding_query": (
            "Hallucinate App MCP dashboard dashboard capability catalog tools/list tools/call "
            "ipfs_accelerate_py ipfs_datasets_py ipfs_kit_py Swissknife Playwright"
        ),
        "ast_query": (
            "hallucinate_app, swissknife, ipfs_accelerate_py, ipfs_datasets_py, ipfs_kit_py, "
            "tools/list, tools/call, daemon health, MCP dashboard"
        ),
        "gap_task": (
            "Create or repair the Hallucinate App dashboard/menu integration so it lists and "
            "calls each backend MCP server tool, then cover it with Playwright."
        ),
    },
    {
        "key": "swissknife-mcp-plus-plus-server-dashboard-interop",
        "title": "Swissknife MCP++ server dashboard interoperability",
        "track": "launch",
        "priority": "P0",
        "submodules": "swissknife, hallucinate_app, Mcp-Plus-Plus, external/ipfs_accelerate, external/ipfs_datasets, external/ipfs_kit",
        "goal": (
            "Swissknife applications consume MCP++ compatible control-plane contracts from "
            "Hallucinate App and the ipfs_accelerate_py, ipfs_datasets_py, and ipfs_kit_py MCP servers."
        ),
        "evidence": (
            "Swissknife applications, Mcp-Plus-Plus, MCP++ compatibility, MCP server dashboard, "
            "dashboard capability catalog, ipfs_accelerate_py, ipfs_datasets_py, ipfs_kit_py, "
            "tools/list, tools/call, control plane"
        ),
        "outputs": (
            "swissknife, Mcp-Plus-Plus, hallucinate_app, external/ipfs_accelerate, "
            "external/ipfs_datasets, external/ipfs_kit, swissknife/test/e2e/mcp-dashboard.spec.ts"
        ),
        "validation": "npm --prefix swissknife run test:e2e:mcp",
        "embedding_query": (
            "Swissknife MCP++ MCP server dashboard control plane tools/list tools/call "
            "ipfs_accelerate_py ipfs_datasets_py ipfs_kit_py"
        ),
        "ast_query": (
            "swissknife, Mcp-Plus-Plus, MCP++, MCP server, tools/list, tools/call, control plane"
        ),
        "gap_task": (
            "Implement the Swissknife-facing MCP++ dashboard contract and tests that prove "
            "Swissknife can enumerate and invoke backend services through the control plane."
        ),
    },
    {
        "key": "cross-device-virtual-desktop-offload-replay",
        "title": "Cross-device virtual desktop offload launch replay",
        "track": "launch",
        "priority": "P0",
        "submodules": "mobile, swissknife, hallucinate_app, external/ipfs_accelerate, external/ipfs_datasets, external/ipfs_kit",
        "goal": (
            "A phone-hosted Swissknife virtual desktop can discover a desktop peer, offload "
            "compute, route Hallucinate App mediation through IPFS/libp2p/MCP++, and produce "
            "a launch readiness receipt."
        ),
        "evidence": (
            "phone-hosted Swissknife virtual desktop, desktop peer offload, mobile phone, "
            "Hallucinate App mediation, IPFS, libp2p, MCP++, launch readiness receipt, "
            "cross-device e2e validation, Playwright launch replay"
        ),
        "outputs": (
            "mobile, swissknife, hallucinate_app, external/ipfs_accelerate, external/ipfs_datasets, "
            "external/ipfs_kit, tests/test_virtual_ai_os_launch_readiness_gate.py"
        ),
        "validation": (
            "PYTHONPATH=external/ipfs_accelerate:external/ipfs_datasets "
            "pytest tests/test_virtual_ai_os_launch_readiness_gate.py -q"
        ),
        "embedding_query": (
            "phone hosted Swissknife virtual desktop desktop peer offload mobile IPFS libp2p "
            "MCP++ launch readiness receipt Playwright"
        ),
        "ast_query": (
            "mobile, swissknife, hallucinate_app, desktop peer offload, launch readiness, Playwright"
        ),
        "gap_task": (
            "Build the deterministic cross-device launch replay and receipt path that proves "
            "phone-to-desktop offload works through the planned control plane."
        ),
    },
    {
        "key": "meta-glasses-control-plane-input-routing",
        "title": "Meta glasses control-plane input routing",
        "track": "launch",
        "priority": "P0",
        "submodules": "external/meta-wearables-dat-android, external/meta-wearables-dat-ios, mobile, swissknife, hallucinate_app",
        "goal": (
            "Meta glasses camera, microphone, headphones, Neural Band, and captouch inputs "
            "route through the mobile phone into Swissknife applications and the control plane "
            "using Bluetooth, Wi-Fi, IPFS/libp2p, and MCP++ compatible envelopes where possible."
        ),
        "evidence": (
            "Meta glasses interface, Meta Wearables DAT, camera, microphone, headphones, "
            "Neural Band, captouch, Bluetooth transport, Wi-Fi transport, mobile phone, "
            "Swissknife applications, IPFS, libp2p, MCP++, control plane"
        ),
        "outputs": (
            "external/meta-wearables-dat-android, external/meta-wearables-dat-ios, mobile, "
            "swissknife, hallucinate_app, tests/test_hallucinate_multimodal_control_todo_queue.py, "
            "tests/test_virtual_ai_os_launch_readiness_gate.py"
        ),
        "validation": (
            "PYTHONPATH=external/ipfs_accelerate:external/ipfs_datasets "
            "pytest tests/test_hallucinate_multimodal_control_todo_queue.py "
            "tests/test_virtual_ai_os_launch_readiness_gate.py -q"
        ),
        "embedding_query": (
            "Meta glasses camera microphone headphones Neural Band captouch Bluetooth Wi-Fi "
            "Swissknife control plane IPFS libp2p MCP++"
        ),
        "ast_query": (
            "Meta Wearables DAT, camera, microphone, headphones, Neural Band, captouch, "
            "Bluetooth, Wi-Fi, Swissknife, control plane"
        ),
        "gap_task": (
            "Research and codify the Meta glasses input contracts, mocks, and transport tests "
            "that let Swissknife applications consume those interaction methods."
        ),
    },
    {
        "key": "hallucinate-daemon-launch-orchestration",
        "title": "Hallucinate App daemon launch orchestration",
        "track": "launch",
        "priority": "P0",
        "submodules": "hallucinate_app, swissknife, external/ipfs_accelerate, external/ipfs_datasets, external/ipfs_kit",
        "goal": (
            "Hallucinate App launches and monitors the ipfs_accelerate_py, ipfs_datasets_py, "
            "and ipfs_kit_py MCP daemons, exposes their health in dashboards, and hands "
            "capability records to Swissknife."
        ),
        "evidence": (
            "Hallucinate App daemon health, daemon launcher, MCP server, MCP dashboard, "
            "ipfs_accelerate_py, ipfs_datasets_py, ipfs_kit_py, dashboard capability catalog, "
            "Swissknife applications, launch Playwright validation gate"
        ),
        "outputs": (
            "hallucinate_app, swissknife, external/ipfs_accelerate, external/ipfs_datasets, "
            "external/ipfs_kit, hallucinate_app/test/e2e/daemon-launch-health.spec.ts"
        ),
        "validation": (
            "PYTHONPATH=external/ipfs_accelerate:external/ipfs_datasets "
            "pytest tests/test_hallucinate_multimodal_control_todo_queue.py -q && "
            "(test ! -f hallucinate_app/package.json || npm --prefix hallucinate_app run test:e2e -- daemon-launch-health.spec.ts) && "
            "(test ! -f swissknife/package.json || npm --prefix swissknife run test:e2e:meta-glasses) && "
            "(test ! -f hallucinate_app/package.json || npm --prefix hallucinate_app run test:e2e -- multimodal-control-surface.spec.ts)"
        ),
        "embedding_query": (
            "Hallucinate App daemon launch health MCP server dashboard ipfs_accelerate_py "
            "ipfs_datasets_py ipfs_kit_py Swissknife"
        ),
        "ast_query": (
            "hallucinate_app, daemon health, MCP server, MCP dashboard, ipfs_accelerate_py, "
            "ipfs_datasets_py, ipfs_kit_py"
        ),
        "gap_task": (
            "Make Hallucinate App own daemon launch and health reporting for the backend MCP "
            "servers, with UI and integration tests that Swissknife can exercise."
        ),
    },
    {
        "key": "objective-heap-autosteer-validation-repair",
        "title": "Objective heap active steering and validation repair",
        "track": "launch",
        "priority": "P0",
        "submodules": "external/ipfs_accelerate, hallucinate_app, swissknife, mobile",
        "goal": (
            "The supervisor actively manages the objective heap and todo boards by adding, "
            "deprioritizing, and repairing goals, subgoals, tasks, and subtasks from validation "
            "results including Playwright launch replays."
        ),
        "evidence": (
            "objective heap, fibonacci priority, supervisor active management, failed validation "
            "repair, Playwright launch replay, HAO task board, MGW task board, VAI task board, "
            "production readiness, launch Playwright validation gate"
        ),
        "outputs": (
            "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor, "
            "tests/test_supervisor_objective_task_janitor.py, "
            "tests/test_reconciliation_guardrail_refresh.py"
        ),
        "validation": (
            "PYTHONPATH=external/ipfs_accelerate "
            "pytest tests/test_supervisor_objective_task_janitor.py "
            "tests/test_reconciliation_guardrail_refresh.py -q"
        ),
        "embedding_query": (
            "objective heap fibonacci priority supervisor active management failed validation "
            "repair Playwright VAI MGW HAO production readiness"
        ),
        "ast_query": (
            "objective heap, supervisor, validation repair, Playwright, VAI, MGW, HAO"
        ),
        "gap_task": (
            "Extend the supervisor loop so failed validation and stale idle lanes generate "
            "mission-aligned follow-up tasks and subgoals instead of generic reconciliation churn."
        ),
    },
)


def normalize_launch_key(value: str) -> str:
    return safe_bundle_key(value).strip("-")


def append_launch_readiness_goals(
    objective_path: Path,
    *,
    repo_root: Path,
    max_goals: int = 8,
    goal_prefix: str | None = None,
) -> ObjectiveTrackingResult:
    """Seed high-value launch-readiness goals for the Swissknife virtual desktop plan."""

    _ = repo_root
    if not objective_path.exists() or max_goals <= 0:
        return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=[])

    text = objective_path.read_text(encoding="utf-8")
    goals = parse_goal_heap(text)
    existing_keys = {
        normalize_launch_key(str(goal.fields.get("launch_key") or ""))
        for goal in goals
        if str(goal.fields.get("launch_key") or "").strip()
    }
    existing_bundles = {
        normalize_launch_key(str(goal.fields.get("bundle") or "").removeprefix("objective/launch/"))
        for goal in goals
        if str(goal.fields.get("bundle") or "").startswith("objective/launch/")
    }
    existing_keys.update(existing_bundles)
    graph = goal_graph(goals)
    root_goal_id = sorted(graph.get("roots") or [goals[0].goal_id if goals else ""])[0]
    goal_prefix = goal_prefix or infer_goal_prefix(goals)
    next_id = next_goal_id(goals, prefix=goal_prefix)
    appended_blocks: list[str] = []
    appended_goal_ids: list[str] = []

    def allocate_goal_id() -> str:
        nonlocal next_id
        current = next_id
        number = int(current[len(goal_prefix) :]) + 1
        next_id = f"{goal_prefix}{number:03d}"
        return current

    for template in LAUNCH_READINESS_GOAL_TEMPLATES:
        launch_key = normalize_launch_key(str(template["key"]))
        if launch_key in existing_keys:
            continue
        goal_id = allocate_goal_id()
        fields = {
            "Status": "active",
            "Parent": root_goal_id,
            "Fib priority": str(fibonacci_priority(1, len(appended_goal_ids))),
            "Track": str(template["track"]),
            "Priority": str(template["priority"]),
            "Bundle": f"objective/launch/{launch_key}",
            "Goal kind": "launch_readiness",
            "Launch key": launch_key,
            "Submodules": str(template["submodules"]),
            "Mission terms": str(template["evidence"]),
            "Goal": str(template["goal"]),
            "Evidence": str(template["evidence"]),
            "Outputs": str(template["outputs"]),
            "Validation": str(template["validation"]),
            "Refinement depth": "1",
            "Embedding query": str(template["embedding_query"]),
            "AST query": str(template["ast_query"]),
            "Parallel lane": f"objective/launch/{launch_key}",
            "Conflict policy": (
                "prefer launch-critical integration evidence; use the LLM merge resolver "
                "when dashboard, daemon, or mobile control-plane edits conflict"
            ),
            "Gap task": str(template["gap_task"]),
        }
        appended_blocks.append(render_goal_block(goal_id=goal_id, title=str(template["title"]), fields=fields))
        appended_goal_ids.append(goal_id)
        existing_keys.add(launch_key)
        if len(appended_goal_ids) >= max_goals:
            break

    if appended_blocks:
        objective_path.write_text(
            text.rstrip() + "\n\n" + "\n\n".join(block.strip() for block in appended_blocks) + "\n",
            encoding="utf-8",
        )
    return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=appended_goal_ids)


def existing_refinement_keys(goals: Sequence[ObjectiveGoal]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for goal in goals:
        for parent in goal.parent_goal_ids:
            for evidence in goal.required_evidence:
                keys.add((parent, normalize_evidence_key(evidence)))
    return keys


def normalize_evidence_key(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def refinement_title(parent_title: str, evidence: str) -> str:
    compact = " ".join(evidence.strip().split())
    if len(compact) > 72:
        compact = compact[:69].rstrip() + "..."
    return f"Prove {compact} for {parent_title}"


def refinement_fields(finding: ObjectiveFinding, *, evidence: str, depth: int, sibling_index: int) -> dict[str, str]:
    outputs = ", ".join(finding.outputs) if finding.outputs else "ipfs_accelerate_py/agent_supervisor, docs, tests"
    return {
        "Status": "active",
        "Parent": finding.goal_id,
        "Fib priority": str(fibonacci_priority(depth, sibling_index)),
        "Track": finding.track,
        "Priority": finding.priority,
        "Bundle": finding.bundle_key,
        "Goal": f"Create concrete implementation, tests, docs, or interface descriptors proving `{evidence}`.",
        "Evidence": evidence,
        "Outputs": outputs,
        "Validation": finding.validation,
        "Refinement depth": str(depth),
        "Embedding query": evidence,
        "AST query": evidence,
        "Parallel lane": finding.parallel_lane,
        "Conflict policy": finding.conflict_policy,
        "Gap task": f"Close the missing objective evidence `{evidence}` with a narrow, verifiable change.",
    }


def append_refinement_goals(
    objective_path: Path,
    findings: Sequence[ObjectiveFinding],
    *,
    max_children_per_finding: int = 3,
    max_depth: int = 4,
    goal_prefix: str | None = None,
) -> ObjectiveTrackingResult:
    """Append child goals for missing evidence terms that are still too broad."""

    if not objective_path.exists() or max_children_per_finding <= 0:
        return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=[])

    text = objective_path.read_text(encoding="utf-8")
    goals = parse_goal_heap(text)
    graph = goal_graph(goals)
    refinement_keys = existing_refinement_keys(goals)
    goals_by_id = {goal.goal_id: goal for goal in goals}
    appended_blocks: list[str] = []
    appended_goal_ids: list[str] = []
    goal_prefix = goal_prefix or infer_goal_prefix(goals)
    next_id = next_goal_id(goals, prefix=goal_prefix)

    def allocate_goal_id() -> str:
        nonlocal next_id
        current = next_id
        number = int(current[len(goal_prefix) :]) + 1
        next_id = f"{goal_prefix}{number:03d}"
        return current

    def ancestors(goal_id: str) -> set[str]:
        pending = list(goals_by_id.get(goal_id, ObjectiveGoal(goal_id, "", {})).parent_goal_ids)
        result: set[str] = set()
        while pending:
            ancestor_id = str(pending.pop())
            if not ancestor_id or ancestor_id in result:
                continue
            result.add(ancestor_id)
            ancestor = goals_by_id.get(ancestor_id)
            if ancestor is not None:
                pending.extend(ancestor.parent_goal_ids)
        return result

    def descendants(goal_id: str) -> set[str]:
        pending = list(graph.get("children", {}).get(goal_id, ()))
        result: set[str] = set()
        while pending:
            descendant_id = str(pending.pop())
            if not descendant_id or descendant_id in result:
                continue
            result.add(descendant_id)
            pending.extend(graph.get("children", {}).get(descendant_id, ()))
        return result

    def lineage_already_owns(goal_id: str, evidence_key: str) -> bool:
        related_ids = ancestors(goal_id) | descendants(goal_id)
        return any(
            evidence_key
            in {
                normalize_evidence_key(item)
                for item in goals_by_id[related_id].required_evidence
            }
            for related_id in related_ids
            if related_id in goals_by_id
        )

    for finding in findings:
        parent_depth = int(graph.get("depths", {}).get(finding.goal_id, finding.graph_depth))
        child_depth = parent_depth + 1
        if child_depth > max_depth:
            continue
        created_for_parent = 0
        for evidence in finding.missing_evidence:
            evidence_key = normalize_evidence_key(evidence)
            key = (finding.goal_id, evidence_key)
            if key in refinement_keys or lineage_already_owns(
                finding.goal_id,
                evidence_key,
            ):
                continue
            goal_id = allocate_goal_id()
            fields = refinement_fields(
                finding,
                evidence=evidence,
                depth=child_depth,
                sibling_index=created_for_parent,
            )
            appended_blocks.append(
                render_goal_block(
                    goal_id=goal_id,
                    title=refinement_title(finding.title, evidence),
                    fields=fields,
                )
            )
            appended_goal_ids.append(goal_id)
            refinement_keys.add(key)
            created_for_parent += 1
            if created_for_parent >= max_children_per_finding:
                break

    if appended_blocks:
        objective_path.write_text(text.rstrip() + "\n\n" + "\n\n".join(block.strip() for block in appended_blocks) + "\n", encoding="utf-8")
    return ObjectiveTrackingResult(objective_path=objective_path, created=False, appended_goal_ids=appended_goal_ids)


def thought_node_id(kind: str, *parts: str) -> str:
    seed = "\0".join(str(part) for part in parts)
    digest = sha1(seed.encode("utf-8")).hexdigest()[:12]
    safe_kind = re.sub(r"[^a-z0-9_]+", "_", kind.lower()).strip("_") or "thought"
    return f"{safe_kind}:{digest}"


def build_objective_thought_graph(goals: Sequence[ObjectiveGoal]) -> dict[str, Any]:
    """Build typed thought nodes from objective goals for integration planning."""

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []

    def add_node(node_id: str, **payload: Any) -> None:
        nodes.setdefault(node_id, {"id": node_id, **payload})

    def add_edge(source: str, target: str, kind: str) -> None:
        edges.append({"from": source, "to": target, "kind": kind})

    for goal in goals:
        goal_node = f"goal:{goal.goal_id}"
        outputs = split_terms(str(goal.fields.get("outputs") or ""))
        validation = str(goal.fields.get("validation") or "").strip()
        interop_pair = split_terms(str(goal.fields.get("interoperability_pair") or ""))
        submodules = split_terms(str(goal.fields.get("submodules") or ""))
        package_manifests = split_terms(str(goal.fields.get("package_manifests") or ""))
        interface_descriptors = split_terms(str(goal.fields.get("interface_descriptors") or ""))
        mcp_descriptors = split_terms(str(goal.fields.get("mcp_descriptors") or ""))
        add_node(
            goal_node,
            kind="goal",
            goal_id=goal.goal_id,
            title=goal.title,
            status=goal.status,
            track=str(goal.fields.get("track") or "ops"),
            priority=str(goal.fields.get("priority") or "P2"),
            thought=(
                str(goal.fields.get("goal") or goal.title)
                or "Decide what implementation evidence proves this objective."
            ),
        )
        for term in goal.required_evidence:
            evidence_node = thought_node_id("evidence", goal.goal_id, term)
            add_node(
                evidence_node,
                kind="evidence_requirement",
                goal_id=goal.goal_id,
                term=term,
                thought=f"Find or create repository evidence proving `{term}`.",
            )
            add_edge(goal_node, evidence_node, "requires_evidence")
        if validation:
            validation_node = thought_node_id("validation", goal.goal_id, validation)
            add_node(
                validation_node,
                kind="validation_strategy",
                goal_id=goal.goal_id,
                command=validation,
                thought="Run this validation before closing the goal.",
            )
            add_edge(goal_node, validation_node, "validated_by")
        for output in outputs:
            surface_node = thought_node_id("code_surface", goal.goal_id, output)
            add_node(
                surface_node,
                kind="code_surface",
                goal_id=goal.goal_id,
                path=output,
                thought=f"Inspect or modify `{output}` as part of this objective.",
            )
            add_edge(goal_node, surface_node, "touches_surface")
        if interop_pair or submodules:
            pair_values = interop_pair or submodules
            interop_node = thought_node_id("interoperability_pair", goal.goal_id, ",".join(pair_values))
            add_node(
                interop_node,
                kind="interoperability_pair",
                goal_id=goal.goal_id,
                submodules=pair_values,
                thought="Prove these components interoperate through contracts and tests.",
            )
            add_edge(goal_node, interop_node, "targets_interoperability")
            test_node = thought_node_id("test_strategy", goal.goal_id, ",".join(pair_values))
            add_node(
                test_node,
                kind="test_strategy",
                goal_id=goal.goal_id,
                submodules=pair_values,
                thought="Write or update integration tests that exercise the shared runtime boundary.",
            )
            add_edge(interop_node, test_node, "needs_test_strategy")
            for manifest in package_manifests:
                manifest_node = thought_node_id("package_manifest", goal.goal_id, manifest)
                add_node(
                    manifest_node,
                    kind="package_manifest",
                    goal_id=goal.goal_id,
                    path=manifest,
                    thought=f"Use `{manifest}` to identify package entrypoints and dependency surfaces.",
                )
                add_edge(interop_node, manifest_node, "uses_package_manifest")
            for descriptor in interface_descriptors:
                descriptor_node = thought_node_id("interface_descriptor", goal.goal_id, descriptor)
                add_node(
                    descriptor_node,
                    kind="interface_descriptor",
                    goal_id=goal.goal_id,
                    path=descriptor,
                    thought=f"Map `{descriptor}` into the interoperability contract.",
                )
                add_edge(interop_node, descriptor_node, "uses_interface_descriptor")
            for descriptor in mcp_descriptors:
                mcp_node = thought_node_id("mcp_descriptor", goal.goal_id, descriptor)
                add_node(
                    mcp_node,
                    kind="mcp_descriptor",
                    goal_id=goal.goal_id,
                    path=descriptor,
                    thought=f"Use `{descriptor}` as an MCP or ORB capability boundary.",
                )
                add_edge(interop_node, mcp_node, "uses_mcp_descriptor")

    return {
        "schema": "ipfs_accelerate_py.agent_supervisor.objective_thought_graph",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": [nodes[node_id] for node_id in sorted(nodes)],
        "edges": edges,
    }


def write_objective_graph_artifact(
    *,
    objective_path: Path,
    graph_path: Path,
) -> dict[str, Any]:
    """Write a JSON graph artifact for the current objective heap."""

    goals = parse_goal_heap(objective_path.read_text(encoding="utf-8")) if objective_path.exists() else []
    graph = goal_graph(goals)
    payload = {
        "schema": "ipfs_accelerate_py.agent_supervisor.objective_graph",
        "generated_at": utc_now(),
        "objective_path": str(objective_path),
        "goal_count": len(goals),
        "active_goal_count": sum(1 for goal in goals if goal.is_schedulable),
        "completed_goal_count": sum(
            1
            for goal in goals
            if goal.lifecycle_state is GoalState.VERIFIED_COMPLETE
        ),
        "heap_schedule": [record.to_dict() for record in objective_heap_schedule(goals)],
        "thought_graph": build_objective_thought_graph(goals),
        "goals": [
            {
                "goal_id": goal.goal_id,
                "title": goal.title,
                "status": goal.status,
                "fib_priority": goal.priority[0],
                "parents": goal.parent_goal_ids,
                "evidence": goal.required_evidence,
                "track": goal.fields.get("track", "ops"),
                "bundle": goal.fields.get("bundle", ""),
                "refinement_depth": goal.fields.get("refinement_depth", str(graph["depths"].get(goal.goal_id, 0))),
            }
            for goal in sorted(goals, key=lambda item: item.priority)
        ],
        "graph": graph,
    }
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_root_evidence(values: Iterable[str]) -> list[str]:
    terms: list[str] = []
    for value in values:
        terms.extend(split_terms(value))
    return terms or list(DEFAULT_ROOT_EVIDENCE)
