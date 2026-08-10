"""DCR-083: derive all statuses from one content-addressed authority projection.

Interfaces
----------
* ``RepairAuthorityProjection@1`` — single projection of task/goal readiness.
* ``GoalCompletion@1`` — derived goal status (never an independent authority).

Predicted symbols: :class:`RepairAuthorityProjection`, :func:`derive_task_status`,
:func:`derive_goal_status`.

Normative rules (fail-closed)
-----------------------------
* Board, objective, baseline, stage, and readiness are *projections* of the
  same sealed evidence graph — never independent authorities.
* Contradictory completion (board complete without admission/validation
  receipts, or receipts without board) reopens the task/goal.
* Runtime model calls remain 0.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    ContractValidationError,
    content_identity,
)


REPAIR_AUTHORITY_PROJECTION_INTERFACE: Final[str] = "RepairAuthorityProjection@1"
GOAL_COMPLETION_INTERFACE: Final[str] = "GoalCompletion@1"
DCR_AUTHORITY_PROJECTION_EVIDENCE: Final[str] = "dcr/authority-projection@1"
DCR_AUTHORITY_PROJECTION_VERSION: Final[int] = 1
AUTHORITY_PROJECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dcr-authority-projection@1"
)
DEFAULT_AUTHORITY_PROJECTION_PATH: Final[str] = (
    "data/agent_supervisor/deterministic_contract_repair/authority-projection.json"
)

_TASK_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "todo",
        "ready",
        "blocked",
        "in_progress",
        "completed",
        "reopened",
        "deferred",
        "abstained",
    }
)
_GOAL_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "open",
        "ready",
        "in_progress",
        "completed",
        "reopened",
        "blocked",
    }
)


class RepairAuthorityProjectionError(ContractValidationError):
    """Malformed authority projection input or contradiction."""


class TaskAuthorityStatus(str, Enum):  # noqa: UP042
    TODO = "todo"
    READY = "ready"
    BLOCKED = "blocked"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REOPENED = "reopened"
    DEFERRED = "deferred"
    ABSTAINED = "abstained"


class GoalAuthorityStatus(str, Enum):  # noqa: UP042
    OPEN = "open"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REOPENED = "reopened"
    BLOCKED = "blocked"


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RepairAuthorityProjectionError(f"{name} must be a non-empty string")
    return value.strip()


def _bool(value: Any) -> bool:
    return value is True


def derive_task_status(
    task: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any] | None = None,
) -> str:
    """Derive one task status from board row + sealed evidence.

    Contradictions reopen.  Explicit ``reopen`` / missing required receipts
    beat a board ``completed`` claim.
    """

    if not isinstance(task, Mapping):
        raise RepairAuthorityProjectionError("task must be a mapping")
    evidence = dict(evidence or {})
    board_status = str(task.get("status") or task.get("board_status") or "todo").lower()
    task_id = str(task.get("task_id") or task.get("id") or "")

    # Evidence claims
    admission_ok = _bool(evidence.get("admission_ok") or evidence.get("admitted"))
    validation_ok = _bool(
        evidence.get("validation_ok")
        or evidence.get("validated")
        or evidence.get("proved_valid")
    )
    publication_ok = _bool(evidence.get("publication_ok") or evidence.get("published"))
    in_progress = _bool(evidence.get("in_progress") or evidence.get("active"))
    deferred = _bool(evidence.get("defer_capability") or evidence.get("deferred"))
    abstained = _bool(evidence.get("abstain_review") or evidence.get("abstained"))
    blocked = _bool(evidence.get("blocked")) or bool(task.get("blocked_by"))
    force_reopen = _bool(evidence.get("reopen") or evidence.get("contradiction"))

    required = set(evidence.get("required_receipts") or ())
    present = set(evidence.get("present_receipts") or ())
    missing_required = sorted(required - present)

    # Contradiction: board completed without required evidence.
    board_completed = board_status in {"completed", "done"}
    if force_reopen:
        return TaskAuthorityStatus.REOPENED.value
    if board_completed and missing_required:
        return TaskAuthorityStatus.REOPENED.value
    if board_completed and not (admission_ok or validation_ok or publication_ok):
        # Board-only completion is not authoritative under DCR-083.
        if evidence.get("require_evidence_for_completion", True):
            return TaskAuthorityStatus.REOPENED.value

    # Evidence-complete completion even if board lags — still projection, but
    # only when required receipts are present and board is not contradictory.
    if validation_ok and admission_ok and not missing_required and not force_reopen:
        if board_status in {"completed", "done", "ready", "todo", "in_progress"}:
            # If board still says todo/ready but evidence is complete, prefer
            # completed only when board also claims completed OR publication ok.
            if board_completed or publication_ok:
                return TaskAuthorityStatus.COMPLETED.value

    if board_completed and (admission_ok or validation_ok or publication_ok):
        return TaskAuthorityStatus.COMPLETED.value

    if deferred:
        return TaskAuthorityStatus.DEFERRED.value
    if abstained:
        return TaskAuthorityStatus.ABSTAINED.value
    if in_progress:
        return TaskAuthorityStatus.IN_PROGRESS.value
    if blocked or board_status in {"blocked", "waiting"}:
        return TaskAuthorityStatus.BLOCKED.value
    if board_status in {"ready", "open"}:
        return TaskAuthorityStatus.READY.value
    if board_status in {"todo", "pending"}:
        # Ready when dependencies satisfied projection says so.
        if _bool(task.get("dependencies_satisfied") or evidence.get("dependencies_satisfied")):
            return TaskAuthorityStatus.READY.value
        return TaskAuthorityStatus.TODO.value
    if board_status in _TASK_STATUSES:
        return board_status
    raise RepairAuthorityProjectionError(
        f"cannot derive task status for {task_id or '<unknown>'}"
    )


def derive_goal_status(
    goal: Mapping[str, Any],
    *,
    task_statuses: Mapping[str, str] | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> str:
    """Derive goal status from member task projections + goal evidence."""

    if not isinstance(goal, Mapping):
        raise RepairAuthorityProjectionError("goal must be a mapping")
    task_statuses = {
        str(k): str(v).lower() for k, v in dict(task_statuses or {}).items()
    }
    evidence = dict(evidence or {})
    member_ids = [
        str(item)
        for item in (
            goal.get("task_ids")
            or goal.get("members")
            or goal.get("child_task_ids")
            or ()
        )
    ]
    board_status = str(goal.get("status") or goal.get("board_status") or "open").lower()

    if _bool(evidence.get("reopen") or evidence.get("contradiction")):
        return GoalAuthorityStatus.REOPENED.value

    statuses = [task_statuses.get(tid, "todo") for tid in member_ids] if member_ids else []
    if statuses:
        if any(status == "reopened" for status in statuses):
            return GoalAuthorityStatus.REOPENED.value
        if all(status == "completed" for status in statuses):
            # Goal board claiming open while all members complete is not a
            # contradiction that reopens; projection reports completed.
            return GoalAuthorityStatus.COMPLETED.value
        if any(status == "in_progress" for status in statuses):
            return GoalAuthorityStatus.IN_PROGRESS.value
        if any(status == "blocked" for status in statuses):
            return GoalAuthorityStatus.BLOCKED.value
        if any(status == "ready" for status in statuses):
            return GoalAuthorityStatus.READY.value
        return GoalAuthorityStatus.OPEN.value

    if board_status in {"completed", "done"}:
        if evidence.get("require_member_completion", True) and member_ids:
            return GoalAuthorityStatus.REOPENED.value
        return GoalAuthorityStatus.COMPLETED.value
    if board_status in _GOAL_STATUSES:
        return board_status
    return GoalAuthorityStatus.OPEN.value


@dataclass(frozen=True)
class RepairAuthorityProjection:
    """One sealed projection of task/goal statuses (RepairAuthorityProjection@1)."""

    projection_id: str
    task_statuses: Mapping[str, str] = field(default_factory=dict)
    goal_statuses: Mapping[str, str] = field(default_factory=dict)
    reopened_task_ids: tuple[str, ...] = ()
    reopened_goal_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    runtime_model_calls: int = 0
    independent_board_authority: bool = False
    SCHEMA: ClassVar[str] = AUTHORITY_PROJECTION_SCHEMA
    INTERFACE: ClassVar[str] = REPAIR_AUTHORITY_PROJECTION_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "projection_id", _text(self.projection_id, "projection_id")
        )
        tasks = {
            str(k): str(v).lower() for k, v in dict(self.task_statuses).items()
        }
        goals = {
            str(k): str(v).lower() for k, v in dict(self.goal_statuses).items()
        }
        for value in tasks.values():
            if value not in _TASK_STATUSES:
                raise RepairAuthorityProjectionError(f"invalid task status {value!r}")
        for value in goals.values():
            if value not in _GOAL_STATUSES:
                raise RepairAuthorityProjectionError(f"invalid goal status {value!r}")
        object.__setattr__(self, "task_statuses", tasks)
        object.__setattr__(self, "goal_statuses", goals)
        object.__setattr__(
            self,
            "reopened_task_ids",
            tuple(str(item) for item in self.reopened_task_ids),
        )
        object.__setattr__(
            self,
            "reopened_goal_ids",
            tuple(str(item) for item in self.reopened_goal_ids),
        )
        object.__setattr__(
            self, "reason_codes", tuple(str(item) for item in self.reason_codes)
        )
        object.__setattr__(self, "runtime_model_calls", 0)
        object.__setattr__(self, "independent_board_authority", False)

    @property
    def content_id(self) -> str:
        return content_identity(self._identity_payload())

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "projection_id": self.projection_id,
            "task_statuses": dict(sorted(self.task_statuses.items())),
            "goal_statuses": dict(sorted(self.goal_statuses.items())),
            "reopened_task_ids": list(self.reopened_task_ids),
            "reopened_goal_ids": list(self.reopened_goal_ids),
            "reason_codes": list(self.reason_codes),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload.update(
            {
                "runtime_model_calls": 0,
                "independent_board_authority": False,
                "goal_completion_interface": GOAL_COMPLETION_INTERFACE,
                "content_id": content_identity(payload),
            }
        )
        return payload


def build_repair_authority_projection(
    *,
    tasks: Sequence[Mapping[str, Any]] = (),
    goals: Sequence[Mapping[str, Any]] = (),
    evidence_by_task: Mapping[str, Mapping[str, Any]] | None = None,
    evidence_by_goal: Mapping[str, Mapping[str, Any]] | None = None,
    projection_id: str | None = None,
) -> RepairAuthorityProjection:
    """Build one content-addressed authority projection for tasks and goals."""

    evidence_by_task = dict(evidence_by_task or {})
    evidence_by_goal = dict(evidence_by_goal or {})
    task_statuses: dict[str, str] = {}
    reopened_tasks: list[str] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            raise RepairAuthorityProjectionError("task rows must be mappings")
        task_id = _text(task.get("task_id") or task.get("id"), "task_id")
        status = derive_task_status(
            task, evidence=evidence_by_task.get(task_id) or task.get("evidence")
        )
        task_statuses[task_id] = status
        if status == TaskAuthorityStatus.REOPENED.value:
            reopened_tasks.append(task_id)

    goal_statuses: dict[str, str] = {}
    reopened_goals: list[str] = []
    for goal in goals:
        if not isinstance(goal, Mapping):
            raise RepairAuthorityProjectionError("goal rows must be mappings")
        goal_id = _text(goal.get("goal_id") or goal.get("id"), "goal_id")
        status = derive_goal_status(
            goal,
            task_statuses=task_statuses,
            evidence=evidence_by_goal.get(goal_id) or goal.get("evidence"),
        )
        goal_statuses[goal_id] = status
        if status == GoalAuthorityStatus.REOPENED.value:
            reopened_goals.append(goal_id)

    reasons = ["single_authority_projection", "board_not_independent"]
    if reopened_tasks or reopened_goals:
        reasons.append("contradiction_reopened")

    pid = projection_id or content_identity(
        {
            "tasks": dict(sorted(task_statuses.items())),
            "goals": dict(sorted(goal_statuses.items())),
        }
    )
    return RepairAuthorityProjection(
        projection_id=pid,
        task_statuses=task_statuses,
        goal_statuses=goal_statuses,
        reopened_task_ids=tuple(sorted(set(reopened_tasks))),
        reopened_goal_ids=tuple(sorted(set(reopened_goals))),
        reason_codes=tuple(reasons),
    )


def materialize_authority_projection(
    *,
    tasks: Sequence[Mapping[str, Any]] | None = None,
    goals: Sequence[Mapping[str, Any]] | None = None,
    destination: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize authority-projection.json evidence for DCR-083."""

    if tasks is None:
        tasks = (
            {
                "task_id": "DCR-080",
                "status": "completed",
                "evidence": {
                    "admission_ok": True,
                    "validation_ok": True,
                    "publication_ok": True,
                },
            },
            {
                "task_id": "DCR-081",
                "status": "completed",
                "evidence": {
                    "require_evidence_for_completion": True,
                    "required_receipts": ("admission",),
                    "present_receipts": (),
                },
            },
            {"task_id": "DCR-083", "status": "ready", "dependencies_satisfied": True},
        )
    if goals is None:
        goals = (
            {
                "goal_id": "DCR-G090",
                "status": "open",
                "task_ids": ("DCR-080", "DCR-081", "DCR-082", "DCR-083", "DCR-084"),
            },
        )
    projection = build_repair_authority_projection(tasks=tasks, goals=goals)
    payload = {
        "schema": AUTHORITY_PROJECTION_SCHEMA,
        "interface": REPAIR_AUTHORITY_PROJECTION_INTERFACE,
        "goal_completion_interface": GOAL_COMPLETION_INTERFACE,
        "evidence_id": DCR_AUTHORITY_PROJECTION_EVIDENCE,
        "version": DCR_AUTHORITY_PROJECTION_VERSION,
        "projection": projection.to_dict(),
        "runtime_model_calls": 0,
        "independent_board_authority": False,
    }
    base = Path(repo_root).resolve() if repo_root is not None else Path.cwd()
    path = (
        Path(destination)
        if destination is not None
        else base.joinpath(*PurePosixPath(DEFAULT_AUTHORITY_PROJECTION_PATH).parts)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "DCR_AUTHORITY_PROJECTION_EVIDENCE",
    "DCR_AUTHORITY_PROJECTION_VERSION",
    "DEFAULT_AUTHORITY_PROJECTION_PATH",
    "GOAL_COMPLETION_INTERFACE",
    "REPAIR_AUTHORITY_PROJECTION_INTERFACE",
    "GoalAuthorityStatus",
    "RepairAuthorityProjection",
    "RepairAuthorityProjectionError",
    "TaskAuthorityStatus",
    "build_repair_authority_projection",
    "derive_goal_status",
    "derive_task_status",
    "materialize_authority_projection",
]
