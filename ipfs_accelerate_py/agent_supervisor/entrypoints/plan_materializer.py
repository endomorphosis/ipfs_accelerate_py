"""Canonical, fail-closed prompt-plan materialization entrypoint.

This module intentionally owns no second planner, parser, or task database.
It joins the bounded prompt planner, formal admission gate, and the existing
Markdown/DuckDB projections.  Its receipts contain identities and counts only:
the prompt body (which is process-local on :class:`PromptSource`) is never
copied into a projection or materialization receipt.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ..prompt.prompt_goal_planner import (
    PromptGoalPlannerConfig,
    PromptGoalPlanningResult,
    generate_prompt_goal_graph,
)
from ..prompt.prompt_plan_admission import (
    PromptPlanAdmissionPolicy,
    PromptPlanAdmissionResult,
    admit_prompt_plan,
)
from ..prompt.prompt_workflow import (
    DirectoryScanReceipt,
    OutputMode,
    PromptGoalGraph,
    PromptTaskRecord,
    PromptWorkflowRequest,
)
from ..planning.formal_plan_compiler import prompt_goal_graph_to_formal_input
from ..task_sources.duckdb_task_source import DuckDBTaskSource
from ..task_sources.markdown_task_source import MarkdownMaterializationResult, MarkdownTaskSource


PLAN_MATERIALIZATION_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/prompt-program-materialization-receipt@1"
)


class PlanMaterializationError(RuntimeError):
    """A plan could not cross the materialization boundary."""


class PlanMaterializationAdmissionError(PlanMaterializationError):
    """An unadmitted graph was offered to a durable projection."""


class PlanMaterializationConflict(PlanMaterializationError):
    """A projection already belongs to a different canonical revision."""


def _cid(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "plan-materializer:sha256:" + hashlib.sha256(raw).hexdigest()


def _safe_receipt_text(value: Any, prompt_body: bytes | None) -> str:
    """Serialize receipt data and deny accidental persistence of raw input."""
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if prompt_body:
        try:
            raw = prompt_body.decode("utf-8")
        except UnicodeDecodeError:
            raw = ""
        if raw and raw in rendered:
            raise PlanMaterializationError("raw prompt body reached a durable projection")
    return rendered


@dataclass(frozen=True)
class CanonicalGoalGraph:
    """Supervisor-readable immutable goal hierarchy."""

    plan_root_cid: str
    root_goal_cid: str
    goal_cids: tuple[str, ...]
    parent_by_goal: Mapping[str, str]
    dependency_goal_cids: Mapping[str, tuple[str, ...]]
    evidence_by_goal: Mapping[str, tuple[str, ...]]
    producer_task_cids: Mapping[str, tuple[str, ...]]

    @classmethod
    def from_graph(cls, graph: PromptGoalGraph, *, plan_root_cid: str | None = None) -> "CanonicalGoalGraph":
        producers: dict[str, list[str]] = {goal.goal_cid: [] for goal in graph.goals}
        for task in graph.tasks:
            producers[task.goal_cid].append(task.task_cid)
        missing = [goal.goal_key for goal in graph.goals if not goal.evidence_cids or not producers[goal.goal_cid]]
        if missing:
            raise PlanMaterializationError("every goal needs evidence and a task producer: " + ", ".join(sorted(missing)))
        return cls(
            plan_root_cid=plan_root_cid or graph.plan_root_cid,
            root_goal_cid=graph.root_goal.goal_cid,
            goal_cids=tuple(sorted(goal.goal_cid for goal in graph.goals)),
            parent_by_goal={goal.goal_cid: goal.parent_goal_cid for goal in graph.goals},
            dependency_goal_cids={goal.goal_cid: goal.dependency_goal_cids for goal in graph.goals},
            evidence_by_goal={goal.goal_cid: goal.evidence_cids for goal in graph.goals},
            producer_task_cids={key: tuple(sorted(value)) for key, value in producers.items()},
        )


@dataclass(frozen=True)
class CanonicalTaskGraph:
    """The scheduler-facing task DAG and mandatory task metadata."""

    plan_root_cid: str
    task_cids: tuple[str, ...]
    dependency_task_cids: Mapping[str, tuple[str, ...]]
    goal_by_task: Mapping[str, str]
    scheduler_fields: Mapping[str, Mapping[str, Any]]

    @classmethod
    def from_graph(cls, graph: PromptGoalGraph, *, plan_root_cid: str | None = None) -> "CanonicalTaskGraph":
        required = ("priority", "track", "parallel_lane", "resource_class")
        fields: dict[str, Mapping[str, Any]] = {}
        for task in graph.tasks:
            values = {name: getattr(task, name) for name in required}
            if any(not value for value in values.values()):
                raise PlanMaterializationError("task is missing scheduler metadata: " + task.task_key)
            if not task.outputs or not task.validations or not task.acceptance or not task.evidence_cids:
                raise PlanMaterializationError("task is missing required outputs, validations, acceptance, or evidence: " + task.task_key)
            fields[task.task_cid] = values
        return cls(
            plan_root_cid=plan_root_cid or graph.plan_root_cid,
            task_cids=tuple(sorted(task.task_cid for task in graph.tasks)),
            dependency_task_cids={task.task_cid: task.dependency_task_cids for task in graph.tasks},
            goal_by_task={task.task_cid: task.goal_cid for task in graph.tasks},
            scheduler_fields=fields,
        )


@dataclass(frozen=True)
class TaskSourceProjectionReceipt:
    kind: str
    plan_root_cid: str
    projection_cid: str
    revision: int
    task_count: int
    goal_count: int
    no_op: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "plan_root_cid": self.plan_root_cid, "projection_cid": self.projection_cid, "revision": self.revision, "task_count": self.task_count, "goal_count": self.goal_count, "no_op": self.no_op}


@dataclass(frozen=True)
class ProgramRevisionCAS:
    """Identity-based revision fence; task sources enforce the durable CAS."""

    plan_root_cid: str
    revision: int
    idempotency_key: str

    def __post_init__(self) -> None:
        if not self.plan_root_cid or self.revision < 1 or not self.idempotency_key:
            raise ValueError("revision CAS requires plan root, positive revision, and idempotency key")


@dataclass(frozen=True)
class PromptProgramMaterialization:
    receipt_id: str
    request_cid: str
    scan_cid: str
    plan_root_cid: str
    revision_cas: ProgramRevisionCAS
    goals: CanonicalGoalGraph
    tasks: CanonicalTaskGraph
    admission: PromptPlanAdmissionResult
    projections: tuple[TaskSourceProjectionReceipt, ...]
    planning: PromptGoalPlanningResult

    def to_dict(self) -> dict[str, Any]:
        # Deliberately do not serialize graph/planner provider text or request.
        return {
            "schema": PLAN_MATERIALIZATION_RECEIPT_SCHEMA,
            "receipt_id": self.receipt_id,
            "request_cid": self.request_cid,
            "scan_cid": self.scan_cid,
            "plan_root_cid": self.plan_root_cid,
            "revision": self.revision_cas.revision,
            "idempotency_key": self.revision_cas.idempotency_key,
            "admission_receipt_id": self.admission.receipt.receipt_id,
            "projections": [item.to_dict() for item in self.projections],
        }


class PromptProgramMaterializer:
    """Compose prompt planning, admission, and canonical task-source writes."""

    def __init__(self, *, planner_config: PromptGoalPlannerConfig | None = None, admission_policy: PromptPlanAdmissionPolicy | None = None) -> None:
        self.planner_config = planner_config
        self.admission_policy = admission_policy

    def materialize(
        self,
        request: PromptWorkflowRequest,
        scan: DirectoryScanReceipt,
        *,
        admission: PromptPlanAdmissionResult | None = None,
        admission_kwargs: Mapping[str, Any] | None = None,
        router: Callable[[str], str] | None = None,
        markdown_path: Path | str | None = None,
        duckdb_path: Path | str | None = None,
        revision: int = 1,
        idempotency_key: str | None = None,
    ) -> PromptProgramMaterialization:
        if request.request_cid != scan.request_cid:
            raise PlanMaterializationError("request and scan identities do not match")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError("revision must be a positive integer")
        planning = generate_prompt_goal_graph(request, scan, router=router, config=self.planner_config)
        graph = planning.graph
        if admission is None:
            options = dict(admission_kwargs or {})
            options.setdefault("repository_tree_id", scan.dirty_worktree_root)
            if self.admission_policy is not None:
                options.setdefault("policy", self.admission_policy)
            admission = admit_prompt_plan(graph, **options)
        if not admission.admitted or admission.admitted_graph is None:
            raise PlanMaterializationAdmissionError("plan admission rejected: " + ", ".join(admission.reason_codes))
        if admission.admitted_graph.plan_root_cid != graph.plan_root_cid:
            raise PlanMaterializationAdmissionError("admission graph does not match planned graph")
        # This check happens before either task-source write.  A planner bug
        # therefore cannot turn a process-local prompt body into a durable
        # Markdown marker, DuckDB JSON record, or history event.
        _safe_receipt_text(graph.to_dict(), request.prompt_source.transient_body)
        # The admission receipt publishes the canonical plan identity.  The
        # planner graph CID remains the candidate identity only.
        plan_root_cid = admission.receipt.final_plan_cid
        goals = CanonicalGoalGraph.from_graph(graph, plan_root_cid=plan_root_cid)
        tasks = CanonicalTaskGraph.from_graph(graph, plan_root_cid=plan_root_cid)
        key = idempotency_key or request.request_cid
        cas = ProgramRevisionCAS(plan_root_cid, revision, key)
        projections: list[TaskSourceProjectionReceipt] = []
        output = request.output_policy
        if output.mode in (OutputMode.MARKDOWN, OutputMode.BOTH):
            path = Path(markdown_path or (Path(output.output_root) / output.markdown_path))
            result: MarkdownMaterializationResult = MarkdownTaskSource(
                path, task_prefix=output.task_prefix, board_namespace=output.board_namespace
            ).materialize(admission, revision=revision)
            projections.append(TaskSourceProjectionReceipt("markdown", plan_root_cid, result.projection.projection_id, revision, len(result.snapshot.tasks), len(graph.goals), result.no_op))
        if output.mode in (OutputMode.DUCKDB, OutputMode.BOTH):
            path = Path(duckdb_path or (Path(output.output_root) / output.duckdb_path))
            source = DuckDBTaskSource(path)
            # DuckDB's direct graph adapter calls the candidate graph CID its
            # plan root.  Give it the same compiler input but bind its root to
            # the admission-published CID, so both durable projections share
            # one authoritative revision identity.
            formal_input = prompt_goal_graph_to_formal_input(
                graph, repository_tree_id=scan.dirty_worktree_root
            )
            formal_input["plan_root_cid"] = plan_root_cid
            installed = source.materialize(formal_input, repository_tree_id=scan.dirty_worktree_root, plan_root_cid=plan_root_cid, expected_absent=False)
            snapshot = source.snapshot()
            projections.append(TaskSourceProjectionReceipt("duckdb", plan_root_cid, snapshot.projection_cid, snapshot.revision, snapshot.task_count, snapshot.goal_count, not bool(installed.get("changed", True))))
        receipt_id = _cid({"request_cid": request.request_cid, "scan_cid": scan.scan_cid, "plan_root_cid": plan_root_cid, "revision": revision, "projections": [item.to_dict() for item in projections]})
        result = PromptProgramMaterialization(receipt_id, request.request_cid, scan.scan_cid, plan_root_cid, cas, goals, tasks, admission, tuple(projections), planning)
        _safe_receipt_text(result.to_dict(), request.prompt_source.transient_body)
        return result


def materialize_prompt_program(*args: Any, **kwargs: Any) -> PromptProgramMaterialization:
    """Functional facade for the production materializer."""
    return PromptProgramMaterializer().materialize(*args, **kwargs)


__all__ = [
    "CanonicalGoalGraph", "CanonicalTaskGraph", "PLAN_MATERIALIZATION_RECEIPT_SCHEMA",
    "PlanMaterializationAdmissionError", "PlanMaterializationConflict", "PlanMaterializationError",
    "ProgramRevisionCAS", "PromptProgramMaterialization", "PromptProgramMaterializer",
    "TaskSourceProjectionReceipt", "materialize_prompt_program",
]
