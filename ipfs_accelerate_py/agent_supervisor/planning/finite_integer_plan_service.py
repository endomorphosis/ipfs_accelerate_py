"""Exact finite-operation projection through the real create-plan pipeline.

This private application profile consumes materials derived by a live repository
owner. It does not authenticate caller observation receipts or grant admission.
The complete observed domain and reviewed operations enter the frozen materials;
selected tasks retain their graph meaning rather than acquiring generic effects.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import PurePosixPath
from typing import Any

from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured

from ..prompt.plan_create_service import (
    PlanCreateMaterials, PlanCreateMode, PlanCreateService, PlanCreateServiceError,
    PlanCreateStage, PlanCreateStageResult,
)
from .adaptive_planner import FrozenPlanningGoal
from .obligation_graph_compiler import (
    ObligationGraph, ObligationStatus, ObservedFact, TypedIntent, obligation_id_for_predicate,
)
from .plan_critic import PlanCritic
from .plan_revision_contracts import PlanCreateRequest
from .symbolic_candidate_planner import SymbolicCandidatePortfolio

FINITE_SERVICE_PROFILE = "finite-integer-owner-observation-preview@1"
_PROFILE = "python-integer-offset-finite@1"
_ROW_FIELDS = frozenset({"requirement_id", "task_id", "producer_id", "operation",
    "path", "function_name", "parameter", "review_ref", "predicate_id",
    "closes_obligation_ids", "depends_on"})
_PLAN_SCHEMA = "ipfs_accelerate_py/agent-supervisor/finite-integer-candidate-plan@1"
_NO_WORK_SCHEMA = "ipfs_accelerate_py/agent-supervisor/finite-integer-no-work-candidate@1"
_DIAGNOSTICS = ("obligation_graph", "portfolio", "candidate_plan", "critique",
                "critic_evidence", "execution_plan")


class FiniteIntegerPlanServiceError(PlanCreateServiceError):
    """The selected finite proposal would weaken a reviewed declaration."""


def _json(value: Any) -> Any:
    """Detach bounded exact inert JSON without invoking caller serializers."""
    pending, count = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if count > 80_000 or depth > 32:
            raise FiniteIntegerPlanServiceError("bounded exact finite-operation JSON required")
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise FiniteIntegerPlanServiceError("exact string operation keys required")
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is int:
            if item.bit_length() > 128:
                raise FiniteIntegerPlanServiceError("bounded exact operation integers required")
        elif type(item) not in {str, bool, type(None)}:
            raise FiniteIntegerPlanServiceError("exact inert operation JSON required")
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                     allow_nan=False).encode()
    if len(raw) > 4 * 1024 * 1024:
        raise FiniteIntegerPlanServiceError("finite-operation JSON byte bound exceeded")
    return json.loads(raw)


def _operation_bindings(value: Any) -> dict[str, Any]:
    detached = _json(value)
    if type(detached) is not dict or not 1 <= len(detached) <= 16:
        raise FiniteIntegerPlanServiceError("nonempty bounded exact operation map required")
    for task_id, row in detached.items():
        if type(row) is not dict or set(row) != _ROW_FIELDS or row["task_id"] != task_id:
            raise FiniteIntegerPlanServiceError("complete operation declaration for each task required")
        for name in _ROW_FIELDS - {"closes_obligation_ids", "depends_on"}:
            text = row[name]
            if (type(text) is not str or not text or text.strip() != text or len(text) > 2048
                    or any(character in text for character in "\x00\r\n\t")):
                raise FiniteIntegerPlanServiceError("bounded exact operation identifiers required")
        if row["operation"] != "update":
            raise FiniteIntegerPlanServiceError("only reviewed finite update operations are supported")
        path = PurePosixPath(row["path"])
        if (path.is_absolute() or "\\" in row["path"] or ".." in path.parts
                or path.as_posix() != row["path"] or row["path"] in {"", "."}):
            raise FiniteIntegerPlanServiceError("canonical repository-relative operation path required")
        for name in ("closes_obligation_ids", "depends_on"):
            ids = row[name]
            if (type(ids) is not list or len(ids) > 16
                    or any(type(item) is not str or not item or len(item) > 2048 for item in ids)
                    or ids != sorted(set(ids)) or (name == "closes_obligation_ids" and not ids)):
                raise FiniteIntegerPlanServiceError("complete canonical operation closure/dependency lists required")
    return detached


@dataclass(frozen=True, slots=True)
class _FiniteNoWorkCandidate:
    request_cid: str
    graph_id: str
    root_obligation_ids: tuple[str, ...]
    operation_bindings_cid: str

    def to_dict(self) -> dict[str, Any]:
        return {"schema": _NO_WORK_SCHEMA, "profile": FINITE_SERVICE_PROFILE,
            "request_cid": self.request_cid, "graph_id": self.graph_id,
            "root_obligation_ids": list(self.root_obligation_ids),
            "operation_bindings_cid": self.operation_bindings_cid,
            "planner_status": "already_complete_in_finite_domain", "task_ids": [],
            "production_admitted": False, "execution_authority": False,
            "completion_authority": False}

    @property
    def portfolio_id(self) -> str:
        return cid_for_structured(self.to_dict())


class FiniteIntegerPlanCreateService(PlanCreateService):
    """Private strict-root, model-disabled finite planning preview.

    The owner adapter must independently authenticate and recheck the complete
    observation. Constructor operations are trusted application wiring, detached
    and compared with the exact frozen material declarations on every stage.
    """

    def __init__(self, *, root_observer, operation_bindings):
        if not callable(root_observer):
            raise FiniteIntegerPlanServiceError("live owner root observer required")
        bindings = _operation_bindings(operation_bindings)
        self._operation_wire = json.dumps(bindings, sort_keys=True, separators=(",", ":"),
                                          ensure_ascii=True).encode()
        self._owner_root_observer = root_observer
        self.materials = None
        self._diagnostics_by_receipt: dict[str, dict[str, Any]] = {}
        for name in _DIAGNOSTICS:
            setattr(self, name, None)
        super().__init__(root_observer=root_observer, require_live_root_observation=True)

    @property
    def operation_bindings(self) -> dict[str, Any]:
        return json.loads(self._operation_wire)

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _DIAGNOSTICS}

    @property
    def planner_status(self) -> str:
        return ("already_complete_in_finite_domain" if isinstance(self.portfolio, _FiniteNoWorkCandidate)
                else "selected" if isinstance(self.portfolio, SymbolicCandidatePortfolio)
                and self.portfolio.selected is not None else "unqualified")

    def _require_materials(self, request, materials):
        if (type(request) is not PlanCreateRequest or type(materials) is not PlanCreateMaterials
                or type(materials.extra) is not dict
                or materials.extra.get("finite_service_profile") != FINITE_SERVICE_PROFILE
                or _operation_bindings(materials.extra.get("finite_operation_bindings")) != self.operation_bindings):
            raise FiniteIntegerPlanServiceError("exact frozen finite service profile and operations required")
        if (materials.query_plan is not None or materials.evidence_bundle is not None
                or materials.obligation_graph is not None or materials.model_provider is not None
                or materials.admission_materials is not None or materials.current_roots is not None
                or materials.evidence_adapters or materials.evidence_queries
                or materials.parallel_request is not None or materials.parallel_tasks is not None
                or materials.workflow_request is not None):
            raise FiniteIntegerPlanServiceError("finite service cannot accept injected stages, static roots or models")
        if (type(materials.intent) is not TypedIntent or type(materials.frozen_goal) is not FrozenPlanningGoal
                or materials.intent.current_root_id != request.roots.dirty_worktree_root
                or materials.frozen_goal.repository_tree_id != request.roots.dirty_worktree_root
                or materials.frozen_goal.policy.require_proof
                or request.budget.max_model_calls != 0):
            raise FiniteIntegerPlanServiceError("source-rooted model-disabled typed finite materials required")
        bindings = self.operation_bindings
        requirements = {row["requirement_id"]: row["predicate_id"] for row in bindings.values()}
        if (len(bindings) != 2 or len(requirements) != 2
                or requirements != dict(materials.intent.metadata.get("requirement_predicate_ids", {}))
                or set(requirements.values()) != set(materials.intent.goal_predicate_ids)):
            raise FiniteIntegerPlanServiceError("both declared finite clause meanings must remain bound")
        match = _json(materials.extra.get("finite_match"))
        if any(type(item) not in {ObservedFact, dict} for item in materials.current_facts):
            raise FiniteIntegerPlanServiceError("exact native facts or detached fact dictionaries required")
        facts = [item.to_dict() if type(item) is ObservedFact else item for item in materials.current_facts]
        if (type(match) is not dict or match.get("profile") != _PROFILE
                or match.get("typed_intent") != materials.intent.to_dict()
                or match.get("current_facts") != _json(facts)
                or match.get("current_root_id") != request.roots.dirty_worktree_root):
            raise FiniteIntegerPlanServiceError("full owner finite match must bind exact typed intent and current facts")
        if set(materials.candidate_context) & {"request_cid", "input_snapshot_cid", "mode", "bounds_digest", "scope_paths"}:
            raise FiniteIntegerPlanServiceError("finite candidate context cannot shadow frozen service inputs")
        if materials.scan is not None:
            expected = {"scan_cid": materials.extra.get("structural_codebase_context_cid"),
                        "structural_codebase": materials.extra.get("structural_codebase")}
            if type(materials.scan) is not dict or materials.scan != expected or not expected["scan_cid"]:
                raise FiniteIntegerPlanServiceError("scan must be the exact owner structural context")
        if (self.require_live_root_observation is not True
                or self.root_observer is not self._owner_root_observer
                or self.receipt_store is not None or type(self.critic) is not PlanCritic):
            raise FiniteIntegerPlanServiceError("private strict-root native critic service required")

    def preview_create(self, request, *, mode=PlanCreateMode.DETERMINISTIC,
                       materials=None, compatibility_alias=""):
        if mode not in (PlanCreateMode.DETERMINISTIC, PlanCreateMode.DETERMINISTIC.value) or compatibility_alias:
            raise FiniteIntegerPlanServiceError("finite owner preview requires the deterministic direct service")
        with self._lock:
            self._require_materials(request, materials)
            self.materials = materials
            for name in _DIAGNOSTICS:
                setattr(self, name, None)
            receipt = super().preview_create(request, mode=mode, materials=materials)
            if receipt.receipt_cid in self._diagnostics_by_receipt:
                for name, value in self._diagnostics_by_receipt[receipt.receipt_cid].items():
                    setattr(self, name, _json(value) if type(value) is dict else value)
            else:
                self._diagnostics_by_receipt[receipt.receipt_cid] = {
                    name: _json(value) if type(value) is dict else value
                    for name, value in self.diagnostics.items()}
            return receipt

    def _require_current_roots(self, request, materials):
        self._require_materials(request, materials)
        return super()._require_current_roots(request, materials)

    def _assert_graph(self, graph):
        if type(graph) is not ObligationGraph:
            raise FiniteIntegerPlanServiceError("native finite obligation graph required")
        bindings = self.operation_bindings
        tasks = {task.candidate_id: task for task in graph.task_candidates}
        producers = {producer.producer_id: producer for producer in graph.producers}
        if (set(tasks) != set(bindings)
                or set(producers) != {row["producer_id"] for row in bindings.values()}
                or set(graph.root_obligation_ids) != {
                    obligation_id_for_predicate(row["predicate_id"]) for row in bindings.values()}):
            raise FiniteIntegerPlanServiceError("complete finite graph/task population must preserve both clauses")
        for task_id, row in bindings.items():
            task = tasks[task_id]
            producer = producers.get(row["producer_id"])
            if (task.producer_id != row["producer_id"]
                    or list(task.closes_obligation_ids) != row["closes_obligation_ids"]
                    or list(task.depends_on_candidate_ids) != row["depends_on"]
                    or row["review_ref"] not in task.provenance_refs
                    or producer is None or row["review_ref"] not in producer.provenance_refs
                    or producer.effect_predicate_ids != (row["predicate_id"],)
                    or producer.task_candidate_ids != (task_id,)
                    or producer.executable is not True or producer.required_predicate_ids
                    or producer.assumption_refs or producer.proof_requirement_refs
                    or producer.validation_requirement_refs or producer.invalidation_selectors):
                raise FiniteIntegerPlanServiceError("reviewed operation differs from graph producer, closure or dependencies")

    def _stage_obligation(self, request, materials, evidence):
        self._require_materials(request, materials)
        result, graph = super()._stage_obligation(request, materials, evidence)
        self.obligation_graph = graph
        self._assert_graph(graph)
        return result, graph

    @staticmethod
    def _complete(graph):
        return bool(graph.root_obligation_ids) and all(
            graph.node(root).status is ObligationStatus.DISCHARGED
            for root in graph.root_obligation_ids)

    def _no_work(self, request, graph):
        return _FiniteNoWorkCandidate(request.request_cid, graph.graph_id,
            graph.root_obligation_ids, cid_for_structured(self.operation_bindings))

    def _stage_candidate(self, request, materials, obligation, *, mode, snapshot):
        self._require_materials(request, materials)
        self._assert_graph(obligation)
        if obligation.planning_blocked or obligation.review_required:
            raise FiniteIntegerPlanServiceError("finite obligation graph remains blocked or requires semantic review")
        if self._complete(obligation):
            portfolio = self._no_work(request, obligation)
            result = PlanCreateStageResult(PlanCreateStage.CANDIDATE, portfolio.portfolio_id, True,
                message="both observed finite clause roots are complete; no task is invented")
        else:
            result, portfolio = super()._stage_candidate(request, materials, obligation, mode=mode, snapshot=snapshot)
        self.portfolio = portfolio
        return result, portfolio

    def _canonical_plan(self, request, portfolio, graph):
        self._assert_graph(graph)
        if isinstance(portfolio, _FiniteNoWorkCandidate):
            if not self._complete(graph) or portfolio != self._no_work(request, graph):
                raise FiniteIntegerPlanServiceError("no-work candidate must bind both exact discharged roots")
            selected_ids = []
        else:
            if type(portfolio) is not SymbolicCandidatePortfolio or portfolio.selected is None:
                raise FiniteIntegerPlanServiceError("a real selected native finite candidate is required")
            if portfolio.request.obligation_graph != graph:
                raise FiniteIntegerPlanServiceError("selected candidate is detached from the exact graph")
            selected = portfolio.selected.symbolic_candidate
            selected_ids = list(selected.schedule.task_ids)
            if not selected_ids or set(selected_ids) != set(selected.task_candidate_ids):
                raise FiniteIntegerPlanServiceError("exact selected native schedule population required")
        bindings, tasks, effects = self.operation_bindings, [], []
        for task_id in selected_ids:
            row = bindings.get(task_id)
            if row is None or not set(row["depends_on"]) <= set(selected_ids):
                raise FiniteIntegerPlanServiceError("selected task drops a declared operation or dependency")
            effect_id = "finite-effect:" + cid_for_structured({
                "schema": "finite-reviewed-operation-effect@1", "request_cid": request.request_cid,
                "operation_binding": row})
            effects.append({"effect_id": effect_id, "task_id": task_id,
                "target_id": row["path"], "operation": row["operation"],
                "predicate_id": row["predicate_id"], "requirement_id": row["requirement_id"],
                "review_ref": row["review_ref"]})
            tasks.append({"task_id": task_id, "producer_id": row["producer_id"],
                "depends_on": row["depends_on"], "closes_obligation_ids": row["closes_obligation_ids"],
                "requirement_id": row["requirement_id"], "review_ref": row["review_ref"],
                "outputs": [row["path"]], "predicted_files": [row["path"]],
                "predicted_symbols": [row["function_name"]], "resource_class": "cpu",
                "estimated_duration_ms": 1000, "expected_effects": [effect_id], "effect_ids": [effect_id]})
        body = {"schema": _PLAN_SCHEMA, "profile": FINITE_SERVICE_PROFILE,
            "request_cid": request.request_cid, "graph_id": graph.graph_id,
            "operation_bindings_cid": cid_for_structured(bindings),
            "required_goal_ids": list(graph.root_obligation_ids), "tasks": tasks, "effects": effects,
            "expected_effect_ids": [effect["effect_id"] for effect in effects],
            "production_admitted": False, "execution_authority": False, "completion_authority": False}
        return {**body, "plan_id": cid_for_structured(body)}

    def _candidate_plan_projection(self, request, portfolio, obligation):
        plan = self._canonical_plan(request, portfolio, obligation)
        self.candidate_plan = _json(plan)
        return plan

    def _stage_critique(self, request, materials, portfolio, obligation, evidence, candidate_plan):
        self._require_materials(request, materials)
        if candidate_plan != self._canonical_plan(request, portfolio, obligation):
            raise FiniteIntegerPlanServiceError("candidate projection changed reviewed task or effect semantics")
        self.critic_evidence = {"schema": "finite-integer-critic-evidence@1",
            "finite_match": _json(materials.extra["finite_match"]),
            "structural_evidence": evidence.to_dict() if callable(getattr(evidence, "to_dict", None)) else evidence,
            "operation_bindings": self.operation_bindings}
        critique = self.critic.critique(candidate_plan, obligation_graph=obligation,
            evidence=self.critic_evidence, required_goal_ids=obligation.root_obligation_ids,
            expected_effects=candidate_plan["expected_effect_ids"])
        self.critique = critique
        passed = critique.accepted and not critique.truncated
        return PlanCreateStageResult(PlanCreateStage.CRITIQUE, critique.critique_id, passed,
            blockers=() if passed else ("finite_critique_not_accepted",),
            message="native critic replayed all finite roots and exact reviewed operation effects"), critique

    def _stage_parallel_plan(self, request, materials, candidate_plan):
        self._require_materials(request, materials)
        if candidate_plan != self._canonical_plan(request, self.portfolio, self.obligation_graph):
            raise FiniteIntegerPlanServiceError("parallel input differs from the exact reviewed projection")
        if isinstance(self.portfolio, _FiniteNoWorkCandidate):
            body = {"schema": "ipfs_accelerate_py/agent-supervisor/finite-integer-no-work-parallel-plan@1",
                "profile": FINITE_SERVICE_PROFILE, "plan_id": candidate_plan["plan_id"],
                "task_ids": [], "admitted": False, "execution_authority": False,
                "completion_authority": False, "status": "no_execution_requested"}
            plan = {**body, "execution_plan_id": cid_for_structured(body)}
            result = PlanCreateStageResult(PlanCreateStage.PARALLEL_PLAN, plan["execution_plan_id"], True,
                message="finite no-work proposal requests no execution plan")
        else:
            result, plan = super()._stage_parallel_plan(request, materials, candidate_plan)
        self.execution_plan = plan
        return result, plan

    def _stage_admission(self, request, materials, candidate_plan, critique, execution_plan):
        self._require_materials(request, materials)
        if candidate_plan != self._canonical_plan(request, self.portfolio, self.obligation_graph):
            raise FiniteIntegerPlanServiceError("admission input differs from the exact reviewed projection")
        body = {"schema": "ipfs_accelerate_py/agent-supervisor/finite-integer-review-only-admission@1",
            "profile": FINITE_SERVICE_PROFILE, "request_cid": request.request_cid,
            "plan_id": candidate_plan["plan_id"], "admitted": False, "verdict": "review_only",
            "production_admitted": False, "execution_authority": False,
            "proof_authority": False, "completion_authority": False}
        admission = {**body, "receipt_id": cid_for_structured(body)}
        return PlanCreateStageResult(PlanCreateStage.ADMISSION, admission["receipt_id"], False,
            blockers=("ir_admission_materials_absent",),
            message="finite owner proposal requires a separate signed evidence admission"), admission


__all__ = ["FINITE_SERVICE_PROFILE", "FiniteIntegerPlanCreateService", "FiniteIntegerPlanServiceError"]
