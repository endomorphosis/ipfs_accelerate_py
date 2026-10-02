"""Fresh finite observations through strict supervisor planning stages.

This additive proposal profile preserves exact source, clause and finite-domain
bindings. Recorded Lean table arithmetic does not become a source-semantics
ProofReceipt, signed task omission, worker permission or completion authority.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any

from ipfs_datasets_py.logic.intent_ir.canonicalize import canonical_intent_ir_bytes
from ipfs_datasets_py.logic.intent_ir.schema import IntentIRDocument
from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured

from ..prompt.plan_create_service import (
    PlanCreateMaterials, PlanCreateMode, freeze_plan_create_input_snapshot,
)
from . import finite_integer_codebase as matcher
from .adaptive_planner import FrozenPlanningGoal
from .obligation_graph_compiler import (
    FactAuthority, FactTruth, ObservedFact, ProducerRule, TaskCandidate, TypedIntent,
    obligation_id_for_predicate, obligation_id_for_producer,
)
from .plan_evaluator import EvidenceAwarePlanPolicy
from .plan_revision_contracts import PlanAuthorityRoots, PlanCreateRequest
from .repository_plan_preview import RepositoryPlanPreviewOwner
from .structural_codebase_context import structural_codebase_context

SCHEMA = "finite-integer-repository-plan-preview@1"
CATALOG_SCHEMA = "supervisor-finite-integer-operation-catalog@1"
_REQUIREMENTS = frozenset({matcher.TYPE_STATEMENT_ID, matcher.OFFSET_STATEMENT_ID})
_FALSE = {name: False for name in (
    "source_semantics_verified", "runtime_behavior_verified", "behavior_authority",
    "proof_authority", "production_admitted", "execution_authority",
    "completion_authority", "mutation_authority", "omission_authority", "worker_launched",
)}


class FiniteIntegerPlanPreviewError(ValueError):
    """The declared request cannot use the closed finite planning profile."""


def finite_integer_prompt_cid(source_text: str) -> str:
    """Bind exact prompt bytes through the service's required DAG-JSON codec."""
    raw = matcher._text(source_text).encode("utf-8")
    return cid_for_structured({"schema": "finite-integer-prompt-source@1",
        "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)})


def finite_integer_intent_cid(document: IntentIRDocument) -> str:
    """Bind complete canonical IR bytes, including native confidence values."""
    if type(document) is not IntentIRDocument:
        raise FiniteIntegerPlanPreviewError("exact native IntentIR required")
    document.validate()
    raw = canonical_intent_ir_bytes(document)
    return cid_for_structured({"schema": "finite-integer-intent-source@1",
        "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)})


def _identifier(value, label):
    if (type(value) is not str or not value or value != value.strip()
            or len(value.encode("utf-8")) > 512
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise FiniteIntegerPlanPreviewError(label + " must be bounded exact text")
    return value


@dataclass(frozen=True, slots=True)
class ReviewedFiniteIntegerOperation:
    """Explicit caller-reviewed proposal declaration, with no authority claim.

    The review reference records application custody; a supplied label is not
    authentication of a human review or evidence of a successful source edit.
    """

    requirement_id: str
    task_id: str
    producer_id: str
    path: str
    function_name: str
    parameter: str
    review_ref: str
    operation: str = "update"

    def __post_init__(self):
        if (type(self.requirement_id) is not str or self.requirement_id not in _REQUIREMENTS
                or type(self.operation) is not str or self.operation != "update"):
            raise FiniteIntegerPlanPreviewError("exact finite clause and reviewed update operation required")
        for name in ("task_id", "producer_id", "review_ref"):
            _identifier(getattr(self, name), name)
        IntegerOffsetContract(self.path, self.function_name, self.parameter, 0)

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class FiniteIntegerOperationCatalog:
    """Complete immutable two-clause declaration; source generations do not drop rows."""

    operations: tuple[ReviewedFiniteIntegerOperation, ...]

    def __post_init__(self):
        if (type(self.operations) is not tuple or len(self.operations) != 2
                or any(type(item) is not ReviewedFiniteIntegerOperation for item in self.operations)
                or {item.requirement_id for item in self.operations} != _REQUIREMENTS
                or len({item.task_id for item in self.operations}) != 2
                or len({item.producer_id for item in self.operations}) != 2
                or len({(item.path, item.function_name, item.parameter) for item in self.operations}) != 1):
            raise FiniteIntegerPlanPreviewError("complete unique same-target two-clause operation catalog required")
        object.__setattr__(self, "operations", tuple(sorted(self.operations, key=lambda item: item.requirement_id)))

    def to_dict(self):
        return {"schema": CATALOG_SCHEMA, "profile": matcher.PROFILE,
                "operations": [item.to_dict() for item in self.operations],
                "authority": "reviewed_proposal_only"}

    @property
    def cid(self):
        return cid_for_structured(self.to_dict())


def _materials(match, catalog, context):
    """Derive exact records from the owned match, never caller supplied facts."""
    from .finite_integer_plan_service import FINITE_SERVICE_PROFILE
    intent = TypedIntent.from_dict(match["typed_intent"])
    facts = tuple(ObservedFact.from_dict(value) for value in match["current_facts"])
    mapping = dict(intent.metadata["requirement_predicate_ids"])
    predicates = {item.predicate_id: item for item in intent.desired_predicates}
    if (set(mapping) != _REQUIREMENTS or set(mapping.values()) != set(predicates)
            or len(predicates) != 2 or intent.current_root_id != context.head.snapshot_cid
            or any(fact.current_root_id != intent.current_root_id
                   or fact.authority is not FactAuthority.BOUNDED_OBSERVATION
                   or fact.truth is not FactTruth.TRUE or fact.predicate != predicates.get(fact.predicate.predicate_id)
                   for fact in facts)):
        raise FiniteIntegerPlanPreviewError("complete exact root-bound finite clauses and owned facts required")
    producers, tasks, bindings = [], [], {}
    observed = {fact.predicate.predicate_id for fact in facts}
    for operation in catalog.operations:
        predicate = predicates[mapping[operation.requirement_id]]
        closes = (obligation_id_for_predicate(predicate.predicate_id)
                  if predicate.predicate_id in observed else
                  obligation_id_for_producer(operation.producer_id, predicate.predicate_id))
        provenance = (*intent.source_refs, catalog.cid, operation.review_ref)
        producers.append(ProducerRule(producer_id=operation.producer_id,
            effect_predicate_ids=(predicate.predicate_id,), task_candidate_ids=(operation.task_id,),
            provenance_refs=provenance))
        tasks.append(TaskCandidate(candidate_id=operation.task_id, producer_id=operation.producer_id,
            closes_obligation_ids=(closes,), provenance_refs=provenance))
        bindings[operation.task_id] = {**operation.to_dict(), "predicate_id": predicate.predicate_id,
                                    "closes_obligation_ids": [closes], "depends_on": []}
    path = catalog.operations[0].path
    goal = FrozenPlanningGoal(goal_id="finite-goal:" + match["query"]["query_cid"],
        goal_content_id=cid_for_structured({"intent": intent.to_dict(), "operation_catalog_cid": catalog.cid}),
        repository_tree_id=intent.current_root_id, policy=EvidenceAwarePlanPolicy(
            acceptance_criteria=intent.goal_predicate_ids, evidence_terms=intent.source_refs,
            supported_semantics=(matcher.PROFILE,), allowed_scopes=("scope:" + path,),
            available_resource_classes=("cpu",), require_validation=True, require_proof=False))
    context_record = context.to_dict()
    return PlanCreateMaterials(
        scan={"scan_cid": context.cid, "structural_codebase": context_record},
        intent=intent, current_facts=facts, producers=tuple(producers), task_candidates=tuple(tasks),
        frozen_goal=goal, candidate_context={"domain": "finite-integer-fixture",
            "repository_paths": [path], "evidence_scope": "explicit_finite_domain_only",
            "task_metadata": {operation.task_id: {"predicted_files": [path],
                "predicted_symbols": [operation.function_name], "scope_ids": ["scope:" + path],
                "resource_classes": ["cpu"]} for operation in catalog.operations}},
        extra={"finite_service_profile": FINITE_SERVICE_PROFILE,
            "root_observation_profile": "repository-live-root-observation@1",
            "operation_catalog": catalog.to_dict(), "operation_catalog_cid": catalog.cid,
            "finite_operation_bindings": bindings, "finite_match": match,
            "structural_codebase_context_cid": context.cid, "structural_codebase": context_record}), bindings


def preview_finite_integer_plan(*, owner: RepositoryPlanPreviewOwner,
        request: PlanCreateRequest, intent_document: IntentIRDocument, source_text: str,
        operation_catalog: FiniteIntegerOperationCatalog, output,
        tool_policy: dict[str, Any], policy_observer: Callable[[PlanCreateRequest], PlanAuthorityRoots]) -> dict[str, Any]:
    """Observe freshly, plan exact residuals and recheck artifacts at every fence.

    This profile accepts no receipt, match, fact, stage, model or service factory.
    Policy observation is trusted ephemeral application wiring. Every public
    call performs fresh native Python/Lean observations; history is not a bypass.
    """
    from .finite_integer_plan_service import FiniteIntegerPlanCreateService
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
        LeaseCancelledError, LeaseTimeoutError,
    )
    if (type(owner) is not RepositoryPlanPreviewOwner or type(request) is not PlanCreateRequest
            or type(intent_document) is not IntentIRDocument
            or type(operation_catalog) is not FiniteIntegerOperationCatalog
            or not callable(policy_observer) or owner.memory_mb < 1024
            or request.budget.max_model_calls != 0 or request.budget.max_tasks < 2
            or request.budget.max_goals < 2 or request.required_analysis_operations
            or request.optional_analysis_operations or request.required_logic_families
            or request.optional_logic_families):
        raise FiniteIntegerPlanPreviewError("typed owner, bounded model-off request and complete reviewed catalog required")
    query = matcher.prepare_finite_integer_query(intent_document=intent_document, source_text=source_text)
    if not query["supported"] or set(query["requirement_ids"]) != _REQUIREMENTS:
        raise FiniteIntegerPlanPreviewError("complete supported finite instruction and native IntentIR required")
    contract = IntegerOffsetContract.from_dict(query["contract"])
    if (request.repository_root != str(owner.repository)
            or request.repository_id != owner.expected_head.repository_id
            or request.roots.repository_root_cid != owner.expected_head.snapshot_cid
            or request.roots.dirty_worktree_root != owner.expected_head.snapshot_cid
            or request.prompt_source_cid != finite_integer_prompt_cid(source_text)
            or request.roots.intent_ir_root != finite_integer_intent_cid(intent_document)
            or request.roots.capability_catalog_root != operation_catalog.cid
            or request.scope_paths != (contract.path,)
            or any((item.path, item.function_name, item.parameter) !=
                   (contract.path, contract.function_name, contract.parameter)
                   for item in operation_catalog.operations)):
        raise FiniteIntegerPlanPreviewError("prompt, IR, operation, scope or repository roots differ")
    # Detach tools and document before the first callback; callers cannot mutate
    # the pending invocation by editing their original containers.
    policy = matcher._json(tool_policy)
    document = json.loads(canonical_intent_ir_bytes(intent_document))
    deadline = time.monotonic() + min(owner.timeout_seconds, request.budget.max_latency_ms / 1000)

    def remaining():
        if owner.cancel_event is not None and owner.cancel_event.is_set():
            raise LeaseCancelledError("finite planning preview cancelled")
        left = deadline - time.monotonic()
        if left <= 0:
            raise LeaseTimeoutError("finite planning preview deadline exceeded")
        return left

    def controls():
        return {"repository_id": owner.expected_head.repository_id, "expected_head": owner.expected_head,
            "scheduler": owner.scheduler, "parent_lease": owner.parent_lease,
            "cancel_event": owner.cancel_event, "timeout_seconds": remaining(), "memory_mb": owner.memory_mb}

    def observe_policy():
        roots = policy_observer(request)
        if type(roots) is not PlanAuthorityRoots:
            raise FiniteIntegerPlanPreviewError("complete independently observed policy roots required")
        roots.require_current(request.roots)
        remaining()
        return roots

    with structural_codebase_context(owner.index, owner.repository, **controls()) as context:
        if request.roots.program_root != context.semantic_state_cid:
            raise FiniteIntegerPlanPreviewError("native semantic program root differs")
        observe_policy()
        match = matcher.match_finite_integer_intent(index=owner.index, repository=owner.repository,
            repository_id=owner.expected_head.repository_id, intent_document=document, source_text=source_text,
            expected_head=owner.expected_head, output=output, tool_policy=policy,
            scheduler=owner.scheduler, parent_lease=owner.parent_lease, cancel_event=owner.cancel_event,
            timeout_seconds=remaining(), memory_mb=owner.memory_mb)
        if match["observation"] is None or match["observation"]["status"] != "observed":
            raise FiniteIntegerPlanPreviewError("source is unsupported or fresh finite Python/Lean observation failed")
        if (match["structural_context"] != context.to_dict()
                or match["query"] != query or match["current_root_id"] != owner.expected_head.snapshot_cid):
            raise FiniteIntegerPlanPreviewError("fresh owned match differs from exact selected query and context")
        match = matcher._json(match)

        def require_observation():
            remaining()
            if match["match_cid"] != cid_for_structured({key: value for key, value in match.items() if key != "match_cid"}):
                raise FiniteIntegerPlanPreviewError("owned finite match mutated during planning")
            matcher._check_observation(observation=match["observation"], index=owner.index,
                head=owner.expected_head, contract=contract, inputs=query["domain_inputs"],
                tool_policy=policy, output=output)
            remaining()

        def observe_roots(value):
            if value != request:
                raise FiniteIntegerPlanPreviewError("service changed the finite request")
            with structural_codebase_context(owner.index, owner.repository, **controls()) as current:
                if current != context:
                    raise FiniteIntegerPlanPreviewError("selected native context changed during planning")
                require_observation()
                roots = observe_policy()
                # The policy callback itself may change artifacts or tools.
                require_observation()
            # Native source/head observation on context exit is another callback
            # boundary; recheck retained artifacts after that boundary as well.
            require_observation()
            return roots

        materials, bindings = _materials(match, operation_catalog, context)
        snapshot = freeze_plan_create_input_snapshot(request, materials=materials)
        service = FiniteIntegerPlanCreateService(root_observer=observe_roots, operation_bindings=bindings)
        preview = service.preview_create(request, mode=PlanCreateMode.DETERMINISTIC, materials=materials)
        if preview.input_snapshot_cid != snapshot.snapshot_cid:
            raise FiniteIntegerPlanPreviewError("service consumed a different frozen finite input")
        stages = {item.stage.value: item for item in preview.stage_results}
        if any(not stages[name].passed for name in ("scan", "query", "evidence", "obligation", "candidate", "critique")):
            raise FiniteIntegerPlanPreviewError("finite service stage failed closed: " + str(preview.to_dict()))
        if preview.admitted or not preview.read_only or preview.wrote_effects:
            raise FiniteIntegerPlanPreviewError("finite profile cannot admit or apply an execution plan")
        diagnostics = service.diagnostics
        require_observation()
        remaining()
    # Context exit independently observes the native source/head. Do not return
    # a proposal if external observation artifacts changed during that check.
    require_observation()
    remaining()
    def record(value):
        return value.to_dict() if callable(getattr(value, "to_dict", None)) else value
    candidate_plan = record(diagnostics["candidate_plan"])
    result = {"schema": SCHEMA, "profile": matcher.PROFILE,
        "match": match, "preview": preview.to_dict(), "input_snapshot": snapshot.to_dict(),
        "operation_catalog": operation_catalog.to_dict(), "operation_catalog_cid": operation_catalog.cid,
        "obligation_graph": record(diagnostics["obligation_graph"]),
        "portfolio": record(diagnostics["portfolio"]), "candidate_plan": candidate_plan,
        "critique": record(diagnostics["critique"]), "critic_evidence": record(diagnostics["critic_evidence"]),
        "execution_plan": record(diagnostics["execution_plan"]),
        "planner_status": "selected" if candidate_plan["tasks"] else "already_complete_in_finite_domain",
        "declared_task_requirement_ids": {item.task_id: item.requirement_id for item in operation_catalog.operations},
        "selected_task_ids": [task["task_id"] for task in candidate_plan["tasks"]],
        "current_facts_count": len(match["current_facts"]), "scope": "explicit_finite_domain_only",
        "model_calls": 0, "training_steps": 0, **_FALSE}
    result["result_cid"] = cid_for_structured(result)
    result = matcher._json(result)
    remaining()
    return result


__all__ = ["SCHEMA", "ReviewedFiniteIntegerOperation", "FiniteIntegerOperationCatalog",
           "FiniteIntegerPlanPreviewError", "finite_integer_prompt_cid", "finite_integer_intent_cid",
           "preview_finite_integer_plan"]
