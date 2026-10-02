"""Strict bounded behavioral repository materials through the native planner.

The v2 snapshot is a proposal artifact. Exact finite observations retain their
bounded authority; kernel-checked model tables do not prove Python equivalence.
The legacy v1 routes and independently signed task population remain intact.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from ipfs_datasets_py.logic.intent_ir.canonicalize import canonical_intent_ir_bytes
from ipfs_datasets_py.logic.intent_ir.schema import IntentIRDocument
from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
from ipfs_datasets_py.logic.software_contracts.codebase_semantic_manifest import load_codebase_semantic_manifest
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_datasets_py.duckdb_control.intent_codebase_catalog import IntentCodebaseCatalog
from ..proof.finite_checked_cache import FiniteCheckedCache
from ..prompt.plan_create_service import PlanCreateMode, freeze_plan_create_input_snapshot
from . import finite_integer_codebase as matcher
from .behavioral_codebase_match import match_behavioral_intent
from .finite_integer_plan_preview import (
    FiniteIntegerOperationCatalog, FiniteIntegerPlanPreviewError,
    finite_integer_prompt_cid, finite_integer_intent_cid, _materials, _FALSE, _REQUIREMENTS,
)
from .finite_integer_plan_service import FiniteIntegerPlanCreateService
from .plan_revision_contracts import PlanAuthorityRoots, PlanCreateRequest, plan_revision_cid
from .repository_plan_preview import RepositoryPlanPreviewOwner
from .structural_codebase_context import structural_codebase_context

SCHEMA = "repository-proof-planning-snapshot@2"
PROFILE = "repository-finite-behavioral-planning@2"


def repository_evidence_binding(*, owner, semantic_manifest_cid, tool_policy, operation_catalog):
    """Deterministic configuration identity; the preview independently observes it."""
    from . import behavioral_codebase_match, finite_integer_plan_service
    from ..proof import finite_checked_cache
    from ipfs_datasets_py.duckdb_control import intent_codebase_catalog
    modules = (behavioral_codebase_match, finite_integer_plan_service, finite_checked_cache, intent_codebase_catalog)
    return dict(schema="repository-evidence-configuration@2", profile=PROFILE,
        source_head=owner.expected_head.to_dict(), semantic_manifest_cid=semantic_manifest_cid,
        model={"enabled": False, "identity": "explicit-model-off@1"},
        tool_policy=matcher._json(tool_policy), operation_catalog_cid=operation_catalog.cid,
        implementations={module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                         for module in modules},
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def _proof_snapshot(behavioral, request, operations, binding):
    match = behavioral["fresh_match"]
    roots = request.roots.to_dict()
    facts = match["current_facts"]
    if any(not fact["provenance_refs"] or fact["current_root_id"] != request.roots.repository_root_cid
           for fact in facts):
        raise FiniteIntegerPlanPreviewError("every current fact requires exact nonempty source roots")
    value = dict(schema=SCHEMA, profile=PROFILE, roots=roots, configuration=binding,
        model=binding["model"], source_head=behavioral["source_head"],
        semantic_manifest_cid=behavioral["semantic_manifest_cid"],
        complete_inventory=behavioral["complete_inventory"],
        query_plan={"query": match["query"], "exact_discovery": behavioral["discovery"]},
        code_obligations=match["typed_intent"], proof_results={
            "finite_observation": match["observation"], "checked_cache": behavioral["checked_cache"],
            "scope": "explicit_finite_domain_only", "source_equivalence_proved": False},
        current_facts=facts, behavioral_match_cid=behavioral["match_cid"],
        requirements=behavioral["requirement_results"], residuals=behavioral["residual_requirements"],
        operations=operations.to_dict(), full_declared_task_ids=sorted(v.task_id for v in operations.operations),
        eligible_current_fact_roots={fact["fact_id"]: list(fact["provenance_refs"]) for fact in facts},
        execution_authority=False, completion_authority=False, omission_authority=False)
    value["snapshot_cid"] = cid_for_structured(value)
    return value


class RepositoryBehavioralPlanCreateService(FiniteIntegerPlanCreateService):
    """Frozen v2 materials join the existing compiler, planner and critic."""

    def __init__(self, *, root_observer, operation_bindings, proof_snapshot):
        self._proof_snapshot = matcher._json(proof_snapshot)
        super().__init__(root_observer=root_observer, operation_bindings=operation_bindings)

    def _require_materials(self, request, materials):
        super()._require_materials(request, materials)
        value = materials.extra.get("repository_proof_snapshot")
        if value != self._proof_snapshot or value.get("schema") != SCHEMA:
            raise FiniteIntegerPlanPreviewError("complete frozen behavioral snapshot required")
        if (value["snapshot_cid"] != cid_for_structured({k: v for k, v in value.items() if k != "snapshot_cid"})
                or value["roots"] != request.roots.to_dict()
                or value["current_facts"] != materials.extra["finite_match"]["current_facts"]
                or value["code_obligations"] != materials.intent.to_dict()
                or request.roots.configuration_root != plan_revision_cid(value["configuration"])):
            raise FiniteIntegerPlanPreviewError("behavioral proof, premise or current-root binding differs")


def preview_behavioral_repository_plan(*, catalog, checked_cache, semantic_manifest_cid, owner: RepositoryPlanPreviewOwner,
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
    if (type(catalog) is not IntentCodebaseCatalog or type(checked_cache) is not FiniteCheckedCache
            or catalog.index is not owner.index or checked_cache.artifacts is not owner.index.artifacts):
        raise FiniteIntegerPlanPreviewError("behavioral owners must share the exact source and artifact authority")
    binding = repository_evidence_binding(owner=owner, semantic_manifest_cid=semantic_manifest_cid,
        tool_policy=tool_policy, operation_catalog=operation_catalog)
    if request.roots.configuration_root != plan_revision_cid(binding):
        raise FiniteIntegerPlanPreviewError("repository evidence configuration root differs")
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
        def check_binding():
            if repository_evidence_binding(owner=owner, semantic_manifest_cid=semantic_manifest_cid,
                    tool_policy=tool_policy, operation_catalog=operation_catalog) != binding:
                raise FiniteIntegerPlanPreviewError("repository evidence producer or configuration changed")
        check_binding()
        roots = policy_observer(request)
        if type(roots) is not PlanAuthorityRoots:
            raise FiniteIntegerPlanPreviewError("complete independently observed policy roots required")
        roots.require_current(request.roots)
        check_binding()
        remaining()
        return roots

    with structural_codebase_context(owner.index, owner.repository, **controls()) as context:
        if request.roots.program_root != context.semantic_state_cid:
            raise FiniteIntegerPlanPreviewError("native semantic program root differs")
        observe_policy()
        behavioral = match_behavioral_intent(catalog=catalog, checked_cache=checked_cache,
            repository=owner.repository, expected_head=owner.expected_head,
            semantic_manifest_cid=semantic_manifest_cid, intent_document=document, source_text=source_text,
            output=output, tool_policy=policy, scheduler=owner.scheduler, parent_lease=owner.parent_lease,
            cancel_event=owner.cancel_event, timeout_seconds=remaining(), memory_mb=owner.memory_mb)
        if catalog.index is not owner.index:
            raise FiniteIntegerPlanPreviewError("behavioral discovery belongs to another source owner")
        match = behavioral["fresh_match"]
        if match["observation"] is None or match["observation"]["status"] != "observed":
            raise FiniteIntegerPlanPreviewError("source is unsupported or fresh finite Python/Lean observation failed")
        if (match["structural_context"] != context.to_dict()
                or match["query"] != query or match["current_root_id"] != owner.expected_head.snapshot_cid):
            raise FiniteIntegerPlanPreviewError("fresh owned match differs from exact selected query and context")
        match = matcher._json(match)
        manifest = load_codebase_semantic_manifest(owner.index, semantic_manifest_cid)
        policy_receipt_cid = manifest["policy_receipt_cid"]

        def require_observation():
            remaining()
            if match["match_cid"] != cid_for_structured({key: value for key, value in match.items() if key != "match_cid"}):
                raise FiniteIntegerPlanPreviewError("owned finite match mutated during planning")
            matcher._check_observation(observation=match["observation"], index=owner.index,
                head=owner.expected_head, contract=contract, inputs=query["domain_inputs"],
                tool_policy=policy, output=output)
            current = catalog.lookup(owner.repository, expected_head=owner.expected_head,
                policy_receipt_cid=policy_receipt_cid, path=contract.path, contract_cid=contract.cid,
                scheduler=owner.scheduler, parent_lease=owner.parent_lease,
                cancel_event=owner.cancel_event, timeout_seconds=remaining())
            if current != behavioral["discovery"]:
                raise FiniteIntegerPlanPreviewError("exact behavioral discovery changed during planning")
            cached = checked_cache.lookup(owner_inputs=dict(index=owner.index,
                repository=owner.repository, expected_head=owner.expected_head, contract=contract,
                inputs=query["domain_inputs"], tool_policy=policy, scheduler=owner.scheduler,
                parent_lease=owner.parent_lease, cancel_event=owner.cancel_event), timeout_seconds=remaining())
            if any(cached.get(key) != value for key, value in behavioral["checked_cache"].items()):
                raise FiniteIntegerPlanPreviewError("checked proof material changed during planning")
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
        proof_snapshot = _proof_snapshot(behavioral, request, operation_catalog, binding)
        materials = replace(materials, extra={**materials.extra,
            "repository_proof_snapshot": proof_snapshot})
        snapshot = freeze_plan_create_input_snapshot(request, materials=materials)
        service = RepositoryBehavioralPlanCreateService(root_observer=observe_roots,
            operation_bindings=bindings, proof_snapshot=proof_snapshot)
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
    result = {"schema": "repository-behavioral-plan-preview@2", "profile": PROFILE,
        "repository_proof_snapshot": proof_snapshot, "behavioral_match": behavioral,
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


__all__ = ["SCHEMA", "PROFILE", "repository_evidence_binding", "preview_behavioral_repository_plan"]
