"""Tests for PlanCreateService@1 (PDR-030 create-plan preview)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import (
    FactAuthority,
    FactTruth,
    ObservedFact,
    ProducerRule,
    TaskCandidate,
    TypedIntent,
    TypedPredicate,
    compile_obligation_graph,
    obligation_id_for_producer,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_admission_service import (
    PLAN_ADMISSION_SERVICE_INTERFACE,
    PlanAdmissionService,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import (
    DirtyTreePolicy,
    FallbackPolicy,
    PlanAuthorityRoots,
    PlanCreateRequest,
    PlanRequestBudget,
    TaskSourceKind,
    plan_revision_cid,
)
from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import (
    CREATE_STAGE_ORDER,
    PLAN_CREATE_SERVICE_INTERFACE,
    WORKFLOW_PREVIEW_COMPATIBILITY_ALIAS,
    PlanCreateInputSnapshot,
    PlanCreateMaterials,
    PlanCreateMode,
    PlanCreatePreviewReceipt,
    PlanCreateService,
    PlanCreateStage,
    PlanCreateStaleRootError,
    PlanCreateVerdict,
    create_default_plan_create_service,
    freeze_plan_create_input_snapshot,
    plan_create_request_from_workflow,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    DirectoryScanPolicy,
    OutputMode,
    PromptOutputPolicy,
    PromptPlanningPolicy,
    PromptSource,
    PromptSupervisorService,
    PromptWorkflowBudget,
    PromptWorkflowRequest,
)


def _cid(label: str) -> str:
    return plan_revision_cid({"fixture": label})


def _roots(**overrides: object) -> PlanAuthorityRoots:
    values: dict[str, object] = {
        "repository_id": "repository:sha256:plan-create-test",
        "repository_root_cid": _cid("repository-root"),
        "dirty_worktree_root": _cid("effective-tree"),
        "task_source_id": "task-source:markdown:plan-create",
        "task_source_revision": _cid("task-source-revision"),
        "policy_root": _cid("policy"),
        "intent_ir_root": _cid("intent"),
        "legal_ir_root": _cid("legal"),
        "security_ir_root": _cid("security"),
        "program_root": _cid("program"),
        "capability_catalog_root": _cid("capability-catalog"),
        "provider_catalog_root": _cid("provider-catalog"),
        "usage_policy_root": _cid("usage"),
        "configuration_root": _cid("configuration"),
    }
    values.update(overrides)
    return PlanAuthorityRoots(**values)  # type: ignore[arg-type]


def _budget(**overrides: int) -> PlanRequestBudget:
    values = {
        "max_goals": 16,
        "max_tasks": 64,
        "max_graph_depth": 8,
        "max_output_paths": 128,
        "max_ready_width": 1,
        "max_repair_rounds": 2,
        "max_scan_bytes": 8 * 1024 * 1024,
        "max_analysis_operations": 16,
        "max_evidence_items": 128,
        "max_logic_families": 4,
        "max_model_calls": 2,
        "max_latency_ms": 80_000,
        "max_provider_tokens": 8_192,
        "max_cost_micros": 800,
    }
    values.update(overrides)
    return PlanRequestBudget(**values)


def _create_request(**overrides: object) -> PlanCreateRequest:
    values: dict[str, object] = {
        "prompt_source_cid": _cid("prompt"),
        "repository_id": "repository:sha256:plan-create-test",
        "repository_root": "/workspace/plan-create",
        "scope_paths": ("ipfs_accelerate_py/agent_supervisor",),
        "dirty_tree_policy": DirtyTreePolicy.OBSERVE_AND_BIND,
        "task_source_kind": TaskSourceKind.BOTH,
        "board_namespace": "plan-create-test",
        "alias_prefix": "PDR",
        "roots": _roots(),
        "budget": _budget(),
        "required_analysis_operations": (),
        "optional_analysis_operations": (),
        "required_logic_families": (),
        "optional_logic_families": (),
        "fallback_policy": FallbackPolicy.FAIL_CLOSED,
        "redacted_source_metadata": {
            "concepts": ["create_plan_preview"],
            "changed_paths": [
                "ipfs_accelerate_py/agent_supervisor/prompt/plan_create_service.py"
            ],
            "symbols": ["PlanCreateService.preview_create"],
        },
        "caller": "principal:test",
        "idempotency_key": "create:preview:1",
    }
    values.update(overrides)
    return PlanCreateRequest(**values)  # type: ignore[arg-type]


def _obligation_materials() -> PlanCreateMaterials:
    goal = TypedPredicate(
        predicate_id="goal:create_plan_preview",
        predicate_type="behavior_state",
        subject_ref="create_plan_preview",
        provenance_refs=("src/plan_create_service.py",),
        proof_requirement_refs=("proof:create_plan_preview",),
        validation_requirement_refs=("validation:create_plan_preview",),
    )
    prerequisite = TypedPredicate(
        predicate_id="goal:query_ready",
        predicate_type="behavior_state",
        subject_ref="query_ready",
        provenance_refs=("src/query.py",),
        proof_requirement_refs=("proof:query_ready",),
        validation_requirement_refs=("validation:query_ready",),
    )
    producer = ProducerRule(
        producer_id="producer:create_plan",
        effect_predicate_ids=(goal.predicate_id,),
        required_predicate_ids=(prerequisite.predicate_id,),
        provenance_refs=("src/plan_create_service.py",),
        proof_requirement_refs=("proof:create_plan_preview",),
    )
    task = TaskCandidate(
        candidate_id="task:implement-create-preview",
        closes_obligation_ids=(
            obligation_id_for_producer(producer.producer_id, goal.predicate_id),
        ),
        producer_id=producer.producer_id,
        provenance_refs=("src/plan_create_service.py",),
    )
    intent = TypedIntent(
        intent_id="intent:create-plan",
        desired_predicates=(goal,),
        source_refs=("intent-source:create",),
        current_root_id=_roots().dirty_worktree_root,
    )
    fact = ObservedFact(
        fact_id="fact:query_ready",
        predicate=prerequisite,
        truth=FactTruth.TRUE,
        authority=FactAuthority.CURRENT_ROOT_FACT,
        provenance_refs=("evidence:query_ready",),
        current_root_id=_roots().dirty_worktree_root,
    )
    return PlanCreateMaterials(
        intent=intent,
        current_facts=(fact,),
        producers=(producer,),
        task_candidates=(task,),
        predicates=(prerequisite,),
        scan={
            "scan_cid": _cid("scan"),
            "tree_id": _roots().dirty_worktree_root,
            "scope_paths": ["ipfs_accelerate_py/agent_supervisor"],
        },
    )


def test_default_factory_wires_production_analysis_and_admission(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "src").mkdir()
    (repository / "src" / "app.py").write_text("def main():\n    return 0\n")

    service = create_default_plan_create_service(
        repository_allowlist=(repository,),
        index_root=tmp_path / "index",
    )

    assert service.INTERFACE == PLAN_CREATE_SERVICE_INTERFACE
    assert service.production_analysis_wired
    assert service.production_admission_wired
    assert isinstance(service.admission_service, PlanAdmissionService)
    assert (
        service.admission_service.INTERFACE == PLAN_ADMISSION_SERVICE_INTERFACE
    )
    assert service.analysis_factory is not None
    assert service.optional_analysis is not None
    assert service.admission_request_factory is not None

    # Factory also wires a workflow supervisor when provided.
    workflow = PromptSupervisorService()
    assert workflow.optional_analysis is None
    service.wire_analysis_factory(service.analysis_factory)
    service.analysis_factory.wire_prompt_supervisor(workflow)
    assert workflow.optional_analysis is service.optional_analysis
    assert workflow.admission_request_factory is service.admission_request_factory


def test_preview_create_runs_all_pipeline_stages_in_order() -> None:
    service = PlanCreateService()
    request = _create_request()
    materials = _obligation_materials()

    receipt = service.preview_create(
        request,
        mode=PlanCreateMode.DETERMINISTIC,
        materials=materials,
    )

    assert isinstance(receipt, PlanCreatePreviewReceipt)
    assert receipt.request_cid == request.request_cid
    assert receipt.stage_order == tuple(stage.value for stage in CREATE_STAGE_ORDER)
    assert {result.stage for result in receipt.stage_results} == set(
        CREATE_STAGE_ORDER
    )
    assert receipt.scan_cid
    assert receipt.query_plan_cid
    assert receipt.evidence_bundle_cid
    assert receipt.obligation_graph_cid
    assert receipt.candidate_portfolio_cid
    assert receipt.critique_cid
    assert receipt.admission_receipt_cid
    assert receipt.execution_plan_cid
    assert receipt.read_only is True
    assert receipt.wrote_effects == ()
    assert receipt.mode is PlanCreateMode.DETERMINISTIC


def test_preview_is_body_free_read_only_and_restart_serializable() -> None:
    store: dict[str, dict[str, object]] = {}
    service = PlanCreateService(receipt_store=store)
    request = _create_request()
    materials = _obligation_materials()

    first = service.preview_create(request, materials=materials)
    payload = first.to_dict()
    # Body-free: no prompt/source bodies and no secret keys.
    encoded = first.to_json()
    for forbidden in (
        "prompt_body",
        "source_body",
        "source_text",
        "api_key",
        "password",
        "private_key",
    ):
        assert forbidden not in encoded
        assert forbidden not in payload

    restored = PlanCreatePreviewReceipt.from_dict(payload)
    assert restored.receipt_cid == first.receipt_cid
    assert restored.to_dict() == first.to_dict()
    assert PlanCreatePreviewReceipt.from_json(encoded).receipt_cid == first.receipt_cid

    # Restart-serializable / idempotent: same inputs return the exact receipt.
    second = service.preview_create(request, materials=materials)
    assert second.receipt_cid == first.receipt_cid
    assert second is first
    assert first.receipt_cid in store
    assert store[first.receipt_cid]["read_only"] is True
    assert store[first.receipt_cid]["wrote_effects"] == []


def test_stale_root_or_policy_fails_rather_than_regenerating() -> None:
    service = PlanCreateService()
    request = _create_request()
    materials = _obligation_materials()
    materials.current_roots = request.roots

    ok = service.preview_create(request, materials=materials)
    assert ok.request_cid == request.request_cid

    stale_roots = replace(
        request.roots,
        policy_root=_cid("stale-policy"),
    )
    stale_materials = PlanCreateMaterials(
        scan=materials.scan,
        intent=materials.intent,
        current_facts=materials.current_facts,
        producers=materials.producers,
        task_candidates=materials.task_candidates,
        predicates=materials.predicates,
        current_roots=stale_roots,
    )
    with pytest.raises(PlanCreateStaleRootError, match="stale root/policy"):
        service.preview_create(request, materials=stale_materials)

    # Cached preview also re-checks roots and refuses silent regeneration.
    with pytest.raises(PlanCreateStaleRootError):
        service.preview_create(
            request,
            materials=PlanCreateMaterials(
                scan=materials.scan,
                intent=materials.intent,
                current_facts=materials.current_facts,
                producers=materials.producers,
                task_candidates=materials.task_candidates,
                predicates=materials.predicates,
                current_roots=stale_roots,
            ),
        )


def test_deterministic_and_model_assisted_share_exact_inputs_and_bounds() -> None:
    service = PlanCreateService()
    request = _create_request()
    materials = _obligation_materials()
    snapshot = freeze_plan_create_input_snapshot(request)

    model_calls: list[object] = []

    def _provider(frozen_request: object) -> dict[str, object]:
        model_calls.append(frozen_request)
        return {"candidates": []}

    materials_model = PlanCreateMaterials(
        scan=materials.scan,
        intent=materials.intent,
        current_facts=materials.current_facts,
        producers=materials.producers,
        task_candidates=materials.task_candidates,
        predicates=materials.predicates,
        model_provider=_provider,
    )

    deterministic = service.preview_create(
        request,
        mode=PlanCreateMode.DETERMINISTIC,
        materials=materials,
    )
    assisted = service.preview_create(
        request,
        mode=PlanCreateMode.MODEL_ASSISTED,
        materials=materials_model,
    )

    assert deterministic.input_snapshot_cid == snapshot.snapshot_cid
    assert assisted.input_snapshot_cid == snapshot.snapshot_cid
    assert deterministic.input_snapshot_cid == assisted.input_snapshot_cid
    assert deterministic.mode is PlanCreateMode.DETERMINISTIC
    assert assisted.mode is PlanCreateMode.MODEL_ASSISTED
    # Mode must not invent a second bounds surface.
    det_candidate = next(
        item
        for item in deterministic.stage_results
        if item.stage is PlanCreateStage.CANDIDATE
    )
    asst_candidate = next(
        item
        for item in assisted.stage_results
        if item.stage is PlanCreateStage.CANDIDATE
    )
    assert snapshot.bounds_digest in det_candidate.detail_ids
    assert snapshot.bounds_digest in asst_candidate.detail_ids
    assert snapshot.snapshot_cid in det_candidate.detail_ids
    assert snapshot.snapshot_cid in asst_candidate.detail_ids


def test_workflow_preview_is_canonical_compatibility_alias() -> None:
    service = PlanCreateService()
    request = _create_request()
    materials = _obligation_materials()

    create_receipt = service.preview_create(
        request,
        materials=materials,
        compatibility_alias=WORKFLOW_PREVIEW_COMPATIBILITY_ALIAS,
    )
    alias_receipt = service.workflow_preview(request, materials=materials)

    assert WORKFLOW_PREVIEW_COMPATIBILITY_ALIAS == "workflow_preview"
    assert alias_receipt.compatibility_alias == WORKFLOW_PREVIEW_COMPATIBILITY_ALIAS
    assert create_receipt.receipt_cid == alias_receipt.receipt_cid
    # Method-level alias binding for discovery surfaces.
    assert service.preview.__func__ is service.preview_create.__func__


def test_workflow_request_projects_into_create_pipeline() -> None:
    service = PlanCreateService()
    roots = _roots()
    workflow_request = PromptWorkflowRequest(
        prompt_source=PromptSource.inline(
            "Create a plan for the service.",
            redacted_metadata={"summary": "workflow alias"},
        ),
        repository_root="/workspace/plan-create",
        directory="/workspace/plan-create/ipfs_accelerate_py",
        repository_root_cid=roots.repository_root_cid,
        allowlist_cid=_cid("allowlist"),
        scan_policy=DirectoryScanPolicy(
            policy_id="scan:plan-create",
            scanner_version="1.0.0",
        ),
        planning_policy=PromptPlanningPolicy(policy_id="planning:deterministic"),
        output_policy=PromptOutputPolicy(
            policy_id="output:plan-create",
            mode=OutputMode.BOTH,
            output_root="/workspace/plan-create",
            allowed_output_roots=("/workspace/plan-create",),
            markdown_path="generated/work.todo.md",
            duckdb_path="generated/work.duckdb",
        ),
        budget=PromptWorkflowBudget(
            max_files=100,
            max_scan_bytes=2 * 1024 * 1024,
            max_file_bytes=256 * 1024,
            max_symbols=100,
            max_prompt_tokens=1_024,
            max_provider_tokens=2_048,
            max_latency_ms=60_000,
            max_goals=8,
            max_tasks=16,
            max_evidence=32,
            max_graph_depth=8,
            max_serialized_bytes=256 * 1024,
            max_rescue_actions=4,
        ),
        caller="principal:test",
        program_root=roots.program_root,
        intent_ir_root=roots.intent_ir_root,
        legal_ir_root=roots.legal_ir_root,
        security_ir_root=roots.security_ir_root,
        policy_root=roots.policy_root,
        dry_run=True,
    )
    projected = plan_create_request_from_workflow(
        workflow_request,
        repository_id=roots.repository_id,
        task_source_id=roots.task_source_id,
        task_source_revision=roots.task_source_revision,
        capability_catalog_root=roots.capability_catalog_root,
        provider_catalog_root=roots.provider_catalog_root,
        usage_policy_root=roots.usage_policy_root,
        configuration_root=roots.configuration_root,
        dirty_worktree_root=roots.dirty_worktree_root,
    )
    assert isinstance(projected, PlanCreateRequest)
    assert projected.roots.policy_root == roots.policy_root

    materials = _obligation_materials()
    receipt = service.workflow_preview(
        workflow_request,
        materials=materials,
        repository_id=roots.repository_id,
        task_source_id=roots.task_source_id,
        task_source_revision=roots.task_source_revision,
        capability_catalog_root=roots.capability_catalog_root,
        provider_catalog_root=roots.provider_catalog_root,
        usage_policy_root=roots.usage_policy_root,
        configuration_root=roots.configuration_root,
        dirty_worktree_root=roots.dirty_worktree_root,
    )
    assert receipt.compatibility_alias == WORKFLOW_PREVIEW_COMPATIBILITY_ALIAS
    assert receipt.stage_order == tuple(stage.value for stage in CREATE_STAGE_ORDER)
    assert receipt.read_only is True


def test_input_snapshot_round_trip_is_content_addressed() -> None:
    request = _create_request()
    snapshot = PlanCreateInputSnapshot.from_request(request)
    restored = PlanCreateInputSnapshot.from_dict(snapshot.to_dict())
    assert restored.snapshot_cid == snapshot.snapshot_cid
    assert restored.bounds_digest == snapshot.bounds_digest
    assert restored.request_cid == request.request_cid


def test_structural_preview_is_review_only_without_ir_admission_materials() -> None:
    service = PlanCreateService()
    receipt = service.preview_create(
        _create_request(),
        materials=_obligation_materials(),
    )
    # Without IR admission materials the service still runs admission but does
    # not silently grant mutation authority.
    assert receipt.verdict in {
        PlanCreateVerdict.REVIEW_ONLY,
        PlanCreateVerdict.REJECTED,
        PlanCreateVerdict.BLOCKED,
    }
    assert receipt.verdict is not PlanCreateVerdict.ADMITTED or receipt.admission_receipt_cid
    admission = next(
        item
        for item in receipt.stage_results
        if item.stage is PlanCreateStage.ADMISSION
    )
    assert "ir_admission_materials_absent" in admission.blockers or admission.passed


def test_live_query_planner_and_obligation_compiler_are_invoked() -> None:
    service = PlanCreateService()
    materials = _obligation_materials()
    graph = compile_obligation_graph(
        materials.intent,
        current_facts=materials.current_facts,
        producers=materials.producers,
        task_candidates=materials.task_candidates,
        predicates=materials.predicates,
        current_root_id=_roots().dirty_worktree_root,
    )
    assert not graph.planning_blocked

    receipt = service.preview_create(
        _create_request(),
        materials=materials,
    )
    # Query stage always uses the live planner unless a typed plan is supplied.
    query = next(
        item for item in receipt.stage_results if item.stage is PlanCreateStage.QUERY
    )
    assert query.passed
    assert query.artifact_cid
    obligation = next(
        item
        for item in receipt.stage_results
        if item.stage is PlanCreateStage.OBLIGATION
    )
    assert obligation.artifact_cid
