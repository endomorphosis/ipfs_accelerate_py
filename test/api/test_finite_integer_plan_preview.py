"""Real finite source observations through the bounded create-plan service.

The authored fixtures are isolated Git repositories, not qualified historical
artifacts. Native Python and Lean are required; this suite has no skip fallback.
"""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import inspect
from pathlib import Path
import shutil
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.planning import (
    finite_integer_codebase as matcher,
    finite_integer_plan_preview as adapter,
    finite_integer_plan_service as service_module,
)
from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import (
    FactAuthority, ObligationGraph, ObligationStatus, ObservedFact,
    obligation_id_for_predicate,
)
from ipfs_accelerate_py.agent_supervisor.planning.repository_plan_preview import RepositoryPlanPreviewOwner
from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import (
    PlanCreateInputSnapshot, PlanCreatePreviewReceipt, PlanCreateVerdict,
)
from ipfs_datasets_py.logic.intent_ir.canonicalize import canonical_intent_ir_bytes
from ipfs_datasets_py.logic.intent_ir.decoder import decode_intent_ir
from ipfs_datasets_py.logic.software_contracts import codebase_finite_integer_observation as native
from ipfs_datasets_py.logic.software_contracts.codebase_ir import StaleCodebaseError
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    LeaseCancelledError, LeaseTimeoutError,
)
from test.api.test_finite_integer_codebase import (
    FALSE_FIELDS, INPUTS, LEAN, PYTHON, finite_git, finite_prepared,
    finite_source, finite_text, finite_tools,
)
from test.api.test_plan_create_semantic_input_identity import _cid, _request


TASK_TYPE = "task:finite:type"
TASK_OFFSET = "task:finite:offset"


def operation_catalog():
    return adapter.FiniteIntegerOperationCatalog(operations=tuple(
        adapter.ReviewedFiniteIntegerOperation(
            requirement_id=requirement, task_id=task, producer_id=producer,
            path="calc.py", function_name="increment", parameter="n",
            review_ref="review:authored-finite-operations", operation="update",
        )
        for requirement, task, producer in (
            (matcher.TYPE_STATEMENT_ID, TASK_TYPE, "producer:finite:type"),
            (matcher.OFFSET_STATEMENT_ID, TASK_OFFSET, "producer:finite:offset"),
        )
    ))


def preview_arguments(prepared, tools, *, source_text=None, head=None, output=None):
    selected = prepared["expected_head"] if head is None else head
    text = finite_text() if source_text is None else source_text
    document = matcher.build_finite_integer_intent(text)
    catalog = operation_catalog()
    manifest = prepared["index"].load(selected.manifest_cid)
    base = _request()
    roots = replace(base.roots, repository_id=selected.repository_id,
        repository_root_cid=selected.snapshot_cid,
        dirty_worktree_root=selected.snapshot_cid,
        program_root=manifest.semantic_state.state_cid,
        intent_ir_root=adapter.finite_integer_intent_cid(document),
        capability_catalog_root=catalog.cid)
    request = replace(base, repository_id=selected.repository_id,
        repository_root=str(prepared["repository"].resolve()), scope_paths=("calc.py",),
        prompt_source_cid=adapter.finite_integer_prompt_cid(text), roots=roots, observe_roots=True,
        budget=replace(base.budget, max_model_calls=0, max_tasks=2,
                       max_goals=2, max_latency_ms=90_000))
    owner = RepositoryPlanPreviewOwner(index=prepared["index"],
        repository=prepared["repository"].resolve(), expected_head=selected,
        scheduler=prepared["scheduler"], timeout_seconds=90, memory_mb=1024)
    return {"owner": owner, "request": request, "intent_document": document,
            "source_text": text, "operation_catalog": catalog,
            "output": prepared["output"] if output is None else output,
            "tool_policy": deepcopy(tools), "policy_observer": lambda bound: bound.roots}


def assert_released(prepared):
    state = prepared["scheduler"].snapshot()
    assert state["active_lease_count"] == state["waiting_request_count"] == 0


def track_publication(monkeypatch):
    services, published = [], []
    cls = service_module.FiniteIntegerPlanCreateService
    original = cls._persist
    def persist(self, *args, **kwargs):
        services.append(self)
        published.append(True)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(cls, "_persist", persist)
    return services, published


def assert_proposal(result, prepared):
    assert result["schema"] == "finite-integer-repository-plan-preview@1"
    assert all(result[name] is False for name in FALSE_FIELDS)
    assert result["production_admitted"] is False
    assert result["worker_launched"] is False
    receipt = PlanCreatePreviewReceipt.from_dict(result["preview"])
    snapshot = PlanCreateInputSnapshot.from_dict(result["input_snapshot"])
    graph = ObligationGraph.from_dict(result["obligation_graph"])
    assert receipt.verdict is PlanCreateVerdict.REVIEW_ONLY
    assert receipt.mode.value == "deterministic" and result["model_calls"] == result["training_steps"] == 0
    assert receipt.read_only and receipt.wrote_effects == ()
    assert receipt.input_snapshot_cid == snapshot.snapshot_cid
    assert graph.current_root_id == snapshot.roots.dirty_worktree_root == prepared["expected_head"].snapshot_cid
    assert len(graph.root_obligation_ids) == 2
    assert len(result["match"]["typed_intent"]["desired_predicates"]) == 2
    assert result["declared_task_requirement_ids"] == {
        TASK_TYPE: matcher.TYPE_STATEMENT_ID, TASK_OFFSET: matcher.OFFSET_STATEMENT_ID}
    assert result["current_facts_count"] == len(result["match"]["current_facts"])
    for row in result["match"]["current_facts"]:
        fact = ObservedFact.from_dict(row)
        assert fact.authority is FactAuthority.BOUNDED_OBSERVATION
        assert fact.current_root_id == graph.current_root_id
        assert result["match"]["observation_cid"] in fact.provenance_refs
    stages = {row.stage.value: row for row in receipt.stage_results}
    assert stages["obligation"].passed and stages["candidate"].passed and stages["critique"].passed
    assert not stages["admission"].passed
    assert_released(prepared)
    return receipt, snapshot, graph


def test_one_current_finite_fact_selects_only_the_reviewed_offset_operation(finite_prepared, finite_tools):
    original = (finite_prepared["repository"] / "calc.py").read_bytes()
    result = adapter.preview_finite_integer_plan(**preview_arguments(finite_prepared, finite_tools))
    receipt, snapshot, graph = assert_proposal(result, finite_prepared)
    assert result["planner_status"] == "selected"
    assert result["current_facts_count"] == 1
    assert result["selected_task_ids"] == [TASK_OFFSET]
    assert result["match"]["residual_clause_ids"] == [matcher.OFFSET_STATEMENT_ID]
    bindings = result["match"]["typed_intent"]["metadata"]["requirement_predicate_ids"]
    assert graph.node(obligation_id_for_predicate(bindings[matcher.TYPE_STATEMENT_ID])).status is ObligationStatus.DISCHARGED
    assert graph.node(obligation_id_for_predicate(bindings[matcher.OFFSET_STATEMENT_ID])).status is not ObligationStatus.DISCHARGED
    candidate = result["candidate_plan"]
    assert [task["task_id"] for task in candidate["tasks"]] == [TASK_OFFSET]
    assert len(candidate["effects"]) == 1
    assert candidate["effects"][0]["operation"] == "update"
    assert candidate["effects"][0]["target_id"] == "calc.py"
    assert candidate["effects"][0]["requirement_id"] == matcher.OFFSET_STATEMENT_ID
    assert candidate["tasks"][0]["requirement_id"] == matcher.OFFSET_STATEMENT_ID
    assert candidate["tasks"][0]["review_ref"] == "review:authored-finite-operations"
    assert (finite_prepared["repository"] / "calc.py").read_bytes() == original
    assert snapshot.material_binding["reuse_supported"] is True
    assert receipt.candidate_portfolio_cid and receipt.critique_cid
    parallel = next(stage for stage in receipt.stage_results if stage.stage.value == "parallel_plan")
    assert not parallel.passed
    assert "parallel:resource_infeasible" in parallel.blockers
    assert "parallel:stale_capacity" in parallel.blockers
    assert result["execution_plan"]["admitted"] is False


def test_newly_captured_successor_has_two_facts_and_an_explicit_empty_plan(finite_prepared, finite_tools):
    first = adapter.preview_finite_integer_plan(**preview_arguments(finite_prepared, finite_tools))
    old = finite_prepared["expected_head"]
    commit = finite_git(finite_prepared["repository"], "rev-parse", "HEAD")
    (finite_prepared["repository"] / "calc.py").write_bytes(finite_source(2))
    assert finite_git(finite_prepared["repository"], "rev-parse", "HEAD") == commit
    successor = finite_prepared["index"].prepare_current(finite_prepared["repository"],
        repository_id=old.repository_id, operation_id="finite-plan-successor",
        expected_head=old, scheduler=finite_prepared["scheduler"]).head
    finite_prepared["expected_head"] = successor
    second = adapter.preview_finite_integer_plan(**preview_arguments(finite_prepared, finite_tools,
        output=finite_prepared["output"].parent / "successor-observation"))
    _, _, graph = assert_proposal(second, finite_prepared)
    assert second["planner_status"] == "already_complete_in_finite_domain"
    assert second["current_facts_count"] == 2 and second["selected_task_ids"] == []
    assert second["match"]["residual_clause_ids"] == []
    assert graph.complete
    assert second["candidate_plan"]["tasks"] == []
    assert second["candidate_plan"]["effects"] == []
    assert second["execution_plan"]["status"] == "no_execution_requested"
    assert second["execution_plan"]["task_ids"] == []
    assert second["execution_plan"]["admitted"] is False
    assert first["preview"]["input_snapshot_cid"] != second["preview"]["input_snapshot_cid"]
    assert first["match"]["source_cid"] != second["match"]["source_cid"]


def test_fresh_repeat_runs_the_native_observer_again_and_never_recovers_facts_from_history(finite_prepared, finite_tools, monkeypatch):
    calls = []
    original = native.observe_finite_integer_source
    def observe(**kwargs):
        calls.append(kwargs["output"])
        return original(**kwargs)
    monkeypatch.setattr(native, "observe_finite_integer_source", observe)
    first = adapter.preview_finite_integer_plan(**preview_arguments(finite_prepared, finite_tools))
    second = adapter.preview_finite_integer_plan(**preview_arguments(finite_prepared, finite_tools,
        output=finite_prepared["output"].parent / "repeat-observation"))
    assert calls == [finite_prepared["output"], finite_prepared["output"].parent / "repeat-observation"]
    assert first["selected_task_ids"] == second["selected_task_ids"] == [TASK_OFFSET]
    assert first["match"]["observation_cid"] != second["match"]["observation_cid"]
    assert first["preview"]["input_snapshot_cid"] != second["preview"]["input_snapshot_cid"]
    assert_proposal(second, finite_prepared)


@pytest.mark.parametrize("binding", ["prompt", "intent", "catalog", "repository_root", "dirty_root", "program", "locator", "scope", "model_budget", "task_budget"])
def test_wrong_request_binding_cannot_run_an_observer_or_create_a_proposal(finite_prepared, finite_tools, monkeypatch, binding):
    args = preview_arguments(finite_prepared, finite_tools)
    request = args["request"]
    root_fields = {"intent": "intent_ir_root", "catalog": "capability_catalog_root",
                   "repository_root": "repository_root_cid", "dirty_root": "dirty_worktree_root",
                   "program": "program_root"}
    if binding in root_fields:
        request = replace(request, roots=replace(request.roots, **{root_fields[binding]: _cid("wrong-binding")}))
    elif binding == "prompt":
        request = replace(request, prompt_source_cid=_cid("different-prompt"))
    elif binding == "locator":
        request = replace(request, repository_root=str(finite_prepared["repository"].parent))
    elif binding == "scope":
        request = replace(request, scope_paths=("elsewhere.py",))
    elif binding == "model_budget":
        request = replace(request, budget=replace(request.budget, max_model_calls=1))
    else:
        request = replace(request, budget=replace(request.budget, max_tasks=1))
    args["request"] = request
    monkeypatch.setattr(native, "observe_finite_integer_source", lambda **kwargs: pytest.fail("wrong request executed"))
    _, published = track_publication(monkeypatch)
    with pytest.raises(adapter.FiniteIntegerPlanPreviewError):
        adapter.preview_finite_integer_plan(**args)
    assert not published and not finite_prepared["output"].exists()
    assert_released(finite_prepared)


@pytest.mark.parametrize("field,value", [("path", "other.py"), ("function_name", "different"), ("parameter", "other")])
def test_reviewed_catalog_cannot_retarget_a_clause(finite_prepared, finite_tools, monkeypatch, field, value):
    args = preview_arguments(finite_prepared, finite_tools)
    catalog = args["operation_catalog"]
    changed = adapter.FiniteIntegerOperationCatalog(operations=tuple(
        replace(item, **{field: value}) for item in catalog.operations))
    args["operation_catalog"] = changed
    args["request"] = replace(args["request"], roots=replace(args["request"].roots,
                                                            capability_catalog_root=changed.cid))
    monkeypatch.setattr(native, "observe_finite_integer_source", lambda **kwargs: pytest.fail("retargeted catalog executed"))
    with pytest.raises(adapter.FiniteIntegerPlanPreviewError):
        adapter.preview_finite_integer_plan(**args)
    assert not finite_prepared["output"].exists()


@pytest.mark.parametrize("control", ["missing_clause", "duplicate_requirement", "duplicate_task", "duplicate_producer", "unknown_requirement", "unreviewed", "wrong_operation"])
def test_catalog_requires_two_distinct_reviewed_bounded_operations(control):
    catalog = operation_catalog()
    first, second = catalog.operations
    with pytest.raises(ValueError):
        if control == "missing_clause":
            adapter.FiniteIntegerOperationCatalog(operations=(first,))
        elif control == "duplicate_requirement":
            adapter.FiniteIntegerOperationCatalog(operations=(first, replace(second, requirement_id=first.requirement_id)))
        elif control == "duplicate_task":
            adapter.FiniteIntegerOperationCatalog(operations=(first, replace(second, task_id=first.task_id)))
        elif control == "duplicate_producer":
            adapter.FiniteIntegerOperationCatalog(operations=(first, replace(second, producer_id=first.producer_id)))
        elif control == "unknown_requirement":
            adapter.FiniteIntegerOperationCatalog(operations=(first, replace(second, requirement_id="different-clause")))
        elif control == "unreviewed":
            replace(first, review_ref="")
        else:
            replace(first, operation="create")


def test_catalog_identity_is_frozen_and_reflects_reviewed_operation_identity():
    catalog = operation_catalog()
    with pytest.raises(FrozenInstanceError):
        catalog.operations = ()
    with pytest.raises(FrozenInstanceError):
        catalog.operations[0].review_ref = "changed"
    changed = adapter.FiniteIntegerOperationCatalog(operations=tuple(
        replace(item, review_ref="review:authored-revision-two") for item in catalog.operations))
    assert catalog.cid != changed.cid
    assert catalog.to_dict() != changed.to_dict()
    assert catalog.cid == operation_catalog().cid


def test_prompt_and_ir_have_distinct_dag_json_roles_binding_exact_native_bytes():
    text = finite_text()
    document = matcher.build_finite_integer_intent(text)
    for schema, raw, actual in (
        ("finite-integer-prompt-source@1", text.encode(), adapter.finite_integer_prompt_cid(text)),
        ("finite-integer-intent-source@1", canonical_intent_ir_bytes(document), adapter.finite_integer_intent_cid(document)),
    ):
        assert actual == cid_for_structured({"schema": schema,
            "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)})
        assert actual.startswith("baguq")
    assert adapter.finite_integer_prompt_cid(text) != adapter.finite_integer_intent_cid(document)
    revised = matcher.build_finite_integer_intent(text, source_revision="authored:2")
    assert adapter.finite_integer_intent_cid(revised) != adapter.finite_integer_intent_cid(document)
    assert adapter.finite_integer_prompt_cid(finite_text(offset=3)) != adapter.finite_integer_prompt_cid(text)


@pytest.mark.parametrize("argument", ["observation", "current_facts", "materials", "service_factory", "evidence_bundle", "receipt"])
def test_public_route_accepts_no_caller_evidence_or_stage_injection(finite_prepared, finite_tools, argument):
    assert argument not in inspect.signature(adapter.preview_finite_integer_plan).parameters
    args = preview_arguments(finite_prepared, finite_tools)
    args[argument] = {"passed": True, "admitted": True}
    with pytest.raises(TypeError, match="unexpected keyword"):
        adapter.preview_finite_integer_plan(**args)
    assert not finite_prepared["output"].exists()


def test_unsupported_captured_module_creates_no_service_proposal_or_native_process(finite_prepared, finite_tools, monkeypatch):
    old = finite_prepared["expected_head"]
    (finite_prepared["repository"] / "calc.py").write_bytes(b"import os\ndef increment(n):\n    return n + 1\n")
    head = finite_prepared["index"].prepare_current(finite_prepared["repository"],
        repository_id=old.repository_id, operation_id="unsupported-module",
        expected_head=old, scheduler=finite_prepared["scheduler"]).head
    finite_prepared["expected_head"] = head
    monkeypatch.setattr(native.BoundedToolRunner, "run", lambda *args, **kwargs: pytest.fail("unsupported module executed"))
    _, published = track_publication(monkeypatch)
    with pytest.raises(adapter.FiniteIntegerPlanPreviewError):
        adapter.preview_finite_integer_plan(**preview_arguments(finite_prepared, finite_tools))
    assert not published
    assert_released(finite_prepared)


def test_valid_ir_with_different_clause_meaning_cannot_borrow_the_full_prompt(finite_prepared, finite_tools, monkeypatch):
    args = preview_arguments(finite_prepared, finite_tools)
    changed = args["intent_document"].to_dict()
    offset = next(row for row in changed["statements"] if row["statement_id"] == matcher.OFFSET_STATEMENT_ID)
    offset["arguments"][-1] = "99"
    document = decode_intent_ir(changed)
    args["intent_document"] = document
    args["request"] = replace(args["request"], roots=replace(args["request"].roots,
        intent_ir_root=adapter.finite_integer_intent_cid(document)))
    monkeypatch.setattr(native, "observe_finite_integer_source", lambda **kwargs: pytest.fail("unsupported IR executed"))
    _, published = track_publication(monkeypatch)
    with pytest.raises(adapter.FiniteIntegerPlanPreviewError):
        adapter.preview_finite_integer_plan(**args)
    assert not published and not args["output"].exists()
    assert_released(finite_prepared)


@pytest.mark.parametrize("stage,mutation", [
    ("_stage_candidate", "source"), ("_stage_critique", "source"),
    ("_stage_candidate", "trace"), ("_stage_critique", "lean"),
    ("_stage_candidate", "tool"),
])
def test_live_service_fences_refuse_source_artifact_or_tool_drift_before_publication(finite_prepared, finite_tools, monkeypatch, stage, mutation):
    tools = finite_tools
    private_python = None
    if mutation == "tool":
        private_python = finite_prepared["output"].parent / "native-python"
        shutil.copy2(PYTHON, private_python)
        tools = native.seal_finite_integer_tools(python_executable=private_python, lean_executable=LEAN)
    args = preview_arguments(finite_prepared, tools)
    cls = service_module.FiniteIntegerPlanCreateService
    original = getattr(cls, stage)
    touched, services = [], []
    def mutate(self, *values, **controls):
        result = original(self, *values, **controls)
        services.append(self)
        touched.append(True)
        if mutation == "source":
            (finite_prepared["repository"] / "calc.py").write_bytes(finite_source(2))
        elif mutation == "trace":
            (args["output"] / "observations.json").write_bytes(b"{}")
        elif mutation == "lean":
            with (args["output"] / "FiniteInteger.olean").open("ab") as stream:
                stream.write(b"tampered")
        else:
            with private_python.open("ab") as stream:
                stream.write(b"tampered")
        return result
    monkeypatch.setattr(cls, stage, mutate)
    _, published = track_publication(monkeypatch)
    with pytest.raises((ValueError, RuntimeError)):
        adapter.preview_finite_integer_plan(**args)
    assert touched == [True]
    assert not published and services[0]._preview_by_key == {}
    assert_released(finite_prepared)


@pytest.mark.parametrize("control", ["partial_roots", "stale_policy", "source_drift"])
def test_independent_policy_callback_cannot_shadow_current_owner_or_full_roots(finite_prepared, finite_tools, monkeypatch, control):
    args = preview_arguments(finite_prepared, finite_tools)
    observed = []
    def observe(request):
        observed.append(True)
        if control == "partial_roots":
            return {"policy_root": request.roots.policy_root}
        if control == "stale_policy":
            return replace(request.roots, policy_root=_cid("new-policy"))
        (finite_prepared["repository"] / "calc.py").write_bytes(finite_source(2))
        return request.roots
    args["policy_observer"] = observe
    _, published = track_publication(monkeypatch)
    with pytest.raises((ValueError, RuntimeError)):
        adapter.preview_finite_integer_plan(**args)
    assert observed and not published
    assert_released(finite_prepared)


def test_policy_callback_tampering_with_fresh_certificate_is_rechecked_after_callback(finite_prepared, finite_tools, monkeypatch):
    args = preview_arguments(finite_prepared, finite_tools)
    touched = []
    def observe(request):
        path = args["output"] / "FiniteInteger.olean"
        if path.exists() and not touched:
            with path.open("ab") as stream:
                stream.write(b"policy-callback-tampered")
            touched.append(True)
        return request.roots
    args["policy_observer"] = observe
    _, published = track_publication(monkeypatch)
    with pytest.raises(ValueError):
        adapter.preview_finite_integer_plan(**args)
    assert touched == [True] and not published
    assert_released(finite_prepared)


def test_final_inner_native_exit_rechecks_artifacts_before_service_publication(finite_prepared, finite_tools, monkeypatch):
    args = preview_arguments(finite_prepared, finite_tools)
    index = finite_prepared["index"]
    cls = service_module.FiniteIntegerPlanCreateService
    admission = cls._stage_admission
    observe_current = index.observe_current
    state = {"admission_finished": False, "exit_armed": False}
    services, touched, native_observations = [], [], []

    def finish_admission(self, *values, **controls):
        result = admission(self, *values, **controls)
        services.append(self)
        state["admission_finished"] = True
        return result

    def observe_policy(request):
        # The completed admission stage identifies the last service fence.
        # Arm only after its policy callback, so the genuine context entry and
        # all pre-exit artifact checks still see the unchanged certificate.
        if state["admission_finished"]:
            state["exit_armed"] = True
        return request.roots

    def observe(*values, **controls):
        result = observe_current(*values, **controls)
        native_observations.append(result.head)
        if state["exit_armed"] and not touched:
            with (args["output"] / "FiniteInteger.olean").open("ab") as stream:
                stream.write(b"tampered-during-final-inner-native-exit")
            touched.append(True)
        return result

    args["policy_observer"] = observe_policy
    monkeypatch.setattr(cls, "_stage_admission", finish_admission)
    monkeypatch.setattr(index, "observe_current", observe)
    _, published = track_publication(monkeypatch)
    with pytest.raises(ValueError):
        adapter.preview_finite_integer_plan(**args)
    assert state["admission_finished"] and touched == [True]
    assert native_observations and all(head == finite_prepared["expected_head"] for head in native_observations)
    assert published == [] and len(services) == 1 and services[0]._preview_by_key == {}
    assert_released(finite_prepared)


def test_final_outer_native_exit_cannot_return_a_proposal_with_changed_artifacts(finite_prepared, finite_tools, monkeypatch):
    args = preview_arguments(finite_prepared, finite_tools)
    index = finite_prepared["index"]
    observe_current = index.observe_current
    services, published = track_publication(monkeypatch)
    touched, native_observations = [], []

    def observe(*values, **controls):
        result = observe_current(*values, **controls)
        native_observations.append(result.head)
        # A proposal has completed the private service before the enclosing
        # native context observes its final exit. Mutate the external artifact
        # only after that real observation has returned a valid current head.
        if published and not touched:
            with (args["output"] / "FiniteInteger.olean").open("ab") as stream:
                stream.write(b"tampered-during-final-outer-native-exit")
            touched.append(True)
        return result

    monkeypatch.setattr(index, "observe_current", observe)
    with pytest.raises(ValueError):
        adapter.preview_finite_integer_plan(**args)
    assert published == touched == [True]
    assert native_observations and all(head == finite_prepared["expected_head"] for head in native_observations)
    assert len(services) == 1
    assert services[0].planner_status == "selected"
    assert_released(finite_prepared)


def test_same_byte_new_head_generation_rejects_old_owner_without_execution(finite_prepared, finite_tools, monkeypatch):
    args = preview_arguments(finite_prepared, finite_tools)
    old = finite_prepared["expected_head"]
    successor = finite_prepared["index"].prepare_current(finite_prepared["repository"],
        repository_id=old.repository_id, operation_id="same-byte-successor",
        expected_head=old, scheduler=finite_prepared["scheduler"]).head
    assert successor.snapshot_cid == old.snapshot_cid and successor != old
    monkeypatch.setattr(native, "observe_finite_integer_source", lambda **kwargs: pytest.fail("stale head executed"))
    with pytest.raises(StaleCodebaseError):
        adapter.preview_finite_integer_plan(**args)
    assert not args["output"].exists()
    assert_released(finite_prepared)


@pytest.mark.parametrize("when", ["entry", "candidate"])
def test_cancellation_cannot_return_or_publish_a_finite_plan(finite_prepared, finite_tools, monkeypatch, when):
    args = preview_arguments(finite_prepared, finite_tools)
    signal = threading.Event()
    args["owner"] = replace(args["owner"], cancel_event=signal)
    if when == "entry":
        signal.set()
        monkeypatch.setattr(native, "observe_finite_integer_source", lambda **kwargs: pytest.fail("cancelled preview executed"))
    else:
        cls = service_module.FiniteIntegerPlanCreateService
        original = cls._stage_candidate
        def cancel(self, *values, **controls):
            result = original(self, *values, **controls)
            signal.set()
            return result
        monkeypatch.setattr(cls, "_stage_candidate", cancel)
    _, published = track_publication(monkeypatch)
    with pytest.raises(LeaseCancelledError):
        adapter.preview_finite_integer_plan(**args)
    assert not published
    assert_released(finite_prepared)


def test_expired_overall_budget_cannot_start_native_execution(finite_prepared, finite_tools, monkeypatch):
    args = preview_arguments(finite_prepared, finite_tools)
    args["owner"] = replace(args["owner"], timeout_seconds=1e-9)
    monkeypatch.setattr(native, "observe_finite_integer_source", lambda **kwargs: pytest.fail("expired preview executed"))
    with pytest.raises(LeaseTimeoutError):
        adapter.preview_finite_integer_plan(**args)
    assert not args["output"].exists()
    assert_released(finite_prepared)


def test_deadline_after_actual_candidate_work_prevents_receipt_publication(finite_prepared, finite_tools, monkeypatch):
    args = preview_arguments(finite_prepared, finite_tools)
    cls = service_module.FiniteIntegerPlanCreateService
    original = cls._stage_candidate
    observed = []
    def expire(self, *values, **controls):
        result = original(self, *values, **controls)
        observed.append(True)
        monotonic = adapter.time.monotonic
        monkeypatch.setattr(adapter.time, "monotonic", lambda: monotonic() + 91)
        return result
    monkeypatch.setattr(cls, "_stage_candidate", expire)
    _, published = track_publication(monkeypatch)
    with pytest.raises(LeaseTimeoutError):
        adapter.preview_finite_integer_plan(**args)
    assert observed == [True] and not published
    assert_released(finite_prepared)


def test_output_directory_cannot_reuse_a_prior_positive_observation(finite_prepared, finite_tools, monkeypatch):
    args = preview_arguments(finite_prepared, finite_tools)
    first = adapter.preview_finite_integer_plan(**args)
    assert first["current_facts_count"] == 1
    monkeypatch.setattr(native, "observe_finite_integer_source", lambda **kwargs: pytest.fail("existing output executed"))
    with pytest.raises(ValueError):
        adapter.preview_finite_integer_plan(**args)
    assert_released(finite_prepared)
