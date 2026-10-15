"""Signed complete preview-to-successor joins with explicit current-owner doubles.

Git, local lifecycle signatures, typed planning stages and full-task replay are
genuine. Source/model/checker observation is explicitly doubled; these controls
grant no native freshness, numerical, worker execution or proof authority.
"""
from copy import deepcopy
from dataclasses import dataclass, fields, replace
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.adaptive_planner import FrozenPlanningGoal
from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import (
    PredicatePolarity, ProducerRule, TaskCandidate, TypedIntent, TypedPredicate,
    obligation_id_for_producer,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_evaluator import EvidenceAwarePlanPolicy
from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import PlanAuthorityRoots
from ipfs_accelerate_py.agent_supervisor.planning.structural_codebase_context import StructuralCodebaseContext
from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import (
    LIVE_ROOT_OBSERVATION_PROFILE, PlanCreateMaterials, PlanCreateService,
    freeze_plan_create_input_snapshot,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.runtime import (
    codebase_preview_successor_admission as bridge_owner,
    codebase_preview_successor_contract as contract,
    codebase_successor_dispatch_admission as successor,
    local_planning_admission as local,
)
from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError
from test.api.test_codebase_successor_dispatch_admission import (
    prohibit_private_source_model_and_numerical_work, successor_case,  # noqa: F401
)
from test.api.test_plan_create_semantic_input_identity import _cid, _request


def _current_arguments(case):
    return {name: case[name] for name in (
        "selection", "root", "completion", "index", "repository", "registry")}


def _task_specs(tasks):
    keys = {task.task_cid: task.task_key for task in tasks}
    return [{"task_key": task.task_key, "scope_paths": list(task.scope_paths),
        "outputs": [{name: getattr(item, name) for name in ("path", "effect", "media_type")}
                    for item in task.outputs],
        "validations": [{name: local._plain(getattr(item, name)) for name in (
            "validation_key", "argv", "cwd", "expected_exit_codes", "policy_cid")}
            for item in task.validations],
        "acceptance": [{name: local._plain(getattr(item, name)) for name in (
            "criterion_key", "criterion", "evidence_cids", "validation_keys")}
            for item in task.acceptance],
        "dependencies": [keys[cid] for cid in task.dependency_task_cids]}
        for task in tasks]


def _author_graph_admission(case, graph):
    roots = {name: getattr(graph, name) for name in ("request_cid", "scan_cid", "program_root")}
    manifest = successor.author_current_successor_manifest(
        **_current_arguments(case), profile_dir=case["profile"], lifecycle_dir=case["lifecycle"],
        task_specs=_task_specs(graph.tasks), planning_roots=roots)
    admission = successor.admit_current_successor_plan(
        **_current_arguments(case), manifest=manifest, graph=graph)
    return manifest, admission


@pytest.fixture
def bridge_case(successor_case, monkeypatch):
    case = successor_case
    head = CodebaseHead.from_dict(case["context"]["selection"]["value"]["current_head"])
    context = StructuralCodebaseContext(head, _cid("independent-current-semantic-state"), {
        "inventory_entries": 2, "captured_entries": 2, "ast_ok": 0, "ast_partial": 0,
        "ast_failed": 0, "opaque_entries": 0, "unindexed_entries": 2,
        "semantic_symbols": 0, "formalized_properties": 0, "checked_properties": 0})
    base = _request()
    roots = replace(base.roots, repository_id=head.repository_id,
        repository_root_cid=head.snapshot_cid, dirty_worktree_root=head.snapshot_cid,
        program_root=context.semantic_state_cid)
    request = replace(base, repository_id=head.repository_id,
        repository_root=str(case["repository"].resolve()), scope_paths=("input.py", "keep.py"),
        roots=roots, observe_roots=True, budget=replace(base.budget, max_model_calls=0))
    predicates = tuple(TypedPredicate(f"requirement:preview:{number}", "reviewed_goal",
        "fixture:target", object_ref=f"independent:requirement:{number}") for number in range(2))
    intent = TypedIntent("intent:independent-preview", predicates, ("source:authored",),
        current_root_id=head.snapshot_cid)
    producers = tuple(ProducerRule(f"producer:preview:{number}", (predicate.predicate_id,))
        for number, predicate in enumerate(predicates))
    candidates = tuple(TaskCandidate(f"candidate:preview:{number}",
        (obligation_id_for_producer(producer.producer_id, predicate.predicate_id),),
        producer_id=producer.producer_id,
        depends_on_candidate_ids=() if number == 0 else ("candidate:preview:0",))
        for number, (producer, predicate) in enumerate(zip(producers, predicates)))
    policy = EvidenceAwarePlanPolicy(acceptance_criteria=intent.goal_predicate_ids,
        evidence_terms=intent.source_refs, allowed_scopes=("scope:repository",),
        available_resource_classes=("cpu",), require_validation=True, require_proof=False)
    materials = PlanCreateMaterials(intent=intent, producers=producers,
        task_candidates=candidates,
        frozen_goal=FrozenPlanningGoal("goal:preview", _cid("independent-authored-goal"),
                                      head.snapshot_cid, policy),
        candidate_context={"domain": "signed-preview-protocol-control",
            "task_metadata": {candidate.candidate_id: {"predicted_files": ["input.py"],
                "scope_ids": ["scope:repository"], "resource_classes": ["cpu"]}
                for candidate in candidates}},
        extra={"root_observation_profile": LIVE_ROOT_OBSERVATION_PROFILE,
               "authored_metadata": "independent fixture"})
    bound = contract.bind_repository_preview_materials(materials, context)
    service = PlanCreateService(root_observer=lambda value: value.roots,
                               require_live_root_observation=True)
    preview = service.preview_create(request, materials=bound)
    assert preview.input_snapshot_cid == freeze_plan_create_input_snapshot(
        request, materials=bound).snapshot_cid
    stages = {stage.stage.value: stage for stage in preview.stage_results}
    assert stages["obligation"].passed is stages["candidate"].passed is True
    # The workflow and create schemas retain their independently authored
    # request/scan identities; the companion records the explicit projection.
    graph = replace(case["graph"], program_root=context.semantic_state_cid)
    manifest, admission = _author_graph_admission(case, graph)
    case["observations"].clear()
    state = {"roots": roots, "checker": _cid("independent-live-checker-policy"),
             "events": [], "roots_callback": None, "checker_callback": None,
             "structural_context": context, "structural_calls": []}

    def roots_observer(value):
        assert value == request
        state["events"].append("roots")
        if state["roots_callback"] is not None:
            state["roots_callback"]()
        return state["roots"]

    def checker_observer(value):
        assert value == request
        state["events"].append("checker")
        if state["checker_callback"] is not None:
            state["checker_callback"]()
        return state["checker"]

    def structural_observation(index, receipt):
        assert index is case["index"]
        assert receipt["head"] == case["context"]["selection"]["value"]["current_head"]
        # Re-observation is allowed while the native closing fence is unused.
        assert case["observations"][-1] == "entry"
        state["structural_calls"].append(receipt)
        return state["structural_context"]

    monkeypatch.setattr(bridge_owner, "_structural_context_from_current", structural_observation)
    # A graph may canonically sort tasks by CID. Preserve the authored
    # dependency order using keys rather than depending on that wire order.
    by_key = {task.task_key: task for task in graph.tasks}
    ordered_tasks = tuple(by_key[task.task_key] for task in case["tasks"])
    candidate_bindings = [{"candidate_id": candidate.candidate_id, "task_cid": task.task_cid}
                          for candidate, task in zip(candidates, ordered_tasks)]
    requirement_bindings = [{"predicate_id": predicate.predicate_id,
        "task_cids": [task.task_cid],
        "acceptance_cids": sorted(item.content_id for item in task.acceptance)}
        for predicate, task in zip(predicates, ordered_tasks)]
    fixture = {"case": case, "state": state, "context": context, "request": request,
        "materials": materials, "preview": preview, "graph": graph, "manifest": manifest,
        "admission": admission, "candidate_bindings": candidate_bindings,
        "requirement_bindings": requirement_bindings, "roots_observer": roots_observer,
        "checker_observer": checker_observer}
    fixture["bridge"] = bridge_owner.author_current_preview_successor_bridge(
        **_author_arguments(fixture))
    case["observations"].clear()
    state["events"].clear()
    state["structural_calls"].clear()
    return fixture


def _author_arguments(fixture):
    return {**_current_arguments(fixture["case"]), **{name: fixture[name] for name in (
        "admission", "request", "materials", "preview", "candidate_bindings",
        "requirement_bindings", "roots_observer", "checker_observer")},
        "checker_policy_cid": fixture["state"]["checker"]}


def _verify_arguments(fixture):
    return {**_current_arguments(fixture["case"]), **{name: fixture[name] for name in (
        "bridge", "admission", "request", "materials", "preview", "roots_observer",
        "checker_observer")}}


def _verify(fixture, **changes):
    return bridge_owner.verify_current_preview_successor_bridge(
        **{**_verify_arguments(fixture), **changes})


def _different_root(roots, name):
    value = "repository:other" if name == "repository_id" else (
        "source:other" if name == "task_source_id" else _cid("changed-root:" + name))
    return replace(roots, **{name: value})


def test_genuine_preview_and_signature_join_every_distinct_typed_task_and_requirement(bridge_case):
    fixture = bridge_case
    receipt = _verify(fixture)
    expected = sorted(task.task_cid for task in fixture["graph"].tasks)
    assert receipt["schema"] == "supervisor-current-preview-successor-bridge-verification@1"
    assert receipt["administrator_task_cids"] == expected
    assert receipt["admission_cid"] == content_identity(fixture["admission"])
    assert receipt["input_snapshot_cid"] == fixture["preview"].input_snapshot_cid
    assert receipt["checker_policy_cid"] == fixture["state"]["checker"]
    mapping = fixture["bridge"]["payload"]["root_mapping"]
    assert mapping["create_request_cid"] == fixture["request"].request_cid
    assert mapping["workflow_request_cid"] == fixture["graph"].request_cid
    assert mapping["create_request_cid"] != mapping["workflow_request_cid"]
    assert mapping["preview_scan_cid"] == fixture["preview"].scan_cid
    assert mapping["workflow_scan_cid"] == fixture["graph"].scan_cid
    assert mapping["preview_scan_cid"] != mapping["workflow_scan_cid"]
    assert receipt["current_facts"] == receipt["removed_task_cids"] == []
    assert receipt["observed_current"] is True and receipt["worker_profile_integrated"] is False
    assert all(value is False for value in receipt["authority"].values())
    assert {row["candidate_id"] for row in fixture["candidate_bindings"]}.isdisjoint(expected)
    assert {row["predicate_id"] for row in fixture["requirement_bindings"]}.isdisjoint(expected)
    assert fixture["case"]["observations"] == ["entry", "close", "exit"] * 2
    assert len(fixture["state"]["structural_calls"]) >= 2
    assert fixture["materials"].scan is None and fixture["materials"].current_facts == ()
    assert all(fixture["case"]["intent"].get_task(cid) is None for cid in expected)


@pytest.mark.parametrize("target,change", [
    ("candidate_bindings", "missing"), ("candidate_bindings", "duplicate"),
    ("candidate_bindings", "unknown_candidate"), ("candidate_bindings", "unknown_task"),
    ("candidate_bindings", "swapped"), ("candidate_bindings", "extra_field"),
    ("requirement_bindings", "missing"), ("requirement_bindings", "duplicate"),
    ("requirement_bindings", "unknown_predicate"), ("requirement_bindings", "missing_task"),
    ("requirement_bindings", "wrong_task"), ("requirement_bindings", "missing_acceptance"),
    ("requirement_bindings", "foreign_acceptance"), ("requirement_bindings", "extra_field"),
])
def test_complete_authored_population_cannot_be_replaced_by_partial_or_ambiguous_maps(
        bridge_case, target, change):
    fixture = bridge_case
    rows = deepcopy(fixture[target])
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif change == "unknown_candidate":
        rows[0]["candidate_id"] = "candidate:unknown"
    elif change == "unknown_task":
        rows[0]["task_cid"] = _cid("unknown-task")
    elif change == "swapped":
        rows[0]["task_cid"], rows[1]["task_cid"] = rows[1]["task_cid"], rows[0]["task_cid"]
    elif change == "unknown_predicate":
        rows[0]["predicate_id"] = "requirement:unknown"
    elif change == "missing_task":
        rows[0]["task_cids"] = []
    elif change == "wrong_task":
        rows[0]["task_cids"] = list(rows[1]["task_cids"])
    elif change == "missing_acceptance":
        rows[0]["acceptance_cids"] = []
    elif change == "foreign_acceptance":
        rows[0]["acceptance_cids"] = [_cid("unrelated-acceptance")]
    else:
        rows[0]["authoritative"] = True
    with pytest.raises(ValueError):
        bridge_owner.author_current_preview_successor_bridge(
            **{**_author_arguments(fixture), target: rows})


@pytest.mark.parametrize("name", [member.name for member in fields(PlanAuthorityRoots)])
@pytest.mark.parametrize("phase", ("entry", "closing"))
def test_every_live_authority_root_is_checked_independently_at_entry_and_close(bridge_case, name, phase):
    fixture = bridge_case
    change = lambda: fixture["state"].update(roots=_different_root(fixture["state"]["roots"], name))
    if phase == "entry":
        change()
    else:
        fixture["case"]["close_callback"] = change
    with pytest.raises(ValueError):
        _verify(fixture)


@pytest.mark.parametrize("phase", ("entry", "closing", "resource_exit"))
def test_checker_policy_drift_withholds_a_current_bridge(bridge_case, phase):
    fixture = bridge_case
    change = lambda: fixture["state"].update(checker=_cid("changed-checker-policy"))
    if phase == "entry":
        change()
    else:
        fixture["case"]["close_callback" if phase == "closing" else "exit_callback"] = change
    with pytest.raises(ValueError):
        _verify(fixture)


@pytest.mark.parametrize("change", ("intent_polarity", "candidate_policy", "task_metadata", "extra_policy",
                                   "frozen_policy", "task_dependency", "predicate_population"))
def test_full_semantic_inputs_remain_bound_beyond_the_claimed_obligation_graph(bridge_case, change):
    fixture = bridge_case
    materials = replace(fixture["materials"])
    if change == "intent_polarity":
        predicates = list(materials.intent.desired_predicates)
        predicates[0] = replace(predicates[0], polarity=PredicatePolarity.NEGATIVE)
        materials.intent = replace(materials.intent, desired_predicates=tuple(predicates))
    elif change == "candidate_policy":
        materials.candidate_context = {**materials.candidate_context, "policy": {"require_proof": True}}
    elif change == "task_metadata":
        materials.candidate_context = deepcopy(materials.candidate_context)
        first = materials.task_candidates[0].candidate_id
        materials.candidate_context["task_metadata"][first]["predicted_files"] = ["keep.py"]
    elif change == "extra_policy":
        materials.extra = {**materials.extra, "policy": {"allowed_operations": ["create"]}}
    elif change == "frozen_policy":
        materials.frozen_goal = replace(materials.frozen_goal,
            policy=replace(materials.frozen_goal.policy, require_validation=False))
    elif change == "task_dependency":
        materials.task_candidates = (materials.task_candidates[0],
            replace(materials.task_candidates[1], depends_on_candidate_ids=()))
    else:
        materials.predicates = (TypedPredicate("predicate:additional", "reviewed_goal", "additional"),)
    with pytest.raises(ValueError):
        _verify(fixture, materials=materials)


@pytest.mark.parametrize("location", ("extra", "candidate_context", "query_plan", "workflow_request"))
def test_structural_material_refusal_does_not_read_arbitrary_dataclass_fields(bridge_case, location):
    fixture, touches = bridge_case, []

    @dataclass
    class HookedRecord:
        value: str

        def __getattribute__(self, name):
            if name == "value":
                touches.append(name)
            return object.__getattribute__(self, name)

    record = HookedRecord("independent hooked field")
    materials = replace(fixture["materials"])
    if location in {"extra", "candidate_context"}:
        setattr(materials, location, {**getattr(materials, location), "nested": record})
    else:
        setattr(materials, location, record)
    with pytest.raises(ValueError):
        contract.bind_repository_preview_materials(materials, fixture["context"])
    assert touches == []


@pytest.mark.parametrize("kind,location", (("dict", "extra"), ("list", "candidate_context")))
def test_structural_material_refusal_does_not_iterate_custom_metadata_containers(bridge_case, kind, location):
    fixture, touches = bridge_case, []

    class HookedDict(dict):
        def items(self):
            touches.append("items")
            return super().items()

        def __iter__(self):
            touches.append("iter")
            return super().__iter__()

        def __len__(self):
            touches.append("len")
            return super().__len__()

    class HookedList(list):
        def __iter__(self):
            touches.append("iter")
            return super().__iter__()

        def __len__(self):
            touches.append("len")
            return super().__len__()

    item = HookedDict({"policy": "independent"}) if kind == "dict" else HookedList(["independent"])
    materials = replace(fixture["materials"], **{
        location: {**getattr(fixture["materials"], location), "nested": item}})
    with pytest.raises(ValueError):
        contract.bind_repository_preview_materials(materials, fixture["context"])
    assert touches == []


def test_structural_material_refusal_does_not_call_a_replaced_typed_record_serializer(bridge_case):
    fixture, touches = bridge_case, []
    goal = replace(fixture["materials"].frozen_goal)
    original = goal.to_dict

    def hooked():
        touches.append("to_dict")
        return original()

    object.__setattr__(goal, "to_dict", hooked)
    materials = replace(fixture["materials"], frozen_goal=goal)
    with pytest.raises(ValueError):
        contract.bind_repository_preview_materials(materials, fixture["context"])
    assert touches == []


def test_passive_finite_ranking_metadata_remains_supported_and_binds_its_full_value(bridge_case):
    fixture = bridge_case
    snapshots = []
    for score in (0.25, 0.75):
        materials = replace(fixture["materials"], extra={**fixture["materials"].extra,
            "ranking": {"score": score, "rank": 2}})
        bound = contract.bind_repository_preview_materials(materials, fixture["context"])
        snapshots.append(freeze_plan_create_input_snapshot(fixture["request"], materials=bound))
    first, second = snapshots
    assert first.material_binding["reuse_supported"] is second.material_binding["reuse_supported"] is True
    assert first.request_cid == second.request_cid and first.roots == second.roots
    assert first.snapshot_cid != second.snapshot_cid
    assert first.material_binding["field_digests"]["extra"] != second.material_binding["field_digests"]["extra"]


@pytest.mark.parametrize("change", ("missing_task", "dependency"))
def test_even_genuine_signed_successor_contracts_must_match_the_entire_preview(bridge_case, change):
    fixture = bridge_case
    tasks = list(fixture["graph"].tasks)
    dependent_position = next(position for position, task in enumerate(tasks) if task.dependency_task_cids)
    if change == "missing_task":
        tasks.pop(dependent_position)
    else:
        tasks[dependent_position] = replace(tasks[dependent_position], dependency_task_cids=())
    graph = replace(fixture["graph"], tasks=tuple(tasks))
    _, admission = _author_graph_admission(fixture["case"], graph)
    assert len(local.verify_local_benchmark_admission(admission)["graph"].tasks) == len(tasks)
    with pytest.raises(ValueError):
        _verify(fixture, admission=admission)


@pytest.mark.parametrize("field", ("program_root", "repository_root_cid"))
def test_current_structural_identity_is_not_replaced_by_a_static_policy_observation(bridge_case, field):
    fixture = bridge_case
    context = fixture["context"]
    if field == "program_root":
        fixture["state"]["structural_context"] = replace(context, semantic_state_cid=_cid("other-semantics"))
    else:
        changed = _cid("other-source-snapshot")
        fixture["state"]["structural_context"] = replace(context,
            head=replace(context.head, snapshot_cid=changed,
                ast_revision_id=f"rev:{context.head.repository_id}:snapshot:{changed}"))
    with pytest.raises(ValueError):
        _verify(fixture)


@pytest.mark.parametrize("phase", ("before_exit", "after_exit"))
def test_live_roots_are_reobserved_through_successor_resource_closure(bridge_case, phase):
    fixture = bridge_case
    change = lambda: fixture["state"].update(
        roots=_different_root(fixture["state"]["roots"], "policy_root"))
    if phase == "before_exit":
        fixture["case"]["close_callback"] = change
    else:
        fixture["case"]["exit_callback"] = change
    with pytest.raises(ValueError):
        _verify(fixture)


def test_application_observer_source_edit_after_native_close_requires_another_native_fence(bridge_case):
    fixture = bridge_case
    case, edited = fixture["case"], []
    manifest = fixture["manifest"]["payload"]

    def native_source_fence():
        # Explicit current-source receiving protocol double. The comparison
        # reads the real fixture Git checkout; no source/model owner is opened.
        current = local._sources(case["repository"], sorted(manifest["sources"]))
        if current != manifest["sources"]:
            raise ValueError("independent native source fence observed callback source edit")

    def edit_after_consumed_native_fence():
        if "close" in case["observations"] and not edited:
            edited.append(True)
            (case["repository"] / "input.py").write_bytes(b"VALUE = 9001\n")

    case["close_callback"] = native_source_fence
    fixture["state"]["roots_callback"] = edit_after_consumed_native_fence
    with pytest.raises(ValueError):
        _verify(fixture)
    assert edited == [True]


@pytest.fixture
def captured_operation(bridge_case, monkeypatch):
    fixture = bridge_case
    capture = {"receipts": [], "bridges": []}
    original = bridge_owner._wire

    def observe(value, **options):
        if type(value) is dict:
            if value.get("schema") == "supervisor-current-preview-successor-bridge-verification@1":
                if all(row is not value for row in capture["receipts"]):
                    capture["receipts"].append(value)
            elif set(value) == {"payload", "binding"} and "bridge_cid" in value.get("payload", {}):
                if all(row is not value for row in capture["bridges"]):
                    capture["bridges"].append(value)
        return original(value, **options)

    monkeypatch.setattr(bridge_owner, "_wire", observe)
    return fixture, capture


@pytest.mark.parametrize("target", ("receipt", "signed_bridge"))
@pytest.mark.parametrize("phase", ("inside_closing", "after_closing", "resource_exit"))
def test_private_custody_refuses_late_receipt_or_signed_bridge_changes(captured_operation, target, phase):
    fixture, capture = captured_operation
    caller = deepcopy(fixture["bridge"])

    def change():
        if target == "receipt":
            capture["receipts"][-1]["administrator_task_cids"].pop()
        else:
            # The test mutates the operation's detached copy, not the caller.
            detached = next(row for row in reversed(capture["bridges"]) if row is not fixture["bridge"])
            detached["payload"]["checker_policy_cid"] = _cid("late-private-bridge-change")

    if phase == "inside_closing":
        fixture["case"]["close_callback"] = change
    elif phase == "resource_exit":
        fixture["case"]["exit_callback"] = change
    with pytest.raises(ValueError):
        with bridge_owner._current_preview_successor_bridge_operation(
                **_verify_arguments(fixture)) as (_, close):
            close()
            if phase == "after_closing":
                change()
    assert fixture["bridge"] == caller


def test_private_operation_requires_a_successful_one_use_close(bridge_case):
    fixture = bridge_case
    with pytest.raises(ValueError, match="closing"):
        with bridge_owner._current_preview_successor_bridge_operation(**_verify_arguments(fixture)):
            pass
    fixture["case"]["observations"].clear()
    with bridge_owner._current_preview_successor_bridge_operation(
            **_verify_arguments(fixture)) as (receipt, close):
        assert close() is receipt
        with pytest.raises(ValueError, match="closed"):
            close()
    assert fixture["case"]["observations"] == ["entry", "close", "exit"] * 2


@pytest.mark.parametrize("phase", ("body", "closing"))
def test_primary_errors_are_preserved_without_publishing_a_successful_bridge(bridge_case, phase):
    fixture = bridge_case
    error = RuntimeError("independent bridge primary failure")
    if phase == "closing":
        def fail():
            raise error
        fixture["case"]["close_callback"] = fail
    with pytest.raises(RuntimeError) as caught:
        with bridge_owner._current_preview_successor_bridge_operation(
                **_verify_arguments(fixture)) as (_, close):
            if phase == "body":
                raise error
            close()
    assert caught.value is error


def test_operation_keeps_caller_bridge_and_admission_detached(bridge_case):
    fixture = bridge_case
    original_bridge, original_admission = deepcopy(fixture["bridge"]), deepcopy(fixture["admission"])
    with bridge_owner._current_preview_successor_bridge_operation(
            **_verify_arguments(fixture)) as (receipt, close):
        fixture["bridge"]["payload"]["checker_policy_cid"] = _cid("caller-change")
        fixture["admission"]["graph"]["tasks"].pop()
        close()
        assert len(receipt["administrator_task_cids"]) == 2
    fixture["bridge"].clear()
    fixture["bridge"].update(original_bridge)
    fixture["admission"].clear()
    fixture["admission"].update(original_admission)


def test_private_operation_rejects_a_changed_process_identity(bridge_case, monkeypatch):
    fixture = bridge_case
    process_id = bridge_owner.os.getpid()
    with pytest.raises(ValueError):
        with monkeypatch.context() as patch:
            with bridge_owner._current_preview_successor_bridge_operation(
                    **_verify_arguments(fixture)) as (_, close):
                # An explicit custody guard control; no actual fork or native
                # process owner is created, and pytest's process view restores.
                patch.setattr(bridge_owner.os, "getpid", lambda: process_id + 1)
                close()


@pytest.mark.parametrize("phase", ("before_entry", "during_roots", "resource_exit"))
def test_cancellation_withholds_a_result_even_after_successor_closing(bridge_case, phase):
    fixture, cancelled = bridge_case, threading.Event()
    if phase == "before_entry":
        cancelled.set()
    elif phase == "during_roots":
        fixture["state"]["roots_callback"] = cancelled.set
    else:
        fixture["case"]["exit_callback"] = cancelled.set
    with pytest.raises(LeaseCancelledError):
        _verify(fixture, cancel_event=cancelled)


def test_unsigned_bridge_change_cannot_borrow_the_original_owner_signature(bridge_case):
    fixture = bridge_case
    bridge = deepcopy(fixture["bridge"])
    bridge["payload"]["checker_policy_cid"] = _cid("unsigned-other-checker")
    with pytest.raises(ValueError):
        _verify(fixture, bridge=bridge)


@pytest.mark.parametrize("change", ("root_projection", "model_lineage", "training_record",
                                   "task_population", "requirement_acceptance", "authority"))
def test_genuine_owner_resigning_cannot_authorize_a_different_joined_contract(bridge_case, change):
    fixture = bridge_case
    payload = deepcopy(fixture["bridge"]["payload"])
    if change == "root_projection":
        payload["root_mapping"]["workflow_request_cid"] = _cid("different-workflow-request")
    elif change == "model_lineage":
        payload["successor_declaration"]["model"]["state_sha256"] = "e" * 64
        payload["successor_declaration_cid"] = content_identity(payload["successor_declaration"])
    elif change == "training_record":
        payload["training_record_cid"] = _cid("different-training-record")
    elif change == "task_population":
        payload["population"]["signed_tasks"] = 1
    elif change == "requirement_acceptance":
        payload["requirement_bindings"][0]["acceptance_cids"] = [_cid("different-actual-acceptance")]
    else:
        payload["authority"][next(iter(payload["authority"]))] = True
    payload["bridge_cid"] = content_identity({key: value for key, value in payload.items() if key != "bridge_cid"})
    resigned = local._signed(payload, fixture["manifest"]["payload"])
    profile = local.verify_local_benchmark_admission(fixture["admission"])["profile"]
    assert local._verify_signature(resigned, profile) == payload
    with pytest.raises(ValueError):
        _verify(fixture, bridge=resigned)


def test_legacy_admission_verifier_cannot_consume_the_companion_as_task_authority(bridge_case):
    with pytest.raises(ValueError):
        local.verify_local_benchmark_admission(bridge_case["bridge"])
