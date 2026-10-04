"""Native finite evidence joins signed local tasks without omission or proof upgrade."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
from ipfs_accelerate_py.agent_supervisor.control.profile_authority import (
    revoke_local_profile, sign_profile_binding,
)
from ipfs_accelerate_py.agent_supervisor.planning import finite_integer_codebase as matcher
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    PromptAcceptanceRecord, PromptGoalGraph, PromptGoalRecord, PromptOutputRecord,
    PromptTaskRecord, PromptValidationRecord,
)
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository, IntentCompletionError
from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex, StaleCodebaseError
from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceSchedulerConfig,
)
from test.api.test_finite_integer_codebase import finite_tools, finite_source, finite_text, finite_git
from test.api.test_finite_integer_plan_preview import preview_arguments


def _module():
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_admission
    return finite_repository_admission


def _build_case(root, tools, *, offset=1):
    root.mkdir()
    repository = root / "repository"
    repository.mkdir()
    (repository / "calc.py").write_bytes(finite_source(offset))
    (repository / "test_type.py").write_text(
        "from calc import increment\nassert all(type(increment(n)) is int for n in [-2,-1,0,1,2])\n")
    (repository / "test_offset.py").write_text(
        "from calc import increment\nassert all(increment(n) == n + 2 for n in [-2,-1,0,1,2])\n")
    for args in (("init", "-q"), ("config", "user.name", "Finite admission fixture"),
                 ("config", "user.email", "fixture@example.invalid"), ("add", "."),
                 ("commit", "-qm", "authored finite checks")):
        finite_git(repository, *args)
    profile, lifecycle = root / "profile", root / "lifecycle"
    Supervisor.init_local(repository=repository, consent=True, profile_dir=profile,
                          lifecycle_dir=lifecycle)
    connection = duckdb.connect(str(root / "source.duckdb"),
        config={"threads": 1, "memory_limit": "64MB"})
    store = DuckDBASTStore(connection=connection)
    artifacts = ImmutableCAS(root / "artifacts")
    index = RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store), artifacts=artifacts,
                                    catalog=CodebaseCatalog(store, artifacts))
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=root / "admission.json", lane_reservations={}, auto_renew_leases=False))
    head = index.prepare_current(repository, repository_id="fixture:finite-signed-admission",
        operation_id="initial", expected_head=None, scheduler=scheduler).head
    case = dict(root=root, repository=repository, index=index, scheduler=scheduler,
                connection=connection, expected_head=head, output=root / "unused-preview",
                profile=profile, lifecycle=lifecycle)
    arguments = preview_arguments(case, tools)
    owner, request = arguments["owner"], arguments["request"]
    policy = content_identity(local.LOCAL_POLICY)
    validations = tuple(PromptValidationRecord(validation_key=key,
        argv=(sys.executable, path), policy_cid=policy)
        for key, path in (("public-type", "test_type.py"), ("public-offset", "test_offset.py")))
    acceptance = tuple(PromptAcceptanceRecord(criterion_key=key, criterion=criterion,
        validation_keys=(validation.validation_key,))
        for key, criterion, validation in zip(("exact-type", "desired-offset"),
            ("All listed outputs are exact integers", "All listed outputs equal n + 2"), validations))
    output = PromptOutputRecord(path="calc.py", effect="modify", media_type="text/x-python")
    goal = PromptGoalRecord(goal_key="FINITE-GOAL", parent_goal_cid="", dependency_goal_cids=(),
        title="Repair the finite offset", objective=finite_text(), rationale="Authored exact public checks",
        scope_paths=("calc.py", "test_offset.py", "test_type.py"), acceptance=acceptance)
    type_task = PromptTaskRecord(task_key="TYPE-TASK", goal_cid=goal.goal_cid,
        dependency_task_cids=(), objective="Retain exact integer outputs", rationale="Explicit administrator task",
        scope_paths=("calc.py",), outputs=(output,), validations=(validations[0],),
        acceptance=(acceptance[0],), evidence_cids=(), policy_roots=(policy,), predicted_files=("calc.py",))
    offset_task = PromptTaskRecord(task_key="OFFSET-TASK", goal_cid=goal.goal_cid,
        dependency_task_cids=(type_task.task_cid,), objective="Return n + 2 on the five declared inputs",
        rationale="Explicit administrator task", scope_paths=("calc.py",), outputs=(output,),
        validations=(validations[1],), acceptance=(acceptance[1],), evidence_cids=(),
        policy_roots=(policy,), predicted_files=("calc.py",))
    roots = dict(request_cid=request.request_cid, program_root=request.roots.program_root,
        scan_cid=content_identity({"sources": local._sources(repository, ["calc.py", "test_type.py", "test_offset.py"])}))
    graph = PromptGoalGraph(**roots, policy_roots=(policy,), goals=(goal,),
                           tasks=(type_task, offset_task), evidence=())
    specs = []
    for task in graph.tasks:
        specs.append(dict(task_key=task.task_key, scope_paths=list(task.scope_paths),
            dependencies=["TYPE-TASK"] if task.task_key == "OFFSET-TASK" else [],
            outputs=[dict(path=output.path, effect=output.effect, media_type=output.media_type)],
            validations=[{name: local._plain(getattr(validation, name)) for name in
                ("validation_key", "argv", "cwd", "expected_exit_codes", "policy_cid")}
                for validation in task.validations],
            acceptance=[{name: local._plain(getattr(row, name)) for name in
                ("criterion_key", "criterion", "evidence_cids", "validation_keys")}
                for row in task.acceptance]))
    manifest = local.author_local_benchmark_manifest(repository=repository, profile_dir=profile,
        lifecycle_dir=lifecycle, task_specs=specs, planning_roots=roots)
    bindings = {matcher.TYPE_STATEMENT_ID: "TYPE-TASK", matcher.OFFSET_STATEMENT_ID: "OFFSET-TASK"}
    case.update(arguments=arguments, owner=owner, request=request, graph=graph, manifest=manifest,
                task_bindings=bindings, tools=deepcopy(tools))
    return case


@pytest.fixture(scope="module")
def signed_case(tmp_path_factory, finite_tools):
    root = tmp_path_factory.mktemp("finite-repository-admission") / "native"
    case = _build_case(root, finite_tools)
    try:
        case["declaration"] = _author(case)
        case["admission"] = _admit(case, output=root / "signed-evidence")
        yield case
    finally:
        state = case["scheduler"].snapshot()
        assert state["active_lease_count"] == state["waiting_request_count"] == 0
        case["connection"].close()


def _author(case, **changes):
    arguments = case["arguments"]
    kwargs = dict(owner=case["owner"], manifest=case["manifest"], request=case["request"],
        intent_document=arguments["intent_document"], source_text=arguments["source_text"],
        operation_catalog=arguments["operation_catalog"], tool_policy=case["tools"],
        task_bindings=case["task_bindings"])
    kwargs.update(changes)
    return _module().author_finite_repository_declaration(**kwargs)


def _admit(case, *, output, **changes):
    kwargs = dict(owner=case["owner"], declaration=case["declaration"], graph=case["graph"],
                  output=output, policy_observer=lambda request: request.roots)
    kwargs.update(changes)
    return _module().admit_finite_repository_plan(**kwargs)


def _rows(intent):
    with intent._connection() as connection:
        return {table: connection.execute("SELECT count(*) FROM " + table).fetchone()[0]
                for table in ("objectives", "tasks", "plans", "goals")}


def _resign(case, payload):
    """A genuine owner signature must not relax the closed admission policy."""
    return {"payload": payload, "binding": sign_profile_binding(
        profile_dir=case["profile"], lifecycle_dir=case["lifecycle"], payload=payload)}


def test_native_signed_evidence_preserves_administrator_population_and_pending_authority(signed_case):
    case, module = signed_case, _module()
    admission = case["admission"]
    verified = module.verify_finite_repository_admission(admission=admission)
    assert verified["observed_current"] is False
    assert admission["receipt"]["payload"]["planning_permitted"] is True
    assert admission["local_admission"]["graph"] == case["graph"].to_dict()
    local_verified = local.verify_local_benchmark_admission(admission["local_admission"])
    assert {task["task_key"] for task in local_verified["manifest"]["tasks"]} == {"TYPE-TASK", "OFFSET-TASK"}
    for flag in ("proof_authority", "completion_authority", "omission_authority"):
        assert admission["receipt"]["payload"][flag] is False
    assert case["scheduler"].snapshot()["active_lease_count"] == 0


def test_native_materialization_retains_both_original_task_guards(signed_case, tmp_path):
    case, module = signed_case, _module()
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        result = module.materialize_finite_repository_plan(owner=case["owner"],
            admission=case["admission"], intent=intent, output=tmp_path / "materialization-evidence",
            policy_observer=lambda request: request.roots)
        assert set(result["task_cids"]) == {task.task_cid for task in case["graph"].tasks}
        assert _rows(intent) == {"objectives": 1, "tasks": 2, "plans": 1, "goals": 1}
        for task in case["graph"].tasks:
            row = intent.get_task(task.task_cid)
            assert row["status"] == "ready"
            assert local.CONTRACT_KEY in row["body"]
            changed = deepcopy(row["body"])
            changed.pop(local.CONTRACT_KEY)
            with pytest.raises(local.LocalPlanningError):
                intent.upsert_task(task_cid=task.task_cid, task_alias=row["task_alias"],
                    goal_cid=row["goal_cid"], body=changed, identity=row["identity"],
                    expected_revision=row["revision"])
            with pytest.raises((IntentCompletionError, local.LocalPlanningError)):
                intent.cas_task_status(task_cid=task.task_cid, expected_revision=row["revision"],
                                      new_status="completed", allow_completion_without_evidence=True)
            assert intent.get_task(task.task_cid)["status"] == "ready"


@pytest.mark.parametrize("mutation", ["prompt", "ir", "catalog", "tools", "task_partition", "head", "model_budget"])
def test_authoring_refuses_malformed_or_rebound_finite_inputs(signed_case, mutation):
    case = signed_case
    changes = {}
    if mutation == "prompt":
        changes["source_text"] = finite_text(offset=3)
    elif mutation == "ir":
        changes["intent_document"] = matcher.build_finite_integer_intent(finite_text(offset=3))
    elif mutation == "catalog":
        catalog = case["arguments"]["operation_catalog"]
        changes["operation_catalog"] = replace(catalog, operations=tuple(
            replace(row, review_ref="review:changed") for row in catalog.operations))
    elif mutation == "tools":
        tools = deepcopy(case["tools"])
        tools["python"]["sha256"] = "0" * 64
        changes["tool_policy"] = tools
    elif mutation == "task_partition":
        changes["task_bindings"] = {key: "TYPE-TASK" for key in case["task_bindings"]}
    elif mutation == "head":
        changes["owner"] = replace(case["owner"], expected_head=replace(case["expected_head"],
            generation=case["expected_head"].generation + 1))
    else:
        changes["request"] = replace(case["request"],
            budget=replace(case["request"].budget, max_model_calls=1))
    with pytest.raises((ValueError, OSError)):
        _author(case, **changes)


@pytest.mark.parametrize("member", ["declaration", "receipt", "local_admission"])
def test_historical_verifier_rejects_untrusted_signature_or_population_tampering(signed_case, member):
    forged = deepcopy(signed_case["admission"])
    if member == "declaration":
        forged[member]["payload"]["task_bindings"][matcher.OFFSET_STATEMENT_ID] = "TYPE-TASK"
    elif member == "receipt":
        forged[member]["payload"]["planning_permitted"] = False
    else:
        forged[member]["graph"]["tasks"] = forged[member]["graph"]["tasks"][:1]
    with pytest.raises(ValueError):
        _module().verify_finite_repository_admission(admission=forged)


@pytest.mark.parametrize("path,value", [
    (("proof_authority",), 0),
    (("planning_permitted",), 1),
    (("policy", "finite_facts_are_context_only"), 1),
    (("policy", "proof_authority"), 0),
    (("semantic_context", "task_population_preserved"), 1),
    (("semantic_context", "proof_authority"), 0),
])
def test_owner_signature_cannot_alias_receipt_booleans_to_numbers(signed_case, path, value):
    forged = deepcopy(signed_case["admission"])
    payload = forged["receipt"]["payload"]
    member = payload
    for key in path[:-1]:
        member = member[key]
    member[path[-1]] = value
    forged["receipt"] = _resign(signed_case, payload)
    with pytest.raises(ValueError):
        _module().verify_finite_repository_admission(admission=forged)


@pytest.mark.parametrize("flag,value", [
    ("proof_authority", 0), ("finite_facts_are_context_only", 1),
])
def test_owner_signature_cannot_alias_declaration_policy_booleans(signed_case, flag, value):
    forged = deepcopy(signed_case["admission"])
    payload = forged["declaration"]["payload"]
    payload["policy"][flag] = value
    forged["declaration"] = _resign(signed_case, payload)
    with pytest.raises(ValueError):
        _module().verify_finite_repository_admission(admission=forged)


def test_owner_resigned_rehashed_evidence_cannot_alias_model_off_feature_authority(signed_case):
    forged = deepcopy(signed_case["admission"])
    evidence = forged["evidence"]
    evidence["feature_context"]["proof_authority"] = 0
    evidence["result_cid"] = cid_for_structured({key: value for key, value in evidence.items()
                                               if key != "result_cid"})
    payload = forged["receipt"]["payload"]
    payload["evidence_cid"] = cid_for_structured(evidence)
    forged["receipt"] = _resign(signed_case, payload)
    with pytest.raises(ValueError):
        _module().verify_finite_repository_admission(admission=forged)


@pytest.mark.parametrize("mutation", [
    "authority_zero", "authority_true", "mode", "unknown_key", "verification_points", "final_order",
])
def test_owner_resigned_rehashed_evidence_cannot_rebind_feature_validation_policy(signed_case, mutation):
    forged = deepcopy(signed_case["admission"])
    evidence = forged["evidence"]
    policy = evidence["feature_validation_policy"]
    if mutation == "authority_zero":
        policy["proof_authority"] = 0
    elif mutation == "authority_true":
        policy["proof_authority"] = True
    elif mutation == "mode":
        policy["mode"] = "frozen"
    elif mutation == "unknown_key":
        policy["allow_unverified_reuse"] = True
    elif mutation == "verification_points":
        policy["numerical_verification_points"] = ["caller_supplied"]
    else:
        policy["final_order"] = "detached_byte_closure_then_native_verification"
    evidence["result_cid"] = cid_for_structured({key: value for key, value in evidence.items()
                                               if key != "result_cid"})
    payload = forged["receipt"]["payload"]
    payload["evidence_cid"] = cid_for_structured(evidence)
    forged["receipt"] = _resign(signed_case, payload)
    with pytest.raises(_module().FiniteRepositoryAdmissionError,
                       match="complete model-off finite evidence or exact authority booleans differ"):
        _module().verify_finite_repository_admission(admission=forged)


def test_owner_resigned_rehashed_unknown_capacity_schema_cannot_auto_upgrade_profile(signed_case):
    forged = deepcopy(signed_case["admission"])
    evidence = forged["evidence"]
    evidence["schema"] = "finite-integer-capacity-bound-plan-preview@3"
    evidence["result_cid"] = cid_for_structured({key: value for key, value in evidence.items()
                                               if key != "result_cid"})
    payload = forged["receipt"]["payload"]
    payload["evidence_cid"] = cid_for_structured(evidence)
    forged["receipt"] = _resign(signed_case, payload)
    with pytest.raises(_module().FiniteRepositoryAdmissionError,
                       match="complete model-off finite evidence or exact authority booleans differ"):
        _module().verify_finite_repository_admission(admission=forged)


def test_retained_observation_artifact_cannot_be_replaced_under_valid_signatures(signed_case):
    admission = signed_case["admission"]
    path = Path(admission["evidence"]["match"]["observation"]["artifacts"]["source"]["path"])
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    try:
        with pytest.raises(ValueError):
            _module().verify_finite_repository_admission(admission=admission)
    finally:
        path.write_bytes(original)


def test_signed_admission_is_historical_after_dirty_edit_and_current_recheck_refuses(signed_case, tmp_path):
    case, module = signed_case, _module()
    path = case["repository"] / "calc.py"
    original = path.read_bytes()
    path.write_bytes(finite_source(9))
    try:
        assert module.verify_finite_repository_admission(admission=case["admission"])["observed_current"] is False
        with pytest.raises((ValueError, StaleCodebaseError)):
            module.verify_current_finite_repository_admission(owner=case["owner"],
                admission=case["admission"], output=tmp_path / "stale-current",
                policy_observer=lambda request: request.roots)
    finally:
        path.write_bytes(original)


def test_genuine_fresh_current_verification_does_not_materialize_or_fit(signed_case, tmp_path):
    result = _module().verify_current_finite_repository_admission(owner=signed_case["owner"],
        admission=signed_case["admission"], output=tmp_path / "fresh-current",
        policy_observer=lambda request: request.roots)
    assert result["observed_current"] is True
    assert signed_case["scheduler"].snapshot()["active_lease_count"] == 0


def test_dropped_original_task_cannot_borrow_valid_finite_evidence(signed_case, tmp_path):
    case = signed_case
    graph = replace(case["graph"], tasks=(case["graph"].tasks[0],))
    with pytest.raises(ValueError):
        _admit(case, output=tmp_path / "omitted-task", graph=graph)


def test_changed_policy_observation_cannot_publish_admission_or_native_rows(signed_case, tmp_path):
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        before = _rows(intent)
        def drift(request):
            return replace(request.roots, policy_root=content_identity({"changed": "policy"}))
        with pytest.raises(ValueError):
            _module().materialize_finite_repository_plan(owner=signed_case["owner"],
                admission=signed_case["admission"], intent=intent, output=tmp_path / "changed-policy",
                policy_observer=drift)
        assert _rows(intent) == before


@pytest.mark.parametrize("mutation", ["source", "retained_evidence"])
def test_late_source_or_evidence_edit_rolls_back_native_materialization(signed_case, tmp_path, monkeypatch, mutation):
    case = signed_case
    path = (case["repository"] / "calc.py" if mutation == "source" else Path(
        case["admission"]["evidence"]["match"]["observation"]["artifacts"]["source"]["path"]))
    changed = []
    original_bytes = path.read_bytes()
    original_materialize = local._materialize_local_benchmark_plan
    def edit_before_owner_commit(*args, **kwargs):
        result = original_materialize(*args, **kwargs)
        changed.append(_rows(kwargs["intent"]))
        path.write_bytes(finite_source(9) if mutation == "source" else original_bytes + b"\n")
        return result
    monkeypatch.setattr(local, "_materialize_local_benchmark_plan", edit_before_owner_commit)
    try:
        with IntentRepository(tmp_path / "intent.duckdb") as intent:
            before = _rows(intent)
            with pytest.raises(ValueError):
                _module().materialize_finite_repository_plan(owner=case["owner"],
                    admission=case["admission"], intent=intent, output=tmp_path / "late-edit",
                    policy_observer=lambda request: request.roots)
            assert changed == [{"objectives": 1, "tasks": 2, "plans": 1, "goals": 1}]
            assert _rows(intent) == before
    finally:
        path.write_bytes(original_bytes)


def test_source_edit_during_real_owner_signing_cannot_publish_declaration(signed_case, monkeypatch):
    case, module = signed_case, _module()
    path, changed = case["repository"] / "calc.py", []
    original_bytes, signer = path.read_bytes(), local._signed
    def mutate_while_signing(payload, manifest):
        if payload["schema"] == module.DECLARATION_SCHEMA:
            path.write_bytes(finite_source(9))
            changed.append(True)
        return signer(payload, manifest)
    monkeypatch.setattr(local, "_signed", mutate_while_signing)
    try:
        with pytest.raises((ValueError, StaleCodebaseError)):
            _author(case)
        assert changed
    finally:
        path.write_bytes(original_bytes)


def test_revoked_active_profile_cannot_replay_or_regrant_signed_finite_admission(tmp_path, finite_tools):
    case = _build_case(tmp_path / "revoked", finite_tools)
    try:
        case["declaration"] = _author(case)
        admission = _admit(case, output=tmp_path / "before-revocation")
        assert _module().verify_finite_repository_admission(admission=admission)["observed_current"] is False
        revoke_local_profile(profile_dir=case["profile"], lifecycle_dir=case["lifecycle"])
        with pytest.raises(ValueError):
            _module().verify_finite_repository_admission(admission=admission)
        with pytest.raises(ValueError):
            _module().verify_current_finite_repository_admission(owner=case["owner"],
                admission=admission, output=tmp_path / "after-revocation",
                policy_observer=lambda request: request.roots)
    finally:
        case["connection"].close()
        assert case["scheduler"].snapshot()["active_lease_count"] == 0


def test_already_complete_source_retains_full_signed_graph_and_cannot_materialize_empty_tasks(tmp_path, finite_tools):
    case = _build_case(tmp_path / "complete", finite_tools, offset=2)
    try:
        case["declaration"] = _author(case)
        admission = _admit(case, output=tmp_path / "no-work")
        assert admission["receipt"]["payload"]["planning_permitted"] is False
        assert admission["local_admission"] is None
        assert admission["graph"] == case["graph"].to_dict()
        assert len(admission["graph"]["tasks"]) == 2
        assert _module().verify_finite_repository_admission(admission=admission)["observed_current"] is False
        changed_task = replace(case["graph"].tasks[1], acceptance=(replace(
            case["graph"].tasks[1].acceptance[0], criterion="Accept any returned value"),))
        changed_goal = replace(case["graph"].goals[0], acceptance=(
            case["graph"].goals[0].acceptance[0], changed_task.acceptance[0]))
        changed_type_task = replace(case["graph"].tasks[0], goal_cid=changed_goal.goal_cid)
        changed_task = replace(changed_task, goal_cid=changed_goal.goal_cid,
                               dependency_task_cids=(changed_type_task.task_cid,))
        changed_graph = replace(case["graph"], goals=(changed_goal,),
                                tasks=(changed_type_task, changed_task))
        with pytest.raises(ValueError):
            _admit(case, graph=changed_graph, output=tmp_path / "no-work-changed-acceptance")
        with IntentRepository(tmp_path / "empty-intent.duckdb") as intent:
            before = _rows(intent)
            with pytest.raises(ValueError):
                _module().materialize_finite_repository_plan(owner=case["owner"], admission=admission,
                    intent=intent, output=tmp_path / "not-empty-materialization",
                    policy_observer=lambda request: request.roots)
            assert _rows(intent) == before
    finally:
        case["connection"].close()
        state = case["scheduler"].snapshot()
        assert state["active_lease_count"] == state["waiting_request_count"] == 0
