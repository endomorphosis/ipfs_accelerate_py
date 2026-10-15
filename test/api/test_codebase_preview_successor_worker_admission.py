"""Genuine signed child admission and native Intent transactions for preview joins.

Git, signatures, complete task graph replay, private receipt storage and native
Intent transactions are genuine. Current source/model/checker observations use
the explicit bridge fixture doubles; these controls grant no native freshness,
numerical, worker execution, semantic proof or production authority.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.runtime import (
    codebase_preview_successor_worker_admission as joined,
    codebase_successor_dispatch_admission as successor,
    local_planning_admission as local,
    router_public_instruction as public,
    supervisor_meta_index as meta,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from test.api.test_codebase_preview_successor_bridge import (
    _author_graph_admission, _different_root, bridge_case,  # noqa: F401
)
from test.api.test_codebase_successor_dispatch_admission import (
    _git, prohibit_private_source_model_and_numerical_work, successor_case,  # noqa: F401
)
from test.api.test_plan_create_semantic_input_identity import _cid


def _current_arguments(fixture):
    return {name: fixture["case"][name] for name in (
        "selection", "root", "completion", "index", "repository", "registry")}


def _bridge_inputs(fixture):
    return joined.PreviewSuccessorBridgeInputs(
        request=fixture["request"], materials=fixture["materials"], preview=fixture["preview"],
        roots_observer=fixture["roots_observer"], checker_observer=fixture["checker_observer"])


def _author(fixture, **changes):
    return joined.author_current_preview_successor_worker_manifest(**{
        **_current_arguments(fixture), "parent_admission": fixture["admission"],
        "bridge": fixture["bridge"], "bridge_inputs": fixture["bridge_inputs"], **changes})


def _admit(fixture, **changes):
    return joined.admit_current_preview_successor_worker_plan(**{
        **_current_arguments(fixture), "manifest": fixture["worker_manifest"],
        "graph": fixture["graph"], "bridge_inputs": fixture["bridge_inputs"], **changes})


def _verify(fixture, **changes):
    return joined.verify_current_preview_successor_worker_admission(**{
        **_current_arguments(fixture), "worker_admission": fixture["worker_admission"],
        "bridge_inputs": fixture["bridge_inputs"], **changes})


def _install(fixture, **changes):
    return joined.materialize_current_preview_successor_worker_plan(**{
        **_current_arguments(fixture), "worker_admission": fixture["worker_admission"],
        "bridge_inputs": fixture["bridge_inputs"], "intent": fixture["case"]["intent"], **changes})


def _prepare(fixture, *, position=0, **changes):
    case = fixture["case"]
    return joined.prepare_current_preview_successor_worker_context(**{
        **_current_arguments(fixture), "worker_admission": fixture["worker_admission"],
        "bridge_inputs": fixture["bridge_inputs"],
        "task_cid": case["tasks"][position].task_cid, "source_path": "keep.py",
        "expected_source_sha256": hashlib.sha256((case["repository"] / "keep.py").read_bytes()).hexdigest(),
        **changes})


@pytest.fixture
def worker_case(bridge_case, monkeypatch):
    fixture = bridge_case
    fixture["bridge_inputs"] = _bridge_inputs(fixture)
    fixture["worker_manifest"] = _author(fixture)
    fixture["worker_admission"] = _admit(fixture)
    # Public preparation also retains the complete historical context. Reading
    # that CAS closure remains an explicit protocol double in this suite.
    def full_context(selection, root, completion, index, evidence_record):
        case = fixture["case"]
        assert selection is case["selection"] and root is case["root"]
        assert completion is case["completion"] and index is case["index"]
        assert evidence_record is None
        return deepcopy(case["context"])
    monkeypatch.setattr(successor, "_full_context", full_context)
    fixture["case"]["observations"].clear()
    fixture["state"]["events"].clear()
    return fixture


def _context(fixture):
    return fixture["worker_manifest"]["payload"]["codebase_preview_successor_context"]


def _binding_ids(fixture):
    context = _context(fixture)
    return {"codebase_preview_successor_context_cid": content_identity(context),
        **{key: context[key] for key in ("parent_admission_cid", "signed_bridge_cid", "bridge_cid")}}


def _assert_no_installed_population(fixture):
    case = fixture["case"]
    assert all(case["intent"].get_task(task.task_cid) is None for task in fixture["graph"].tasks)
    assert case["intent"].get_plan(fixture["worker_admission"]["receipt"]["payload"]["plan_id"]) is None


def test_child_manifest_binds_exact_signed_parent_and_companion_without_circular_identity(worker_case):
    fixture = worker_case
    verified = local.verify_local_benchmark_admission(fixture["worker_admission"])
    manifest, receipt = verified["manifest"], verified["receipt"]
    parent = fixture["manifest"]["payload"]
    context = _context(fixture)
    assert manifest["schema"] == local.PREVIEW_SUCCESSOR_MANIFEST_SCHEMA == "supervisor-local-benchmark-manifest@7"
    assert set(manifest) == set(parent) | {"codebase_preview_successor_context"}
    assert {key: value for key, value in manifest.items() if key not in {
        "schema", "codebase_preview_successor_context"}} == {
        key: value for key, value in parent.items() if key != "schema"}
    assert set(context) == {"schema", "parent_admission", "parent_admission_cid", "signed_bridge",
        "signed_bridge_cid", "bridge_cid"}
    assert context["parent_admission"] == fixture["admission"]
    assert context["parent_admission_cid"] == content_identity(fixture["admission"])
    assert context["signed_bridge"] == fixture["bridge"]
    assert context["signed_bridge_cid"] == content_identity(fixture["bridge"])
    assert context["bridge_cid"] == fixture["bridge"]["payload"]["bridge_cid"]
    assert context["signed_bridge"]["payload"]["admission_cid"] == context["parent_admission_cid"]
    assert content_identity(fixture["worker_admission"]) != context["parent_admission_cid"]
    assert receipt["schema"] == local.PREVIEW_SUCCESSOR_PLANNING_RECEIPT_SCHEMA == "supervisor-local-planning-receipt@5"
    assert {key: receipt[key] for key in _binding_ids(fixture)} == _binding_ids(fixture)
    assert verified["graph"].to_dict() == fixture["graph"].to_dict()
    assert receipt["administrator_task_cids"] == sorted(task.task_cid for task in fixture["graph"].tasks)
    assert receipt["current_facts"] == receipt["removed_task_cids"] == []
    assert receipt["runtime_requirements_preserved"] is True
    assert receipt["pending_requirements"] == fixture["admission"]["receipt"]["payload"]["pending_requirements"]
    assert all(row["phase"] == "post_execution" and row["required"] is True
        for row in receipt["pending_requirements"])
    assert all(receipt[key] is False for key in (
        "completion_authority", "code_proof_authority", "production_activation"))
    _assert_no_installed_population(fixture)


def test_materialization_keeps_every_task_validation_acceptance_and_versioned_pending_reference(worker_case):
    fixture = worker_case
    result = _install(fixture)
    case = fixture["case"]
    expected = sorted(task.task_cid for task in fixture["graph"].tasks)
    assert sorted(result["task_cids"]) == expected
    assert result["administrator_task_population_preserved"] is result["runtime_requirements_preserved"] is True
    assert result["current_facts"] == result["removed_task_cids"] == []
    reference = case["intent"].get_plan(result["plan_id"])["body"]["local_planning_receipt_ref"]
    assert reference["schema"] == local.PREVIEW_SUCCESSOR_PLANNING_RECEIPT_REFERENCE_SCHEMA == (
        "supervisor-local-planning-receipt-reference@5")
    assert {key: reference[key] for key in _binding_ids(fixture)} == _binding_ids(fixture)
    assert local.load_local_planning_receipt(reference, manifest=fixture["worker_manifest"]) == (
        fixture["worker_admission"]["receipt"])
    for task in fixture["graph"].tasks:
        installed = case["intent"].get_task(task.task_cid)
        pending, manifest, _, _ = local._contract(installed["body"], task.task_cid)
        assert pending["schema"] == local.PREVIEW_SUCCESSOR_CONTRACT_SCHEMA == (
            "supervisor-local-pending-completion@3")
        assert {key: pending[key] for key in _binding_ids(fixture)} == _binding_ids(fixture)
        retained_plan = pending["preview_successor_plan"]
        assert retained_plan == {"schema": "supervisor-local-preview-successor-plan@1",
            "graph": fixture["worker_admission"]["graph"],
            "receipt": fixture["worker_admission"]["receipt"]}
        assert {key: retained_plan["receipt"]["payload"][key] for key in _binding_ids(fixture)} == (
            _binding_ids(fixture))
        assert pending["task_spec"] == next(spec for spec in fixture["manifest"]["payload"]["tasks"]
            if spec["task_key"] == task.task_key)
        assert pending["dependencies"] == list(task.dependency_task_cids)
        assert pending["pending_requirements"] == reference["pending_requirements"]
        assert pending["pending_cid"] == reference["pending_cid"]
        assert pending["planning_receipt_cid"] == reference["receipt_cid"]
        assert manifest["codebase_preview_successor_context"] == _context(fixture)


def test_current_worker_verifier_retains_full_population_and_denies_proof_authority(worker_case):
    fixture = worker_case
    receipt = _verify(fixture)
    assert receipt["administrator_task_cids"] == sorted(task.task_cid for task in fixture["graph"].tasks)
    assert receipt["pending_cid"] == fixture["worker_admission"]["receipt"]["payload"]["pending_cid"]
    assert receipt["signed_bridge_cid"] == _context(fixture)["signed_bridge_cid"]
    assert receipt["bridge_cid"] == _context(fixture)["bridge_cid"]
    assert receipt["observed_current"] is True
    assert receipt["current_facts"] == receipt["removed_task_cids"] == []
    assert all(value is False for value in receipt["authority"].values())
    assert len(fixture["state"]["events"]) >= 2
    assert fixture["case"]["observations"].count("entry") >= 2


@pytest.mark.parametrize("field", ("codebase_preview_successor_context_cid", "parent_admission_cid",
                                 "signed_bridge_cid", "bridge_cid"))
def test_pending_contract_and_receipt_reference_reject_every_changed_join_identity(worker_case, field):
    fixture = worker_case
    result = _install(fixture)
    case = fixture["case"]
    task = fixture["graph"].tasks[0]
    body = deepcopy(case["intent"].get_task(task.task_cid)["body"])
    pending = body[local.CONTRACT_KEY]["payload"]
    pending[field] = _cid("foreign-pending-" + field)
    body[local.CONTRACT_KEY] = local._signed(pending, fixture["worker_manifest"]["payload"])
    with pytest.raises(ValueError):
        local._contract(body, task.task_cid)
    reference = deepcopy(case["intent"].get_plan(result["plan_id"])["body"]["local_planning_receipt_ref"])
    reference[field] = _cid("foreign-reference-" + field)
    with pytest.raises(ValueError):
        local.load_local_planning_receipt(reference, manifest=fixture["worker_manifest"])


def test_pending_and_reference_cannot_downgrade_or_grow_unknown_authority_fields(worker_case):
    fixture = worker_case
    result = _install(fixture)
    case = fixture["case"]
    task = fixture["graph"].tasks[0]
    original_body = case["intent"].get_task(task.task_cid)["body"]
    original_reference = case["intent"].get_plan(result["plan_id"])["body"]["local_planning_receipt_ref"]
    for change in ("downgrade", "unknown_authority"):
        body = deepcopy(original_body)
        pending = body[local.CONTRACT_KEY]["payload"]
        reference = deepcopy(original_reference)
        if change == "downgrade":
            pending["schema"] = local.CONTRACT_SCHEMA
            reference["schema"] = local.SUCCESSOR_PLANNING_RECEIPT_REFERENCE_SCHEMA
        else:
            pending["semantic_proof_authority"] = True
            reference["native_currentness_token"] = True
        body[local.CONTRACT_KEY] = local._signed(pending, fixture["worker_manifest"]["payload"])
        with pytest.raises(ValueError):
            local._contract(body, task.task_cid)
        with pytest.raises(ValueError):
            local.load_local_planning_receipt(reference, manifest=fixture["worker_manifest"])


@pytest.mark.parametrize("target", ("parent", "bridge", "graph"))
def test_child_authoring_cannot_swap_a_genuinely_signed_parent_companion_or_graph(worker_case, target):
    fixture = worker_case
    graph = replace(fixture["graph"], tasks=tuple(
        task for task in fixture["graph"].tasks if not task.dependency_task_cids))
    if target == "parent":
        _, admission = _author_graph_admission(fixture["case"], graph)
        with pytest.raises(ValueError):
            _author(fixture, parent_admission=admission)
    elif target == "bridge":
        payload = deepcopy(fixture["bridge"]["payload"])
        payload["checker_policy_cid"] = _cid("genuinely-resigned-other-checker")
        payload["bridge_cid"] = content_identity({key: value for key, value in payload.items()
            if key != "bridge_cid"})
        bridge = local._signed(payload, fixture["manifest"]["payload"])
        assert local._verify_signature(bridge, local.verify_local_benchmark_admission(
            fixture["admission"])["profile"]) == payload
        with pytest.raises(ValueError):
            _author(fixture, bridge=bridge)
    else:
        with pytest.raises(ValueError):
            _admit(fixture, graph=graph)


@pytest.mark.parametrize("change", ("unknown_manifest", "unknown_context", "parent_cid", "bridge_cid",
                                   "signed_bridge_cid", "downgrade"))
def test_signed_child_manifest_rejects_unknown_fields_changed_identities_and_profile_downgrade(worker_case, change):
    fixture = worker_case
    payload = deepcopy(fixture["worker_manifest"]["payload"])
    context = payload["codebase_preview_successor_context"]
    if change == "unknown_manifest":
        payload["proof_authority"] = True
    elif change == "unknown_context":
        context["worker_freshness_token"] = True
    elif change == "parent_cid":
        context["parent_admission_cid"] = _cid("different-parent-admission")
    elif change in {"bridge_cid", "signed_bridge_cid"}:
        context[change] = _cid("different-child-" + change)
    else:
        payload["schema"] = local.SUCCESSOR_MANIFEST_SCHEMA
    signed = local._signed(payload, fixture["worker_manifest"]["payload"])
    with pytest.raises(ValueError):
        _admit(fixture, manifest=signed)


@pytest.mark.parametrize("target", ("task_population", "pending", "authority", "unknown_receipt", "downgrade"))
def test_owner_resigning_cannot_change_complete_child_planning_contract(worker_case, target):
    fixture = worker_case
    admission = deepcopy(fixture["worker_admission"])
    payload = admission["receipt"]["payload"]
    if target == "task_population":
        payload["administrator_task_cids"].pop()
    elif target == "pending":
        payload["pending_requirements"][0]["required"] = False
    elif target == "authority":
        payload["completion_authority"] = True
    elif target == "unknown_receipt":
        payload["verified_semantic_proof"] = True
    else:
        payload["schema"] = local.SUCCESSOR_PLANNING_RECEIPT_SCHEMA
    admission["receipt"] = local._signed(payload, admission["manifest"]["payload"])
    with pytest.raises(ValueError):
        _verify(fixture, worker_admission=admission)
    _assert_no_installed_population(fixture)


@pytest.mark.parametrize("operation", ("author", "admit", "verify", "install", "prepare"))
def test_explicit_worker_profile_requires_current_bridge_inputs_at_every_public_boundary(worker_case, operation):
    fixture = worker_case
    call = {"author": _author, "admit": _admit, "verify": _verify,
        "install": _install, "prepare": _prepare}[operation]
    with pytest.raises(ValueError):
        call(fixture, bridge_inputs=None)
    _assert_no_installed_population(fixture)


@pytest.mark.parametrize("target", ("root", "checker", "source"))
def test_observation_drift_after_full_plan_staging_withholds_committed_intent_population(
        worker_case, monkeypatch, target):
    fixture = worker_case
    case = fixture["case"]
    staged = []
    prepare_plan = local._materialize_local_benchmark_plan

    def stage_then_mutate(**arguments):
        # The production installer records the complete native write set
        # before acquiring the target Intent write lock. No target row exists
        # at this seam. Root/checker drift fails closing preparation; source
        # drift fails source replay or the native fence, with no committed rows.
        assert not isinstance(arguments["intent"], IntentRepository)
        result = prepare_plan(**arguments)
        staged.append(sorted(result["task_cids"]))
        assert all(case["intent"].get_task(task.task_cid) is None
            for task in fixture["graph"].tasks)
        if target == "root":
            fixture["state"]["roots"] = _different_root(fixture["state"]["roots"], "policy_root")
        elif target == "checker":
            fixture["state"]["checker"] = _cid("late-staged-checker-drift")
        else:
            (case["repository"] / "input.py").write_bytes(b"VALUE = 98765\n")
        return result

    def source_closing_fence():
        sources = fixture["manifest"]["payload"]["sources"]
        if local._sources(case["repository"], sorted(sources)) != sources:
            raise ValueError("independent native source closing fence observed late staging edit")

    monkeypatch.setattr(local, "_materialize_local_benchmark_plan", stage_then_mutate)
    case["close_callback"] = source_closing_fence
    with pytest.raises(ValueError):
        _install(fixture)
    assert staged == [sorted(task.task_cid for task in fixture["graph"].tasks)]
    _assert_no_installed_population(fixture)


def test_final_native_failure_after_real_task_writes_rolls_back_the_entire_intent_transaction(
        worker_case, monkeypatch):
    fixture = worker_case
    case = fixture["case"]
    persisted = []
    native_upsert = IntentRepository.upsert_task

    def upsert_then_fail_final_fence(owner, **arguments):
        result = native_upsert(owner, **arguments)
        if owner.uses_bound_connection and owner.owner_id == case["intent"].owner_id:
            persisted.append(arguments["task_cid"])
            case["close_error"] = "late native closing failure after genuine task writes"
        return result

    monkeypatch.setattr(IntentRepository, "upsert_task", upsert_then_fail_final_fence)
    with pytest.raises(ValueError, match="late native closing failure"):
        _install(fixture)
    assert sorted(persisted) == sorted(task.task_cid for task in fixture["graph"].tasks)
    _assert_no_installed_population(fixture)


def test_installation_observers_can_reenter_target_intent_write_owner_before_persistence(worker_case):
    fixture = worker_case
    case = fixture["case"]
    observed_rows = []

    def reenter_target_write_owner():
        # Acquiring the genuine target write transaction would time out if the
        # application observer ran while its file lock was already held. The
        # observer performs a read and leaves all target rows unchanged.
        with case["intent"]._connection(write=True) as connection:
            observed_rows.append(connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])

    fixture["state"]["roots_callback"] = reenter_target_write_owner
    result = _install(fixture)
    assert observed_rows and all(rows == 0 for rows in observed_rows)
    assert sorted(result["task_cids"]) == sorted(task.task_cid for task in fixture["graph"].tasks)
    assert all(case["intent"].get_task(task.task_cid) is not None for task in fixture["graph"].tasks)


@pytest.mark.parametrize("target", ("root", "checker"))
def test_closing_and_resource_exit_observations_do_not_become_reusable_worker_freshness(
        worker_case, target):
    fixture = worker_case
    def mutate():
        if target == "root":
            fixture["state"]["roots"] = _different_root(fixture["state"]["roots"], "policy_root")
        else:
            fixture["state"]["checker"] = _cid("worker-boundary-checker-drift")
    # The first native closing and resource cleanup are consumed before the
    # second application observation and final native-only fence.
    fixture["case"]["exit_callback"] = mutate
    with pytest.raises(ValueError):
        _verify(fixture)


def test_private_worker_operation_requires_preparation_and_a_successful_closing_fence(worker_case):
    fixture = worker_case
    arguments = {**_current_arguments(fixture), "worker_admission": fixture["worker_admission"],
        "bridge_inputs": fixture["bridge_inputs"]}
    with pytest.raises(ValueError):
        with joined._current_preview_successor_worker_operation(**arguments) as (_, close):
            close()
    with pytest.raises(ValueError):
        with joined._current_preview_successor_worker_operation(**arguments) as (_, close):
            close.prepare()


def test_private_worker_preparation_and_native_closing_are_both_single_use(worker_case):
    fixture = worker_case
    arguments = {**_current_arguments(fixture), "worker_admission": fixture["worker_admission"],
        "bridge_inputs": fixture["bridge_inputs"]}
    with joined._current_preview_successor_worker_operation(**arguments) as (receipt, close):
        close.prepare()
        with pytest.raises(ValueError):
            close.prepare()
        assert close() is receipt
        with pytest.raises(ValueError):
            close()


def test_final_native_closing_runs_no_application_observer_after_unlocked_preparation(worker_case):
    fixture = worker_case
    arguments = {**_current_arguments(fixture), "worker_admission": fixture["worker_admission"],
        "bridge_inputs": fixture["bridge_inputs"]}
    with joined._current_preview_successor_worker_operation(**arguments) as (_, close):
        close.prepare()
        events = list(fixture["state"]["events"])
        native_closes = fixture["case"]["observations"].count("close")
        close()
        assert fixture["state"]["events"] == events
        assert fixture["case"]["observations"].count("close") == native_closes + 1
    assert fixture["state"]["events"] == events


@pytest.mark.parametrize("target", ("materials", "callback", "receipt"))
def test_private_final_fence_rechecks_passive_custody_after_unlocked_preparation(worker_case, target):
    fixture = worker_case
    arguments = {**_current_arguments(fixture), "worker_admission": fixture["worker_admission"],
        "bridge_inputs": fixture["bridge_inputs"]}
    with pytest.raises(ValueError):
        with joined._current_preview_successor_worker_operation(**arguments) as (receipt, close):
            close.prepare()
            if target == "materials":
                fixture["materials"].extra["authored_metadata"] = "changed after preparation"
            elif target == "callback":
                object.__setattr__(fixture["bridge_inputs"], "checker_observer",
                    lambda _: fixture["state"]["checker"])
            else:
                receipt["administrator_task_cids"].pop()
            close()


def test_replacing_the_closing_callable_cannot_forge_a_successful_native_fence(worker_case):
    fixture = worker_case
    arguments = {**_current_arguments(fixture), "worker_admission": fixture["worker_admission"],
        "bridge_inputs": fixture["bridge_inputs"]}
    with pytest.raises(ValueError, match="closing"):
        with joined._current_preview_successor_worker_operation(**arguments) as (receipt, close):
            close.prepare()
            close._close = lambda: receipt
            assert close() is receipt


def test_final_worker_fence_refuses_a_changed_thread_after_preparation(worker_case, monkeypatch):
    fixture = worker_case
    arguments = {**_current_arguments(fixture), "worker_admission": fixture["worker_admission"],
        "bridge_inputs": fixture["bridge_inputs"]}
    thread_id = joined.threading.get_ident()
    with pytest.raises(ValueError, match="thread"):
        with monkeypatch.context() as patch:
            with joined._current_preview_successor_worker_operation(**arguments) as (_, close):
                close.prepare()
                # A custody control only: no new process or thread is created.
                patch.setattr(joined.threading, "get_ident", lambda: thread_id + 1)
                close()


def test_producer_returned_admission_cannot_mutate_while_a_detached_copy_is_verified(worker_case, monkeypatch):
    fixture = worker_case
    captured = []
    native_admit = local.admit_local_benchmark_plan

    def admit_and_capture(**arguments):
        result = native_admit(**arguments)
        captured.append(result)
        return result

    def mutate_producer_copy():
        if captured:
            captured[-1]["receipt"]["payload"]["runtime_requirements_preserved"] = False

    monkeypatch.setattr(local, "admit_local_benchmark_plan", admit_and_capture)
    fixture["state"]["roots_callback"] = mutate_producer_copy
    with pytest.raises(ValueError):
        _admit(fixture)
    assert len(captured) == 1
    _assert_no_installed_population(fixture)


def test_versioned_public_worker_artifact_replays_the_complete_signed_join_without_private_owners(
        worker_case, monkeypatch, tmp_path):
    fixture = worker_case
    descriptor = _prepare(fixture)
    artifact = Path(descriptor["artifact"])
    payload = json.loads(artifact.read_bytes())
    assert payload["schema"] == public.PREVIEW_SUCCESSOR_SCHEMA == "supervisor-public-instruction@5"
    assert payload["manifest"] == fixture["worker_manifest"]
    assert payload["inventory_plan_admission"] == {"graph": fixture["worker_admission"]["graph"],
        "receipt": fixture["worker_admission"]["receipt"]}
    assert descriptor["signed_bridge_cid"] == _context(fixture)["signed_bridge_cid"]
    workspace = tmp_path / "public-worker"
    _git(fixture["case"]["repository"], "worktree", "add", "--detach", "-q", str(workspace))

    def forbidden(*_, **__):
        pytest.fail("public preview-successor replay accessed a private or metadata owner")

    monkeypatch.setattr(local, "load_local_profile", forbidden)
    monkeypatch.setattr(local, "_signed", forbidden)
    monkeypatch.setattr(joined, "_current_preview_successor_worker_operation", forbidden)
    monkeypatch.setenv(meta.ENV_DUCKDB, str(tmp_path / "unopened-public-meta.duckdb"))
    monkeypatch.setattr(meta.SupervisorMetaIndex, "from_env", classmethod(forbidden))
    monkeypatch.setattr(meta, "_connect", forbidden)
    task = fixture["case"]["tasks"][0]
    block, receipt = public.load_public_instruction(artifact=artifact, expected_sha256=descriptor["sha256"],
        task_cid=task.task_cid, prompt=json.dumps({"objective_id": task.task_key}), workspace=workspace)
    assert "VALUE = 1" in block
    assert receipt["extra_provider_calls"] == 0
    assert receipt["schema"] == "supervisor-public-instruction-inclusion@5"
    assert all(payload[key] is False for key in (
        "completion_authority", "publication_authority", "scope_expansion_authority"))
