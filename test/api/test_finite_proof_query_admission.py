"""Genuine solver evidence at the new finite signed receiving transaction.

The fixture uses real Z3/CVC5, native source/CAS/query owners, real local
signatures and actual DuckDB transactions. Corruption controls edit retained
inputs or run genuine owner rebuilds; they never supply a successful checker.
"""
from copy import deepcopy
from pathlib import Path
import ast
import inspect
import json
import os
import shutil
import stat
import tempfile

import pytest

from ipfs_accelerate_py.agent_supervisor.control.profile_authority import sign_profile_binding
from ipfs_accelerate_py.agent_supervisor.planning import finite_integer_codebase as matcher
from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_admission as finite
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_datasets_py.duckdb_control.codebase_verification_catalog import CodebaseVerificationCatalog
from ipfs_datasets_py.logic.software_contracts import codebase_applicability, codebase_verification
from ipfs_datasets_py.logic.software_contracts import codebase_finite_integer_observation as observation
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex, StaleCodebaseError

from test.api.test_finite_integer_codebase import (
    LEAN, PYTHON, finite_source, finite_tools,
)
from test.api.test_finite_repository_admission import _author, _build_case


def _module():
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_proof_query_admission
    return finite_proof_query_admission


def build_proof_query_case(root, tools, *, publish_app=True, contract=None, domain=None, offset=1):
    """Shared genuine setup for the additive planner and admission controls."""
    from ipfs_accelerate_py.agent_supervisor.planning.finite_proof_query_join import derive_finite_proof_spec

    assert all(shutil.which(name) is not None for name in ("z3", "cvc5")), (
        "the native Z3/CVC5 pair is required; this qualification cannot skip it"
    )
    case = _build_case(Path(root), tools, offset=offset)
    try:
        derived_contract, derived_domain = derive_finite_proof_spec(
            intent_document=case["arguments"]["intent_document"],
            source_text=case["arguments"]["source_text"],
        )
        contract = derived_contract if contract is None else contract
        domain = derived_domain if domain is None else domain
        controls = dict(expected_head=case["expected_head"], scheduler=case["scheduler"])
        record = codebase_verification.verify_current_codebase_unit(
            case["index"], case["repository"], path="calc.py", contracts=[contract], **controls
        )
        applicability = None
        if publish_app:
            applicability = codebase_applicability.verify_current_codebase_applicability(
                case["index"], case["repository"], verification_cid=record.artifact_cid,
                domains=[domain], **controls
            )
        catalog = CodebaseVerificationCatalog(case["index"])
        projection = catalog.publish(
            case["repository"], verification_cid=record.artifact_cid,
            applicability_cid=None if applicability is None else applicability.artifact_cid,
            operation_id="finite-proof-query-fixture", **controls
        )
        case.update(
            verification_catalog=catalog, verification_record=record,
            applicability_record=applicability, proof_contract=contract,
            proof_domain=domain, proof_projection=projection,
        )
        case["declaration"] = _author(case)
        return case
    except BaseException:
        case["connection"].close()
        raise


def close_proof_query_case(case):
    state = case["scheduler"].snapshot()
    assert state["active_lease_count"] == state["waiting_request_count"] == 0
    case["connection"].close()


@pytest.fixture(scope="session")
def proof_query_native_case(tmp_path_factory):
    assert PYTHON.is_file() and LEAN.is_file(), "native Python and Lean are required"
    tools = observation.seal_finite_integer_tools(python_executable=PYTHON, lean_executable=LEAN)
    case = build_proof_query_case(tmp_path_factory.mktemp("finite-proof-query-shared") / "native", tools)
    try:
        yield case
    finally:
        close_proof_query_case(case)


@pytest.fixture(scope="module")
def proof_query_signed_case(proof_query_native_case):
    case = proof_query_native_case
    case["proof_query_admission"] = _admit(case, output=case["root"] / "new-signed-admission")
    return case


def _admit(case, *, output, **changes):
    kwargs = dict(
        owner=case["owner"], declaration=case["declaration"], graph=case["graph"],
        verification_catalog=case["verification_catalog"], output=output,
        policy_observer=lambda request: request.roots,
    )
    kwargs.update(changes)
    return _module().admit_finite_proof_query_plan(**kwargs)


def _verify(case, admission, *, output, **changes):
    kwargs = dict(
        owner=case["owner"], admission=admission,
        verification_catalog=case["verification_catalog"], output=output,
        policy_observer=lambda request: request.roots,
    )
    kwargs.update(changes)
    return _module().verify_current_finite_proof_query_admission(**kwargs)


def _materialize(case, admission, *, intent, output, **changes):
    kwargs = dict(
        owner=case["owner"], admission=admission, intent=intent,
        verification_catalog=case["verification_catalog"], output=output,
        policy_observer=lambda request: request.roots,
    )
    kwargs.update(changes)
    return _module().materialize_finite_proof_query_plan(**kwargs)


def _rows(intent):
    with intent._connection() as connection:
        return {table: connection.execute("SELECT count(*) FROM " + table).fetchone()[0]
                for table in ("objectives", "goals", "plans", "tasks", "task_dependencies")}


def _rebuild(case):
    return case["verification_catalog"].rebuild_current(
        case["repository"], expected_head=case["expected_head"], scheduler=case["scheduler"]
    )


def _resign(case, payload):
    return {"payload": payload, "binding": sign_profile_binding(
        profile_dir=case["profile"], lifecycle_dir=case["lifecycle"], payload=payload
    )}


def test_genuine_signed_join_preserves_original_tasks_pending_checks_and_model_off(proof_query_signed_case):
    case = proof_query_signed_case
    admission = case["proof_query_admission"]
    assert set(admission) == {"schema", "finite_admission", "indexed_plan", "receipt"}
    base = admission["finite_admission"]
    verified = finite.verify_finite_repository_admission(admission=base)
    semantic = verified["semantic_context"]
    assert semantic["eligible_requirement_ids"] == [matcher.TYPE_STATEMENT_ID]
    assert semantic["residual_requirement_ids"] == [matcher.OFFSET_STATEMENT_ID]
    assert base["graph"] == case["graph"].to_dict()
    native = local.verify_local_benchmark_admission(base["local_admission"])
    assert {row["task_key"] for row in native["manifest"]["tasks"]} == {"TYPE-TASK", "OFFSET-TASK"}
    pending = native["receipt"]["pending_requirements"]
    assert pending and all(row["required"] is True and row["phase"] == "post_execution" for row in pending)
    assert base["evidence"]["feature_context"]["mode"] == "model_off"
    for flag in ("proof_authority", "execution_authority", "completion_authority", "omission_authority"):
        assert base["receipt"]["payload"][flag] is False
        assert admission["receipt"]["payload"][flag] is False
    _, profile, *_ = finite._declaration(base["declaration"])
    assert local._verify_signature(admission["receipt"], profile) == admission["receipt"]["payload"]


def test_native_materialization_stores_both_references_and_all_original_contracts(proof_query_signed_case, tmp_path):
    case = proof_query_signed_case
    base_local = case["proof_query_admission"]["finite_admission"]["local_admission"]
    local_verified = local.verify_local_benchmark_admission(base_local)
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        result = _materialize(case, case["proof_query_admission"], intent=intent,
                              output=tmp_path / "materialization")
        assert set(result["task_cids"]) == {row.task_cid for row in case["graph"].tasks}
        assert _rows(intent) == {"objectives": 1, "goals": 1, "plans": 1, "tasks": 2, "task_dependencies": 1}
        plan = intent.get_plan(result["plan_id"])
        assert {"local_planning_receipt_ref", "finite_repository_admission_ref",
                "finite_proof_query_admission_ref"} <= set(plan["body"])
        assert plan["body"]["finite_repository_admission_ref"]["admission_cid"] == cid_for_structured(
            case["proof_query_admission"]["finite_admission"]
        )
        for original in case["graph"].tasks:
            row = intent.get_task(original.task_cid)
            assert row["status"] == "ready"
            assert row["dependencies"] == tuple(original.dependency_task_cids)
            signed_contract = row["body"][local.CONTRACT_KEY]
            contract = local._verify_signature(signed_contract, local_verified["profile"])
            assert contract == local._pending_contract_payload(
                admission=base_local, verified=local_verified, task=original,
                intent_owner_id=intent.owner_id,
            )
            assert contract["pending_requirements"]
            assert all(item["required"] is True and item["phase"] == "post_execution"
                       for item in contract["pending_requirements"])


def test_receiving_gate_reobserves_current_semantics_without_changing_task_population(proof_query_signed_case, tmp_path):
    case = proof_query_signed_case
    admission = case["proof_query_admission"]
    original = deepcopy(admission)
    current = _verify(case, admission, output=tmp_path / "genuine-current")
    assert current["observed_current"] is True
    assert current["admission_cid"] == cid_for_structured(admission)
    assert current["semantic_context"]["eligible_requirement_ids"] == [matcher.TYPE_STATEMENT_ID]
    assert current["semantic_context"]["residual_requirement_ids"] == [matcher.OFFSET_STATEMENT_ID]
    assert admission == original
    assert all(current[name] is False for name in (
        "proof_authority", "execution_authority", "completion_authority", "omission_authority",
        "training_executed", "model_inference_executed",
    ))


def test_last_successful_native_observation_cannot_hide_late_working_source_edit(
        proof_query_signed_case, tmp_path, monkeypatch):
    """Mutate after the actual final scan, before detached receiving custody."""
    from ipfs_accelerate_py.agent_supervisor.planning import finite_proof_query_join as join

    case = proof_query_signed_case
    module = _module()
    tree = ast.parse(Path(join.__file__).read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_preview_owned_finite_proof_join")
    # Select the final top-level require_current call after the outer context.
    # This locates the real boundary without substituting any native result.
    calls = [node for node in function.body if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
             and node.value.func.id == "require_current"]
    assert len(calls) == 1
    last_line = calls[0].lineno
    path = case["repository"] / "calc.py"
    original = path.read_bytes()
    observe = RepositoryCodebaseIndex.observe_current
    replay = module._replay
    armed, fired, replay_returned = [], [], []

    def after_native_observation(self, *args, **kwargs):
        result = observe(self, *args, **kwargs)
        if self is case["index"] and armed and not fired:
            frame = inspect.currentframe().f_back
            final_preview = closing_capture = False
            while frame is not None:
                if frame.f_code is join._preview_owned_finite_proof_join.__code__:
                    final_preview = frame.f_lineno == last_line
                if frame.f_code is join.capture_current_finite_proof_query.__code__:
                    closing_capture = "result" in frame.f_locals
                frame = frame.f_back
            if final_preview and closing_capture:
                assert result.head == case["expected_head"]
                path.write_bytes(finite_source(11))
                fired.append(True)
        return result

    def arm_real_receiving_replay(*args, **kwargs):
        armed.append(True)
        try:
            result = replay(*args, **kwargs)
            replay_returned.append(True)
            return result
        finally:
            armed.pop()

    monkeypatch.setattr(RepositoryCodebaseIndex, "observe_current", after_native_observation)
    monkeypatch.setattr(module, "_replay", arm_real_receiving_replay)
    try:
        with pytest.raises(StaleCodebaseError, match="source custody.*working:calc.py"):
            _verify(case, case["proof_query_admission"], output=tmp_path / "last-source-observation")
        assert fired == [True]
        assert replay_returned == [True], "the real native query/planner replay completed before custody refused"
        assert case["scheduler"].snapshot()["active_lease_count"] == 0
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize("operation", ["current_verify", "materialize"])
def test_final_native_source_custody_epoch_rebuild_refuses_and_rolls_back(
        proof_query_native_case, tmp_path, monkeypatch, operation):
    """Rebuild the real query index after replay, during its last AST lookup."""
    from ipfs_accelerate_py.agent_supervisor.planning import finite_integer_source_custody as source_custody

    case = build_proof_query_case(tmp_path / "final-custody-native", proof_query_native_case["tools"])
    receipt = {"schema": "finite-proof-query-final-custody-control@1", "operation": operation,
               "replay_returned": [], "native_plan_written": [], "lookup_fired": []}
    receipt_path = tmp_path / "final-custody-epoch-control.json"
    try:
        # The original signed admission is complete before any callback patch.
        admission = _admit(case, output=tmp_path / "final-custody-admission")
        page = admission["indexed_plan"]["proof_query_closure"]["discovery_query"]["page"]
        receipt.update(epoch_before=page["epoch"], inventory_cid_before=page["inventory_cid"])
        module = _module()
        real_replay = module._replay
        real_lookup = RepositoryCodebaseIndex.lookup
        real_upsert = IntentRepository.upsert_plan
        armed = [False]

        def after_actual_plan_write(self, **kwargs):
            result = real_upsert(self, **kwargs)
            if "finite_proof_query_admission_ref" in dict(kwargs.get("body") or {}):
                assert not receipt["native_plan_written"]
                receipt["native_plan_written"].append(True)
            return result

        def after_genuine_replay(*args, **kwargs):
            result = real_replay(*args, **kwargs)
            after_write = bool(receipt["native_plan_written"])
            receipt["replay_returned"].append({"after_native_plan_write": after_write})
            armed[0] = operation == "current_verify" or after_write
            return result

        def after_actual_custody_lookup(self, *args, **kwargs):
            result = real_lookup(self, *args, **kwargs)
            if self is case["index"] and armed[0] and not receipt["lookup_fired"]:
                frame = inspect.currentframe().f_back
                native = requiring_custody = False
                while frame is not None:
                    native |= frame.f_code is source_custody._native.__code__
                    requiring_custody |= frame.f_code is source_custody.FrozenSourceCustody.require_current.__code__
                    frame = frame.f_back
                if native and requiring_custody:
                    # Mark before rebuilding because the genuine owner rebuild
                    # performs its own native source observations and lookups.
                    receipt["lookup_fired"].append({"path": args[1], "after_genuine_replay": True})
                    inventory = _rebuild(case)
                    receipt.update(epoch_after=inventory.epoch, inventory_cid_after=inventory.inventory_cid)
            return result

        monkeypatch.setattr(IntentRepository, "upsert_plan", after_actual_plan_write)
        monkeypatch.setattr(module, "_replay", after_genuine_replay)
        monkeypatch.setattr(RepositoryCodebaseIndex, "lookup", after_actual_custody_lookup)
        if operation == "current_verify":
            with pytest.raises(ValueError) as refused:
                _verify(case, admission, output=tmp_path / "final-custody-receiving")
            assert receipt["replay_returned"] == [{"after_native_plan_write": False}]
            assert receipt["native_plan_written"] == []
        else:
            with IntentRepository(tmp_path / "final-custody-intent.duckdb") as intent:
                with pytest.raises(ValueError) as refused:
                    _materialize(case, admission, intent=intent,
                                 output=tmp_path / "final-custody-materialization")
                receipt["native_rows_after_refusal"] = _rows(intent)
                assert receipt["native_rows_after_refusal"] == {
                    "objectives": 0, "goals": 0, "plans": 0, "tasks": 0, "task_dependencies": 0,
                }
            assert receipt["native_plan_written"] == [True]
            assert receipt["replay_returned"] == [
                {"after_native_plan_write": False}, {"after_native_plan_write": True},
            ]
        receipt.update(refusal_type=type(refused.value).__name__, refusal_message=str(refused.value))
        assert receipt["lookup_fired"] == [{"path": "calc.py", "after_genuine_replay": True}]
        assert receipt["epoch_after"] > receipt["epoch_before"]
        assert receipt["inventory_cid_after"] != receipt["inventory_cid_before"]
        receipt["qualification"] = "refused_after_real_late_epoch_rebuild"
    finally:
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        close_proof_query_case(case)


def test_dirty_source_cannot_reuse_signed_proof_query_admission(proof_query_signed_case, tmp_path):
    case = proof_query_signed_case
    path = case["repository"] / "calc.py"
    original = path.read_bytes()
    path.write_bytes(finite_source(9))
    try:
        with pytest.raises(ValueError):
            _verify(case, case["proof_query_admission"], output=tmp_path / "dirty-source")
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize("flag,value", [
    ("proof_authority", True), ("proof_authority", 0),
    ("execution_authority", True), ("completion_authority", 0),
    ("omission_authority", True),
])
def test_genuine_resign_does_not_relax_join_authority(proof_query_signed_case, tmp_path, flag, value):
    case = proof_query_signed_case
    changed = deepcopy(case["proof_query_admission"])
    changed["receipt"]["payload"][flag] = value
    changed["receipt"] = _resign(case, changed["receipt"]["payload"])
    with pytest.raises(ValueError):
        _verify(case, changed, output=tmp_path / "resigned-authority")


def test_original_task_omission_cannot_borrow_new_signed_join(proof_query_signed_case, tmp_path):
    case = proof_query_signed_case
    changed = deepcopy(case["proof_query_admission"])
    changed["finite_admission"]["graph"]["tasks"] = changed["finite_admission"]["graph"]["tasks"][:1]
    with pytest.raises(ValueError):
        _verify(case, changed, output=tmp_path / "omitted-task")


@pytest.mark.parametrize("mutation", [
    "omitted_record", "swapped_record", "omitted_key", "wrong_contract",
    "narrowed_domain", "contradictory_premise",
])
def test_genuine_resign_and_rehash_cannot_replace_independent_current_proof_query(
        proof_query_signed_case, tmp_path, mutation):
    case = proof_query_signed_case
    changed = deepcopy(case["proof_query_admission"])
    indexed = changed["indexed_plan"]
    closure = indexed["proof_query_closure"]
    if mutation == "omitted_record":
        closure["verification"] = {}
        closure["verification_cid"] = cid_for_structured({})
    elif mutation == "swapped_record":
        closure["verification"], closure["applicability"] = (
            closure["applicability"], closure["verification"],
        )
        closure["verification_cid"], closure["applicability_cid"] = (
            closure["applicability_cid"], closure["verification_cid"],
        )
    elif mutation == "omitted_key":
        assert closure["canonical_key_membership"]
        closure["canonical_key_membership"] = closure["canonical_key_membership"][:-1]
    elif mutation == "wrong_contract":
        closure["contract"]["postconditions"] = ["result == n + 9"]
        closure["contract_cid"] = cid_for_structured(closure["contract"])
    elif mutation == "narrowed_domain":
        closure["domain"]["predicates"] = ["n == 0"]
        closure["domain_cid"] = cid_for_structured(closure["domain"])
    else:
        closure["contract"]["preconditions"] = ["False"]
        closure["contract_cid"] = cid_for_structured(closure["contract"])
    closure["closure_cid"] = cid_for_structured({key: value for key, value in closure.items()
                                               if key != "closure_cid"})
    indexed["proof_query_closure_cid"] = closure["closure_cid"]
    indexed["result_cid"] = cid_for_structured({key: value for key, value in indexed.items()
                                               if key != "result_cid"})
    payload = changed["receipt"]["payload"]
    payload["indexed_plan_cid"] = indexed["result_cid"]
    payload["proof_query_closure_cid"] = closure["closure_cid"]
    changed["receipt"] = _resign(case, payload)
    # The real signature and outer complete-input CIDs pass. Native query
    # reconstruction must still reject the substituted complete proof body.
    assert _module()._received(changed)[0] == changed
    with pytest.raises(ValueError):
        _verify(case, changed, output=tmp_path / "resigned-proof-body")


def test_bound_or_foreign_native_transaction_cannot_materialize_join(proof_query_signed_case, tmp_path):
    case = proof_query_signed_case
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        with intent._connection(write=True) as connection:
            with IntentRepository(bound_connection=connection) as bound:
                with pytest.raises(ValueError):
                    _materialize(case, case["proof_query_admission"], intent=bound,
                                  output=tmp_path / "bound-transaction")
        assert _rows(intent) == {table: 0 for table in _rows(intent)}


def test_real_catalog_rebuild_makes_signed_query_epoch_stale(proof_query_signed_case, tmp_path):
    case = proof_query_signed_case
    # An independent clone is necessary: this real epoch advance is irreversible
    # for the shared signed fixture and must not invalidate unrelated controls.
    isolated = build_proof_query_case(tmp_path / "epoch-native", case["tools"])
    try:
        admission = _admit(isolated, output=tmp_path / "epoch-admission")
        _rebuild(isolated)
        with pytest.raises(ValueError):
            _verify(isolated, admission, output=tmp_path / "epoch-receiving")
    finally:
        close_proof_query_case(isolated)


def test_already_complete_proof_join_retains_full_graph_and_cannot_omit_native_tasks(
        proof_query_signed_case, tmp_path):
    isolated = build_proof_query_case(tmp_path / "complete-native", proof_query_signed_case["tools"], offset=2)
    try:
        admission = _admit(isolated, output=tmp_path / "complete-admission")
        assert admission["finite_admission"]["local_admission"] is None
        assert admission["finite_admission"]["graph"] == isolated["graph"].to_dict()
        assert len(admission["finite_admission"]["graph"]["tasks"]) == 2
        assert admission["receipt"]["payload"]["planning_permitted"] is False
        assert admission["receipt"]["payload"]["no_work_review_only"] is True
        assert admission["indexed_plan"]["selected_task_ids"] == []
        assert admission["indexed_plan"]["removed_task_ids"] == []
        with IntentRepository(tmp_path / "complete-intent.duckdb") as intent:
            with pytest.raises(ValueError):
                _materialize(isolated, admission, intent=intent, output=tmp_path / "complete-materialization")
            assert _rows(intent) == {"objectives": 0, "goals": 0, "plans": 0, "tasks": 0, "task_dependencies": 0}
    finally:
        close_proof_query_case(isolated)


@pytest.mark.parametrize("mutation", ["query_epoch", "source_bytes", "proof_cas_inode"])
def test_late_native_plan_write_mutation_rolls_back_entire_population(
        proof_query_signed_case, tmp_path, monkeypatch, mutation):
    case = proof_query_signed_case
    isolated = build_proof_query_case(tmp_path / "late-native", case["tools"])
    source = isolated["repository"] / "calc.py"
    original_bytes = source.read_bytes()
    try:
        admission = _admit(isolated, output=tmp_path / "late-admission")
        real_upsert = IntentRepository.upsert_plan
        fired = []

        def after_actual_write(self, **kwargs):
            result = real_upsert(self, **kwargs)
            if "finite_proof_query_admission_ref" in dict(kwargs.get("body") or {}):
                assert not fired
                fired.append(True)
                if mutation == "query_epoch":
                    _rebuild(isolated)
                elif mutation == "source_bytes":
                    source.write_bytes(finite_source(7))
                else:
                    closure = admission["indexed_plan"]["proof_query_closure"]
                    path = isolated["index"].artifacts.path_for(closure["verification_cid"])
                    before = path.stat()
                    raw = path.read_bytes()
                    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as replacement:
                        replacement.write(raw)
                        replacement.flush()
                        os.fchmod(replacement.fileno(), stat.S_IMODE(before.st_mode))
                        os.fsync(replacement.fileno())
                        temporary = Path(replacement.name)
                    os.replace(temporary, path)
                    assert path.read_bytes() == raw and path.stat().st_ino != before.st_ino
            return result

        monkeypatch.setattr(IntentRepository, "upsert_plan", after_actual_write)
        with IntentRepository(tmp_path / "intent.duckdb") as intent:
            with pytest.raises(ValueError):
                _materialize(isolated, admission, intent=intent, output=tmp_path / "late-materialization")
            assert fired == [True]
            assert _rows(intent) == {"objectives": 0, "goals": 0, "plans": 0, "tasks": 0, "task_dependencies": 0}
    finally:
        source.write_bytes(original_bytes)
        close_proof_query_case(isolated)
