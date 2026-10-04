"""Real indexed obligations bound to one freshly observed finite preview.

The fixtures require native Python, Lean, Z3 and CVC5. The solver sidecars
remain conditional mathematical evidence; only the fresh finite observations
can supply BOUNDED_OBSERVATION facts. Corruption controls begin with genuinely
produced native records and never synthesize a successful checker result.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.planning import finite_integer_codebase as matcher
from ipfs_accelerate_py.agent_supervisor.planning import finite_proof_query_join as join
from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import (
    FactAuthority, ObservedFact,
)
from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import (
    PlanCreateInputSnapshot, PlanCreatePreviewReceipt, freeze_plan_create_input_snapshot,
)
from ipfs_datasets_py.duckdb_control.codebase_verification_catalog import (
    CodebaseVerificationCatalog, CodebaseVerificationCatalogError,
)
from ipfs_datasets_py.logic.common.canonical_cache_key import (
    CanonicalProofCacheKey, REQUIRED_IDENTITY_FIELDS,
)
from ipfs_datasets_py.logic.ir_core.protocols import ExecutionBounds
from ipfs_datasets_py.logic.software_contracts import codebase_applicability, codebase_verification
from ipfs_datasets_py.logic.software_contracts.codebase_ir import StaleCodebaseError
from ipfs_datasets_py.logic.software_contracts.content import (
    canonical_dag_json_bytes, cid_for_structured,
)
from ipfs_datasets_py.logic.software_verification.applicability import RequestedInputDomain
from ipfs_datasets_py.logic.software_verification.pipeline import ContractSpec

from test.api.test_finite_integer_codebase import INPUTS, finite_source, finite_text, finite_tools
from test.api.test_finite_integer_plan_preview import TASK_OFFSET
from test.api.test_finite_proof_query_admission import (
    build_proof_query_case, close_proof_query_case, proof_query_native_case,
)


def _capture(case, **changes):
    arguments = case["arguments"]
    kwargs = dict(
        owner=case["owner"], verification_catalog=case["verification_catalog"],
        intent_document=arguments["intent_document"], source_text=arguments["source_text"],
    )
    kwargs.update(changes)
    return join.capture_current_finite_proof_query(**kwargs)


def _observe(case, output):
    arguments = case["arguments"]
    return matcher.match_finite_integer_intent(
        index=case["index"], repository=case["repository"],
        repository_id=case["expected_head"].repository_id,
        expected_head=case["expected_head"], intent_document=arguments["intent_document"],
        source_text=arguments["source_text"], output=output,
        tool_policy=case["tools"], scheduler=case["scheduler"],
        timeout_seconds=90, memory_mb=1024,
    )


def _preview(case, match, **changes):
    arguments = case["arguments"]
    kwargs = dict(
        owner=case["owner"], request=case["request"],
        intent_document=arguments["intent_document"], source_text=arguments["source_text"],
        operation_catalog=arguments["operation_catalog"], match=match,
        verification_catalog=case["verification_catalog"],
        policy_observer=lambda request: request.roots,
    )
    kwargs.update(changes)
    return join._preview_owned_finite_proof_join(**kwargs)


def _released(case):
    state = case["scheduler"].snapshot()
    assert state["active_lease_count"] == state["waiting_request_count"] == 0


def _content_bound(record, field):
    assert record[field] == cid_for_structured({
        key: value for key, value in record.items() if key != field
    })


@pytest.fixture(scope="module")
def planning_native(proof_query_native_case):
    case = proof_query_native_case
    match = _observe(case, case["root"] / "planning-finite-observation")
    assert match["eligible_clause_ids"] == [matcher.TYPE_STATEMENT_ID]
    assert match["residual_clause_ids"] == [matcher.OFFSET_STATEMENT_ID]
    assert match["observation"]["kernel_checked_model_table"] is True
    _released(case)
    return case, match


@pytest.fixture(scope="module")
def planning_tamper_case(tmp_path_factory, finite_tools):
    case = build_proof_query_case(
        tmp_path_factory.mktemp("finite-proof-query-planning-controls") / "native", finite_tools,
    )
    try:
        match = _observe(case, case["root"] / "corruption-finite-observation")
        yield case, match
    finally:
        close_proof_query_case(case)


@pytest.fixture(scope="module")
def ambiguous_native_case(tmp_path_factory, finite_tools):
    case = build_proof_query_case(
        tmp_path_factory.mktemp("finite-proof-query-native-ambiguity") / "native", finite_tools,
    )
    try:
        controls = dict(expected_head=case["expected_head"], scheduler=case["scheduler"])
        second = codebase_verification.verify_current_codebase_unit(
            case["index"], case["repository"], path="calc.py", contracts=[case["proof_contract"]],
            bounds=ExecutionBounds(max_steps=99_999), **controls,
        )
        applicability = codebase_applicability.verify_current_codebase_applicability(
            case["index"], case["repository"], verification_cid=second.artifact_cid,
            domains=[case["proof_domain"]], **controls,
        )
        projection = case["verification_catalog"].publish(
            case["repository"], verification_cid=second.artifact_cid,
            applicability_cid=applicability.artifact_cid,
            operation_id="second-real-proof-with-distinct-bounds", **controls,
        )
        assert projection.projection_cid != case["proof_projection"].projection_cid
        assert second.to_dict()["canonical_keys"] != case["verification_record"].to_dict()["canonical_keys"]
        match = _observe(case, case["root"] / "ambiguous-finite-observation")
        yield case, match
    finally:
        close_proof_query_case(case)


@pytest.fixture
def callback_case(tmp_path, finite_tools):
    case = build_proof_query_case(tmp_path / "native", finite_tools)
    try:
        match = _observe(case, case["root"] / "callback-finite-observation")
        yield case, match
    finally:
        close_proof_query_case(case)


def test_derived_contract_and_domain_bind_every_authored_finite_input():
    document = matcher.build_finite_integer_intent(finite_text())
    contract, domain = join.derive_finite_proof_spec(
        intent_document=document, source_text=finite_text(),
    )
    assert type(contract) is ContractSpec and type(domain) is RequestedInputDomain
    assert contract.function_name == domain.function_name == "increment"
    assert contract.postconditions == ("result == n + 2",)
    assert contract.preconditions == ()
    # These predicates are evaluated only to verify this tiny authored input
    # bridge. They are not a runtime implementation or an unrestricted parser.
    assert [n for n in range(-4, 5) if all(
        eval(predicate, {"__builtins__": {}}, {"n": n}) for predicate in domain.predicates
    )] == INPUTS


@pytest.mark.parametrize("damage", ["missing_requirement", "changed_domain", "changed_source"])
def test_invalid_or_misaligned_input_is_rejected_before_catalog_access(
        proof_query_native_case, monkeypatch, damage):
    case = proof_query_native_case
    arguments = case["arguments"]
    document, text = arguments["intent_document"], arguments["source_text"]
    if damage == "missing_requirement":
        document = replace(document, statements=document.statements[:1])
    elif damage == "changed_domain":
        text = text.replace("[-2,-1,0,1,2]", "[-1,0,1]")
    else:
        text = text.replace("n + 2", "n + 3")
    monkeypatch.setattr(CodebaseVerificationCatalog, "query_current", lambda *a, **k:
        pytest.fail("invalid finite input reached the native query"))
    with pytest.raises(ValueError):
        _capture(case, intent_document=document, source_text=text)
    _released(case)


def test_genuine_closure_keeps_native_pages_full_obligations_and_complete_keys(proof_query_native_case):
    case = proof_query_native_case
    closure = _capture(case)
    _content_bound(closure, "closure_cid")
    assert closure["head"] == case["expected_head"].to_dict()
    assert closure["head_cid"] == cid_for_structured(closure["head"])
    assert closure["current_root_id"] == case["expected_head"].snapshot_cid
    assert closure["contract"] == case["proof_contract"].to_dict()
    assert closure["domain"] == case["proof_domain"].to_dict()
    assert closure["contract_cid"] == cid_for_structured(closure["contract"])
    assert closure["domain_cid"] == cid_for_structured(closure["domain"])
    assert closure["verification"] == case["verification_record"].to_dict()
    assert closure["applicability"] == case["applicability_record"].to_dict()
    assert closure["projection"] == case["proof_projection"].to_dict()
    verification, applicability = closure["verification"], closure["applicability"]
    assert len(verification["solver_artifacts"]) == len(verification["canonical_keys"]) == 1
    assert len(applicability["checks"]) == len(applicability["canonical_keys"]) == 4
    assert [row["kind"] for row in applicability["checks"]] == [
        "premises_satisfiable", "domain_satisfiable", "domain_implies_preconditions",
        "property_on_requested_domain",
    ]
    for key in verification["canonical_keys"] + applicability["canonical_keys"]:
        assert set(REQUIRED_IDENTITY_FIELDS) <= set(key)
        native_key = CanonicalProofCacheKey.from_dict(key)
        assert native_key.to_dict() == key
    membership = closure["canonical_key_membership"]
    assert len(membership) == 5
    assert [row["phase"] for row in membership] == ["verification"] + ["applicability"] * 4
    assert [row["key"] for row in membership] == verification["canonical_keys"] + applicability["canonical_keys"]
    assert [row["key_id"] for row in membership] == [
        CanonicalProofCacheKey.from_dict(key).key_id
        for key in verification["canonical_keys"] + applicability["canonical_keys"]
    ]
    assert membership[0]["obligation_id"] == verification["pipeline_result"]["obligation_results"][0]["vc_obligation"]["obligation_id"]
    assert [row["obligation_id"] for row in membership[1:]] == [
        row["obligation"]["smt_obligation"]["obligation_id"] for row in applicability["checks"]
    ]
    assert closure["domain_bridge"] == {
        "schema": "finite-input-to-conditional-domain-bridge@1",
        "finite_query_cid": closure["finite_query"]["query_cid"],
        "finite_domain_cid": closure["finite_query"]["domain_cid"],
        "domain_inputs": INPUTS, "parameter": "n",
        "requested_domain_cid": closure["domain_cid"],
        "predicates": list(case["proof_domain"].predicates),
    }
    assert verification["source_binding"]["entry"]["source_cid"] == applicability["source_binding"]["entry"]["source_cid"]
    for phase in ("discovery_query", "exact_query"):
        page = closure[phase]["page"]
        assert page["complete"] is True and page["next_cursor"] is page["start_cursor"] is None
        assert len(page["entries"]) == 1
        assert page["head"] == closure["head"]
        _content_bound(page, "page_cid")
    assert closure["discovery_query"]["page"]["epoch"] == closure["exact_query"]["page"]["epoch"]
    assert all(value is False for value in closure["authority"].values())
    _released(case)


def test_capture_replays_native_index_without_launching_a_prover(proof_query_native_case, monkeypatch):
    queried = []
    original = CodebaseVerificationCatalog.query_current
    original_popen_init = subprocess.Popen.__init__
    solver_paths = {
        str(Path(pin[field]).resolve())
        for record in (proof_query_native_case["verification_record"],
                       proof_query_native_case["applicability_record"])
        for pin in record.to_dict()["environment"]["executables"]
        for field in ("path", "launcher_path")
    }
    solver_launches = []

    def query(catalog, *args, **kwargs):
        page = original(catalog, *args, **kwargs)
        queried.append((kwargs, page))
        return page

    def popen_init(process, command, *args, **kwargs):
        argv = [command] if isinstance(command, (str, bytes, Path)) else list(command)
        if kwargs.get("executable") is not None:
            argv.append(kwargs["executable"])
        # The bounded runner retains the original Popen class and may prefix
        # its solver argv with prlimit. Inspect every exact executable path
        # while retaining genuine Git observations and the native producers.
        matched = []
        for argument in argv:
            if isinstance(argument, (str, bytes, Path)):
                path = Path(argument.decode() if isinstance(argument, bytes) else argument)
                if path.is_absolute() and str(path.resolve()) in solver_paths:
                    matched.append(str(path.resolve()))
        if matched:
            solver_launches.extend(matched)
            pytest.fail("indexed closure lookup launched a native solver")
        return original_popen_init(process, command, *args, **kwargs)

    monkeypatch.setattr(CodebaseVerificationCatalog, "query_current", query)
    monkeypatch.setattr(subprocess.Popen, "__init__", popen_init)
    closure = _capture(proof_query_native_case)
    assert len(queried) == 2
    assert all(row[0]["page_size"] == 2 and row[0]["cursor"] is None for row in queried)
    assert solver_launches == []
    assert closure["verification_cid"] == proof_query_native_case["verification_record"].artifact_cid
    _released(proof_query_native_case)


def test_owned_native_preview_binds_query_closure_and_keeps_fact_authority(planning_native):
    case, match = planning_native
    result = _preview(case, match)
    closure = result["proof_query_closure"]
    snapshot = PlanCreateInputSnapshot.from_dict(result["input_snapshot"])
    receipt = PlanCreatePreviewReceipt.from_dict(result["preview"])
    assert receipt.input_snapshot_cid == snapshot.snapshot_cid
    assert snapshot.material_binding["reuse_supported"] is True
    assert "extra" in snapshot.material_binding["field_digests"]
    assert result["selected_task_ids"] == [TASK_OFFSET]
    assert len(result["declared_task_requirement_ids"]) == len(result["operation_catalog"]["operations"]) == 2
    assert result["match"] == match
    assert len(match["current_facts"]) == 1
    for row in match["current_facts"]:
        assert ObservedFact.from_dict(row).authority is FactAuthority.BOUNDED_OBSERVATION
    assert match["residual_clause_ids"] == [matcher.OFFSET_STATEMENT_ID]
    assert [row["statement_id"] for row in result["requirement_ledger"]] == match["query"]["requirement_ids"]
    assert [row["status"] for row in result["requirement_ledger"]] == [
        row["status"] for row in match["clause_results"]
    ]
    assert sum(len(row["fact_ids"]) for row in result["requirement_ledger"]) == 1
    assert receipt.read_only is True and receipt.admitted is False and receipt.wrote_effects == ()
    assert result["model_calls"] == result["training_steps"] == result["solver_calls"] == 0
    assert all(value is False for value in closure["authority"].values())
    _released(case)


def test_extra_digest_binds_full_proof_query_content_and_changes_with_one_key(
        planning_native, monkeypatch):
    case, match = planning_native
    captured = []
    original_service = join.FiniteIntegerPlanCreateService

    def service(**kwargs):
        actual = original_service(**kwargs)
        captured.append(actual)
        return actual

    monkeypatch.setattr(join, "FiniteIntegerPlanCreateService", service)
    result = _preview(case, match)
    assert len(captured) == 1
    materials = captured[0].materials
    assert materials.extra[join.MATERIAL_KEY] == result["proof_query_closure"]
    assert materials.extra[join.MATERIAL_KEY + "_cid"] == result["proof_query_closure_cid"]
    assert materials.extra["finite_proof_query_profile"] == join.PROFILE
    snapshot = freeze_plan_create_input_snapshot(case["request"], materials=materials)
    assert snapshot.to_dict() == result["input_snapshot"]
    changed_extra = deepcopy(materials.extra)
    closure = changed_extra[join.MATERIAL_KEY]
    closure["canonical_key_membership"][0]["key"]["bounds"] = "sha256:" + "0" * 64
    closure["closure_cid"] = cid_for_structured({
        key: value for key, value in closure.items() if key != "closure_cid"
    })
    changed_extra[join.MATERIAL_KEY + "_cid"] = closure["closure_cid"]
    changed = freeze_plan_create_input_snapshot(
        case["request"], materials=replace(materials, extra=changed_extra),
    )
    assert changed.snapshot_cid != snapshot.snapshot_cid
    assert changed.material_binding["field_digests"]["extra"] != snapshot.material_binding["field_digests"]["extra"]
    assert {key: value for key, value in changed.material_binding["field_digests"].items() if key != "extra"} == {
        key: value for key, value in snapshot.material_binding["field_digests"].items() if key != "extra"
    }
    _released(case)


@pytest.mark.parametrize("case_name", ["ambiguous", "incomplete"])
def test_actual_native_non_singleton_or_paginated_inventory_refuses_before_plan(
        ambiguous_native_case, monkeypatch, case_name):
    case, match = ambiguous_native_case
    pages = []
    original = CodebaseVerificationCatalog.query_current

    def query(catalog, *args, **kwargs):
        if case_name == "incomplete":
            kwargs = {**kwargs, "page_size": 1}
        page = original(catalog, *args, **kwargs)
        pages.append(page)
        return page

    monkeypatch.setattr(CodebaseVerificationCatalog, "query_current", query)
    monkeypatch.setattr(join, "FiniteIntegerPlanCreateService", lambda **kwargs:
        pytest.fail("incomplete or ambiguous native proof population reached planning"))
    with pytest.raises(join.FiniteProofQueryJoinError):
        _preview(case, match)
    assert len(pages) == 1
    if case_name == "incomplete":
        assert len(pages[0].entries) == 1 and pages[0].next_cursor is not None
        assert pages[0].complete is False
    else:
        assert len(pages[0].entries) == 2 and pages[0].complete is True
    _released(case)


@pytest.mark.parametrize("damage", ["source", "contract", "domain", "full_key"])
def test_native_normalized_inventory_corruption_refuses_before_planning(
        planning_tamper_case, monkeypatch, damage):
    case, match = planning_tamper_case
    cx = case["connection"]
    if damage == "source":
        table, column, predicate = "codebase_verification_query.dependencies", "value", "kind='source'"
        changed = case["index"].artifacts.put_bytes(b"a different captured source\n")
    elif damage in {"contract", "domain"}:
        table, column, predicate = "codebase_verification_query.entries", damage + "_cid", "TRUE"
        changed = cid_for_structured({"different": damage})
    else:
        table, column, predicate = "codebase_verification_query.keys", "key_json", "phase='verification'"
        raw = cx.execute(f"SELECT {column} FROM {table} WHERE {predicate}").fetchone()[0]
        key = json.loads(raw)
        key["bounds"] = "sha256:" + "0" * 64
        changed = canonical_dag_json_bytes(key).decode()
    originals = cx.execute(f"SELECT * FROM {table} WHERE {predicate}").fetchall()
    targeted = cx.execute(f"SELECT entry_id,{column} FROM {table} WHERE {predicate}").fetchall()
    assert len(originals) == len(targeted) == 1
    entry_id, original_value = targeted[0]
    monkeypatch.setattr(join, "FiniteIntegerPlanCreateService", lambda **kwargs:
        pytest.fail("corrupted native proof inventory reached planning"))
    try:
        cx.execute(f"UPDATE {table} SET {column}=? WHERE entry_id=? AND {column}=? AND {predicate}",
                   [changed, entry_id, original_value])
        with pytest.raises((ValueError, CodebaseVerificationCatalogError)):
            _preview(case, match)
    finally:
        cx.execute(f"UPDATE {table} SET {column}=? WHERE entry_id=? AND {column}=? AND {predicate}",
                   [original_value, entry_id, changed])
    assert cx.execute(f"SELECT * FROM {table} WHERE {predicate}").fetchall() == originals
    assert _capture(case)["verification_cid"] == case["verification_record"].artifact_cid
    _released(case)


def test_caller_cannot_upgrade_genuinely_observed_fact_to_proved(planning_native):
    case, match = planning_native
    forged = deepcopy(match)
    forged["current_facts"][0]["authority"] = FactAuthority.PROOF_RECEIPT.value
    forged["match_cid"] = cid_for_structured({key: value for key, value in forged.items() if key != "match_cid"})
    with pytest.raises(ValueError):
        _preview(case, forged)
    _released(case)


@pytest.mark.parametrize("mutation", ["source_literal", "normalized_epoch"])
def test_genuine_policy_callback_mutation_refuses_the_proof_join(callback_case, mutation):
    case, match = callback_case
    closure_before = _capture(case)
    path = case["repository"] / "calc.py"
    original_source = path.read_bytes()
    changed = []

    def policy(request):
        if not changed:
            if mutation == "source_literal":
                path.write_bytes(finite_source(2))
                changed.append({"sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            else:
                inventory = case["verification_catalog"].rebuild_current(
                    case["repository"], expected_head=case["expected_head"], scheduler=case["scheduler"],
                )
                changed.append(inventory)
        return request.roots

    try:
        with pytest.raises((ValueError, StaleCodebaseError)):
            _preview(case, match, policy_observer=policy)
        assert len(changed) == 1
    finally:
        if mutation == "source_literal":
            path.write_bytes(original_source)
    closure_after = _capture(case)
    if mutation == "normalized_epoch":
        assert closure_after["discovery_query"]["page"]["inventory_cid"] != closure_before["discovery_query"]["page"]["inventory_cid"]
        assert closure_after["discovery_query"]["page"]["epoch"] > closure_before["discovery_query"]["page"]["epoch"]
        assert closure_after["closure_cid"] != closure_before["closure_cid"]
    else:
        assert closure_after["closure_cid"] == closure_before["closure_cid"]
    assert match["eligible_clause_ids"] == [matcher.TYPE_STATEMENT_ID]
    _released(case)
