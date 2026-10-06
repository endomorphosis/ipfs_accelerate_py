"""Closed metadata for rejected signed task contracts; no provider required."""

from copy import deepcopy
from dataclasses import replace
import json
import subprocess
from types import SimpleNamespace

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario as scenario
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import (
    original as original, _proposal_graph, _proposal_json,
)
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local


REJECTION = "proposal changed signed scope, acceptance, dependency or command"
PRIVATE = "PRIVATE-proposed-body-not-for-diagnostic"


def _rows(error):
    assert type(error) is local.LocalPlanningError
    assert error.args == (REJECTION,)
    projection = local.project_task_contract_mismatch(error)
    assert projection["schema"] == "supervisor-local-task-contract-mismatch@1"
    assert PRIVATE not in json.dumps(projection)
    return {row["field"]: row for row in projection["fields"]}


@pytest.mark.parametrize("field,expected_count,observed_count,members", [
    ("scope_paths", 2, 1, []),
    ("outputs", 1, 1, ["media_type"]),
    ("validations", 1, 1, ["argv"]),
    ("acceptance", 1, 1, ["criterion"]),
    ("assumptions", 0, 1, []),
])
def test_real_signed_admission_retains_exact_rejected_field_without_bodies(
        scenario, field, expected_count, observed_count, members):
    graph = scenario["graph"]
    task = graph.tasks[0]
    if field == "scope_paths":
        changed = (PRIVATE,)
    elif field == "outputs":
        changed = (replace(task.outputs[0], media_type=PRIVATE),)
    elif field == "validations":
        changed = (replace(task.validations[0], argv=("python3", "-c", PRIVATE)),)
    elif field == "acceptance":
        changed = (replace(task.acceptance[0], criterion=PRIVATE),)
    elif field == "assumptions":
        changed = (PRIVATE,)
    else:
        changed = (local.content_identity({"private": PRIVATE}),)
    graph = replace(graph, tasks=(replace(task, **{field: changed}),))
    watermark = scenario["intent"].event_watermark()
    with pytest.raises(local.LocalPlanningError, match="changed signed") as captured:
        local.admit_local_benchmark_plan(graph=graph, manifest=scenario["manifest"])
    rows = _rows(captured.value)
    assert set(rows) == {field}
    assert rows[field]["expected_count"] == expected_count
    assert rows[field]["observed_count"] == observed_count
    assert rows[field]["changed_members"] == members
    assert rows[field]["counts_capped"] is False
    assert scenario["intent"].event_watermark() == watermark
    # Neither content hashes nor path/command/criterion values are disclosed.
    serialized = json.dumps(rows)
    assert "answer.py" not in serialized and "public-answer" not in serialized
    assert local.content_identity({"private": PRIVATE}) not in serialized


def test_missing_task_policy_root_is_rejected_after_typed_graph_validation(original):
    prepared = prep.prepare(repository=original[0], instruction=original[1], state=original[2])
    graph = _proposal_graph(prepared)
    task = graph.tasks[0]
    assert len(task.policy_roots) == len(graph.policy_roots) > 1
    graph = replace(graph, tasks=(replace(task, policy_roots=(task.policy_roots[0],)),))
    with pytest.raises(local.LocalPlanningError, match="changed signed") as captured:
        local.admit_local_benchmark_plan(graph=graph, manifest=prepared["manifest"])
    rows = _rows(captured.value)
    assert rows == {"policy_roots": {"field": "policy_roots", "comparison": "ordered_equal",
        "expected_count": len(task.policy_roots), "observed_count": 1,
        "counts_capped": False, "changed_members": []}}
    assert all(cid not in json.dumps(rows) for cid in task.policy_roots)


def test_unallowed_evidence_diagnostic_is_closed_and_subset_semantics_are_preserved():
    # Typed graph construction already rejects unknown evidence references;
    # exercise the defensive task-level diagnostic without bypassing that gate.
    contract = {field: [] for field in ("scope_paths", "outputs", "validations", "acceptance", "dependencies")}
    task = SimpleNamespace(assumptions=(), evidence_cids=(PRIVATE,), policy_roots=("policy",))
    error = local.LocalPlanningError(REJECTION)
    error.task_contract_mismatch = local._task_contract_mismatch(task=task, expected=contract,
        observed=contract, policy_roots=("policy",), allowed_evidence=(), strict=True)
    rows = _rows(error)
    assert rows == {"evidence_cids": {"field": "evidence_cids", "comparison": "allowed_subset",
        "expected_count": 0, "observed_count": 1, "counts_capped": False, "changed_members": []}}
    task.evidence_cids = ()
    assert local._task_contract_mismatch(task=task, expected=contract, observed=contract,
        policy_roots=("policy",), allowed_evidence=(SimpleNamespace(evidence_cid=PRIVATE),),
        strict=True).fields == ()


@pytest.mark.parametrize("field", ["evidence_cids", "policy_roots"])
def test_unknown_graph_references_still_fail_before_task_contract_diagnostics(scenario, field):
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import PromptGraphError

    task = replace(scenario["graph"].tasks[0], **{field: (local.content_identity({"private": PRIVATE}),)})
    with pytest.raises(PromptGraphError) as captured:
        replace(scenario["graph"], tasks=(task,))
    assert local.project_task_contract_mismatch(captured.value) is None


def test_dependency_mismatch_retains_counts_without_task_keys_or_cids(scenario):
    graph = scenario["graph"]
    first = graph.tasks[0]
    second = replace(first, task_key="PRIVATE-second-task")
    manifest = deepcopy(scenario["manifest"]["payload"])
    other = deepcopy(manifest["tasks"][0])
    other["task_key"] = second.task_key
    manifest["tasks"].append(other)
    first = replace(first, dependency_task_cids=(second.task_cid,))
    graph = replace(graph, tasks=(first, second))
    manifest = local._signed(manifest, manifest)
    with pytest.raises(local.LocalPlanningError, match="changed signed") as captured:
        local.admit_local_benchmark_plan(graph=graph, manifest=manifest)
    rows = _rows(captured.value)
    assert rows == {"dependencies": {"field": "dependencies", "comparison": "unordered_equal",
        "expected_count": 0, "observed_count": 1, "counts_capped": False, "changed_members": []}}
    assert second.task_key not in json.dumps(rows) and second.task_cid not in json.dumps(rows)


def test_multiple_changed_fields_are_retained_at_first_rejected_task(scenario):
    graph = scenario["graph"]
    task = graph.tasks[0]
    changed = replace(task, scope_paths=(PRIVATE,), assumptions=(PRIVATE,),
        validations=(replace(task.validations[0], argv=("python3", PRIVATE), cwd=PRIVATE),),
        outputs=(replace(task.outputs[0], path=PRIVATE, media_type=PRIVATE),),
        acceptance=(replace(task.acceptance[0], criterion=PRIVATE),))
    with pytest.raises(local.LocalPlanningError, match="changed signed") as captured:
        local.admit_local_benchmark_plan(graph=replace(graph, tasks=(changed,)), manifest=scenario["manifest"])
    rows = _rows(captured.value)
    assert set(rows) == {"scope_paths", "outputs", "validations", "acceptance", "assumptions"}
    assert rows["outputs"]["changed_members"] == ["path", "media_type"]
    assert rows["validations"]["changed_members"] == ["argv", "cwd"]


def test_reordered_signed_collections_still_admit_and_have_no_diagnostic(scenario):
    payload = deepcopy(scenario["manifest"]["payload"])
    payload["tasks"][0]["scope_paths"].reverse()
    signed = local._signed(payload, payload)
    admitted = local.admit_local_benchmark_plan(graph=scenario["graph"], manifest=signed)
    assert admitted["receipt"]["payload"]["planning_permitted"] is True
    assert "task_contract_mismatch" not in admitted["receipt"]["payload"]


def _diagnostic_error():
    error = local.LocalPlanningError(REJECTION)
    error.task_contract_mismatch = local.TaskContractMismatch((local.TaskContractFieldMismatch(
        field="validations", comparison="unordered_equal", expected_count=1,
        observed_count=1, counts_capped=False, changed_members=("argv",)),))
    return error


@pytest.mark.parametrize("mutation", [
    "foreign_error", "subclass", "other_message", "extra_error_arg", "mapping", "empty",
    "too_many", "list_fields", "mapping_row", "unknown_field", "duplicate_field",
    "wrong_comparison", "bool_count", "negative_count", "large_count", "float_count",
    "string_count", "bad_cap", "unsupported_cap", "list_members", "unknown_member",
    "foreign_member", "duplicate_member", "nested_member", "too_many_members",
])
def test_projection_rejects_untyped_or_unbounded_metadata(mutation):
    error = _diagnostic_error()
    original = error.task_contract_mismatch
    row = original.fields[0]
    if mutation == "foreign_error":
        error = ValueError(REJECTION)
        error.task_contract_mismatch = original
    elif mutation == "subclass":
        class Foreign(local.LocalPlanningError):
            pass
        error = Foreign(REJECTION)
        error.task_contract_mismatch = original
    elif mutation == "other_message":
        error.args = (PRIVATE,)
    elif mutation == "extra_error_arg":
        error.args = (REJECTION, PRIVATE)
    elif mutation == "mapping":
        error.task_contract_mismatch = {"fields": [PRIVATE]}
    elif mutation in {"empty", "too_many", "list_fields", "mapping_row", "duplicate_field"}:
        fields = {"empty": (), "too_many": (row,) * 9, "list_fields": [row],
            "mapping_row": ({"field": PRIVATE},), "duplicate_field": (row, row)}[mutation]
        error.task_contract_mismatch = local.TaskContractMismatch(fields)
    else:
        changes = {
            "unknown_field": {"field": PRIVATE},
            "wrong_comparison": {"comparison": PRIVATE},
            "bool_count": {"observed_count": True},
            "negative_count": {"observed_count": -1},
            "large_count": {"observed_count": 65536},
            "float_count": {"observed_count": 1.0},
            "string_count": {"observed_count": PRIVATE},
            "bad_cap": {"counts_capped": 1},
            "unsupported_cap": {"counts_capped": True},
            "list_members": {"changed_members": ["argv"]},
            "unknown_member": {"changed_members": (PRIVATE,)},
            "foreign_member": {"changed_members": ("criterion",)},
            "duplicate_member": {"changed_members": ("argv", "argv")},
            "nested_member": {"changed_members": ({"argv": PRIVATE},)},
            "too_many_members": {"changed_members": ("argv",) * 6},
        }[mutation]
        error.task_contract_mismatch = local.TaskContractMismatch((replace(row, **changes),))
    assert local.project_task_contract_mismatch(error) is None


def test_diagnostic_counts_are_capped_and_projection_is_detached():
    task = SimpleNamespace(assumptions=(PRIVATE,) * 65536, evidence_cids=(), policy_roots=("policy",))
    contract = {field: [] for field in ("scope_paths", "outputs", "validations", "acceptance", "dependencies")}
    error = local.LocalPlanningError(REJECTION)
    error.task_contract_mismatch = local._task_contract_mismatch(task=task, expected=contract,
        observed=contract, policy_roots=("policy",), allowed_evidence=(), strict=False)
    rows = _rows(error)
    assert rows["assumptions"]["expected_count"] == 0
    assert rows["assumptions"]["observed_count"] == 65535
    assert rows["assumptions"]["counts_capped"] is True
    rows["assumptions"]["changed_members"].append(PRIVATE)
    assert _rows(error)["assumptions"]["changed_members"] == []


@pytest.mark.parametrize("strict", [False, True])
def test_inventory_diagnostic_uses_existing_type_exact_equality(strict):
    task = SimpleNamespace(assumptions=(), evidence_cids=(), policy_roots=("policy",))
    expected = {field: [] for field in ("scope_paths", "outputs", "validations", "acceptance", "dependencies")}
    expected["validations"] = [{"validation_key": "check", "argv": ["python3"], "cwd": ".",
        "expected_exit_codes": [0], "policy_cid": "policy"}]
    observed = deepcopy(expected)
    observed["validations"][0]["expected_exit_codes"] = [False]
    diagnostic = local._task_contract_mismatch(task=task, expected=expected, observed=observed,
        policy_roots=("policy",), allowed_evidence=(), strict=strict)
    if strict:
        error = local.LocalPlanningError(REJECTION)
        error.task_contract_mismatch = diagnostic
        assert _rows(error)["validations"]["changed_members"] == ["expected_exit_codes"]
    else:
        assert diagnostic.fields == ()


def test_record_association_change_identifies_field_without_claiming_member_alignment():
    task = SimpleNamespace(assumptions=(), evidence_cids=(), policy_roots=("policy",))
    expected = {field: [] for field in ("scope_paths", "outputs", "validations", "acceptance", "dependencies")}
    expected["outputs"] = [{"path": "a", "effect": "modify", "media_type": "text/plain"},
        {"path": "b", "effect": "create", "media_type": "text/plain"}]
    observed = deepcopy(expected)
    observed["outputs"][0]["effect"], observed["outputs"][1]["effect"] = "create", "modify"
    error = local.LocalPlanningError(REJECTION)
    error.task_contract_mismatch = local._task_contract_mismatch(task=task, expected=expected,
        observed=observed, policy_roots=("policy",), allowed_evidence=(), strict=True)
    rows = _rows(error)
    assert set(rows) == {"outputs"} and rows["outputs"]["changed_members"] == []


def test_provider_plan_persists_rejection_diagnostic_without_admission_or_materialization(original, monkeypatch):
    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    native_run = subprocess.run
    def reported_version(argv, *args, **kwargs):
        if argv == ["codex", "--version"]:
            return subprocess.CompletedProcess(argv, 0, "codex-cli 0.160.0\n", "")
        return native_run(argv, *args, **kwargs)
    monkeypatch.setattr(subprocess, "run", reported_version)
    calls = []
    monkeypatch.setattr(local, "materialize_local_benchmark_plan", lambda **kwargs: calls.append(kwargs))
    proposal = json.loads(_proposal_json(prepared))
    proposal["tasks"][0]["acceptance"][0]["criterion"] = PRIVATE
    result = prep.plan(state, provider_callable=lambda *args, **kwargs: {
        "text": json.dumps(proposal), "observation": {}, "execution_receipt": None})
    assert result["qualified"] is False
    assert result["failure"] == {"type": "LocalPlanningError", "message": REJECTION}
    assert result["provider_calls"] == 1 and result["provider_receipt"]["outcome"] == "provider"
    diagnostic = result["task_contract_mismatch"]
    assert diagnostic["fields"] == [{"field": "acceptance", "comparison": "unordered_equal",
        "expected_count": 1, "observed_count": 1, "counts_capped": False, "changed_members": ["criterion"]}]
    assert json.loads((state / "planning-result.json").read_text())["task_contract_mismatch"] == diagnostic
    assert PRIVATE not in json.dumps(diagnostic)
    assert calls == [] and not (state / "admission.json").exists()
    assert not (state / "intent.duckdb").exists()


@pytest.mark.parametrize("typed", [False, True])
def test_symbolic_failure_persists_only_typed_bounded_metadata(original, monkeypatch, typed):
    from ipfs_accelerate_py.agent_supervisor.planning import intent_symbolic_planning

    root, instruction, state = original
    prepared = prep.prepare(repository=root, instruction=instruction, state=state)
    # An explicit adapter fixture: the real admission path is exercised above.
    prepared["intent_requirement_contract"] = {"schema": "intent-plan-requirement-contract@2"}
    monkeypatch.setattr(prep, "_load_prepared", lambda path: prepared)
    error = _diagnostic_error()
    if not typed:
        error.task_contract_mismatch = {"field": PRIVATE}
    def rejected(*args, **kwargs):
        raise error
    monkeypatch.setattr(intent_symbolic_planning, "build_intent_symbolic_plan", rejected)
    result = prep._plan_symbolic(state=state, prepared=prepared, initial=None, timeout_seconds=30)
    assert result["qualified"] is False and result["provider_calls"] == 0
    assert result["failure"] == {"type": "LocalPlanningError", "message": REJECTION}
    saved = json.loads((state / "planning-result.json").read_text())
    if typed:
        assert saved["task_contract_mismatch"] == local.project_task_contract_mismatch(error)
    else:
        assert "task_contract_mismatch" not in result and "task_contract_mismatch" not in saved
    assert not (state / "admission.json").exists() and not (state / "intent.duckdb").exists()
