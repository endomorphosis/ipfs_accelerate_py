"""Declared planning composes with signed selected context and full custody.

Git, signing, vector/semantic producers, admission and Intent are real. The
source programs and public requirement are authored; no model, target program,
training, Docker, native coding child or official verifier is dispatched.
"""
from copy import deepcopy
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
from benchmarks.agent_supervisor.container_coding.test_terminal_declared_contract_planning import no_cli, forbid_provider
from benchmarks.agent_supervisor.container_coding.test_terminal_signed_semantic_scope import make_case
from benchmarks.agent_supervisor.container_coding.terminal_planning_policy import DECLARED_PLANNING_MODE
from ipfs_accelerate_py.agent_supervisor.runtime import declared_contract_planning as declared
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.terminal_semantic_scope import replay_terminal_semantic_scope
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


@pytest.fixture
def selected_declared(tmp_path, monkeypatch):
    native_prepare = prep.prepare

    def explicit_declared_prepare(*args, **kwargs):
        assert "planning_mode" not in kwargs
        return native_prepare(*args, **kwargs, planning_mode=DECLARED_PLANNING_MODE)

    monkeypatch.setattr(prep, "prepare", explicit_declared_prepare)
    case = make_case(tmp_path)
    assert case.prepared["planning_mode"] == DECLARED_PLANNING_MODE
    assert case.prepared["planning_strategy"] == "declared_contract"
    assert case.prepared["request"]["planning_policy"]["policy_id"] == declared.POLICY_ID
    assert case.prepared["request"]["planning_policy"]["allow_model"] is False
    no_cli(monkeypatch)
    return case


def test_declared_plan_keeps_full_signed_contract_and_reuses_exact_selected_context(selected_declared):
    case = selected_declared
    receipt = prep.initial_context(state=case.state)
    loaded = initial.load_initial_context(state=case.state, prepared=case.prepared, require_empty_owner=True)
    selection = replay_terminal_semantic_scope(case.prepared)
    manifest = deepcopy(case.prepared["manifest"])
    spec = deepcopy(case.prepared["spec"])
    assert len(manifest["payload"]["sources"]) == len(case.prepared["worker_inputs"]) == 73
    assert (selection["full_count"], selection["selected_count"], selection["frontier_count"]) == (73, 5, 68)
    assert set(loaded["semantic"]["manifest"]) == set(selection["selected_paths"])
    assert loaded["semantic"]["program_paths"] == case.selected
    assert set(loaded["indexed"]["source_sha256"]) == set(case.selected)
    semantic_path = case.root/loaded["descriptor"]["metadata"]["Semantic context artifact"]
    vector_path = case.root/".runtime/terminal-vectors/result.json"
    frozen_artifacts = (semantic_path.read_bytes(), vector_path.read_bytes())
    result = prep.plan(case.state, provider_callable=forbid_provider)
    assert result["qualified"] is True, result
    assert result["provider_calls"] == 0 and result["provider_receipt"] is None
    assert result["execution_receipt"] is result["model_request_sha256"] is None
    assert (result["goals"], result["tasks"]) == (2, 1)
    assert result["initial_indexed_context"]["supplied_to_router"] is False
    assert result["initial_indexed_context"]["used_for_task_selection"] is False
    assert result["initial_indexed_context"]["index_id"] == receipt["index_id"]
    mechanical = result["declared_contract_planning"]
    assert mechanical["formal_plan_consistent"] is True
    assert mechanical["manifest_cid"] == local.content_identity(manifest)
    assert mechanical["source_obligations_discharged"] == mechanical["provider_calls"] == mechanical["model_calls"] == 0
    for key in ("instruction_interpreted", "source_semantics_verified", "semantic_alignment_verified",
                "proof_authority", "execution_authority", "completion_authority"):
        assert result[key] is mechanical[key] is False
    admission = json.loads((case.state/"admission.json").read_bytes())
    checked = local.verify_local_benchmark_admission(admission, initial=True)
    assert checked["manifest"] == manifest["payload"]
    assert checked["receipt"]["pending_requirements"]
    assert all(row["phase"] == "post_execution" for row in checked["receipt"]["pending_requirements"])
    task = checked["graph"].tasks[0]
    assert task.task_key == spec["task_key"] and list(task.scope_paths) == sorted(spec["scope_paths"])
    for name in ("outputs", "validations", "acceptance"):
        expected = spec[name]
        actual = [row.to_dict() for row in getattr(task, name)]
        assert len(actual) == len(expected)
        assert [{key: row[key] for key in authored} for row, authored in zip(actual, expected)] == expected
    context = prep.context(state=case.state)
    assert context["terminal_semantic_scope"] == selection
    assert context["initial_indexes_reused"] is True and context["new_embedding_calls"] == 0
    assert context["provider_calls"] == 0 and context["nomination_only"] is True
    assert context["semantic_equivalence_claimed"] is False
    assert loaded["semantic"]["completion_authority"] is False
    assert loaded["retrieval"]["nomination_only"] is True
    assert all(loaded["retrieval"][key] is False for key in (
        "semantic_authority", "execution_authority", "completion_authority"))
    assert all(context["terminal_semantic_scope"][key] is False for key in (
        "proof_authority", "execution_authority", "completion_authority", "omission_authority"))
    assert context["doctor_repair"]["completion_authority"] is False
    assert context["semantic_root_cid"] == receipt["semantic_root_cid"]
    assert context["index_id"] == receipt["index_id"]
    assert context["doctor_repair"]["automatic_repair_eligible"] is False
    assert (semantic_path.read_bytes(), vector_path.read_bytes()) == frozen_artifacts
    assert prep._load_prepared(case.state) == case.prepared
    assert case.prepared["manifest"] == manifest and case.prepared["spec"] == spec
    assert {name:(case.root/name).read_bytes() for name in case.source} == case.source
    with IntentRepository(case.state/"intent.duckdb", install_schema=False) as intent:
        projection = intent.plan_projection()
    assert len(projection["tasks"]) == 1 and len(projection["goals"]) == 2
    assert projection["tasks"][0]["status"] == "ready"
    assert not (case.state/"provider-request.json").exists()
    assert not (case.state/"planner-provider-request.json").exists()
    assert not (case.state/"provider-response.txt").exists()


def test_unsigned_mode_downgrade_refuses_before_plan_or_admission(selected_declared):
    case = selected_declared
    prep.initial_context(state=case.state)
    changed = deepcopy(case.prepared)
    changed["planning_mode"] = "model"
    changed["planning_strategy"] = "direct"
    prep._write(case.state/"prepared.json", changed)
    with pytest.raises(ValueError, match="signed planning policy"):
        prep.plan(case.state, provider_callable=forbid_provider)
    assert not (case.state/"planner-invoked.json").exists()
    assert not (case.state/"admission.json").exists()
    with IntentRepository(case.state/"intent.duckdb", install_schema=False) as intent:
        assert intent.plan_projection()["tasks"] == []


def test_current_omitted_source_drift_refuses_before_declared_plan(selected_declared):
    case = selected_declared
    prep.initial_context(state=case.state)
    path = case.root/"module_069.py"
    path.write_bytes(path.read_bytes()+b"\n# changed after initial selection\n")
    with pytest.raises(ValueError):
        prep.plan(case.state, provider_callable=forbid_provider)
    assert not (case.state/"planner-invoked.json").exists()
    assert not (case.state/"admission.json").exists()
    assert len(case.prepared["manifest"]["payload"]["sources"]) == 73
    with IntentRepository(case.state/"intent.duckdb", install_schema=False) as intent:
        assert intent.plan_projection()["tasks"] == []
