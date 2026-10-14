"""Real signed preparation/admission/storage without a planning provider."""
from copy import deepcopy
from dataclasses import replace
import json
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.test_terminal_task_profile import original
from benchmarks.agent_supervisor.container_coding.terminal_planning_policy import DECLARED_PLANNING_MODE
from ipfs_accelerate_py.agent_supervisor.runtime import declared_contract_planning as declared
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


def prepare(tmp_path, *, mode=DECLARED_PLANNING_MODE, modify=False):
    root, instruction, state, profile = original(tmp_path)
    if modify:
        profile["outputs"][0].update(path="source.py", effect="modify")
    prepared = prep.prepare(repository=root, instruction=instruction, state=state,
        task_profile=profile, disable_intent_autoencoder=True, planning_mode=mode)
    return root, instruction, state, prepared


def forbid_provider(*args, **kwargs):
    pytest.fail("declared-contract planning must not invoke a provider")


def no_cli(monkeypatch):
    real = subprocess.run
    def run(argv, *args, **kwargs):
        if str(argv[0]).endswith(("codex", "grok")):
            pytest.fail("declared-contract planning must not query a model CLI")
        return real(argv, *args, **kwargs)
    monkeypatch.setattr(subprocess, "run", run)


@pytest.mark.parametrize("modify", [False, True])
def test_exact_declared_plan_admitted_and_materialized_without_provider(tmp_path, monkeypatch, modify):
    root, instruction, state, prepared = prepare(tmp_path, modify=modify)
    no_cli(monkeypatch)
    assert prepared["planning_mode"] == DECLARED_PLANNING_MODE
    assert prepared["planning_strategy"] == "declared_contract"
    assert prepared["request"]["planning_policy"]["allow_model"] is False
    assert prepared["request"]["planning_policy"]["policy_id"] == declared.POLICY_ID
    assert (root / prep.INSTRUCTION).read_text() == instruction.read_text() == prepared["query"]
    assert "intent_requirement_contract" not in prepared
    assert not (state / "provider-request.json").exists()
    candidate = declared.build_declared_contract_plan(manifest=prepared["manifest"])
    assert candidate == declared.build_declared_contract_plan(manifest=prepared["manifest"])
    result = prep.plan(state, provider_callable=forbid_provider)
    assert result["qualified"] and result["provider_calls"] == 0
    assert result["goals"] == 2 and result["tasks"] == 1
    assert result["execution_receipt"] is result["model_request_sha256"] is None
    assert result["declared_contract_planning"] == candidate["receipt"]
    assert result["declared_contract_planning"]["formal_plan_consistent"] is True
    for key in ("instruction_interpreted", "source_semantics_verified", "proof_authority",
                "execution_authority", "completion_authority"):
        assert result[key] is False and candidate["receipt"][key] is False
    admission = json.loads((state / "admission.json").read_text())
    checked = local.verify_local_benchmark_admission(admission, initial=True)
    assert checked["graph"] == candidate["graph"]
    assert checked["receipt"]["pending_requirements"]
    assert all(row["phase"] == "post_execution" for row in checked["receipt"]["pending_requirements"])
    assert checked["receipt"]["completion_authority"] is False
    task = checked["graph"].tasks[0]
    for name in ("outputs", "validations", "acceptance"):
        expected = prepared["spec"][name]
        actual = [x.to_dict() for x in getattr(task, name)]
        assert [{key: row[key] for key in spec} for row, spec in zip(actual, expected)] == expected
    with IntentRepository(state / "intent.duckdb", install_schema=False) as intent:
        projected = intent.plan_projection()
    assert len(projected["tasks"]) == 1 and len(projected["goals"]) == 2
    assert not (state / "planner-provider-request.json").exists()
    assert not (state / "provider-response.txt").exists()
    assert json.loads((state / "planning-result.json").read_text()) == result
    with pytest.raises(FileExistsError):
        prep.plan(state, provider_callable=forbid_provider)


def test_planning_callable_is_unnecessary_when_explicit_signed_mode_selected(tmp_path, monkeypatch):
    _, _, state, _ = prepare(tmp_path)
    no_cli(monkeypatch)
    assert prep.plan(state)["qualified"] is True


def test_model_default_is_preserved_and_cannot_silently_use_declared_helper(tmp_path, monkeypatch):
    _, _, state, prepared = prepare(tmp_path, mode="model")
    assert "planning_mode" not in prepared
    assert prepared["planning_strategy"] == "direct"
    assert prepared["request"]["planning_policy"]["allow_model"] is True
    assert (state / "provider-request.json").exists()
    with pytest.raises(local.LocalPlanningError, match="signed planning policy"):
        declared.build_declared_contract_plan(manifest=prepared["manifest"])
    with pytest.raises(ValueError, match="isolated worker router callable"):
        prep.plan(state)


@pytest.mark.parametrize("mutation", ["remove_mode", "model_mode", "strategy", "request_policy", "signed_policy"])
def test_prepared_mode_cannot_override_signed_policy(tmp_path, mutation):
    _, _, state, prepared = prepare(tmp_path)
    altered = deepcopy(prepared)
    if mutation == "remove_mode":
        altered.pop("planning_mode")
    elif mutation == "model_mode":
        altered["planning_mode"] = "model"
    elif mutation == "strategy":
        altered["planning_strategy"] = "direct"
    elif mutation == "request_policy":
        altered["request"]["planning_policy"]["allow_model"] = True
    else:
        altered["manifest"]["payload"]["planning_inputs"]["request"]["planning_policy"]["allow_model"] = True
    prep._write(state / "prepared.json", altered)
    with pytest.raises(ValueError):
        prep.plan(state, provider_callable=forbid_provider)
    assert not (state / "admission.json").exists()


@pytest.mark.parametrize("path", ["source.py", prep.INSTRUCTION, prep.SMOKE])
def test_changed_signed_source_refuses_before_skeleton_or_model(tmp_path, path):
    root, _, state, _ = prepare(tmp_path)
    with (root / path).open("a") as stream:
        stream.write("\n# changed\n")
    with pytest.raises(ValueError):
        prep.plan(state, provider_callable=forbid_provider)
    assert not (state / "planner-invoked.json").exists()
    assert not (state / "admission.json").exists()


@pytest.mark.parametrize("mode", [None, "declared-contract@2", "", True])
def test_invalid_mode_rejected_before_mutation(tmp_path, mode):
    root, instruction, state, profile = original(tmp_path)
    before = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"])
    with pytest.raises(ValueError):
        prep.prepare(repository=root, instruction=instruction, state=state,
            task_profile=profile, planning_mode=mode)
    assert not state.exists() and not (root / prep.INSTRUCTION).exists()
    assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"]) == before


def test_explicit_intent_contract_cannot_be_bypassed(tmp_path):
    root, instruction, state, profile = original(tmp_path)
    with pytest.raises(ValueError):
        prep.prepare(repository=root, instruction=instruction, state=state,
            task_profile=profile, planning_mode=DECLARED_PLANNING_MODE,
            intent_requirement_contract=tmp_path / "not-opened.json")
    assert not state.exists() and not (root / prep.INSTRUCTION).exists()


def test_multitask_rejected_before_preprocessing_or_signing(tmp_path):
    root, instruction, state, profile = original(tmp_path)
    profile["schema"] = prep.MULTITASK_SCHEMA
    with pytest.raises(ValueError, match="one independent signed task"):
        prep.prepare(repository=root, instruction=instruction, state=state,
            task_profile=profile, planning_mode=DECLARED_PLANNING_MODE)
    assert not state.exists() and not (root / prep.INSTRUCTION).exists()


def test_independent_admission_rejects_candidate_drift(tmp_path, monkeypatch):
    _, _, state, _ = prepare(tmp_path)
    build = declared.build_declared_contract_plan
    def drift(**kwargs):
        value = build(**kwargs)
        task = value["graph"].tasks[0]
        # Semantically changing the accepted criterion is refused even though
        # all signed inputs and the initially compiled candidate were valid.
        acceptance = replace(task.acceptance[0], criterion="invented weaker check")
        value["graph"] = replace(value["graph"], tasks=(replace(task, acceptance=(acceptance,)),))
        return value
    monkeypatch.setattr(declared, "build_declared_contract_plan", drift)
    result = prep.plan(state, provider_callable=forbid_provider)
    assert result["qualified"] is False and result["provider_calls"] == 0
    assert result["task_contract_mismatch"]
    assert not (state / "admission.json").exists()


def test_source_change_during_construction_refuses_admission(tmp_path, monkeypatch):
    root, _, state, _ = prepare(tmp_path)
    build = declared.build_declared_contract_plan
    def drift(**kwargs):
        value = build(**kwargs)
        (root / "source.py").write_text("def changed(): return 2\n")
        return value
    monkeypatch.setattr(declared, "build_declared_contract_plan", drift)
    result = prep.plan(state, provider_callable=forbid_provider)
    assert result["qualified"] is False and result["provider_calls"] == 0
    assert not (state / "admission.json").exists()
