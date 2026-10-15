"""Authored router responses through native parsing, admission and storage.

This covers the model path that the provider-free declared-contract owner
fixture deliberately bypasses. No live provider or benchmark oracle is used.
"""
from copy import deepcopy
import json
import subprocess

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.local_live_planner import preflight_proposal
from benchmarks.agent_supervisor.container_coding.test_terminal_public_acceptance_profile import prepared_public_case
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import _select_evidence
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import DirectoryScanReceipt, PromptWorkflowRequest
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local


def _authored_proposal(case):
    request = PromptWorkflowRequest.from_dict(case.prepared["request"])
    scan = DirectoryScanReceipt.from_dict(case.prepared["scan"])
    evidence = _select_evidence(request, scan, prep._config(case.repository))
    proposal = preflight_proposal({"spec": case.prepared["spec"], "evidence": evidence})
    proposal["root_goal_key"] = "TB-GOAL"
    proposal["goals"][0].update(goal_key="TB-GOAL", title="Public task", objective="Fulfill public task")
    proposal["goals"][1].update(goal_key="TB-SUBGOAL", parent_goal_key="TB-GOAL",
        title="Implement", objective="Fulfill public task")
    proposal["tasks"][0].update(task_key="TB-CODE-TASK", goal_key="TB-SUBGOAL",
        objective="Fulfill public task", predicted_files=["eigen.py"])
    # Break the fixture's shared lists before deliberately altering one node.
    return json.loads(json.dumps(proposal))


@pytest.mark.parametrize("mutation", [None, "task-omission", "task-alteration", "goal-alteration"])
def test_single_task_three_checks_use_model_parser_and_native_admission(tmp_path, monkeypatch, mutation):
    case = prepared_public_case(tmp_path)
    assert case.prepared["request"]["budget"]["max_tasks"] == 1
    assert case.prepared["planning_strategy"] == "direct"
    assert len(case.prepared["spec"]["acceptance"]) == 3
    proposal = _authored_proposal(case)
    if mutation == "task-omission":
        proposal["tasks"][0]["acceptance"].pop()
    elif mutation == "task-alteration":
        proposal["tasks"][0]["acceptance"][2]["criterion"] = "An altered public performance criterion"
    elif mutation == "goal-alteration":
        proposal["goals"][0]["acceptance"][2]["criterion"] = "An altered public performance criterion"
    native_run = subprocess.run

    def version(argv, *args, **kwargs):
        if argv == ["codex", "--version"]:
            return subprocess.CompletedProcess(argv, 0, "codex-cli " + prep.CLI_VERSION + "\n", "")
        return native_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", version)
    calls = []

    def authored_router_response(prompt, **kwargs):
        calls.append(kwargs)
        return {"text": json.dumps(proposal), "observation": {}, "execution_receipt": None}

    result = prep.plan(case.state, provider_callable=authored_router_response)
    assert len(calls) == result["provider_calls"] == 1  # Authored callable, no generation.
    assert result["provider_receipt"]["outcome"] == "provider"
    assert result["provider_receipt"]["fallback"]["used"] is False
    if mutation is not None:
        assert result["qualified"] is False
        assert result["failure"]["type"] == "LocalPlanningError"
        assert not (case.state / "admission.json").exists()
        assert not (case.state / "intent.duckdb").exists()
        if mutation.startswith("task-"):
            assert "task_contract_mismatch" in result
        return

    assert result["qualified"] is True, result
    assert result["goals"] == 2 and result["tasks"] == 1
    assert len(result["task_cids"]) == 1
    admission = json.loads((case.state / "admission.json").read_text())
    verified = local.verify_local_benchmark_admission(admission)
    expected = {row["criterion_key"]: deepcopy(row) for row in case.prepared["spec"]["acceptance"]}
    for node in (*verified["graph"].goals, *verified["graph"].tasks):
        assert {criterion.criterion_key: {key: value for key, value in criterion.to_dict().items()
            if key in expected[criterion.criterion_key]} for criterion in node.acceptance} == expected
    assert verified["receipt"]["planning_permitted"] is True
    assert verified["receipt"]["completion_authority"] is False
    assert {row["phase"] for row in verified["receipt"]["pending_requirements"]} == {"post_execution"}
    assert (case.state / "intent.duckdb").is_file()
