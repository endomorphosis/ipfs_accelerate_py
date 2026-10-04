"""Signed absent-to-created outputs preserve the original local admission gate."""

from copy import deepcopy
from dataclasses import replace

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario as scenario
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import PromptOutputRecord
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentCompletionError


def _with_creation(scenario, name="report.jsonl"):
    original = scenario["manifest"]["payload"]
    specs = deepcopy(original["tasks"])
    if name not in specs[0]["scope_paths"]:
        specs[0]["scope_paths"].append(name)
    specs[0]["outputs"] = [row for row in specs[0]["outputs"] if row["path"] != name]
    specs[0]["outputs"].append({"path": name, "effect": "create", "media_type": "application/json"})
    graph = scenario["graph"]
    goal = replace(graph.goals[0], scope_paths=tuple(specs[0]["scope_paths"]))
    task = replace(graph.tasks[0], goal_cid=goal.goal_cid, scope_paths=tuple(specs[0]["scope_paths"]),
                   outputs=tuple(PromptOutputRecord(**row) for row in specs[0]["outputs"]))
    graph = replace(graph, goals=(goal,), tasks=(task,))
    manifest = local.author_local_benchmark_manifest(
        repository=scenario["repository"], profile_dir=original["profile_dir"],
        lifecycle_dir=original["lifecycle_dir"], task_specs=specs,
        planning_roots=original["planning_roots"],
    )
    return graph, manifest


def test_exact_declared_create_survives_real_native_validation_and_completion(scenario):
    graph, manifest = _with_creation(scenario)
    root, intent = scenario["repository"], scenario["intent"]
    assert manifest["payload"]["schema"] == local.CREATE_MANIFEST_SCHEMA
    assert manifest["payload"]["created_outputs"] == ["report.jsonl"]
    assert "report.jsonl" not in manifest["payload"]["sources"]
    assert not (root / "report.jsonl").exists()
    admission = local.admit_local_benchmark_plan(graph=graph, manifest=manifest)
    result = local.materialize_local_benchmark_plan(admission=admission, intent=intent)
    cid = result["task_cids"][0]
    intent.cas_task_status(task_cid=cid, expected_revision=intent.get_task(cid)["revision"], new_status="in_progress")
    (root / "answer.py").write_text("def answer():\n    return 2\n")
    # Even a legitimately passing public check cannot waive an absent declared
    # output. The output requirement is separate from check authoring quality.
    passed = local.run_local_task_validations(intent=intent, task_cid=cid, attempt_id="before-create")
    assert passed["passed"] is True
    with pytest.raises(IntentCompletionError):
        intent.cas_task_status(task_cid=cid, expected_revision=intent.get_task(cid)["revision"],
                               new_status="completed", allow_completion_without_evidence=True)
    (root / "report.jsonl").write_text('{"observed":"answer now returns two"}\n')
    current = local.observe_local_manifest_sources(root, manifest["payload"])
    assert "report.jsonl" in current
    finished = local.run_local_task_validations(intent=intent, task_cid=cid, attempt_id="after-create")
    assert finished["passed"] is True
    assert finished["source_tree_id"] != passed["source_tree_id"]
    intent.cas_task_status(task_cid=cid, expected_revision=intent.get_task(cid)["revision"], new_status="completed")
    assert intent.get_task(cid)["status"] == "completed"


@pytest.mark.parametrize("preexisting", ["untracked", "baseline"])
def test_created_output_must_actually_be_absent_before_admission(scenario, preexisting):
    if preexisting == "untracked":
        (scenario["repository"] / "report.jsonl").write_text("not absent\n")
        name = "report.jsonl"
    else:
        name = "answer.py"
    with pytest.raises(local.LocalPlanningError):
        _with_creation(scenario, name=name)


@pytest.mark.parametrize("mutation", ["symlink", "undeclared", "public_check", "directory"])
def test_created_scope_cannot_hide_foreign_or_immutable_changes(scenario, mutation):
    _graph, manifest = _with_creation(scenario)
    root = scenario["repository"]
    if mutation == "symlink":
        (root / "report.jsonl").symlink_to(root / "answer.py")
    elif mutation == "undeclared":
        (root / "foreign.py").write_text("print('outside signed scope')\n")
    elif mutation == "public_check":
        (root / "test_answer.py").write_text("pass\n")
    else:
        (root / "report.jsonl").mkdir()
    with pytest.raises(local.LocalPlanningError):
        local._manifest(manifest)


def test_owner_signature_does_not_authorize_extra_creation(scenario):
    _graph, manifest = _with_creation(scenario)
    altered = deepcopy(manifest["payload"])
    altered["created_outputs"].append("foreign.py")
    resigned = local._signed(altered, altered)
    with pytest.raises(local.LocalPlanningError, match="declared-create"):
        local._manifest(resigned)
