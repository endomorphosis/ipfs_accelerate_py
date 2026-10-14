"""Native immutable graph revisions; declaration never supplies admission."""
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.prompt.acceptance_bindings import (
    AcceptanceBindingError, bind_acceptance_checker,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import parse_prompt_goal_graph
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_plan_admission import admit_prompt_plan
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    EvidenceAuthority, PromptAcceptanceRecord, PromptEvidenceRecord, PromptGoalGraph,
    PromptValidationRecord, prompt_workflow_cid,
)
from test.api.test_agent_supervisor_prompt_goal_planner import _encoded_proposal, _request, _scan

KEY = "criterion:fenced-owner"


def fixture():
    request = _request()
    scan = _scan(request)
    graph = parse_prompt_goal_graph(_encoded_proposal(scan), request, scan)
    extra = PromptAcceptanceRecord(KEY, "Exact owner claims and whole-batch rollback must hold.")
    root = replace(graph.root_goal, acceptance=(*graph.root_goal.acceptance, extra))
    task = replace(graph.tasks[0], goal_cid=root.goal_cid,
                   acceptance=(*graph.tasks[0].acceptance, extra))
    child = replace(root, goal_key="goal:child", parent_goal_cid=root.goal_cid,
                    acceptance=graph.root_goal.acceptance)
    dependent = replace(graph.tasks[0], task_key="task:dependent", goal_cid=child.goal_cid,
                        dependency_task_cids=(task.task_cid,))
    graph = replace(graph, goals=(root, child), tasks=(task, dependent))
    command = PromptValidationRecord("validation:fenced-owner",
        ("python3", "scripts/ops/legal_ir/check_span_claim_acceptance.py", "--manifest", "manifest.json"))
    evidence = PromptEvidenceRecord("evidence:fenced-owner", "checked_component_observation",
        prompt_workflow_cid({"fixture": "engineering-test-source"}),
        "Private native control observations; no legal/model/kernel admission.",
        claim_keys=(KEY,), authority=EvidenceAuthority.SCAN_ADVISORY)
    return request, scan, graph, command, evidence


def bind(graph, command, evidence, **changes):
    arguments = dict(expected_plan_root=graph.plan_root_cid, criterion_key=KEY,
                     producer_task_key="task:retry-planner", validation=command, evidence=evidence)
    arguments.update(changes)
    return bind_acceptance_checker(graph, **arguments)


def test_complete_hierarchy_and_task_cids_are_rebound_without_other_acceptance_changes():
    _, _, graph, command, evidence = fixture()
    original = graph.to_dict()
    result = bind(graph, command, evidence)
    assert graph.to_dict() == original
    assert result.graph.plan_root_cid != graph.plan_root_cid
    assert set(result.goal_cid_map) == {item.goal_cid for item in graph.goals}
    assert set(result.task_cid_map) == {item.task_cid for item in graph.tasks}
    assert PromptGoalGraph.from_dict(result.graph.to_dict()).to_dict() == result.graph.to_dict()
    old_goals = {item.goal_key: item for item in graph.goals}
    old_tasks = {item.task_key: item for item in graph.tasks}
    for goal in result.graph.goals:
        before = old_goals[goal.goal_key]
        assert goal.parent_goal_cid == result.goal_cid_map.get(before.parent_goal_cid, "")
        for item in goal.acceptance:
            old = next(old for old in before.acceptance if old.criterion_key == item.criterion_key)
            if item.criterion_key == KEY:
                assert item.criterion == old.criterion
                assert item.validation_keys == (command.validation_key,)
                assert item.evidence_cids == (evidence.evidence_cid,)
                assert evidence.evidence_cid in goal.evidence_cids
            else:
                assert item == old
    for task in result.graph.tasks:
        before = old_tasks[task.task_key]
        assert task.goal_cid == result.goal_cid_map[before.goal_cid]
        assert task.dependency_task_cids == tuple(result.task_cid_map[cid] for cid in before.dependency_task_cids)
        assert set(before.validations).issubset(task.validations)
        assert (command in task.validations) is (task.task_key == "task:retry-planner")
        assert task.objective == before.objective and task.scope_paths == before.scope_paths
        assert task.outputs == before.outputs and task.policy_roots == before.policy_roots
        assert (evidence.evidence_cid in task.evidence_cids) is (task.task_key == "task:retry-planner")
    assert result.graph.unresolved_questions == graph.unresolved_questions
    assert result.graph.uncertainty_debt == graph.uncertainty_debt
    assert result.to_dict()["criterion_satisfied"] is False
    assert result.to_dict()["admitted"] is False
    with pytest.raises(TypeError):
        result.task_cid_map["new"] = "wrong"


@pytest.mark.parametrize("change", [
    {"expected_plan_root": prompt_workflow_cid({"stale": True})},
    {"producer_task_key": "missing"}, {"criterion_key": "missing"},
])
def test_stale_or_missing_identity_refuses_without_mutating_graph(change):
    _, _, graph, command, evidence = fixture()
    old = graph.to_dict()
    with pytest.raises(AcceptanceBindingError):
        bind(graph, command, evidence, **change)
    assert graph.to_dict() == old


def test_validation_and_evidence_keys_cannot_silently_replace_existing_records():
    _, _, graph, command, evidence = fixture()
    with pytest.raises(AcceptanceBindingError, match="validation key already"):
        bind(graph, replace(command, validation_key=graph.tasks[0].validations[0].validation_key), evidence)
    with pytest.raises(AcceptanceBindingError, match="evidence key already"):
        bind(graph, command, replace(evidence, evidence_key=graph.evidence[0].evidence_key))
    with pytest.raises(AcceptanceBindingError, match="claim key"):
        bind(graph, command, replace(evidence, claim_keys=()))


def test_ambiguous_producer_or_mismatched_mirrored_text_is_not_bound():
    _, _, graph, command, evidence = fixture()
    root = graph.root_goal
    wrong_root = replace(root, acceptance=tuple(replace(item, criterion="Different meaning.")
        if item.criterion_key == KEY else item for item in root.acceptance))
    goals = tuple(wrong_root if item == root else replace(item, parent_goal_cid=wrong_root.goal_cid)
                  for item in graph.goals)
    tasks = tuple(replace(item, goal_cid=wrong_root.goal_cid) if item.goal_cid == root.goal_cid
                  else replace(item, goal_cid=next(goal.goal_cid for goal in goals if goal.goal_key == "goal:child"))
                  for item in graph.tasks)
    # Task dependency changes also need their native CID crosswalk.
    producer = next(item for item in tasks if item.task_key == "task:retry-planner")
    tasks = tuple(replace(item, dependency_task_cids=(producer.task_cid,))
                  if item.task_key == "task:dependent" else item for item in tasks)
    conflict = replace(graph, goals=goals, tasks=tasks)
    with pytest.raises(AcceptanceBindingError, match="text conflicts"):
        bind(conflict, command, evidence)
    duplicate = next(item for item in graph.tasks if item.task_key == "task:dependent")
    extra = next(item for item in graph.tasks if item.task_key == "task:retry-planner")
    duplicate = replace(duplicate, acceptance=extra.acceptance)
    ambiguous = replace(graph, tasks=tuple(duplicate if item.task_key == duplicate.task_key else item for item in graph.tasks))
    with pytest.raises(AcceptanceBindingError, match="one exact producer"):
        bind(ambiguous, command, evidence)


def test_advisory_binding_cannot_upgrade_authority_or_satisfy_native_admission():
    request, scan, graph, command, evidence = fixture()
    with pytest.raises(AcceptanceBindingError, match="remain advisory"):
        bind(graph, command, replace(evidence, authority=EvidenceAuthority.AUTHORITATIVE))
    result = bind(graph, command, evidence)
    receipt = admit_prompt_plan(result.graph, repository_tree_id=scan.dirty_worktree_root,
        workflow_request=request, scan_receipt=scan)
    assert receipt.admitted is False
    assert receipt.ir_receipt is None
    assert any(item.code == "unknown_mandatory_state" for item in receipt.receipt.findings)
    assert not any(item.code == "evidence_untraced"
                   and evidence.evidence_cid in item.source_ids
                   and item.message in ("goal acceptance evidence is not bound to the goal",
                                        "task acceptance evidence is not bound to the task")
                   for item in receipt.receipt.findings)


def test_matching_criterion_outside_producer_ancestry_refuses():
    _, _, graph, command, evidence = fixture()
    child = next(goal for goal in graph.goals if goal.parent_goal_cid)
    selected = next(item for item in graph.root_goal.acceptance if item.criterion_key == KEY)
    changed = replace(child, acceptance=(*child.acceptance, selected))
    goals = tuple(changed if goal == child else goal for goal in graph.goals)
    tasks = tuple(replace(task, goal_cid=changed.goal_cid) if task.goal_cid == child.goal_cid
                  else task for task in graph.tasks)
    with pytest.raises(AcceptanceBindingError, match="outside the producer goal ancestry"):
        bind(replace(graph, goals=goals, tasks=tasks), command, evidence)


def test_both_parent_and_child_mirrors_bind_to_the_same_descendant_producer():
    _, _, graph, command, evidence = fixture()
    child = next(goal for goal in graph.goals if goal.parent_goal_cid)
    selected = next(item for item in graph.root_goal.acceptance if item.criterion_key == KEY)
    changed = replace(child, acceptance=(*child.acceptance, selected))
    producer = next(task for task in graph.tasks if task.task_key == "task:retry-planner")
    descendant = replace(producer, goal_cid=changed.goal_cid)
    dependent = next(task for task in graph.tasks if task != producer)
    dependent = replace(dependent, goal_cid=changed.goal_cid,
                        dependency_task_cids=(descendant.task_cid,))
    graph = replace(graph, goals=(graph.root_goal, changed), tasks=(descendant, dependent))
    result = bind(graph, command, evidence)
    assert len(result.graph.goals) == 2
    for goal in result.graph.goals:
        item = next(item for item in goal.acceptance if item.criterion_key == KEY)
        assert command.validation_key in item.validation_keys
        assert evidence.evidence_cid in item.evidence_cids
        assert evidence.evidence_cid in goal.evidence_cids


def test_existing_checks_and_evidence_survive_binding_and_goal_dependency_rewrite():
    _, _, graph, command, evidence = fixture()
    producer = next(task for task in graph.tasks if task.task_key == "task:retry-planner")
    prior_key = producer.validations[0].validation_key
    prior_evidence = graph.evidence[0].evidence_cid
    selected = next(item for item in producer.acceptance if item.criterion_key == KEY)
    selected = replace(selected, validation_keys=(prior_key,), evidence_cids=(prior_evidence,))
    def retain(items):
        return tuple(selected if item.criterion_key == KEY else item for item in items)
    root = replace(graph.root_goal, acceptance=retain(graph.root_goal.acceptance))
    child = next(goal for goal in graph.goals if goal.parent_goal_cid)
    child = replace(child, parent_goal_cid=root.goal_cid,
                    dependency_goal_cids=(root.goal_cid,))
    producer = replace(producer, goal_cid=root.goal_cid, acceptance=retain(producer.acceptance))
    dependent = next(task for task in graph.tasks if task.task_key != producer.task_key)
    dependent = replace(dependent, goal_cid=child.goal_cid,
                        dependency_task_cids=(producer.task_cid,))
    graph = replace(graph, goals=(root, child), tasks=(producer, dependent))
    result = bind(graph, command, evidence)
    for record in (*result.graph.goals, *result.graph.tasks):
        for item in record.acceptance:
            if item.criterion_key == KEY:
                assert set(item.validation_keys) == {prior_key, command.validation_key}
                assert set(item.evidence_cids) == {prior_evidence, evidence.evidence_cid}
    changed_child = next(goal for goal in result.graph.goals if goal.parent_goal_cid)
    assert changed_child.dependency_goal_cids == (result.graph.root_goal.goal_cid,)
