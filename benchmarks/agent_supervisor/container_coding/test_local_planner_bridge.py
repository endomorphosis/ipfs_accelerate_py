"""Real native directory scanner/parser joins the signed local-only policy."""

from copy import deepcopy
import json

import pytest

from benchmarks.agent_supervisor.container_coding.local_live_planner import (
    prepare,
    preflight_proposal,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import parse_prompt_goal_graph
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local


@pytest.fixture
def prepared(tmp_path):
    return prepare(tmp_path / "qualification")


def graph_for(prepared):
    return parse_prompt_goal_graph(
        json.dumps(preflight_proposal(prepared)),
        prepared["request"],
        prepared["scan"],
        config=prepared["config"],
        constraint_summaries=prepared["constraints"],
    )


def test_real_scanner_root_is_accepted_by_real_native_parser(prepared):
    graph = graph_for(prepared)
    assert graph.program_root == prepared["scan"].program_root
    assert graph.program_root.startswith("b")
    assert len(graph.goals) == 2 and len(graph.tasks) == 1
    assert graph.evidence == prepared["evidence"]
    admission = local.admit_local_benchmark_plan(graph=graph, manifest=prepared["manifest"])
    verified = local.verify_local_benchmark_admission(admission)
    assert verified["receipt"]["completion_authority"] is False
    assert verified["receipt"]["code_proof_authority"] is False
    assert all(
        row["required"] and row["phase"] == "post_execution"
        for row in verified["receipt"]["pending_requirements"]
    )
    assert all(
        row["external_ir_assurance"] == "unavailable"
        for row in verified["manifest"]["planning_inputs"]["domain_declarations"].values()
    )


@pytest.mark.parametrize("mutation", ["missing", "tampered", "foreign_root", "extra_obligation"])
def test_owner_signed_planner_inputs_cannot_launder_missing_or_foreign_authority(
    prepared, mutation
):
    forged = deepcopy(prepared["manifest"])
    inputs = forged["payload"]["planning_inputs"]
    if mutation == "missing":
        inputs["selected_evidence"].pop()
    elif mutation == "tampered":
        inputs["selected_evidence"][0]["summary"] = "invented source observation"
    elif mutation == "foreign_root":
        inputs["request"]["security_ir_root"] = local.content_identity({"foreign": "policy"})
    else:
        inputs["domain_declarations"]["legal"]["proof_obligations"] = ["must-not-be-deferred"]
    forged = local._signed(forged["payload"], forged["payload"])
    with pytest.raises(ValueError):
        local.admit_local_benchmark_plan(graph=graph_for(prepared), manifest=forged)


def test_graph_cannot_drop_bound_descriptive_evidence(prepared):
    from dataclasses import replace

    graph = graph_for(prepared)
    # Preserve the referenced evidence, remove a different independently bound
    # scan input. This remains a valid PromptGoalGraph but is not this manifest.
    used = set(graph.tasks[0].evidence_cids)
    removable = next(row for row in graph.evidence if row.evidence_cid not in used)
    graph = replace(graph, evidence=tuple(row for row in graph.evidence if row != removable))
    with pytest.raises(local.LocalPlanningError, match="externally evidenced"):
        local.admit_local_benchmark_plan(graph=graph, manifest=prepared["manifest"])


@pytest.mark.parametrize("failure", ["provider", "parse"])
def test_disabled_fallback_preserves_real_provider_failure_receipt(prepared, failure):
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import (
        PromptGoalPlannerError,
        generate_prompt_goal_graph,
    )

    calls = []

    def router(prompt):
        calls.append(prompt)
        if failure == "provider":
            raise TimeoutError("observed provider timeout")
        return "not a proposal"

    with pytest.raises(PromptGoalPlannerError, match="local fallback is disabled") as raised:
        generate_prompt_goal_graph(
            prepared["request"],
            prepared["scan"],
            router=router,
            config=prepared["config"],
            constraint_summaries=prepared["constraints"],
        )
    assert len(calls) == 1
    receipt = raised.value.provider_receipt
    assert receipt["attempted"] is True
    assert receipt["status"] != "succeeded"
    assert receipt["request_bytes"] == len(calls[0].encode())
    assert receipt["response_bytes"] == (len("not a proposal") if failure == "parse" else 0)


@pytest.mark.parametrize("failure", ["policy", "unavailable", "request_budget"])
def test_disabled_fallback_never_manufactures_graph_without_provider(prepared, failure):
    from dataclasses import replace
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import (
        PromptGoalPlannerError,
        generate_prompt_goal_graph,
    )

    request, scan, config = prepared["request"], prepared["scan"], prepared["config"]
    if failure == "policy":
        request = replace(
            request, planning_policy=replace(request.planning_policy, allow_model=False)
        )
        scan = replace(scan, request_cid=request.request_cid)
    if failure == "request_budget":
        config = replace(config, max_provider_request_bytes=512)
    calls = []
    with pytest.raises(PromptGoalPlannerError):
        generate_prompt_goal_graph(
            request,
            scan,
            config=config,
            capabilities={"available": False} if failure == "unavailable" else None,
            router=lambda prompt: calls.append(prompt) or "{}",
        )
    assert calls == []


def test_requested_deterministic_plan_accepts_actual_repository_root_scope(prepared):
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import (
        deterministic_prompt_goal_graph,
    )

    graph = deterministic_prompt_goal_graph(
        prepared["request"], prepared["scan"], config=prepared["config"]
    )
    assert "." not in graph.goals[0].scope_paths
    assert set(graph.goals[0].scope_paths) <= {"answer.py", "test_answer.py"}
    assert graph.tasks
