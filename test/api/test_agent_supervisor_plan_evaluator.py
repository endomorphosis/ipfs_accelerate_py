from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.objective_daemon import (
    persist_objective_plan_evaluations,
    plan_objective_records,
)
from ipfs_accelerate_py.agent_supervisor.plan_evaluator import (
    PlanBranch,
    evaluate_plan_branches,
)
from ipfs_accelerate_py.agent_supervisor.task_proposal_router import (
    build_structured_plan_prompt,
    generate_structured_plan_branches,
    parse_structured_plan_branches,
)


def _branch_payload(
    branch_id: str,
    *,
    expected_objective_delta: float = 0.7,
    estimated_cost: float = 2.0,
    risk: float = 0.2,
    source: str = "llm_router",
) -> dict[str, Any]:
    return {
        "branch_id": branch_id,
        "summary": f"Implement {branch_id} with focused production and test changes.",
        "predicted_files": [f"src/{branch_id}.py", f"tests/test_{branch_id}.py"],
        "predicted_symbols": [f"{branch_id}.build", f"test_{branch_id}"],
        "dependencies": ["REF-039"],
        "validation_commands": [f"python -m pytest tests/test_{branch_id}.py -q"],
        "validation_proof": [
            "The focused test executes the changed symbol and verifies its observable result."
        ],
        "estimated_cost": estimated_cost,
        "risk": risk,
        "expected_objective_delta": expected_objective_delta,
        "source": source,
    }


def _branch(branch_id: str, **overrides: Any) -> PlanBranch:
    return PlanBranch.from_dict(_branch_payload(branch_id, **overrides))


def test_plan_branch_schema_round_trips_all_scheduler_evidence() -> None:
    payload = _branch_payload("complete")

    branch = PlanBranch.from_dict(payload)

    assert branch.to_dict() == payload
    assert branch.predicted_files == tuple(payload["predicted_files"])
    assert branch.predicted_symbols == tuple(payload["predicted_symbols"])
    assert branch.dependencies == ("REF-039",)
    assert branch.validation_commands == tuple(payload["validation_commands"])
    assert branch.validation_proof == tuple(payload["validation_proof"])
    assert branch.estimated_cost == 2.0
    assert branch.risk == 0.2
    assert branch.expected_objective_delta == 0.7


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda item: item.pop("predicted_files"), "predicted_files"),
        (lambda item: item.update(predicted_symbols=[]), "predicted_symbols"),
        (lambda item: item.update(validation_commands=[]), "validation_commands"),
        (lambda item: item.update(validation_proof=[]), "validation_proof"),
        (lambda item: item.update(estimated_cost=-1), "estimated_cost"),
        (lambda item: item.update(risk=1.5), "risk"),
        (lambda item: item.update(expected_objective_delta=-0.1), "expected_objective_delta"),
    ],
)
def test_plan_branch_schema_rejects_missing_or_invalid_execution_evidence(
    mutation: Any,
    match: str,
) -> None:
    payload = _branch_payload("invalid")
    mutation(payload)

    with pytest.raises((TypeError, ValueError), match=match):
        PlanBranch.from_dict(payload)


def test_structured_prompt_requests_multiple_strict_schema_candidates() -> None:
    prompt = build_structured_plan_prompt(
        {
            "goal_id": "G9.S2",
            "task_id": "REF-041",
            "title": "Generate structured plan branches",
            "acceptance": "Ready subgoals have evaluated alternatives.",
            "outputs": ["plan_evaluator.py"],
        },
        branch_count=4,
    )

    assert "4" in prompt
    for field in _branch_payload("schema"):
        assert field in prompt
    assert "JSON" in prompt


def test_parser_accepts_fenced_router_json_and_validates_every_branch() -> None:
    raw = "Router result:\n```json\n" + json.dumps(
        {"branches": [_branch_payload("one"), _branch_payload("two")]}
    ) + "\n```\n"

    branches = parse_structured_plan_branches(raw)

    assert [branch.branch_id for branch in branches] == ["one", "two"]
    assert all(isinstance(branch, PlanBranch) for branch in branches)
    assert all(branch.source == "llm_router" for branch in branches)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda branch: branch.update(unexpected="not in schema"),
        lambda branch: branch.update(source="untrusted_provider_label"),
    ],
)
def test_parser_rejects_unknown_fields_and_untrusted_source(mutation: Any) -> None:
    branch = _branch_payload("strict")
    mutation(branch)

    with pytest.raises(ValueError):
        parse_structured_plan_branches(json.dumps({"branches": [branch]}))


def test_evaluator_is_deterministic_and_retains_rejected_rationales() -> None:
    best = _branch(
        "best",
        expected_objective_delta=0.95,
        estimated_cost=1.0,
        risk=0.05,
    )
    expensive = _branch(
        "expensive",
        expected_objective_delta=0.4,
        estimated_cost=9.0,
        risk=0.3,
    )
    risky = _branch(
        "risky",
        expected_objective_delta=0.8,
        estimated_cost=2.0,
        risk=0.95,
    )

    forward = evaluate_plan_branches([risky, best, expensive])
    reverse = evaluate_plan_branches([expensive, best, risky])

    assert forward.selected.branch_id == "best"
    assert reverse.selected.branch_id == "best"
    assert {item.branch_id for item in forward.rejected} == {"expensive", "risky"}
    assert forward.to_dict() == reverse.to_dict()
    assert set(forward.scores) == {"best", "expensive", "risky"}
    assert set(forward.rationales) == {"best", "expensive", "risky"}
    assert all(forward.rationales[branch_id] for branch_id in forward.rationales)


def test_evaluator_uses_stable_branch_id_tiebreak() -> None:
    alpha = _branch("alpha")
    zeta = _branch("zeta")

    first = evaluate_plan_branches([zeta, alpha])
    second = evaluate_plan_branches([alpha, zeta])

    assert first.selected.branch_id == second.selected.branch_id == "alpha"
    assert first.to_dict() == second.to_dict()


def test_evaluation_profile_g_payload_uses_content_safe_integer_scores() -> None:
    evaluation = evaluate_plan_branches([_branch("selected"), _branch("other")])

    payload = evaluation.to_dict(profile_g=True)
    selected = payload["selected"]

    assert selected["branch_id"] == evaluation.selected.branch_id
    assert payload["scores"][selected["branch_id"]] == evaluation.scores[selected["branch_id"]]
    assert "estimated_cost" not in selected
    assert "risk" not in selected
    assert "expected_objective_delta" not in selected
    assert isinstance(selected["estimated_cost_millionths"], int)
    assert isinstance(selected["risk_millionths"], int)
    assert isinstance(selected["expected_objective_delta_millionths"], int)
    assert all(isinstance(score, int) for score in evaluation.scores.values())


def test_router_generates_multiple_validated_branches_for_an_eligible_subgoal() -> None:
    prompts: list[str] = []

    def router(prompt: str) -> str:
        prompts.append(prompt)
        return json.dumps(
            {"branches": [_branch_payload("direct"), _branch_payload("layered")]}
        )

    result = generate_structured_plan_branches(
        {"task_id": "REF-041", "title": "Structured branch planning"},
        router=router,
        branch_count=2,
    )

    assert len(prompts) == 1
    assert "REF-041" in prompts[0]
    assert [branch.branch_id for branch in result.branches] == ["direct", "layered"]
    assert result.used_fallback is False
    assert not result.router_error
    assert result.raw_response


def test_router_failure_uses_deterministic_fallback_without_blocking_ready_work() -> None:
    fallback_calls: list[tuple[object, int]] = []

    def unavailable_router(_prompt: str) -> str:
        raise TimeoutError("provider capacity exhausted")

    def fallback_planner(subgoal: object, branch_count: int) -> list[PlanBranch]:
        fallback_calls.append((subgoal, branch_count))
        return [
            _branch("fallback-a", source="deterministic_fallback"),
            _branch("fallback-b", source="deterministic_fallback"),
        ]

    subgoal = {"task_id": "READY-001", "claimable": True}
    result = generate_structured_plan_branches(
        subgoal,
        router=unavailable_router,
        fallback_planner=fallback_planner,
        branch_count=2,
    )

    assert fallback_calls == [(subgoal, 2)]
    assert result.used_fallback is True
    assert "provider capacity exhausted" in result.router_error
    assert [branch.branch_id for branch in result.branches] == [
        "fallback-a",
        "fallback-b",
    ]
    assert all(branch.source == "deterministic_fallback" for branch in result.branches)


@pytest.mark.parametrize(
    "raw",
    ["not json", '{"branches": []}', '{"branches": [{"branch_id": "bad"}]}'],
)
def test_malformed_or_empty_router_output_falls_back(raw: str) -> None:
    result = generate_structured_plan_branches(
        {"task_id": "READY-002"},
        router=lambda _prompt: raw,
        fallback_planner=lambda _subgoal, _count: [
            _branch("safe", source="deterministic_fallback")
        ],
    )

    assert result.used_fallback is True
    assert result.router_error
    assert [branch.branch_id for branch in result.branches] == ["safe"]


def _objective_record(task_id: str) -> SimpleNamespace:
    finding = SimpleNamespace(
        goal_id="G9.S2",
        title=f"Plan {task_id}",
        summary=f"Create evaluated alternatives for {task_id}.",
        goal="Every ready subgoal has an evidence-backed selected branch.",
        priority="P1",
        track="G9",
        missing_evidence=["Only one constant-scored branch exists."],
        predicted_files=[f"src/{task_id.lower()}.py"],
        outputs=[f"tests/test_{task_id.lower()}.py"],
        ast_symbols=[f"build_{task_id.lower()}"],
        parent_goal_ids=["G9.S1"],
        validation=f"python -m pytest tests/test_{task_id.lower()}.py -q",
    )
    return SimpleNamespace(task_id=task_id, finding=finding)


def test_objective_planning_isolates_router_failure_and_keeps_later_work_ready() -> None:
    router_calls = 0

    def router(_prompt: str) -> str:
        nonlocal router_calls
        router_calls += 1
        if router_calls == 1:
            raise ConnectionError("first provider call failed")
        return json.dumps(
            {"branches": [_branch_payload("later-best"), _branch_payload("later-alt")]}
        )

    decisions = plan_objective_records(
        [_objective_record("READY-001"), _objective_record("READY-002")],
        branch_count=2,
        router=router,
    )

    assert [item["task_id"] for item in decisions] == ["READY-001", "READY-002"]
    assert decisions[0]["used_fallback"] is True
    assert "first provider call failed" in decisions[0]["router_error"]
    assert decisions[0]["selected"]["branch"]
    assert decisions[1]["used_fallback"] is False
    assert decisions[1]["router_error"] is None
    assert decisions[1]["selected"]["branch"]["branch_id"] in {"later-best", "later-alt"}
    assert len(decisions[1]["rejected"]) == 1
    assert decisions[1]["selection_rationale"]


def test_daemon_persists_selected_and_rejected_branches_for_scheduler_visibility(
    tmp_path: Path,
) -> None:
    decision = {
        "task_id": "READY-001",
        "goal_id": "G9.S2",
        "source": "llm_router",
        "used_fallback": False,
        "router_error": None,
        "evaluator_version": "plan-evaluator-v1",
        "selected": {
            "branch": _branch_payload("selected"),
            "score_millionths": 900_000,
            "rationale": ["highest deterministic utility"],
        },
        "rejected": [
            {
                "branch": _branch_payload("rejected"),
                "score_millionths": 400_000,
                "rationale": ["higher cost and lower objective delta"],
            }
        ],
        "selection_rationale": ["highest deterministic utility"],
    }
    artifact = tmp_path / "state" / "plan-evaluations.json"
    bundle_index = tmp_path / "bundles" / "index.json"
    bundle_index.parent.mkdir(parents=True)
    bundle_index.write_text(
        json.dumps(
            {
                "bundles": {
                    "g9/s2": {
                        "tasks": [
                            {"task_id": "READY-001"},
                            {"task_id": "UNRELATED"},
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    persist_objective_plan_evaluations(
        artifact,
        [decision],
        bundle_index_path=bundle_index,
    )

    persisted = json.loads(artifact.read_text(encoding="utf-8"))
    projected = json.loads(bundle_index.read_text(encoding="utf-8"))
    task = projected["bundles"]["g9/s2"]["tasks"][0]
    assert persisted["evaluation_count"] == 1
    assert persisted["evaluations"][0]["selected"]["branch"]["branch_id"] == "selected"
    assert persisted["evaluations"][0]["rejected"][0]["branch"]["branch_id"] == "rejected"
    assert task["selected_plan_branch"]["branch_id"] == "selected"
    assert task["selected_plan_evaluation"]["score_millionths"] == 900_000
    assert task["rejected_plan_branches"][0]["branch"]["branch_id"] == "rejected"
    assert task["plan_selection_rationale"] == ["highest deterministic utility"]
    assert "plan_evaluation" not in projected["bundles"]["g9/s2"]["tasks"][1]
