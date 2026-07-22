from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.goal_coverage import (
    UNMAPPED_GOAL_ID,
    goal_coverage_work_seeds,
)
from ipfs_accelerate_py.agent_supervisor.objective_graph import (
    ObjectiveGenerationLimits,
    ObjectiveWorkKind,
    ObjectiveWorkProposal,
    materialize_bounded_objective_work,
)
from ipfs_accelerate_py.agent_supervisor.objective_daemon import (
    load_objective_generation_work,
    materialize_objective_generation_cycle,
)
from ipfs_accelerate_py.agent_supervisor.plan_evaluator import (
    AnalysisProposal,
    ObjectiveWorkEvaluationPolicy,
)
from ipfs_accelerate_py.agent_supervisor.task_proposal_router import (
    analysis_proposals_to_objective_work,
)


def _proposal(**overrides: object) -> ObjectiveWorkProposal:
    values: dict[str, object] = {
        "kind": "task",
        "title": "Add API proof",
        "parent_goal_id": "G1",
        "parent_objective_terms": ("API is available",),
        "expected_evidence_delta": ("API validation receipt",),
        "dependencies": ("bootstrap",),
        "predicted_files": ("src/api.py",),
        "predicted_symbols": ("ApiClient",),
        "validation_commands": ("pytest -q",),
        "confidence": 0.8,
        "estimated_cost": 2.0,
        "novelty": 0.9,
        "depth": 1,
        "estimated_tokens": 100,
    }
    values.update(overrides)
    return ObjectiveWorkProposal(**values)  # type: ignore[arg-type]


def test_coverage_and_contradictions_create_complete_hierarchical_work() -> None:
    coverage = {
        "criteria": [
            {
                "goal_id": "G1",
                "criterion_id": "criterion-1",
                "criterion": "The API has current validation proof",
                "status": "uncovered",
                "missing_surfaces": ["predicted_files", "validation_receipts"],
            }
        ],
        "edges": [
            {
                "criterion_id": "criterion-1",
                "surface": "predicted_file",
                "value": "src/api.py",
            },
            {
                "criterion_id": "criterion-1",
                "surface": "ast_symbol",
                "value": "ApiClient",
            },
            {
                "criterion_id": "criterion-1",
                "surface": "validation_command",
                "value": "pytest tests/test_api.py -q",
            },
        ],
        "finding_assignments": [
            {
                "finding_id": "finding-1",
                "goal_id": UNMAPPED_GOAL_ID,
                "confidence": 0.7,
                "finding": {
                    "title": "Cover the unsupported event surface",
                    "missing_evidence": ["event delivery receipt"],
                    "predicted_files": ["src/events.py"],
                    "predicted_symbols": ["EventSink"],
                    "validation": ["pytest tests/test_events.py -q"],
                },
            }
        ],
    }
    goals = [
        {
            "goal_id": "G1",
            "fields": {
                "acceptance": "The API has current validation proof",
                "outputs": "src/api.py",
                "validation": "pytest tests/test_api.py -q",
                "graph_depth": "0",
            },
        }
    ]
    contradictions = [
        {
            "goal_id": "G1",
            "kind": "failed_validation",
            "summary": "The API validation regressed.",
            "impacted_criteria": ["The API has current validation proof"],
            "invalidated_evidence": ["receipt-old"],
            "source_receipt_id": "receipt-failed",
        }
    ]

    first = goal_coverage_work_seeds(
        coverage,
        goals=goals,
        contradictions=contradictions,
    )
    second = goal_coverage_work_seeds(
        coverage,
        goals=goals,
        contradictions=contradictions,
    )

    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert {item.kind for item in first} == {
        ObjectiveWorkKind.GOAL,
        ObjectiveWorkKind.SUBGOAL,
        ObjectiveWorkKind.TASK,
    }
    surface_tasks = [item for item in first if item.parent_goal_id.startswith("objective-work:")]
    assert surface_tasks
    assert all(item.dependencies == (item.parent_goal_id,) for item in surface_tasks)
    for item in first:
        payload = item.to_dict()
        assert payload["canonical_id"].startswith("objective-work:")
        assert payload["semantic_key"].startswith("objective-work/v1/")
        assert "parent_objective_terms" in payload
        assert payload["expected_evidence_delta"]
        assert "dependencies" in payload
        assert "predicted_files" in payload
        assert "predicted_symbols" in payload
        assert "validation" in payload
        assert "confidence" in payload
        assert "cost" in payload
        assert "novelty" in payload


def test_canonical_and_semantic_duplicates_are_suppressed_across_cycles() -> None:
    original = _proposal()
    exact = ObjectiveWorkProposal.from_dict(original.to_dict())
    semantic = _proposal(title="Prove the API with equivalent evidence")

    assert semantic.semantic_key == original.semantic_key
    assert semantic.canonical_id != original.canonical_id

    result = materialize_bounded_objective_work(
        [exact, semantic],
        existing_work=[original],
        limits=ObjectiveGenerationLimits(semantic_similarity_threshold=0.5),
    )

    assert not result.accepted
    assert {item.reason for item in result.rejected} == {
        "canonical_duplicate",
        "semantic_duplicate",
    }


@pytest.mark.parametrize(
    ("proposal_overrides", "limit_overrides", "current_open", "reason"),
    [
        ({"depth": 3}, {"max_depth": 2}, 0, "depth_limit"),
        ({"retry_count": 2}, {"max_retries": 1}, 0, "retry_limit"),
        ({"estimated_tokens": 101}, {"token_budget": 100}, 0, "token_budget"),
        ({}, {"max_open_work": 1}, 1, "open_work_limit"),
        ({}, {"max_breadth_per_parent": 0}, 0, "breadth_limit"),
        ({}, {"max_new_work": 0}, 0, "cycle_limit"),
    ],
)
def test_each_finite_generation_limit_is_fail_closed(
    proposal_overrides: dict[str, object],
    limit_overrides: dict[str, object],
    current_open: int,
    reason: str,
) -> None:
    result = materialize_bounded_objective_work(
        [_proposal(**proposal_overrides)],
        current_open_work=current_open,
        limits=ObjectiveGenerationLimits(**limit_overrides),  # type: ignore[arg-type]
    )
    assert not result.accepted
    assert [item.reason for item in result.rejected] == [reason]
    assert result.exhausted


def test_persisted_work_identity_is_revalidated() -> None:
    payload = _proposal().to_dict()
    payload["semantic_key"] = "objective-work/v1/tampered"
    with pytest.raises(ValueError, match="semantic_key"):
        ObjectiveWorkProposal.from_dict(payload)


def test_llm_router_proposals_preserve_complete_work_metadata() -> None:
    proposal = AnalysisProposal.from_dict(
        {
            "branch": {
                "branch_id": "router-plan",
                "summary": "Implement the uncovered API evidence.",
                "predicted_files": ["src/api.py"],
                "predicted_symbols": ["ApiClient"],
                "dependencies": ["REF-205"],
                "validation_commands": ["pytest tests/test_api.py -q"],
                "validation_proof": ["API receipt is current and provenance-backed"],
                "estimated_cost": 2.5,
                "risk": 0.1,
                "expected_objective_delta": 0.8,
                "source": "llm_router",
            },
            "confidence": 0.9,
            "novelty": 0.85,
            "objective_terms": ["API evidence"],
        }
    )

    (work,) = analysis_proposals_to_objective_work(
        [proposal],
        parent_goal_id="G1",
        depth=2,
        estimated_tokens=512,
        retry_count=1,
    )

    assert work.source == "llm_router"
    assert work.parent_objective_terms == ("API evidence",)
    assert work.expected_evidence_delta == (
        "API receipt is current and provenance-backed",
    )
    assert work.dependencies == ("REF-205",)
    assert work.predicted_files == ("src/api.py",)
    assert work.predicted_symbols == ("ApiClient",)
    assert work.validation_commands == ("pytest tests/test_api.py -q",)
    assert work.confidence == 0.9
    assert work.estimated_cost == 2.5
    assert work.novelty == 0.85
    assert work.estimated_tokens == 512
    assert work.retry_count == 1


def test_daemon_generation_ledger_prevents_cross_cycle_regeneration(tmp_path) -> None:
    artifact = tmp_path / "state" / "objective_generation.json"
    policy = ObjectiveWorkEvaluationPolicy(
        min_confidence=0.0,
        min_novelty=0.0,
        max_proposals=3,
        max_total_cost=10.0,
        max_open_work=10,
        current_open_work=0,
        remaining_token_budget=1000,
    )

    first, first_artifact = materialize_objective_generation_cycle(
        [_proposal()],
        artifact_path=artifact,
        evaluation_policy=policy,
        objective_terms=["API is available"],
    )
    second, second_artifact = materialize_objective_generation_cycle(
        [_proposal(title="Cosmetically renamed API proof")],
        artifact_path=artifact,
        evaluation_policy=policy,
        objective_terms=["API is available"],
    )

    assert len(first.accepted) == 1
    assert not second.accepted
    assert first_artifact["cycle_count"] == 1
    assert second_artifact["cycle_count"] == 2
    assert second_artifact["generated_work_count"] == 1
    assert second_artifact["last_evaluation"]["rejected"][0]["reason"] in {
        "duplicate_canonical_identity",
        "duplicate_semantic_work",
    }
    assert load_objective_generation_work(artifact) == tuple(
        second_artifact["generated_work"]
    )
    assert json.loads(artifact.read_text(encoding="utf-8"))["cycle_count"] == 2
