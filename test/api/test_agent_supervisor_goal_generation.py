from __future__ import annotations

import json

import pytest

import ipfs_accelerate_py.agent_supervisor.objective_tracker as objective_tracker_module
from ipfs_accelerate_py.agent_supervisor.goal_coverage import (
    UNMAPPED_GOAL_ID,
    goal_coverage_work_seeds,
)
from ipfs_accelerate_py.agent_supervisor.objective_graph import (
    ObjectiveGenerationLimits,
    ObjectiveGoalMaterializationPolicy,
    ObjectiveWorkKind,
    ObjectiveWorkProposal,
    materialize_bounded_objective_work,
    objective_goal_content_id,
    preview_objective_goal_materialization,
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.objective_daemon import (
    load_objective_admission_records,
    load_objective_generation_work,
    materialize_admitted_objective_work,
    materialize_objective_generation_cycle,
)
from ipfs_accelerate_py.agent_supervisor.objective_tracker import (
    ObjectiveMaterializationTransactionState,
    commit_objective_goal_materialization,
    objective_materialization_tree_identity,
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


def _objective_heap() -> str:
    return """# Objective Heap

## ROOT Frozen objective

- Status: active
- Goal: Deliver the root objective
- Evidence: root proof
- Graph depth: 0
"""


def _hierarchical_goal_work() -> tuple[ObjectiveWorkProposal, ObjectiveWorkProposal]:
    goal = _proposal(
        kind="goal",
        title="Establish API evidence goal",
        parent_goal_id="ROOT",
        parent_objective_terms=("Deliver the root objective",),
        expected_evidence_delta=("API evidence",),
        dependencies=("bootstrap",),
        predicted_files=("src/api.py",),
        predicted_symbols=("ApiClient",),
        validation_commands=("pytest tests/test_api.py -q",),
        depth=1,
        source_id="proposal:api-goal",
    )
    subgoal = _proposal(
        kind="subgoal",
        title="Verify API evidence",
        parent_goal_id=goal.canonical_id,
        parent_objective_terms=("API evidence",),
        expected_evidence_delta=("API validation receipt",),
        dependencies=(goal.canonical_id,),
        predicted_files=("tests/test_api.py",),
        predicted_symbols=("test_api",),
        validation_commands=("pytest tests/test_api.py -q",),
        depth=2,
        source_id="proposal:api-subgoal",
    )
    return goal, subgoal


def test_goal_materialization_is_preview_first_lossless_and_transactional(
    tmp_path,
) -> None:
    objective_path = tmp_path / "repo" / "objective-heap.md"
    objective_path.parent.mkdir()
    objective_path.write_text(_objective_heap(), encoding="utf-8")
    original = objective_path.read_bytes()
    goal, subgoal = _hierarchical_goal_work()

    preview = preview_objective_goal_materialization(
        objective_path.read_text(encoding="utf-8"),
        (subgoal, goal),  # provider order cannot change hierarchy
        policy=ObjectiveGoalMaterializationPolicy(root_goal_id="ROOT"),
    )

    assert preview.ready
    assert objective_path.read_bytes() == original
    assert preview.admitted_proposal_ids == (goal.canonical_id, subgoal.canonical_id)
    result = commit_objective_goal_materialization(
        repo_root=objective_path.parent,
        objective_path=objective_path,
        journal_path=tmp_path / "admission-journal.json",
        preview=preview,
    )

    assert result.state is ObjectiveMaterializationTransactionState.COMMITTED
    assert result.changed
    materialized = {
        item.goal_id: item
        for item in parse_goal_heap(objective_path.read_text(encoding="utf-8"))
    }
    assert materialized[goal.canonical_id].parent_goal_ids == ["ROOT"]
    child = materialized[subgoal.canonical_id]
    assert child.parent_goal_ids == [goal.canonical_id]
    assert child.dependencies == [goal.canonical_id]
    assert child.required_evidence == ["API validation receipt"]
    assert child.semantic_key == subgoal.semantic_key
    assert child.canonical_proposal_id == subgoal.canonical_id
    assert child.lifecycle_owner == "objective_daemon"
    assert child.status == "active"

    replay = commit_objective_goal_materialization(
        repo_root=objective_path.parent,
        objective_path=objective_path,
        journal_path=tmp_path / "admission-journal.json",
        preview=preview,
    )
    assert replay.committed and replay.resumed and not replay.changed
    assert objective_path.read_text(encoding="utf-8").count(
        f"## {subgoal.canonical_id} "
    ) == 1


def test_goal_materialization_keeps_semantic_and_structural_breadth_bounds() -> None:
    goal, _subgoal = _hierarchical_goal_work()
    first = preview_objective_goal_materialization(
        _objective_heap(),
        (goal,),
        root_goal_id="ROOT",
    )
    assert first.ready

    semantic_replay = _proposal(
        kind="goal",
        title="Reworded API evidence goal",
        parent_goal_id=goal.parent_goal_id,
        parent_objective_terms=goal.parent_objective_terms,
        expected_evidence_delta=goal.expected_evidence_delta,
        dependencies=goal.dependencies,
        predicted_files=goal.predicted_files,
        predicted_symbols=goal.predicted_symbols,
        validation_commands=goal.validation_commands,
        depth=goal.depth,
        source_id="proposal:different-provider-identity",
    )
    assert semantic_replay.canonical_id != goal.canonical_id
    assert semantic_replay.semantic_key == goal.semantic_key
    duplicate = preview_objective_goal_materialization(
        first.candidate_text,
        (semantic_replay,),
        root_goal_id="ROOT",
    )
    assert not duplicate.ready
    assert duplicate.candidate_text == first.candidate_text
    assert [item.reason for item in duplicate.rejected] == ["semantic_duplicate"]

    terminal_child_heap = (
        _objective_heap()
        + "\n## TERMINAL Historical child\n\n"
        "- Status: verified\n"
        "- Goal: Historical bounded branch\n"
        "- Parents: ROOT\n"
        "- Graph depth: 1\n"
    )
    breadth = preview_objective_goal_materialization(
        terminal_child_heap,
        (goal,),
        policy=ObjectiveGoalMaterializationPolicy(
            root_goal_id="ROOT",
            limits=ObjectiveGenerationLimits(max_breadth_per_parent=1),
        ),
    )
    assert not breadth.ready
    assert breadth.candidate_text == terminal_child_heap
    assert [item.reason for item in breadth.rejected] == ["breadth_limit"]


def test_stale_heap_and_lease_conflict_fail_closed_and_remain_resumable(
    tmp_path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    objective_path = repo_root / "objective-heap.md"
    objective_path.write_text(_objective_heap(), encoding="utf-8")
    goal, _subgoal = _hierarchical_goal_work()
    preview = preview_objective_goal_materialization(
        objective_path.read_text(encoding="utf-8"),
        (goal,),
        root_goal_id="ROOT",
    )

    objective_path.write_text(
        _objective_heap() + "\n## HUMAN Concurrent goal\n\n- Status: active\n- Parent: ROOT\n",
        encoding="utf-8",
    )
    concurrent = objective_path.read_bytes()
    stale = commit_objective_goal_materialization(
        repo_root=repo_root,
        objective_path=objective_path,
        journal_path=tmp_path / "stale-journal.json",
        preview=preview,
    )
    assert stale.state is ObjectiveMaterializationTransactionState.BLOCKED
    assert stale.reason_codes == ("stale_objective_heap",)
    assert objective_path.read_bytes() == concurrent

    objective_path.write_text(_objective_heap(), encoding="utf-8")
    tree_journal = tmp_path / "tree-journal.json"
    tree_preview = preview_objective_goal_materialization(
        _objective_heap(),
        (goal,),
        root_goal_id="ROOT",
    )
    expected_tree = objective_materialization_tree_identity(
        repo_root,
        objective_path=objective_path,
        journal_path=tree_journal,
    ).tree_id
    (repo_root / "concurrent-source.py").write_text(
        "CONCURRENT = True\n", encoding="utf-8"
    )
    stale_tree = commit_objective_goal_materialization(
        repo_root=repo_root,
        objective_path=objective_path,
        journal_path=tree_journal,
        preview=tree_preview,
        expected_repository_tree_id=expected_tree,
    )
    assert stale_tree.state is ObjectiveMaterializationTransactionState.BLOCKED
    assert stale_tree.reason_codes == ("stale_repository_tree",)
    assert objective_path.read_text(encoding="utf-8") == _objective_heap()

    fresh = preview_objective_goal_materialization(
        objective_path.read_text(encoding="utf-8"),
        (goal,),
        root_goal_id="ROOT",
    )
    journal = tmp_path / "lease-journal.json"
    blocked = commit_objective_goal_materialization(
        repo_root=repo_root,
        objective_path=objective_path,
        journal_path=journal,
        preview=fresh,
        lease_guard=lambda token: False,
        expected_lease_token="fence-7",
    )
    assert blocked.state is ObjectiveMaterializationTransactionState.PREPARED
    assert blocked.resumable
    assert objective_path.read_text(encoding="utf-8") == _objective_heap()

    resumed = commit_objective_goal_materialization(
        repo_root=repo_root,
        objective_path=objective_path,
        journal_path=journal,
        preview=fresh,
        lease_guard=lambda token: {"fencing_token": token},
        expected_lease_token="fence-7",
    )
    assert resumed.committed and resumed.resumed
    assert goal.canonical_id in {
        item.goal_id
        for item in parse_goal_heap(objective_path.read_text(encoding="utf-8"))
    }


def test_partial_heap_write_remains_prepared_and_resumes_exactly(
    tmp_path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    objective_path = repo_root / "objective-heap.md"
    objective_path.write_text(_objective_heap(), encoding="utf-8")
    goal, subgoal = _hierarchical_goal_work()
    preview = preview_objective_goal_materialization(
        _objective_heap(),
        (goal, subgoal),
        root_goal_id="ROOT",
    )
    journal = tmp_path / "partial-journal.json"
    partial_text = (
        _objective_heap().rstrip()
        + "\n\n"
        + preview.materialized[0].rendered_block.strip()
        + "\n"
    )
    real_rewrite = objective_tracker_module._atomic_rewrite
    monkeypatch.setattr(
        objective_tracker_module,
        "_atomic_rewrite",
        lambda path, _text: path.write_text(partial_text, encoding="utf-8"),
    )

    partial = commit_objective_goal_materialization(
        repo_root=repo_root,
        objective_path=objective_path,
        journal_path=journal,
        preview=preview,
    )
    assert partial.state is ObjectiveMaterializationTransactionState.PREPARED
    assert partial.resumable
    assert partial.reason_codes[0] == "partial_write"
    assert goal.canonical_id in objective_path.read_text(encoding="utf-8")
    assert subgoal.canonical_id not in objective_path.read_text(encoding="utf-8")

    monkeypatch.setattr(objective_tracker_module, "_atomic_rewrite", real_rewrite)
    resumed = commit_objective_goal_materialization(
        repo_root=repo_root,
        objective_path=objective_path,
        journal_path=journal,
        preview=preview,
    )
    assert resumed.committed and resumed.resumed
    materialized_ids = [
        item.goal_id
        for item in parse_goal_heap(objective_path.read_text(encoding="utf-8"))
    ]
    assert materialized_ids.count(goal.canonical_id) == 1
    assert materialized_ids.count(subgoal.canonical_id) == 1


def test_shadow_never_mutates_and_assist_persists_review_only(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    objective_path = repo_root / "objective-heap.md"
    objective_path.write_text(_objective_heap(), encoding="utf-8")
    generation_path = tmp_path / "state" / "objective-generation.json"
    goal, subgoal = _hierarchical_goal_work()
    original = objective_path.read_bytes()

    shadow = materialize_admitted_objective_work(
        (goal, subgoal),
        repo_root=repo_root,
        objective_path=objective_path,
        generation_path=generation_path,
        mode="shadow",
        root_goal_id="ROOT",
    )
    assert shadow.status == "shadow"
    assert objective_path.read_bytes() == original
    assert not generation_path.exists()

    assist = materialize_admitted_objective_work(
        (goal, subgoal),
        repo_root=repo_root,
        objective_path=objective_path,
        generation_path=generation_path,
        mode="assist",
        root_goal_id="ROOT",
    )
    assert assist.status == "review_required"
    assert assist.review_persisted and assist.resumable
    assert objective_path.read_bytes() == original
    records = load_objective_admission_records(generation_path)
    assert set(records) == {goal.canonical_id, subgoal.canonical_id}
    assert {item["status"] for item in records.values()} == {"review_required"}
    assert all(item["preview"] for item in records.values())


def test_auto_safe_without_bound_authority_fails_closed_and_is_reviewable(
    tmp_path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    objective_path = repo_root / "objective-heap.md"
    objective_path.write_text(_objective_heap(), encoding="utf-8")
    generation_path = tmp_path / "state" / "objective-generation.json"
    goal, _subgoal = _hierarchical_goal_work()
    root = parse_goal_heap(_objective_heap())[0]

    result = materialize_admitted_objective_work(
        (goal,),
        repo_root=repo_root,
        objective_path=objective_path,
        generation_path=generation_path,
        mode="auto_safe",
        root_goal_id="ROOT",
        expected_root_content_id=objective_goal_content_id(root),
        new_assumption_ids=("assumption:hidden",),
        unsupported_semantics=("formula:invented",),
        hard_policy_gates={"scope": False},
    )

    assert result.status == "rejected"
    assert {
        "new_assumptions",
        "unsupported_semantics",
        "hard_policy_gate:scope",
        "unresolved_authoritative_receipts",
    }.issubset(result.reason_codes)
    assert objective_path.read_text(encoding="utf-8") == _objective_heap()
    record = load_objective_admission_records(generation_path)[goal.canonical_id]
    assert record["status"] == "rejected"
    assert record["lifecycle_owner"] == "objective_daemon"
