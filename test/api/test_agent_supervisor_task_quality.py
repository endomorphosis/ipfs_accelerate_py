from __future__ import annotations

import copy

import pytest

from ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery import (
    ensure_task_blocks_present,
    next_task_id,
    refill_open_task_capacity,
    task_header_prefix,
    task_id_prefix,
    task_ids_from_todo_text,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    EvidenceSourcePolicy,
    next_task_id as next_objective_task_id,
    normalize_task_id_prefix as normalize_objective_task_id_prefix,
    task_markdown_heading_prefix,
)
from ipfs_accelerate_py.agent_supervisor.task_quality import (
    TASK_SPLIT_REFILL_REQUIREMENT_ID,
    TASK_WORK_CONTRACT_SCHEMA,
    TaskCandidate,
    TaskQualityPolicy,
    TaskSplitRefillEvidence,
    admit_task_candidate,
    can_coalesce_tasks,
    canonical_semantic_identity,
    coalesce_task_candidates,
    prove_task_split_refill,
    refine_task_candidates,
    score_task_candidate,
    split_task_candidate,
)


def _candidate(**overrides: object) -> TaskCandidate:
    payload: dict[str, object] = {
        "title": "Make cache admission deterministic",
        "goal_id": "CACHE-G001",
        "context_paths": ["src/cache.py", "test/cache.py"],
        "outputs": ["src/cache.py", "test/cache.py"],
        "validation_commands": ["python -m pytest test/cache.py -q"],
        "merge_fate": "cache-admission",
        "acceptance_criteria": [
            "Equivalent inputs share one cache entry",
            "Concurrent callers observe the same result",
        ],
        "predicted_paths": ["src/cache.py", "test/cache.py"],
        "predicted_symbols": ["Cache.admit", "test_cache_admission"],
        "preconditions": ["The cache key has been canonicalized"],
        "effects": ["One canonical cache entry is persisted"],
        "evidence_subset": ["cache-admission-unit-tests"],
        "resource_class": "cpu-medium",
        "token_class": "medium",
        "estimated_tokens": 2400,
        "dependencies": ["CACHE-001"],
        "conflicts": ["cache-writer"],
        "estimated_context_tokens": 1200,
        "estimated_validation_seconds": 30,
    }
    payload.update(overrides)
    return TaskCandidate.from_mapping(payload)


def test_candidate_contract_has_stable_canonical_semantic_identity():
    first = _candidate()
    reordered = _candidate(
        context_paths=["test/cache.py", "src/cache.py"],
        outputs=["test/cache.py", "src/cache.py"],
        acceptance_criteria=list(reversed(first.acceptance)),
        predicted_paths=list(reversed(first.predicted_paths)),
        predicted_symbols=list(reversed(first.predicted_symbols)),
        dependencies=["CACHE-001", "CACHE-001"],
    )

    assert first.semantic_identity == canonical_semantic_identity(first)
    assert reordered.semantic_identity == first.semantic_identity
    assert first.preconditions
    assert first.effects
    assert first.evidence_subset
    assert first.resource_class == "cpu-medium"
    assert first.token_class == "medium"


def test_work_contract_binds_exact_coherent_subset_scope_and_predicted_costs():
    candidate = _candidate()
    contract = candidate.work_contract

    assert contract == {
        "schema": TASK_WORK_CONTRACT_SCHEMA,
        "goal_id": "cache g001",
        "acceptance_effect_subset": {
            "acceptance": [
                "concurrent callers observe the same result",
                "equivalent inputs share one cache entry",
            ],
            "effects": ["one canonical cache entry is persisted"],
            "evidence_subset": ["cache admission unit tests"],
        },
        "predicted_scope": {
            "paths": ["src/cache.py", "test/cache.py"],
            "symbols": ["cache admit", "test cache admission"],
            "context_paths": ["src/cache.py", "test/cache.py"],
        },
        "predicted_costs": {
            "context_tokens": 1200,
            "validation_seconds": 30,
            "task_tokens": 2400,
            "resource_class": "cpu-medium",
            "token_class": "medium",
            "dependency_count": 1,
            "conflict_count": 1,
        },
        "execution_boundary": {
            "preconditions": ["the cache key has been canonicalized"],
            "dependencies": ["CACHE-001"],
            "conflicts": ["cache-writer"],
            "validation_commands": ["python -m pytest test/cache.py -q"],
            "merge_fate": "cache admission",
        },
    }
    assert candidate.predicted_costs_complete is True
    assert candidate.work_contract_id

    reordered = _candidate(
        acceptance_criteria=list(reversed(candidate.acceptance)),
        effects=list(reversed(candidate.effects)),
        evidence_subset=list(reversed(candidate.evidence_subset)),
        predicted_paths=list(reversed(candidate.predicted_paths)),
        predicted_symbols=list(reversed(candidate.predicted_symbols)),
        context_paths=list(reversed(candidate.context_paths)),
    )
    assert reordered.work_contract == contract
    assert reordered.work_contract_id == candidate.work_contract_id


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("predicted_paths", ["src/cache.py", "test/cache_v2.py"]),
        ("predicted_symbols", ["Cache.admit", "test_cache_admission_v2"]),
        ("context_paths", ["src/cache.py", "test/cache_v2.py"]),
        ("estimated_context_tokens", 1201),
        ("estimated_validation_seconds", 31),
        ("estimated_tokens", 2401),
    ],
)
def test_canonical_identity_changes_with_predicted_scope_or_any_mandatory_cost(
    field_name: str,
    changed_value: object,
):
    baseline = _candidate()
    changed = _candidate(**{field_name: changed_value})

    assert changed.work_contract != baseline.work_contract
    assert changed.work_contract_id != baseline.work_contract_id
    assert changed.semantic_identity != baseline.semantic_identity
    assert changed.canonical_task_key != baseline.canonical_task_key
    assert changed.canonical_task_cid != baseline.canonical_task_cid


@pytest.mark.parametrize(
    ("field_name", "expected_reason"),
    [
        (
            "estimated_context_tokens",
            "missing_estimated_context_tokens",
        ),
        (
            "estimated_validation_seconds",
            "missing_estimated_validation_seconds",
        ),
        ("estimated_tokens", "missing_estimated_tokens"),
    ],
)
def test_admission_fails_closed_with_specific_reason_for_each_zero_cost(
    field_name: str,
    expected_reason: str,
):
    candidate = _candidate(**{field_name: 0})
    decision = admit_task_candidate(candidate)

    assert candidate.predicted_costs_complete is False
    assert decision.accepted is False
    assert expected_reason in decision.rejection_reasons
    reason = next(item for item in decision.rejections if item.code == expected_reason)
    assert reason.semantic_identity == candidate.semantic_identity
    assert field_name in reason.detail


def test_candidate_projection_round_trip_preserves_identity_and_rejects_tampering():
    candidate = _candidate(source_id="ASI-051")
    projection = candidate.to_dict()

    restored = TaskCandidate.from_dict(projection)

    assert restored == candidate
    assert restored.semantic_identity == projection["canonical_semantic_identity"]
    assert projection["canonical_task_key"] == candidate.canonical_task_key
    assert projection["canonical_task_cid"] == candidate.canonical_task_cid
    assert projection["task_cid"] == candidate.canonical_task_cid
    assert projection["source_id"] == "ASI-051"

    projection["canonical_semantic_identity"] = "task-quality/v1/not-the-content"
    with pytest.raises(
        ValueError,
        match="semantic_identity does not match canonical semantic task content",
    ):
        TaskCandidate.from_dict(projection)

    projection = candidate.to_dict()
    projection["work_contract"]["predicted_costs"]["task_tokens"] += 1
    with pytest.raises(
        ValueError,
        match="work_contract does not match canonical task work content",
    ):
        TaskCandidate.from_dict(projection)

    projection = candidate.to_dict()
    projection["work_contract_id"] = "not-the-canonical-contract"
    with pytest.raises(
        ValueError,
        match="work_contract_id does not match canonical task work content",
    ):
        TaskCandidate.from_dict(projection)

    for field_name in ("canonical_task_key", "canonical_task_cid"):
        projection = candidate.to_dict()
        projection[field_name] = f"tampered-{field_name}"
        with pytest.raises(ValueError, match=field_name):
            TaskCandidate.from_dict(projection)


def test_quality_score_is_multidimensional_and_rejects_incomplete_contracts():
    complete = score_task_candidate(_candidate())
    incomplete_candidate = _candidate(
        acceptance_criteria=[],
        preconditions=[],
        effects=[],
        evidence_subset=[],
        resource_class="",
        token_class="",
        validation_commands=[],
    )
    incomplete = score_task_candidate(incomplete_candidate)
    complete_decision = admit_task_candidate(_candidate())
    incomplete_decision = admit_task_candidate(incomplete_candidate)

    assert complete_decision.accepted is True
    assert complete.total > incomplete.total
    assert complete.acceptance_coverage > 0
    assert complete.coherent_effects > 0
    assert complete.breadth_fit > 0
    assert complete.context_cost > 0
    assert complete.validation_cost > 0
    assert complete.dependency_cost > 0
    assert complete.conflict_cost > 0
    assert complete.resource_fit == 1.0
    assert complete.historical_novelty == 1.0
    assert complete.historical_failure_safety == 1.0
    assert incomplete_decision.accepted is False
    assert incomplete_decision.rejections
    assert all(reason.code and reason.detail for reason in incomplete_decision.rejections)


def test_quality_dimensions_respond_to_scope_cost_dependency_conflict_and_resources():
    policy = TaskQualityPolicy(
        max_predicted_paths=2,
        max_predicted_symbols=2,
        max_context_paths=2,
        max_context_tokens=2_000,
        max_validation_seconds=60,
        max_dependencies=2,
        max_conflicts=2,
    )
    baseline = score_task_candidate(_candidate(), policy=policy)
    expensive = score_task_candidate(
        _candidate(
            context_paths=[f"src/context_{index}.py" for index in range(5)],
            predicted_paths=[f"src/component_{index}.py" for index in range(5)],
            predicted_symbols=[f"Component{index}.run" for index in range(5)],
            estimated_context_tokens=10_000,
            estimated_validation_seconds=300,
            dependencies=[f"DEP-{index}" for index in range(5)],
            conflicts=[f"surface-{index}" for index in range(5)],
            resource_class="unsupported-accelerator",
        ),
        policy=policy,
    )

    assert expensive.breadth_fit < baseline.breadth_fit
    assert expensive.context_cost < baseline.context_cost
    assert expensive.validation_cost < baseline.validation_cost
    assert expensive.dependency_cost < baseline.dependency_cost
    assert expensive.conflict_cost < baseline.conflict_cost
    assert expensive.resource_fit < baseline.resource_fit
    decision = admit_task_candidate(
        _candidate(resource_class="unsupported-accelerator"),
        policy=policy,
    )
    assert "invalid_resource_class" in decision.rejection_reasons


def test_duplicate_and_failure_history_reduce_quality_and_explain_rejection():
    clean = score_task_candidate(_candidate())
    duplicate = score_task_candidate(
        _candidate(),
        historical_tasks=(_candidate(title="Already implemented"),),
    )
    repeatedly_failed = score_task_candidate(
        _candidate(),
        historical_failures=(
            {
                "candidate": _candidate(title="Failed wording"),
                "outcome": "failed",
                "failure_reason": "unchanged validation failure",
            },
        ),
    )
    duplicate_decision = admit_task_candidate(
        _candidate(),
        historical_tasks=(_candidate(title="Already implemented"),),
    )
    failed_decision = admit_task_candidate(
        _candidate(),
        historical_failures=(_candidate(title="Failed wording"),),
    )

    assert duplicate.total < clean.total
    assert repeatedly_failed.total < clean.total
    assert duplicate.duplicate_similarity == 1.0
    assert repeatedly_failed.failure_similarity == 1.0
    assert duplicate_decision.accepted is False
    assert failed_decision.accepted is False
    assert "historical_duplicate" in duplicate_decision.rejection_reasons
    assert "historical_failure" in failed_decision.rejection_reasons


def test_over_broad_task_splits_deterministically_and_preserves_dependencies():
    broad = _candidate(
        acceptance_criteria=[f"criterion-{index}" for index in range(7)],
        predicted_paths=[f"src/component_{index}.py" for index in range(7)],
        predicted_symbols=[f"Component{index}.run" for index in range(7)],
        effects=[f"effect-{index}" for index in range(7)],
        evidence_subset=[f"evidence-{index}" for index in range(7)],
        dependencies=["CACHE-001", "CACHE-002"],
        estimated_tokens=14_000,
    )
    policy = TaskQualityPolicy(
        max_predicted_paths=2,
        max_predicted_symbols=2,
        max_estimated_tokens=4_000,
    )

    first = split_task_candidate(broad, policy=policy)
    repeated = split_task_candidate(broad, policy=policy)

    assert len(first) > 1
    assert [item.semantic_identity for item in first] == [
        item.semantic_identity for item in repeated
    ]
    assert set().union(*(set(item.acceptance) for item in first)) == set(
        broad.acceptance
    )
    assert set().union(*(set(item.predicted_paths) for item in first)) == (
        set(broad.predicted_paths) | set(broad.outputs)
    )
    assert all(
        set(item.dependencies).issuperset(broad.dependencies)
        for item in first
    )
    assert all(item.goal_id == broad.goal_id for item in first)


def test_split_recomputes_child_contracts_and_preserves_full_source_coverage():
    source = _candidate(
        acceptance_criteria=[f"criterion-{index}" for index in range(4)],
        effects=[f"effect-{index}" for index in range(4)],
        evidence_subset=[f"evidence-{index}" for index in range(4)],
        context_paths=[f"src/part_{index}.py" for index in range(4)],
        outputs=[f"src/part_{index}.py" for index in range(4)],
        predicted_paths=[f"src/part_{index}.py" for index in range(4)],
        predicted_symbols=[f"Part{index}.run" for index in range(4)],
        estimated_context_tokens=2_400,
        estimated_validation_seconds=40,
        estimated_tokens=8_000,
    )
    policy = TaskQualityPolicy(
        max_predicted_paths=2,
        max_predicted_symbols=2,
        max_acceptance_criteria=2,
        max_effects=2,
        max_evidence_items=2,
        max_context_paths=2,
        max_context_tokens=1_200,
        max_estimated_tokens=4_000,
    )

    children = split_task_candidate(source, policy=policy)

    assert len(children) == 2
    assert len({child.work_contract_id for child in children}) == len(children)
    assert source.work_contract_id not in {
        child.work_contract_id for child in children
    }
    assert all(child.predicted_costs_complete for child in children)
    assert all(
        TaskCandidate.from_dict(child.to_dict()).work_contract_id
        == child.work_contract_id
        for child in children
    )

    for subset_key in ("acceptance", "effects", "evidence_subset"):
        assert set().union(
            *(
                set(child.work_contract["acceptance_effect_subset"][subset_key])
                for child in children
            )
        ) == set(source.work_contract["acceptance_effect_subset"][subset_key])
    for scope_key in ("paths", "symbols", "context_paths"):
        assert set().union(
            *(
                set(child.work_contract["predicted_scope"][scope_key])
                for child in children
            )
        ) == set(source.work_contract["predicted_scope"][scope_key])
    for boundary_key in (
        "preconditions",
        "dependencies",
        "conflicts",
        "validation_commands",
    ):
        assert set().union(
            *(
                set(child.work_contract["execution_boundary"][boundary_key])
                for child in children
            )
        ) == set(source.work_contract["execution_boundary"][boundary_key])
    assert all(
        child.work_contract["execution_boundary"]["merge_fate"]
        == source.work_contract["execution_boundary"]["merge_fate"]
        for child in children
    )

    child_costs = [child.work_contract["predicted_costs"] for child in children]
    source_costs = source.work_contract["predicted_costs"]
    assert sum(cost["context_tokens"] for cost in child_costs) == (
        source_costs["context_tokens"]
    )
    assert sum(cost["task_tokens"] for cost in child_costs) == (
        source_costs["task_tokens"]
    )
    assert {cost["validation_seconds"] for cost in child_costs} == {
        source_costs["validation_seconds"]
    }


def test_tiny_tasks_coalesce_only_with_shared_execution_and_merge_fate():
    left = _candidate(
        title="Add cache hit assertion",
        acceptance_criteria=["A cache hit is observable"],
        predicted_symbols=["test_cache_hit"],
        effects=["Cache-hit behavior is covered"],
        evidence_subset=["cache-hit-test"],
        estimated_tokens=300,
    )
    right = _candidate(
        title="Add cache miss assertion",
        acceptance_criteria=["A cache miss is observable"],
        predicted_symbols=["test_cache_miss"],
        effects=["Cache-miss behavior is covered"],
        evidence_subset=["cache-miss-test"],
        estimated_tokens=350,
    )
    policy = TaskQualityPolicy(tiny_max_paths=2)

    assert can_coalesce_tasks(left, right, policy=policy)
    coalesced = coalesce_task_candidates((right, left), policy=policy)
    coalesced_repeated = coalesce_task_candidates((left, right), policy=policy)
    assert coalesced.semantic_identity == coalesced_repeated.semantic_identity
    assert set(coalesced.acceptance) == {
        *left.acceptance,
        *right.acceptance,
    }
    assert coalesced.dependencies == left.dependencies

    shared_fields = {
        "goal_id": "OTHER-GOAL",
        "context_paths": ["src/other.py"],
        "outputs": ["src/other.py"],
        "validation_commands": ["python -m pytest test/other.py -q"],
        "merge_fate": "other-merge",
    }
    for field_name, different_value in shared_fields.items():
        assert not can_coalesce_tasks(
            left,
            _candidate(
                title=f"Different {field_name}",
                **{field_name: different_value},
            ),
            policy=policy,
        )


def test_coalesce_recomputes_contract_and_preserves_all_source_contract_coverage():
    left = _candidate(
        source_id="CACHE-LEFT",
        title="Prove deterministic cache hits",
        acceptance_criteria=["Cache hits are deterministic"],
        effects=["Cache-hit state is observable"],
        evidence_subset=["cache-hit-proof"],
        predicted_symbols=["test_cache_hit"],
        dependencies=["CACHE-001"],
        conflicts=["cache-hit-writer"],
        estimated_context_tokens=400,
        estimated_validation_seconds=10,
        estimated_tokens=300,
    )
    right = _candidate(
        source_id="CACHE-RIGHT",
        title="Prove deterministic cache misses",
        acceptance_criteria=["Cache misses are deterministic"],
        effects=["Cache-miss state is observable"],
        evidence_subset=["cache-miss-proof"],
        predicted_symbols=["test_cache_miss"],
        dependencies=["CACHE-002"],
        conflicts=["cache-miss-writer"],
        estimated_context_tokens=600,
        estimated_validation_seconds=20,
        estimated_tokens=350,
    )

    merged = coalesce_task_candidates(
        (right, left),
        policy=TaskQualityPolicy(tiny_max_paths=2),
    )
    repeated = coalesce_task_candidates(
        (left, right),
        policy=TaskQualityPolicy(tiny_max_paths=2),
    )

    assert merged.work_contract == repeated.work_contract
    assert merged.work_contract_id == repeated.work_contract_id
    assert merged.work_contract_id not in {
        left.work_contract_id,
        right.work_contract_id,
    }
    assert TaskCandidate.from_dict(merged.to_dict()) == merged

    source_contracts = (left.work_contract, right.work_contract)
    for subset_key in ("acceptance", "effects", "evidence_subset"):
        expected = set().union(
            *(
                set(contract["acceptance_effect_subset"][subset_key])
                for contract in source_contracts
            )
        )
        assert set(
            merged.work_contract["acceptance_effect_subset"][subset_key]
        ) == expected
    for scope_key in ("paths", "symbols", "context_paths"):
        expected = set().union(
            *(
                set(contract["predicted_scope"][scope_key])
                for contract in source_contracts
            )
        )
        assert set(merged.work_contract["predicted_scope"][scope_key]) == expected
    for boundary_key in (
        "preconditions",
        "dependencies",
        "conflicts",
        "validation_commands",
    ):
        expected = set().union(
            *(
                set(contract["execution_boundary"][boundary_key])
                for contract in source_contracts
            )
        )
        assert set(
            merged.work_contract["execution_boundary"][boundary_key]
        ) == expected

    merged_costs = merged.work_contract["predicted_costs"]
    assert merged_costs["context_tokens"] == 600
    assert merged_costs["validation_seconds"] == 20
    assert merged_costs["task_tokens"] == 650
    assert merged_costs["dependency_count"] == 2
    assert merged_costs["conflict_count"] == 2


def test_refinement_rejects_existing_semantic_duplicate_and_bounds_open_work():
    existing = _candidate()
    duplicate_wording = _candidate(title="Rephrased cache task")
    novel = _candidate(
        title="Make cache eviction deterministic",
        goal_id="CACHE-G002",
        merge_fate="cache-eviction",
        acceptance_criteria=["Expired entries are evicted deterministically"],
        effects=["Expired entries are removed in key order"],
        evidence_subset=["cache-eviction-tests"],
        predicted_symbols=["Cache.evict", "test_cache_eviction"],
    )

    duplicate_result = refine_task_candidates(
        (duplicate_wording, novel),
        existing_tasks=(existing,),
        current_open_work=0,
    )
    duplicate_codes = {
        reason.code
        for rejection in duplicate_result.rejected
        for reason in rejection.rejections
    }
    assert "historical_duplicate" in duplicate_codes
    assert novel.semantic_identity in {
        candidate.semantic_identity for candidate in duplicate_result.accepted
    }

    pressure_result = refine_task_candidates(
        (novel,),
        policy=TaskQualityPolicy(max_open_work=3),
        current_open_work=3,
    )
    assert pressure_result.accepted == ()
    assert pressure_result.rejected
    assert pressure_result.final_open_work == 3


def test_refinement_projection_retains_canonical_source_lineage_after_resizing():
    broad = _candidate(
        source_id="ASI-051",
        predicted_paths=[f"src/part_{index}.py" for index in range(4)],
        outputs=[f"src/part_{index}.py" for index in range(4)],
        predicted_symbols=[f"Part{index}.run" for index in range(4)],
        estimated_tokens=8_000,
    )
    result = refine_task_candidates(
        (broad,),
        policy=TaskQualityPolicy(
            max_predicted_paths=2,
            max_predicted_symbols=2,
            max_estimated_tokens=4_000,
            coalesce_tiny=False,
        ),
    )

    assert len(result.accepted) == 2
    assert all(
        decision.source_identities == (broad.semantic_identity,)
        for decision in result.decisions
    )
    projection = result.to_dict()
    assert {
        item["candidate"]["canonical_semantic_identity"]
        for item in projection["decisions"]
    } == {item.semantic_identity for item in result.accepted}
    assert all(
        item["source_identities"] == [broad.semantic_identity]
        for item in projection["decisions"]
    )


def test_refinement_projection_retains_all_sources_when_tiny_tasks_coalesce():
    left = _candidate(
        source_id="ASI-051-A",
        title="Cover cache hit",
        acceptance_criteria=["Cache hits are covered"],
        effects=["Cache-hit coverage exists"],
        evidence_subset=["cache-hit-evidence"],
        predicted_symbols=["test_cache_hit"],
        estimated_tokens=300,
    )
    right = _candidate(
        source_id="ASI-051-B",
        title="Cover cache miss",
        acceptance_criteria=["Cache misses are covered"],
        effects=["Cache-miss coverage exists"],
        evidence_subset=["cache-miss-evidence"],
        predicted_symbols=["test_cache_miss"],
        estimated_tokens=300,
    )

    result = refine_task_candidates(
        (right, left),
        policy=TaskQualityPolicy(tiny_max_paths=2),
    )

    assert len(result.accepted) == 1
    decision = result.decisions[0]
    assert decision.status.value == "coalesced"
    assert set(decision.source_identities) == {
        left.semantic_identity,
        right.semantic_identity,
    }
    assert result.to_dict()["decisions"][0]["source_identities"] == list(
        decision.source_identities
    )


def test_broad_split_refill_evidence_proves_zero_duplicate_admission():
    broad = _candidate(
        source_id="ASI-034",
        acceptance_criteria=[f"criterion-{index}" for index in range(6)],
        effects=[f"effect-{index}" for index in range(6)],
        evidence_subset=[f"evidence-{index}" for index in range(6)],
        context_paths=[f"src/part_{index}.py" for index in range(6)],
        outputs=[f"src/part_{index}.py" for index in range(6)],
        predicted_paths=[f"src/part_{index}.py" for index in range(6)],
        predicted_symbols=[f"Part{index}.run" for index in range(6)],
        estimated_tokens=12_000,
    )
    policy = TaskQualityPolicy(
        max_predicted_paths=2,
        max_predicted_symbols=2,
        max_acceptance_criteria=2,
        max_effects=2,
        max_evidence_items=2,
        max_context_paths=2,
        max_estimated_tokens=4_000,
        max_new_work=8,
        coalesce_tiny=False,
    )

    evidence = prove_task_split_refill(
        broad,
        policy=policy,
        initial_open_work=3,
        repository_tree="git-tree-asi-034",
    )
    repeated = prove_task_split_refill(
        broad.to_dict(),
        policy=policy,
        initial_open_work=3,
        repository_tree="git-tree-asi-034",
    )

    assert evidence.verify_integrity()
    assert evidence.evidence_id == repeated.evidence_id
    assert evidence.proved_requirement_ids == (
        TASK_SPLIT_REFILL_REQUIREMENT_ID,
    )
    first = evidence.first_admission
    refill = evidence.refill_admission
    assert len(first["accepted"]) == 3
    assert refill["accepted"] == []
    assert refill["initial_open_work"] == first["final_open_work"]
    assert refill["final_open_work"] == first["final_open_work"]
    assert {
        item["candidate"]["canonical_task_cid"]
        for item in refill["decisions"]
    } == {
        item["canonical_task_cid"] for item in first["accepted"]
    }
    assert all(
        {
            rejection["reason"]
            for rejection in item["rejections"]
        }
        & {"historical_duplicate", "duplicate_semantic_identity"}
        for item in refill["decisions"]
    )

    decision = EvidenceSourcePolicy().validate_completion_evidence(
        TASK_SPLIT_REFILL_REQUIREMENT_ID,
        evidence,
        repository_tree="git-tree-asi-034",
        policy_id=policy.policy_id,
    )
    assert decision.satisfies is True


def test_broad_split_refill_evidence_fails_closed_for_partial_or_forged_receipts():
    broad = _candidate(
        acceptance_criteria=[f"criterion-{index}" for index in range(4)],
        effects=[f"effect-{index}" for index in range(4)],
        evidence_subset=[f"evidence-{index}" for index in range(4)],
        context_paths=[f"src/part_{index}.py" for index in range(4)],
        outputs=[f"src/part_{index}.py" for index in range(4)],
        predicted_paths=[f"src/part_{index}.py" for index in range(4)],
        predicted_symbols=[f"Part{index}.run" for index in range(4)],
        estimated_tokens=8_000,
    )
    complete_policy = TaskQualityPolicy(
        max_predicted_paths=2,
        max_predicted_symbols=2,
        max_acceptance_criteria=2,
        max_effects=2,
        max_evidence_items=2,
        max_context_paths=2,
        max_estimated_tokens=4_000,
        max_new_work=4,
        coalesce_tiny=False,
    )
    valid = TaskSplitRefillEvidence.create(
        broad,
        policy=complete_policy,
        repository_tree="git-tree-asi-034",
    )
    capacity_limited = TaskSplitRefillEvidence.create(
        broad,
        policy=TaskQualityPolicy(
            max_predicted_paths=2,
            max_predicted_symbols=2,
            max_acceptance_criteria=2,
            max_effects=2,
            max_evidence_items=2,
            max_context_paths=2,
            max_estimated_tokens=4_000,
            max_new_work=1,
            coalesce_tiny=False,
        ),
        repository_tree="git-tree-asi-034",
    )
    unbound = TaskSplitRefillEvidence.create(
        broad,
        policy=complete_policy,
    )

    assert valid.proved_requirement_ids
    assert capacity_limited.proved_requirement_ids == ()
    assert unbound.verify_integrity()
    assert unbound.proved_requirement_ids == ()
    restored = TaskSplitRefillEvidence.from_dict(valid.to_dict())
    assert restored.verify_integrity()
    assert restored.proved_requirement_ids == ()
    restored_decision = EvidenceSourcePolicy().validate_completion_evidence(
        TASK_SPLIT_REFILL_REQUIREMENT_ID,
        restored,
        repository_tree="git-tree-asi-034",
        policy_id=complete_policy.policy_id,
    )
    assert restored_decision.satisfies is False
    assert "receipt_producer_authority_missing" in restored_decision.reason_codes

    forged = {
        "schema": "caller-authored-lookalike@1",
        "evidence_id": "fake-evidence",
        "requirement_id": TASK_SPLIT_REFILL_REQUIREMENT_ID,
        "repository_tree": "git-tree-asi-034",
        "policy_id": complete_policy.policy_id,
        "source_tier": "validation",
        "status": "passed",
        "complete": True,
        "coverage_complete": True,
    }
    forged_decision = EvidenceSourcePolicy().validate_completion_evidence(
        TASK_SPLIT_REFILL_REQUIREMENT_ID,
        forged,
        repository_tree="git-tree-asi-034",
        policy_id=complete_policy.policy_id,
    )
    assert forged_decision.satisfies is False
    assert "receipt_producer_authority_missing" in forged_decision.reason_codes

    tampered = copy.deepcopy(valid.to_dict())
    tampered["refill_admission"]["accepted"].append(
        tampered["first_admission"]["accepted"][0]
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        TaskSplitRefillEvidence.from_dict(tampered)


def test_legacy_heading_prefix_is_normalized_once_and_ids_remain_monotonic():
    todo = """# Work

## ASI-007 Existing
- Status: completed

## ASI-010 Existing
- Status: todo

## OTHER-999 Unrelated
- Status: todo
"""

    assert task_id_prefix("## ASI-") == "ASI-"
    assert task_id_prefix("## ## ASI-") == "ASI-"
    assert task_header_prefix("ASI-") == "## ASI-"
    assert task_header_prefix("## ASI-") == "## ASI-"
    assert "## ##" not in task_header_prefix("## ## ASI-")
    assert task_ids_from_todo_text(todo, task_prefix="## ASI-") == [
        "ASI-007",
        "ASI-010",
    ]
    assert next_task_id(
        todo,
        task_prefix="## ASI-",
        reserved_task_ids=("## ASI-011", "OTHER-1000"),
    ) == "ASI-012"
    assert normalize_objective_task_id_prefix("## ## ASI-") == "ASI-"
    assert task_markdown_heading_prefix("## ## ASI-") == "## ASI-"
    assert next_objective_task_id(
        todo,
        task_prefix="## ASI-",
        reserved_task_ids=("## ASI-011", "OTHER-1000"),
    ) == "ASI-012"


def test_refill_capacity_bounds_open_work_pressure():
    assert refill_open_task_capacity(
        current_open=0,
        min_open_tasks=5,
        max_findings=20,
    ) == 6
    assert refill_open_task_capacity(
        current_open=5,
        min_open_tasks=5,
        max_findings=20,
    ) == 1
    assert refill_open_task_capacity(
        current_open=6,
        min_open_tasks=5,
        max_findings=20,
    ) == 0
    assert refill_open_task_capacity(
        current_open=0,
        min_open_tasks=5,
        max_findings=3,
    ) == 3


def test_refill_deduplicates_semantic_aliases_and_normalizes_block_heading(tmp_path):
    todo_path = tmp_path / "todo.md"
    todo_path.write_text(
        """# Work

## ASI-001 Existing wording

- Status: todo
- Canonical task key: task/v1/shared-work
""",
        encoding="utf-8",
    )

    assert not ensure_task_blocks_present(
        todo_path,
        (
            (
                "## ASI-002",
                """## ## ASI-002 Rephrased duplicate

- Status: todo
- Semantic identity: task/v1/shared-work
""",
            ),
        ),
    )
    assert "ASI-002" not in todo_path.read_text(encoding="utf-8")

    assert ensure_task_blocks_present(
        todo_path,
        (
            (
                "## ASI-003",
                """## ## ASI-003 Distinct work

- Status: todo
- Semantic identity: task/v1/distinct-work
""",
            ),
        ),
    )
    updated = todo_path.read_text(encoding="utf-8")
    assert "## ASI-003 Distinct work" in updated
    assert "## ## ASI-" not in updated
