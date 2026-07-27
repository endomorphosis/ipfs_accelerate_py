from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.task_quality import (
    TaskCandidate,
    TaskCostMeasurement,
    TaskGranularityRun,
    TaskQualityPolicy,
    calibrate_task_granularity,
    can_coalesce_tasks,
    coalesce_task_candidates,
    compare_task_granularity_runs,
    is_over_broad,
    propagate_task_completion,
    refine_task_candidates,
)


TREE = "git-tree-granularity"
FEATURES = ("python:3.12", "pytest:9", "linux:x86_64")


def _candidate(**overrides: object) -> TaskCandidate:
    payload: dict[str, object] = {
        "title": "Calibrate cache task granularity",
        "goal_id": "CACHE-GRANULARITY",
        "acceptance_criteria": ["criterion-0"],
        "preconditions": ["The task source is frozen"],
        "effects": ["effect-0"],
        "evidence_subset": ["evidence-0"],
        "context_paths": ["src/cache.py"],
        "outputs": ["src/cache.py"],
        "predicted_paths": ["src/cache.py"],
        "predicted_symbols": ["Cache.run"],
        "predicted_interfaces": ["CacheProtocol.run"],
        "validation_commands": ["python -m pytest test/cache.py -q"],
        "proof_obligations": ["Cache completion is deterministic"],
        "proof_commands": ["python -m pytest test/cache_proof.py -q"],
        "resource_class": "cpu-medium",
        "token_class": "medium",
        "merge_fate": "cache-family",
        "estimated_context_tokens": 400,
        "estimated_validation_seconds": 10,
        "estimated_proof_seconds": 10,
        "estimated_tokens": 600,
        "estimated_merge_risk_millionths": 50_000,
        "dependencies": ["CACHE-BASE"],
    }
    payload.update(overrides)
    return TaskCandidate.from_mapping(payload)


def _measurement(
    policy: TaskQualityPolicy,
    *,
    fixture_id: str = "matching",
    repository_tree: str = TREE,
    policy_id: str | None = None,
    features: tuple[str, ...] = FEATURES,
    acceptance_count: int = 2,
    context_path_count: int = 2,
    context_tokens: int = 1_000,
    predicted_file_count: int = 2,
    predicted_symbol_count: int = 2,
    predicted_interface_count: int = 2,
    validation_seconds: int = 10,
    proof_item_count: int = 1,
    proof_seconds: int = 10,
    task_tokens: int = 1_000,
    merge_risk_millionths: int = 100_000,
    model_calls: int = 1,
    accepted_criteria: int = 2,
) -> TaskCostMeasurement:
    return TaskCostMeasurement(
        fixture_id=fixture_id,
        repository_tree=repository_tree,
        policy_id=policy_id or policy.policy_id,
        toolchain_features=features,
        acceptance_count=acceptance_count,
        context_path_count=context_path_count,
        context_tokens=context_tokens,
        predicted_file_count=predicted_file_count,
        predicted_symbol_count=predicted_symbol_count,
        predicted_interface_count=predicted_interface_count,
        validation_seconds=validation_seconds,
        proof_item_count=proof_item_count,
        proof_seconds=proof_seconds,
        task_tokens=task_tokens,
        merge_risk_millionths=merge_risk_millionths,
        model_calls=model_calls,
        accepted_criteria=accepted_criteria,
    )


def test_contract_binds_exact_subset_scope_interfaces_proof_cost_and_merge_fate():
    candidate = _candidate()
    contract = candidate.work_contract

    assert contract["acceptance_effect_subset"] == {
        "acceptance": ["criterion 0"],
        "effects": ["effect 0"],
        "evidence_subset": ["evidence 0"],
    }
    assert contract["predicted_scope"] == {
        "paths": ["src/cache.py"],
        "symbols": ["cache run"],
        "context_paths": ["src/cache.py"],
        "interfaces": ["cacheprotocol run"],
    }
    assert contract["execution_boundary"]["validation_commands"]
    assert contract["predicted_proof"] == {
        "obligations": ["cache completion is deterministic"],
        "commands": ["python -m pytest test/cache_proof.py -q"],
        "estimated_seconds": 10,
    }
    assert contract["predicted_costs"]["resource_class"] == "cpu-medium"
    assert contract["predicted_costs"]["task_tokens"] == 600
    assert contract["predicted_costs"]["merge_risk_millionths"] == 50_000
    assert contract["execution_boundary"]["merge_fate"] == "cache family"

    changed = replace(
        candidate,
        predicted_interfaces=("CacheProtocol.close",),
        semantic_identity="",
    )
    assert changed.work_contract_id != candidate.work_contract_id
    assert changed.semantic_identity != candidate.semantic_identity


def test_calibration_uses_only_exact_tree_policy_and_toolchain_history():
    policy = TaskQualityPolicy()
    matching = _measurement(policy)
    foreign_tree = _measurement(
        policy,
        fixture_id="foreign-tree",
        repository_tree="git-tree-other",
        acceptance_count=12,
        task_tokens=20_000,
    )
    foreign_policy = _measurement(
        policy,
        fixture_id="foreign-policy",
        policy_id="policy:other",
        context_tokens=20_000,
    )
    foreign_toolchain = _measurement(
        policy,
        fixture_id="foreign-toolchain",
        features=("python:3.13", "pytest:9", "linux:x86_64"),
        validation_seconds=1_000,
    )

    first = calibrate_task_granularity(
        (foreign_tree, matching, foreign_policy, foreign_toolchain),
        repository_tree=TREE,
        policy=policy,
        toolchain_features=FEATURES,
    )
    repeated = calibrate_task_granularity(
        (foreign_toolchain, foreign_policy, matching, foreign_tree),
        repository_tree=TREE,
        policy=policy,
        toolchain_features=reversed(FEATURES),
    )

    assert first.calibration_id == repeated.calibration_id
    matching_only = calibrate_task_granularity(
        (matching,),
        repository_tree=TREE,
        policy=policy,
        toolchain_features=FEATURES,
    )
    assert matching_only.calibration_id == first.calibration_id
    assert first.matching_measurement_ids == (matching.measurement_id,)
    assert set(first.excluded_measurement_ids) == {
        foreign_tree.measurement_id,
        foreign_policy.measurement_id,
        foreign_toolchain.measurement_id,
    }
    assert first.effective_policy.max_acceptance_criteria == 2
    assert first.effective_policy.max_context_tokens == 1_000
    assert first.effective_policy.max_validation_seconds == 10
    assert first.effective_policy.max_proof_seconds == 10
    assert first.effective_policy.max_estimated_tokens == 1_000
    assert first.effective_policy.max_merge_risk_millionths == 100_000

    with pytest.raises(ValueError, match="no task cost measurements match"):
        calibrate_task_granularity(
            (foreign_tree, foreign_policy, foreign_toolchain),
            repository_tree=TREE,
            policy=policy,
            toolchain_features=FEATURES,
        )


def test_measured_bounds_split_every_cost_surface_and_preserve_coverage():
    policy = TaskQualityPolicy(coalesce_tiny=False)
    measurement = _measurement(policy)
    broad = _candidate(
        source_id="BROAD",
        acceptance_criteria=[f"criterion-{index}" for index in range(4)],
        effects=[f"effect-{index}" for index in range(4)],
        evidence_subset=[f"evidence-{index}" for index in range(4)],
        context_paths=[f"src/part_{index}.py" for index in range(4)],
        outputs=[f"src/part_{index}.py" for index in range(4)],
        predicted_paths=[f"src/part_{index}.py" for index in range(4)],
        predicted_symbols=[f"Part{index}.run" for index in range(4)],
        predicted_interfaces=[f"PartProtocol{index}" for index in range(4)],
        proof_obligations=[f"proof-{index}" for index in range(4)],
        estimated_context_tokens=4_000,
        estimated_validation_seconds=40,
        estimated_proof_seconds=40,
        estimated_tokens=4_000,
        estimated_merge_risk_millionths=400_000,
    )

    first = refine_task_candidates(
        (broad,),
        policy=policy,
        cost_measurements=(measurement,),
        repository_tree=TREE,
        toolchain_features=FEATURES,
    )
    repeated = refine_task_candidates(
        (broad,),
        policy=policy,
        cost_measurements=(measurement,),
        repository_tree=TREE,
        toolchain_features=reversed(FEATURES),
    )

    assert len(first.accepted) == 4
    assert first.granularity_calibration_id
    assert [item.canonical_task_cid for item in first.accepted] == [
        item.canonical_task_cid for item in repeated.accepted
    ]
    assert len({item.semantic_identity for item in first.accepted}) == 4
    assert all(item.dependencies == broad.dependencies for item in first.accepted)
    assert all(item.merge_fate == broad.merge_fate for item in first.accepted)
    assert all(item.estimated_context_tokens <= 1_000 for item in first.accepted)
    assert all(item.estimated_validation_seconds <= 10 for item in first.accepted)
    assert all(item.estimated_proof_seconds <= 10 for item in first.accepted)
    assert all(item.estimated_tokens <= 1_000 for item in first.accepted)
    assert all(
        item.estimated_merge_risk_millionths <= 100_000
        for item in first.accepted
    )

    for field in (
        "acceptance",
        "effects",
        "evidence_subset",
        "context_paths",
        "predicted_paths",
        "predicted_symbols",
        "predicted_interfaces",
        "proof_obligations",
    ):
        assert {
            value for child in first.accepted for value in getattr(child, field)
        } == set(getattr(broad, field))


@pytest.mark.parametrize(
    "overrides",
    (
        {"acceptance_criteria": ["criterion-0", "criterion-1", "criterion-2"]},
        {"estimated_context_tokens": 2_500},
        {
            "predicted_interfaces": [
                "Interface0",
                "Interface1",
                "Interface2",
                "Interface3",
                "Interface4",
            ]
        },
        {"estimated_validation_seconds": 30},
        {
            "proof_obligations": ["proof-0", "proof-1", "proof-2"],
        },
        {"estimated_proof_seconds": 30},
        {"estimated_merge_risk_millionths": 250_000},
    ),
)
def test_each_measured_dimension_independently_forces_valid_split(
    overrides: dict[str, object],
):
    policy = TaskQualityPolicy(coalesce_tiny=False)
    calibration = calibrate_task_granularity(
        (_measurement(policy),),
        repository_tree=TREE,
        policy=policy,
        toolchain_features=FEATURES,
    )
    source = _candidate(**overrides)

    result = refine_task_candidates(
        (source,),
        policy=policy,
        calibration=calibration,
        repository_tree=TREE,
        toolchain_features=FEATURES,
    )

    assert len(result.accepted) >= 2
    assert not result.rejected
    assert len({task.semantic_identity for task in result.accepted}) == len(
        result.accepted
    )
    assert all(
        not is_over_broad(
            task,
            policy=policy,
            calibration=calibration,
        )
        for task in result.accepted
    )
    assert {
        criterion
        for task in result.accepted
        for criterion in task.acceptance
    } == set(source.acceptance)
    assert all(task.dependencies == source.dependencies for task in result.accepted)


def test_coalesce_requires_compatible_tiny_work_and_stays_inside_measured_bounds():
    policy = TaskQualityPolicy(tiny_max_paths=2)
    calibration = calibrate_task_granularity(
        (_measurement(policy, acceptance_count=4, proof_item_count=2),),
        repository_tree=TREE,
        policy=policy,
        toolchain_features=FEATURES,
    )
    left = _candidate(
        source_id="LEFT",
        title="Cover cache hit",
        acceptance_criteria=["cache-hit"],
        effects=["hit-covered"],
        evidence_subset=["hit-evidence"],
        predicted_symbols=["test_hit"],
        predicted_interfaces=["HitProtocol"],
        proof_obligations=["hit-proof"],
        estimated_tokens=300,
    )
    right = _candidate(
        source_id="RIGHT",
        title="Cover cache miss",
        acceptance_criteria=["cache-miss"],
        effects=["miss-covered"],
        evidence_subset=["miss-evidence"],
        predicted_symbols=["test_miss"],
        predicted_interfaces=["MissProtocol"],
        proof_obligations=["miss-proof"],
        estimated_tokens=300,
    )

    assert can_coalesce_tasks(
        left, right, policy=policy, calibration=calibration
    )
    merged = coalesce_task_candidates(
        (right, left), policy=policy, calibration=calibration
    )
    repeated = coalesce_task_candidates(
        (left, right), policy=policy, calibration=calibration
    )
    assert merged.semantic_identity == repeated.semantic_identity
    assert set(merged.acceptance) == {*left.acceptance, *right.acceptance}
    assert merged.estimated_tokens == 600

    incompatible = replace(
        right,
        proof_commands=("prove-with-another-tool",),
        semantic_identity="",
    )
    assert not can_coalesce_tasks(
        left, incompatible, policy=policy, calibration=calibration
    )
    too_expensive = replace(right, estimated_tokens=800, semantic_identity="")
    assert not can_coalesce_tasks(
        left, too_expensive, policy=policy, calibration=calibration
    )


def test_completion_propagates_only_after_exact_bound_descendants_complete():
    policy = TaskQualityPolicy(coalesce_tiny=False)
    broad = _candidate(
        source_id="BROAD",
        acceptance_criteria=["criterion-a", "criterion-b"],
        outputs=["src/a.py", "src/b.py"],
        predicted_paths=["src/a.py", "src/b.py"],
        predicted_symbols=["A.run", "B.run"],
        predicted_interfaces=["AProtocol", "BProtocol"],
        proof_obligations=["proof-a", "proof-b"],
        estimated_context_tokens=2_000,
        estimated_validation_seconds=20,
        estimated_proof_seconds=20,
        estimated_tokens=2_000,
        estimated_merge_risk_millionths=200_000,
    )
    result = refine_task_candidates(
        (broad,),
        policy=policy,
        cost_measurements=(
            _measurement(
                policy,
                acceptance_count=1,
                accepted_criteria=1,
            ),
        ),
        repository_tree=TREE,
        toolchain_features=FEATURES,
    )
    assert len(result.accepted) == 2

    partial = propagate_task_completion(
        result, (result.accepted[0].canonical_task_cid,)
    )
    assert partial.completed_source_identities == ()
    assert partial.incomplete_source_identities == (broad.semantic_identity,)
    assert len(partial.completed_acceptance) == 1

    complete = propagate_task_completion(
        result, tuple(item.canonical_task_key for item in result.accepted)
    )
    repeated = propagate_task_completion(
        result, reversed(tuple(item.semantic_identity for item in result.accepted))
    )
    assert complete.propagation_id == repeated.propagation_id
    assert complete.completed_source_identities == (broad.semantic_identity,)
    assert set(complete.completed_acceptance) == set(broad.acceptance)

    with pytest.raises(ValueError, match="unknown or unaccepted"):
        propagate_task_completion(result, (broad.merge_fate,))


def test_paired_fixture_proves_zero_duplicates_and_fewer_calls_per_criterion():
    left = _candidate(
        title="Cover cache hit",
        acceptance_criteria=["cache-hit"],
        effects=["hit-covered"],
        evidence_subset=["hit-evidence"],
        predicted_symbols=["test_hit"],
        predicted_interfaces=["HitProtocol"],
        proof_obligations=["hit-proof"],
        estimated_tokens=300,
    )
    right = _candidate(
        title="Cover cache miss",
        acceptance_criteria=["cache-miss"],
        effects=["miss-covered"],
        evidence_subset=["miss-evidence"],
        predicted_symbols=["test_miss"],
        predicted_interfaces=["MissProtocol"],
        proof_obligations=["miss-proof"],
        estimated_tokens=300,
    )
    merged = coalesce_task_candidates(
        (left, right), policy=TaskQualityPolicy(tiny_max_paths=2)
    )
    baseline = TaskGranularityRun(
        fixture_id="paired-cache",
        tasks=(left, right),
        completed_tasks=(left.canonical_task_cid, right.canonical_task_cid),
        model_calls=2,
    )
    candidate = TaskGranularityRun(
        fixture_id="paired-cache",
        tasks=(merged,),
        completed_tasks=(merged.canonical_task_cid,),
        model_calls=1,
    )

    comparison = compare_task_granularity_runs(baseline, candidate)
    assert comparison.qualifies
    assert candidate.duplicate_semantic_task_count == 0
    assert candidate.calls_per_accepted_criterion < (
        baseline.calls_per_accepted_criterion
    )

    duplicate_run = TaskGranularityRun(
        fixture_id="paired-cache",
        tasks=(merged, merged),
        completed_tasks=(merged.semantic_identity,),
        model_calls=1,
    )
    assert not compare_task_granularity_runs(
        baseline, duplicate_run
    ).zero_duplicate_semantic_tasks

    incomplete_run = TaskGranularityRun(
        fixture_id="paired-cache",
        tasks=(merged,),
        completed_tasks=(),
        model_calls=1,
    )
    assert incomplete_run.to_dict()["calls_per_accepted_criterion"] is None
    assert not compare_task_granularity_runs(
        baseline, incomplete_run
    ).completion_exact
