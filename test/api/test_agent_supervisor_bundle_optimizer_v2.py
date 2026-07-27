from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.bundle_optimizer import (
    BundleOptimizationPolicy,
    BundlePlanningChange,
    BundlePlanningChangeKind,
    optimize_task_bundles,
    rebundle_pending_work,
)


def _task(task_id: str, **overrides: object) -> dict[str, object]:
    suffix = task_id.casefold()
    payload: dict[str, object] = {
        "task_id": task_id,
        "canonical_task_cid": f"cid-{suffix}",
        "canonical_task_key": f"task/v2/{suffix}",
        "title": "Implement shared objective",
        "goal_id": "ASI-G260",
        "acceptance": [f"{task_id} accepted"],
        "effects": [f"{task_id} effect"],
        "outputs": [f"src/{suffix}.py"],
        "predicted_paths": [f"src/{suffix}.py"],
        "predicted_symbols": [f"{task_id}.run"],
        "context_paths": [f"context/{suffix}.md"],
        "immutable_context_keys": [f"context-cid/{suffix}"],
        "artifact_locality_keys": [f"artifact/{suffix}"],
        "validation_commands": [f"pytest test/{suffix}.py"],
        "resource_class": "cpu-medium",
        "provider_batch_key": "provider/batch/default",
        "estimated_context_tokens": 100,
        "estimated_tokens": 200,
        "estimated_validation_seconds": 5,
        "merge_fate": "merge/main",
        "merge_pressure": 1,
        "dependencies": [],
        "conflicts": [],
        "status": "pending",
    }
    payload.update(overrides)
    return payload


def _members(result: object) -> set[frozenset[str]]:
    return {frozenset(bundle.task_cids) for bundle in result.bundles}


def test_multifactor_plan_beats_title_and_goal_only_context_cost() -> None:
    tasks = [
        _task(
            "A",
            immutable_context_keys=["context-cid/alpha"],
            artifact_locality_keys=["artifact/alpha"],
            context_paths=["context/alpha.md"],
            validation_commands=["pytest test/alpha.py"],
        ),
        _task(
            "B",
            immutable_context_keys=["context-cid/beta"],
            artifact_locality_keys=["artifact/beta"],
            context_paths=["context/beta.md"],
            validation_commands=["pytest test/beta.py"],
        ),
        _task(
            "C",
            immutable_context_keys=["context-cid/alpha"],
            artifact_locality_keys=["artifact/alpha"],
            context_paths=["context/alpha.md"],
            validation_commands=["pytest test/alpha.py"],
        ),
        _task(
            "D",
            immutable_context_keys=["context-cid/beta"],
            artifact_locality_keys=["artifact/beta"],
            context_paths=["context/beta.md"],
            validation_commands=["pytest test/beta.py"],
        ),
    ]

    plan = optimize_task_bundles(
        tuple(reversed(tasks)),
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=2),
    )

    assert _members(plan) == {
        frozenset(("cid-a", "cid-c")),
        frozenset(("cid-b", "cid-d")),
    }
    assert plan.metrics["context_model_cost"] == 200
    assert plan.metrics["artifact_reuse_millionths"] == 500_000
    for baseline in ("title_only", "goal_only"):
        assert (
            plan.metrics["context_model_cost"]
            < plan.heuristic_metrics[baseline]["context_model_cost"]
        )
        assert plan.regression_guards[
            f"no_conflict_rate_regression_vs_{baseline}"
        ]
        assert plan.regression_guards[
            f"no_model_call_regression_vs_{baseline}"
        ]


def test_context_cost_reuse_requires_an_immutable_content_identity() -> None:
    mutable = (
        _task(
            "MUTABLE-A",
            context_paths=["context/shared.md"],
            immutable_context_keys=[],
        ),
        _task(
            "MUTABLE-B",
            context_paths=["context/shared.md"],
            immutable_context_keys=[],
        ),
    )
    immutable = tuple(
        {
            **task,
            "immutable_context_keys": ["context-cid/shared"],
        }
        for task in mutable
    )
    policy = BundleOptimizationPolicy(max_tasks_per_bundle=2)

    mutable_plan = optimize_task_bundles(mutable, policy=policy)
    immutable_plan = optimize_task_bundles(immutable, policy=policy)

    assert mutable_plan.metrics["context_model_cost"] == 200
    assert immutable_plan.metrics["context_model_cost"] == 100
    assert immutable_plan.bundles[0].shared_immutable_context_keys == (
        "context-cid/shared",
    )


def test_existing_prerequisite_wave_does_not_gain_redundant_conflict_depth() -> None:
    root = _task(
        "ROOT",
        outputs=["src/shared.py"],
        predicted_paths=["src/shared.py"],
    )
    child = _task(
        "CHILD",
        outputs=["src/shared.py"],
        predicted_paths=["src/shared.py"],
        dependencies=["cid-root"],
    )
    independent = _task("INDEPENDENT")

    plan = optimize_task_bundles(
        (child, independent, root),
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=1),
    )
    wave = {
        cid: bundle.execution_wave
        for bundle in plan.bundles
        for cid in bundle.task_cids
    }

    assert wave["cid-root"] == 0
    assert wave["cid-child"] == 1
    assert wave["cid-independent"] == 0
    assert plan.metrics["critical_path_wave_count"] == 2
    assert plan.metrics["blocking_conflict_count"] == 1
    assert plan.metrics["merge_conflict_rate_millionths"] == 0


def test_path_symbol_and_interface_conflicts_are_all_serialized() -> None:
    tasks = (
        _task(
            "PATH-A",
            outputs=["src/path.py"],
            predicted_paths=["src/path.py"],
        ),
        _task(
            "PATH-B",
            outputs=["src/path.py"],
            predicted_paths=["src/path.py"],
        ),
        _task(
            "SYMBOL-A",
            global_ast_symbols=["Registry.publish"],
        ),
        _task(
            "SYMBOL-B",
            global_ast_symbols=["Registry.publish"],
        ),
        _task(
            "INTERFACE-A",
            interfaces=["RuntimeAPI@2"],
        ),
        _task(
            "INTERFACE-B",
            required_interfaces=["RuntimeAPI@2"],
        ),
    )

    plan = optimize_task_bundles(
        tasks,
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=1),
    )
    overlaps = {
        key
        for edge in plan.conflict_graph["edges"]
        if edge["blocks_concurrency"]
        for key in edge["overlaps"]
    }
    wave = {
        cid: bundle.execution_wave
        for bundle in plan.bundles
        for cid in bundle.task_cids
    }

    assert {"files", "ast_symbols", "interfaces"}.issubset(overlaps)
    assert wave["cid-path-a"] != wave["cid-path-b"]
    assert wave["cid-symbol-a"] != wave["cid-symbol-b"]
    assert wave["cid-interface-a"] != wave["cid-interface-b"]
    assert plan.metrics["merge_conflict_rate_millionths"] == 0


def test_merge_pressure_is_bounded_and_reduced_only_with_locality() -> None:
    first = _task("FIRST", merge_pressure=3)
    second = _task(
        "SECOND",
        merge_pressure=3,
        immutable_context_keys=["context-cid/first"],
        artifact_locality_keys=["artifact/first"],
        context_paths=["context/first.md"],
        validation_commands=["pytest test/first.py"],
    )

    separated = optimize_task_bundles(
        (first, second),
        policy=BundleOptimizationPolicy(
            max_tasks_per_bundle=2,
            max_merge_pressure_per_bundle=5,
        ),
    )
    grouped = optimize_task_bundles(
        (second, first),
        policy=BundleOptimizationPolicy(
            max_tasks_per_bundle=2,
            max_merge_pressure_per_bundle=6,
        ),
    )

    assert len(separated.bundles) == 2
    assert len(grouped.bundles) == 1
    assert grouped.bundles[0].merge_pressure == 6
    assert grouped.metrics["merge_pressure_reduction"] == 3


def test_typed_rebundle_changes_pending_only_and_is_deterministic() -> None:
    initial_tasks = [_task("ACTIVE"), _task("PENDING-A"), _task("PENDING-B")]
    previous = optimize_task_bundles(
        initial_tasks,
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=1),
    )
    active_bundle = next(
        bundle for bundle in previous.bundles if "cid-active" in bundle.task_cids
    )
    changed_tasks = [
        {
            **initial_tasks[0],
            "status": "active",
        },
        {
            **initial_tasks[1],
            "immutable_context_keys": ["context-cid/pending"],
            "artifact_locality_keys": ["artifact/pending"],
            "context_paths": ["context/pending.md"],
            "validation_commands": ["pytest test/pending.py"],
        },
        {
            **initial_tasks[2],
            "immutable_context_keys": ["context-cid/pending"],
            "artifact_locality_keys": ["artifact/pending"],
            "context_paths": ["context/pending.md"],
            "validation_commands": ["pytest test/pending.py"],
        },
    ]
    changes = (
        BundlePlanningChange(
            BundlePlanningChangeKind.CONTEXT_CHANGED,
            ("cid-pending-a", "cid-pending-b"),
            "revision-2",
        ),
        BundlePlanningChange(
            BundlePlanningChangeKind.ARTIFACT_CHANGED,
            ("cid-pending-b", "cid-pending-a"),
            "revision-2",
        ),
    )

    first = rebundle_pending_work(
        changed_tasks,
        previous_plan=previous,
        changes=changes,
        active_task_cids=("cid-active",),
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=2),
    )
    second = rebundle_pending_work(
        tuple(reversed(changed_tasks)),
        previous_plan=previous,
        changes=tuple(reversed(changes)),
        active_task_cids=("cid-active",),
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=2),
    )

    assert first.active_bundles == (active_bundle,)
    assert first.active_bundles[0].bundle_cid == active_bundle.bundle_cid
    assert _members(first.plan) == {
        frozenset(("cid-active",)),
        frozenset(("cid-pending-a", "cid-pending-b")),
    }
    assert first.to_dict() == second.to_dict()
    covered = [
        cid for bundle in first.plan.bundles for cid in bundle.task_cids
    ]
    assert len(covered) == len(set(covered)) == 3


def test_typed_rebundle_rejects_active_mutation_and_untyped_population_change() -> None:
    tasks = [_task("ACTIVE"), _task("PENDING")]
    previous = optimize_task_bundles(
        tasks,
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=1),
    )
    active_changed = [
        {**tasks[0], "context_paths": ["context/mutated.md"]},
        tasks[1],
    ]
    change = BundlePlanningChange(
        BundlePlanningChangeKind.CONTEXT_CHANGED,
        ("cid-active",),
        "revision-2",
    )

    with pytest.raises(ValueError, match="cannot mutate active"):
        rebundle_pending_work(
            active_changed,
            previous_plan=previous,
            changes=(change,),
            active_task_cids=("cid-active",),
        )
    with pytest.raises(ValueError, match="admission/removal"):
        rebundle_pending_work(
            (*tasks, _task("NEW")),
            previous_plan=previous,
            changes=(
                BundlePlanningChange(
                    BundlePlanningChangeKind.CONTEXT_CHANGED,
                    ("cid-pending",),
                    "revision-2",
                ),
            ),
        )


def test_plan_serialization_is_canonical_and_json_safe() -> None:
    tasks = (
        _task("ONE", immutable_context_keys=["context-cid/shared"]),
        _task("TWO", immutable_context_keys=["context-cid/shared"]),
    )
    policy = BundleOptimizationPolicy(max_tasks_per_bundle=2)
    first = optimize_task_bundles(tasks, policy=policy)
    second = optimize_task_bundles(tuple(reversed(tasks)), policy=policy)

    assert first.to_dict() == second.to_dict()
    assert json.loads(json.dumps(first.to_dict())) == first.to_dict()
