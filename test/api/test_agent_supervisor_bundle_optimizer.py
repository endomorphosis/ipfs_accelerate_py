from __future__ import annotations

import copy

import pytest

from ipfs_accelerate_py.agent_supervisor.bundle_optimizer import (
    PACKET_COMPLETION_BINDING_REQUIREMENT_ID,
    BundleOptimizationPolicy,
    PacketCompletionBindingEvidence,
    optimize_task_bundles,
    propagate_goal_packet_completion,
)
from ipfs_accelerate_py.agent_supervisor.todo_vector_index import (
    TodoIndexRecord,
    build_execution_packet,
)


def _task(task_id: str, **overrides: object) -> dict[str, object]:
    suffix = task_id.casefold().replace("_", "-")
    payload: dict[str, object] = {
        "task_id": task_id,
        "canonical_task_cid": f"cid-{suffix}",
        "canonical_task_key": f"task/v1/{suffix}",
        "title": f"Implement {task_id}",
        "goal_id": "ASI-G050",
        "acceptance": [f"{task_id} is implemented"],
        "effects": [f"{task_id} behavior is available"],
        "evidence_subset": [f"evidence-{suffix}"],
        "outputs": [f"src/{suffix}.py"],
        "predicted_paths": [f"src/{suffix}.py"],
        "predicted_symbols": [f"{task_id}.run"],
        "context_paths": ["src/shared.py"],
        "validation_commands": ["python -m pytest test/shared.py -q"],
        "resource_class": "cpu-medium",
        "token_class": "medium",
        "estimated_context_tokens": 100,
        "estimated_tokens": 500,
        "estimated_validation_seconds": 10,
        "merge_fate": "objective/ASI-G050",
        "dependencies": [],
        "conflicts": [],
    }
    payload.update(overrides)
    return payload


def _bundled_task_cids(result: object) -> set[str]:
    return {
        task_cid
        for bundle in result.bundles
        for task_cid in bundle.task_cids
    }


def test_optimizer_requires_admission_to_supply_canonical_identity():
    with pytest.raises(ValueError, match="requires canonical_task_key"):
        optimize_task_bundles(
            [
                {
                    "task_id": "RAW-001",
                    "title": "Unadmitted work",
                    "outputs": ["src/raw.py"],
                }
            ]
        )


def test_optimizer_groups_shared_context_and_reuses_validation_without_losing_identity():
    first = _task("ASI-051-A")
    second = _task(
        "ASI-051-B",
        outputs=["src/second.py"],
        predicted_paths=["src/second.py"],
        predicted_symbols=["Second.run"],
    )
    unrelated = _task(
        "ASI-051-C",
        outputs=["tools/unrelated.py"],
        predicted_paths=["tools/unrelated.py"],
        predicted_symbols=["Unrelated.run"],
        context_paths=["tools/context.py"],
        validation_commands=["python -m pytest test/unrelated.py -q"],
        merge_fate="objective/OTHER",
    )

    result = optimize_task_bundles(
        (unrelated, second, first),
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=2),
    )

    assert _bundled_task_cids(result) == {
        first["canonical_task_cid"],
        second["canonical_task_cid"],
        unrelated["canonical_task_cid"],
    }
    shared_bundle = next(
        bundle
        for bundle in result.bundles
        if first["canonical_task_cid"] in bundle.task_cids
    )
    assert set(shared_bundle.task_cids) == {
        first["canonical_task_cid"],
        second["canonical_task_cid"],
    }
    assert shared_bundle.validation_commands == (
        "python -m pytest test/shared.py -q",
    )
    assert set(shared_bundle.canonical_task_keys) == {
        first["canonical_task_key"],
        second["canonical_task_key"],
    }

    projection = result.to_dict()
    projected = {
        task_cid
        for bundle in projection["bundles"]
        for task_cid in bundle["task_cids"]
    }
    assert projected == _bundled_task_cids(result)


def test_optimizer_is_deterministic_across_input_order_and_display_id_aliases():
    first = _task("DISPLAY-A")
    second = _task(
        "DISPLAY-B",
        outputs=["src/b.py"],
        predicted_paths=["src/b.py"],
        predicted_symbols=["B.run"],
    )
    policy = BundleOptimizationPolicy(max_tasks_per_bundle=2)

    initial = optimize_task_bundles((first, second), policy=policy)
    reordered = optimize_task_bundles((second, first), policy=policy)
    renamed = optimize_task_bundles(
        (
            {**first, "task_id": "RENAMED-A"},
            {**second, "task_id": "RENAMED-B"},
        ),
        policy=policy,
    )

    assert [bundle.bundle_cid for bundle in initial.bundles] == [
        bundle.bundle_cid for bundle in reordered.bundles
    ]
    assert [bundle.bundle_cid for bundle in initial.bundles] == [
        bundle.bundle_cid for bundle in renamed.bundles
    ]
    assert [bundle.task_cids for bundle in initial.bundles] == [
        bundle.task_cids for bundle in renamed.bundles
    ]


def test_dependency_waves_preserve_independent_critical_path_width():
    root_a = _task(
        "ROOT-A",
        context_paths=["src/a.py"],
        validation_commands=["test-a"],
        merge_fate="lane-a",
    )
    root_b = _task(
        "ROOT-B",
        context_paths=["src/b.py"],
        validation_commands=["test-b"],
        merge_fate="lane-b",
    )
    child_a = _task(
        "CHILD-A",
        context_paths=["src/a-child.py"],
        validation_commands=["test-a-child"],
        merge_fate="lane-a-child",
        dependencies=[root_a["canonical_task_cid"]],
    )
    child_b = _task(
        "CHILD-B",
        context_paths=["src/b-child.py"],
        validation_commands=["test-b-child"],
        merge_fate="lane-b-child",
        dependencies=[root_b["canonical_task_cid"]],
    )

    result = optimize_task_bundles(
        (child_b, root_a, child_a, root_b),
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=1),
    )
    wave_by_task = {
        bundle.task_cids[0]: bundle.execution_wave for bundle in result.bundles
    }

    assert wave_by_task[root_a["canonical_task_cid"]] == 0
    assert wave_by_task[root_b["canonical_task_cid"]] == 0
    assert wave_by_task[child_a["canonical_task_cid"]] == 1
    assert wave_by_task[child_b["canonical_task_cid"]] == 1
    assert sum(wave == 0 for wave in wave_by_task.values()) == 2
    assert sum(wave == 1 for wave in wave_by_task.values()) == 2


def test_conflicting_tasks_are_serialized_even_when_context_reuse_is_high():
    first = _task(
        "WRITER-A",
        outputs=["src/shared.py"],
        predicted_paths=["src/shared.py"],
        conflicts=["shared-writer"],
    )
    second = _task(
        "WRITER-B",
        outputs=["src/shared.py"],
        predicted_paths=["src/shared.py"],
        conflicts=["shared-writer"],
    )

    result = optimize_task_bundles(
        (first, second),
        policy=BundleOptimizationPolicy(
            max_tasks_per_bundle=2,
            allow_internal_conflicts=False,
        ),
    )
    containing = {
        task_cid: bundle
        for bundle in result.bundles
        for task_cid in bundle.task_cids
    }

    assert containing[first["canonical_task_cid"]].bundle_cid != containing[
        second["canonical_task_cid"]
    ].bundle_cid
    assert containing[first["canonical_task_cid"]].execution_wave != containing[
        second["canonical_task_cid"]
    ].execution_wave
    assert (
        containing[first["canonical_task_cid"]].conflict_weight > 0
        or containing[second["canonical_task_cid"]].conflict_weight > 0
    )


def test_conflict_serialization_never_reverses_a_dependency_edge():
    parent = _task(
        "PARENT",
        canonical_task_cid="cid-z-parent",
        canonical_task_key="task/v1/z-parent",
        outputs=["src/shared.py"],
        predicted_paths=["src/shared.py"],
        conflicts=["shared-writer"],
    )
    child = _task(
        "CHILD",
        canonical_task_cid="cid-a-child",
        canonical_task_key="task/v1/a-child",
        outputs=["src/shared.py"],
        predicted_paths=["src/shared.py"],
        dependencies=["cid-z-parent"],
        conflicts=["shared-writer"],
    )

    result = optimize_task_bundles(
        (child, parent),
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=1),
    )
    wave_by_task = {
        task_cid: bundle.execution_wave
        for bundle in result.bundles
        for task_cid in bundle.task_cids
    }

    assert wave_by_task["cid-z-parent"] == 0
    assert wave_by_task["cid-a-child"] > wave_by_task["cid-z-parent"]


def test_packet_completion_propagates_only_to_explicit_canonical_bindings():
    aggregate = _task(
        "PACKET",
        goal_packet_key="goal_packet/task-quality/shared",
        goal_packet_role="packet_aggregate",
        merge_family="goal_packet/task-quality/shared",
        completion_task_bindings=["cid-bound"],
    )
    bound = _task(
        "BOUND",
        canonical_task_cid="cid-bound",
        canonical_task_key="task/v1/bound",
        goal_packet_key="goal_packet/task-quality/shared",
        goal_packet_role="packet_member",
        merge_family="goal_packet/task-quality/shared",
    )
    unbound = _task(
        "UNBOUND",
        canonical_task_cid="cid-unbound",
        canonical_task_key="task/v1/unbound",
        goal_packet_key="goal_packet/task-quality/shared",
        goal_packet_role="packet_member",
        merge_family="goal_packet/task-quality/shared",
    )
    same_family_only = _task(
        "SAME-FAMILY",
        canonical_task_cid="cid-same-family",
        canonical_task_key="task/v1/same-family",
        goal_packet_key="",
        merge_family="goal_packet/task-quality/shared",
    )
    tasks = (aggregate, bound, unbound, same_family_only)

    completion = propagate_goal_packet_completion(
        tasks,
        completed_task_cids=(aggregate["canonical_task_cid"],),
    )

    assert set(completion.completed_task_cids) == {
        aggregate["canonical_task_cid"],
        bound["canonical_task_cid"],
    }
    assert completion.propagated_task_cids == (bound["canonical_task_cid"],)
    assert unbound["canonical_task_cid"] not in completion.completed_task_cids
    assert same_family_only["canonical_task_cid"] not in completion.completed_task_cids
    assert completion.evidence.proved_requirement_ids == (
        PACKET_COMPLETION_BINDING_REQUIREMENT_ID,
    )


def test_packet_completion_evidence_fails_closed_for_unbound_or_tampered_projection():
    aggregate = _task(
        "PACKET",
        goal_packet_key="goal_packet/task-quality/shared",
        goal_packet_role="packet_aggregate",
        completion_task_bindings=["cid-bound"],
    )
    bound = _task(
        "BOUND",
        canonical_task_cid="cid-bound",
        canonical_task_key="task/v1/bound",
        goal_packet_key="goal_packet/task-quality/shared",
        goal_packet_role="packet_member",
    )
    unbound = _task(
        "UNBOUND",
        canonical_task_cid="cid-unbound",
        canonical_task_key="task/v1/unbound",
        goal_packet_key="goal_packet/task-quality/shared",
        goal_packet_role="packet_member",
    )
    tasks = (aggregate, bound, unbound)

    valid = PacketCompletionBindingEvidence.create(
        tasks,
        completed_task_cids=(aggregate["canonical_task_cid"],),
        propagated_task_cids=(bound["canonical_task_cid"],),
    )
    invalid = PacketCompletionBindingEvidence.create(
        tasks,
        completed_task_cids=(aggregate["canonical_task_cid"],),
        propagated_task_cids=(
            bound["canonical_task_cid"],
            unbound["canonical_task_cid"],
        ),
    )

    assert valid.verify_integrity()
    assert valid.proved_requirement_ids == (
        PACKET_COMPLETION_BINDING_REQUIREMENT_ID,
    )
    assert invalid.verify_integrity()
    assert invalid.proved_requirement_ids == ()

    projection = copy.deepcopy(valid.to_dict())
    restored = PacketCompletionBindingEvidence.from_dict(projection)
    assert restored == valid
    assert restored.proved_requirement_ids == ()
    projection["propagated_task_cids"].append(unbound["canonical_task_cid"])
    with pytest.raises(ValueError):
        PacketCompletionBindingEvidence.from_dict(projection)
    key_tamper = copy.deepcopy(valid.to_dict())
    key_tamper["task_population"][0]["canonical_task_key"] = "task/v1/forged"
    with pytest.raises(ValueError):
        PacketCompletionBindingEvidence.from_dict(key_tamper)


def test_vector_execution_packet_projects_exact_canonical_completion_binding():
    common = {
        "status": "todo",
        "priority": "P1",
        "track": "task-generation",
        "source_line": 1,
        "bundle_key": "objective/task-generation",
        "goal_packet_key": "goal_packet/task-generation/shared",
        "merge_family": "goal_packet/task-generation/shared",
    }
    aggregate = TodoIndexRecord(
        task_id="PACKET",
        title="Packet aggregate",
        canonical_task_key="task/v1/packet",
        task_cid="cid-packet",
        semantic_identity="semantic-packet",
        goal_packet_role="packet_aggregate",
        candidate_kind="goal_packet_aggregate",
        completion_task_bindings=["semantic-bound"],
        **common,
    )
    bound = TodoIndexRecord(
        task_id="BOUND",
        title="Bound member",
        canonical_task_key="task/v1/bound",
        task_cid="cid-bound",
        semantic_identity="semantic-bound",
        goal_packet_role="packet_member",
        **common,
    )
    unbound = TodoIndexRecord(
        task_id="UNBOUND",
        title="Unbound member",
        canonical_task_key="task/v1/unbound",
        task_cid="cid-unbound",
        semantic_identity="semantic-unbound",
        goal_packet_role="packet_member",
        **common,
    )

    packet = build_execution_packet(
        context={
            "context_key": "context/shared",
            "source_type": "goal_packet_key",
            "source_key": common["goal_packet_key"],
        },
        records=(unbound, bound, aggregate),
    )

    assert packet is not None
    assert set(packet["active_task_cids"]) == {
        "cid-packet",
        "cid-bound",
        "cid-unbound",
    }
    assert packet["primary_task_cid"] == "cid-packet"
    assert packet["completion_binding"]["bound_sibling_task_cids"] == [
        "cid-bound"
    ]
    assert packet["completion_binding"]["bound_sibling_task_ids"] == ["BOUND"]
    assert packet["completion_binding"]["canonical_task_keys"] == {
        "cid-packet": "task/v1/packet",
        "cid-bound": "task/v1/bound",
    }
    assert "cid-unbound" not in packet["completion_binding"][
        "bound_sibling_task_cids"
    ]
