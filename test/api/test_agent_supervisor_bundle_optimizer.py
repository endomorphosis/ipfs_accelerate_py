from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.objectives.bundle_optimizer import (
    CRITICAL_PATH_WIDTH_REQUIREMENT_ID,
    PACKET_COMPLETION_BINDING_REQUIREMENT_ID,
    BundleOptimizationPolicy,
    CriticalPathWidthEvidence,
    PacketCompletionBindingEvidence,
    optimize_task_bundles,
    prove_critical_path_width,
    propagate_goal_packet_completion,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    EvidenceSourcePolicy,
    TASK_GENERATION_ACCEPTANCE_CRITERIA,
    TASK_GENERATION_CHILD_GOAL_IDS,
    TASK_GENERATION_COMPLETION_ANALYZER_VERSION,
    TASK_GENERATION_COMPLETION_CONFIGURATION_REVISION,
    TASK_GENERATION_OBJECTIVE_ID,
    TASK_GENERATION_OBJECTIVE_REVISION,
    TASK_GENERATION_PRODUCING_TASK_IDS,
    TASK_GENERATION_REQUIRED_EXHAUSTIVE_RECEIPTS,
    evaluate_task_generation_completion,
    task_generation_evidence_producer_bindings,
)
from ipfs_accelerate_py.agent_supervisor.objectives.goal_completion import (
    CompletionEvidence,
    GoalState,
)
from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import (
    ConflictWaveProjection,
    build_conflict_surface,
    build_task_work_contract,
    project_conflict_free_wave,
    rehydrate_task_work_contract_projection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.todo_vector_index import (
    TodoIndexRecord,
    build_execution_packet,
)
from ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor import (
    optimize_bundle_payloads,
    plan_bundle_lanes,
)
from ipfs_accelerate_py.agent_supervisor.merge.lease_coordination import (
    adapt_goal_bundle,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_identity import (
    canonical_bundle_identity,
)
from ipfs_accelerate_py.agent_supervisor.planning.task_quality import TaskCandidate


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


def test_objective_graph_depth_never_manufactures_dependency_waves():
    first = _task("HIERARCHY-A", graph_depth=9)
    second = _task(
        "HIERARCHY-B",
        graph_depth=9,
        outputs=["src/hierarchy-b.py"],
        predicted_paths=["src/hierarchy-b.py"],
        predicted_symbols=["HierarchyB.run"],
    )

    result = optimize_task_bundles(
        (first, second),
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=1),
    )

    assert {bundle.execution_wave for bundle in result.bundles} == {0}
    assert result.metrics["critical_path_wave_count"] == 1


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
        "acceptance_criteria": ["packet projection remains coherent"],
        "effects": ["packet work is executed"],
        "predicted_files": ["src/packet_projection.py"],
        "predicted_symbols": ["PacketProjection.run"],
        "estimated_context_tokens": 100,
        "estimated_tokens": 300,
        "estimated_validation_seconds": 5,
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
    assert len(packet["task_work_contracts"]) == 3
    assert len(set(packet["task_work_contract_ids"])) == 3
    assert {
        contract["canonical_task_cid"]
        for contract in packet["task_work_contracts"]
    } == {"cid-packet", "cid-bound", "cid-unbound"}
    assert all(
        contract["work_contract"]["acceptance_effect_subset"] == {
            "acceptance": ["packet projection remains coherent"],
            "effects": ["packet work is executed"],
            "evidence_subset": [],
        }
        for contract in packet["task_work_contracts"]
    )
    # The aggregate explicitly covers BOUND, so packet execution costs count
    # only the aggregate and independent UNBOUND work.
    assert packet["estimated_costs"] == {
        "estimated_context_tokens": 200,
        "estimated_tokens": 600,
        "estimated_validation_seconds": 10,
    }


def test_optimizer_uses_evidence_provider_and_resource_compatibility():
    shared_evidence = _task(
        "EVIDENCE-A",
        context_paths=[],
        validation_commands=[],
        evidence_subset=["proof/shared"],
        provider_id="provider-a",
        provider_route="chat",
        model_id="model-a",
    )
    compatible = _task(
        "EVIDENCE-B",
        outputs=["src/evidence-b.py"],
        predicted_paths=["src/evidence-b.py"],
        predicted_symbols=["EvidenceB.run"],
        context_paths=[],
        validation_commands=[],
        evidence_subset=["proof/shared"],
        provider_id="provider-a",
        provider_route="chat",
        model_id="model-a",
    )
    other_provider = _task(
        "EVIDENCE-C",
        outputs=["src/evidence-c.py"],
        predicted_paths=["src/evidence-c.py"],
        predicted_symbols=["EvidenceC.run"],
        context_paths=[],
        validation_commands=[],
        evidence_subset=["proof/shared"],
        provider_id="provider-b",
        provider_route="chat",
        model_id="model-a",
    )
    other_resource = _task(
        "EVIDENCE-D",
        outputs=["src/evidence-d.py"],
        predicted_paths=["src/evidence-d.py"],
        predicted_symbols=["EvidenceD.run"],
        context_paths=[],
        validation_commands=[],
        evidence_subset=["proof/shared"],
        provider_id="provider-a",
        provider_route="chat",
        model_id="model-a",
        resource_class="gpu-large",
    )

    result = optimize_task_bundles(
        (other_resource, compatible, other_provider, shared_evidence),
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=4),
    )
    containing = {
        cid: bundle for bundle in result.bundles for cid in bundle.task_cids
    }

    shared_bundle = containing[shared_evidence["canonical_task_cid"]]
    assert set(shared_bundle.task_cids) == {
        shared_evidence["canonical_task_cid"],
        compatible["canonical_task_cid"],
    }
    assert shared_bundle.shared_evidence_keys == ("proof/shared",)
    assert containing[other_provider["canonical_task_cid"]] != shared_bundle
    assert containing[other_resource["canonical_task_cid"]] != shared_bundle
    assert len(shared_bundle.provider_batch_keys) == 1


def test_conflict_coloring_preserves_independent_width_for_path_graph():
    left = _task(
        "PATH-A",
        context_paths=[],
        validation_commands=[],
        conflicts=["edge-ab"],
    )
    middle = _task(
        "PATH-B",
        outputs=["src/path-b.py"],
        predicted_paths=["src/path-b.py"],
        predicted_symbols=["PathB.run"],
        context_paths=[],
        validation_commands=[],
        conflicts=["edge-ab", "edge-bc"],
    )
    right = _task(
        "PATH-C",
        outputs=["src/path-c.py"],
        predicted_paths=["src/path-c.py"],
        predicted_symbols=["PathC.run"],
        context_paths=[],
        validation_commands=[],
        conflicts=["edge-bc"],
    )

    result = optimize_task_bundles(
        (right, left, middle),
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=1),
    )
    wave_by_cid = {
        bundle.task_cids[0]: bundle.execution_wave for bundle in result.bundles
    }

    assert result.metrics["critical_path_wave_count"] == 2
    assert wave_by_cid[left["canonical_task_cid"]] == wave_by_cid[
        right["canonical_task_cid"]
    ]
    assert wave_by_cid[middle["canonical_task_cid"]] != wave_by_cid[
        left["canonical_task_cid"]
    ]
    assert max(result.execution_width_by_wave.values()) == 2
    assert result.metrics["merge_conflict_rate_millionths"] == 0
    assert (
        result.comparison.current_metrics["merge_conflict_rate_millionths"]
        == 1_000_000
    )


def test_critical_path_width_evidence_proves_independent_path_endpoints():
    left = _task(
        "WIDTH-A",
        context_paths=[],
        validation_commands=[],
        conflicts=["edge-ab"],
    )
    middle = _task(
        "WIDTH-B",
        outputs=["src/width-b.py"],
        predicted_paths=["src/width-b.py"],
        predicted_symbols=["WidthB.run"],
        context_paths=[],
        validation_commands=[],
        conflicts=["edge-ab", "edge-bc"],
    )
    right = _task(
        "WIDTH-C",
        outputs=["src/width-c.py"],
        predicted_paths=["src/width-c.py"],
        predicted_symbols=["WidthC.run"],
        context_paths=[],
        validation_commands=[],
        conflicts=["edge-bc"],
    )
    policy = BundleOptimizationPolicy(max_tasks_per_bundle=1)

    evidence = prove_critical_path_width(
        (right, left, middle),
        policy=policy,
        repository_tree="git-tree-asi-034",
    )
    repeated = prove_critical_path_width(
        (middle, right, left),
        policy=policy,
        repository_tree="git-tree-asi-034",
    )

    assert evidence.verify_integrity()
    assert evidence.evidence_id == repeated.evidence_id
    assert evidence.proved_requirement_ids == (
        CRITICAL_PATH_WIDTH_REQUIREMENT_ID,
    )
    assert evidence.independent_width_by_dependency_wave == {"0": 2}
    assert evidence.effective_task_waves[left["canonical_task_cid"]] == (
        evidence.effective_task_waves[right["canonical_task_cid"]]
    )
    assert evidence.effective_task_waves[middle["canonical_task_cid"]] != (
        evidence.effective_task_waves[left["canonical_task_cid"]]
    )
    assert all(
        evidence.effective_task_waves[left_cid]
        != evidence.effective_task_waves[right_cid]
        for left_cid, right_cid in evidence.blocking_conflict_pairs
    )

    decision = EvidenceSourcePolicy().validate_completion_evidence(
        CRITICAL_PATH_WIDTH_REQUIREMENT_ID,
        evidence,
        repository_tree="git-tree-asi-034",
        policy_id=policy.policy_id,
    )
    assert decision.satisfies is True


def test_critical_path_width_evidence_fails_closed_without_width_or_authority():
    first = _task("SERIAL-A", conflicts=["only-edge"])
    second = _task(
        "SERIAL-B",
        outputs=["src/serial-b.py"],
        predicted_paths=["src/serial-b.py"],
        predicted_symbols=["SerialB.run"],
        conflicts=["only-edge"],
    )
    no_independent_width = CriticalPathWidthEvidence.create(
        (first, second),
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=1),
        repository_tree="git-tree-asi-034",
    )
    independent = CriticalPathWidthEvidence.create(
        (
            _task("PARALLEL-A"),
            _task(
                "PARALLEL-B",
                outputs=["src/parallel-b.py"],
                predicted_paths=["src/parallel-b.py"],
                predicted_symbols=["ParallelB.run"],
            ),
        ),
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=1),
        repository_tree="git-tree-asi-034",
    )

    assert no_independent_width.verify_integrity()
    assert no_independent_width.proved_requirement_ids == ()
    assert independent.proved_requirement_ids
    unbound = CriticalPathWidthEvidence.create(
        (
            _task("UNBOUND-A"),
            _task(
                "UNBOUND-B",
                outputs=["src/unbound-b.py"],
                predicted_paths=["src/unbound-b.py"],
                predicted_symbols=["UnboundB.run"],
            ),
        ),
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=1),
    )
    assert unbound.verify_integrity()
    assert unbound.proved_requirement_ids == ()
    restored = CriticalPathWidthEvidence.from_dict(independent.to_dict())
    assert restored.verify_integrity()
    assert restored.proved_requirement_ids == ()
    restored_decision = EvidenceSourcePolicy().validate_completion_evidence(
        CRITICAL_PATH_WIDTH_REQUIREMENT_ID,
        restored,
        repository_tree="git-tree-asi-034",
        policy_id=BundleOptimizationPolicy(max_tasks_per_bundle=1).policy_id,
    )
    assert restored_decision.satisfies is False
    assert "receipt_producer_authority_missing" in restored_decision.reason_codes

    forged = {
        "schema": "caller-width-lookalike@1",
        "evidence_id": "fake-width",
        "requirement_id": CRITICAL_PATH_WIDTH_REQUIREMENT_ID,
        "repository_tree": "git-tree-asi-034",
        "policy_id": BundleOptimizationPolicy(max_tasks_per_bundle=1).policy_id,
        "source_tier": "validation",
        "status": "passed",
        "complete": True,
        "coverage_complete": True,
    }
    forged_decision = EvidenceSourcePolicy().validate_completion_evidence(
        CRITICAL_PATH_WIDTH_REQUIREMENT_ID,
        forged,
        repository_tree="git-tree-asi-034",
        policy_id=forged["policy_id"],
    )
    assert forged_decision.satisfies is False
    assert "receipt_producer_authority_missing" in forged_decision.reason_codes

    tampered = copy.deepcopy(independent.to_dict())
    cid = tampered["task_population"][0]["canonical_task_cid"]
    tampered["planned_task_waves"][cid] = 99
    with pytest.raises(ValueError, match="digest mismatch"):
        CriticalPathWidthEvidence.from_dict(tampered)


def test_critical_path_width_proof_binds_closed_dag_and_exact_conflict_edges():
    left = _task("DAG-WIDTH-A", conflicts=["left-middle"])
    middle = _task(
        "DAG-WIDTH-B",
        outputs=["src/dag-width-b.py"],
        predicted_paths=["src/dag-width-b.py"],
        predicted_symbols=["DagWidthB.run"],
        conflicts=["left-middle", "middle-right"],
    )
    right = _task(
        "DAG-WIDTH-C",
        outputs=["src/dag-width-c.py"],
        predicted_paths=["src/dag-width-c.py"],
        predicted_symbols=["DagWidthC.run"],
        conflicts=["middle-right"],
    )
    left_child = _task(
        "DAG-WIDTH-D",
        dependencies=[left["canonical_task_cid"]],
        outputs=["src/dag-width-d.py"],
        predicted_paths=["src/dag-width-d.py"],
        predicted_symbols=["DagWidthD.run"],
    )
    right_child = _task(
        "DAG-WIDTH-E",
        dependencies=[right["canonical_task_cid"]],
        outputs=["src/dag-width-e.py"],
        predicted_paths=["src/dag-width-e.py"],
        predicted_symbols=["DagWidthE.run"],
    )

    evidence = prove_critical_path_width(
        (right_child, middle, left, left_child, right),
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=1),
        repository_tree="git-tree-asi-035",
    )

    assert evidence.proved_requirement_ids == (
        CRITICAL_PATH_WIDTH_REQUIREMENT_ID,
    )
    original_edges = {
        (parent, child)
        for child, parents in evidence.dependency_task_cids.items()
        for parent in parents
    }
    serialized_edges = {
        (parent, child)
        for child, parents in evidence.serialized_dependencies.items()
        for parent in parents
    }
    conflict_edges = serialized_edges - original_edges
    assert len(conflict_edges) == len(evidence.blocking_conflict_pairs) == 2
    assert original_edges.issubset(serialized_edges)
    assert evidence.effective_task_waves[left["canonical_task_cid"]] == (
        evidence.effective_task_waves[right["canonical_task_cid"]]
    )
    assert {
        cid
        for bundle in evidence.bundle_population
        for cid in bundle["task_cids"]
    } == {
        task["canonical_task_cid"]
        for task in (left, middle, right, left_child, right_child)
    }


def test_critical_path_width_proof_rejects_unresolved_dependency_population():
    independent = _task("CLOSED-DAG-A")
    unresolved = _task(
        "CLOSED-DAG-B",
        dependencies=["cid-outside-proof-population"],
        outputs=["src/closed-dag-b.py"],
        predicted_paths=["src/closed-dag-b.py"],
        predicted_symbols=["ClosedDagB.run"],
    )

    evidence = prove_critical_path_width(
        (independent, unresolved),
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=1),
        repository_tree="git-tree-asi-035",
    )

    assert evidence.verify_integrity()
    assert evidence.proved_requirement_ids == ()
    assert evidence.unresolved_dependency_references == {
        unresolved["canonical_task_cid"]: ("cid-outside-proof-population",)
    }


def test_width_projection_round_trip_requires_exact_canonical_replay():
    projection = project_conflict_free_wave(
        ("cid-c", "cid-a", "cid-b"),
        (("cid-b", "cid-c"), ("cid-a", "cid-b")),
        dependency_wave=4,
    )
    repeated = project_conflict_free_wave(
        reversed(projection.task_cids),
        reversed(projection.blocking_conflict_pairs),
        dependency_wave=4,
    )

    assert projection.to_dict() == repeated.to_dict()
    assert ConflictWaveProjection.from_dict(projection.to_dict()) == projection
    tampered = projection.to_dict()
    tampered["color_by_task_cid"]["cid-a"] = 7
    with pytest.raises(ValueError, match="canonical replay"):
        ConflictWaveProjection.from_dict(tampered)

    with pytest.raises(ValueError, match="inside one dependency wave"):
        project_conflict_free_wave(("cid-a",), (("cid-a", "cid-b"),))


def test_optimizer_serializes_global_ast_conflicts_across_disjoint_files():
    first = _task(
        "AST-A",
        outputs=["src/ast_a.py"],
        predicted_paths=["src/ast_a.py"],
        global_ast_symbols=["SharedProtocol.dispatch"],
    )
    second = _task(
        "AST-B",
        outputs=["src/ast_b.py"],
        predicted_paths=["src/ast_b.py"],
        global_ast_symbols=["SharedProtocol.dispatch"],
    )

    result = optimize_task_bundles(
        (first, second),
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=2),
    )
    waves = {
        task_cid: bundle.execution_wave
        for bundle in result.bundles
        for task_cid in bundle.task_cids
    }

    assert len(result.bundles) == 2
    assert len(set(waves.values())) == 2
    assert result.metrics["blocking_conflict_count"] == 1
    assert result.conflict_graph["edges"][0]["overlaps"]["ast_symbols"] == [
        "SharedProtocol.dispatch"
    ]


def test_plan_metrics_measure_real_bundle_reuse_and_compare_current_planner():
    first = _task("METRIC-A", status="completed", work_item_count=2)
    second = _task(
        "METRIC-B",
        status="completed",
        work_item_count=3,
        outputs=["src/metric-b.py"],
        predicted_paths=["src/metric-b.py"],
        predicted_symbols=["MetricB.run"],
    )
    singleton = optimize_task_bundles(
        (first, second),
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=1),
    )
    grouped = optimize_task_bundles(
        (first, second),
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=2),
    )

    assert singleton.metrics["context_reuse_millionths"] == 0
    assert singleton.metrics["validation_reuse_millionths"] == 0
    assert grouped.metrics["context_reuse_millionths"] == 500_000
    assert grouped.metrics["validation_reuse_millionths"] == 500_000
    assert grouped.metrics["accepted_work_item_count"] == 5
    assert grouped.metrics["model_calls_per_work_item_millionths"] == 200_000
    assert grouped.metrics["bundle_completion_millionths"] == 1_000_000
    assert grouped.comparison.improvements["model_call_count"] == 1


def test_optimizer_models_packet_aggregate_and_exact_covered_siblings():
    aggregate = _task(
        "AGGREGATE",
        goal_packet_key="packet/shared",
        goal_packet_role="packet_aggregate",
        candidate_kind="goal_packet_aggregate",
        completion_task_bindings=["cid-covered"],
    )
    covered = _task(
        "COVERED",
        canonical_task_cid="cid-covered",
        canonical_task_key="task/v1/covered",
        goal_packet_key="packet/shared",
        goal_packet_role="packet_member",
        outputs=["src/covered.py"],
        predicted_paths=["src/covered.py"],
    )
    same_packet_unbound = _task(
        "UNBOUND-PACKET",
        goal_packet_key="packet/shared",
        goal_packet_role="packet_member",
        outputs=["src/unbound-packet.py"],
        predicted_paths=["src/unbound-packet.py"],
    )

    result = optimize_task_bundles(
        (same_packet_unbound, aggregate, covered),
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=3),
    )

    assert len(result.packet_aggregates) == 1
    projection = result.packet_aggregates[0]
    assert projection.aggregate_task_cid == aggregate["canonical_task_cid"]
    assert projection.covered_sibling_task_cids == ("cid-covered",)
    assert same_packet_unbound["canonical_task_cid"] not in (
        projection.covered_sibling_task_cids
    )
    aggregate_bundle = next(
        bundle
        for bundle in result.bundles
        if aggregate["canonical_task_cid"] in bundle.task_cids
    )
    assert aggregate_bundle.packet_aggregate_task_cids == (
        aggregate["canonical_task_cid"],
    )
    assert aggregate_bundle.covered_sibling_task_cids == ("cid-covered",)


def test_vector_packet_keeps_exact_binding_outside_bounded_active_slice():
    common = {
        "priority": "P1",
        "track": "bundling",
        "source_line": 1,
        "bundle_key": "objective/bundling",
        "goal_packet_key": "packet/large",
        "merge_family": "packet/large",
    }
    siblings = [
        TodoIndexRecord(
            task_id=f"MEMBER-{index}",
            title=f"Member {index}",
            status="completed" if index == 6 else "todo",
            canonical_task_key=f"task/v1/member-{index}",
            task_cid=f"cid-member-{index}",
            semantic_identity=f"semantic-member-{index}",
            goal_packet_role="packet_member",
            **common,
        )
        for index in range(7)
    ]
    aggregate = TodoIndexRecord(
        task_id="LARGE-PACKET",
        title="Large packet",
        status="todo",
        canonical_task_key="task/v1/large-packet",
        task_cid="cid-large-packet",
        semantic_identity="semantic-large-packet",
        goal_packet_role="packet_aggregate",
        candidate_kind="goal_packet_aggregate",
        completion_task_bindings=[
            sibling.semantic_identity for sibling in siblings
        ],
        **common,
    )

    packet = build_execution_packet(
        context={"context_key": "context/large"},
        records=(*siblings, aggregate),
        max_tasks=3,
    )

    assert packet is not None
    assert len(packet["active_task_ids"]) == 3
    assert set(packet["completion_binding"]["bound_sibling_task_cids"]) == {
        sibling.task_cid for sibling in siblings
    }
    assert "cid-member-6" in packet["completion_binding"][
        "bound_sibling_task_cids"
    ]


def test_vector_packet_marks_conflicting_editors_for_serial_execution():
    common = {
        "status": "todo",
        "priority": "P1",
        "track": "bundling",
        "source_line": 1,
        "bundle_key": "objective/conflicting",
        "outputs": ["src/shared.py"],
    }
    first = TodoIndexRecord(
        task_id="VECTOR-A",
        title="First editor",
        canonical_task_key="task/v1/vector-a",
        task_cid="cid-vector-a",
        **common,
    )
    second = TodoIndexRecord(
        task_id="VECTOR-B",
        title="Second editor",
        canonical_task_key="task/v1/vector-b",
        task_cid="cid-vector-b",
        **common,
    )

    packet = build_execution_packet(
        context={"context_key": "context/conflicting", "merge_ready": True},
        records=(first, second),
    )

    assert packet is not None
    assert packet["merge_ready"] is False
    assert packet["serial_execution_required"] is True
    assert packet["blocking_conflict_count"] == 1
    assert set(packet["conflict_lane_by_task_cid"]) == {
        "cid-vector-a",
        "cid-vector-b",
    }
    assert packet["independent_width_by_dependency_wave"] == {"0": 1}
    assert packet["conflict_width_projections"][0]["color_count"] == 2


def test_vector_packet_uses_canonical_dag_waves_and_routes_width_producer():
    common = {
        "status": "todo",
        "priority": "P1",
        "track": "bundling",
        "source_line": 1,
        "bundle_key": "objective/dependency-width",
        "graph_depth": 7,
        "missing_evidence": [CRITICAL_PATH_WIDTH_REQUIREMENT_ID],
    }
    root = TodoIndexRecord(
        task_id="VECTOR-ROOT",
        title="Root",
        canonical_task_key="task/v1/vector-root",
        task_cid="cid-vector-root",
        outputs=["src/vector-root.py"],
        **common,
    )
    child = TodoIndexRecord(
        task_id="VECTOR-CHILD",
        title="Child",
        canonical_task_key="task/v1/vector-child",
        task_cid="cid-vector-child",
        dependency_task_cids=[root.task_cid],
        outputs=["src/vector-child.py"],
        **common,
    )

    packet = build_execution_packet(
        context={"context_key": "context/dependency-width"},
        records=(child, root),
    )

    assert packet is not None
    assert packet["dependency_projection_complete"] is True
    assert packet["dependency_wave_by_task_cid"] == {
        root.task_cid: 0,
        child.task_cid: 1,
    }
    assert packet["independent_width_by_dependency_wave"] == {"0": 1, "1": 1}
    expected_binding = {
        CRITICAL_PATH_WIDTH_REQUIREMENT_ID: (
            "bundle_optimizer.prove_critical_path_width:"
            "CriticalPathWidthEvidence"
        )
    }
    assert packet["evidence_producer_bindings"] == expected_binding
    assert task_generation_evidence_producer_bindings(
        [CRITICAL_PATH_WIDTH_REQUIREMENT_ID, "not-registered"]
    ) == expected_binding


def test_vector_packet_withholds_unresolved_tasks_from_width_projection():
    independent = TodoIndexRecord(
        task_id="VECTOR-INDEPENDENT",
        title="Independent",
        status="todo",
        priority="P1",
        track="bundling",
        source_line=1,
        bundle_key="objective/dependency-width",
        canonical_task_key="task/v1/vector-independent",
        task_cid="cid-vector-independent",
        outputs=["src/vector-independent.py"],
    )
    unresolved = TodoIndexRecord(
        task_id="VECTOR-UNRESOLVED",
        title="Unresolved",
        status="todo",
        priority="P1",
        track="bundling",
        source_line=2,
        bundle_key="objective/dependency-width",
        canonical_task_key="task/v1/vector-unresolved",
        task_cid="cid-vector-unresolved",
        dependency_task_cids=["cid-not-in-packet-population"],
        graph_depth=11,
        outputs=["src/vector-unresolved.py"],
    )

    packet = build_execution_packet(
        context={"context_key": "context/dependency-width"},
        records=(unresolved, independent),
    )

    assert packet is not None
    assert packet["dependency_projection_complete"] is False
    assert packet["dependency_wave_by_task_cid"] == {
        independent.task_cid: 0,
    }
    assert packet["dependency_projection_diagnostics"] == {
        unresolved.task_cid: [
            "unresolved_dependency:cid-not-in-packet-population"
        ]
    }
    projected_cids = {
        cid
        for projection in packet["conflict_width_projections"]
        for cid in projection["task_cids"]
    }
    assert projected_cids == {independent.task_cid}


def test_bundle_supervisor_projects_optimizer_slices_and_comparison():
    first = _task("SUPERVISOR-A")
    second = _task(
        "SUPERVISOR-B",
        outputs=["src/supervisor-b.py"],
        predicted_paths=["src/supervisor-b.py"],
        predicted_symbols=["SupervisorB.run"],
    )
    payloads = optimize_bundle_payloads(
        [
            {
                "bundle_key": "objective/shared",
                "parallel_lane": "shared",
                "tasks": [first, second],
            }
        ],
        policy=BundleOptimizationPolicy(max_tasks_per_bundle=2),
    )

    assert len(payloads) == 1
    optimization = payloads[0]["bundle_optimization"]
    assert optimization["applied"] is True
    assert set(payloads[0]["execution_slice_task_cids"]) == {
        first["canonical_task_cid"],
        second["canonical_task_cid"],
    }
    assert optimization["metrics"]["model_call_count"] == 1
    assert (
        optimization["comparison"]["current_planner"]["model_call_count"] == 1
    )


def test_bundle_supervisor_rebuilds_contract_for_derived_planning_projection():
    source = _task("SUPERVISOR-PROJECTED")
    contract = build_task_work_contract(source)
    projected = {
        **source,
        "work_contract": contract._material(),
        "work_contract_id": contract.work_contract_id,
        "task_work_contract": contract.to_dict(),
        "task_work_contract_id": contract.task_work_contract_id,
        # Dependency planning adds canonical CIDs after admission, so the
        # source contract no longer describes this execution projection.
        "dependency_task_cids": ["cid-upstream"],
    }
    rehydrated = rehydrate_task_work_contract_projection(projected)

    [optimized] = optimize_bundle_payloads(
        [
            {
                "bundle_key": "objective/projected",
                "parallel_lane": "projected",
                "tasks": [projected],
            }
        ]
    )

    assert optimized["bundle_optimization"]["applied"] is True
    assert rehydrated["dependency_task_cids"] == []
    assert (
        build_task_work_contract(rehydrated).task_work_contract_id
        == contract.task_work_contract_id
    )
    assert optimized["execution_slice_task_cids"] == [
        source["canonical_task_cid"]
    ]
    assert projected["work_contract_id"] == contract.work_contract_id


def test_bundle_supervisor_rehydrates_stale_nested_contract_aliases():
    source = _task("SUPERVISOR-NESTED-PROJECTION")
    contract = build_task_work_contract(source)
    projected = {
        **source,
        "work_contract": contract._material(),
        "work_contract_id": contract.work_contract_id,
        "task_work_contract": contract.to_dict(),
        "task_work_contract_id": contract.task_work_contract_id,
        "dependency_task_cids": ["cid-upstream"],
        "metadata": {
            "Depends on": ["cid-upstream"],
            "Outputs": ["stale/generated-projection.py"],
            "Acceptance": ["stale projected acceptance"],
            "unrelated_audit_field": "preserved",
        },
    }

    rehydrated = rehydrate_task_work_contract_projection(projected)

    assert rehydrated["metadata"] == {"unrelated_audit_field": "preserved"}
    assert rehydrated["dependency_task_cids"] == list(contract.dependencies)
    assert (
        build_task_work_contract(rehydrated).task_work_contract_id
        == contract.task_work_contract_id
    )


def test_conflict_surface_prefers_canonical_cid_over_stale_compatibility_alias():
    source = _task("SUPERVISOR-CANONICAL-CID")
    contract = build_task_work_contract(source)
    projected = {
        **source,
        "task_cid": "cid-stale-compatibility-projection",
        "work_contract": contract._material(),
        "work_contract_id": contract.work_contract_id,
        "task_work_contract": contract.to_dict(),
        "task_work_contract_id": contract.task_work_contract_id,
    }

    surface = build_conflict_surface(projected)

    assert surface.task_cid == source["canonical_task_cid"]
    assert surface.task_work_contract_id == contract.task_work_contract_id


def test_bundle_optimizer_preserves_width_for_disjoint_managed_submodule_tasks():
    first = _task(
        "SUBMODULE-ALPHA",
        outputs=["vendor/runtime/src/alpha.py"],
        predicted_paths=["vendor/runtime/src/alpha.py"],
        files=["vendor/runtime/src/alpha.py"],
        submodules=["vendor/runtime"],
        interfaces=["AlphaAPI@1"],
        goal_id="GOAL-ALPHA",
        context_paths=["vendor/runtime/src/alpha.py"],
        validation_commands=["pytest tests/test_alpha.py"],
        merge_fate="objective/GOAL-ALPHA",
    )
    second = _task(
        "SUBMODULE-BETA",
        outputs=["vendor/runtime/src/beta.py"],
        predicted_paths=["vendor/runtime/src/beta.py"],
        files=["vendor/runtime/src/beta.py"],
        submodules=["vendor/runtime"],
        interfaces=["BetaAPI@1"],
        goal_id="GOAL-BETA",
        context_paths=["vendor/runtime/src/beta.py"],
        validation_commands=["pytest tests/test_beta.py"],
        merge_fate="objective/GOAL-BETA",
        resource_class="cpu-large",
    )
    source = [
        {
            "bundle_key": "objective/managed-submodule",
            "parallel_lane": "managed-submodule",
            "tasks": [first, second],
        }
    ]

    conservative = optimize_bundle_payloads(source)
    concurrent = optimize_bundle_payloads(
        source,
        managed_submodule_paths=("vendor/runtime",),
        allow_disjoint_submodule_concurrency=True,
    )

    assert sorted(
        payload["optimizer_execution_wave"] for payload in conservative
    ) == [0, 1]
    assert sorted(
        payload["optimizer_execution_wave"] for payload in concurrent
    ) == [0, 0]
    assert all(
        payload["bundle_optimization"]["metrics"][
            "blocking_conflict_count"
        ]
        == 0
        for payload in concurrent
    )


def test_implemented_bundle_planner_serializes_disjoint_managed_submodule_tasks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = _task(
        "SUBMODULE-ALPHA",
        outputs=["vendor/runtime/src/alpha.py"],
        predicted_paths=["vendor/runtime/src/alpha.py"],
        files=["vendor/runtime/src/alpha.py"],
        submodules=["vendor/runtime"],
        interfaces=["AlphaAPI@1"],
        goal_id="GOAL-ALPHA",
        context_paths=["vendor/runtime/src/alpha.py"],
        validation_commands=["pytest tests/test_alpha.py"],
        merge_fate="objective/GOAL-ALPHA",
    )
    second = _task(
        "SUBMODULE-BETA",
        outputs=["vendor/runtime/src/beta.py"],
        predicted_paths=["vendor/runtime/src/beta.py"],
        files=["vendor/runtime/src/beta.py"],
        submodules=["vendor/runtime"],
        interfaces=["BetaAPI@1"],
        goal_id="GOAL-BETA",
        context_paths=["vendor/runtime/src/beta.py"],
        validation_commands=["pytest tests/test_beta.py"],
        merge_fate="objective/GOAL-BETA",
        resource_class="cpu-large",
    )
    source = [
        {
            "bundle_key": "objective/managed-submodule",
            "parallel_lane": "managed-submodule",
            "todo_path": "managed-submodule.todo.md",
            "tasks": [first, second],
        }
    ]
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor.build_bundle_task_payloads",
        lambda _path: source,
    )

    lanes = plan_bundle_lanes(
        bundle_index_path=tmp_path / "index.json",
        repo_root=tmp_path,
        state_root=tmp_path / "state",
        worktree_root=tmp_path / "worktrees",
        log_dir=tmp_path / "logs",
        implement=True,
        worktree_submodule_paths=("vendor/runtime",),
        allow_disjoint_submodule_concurrency=True,
    )

    assert sorted(lane.optimizer_execution_wave for lane in lanes) == [0, 1]
    assert all(
        other.bundle_key in lane.conflicting_task_ids
        for lane in lanes
        for other in lanes
        if other.bundle_key != lane.bundle_key
    )
    assert all("--implement" in lane.command for lane in lanes)


def test_bundle_supervisor_optimizer_cannot_resurrect_completed_members():
    completed = _task("SUPERVISOR-COMPLETED")
    ready = _task(
        "SUPERVISOR-READY",
        outputs=["src/supervisor-ready.py"],
        predicted_paths=["src/supervisor-ready.py"],
    )
    completed_cid = str(completed["canonical_task_cid"])
    ready_cid = str(ready["canonical_task_cid"])
    source = {
        "bundle_key": "objective/completion-overlay",
        "parallel_lane": "completion-overlay",
        # Source taskboards remain immutable and can therefore still say
        # ``todo`` after a durable completion receipt was published.
        "tasks": [completed, ready],
        "completed_member_task_cids": [completed_cid],
        "completed_member_task_ids": [completed["task_id"]],
        "ready_member_task_cids": [ready_cid],
        "ready_member_task_ids": [ready["task_id"]],
        "execution_slice_task_cids": [ready_cid],
        "execution_slice_task_ids": [ready["task_id"]],
        "claimable": True,
    }

    [optimized] = optimize_bundle_payloads([source])

    assert optimized["completed_member_task_cids"] == [completed_cid]
    assert optimized["completed_member_task_ids"] == [completed["task_id"]]
    assert optimized["execution_slice_task_cids"] == [ready_cid]
    assert optimized["execution_slice_task_ids"] == [ready["task_id"]]
    assert completed_cid not in optimized["execution_slice_task_cids"]

    completed_only = {
        **source,
        "ready_member_task_cids": [],
        "ready_member_task_ids": [],
        "execution_slice_task_cids": [],
        "execution_slice_task_ids": [],
    }
    [drained] = optimize_bundle_payloads([completed_only])

    assert drained["completed_member_task_cids"] == [completed_cid]
    assert drained["completed_member_task_ids"] == [completed["task_id"]]
    assert drained["execution_slice_task_cids"] == []
    assert drained["execution_slice_task_ids"] == []


def test_split_optimizer_slices_receive_distinct_execution_identities():
    first = _task("SLICE-A", provider_id="provider-a")
    second = _task(
        "SLICE-B",
        provider_id="provider-b",
        outputs=["src/slice-b.py"],
        predicted_paths=["src/slice-b.py"],
    )
    payloads = optimize_bundle_payloads(
        [
            {
                "bundle_key": "objective/sliced",
                "parallel_lane": "sliced",
                "tasks": [first, second],
                "profile_g": {
                    "task_cid": "legacy-shared-task-cid",
                    "task_spec_cid": "legacy-shared-task-spec-cid",
                },
            }
        ]
    )

    assert len(payloads) == 2
    assert len({payload["bundle_key"] for payload in payloads}) == 2
    assert all("profile_g" not in payload for payload in payloads)
    assert all(payload["source_profile_g_ref"] for payload in payloads)
    assert all(
        adapt_goal_bundle(payload, created_at_ms=1_783_872_000_000)[
            "task_cid"
        ]
        for payload in payloads
    )
    identities = {
        canonical_bundle_identity(payload).canonical_task_cid
        for payload in payloads
    }
    assert len(identities) == 2


G050_NOW = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)
G050_REPOSITORY_ID = "repository:ipfs-accelerate-py"
G050_REPOSITORY_TREE = "tree:sha256:asi-084-current"


def _g050_binding() -> dict[str, str]:
    return {
        "repository_id": G050_REPOSITORY_ID,
        "tree_id": G050_REPOSITORY_TREE,
        "objective_id": TASK_GENERATION_OBJECTIVE_ID,
        "objective_revision": TASK_GENERATION_OBJECTIVE_REVISION,
        "analyzer_version": TASK_GENERATION_COMPLETION_ANALYZER_VERSION,
        "configuration_revision": (
            TASK_GENERATION_COMPLETION_CONFIGURATION_REVISION
        ),
    }


def _g050_completion_packet() -> dict[str, object]:
    validation_command = (
        "python -m pytest test/api/test_agent_supervisor_task_quality.py "
        "test/api/test_agent_supervisor_bundle_optimizer.py -q"
    )
    evidence = tuple(
        CompletionEvidence(
            acceptance_criterion=criterion,
            producing_task_or_scan="ASI-084",
            producer_kind="task",
            validation_receipt={
                "status": "passed",
                "tree_id": G050_REPOSITORY_TREE,
                "command": validation_command,
            },
            validation_passed=True,
            repository_id=G050_REPOSITORY_ID,
            repository_tree=G050_REPOSITORY_TREE,
            freshness={"fresh": True},
            observed_at=G050_NOW - timedelta(minutes=2),
            provenance_cid=f"validation:asi-084:{index}",
        )
        for index, criterion in enumerate(
            TASK_GENERATION_ACCEPTANCE_CRITERIA,
            start=1,
        )
    )
    coverage = {
        "verified": True,
        "repository_id": G050_REPOSITORY_ID,
        "repository_tree": G050_REPOSITORY_TREE,
        "evaluated_at": (G050_NOW - timedelta(minutes=1)).isoformat(),
        "criteria": [
            {
                "criterion": criterion,
                "status": "verified",
                "verified": True,
                "implementation": (
                    "ipfs_accelerate_py/agent_supervisor/"
                    + (
                        "task_quality.py"
                        if index < 4
                        else "bundle_optimizer.py"
                    )
                ),
                "validation": validation_command,
                "validation_receipt_ids": [
                    f"validation:asi-084:{index}"
                ],
            }
            for index, criterion in enumerate(
                TASK_GENERATION_ACCEPTANCE_CRITERIA,
                start=1,
            )
        ],
    }
    children = [
        {
            "goal_id": goal_id,
            "state": "verified_complete",
            "verified": True,
            "completion_gate": {
                "passed": True,
                "evaluated_evidence": {
                    "repository_id": G050_REPOSITORY_ID,
                    "repository_tree": G050_REPOSITORY_TREE,
                    "evaluated_at": (
                        G050_NOW - timedelta(minutes=3)
                    ).isoformat(),
                    "validation_evidence": [
                        {
                            "valid": True,
                            "evidence": {
                                "repository_id": G050_REPOSITORY_ID,
                                "repository_tree": G050_REPOSITORY_TREE,
                            },
                        }
                    ],
                },
            },
            "proof_requirements": [
                {
                    "repository_tree": G050_REPOSITORY_TREE,
                    "provenance_id": f"proof:{goal_id}",
                    "required_assurance": "solver_checked",
                    "authoritative_assurance": "solver_checked",
                    "assurance_satisfied": True,
                    "contradicted": False,
                    "proof_verdict": "proved",
                    "freshness": "current",
                    "reason_codes": [],
                }
            ],
        }
        for goal_id in TASK_GENERATION_CHILD_GOAL_IDS
    ]
    binding = _g050_binding()
    members = [
        {
            "member_id": "asi-084-exhaustive",
            "evidence_channel": "implementation-validation",
            "receipt_cid": "scan:asi-084:exhaustive",
            "binding": dict(binding),
            "scan_mode": "exhaustive",
            "healthy": True,
            "safe_for_completion_reasoning": True,
            "finished_at": (G050_NOW - timedelta(minutes=4)).isoformat(),
        },
        {
            "member_id": "asi-084-audit",
            "evidence_channel": "independent-audit",
            "receipt_cid": "scan:asi-084:audit",
            "binding": dict(binding),
            "scan_mode": "exhaustive",
            "healthy": True,
            "safe_for_completion_reasoning": True,
            "finished_at": (G050_NOW - timedelta(minutes=3)).isoformat(),
        },
    ]
    return {
        "repository_id": G050_REPOSITORY_ID,
        "repository_tree": G050_REPOSITORY_TREE,
        "producing_tasks": [
            {"task_id": task_id, "status": "completed"}
            for task_id in TASK_GENERATION_PRODUCING_TASK_IDS
        ],
        "child_goals": children,
        "evidence": evidence,
        "tasks_complete": True,
        "coverage": coverage,
        "analyzer_health": {
            "status": "healthy",
            "healthy": True,
            "safe_for_completion_reasoning": True,
            "binding": dict(binding),
        },
        "exhaustion_quorum": {
            "required_members": TASK_GENERATION_REQUIRED_EXHAUSTIVE_RECEIPTS,
            "member_count": len(members),
            "satisfied": True,
            "quorum_met": True,
            "binding": dict(binding),
            "members": members,
        },
        "now": G050_NOW,
        "freshness_seconds": 3600,
    }


def _assert_g050_rejected(packet: dict[str, object]) -> None:
    decision = evaluate_task_generation_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **packet,
    )
    assert decision.verified is False
    assert decision.state is not GoalState.VERIFIED_COMPLETE
    assert decision.reason_codes


def test_g050_parent_completion_requires_closed_current_tree_proof_packet():
    assert TASK_GENERATION_OBJECTIVE_ID == "ASI-G050"
    assert TASK_GENERATION_PRODUCING_TASK_IDS == ("ASI-013", "ASI-014")
    assert TASK_GENERATION_CHILD_GOAL_IDS == (
        "ASI-G106",
        "ASI-G107",
        "ASI-G108",
    )
    assert TASK_GENERATION_REQUIRED_EXHAUSTIVE_RECEIPTS == 2
    assert len(TASK_GENERATION_ACCEPTANCE_CRITERIA) == 5

    packet = _g050_completion_packet()
    provisional = evaluate_task_generation_completion(
        current_state=GoalState.ACTIVE,
        **packet,
    )
    assert provisional.state is GoalState.PROVISIONALLY_COMPLETE
    assert provisional.verified is False
    assert "provisional_transition_required" in provisional.reason_codes
    assert provisional.acceptance_criteria == (
        TASK_GENERATION_ACCEPTANCE_CRITERIA
    )
    assert provisional.gate is not None and provisional.gate.passed

    verified = evaluate_task_generation_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **packet,
    )
    assert verified.state is GoalState.VERIFIED_COMPLETE
    assert verified.verified is True
    assert verified.gate is not None and verified.gate.passed
    evaluated = verified.gate.evaluated_evidence
    assert evaluated["repository_id"] == G050_REPOSITORY_ID
    assert evaluated["repository_tree"] == G050_REPOSITORY_TREE
    assert evaluated["acceptance_criteria"] == list(
        TASK_GENERATION_ACCEPTANCE_CRITERIA
    )
    assert {
        child["goal_id"] for child in evaluated["child_goals"]
    } == set(TASK_GENERATION_CHILD_GOAL_IDS)

    invalidated = copy.deepcopy(packet)
    stale_evidence = list(invalidated["evidence"])
    stale_evidence[0] = replace(
        stale_evidence[0],
        freshness={"fresh": False},
        observed_at=G050_NOW - timedelta(hours=2),
    )
    invalidated["evidence"] = tuple(stale_evidence)
    reopened = evaluate_task_generation_completion(
        current_state=GoalState.VERIFIED_COMPLETE,
        **invalidated,
    )
    assert reopened.state is GoalState.REOPENED
    assert reopened.verified is False
    assert reopened.reason_codes


@pytest.mark.parametrize(
    "producer_failure",
    ["tasks_not_complete", "missing", "wrong", "duplicate", "incomplete"],
)
def test_g050_parent_rejects_incomplete_wrong_or_duplicate_producers(
    producer_failure: str,
):
    packet = _g050_completion_packet()
    producers = copy.deepcopy(packet["producing_tasks"])
    if producer_failure == "tasks_not_complete":
        packet["tasks_complete"] = False
    elif producer_failure == "missing":
        packet["producing_tasks"] = producers[:-1]
    elif producer_failure == "wrong":
        producers[-1]["task_id"] = "ASI-084"
        packet["producing_tasks"] = producers
    elif producer_failure == "duplicate":
        producers[-1]["task_id"] = producers[0]["task_id"]
        packet["producing_tasks"] = producers
    else:
        producers[-1]["status"] = "active"
        packet["producing_tasks"] = producers

    _assert_g050_rejected(packet)


@pytest.mark.parametrize(
    "evidence_failure",
    ["missing", "wrong", "duplicate", "stale", "failed", "foreign"],
)
def test_g050_parent_rejects_each_invalid_submitted_criterion_evidence(
    evidence_failure: str,
):
    packet = _g050_completion_packet()
    evidence = list(packet["evidence"])
    if evidence_failure == "missing":
        evidence.pop()
    elif evidence_failure == "wrong":
        evidence[-1] = replace(
            evidence[-1],
            acceptance_criterion="caller-selected substitute criterion",
        )
    elif evidence_failure == "duplicate":
        evidence[-1] = replace(
            evidence[-1],
            acceptance_criterion=evidence[0].acceptance_criterion,
            provenance_cid=evidence[0].provenance_cid,
        )
    elif evidence_failure == "stale":
        evidence[0] = replace(
            evidence[0],
            freshness={"fresh": False},
            observed_at=G050_NOW - timedelta(hours=2),
        )
    elif evidence_failure == "failed":
        evidence[0] = replace(
            evidence[0],
            validation_passed=False,
            validation_receipt={
                "status": "failed",
                "tree_id": G050_REPOSITORY_TREE,
            },
        )
    else:
        evidence[0] = replace(
            evidence[0],
            repository_id="repository:foreign",
            repository_tree="tree:sha256:foreign",
        )
    packet["evidence"] = tuple(evidence)

    _assert_g050_rejected(packet)


@pytest.mark.parametrize(
    "coverage_failure",
    ["missing_row", "duplicate_row", "missing_implementation", "unbound_receipt"],
)
def test_g050_parent_rejects_incomplete_or_unbound_coverage(
    coverage_failure: str,
):
    packet = _g050_completion_packet()
    coverage = copy.deepcopy(packet["coverage"])
    rows = coverage["criteria"]
    if coverage_failure == "missing_row":
        rows.pop()
    elif coverage_failure == "duplicate_row":
        rows[-1] = copy.deepcopy(rows[0])
    elif coverage_failure == "missing_implementation":
        rows[0]["implementation"] = ""
    else:
        rows[0]["validation_receipt_ids"] = ["validation:foreign"]
    packet["coverage"] = coverage

    _assert_g050_rejected(packet)


@pytest.mark.parametrize(
    "analyzer_failure",
    ["missing", "unhealthy", "unsafe", "mismatched_binding"],
)
def test_g050_parent_rejects_missing_unhealthy_unsafe_or_foreign_analyzer(
    analyzer_failure: str,
):
    packet = _g050_completion_packet()
    health = copy.deepcopy(packet["analyzer_health"])
    if analyzer_failure == "missing":
        health = {}
    elif analyzer_failure == "unhealthy":
        health["status"] = "degraded"
        health["healthy"] = False
    elif analyzer_failure == "unsafe":
        health["safe_for_completion_reasoning"] = False
    else:
        health["binding"]["analyzer_version"] = "foreign-analyzer@1"
    packet["analyzer_health"] = health

    _assert_g050_rejected(packet)


def test_g050_parent_rejects_caller_lowered_exhaustive_quorum():
    with pytest.raises(
        ValueError,
        match="must equal the configured ASI-G050 count",
    ):
        evaluate_task_generation_completion(
            required_exhaustive_receipts=1,
            **_g050_completion_packet(),
        )


@pytest.mark.parametrize(
    "quorum_failure",
    [
        "insufficient",
        "duplicate_member",
        "duplicate_channel",
        "duplicate_receipt",
        "unhealthy",
        "unsafe",
        "non_exhaustive",
        "stale",
        "foreign",
        "foreign_member",
    ],
)
def test_g050_parent_rejects_nonindependent_or_unhealthy_exhaustive_quorum(
    quorum_failure: str,
):
    packet = _g050_completion_packet()
    quorum = copy.deepcopy(packet["exhaustion_quorum"])
    members = quorum["members"]
    if quorum_failure == "insufficient":
        members.pop()
        quorum["member_count"] = 1
    elif quorum_failure == "duplicate_member":
        members[1]["member_id"] = members[0]["member_id"]
    elif quorum_failure == "duplicate_channel":
        members[1]["evidence_channel"] = members[0]["evidence_channel"]
    elif quorum_failure == "duplicate_receipt":
        members[1]["receipt_cid"] = members[0]["receipt_cid"]
    elif quorum_failure == "unhealthy":
        members[1]["healthy"] = False
    elif quorum_failure == "unsafe":
        members[1]["safe_for_completion_reasoning"] = False
    elif quorum_failure == "non_exhaustive":
        members[1]["scan_mode"] = "partial"
    elif quorum_failure == "stale":
        members[1]["finished_at"] = (
            G050_NOW - timedelta(hours=2)
        ).isoformat()
    elif quorum_failure == "foreign":
        quorum["binding"]["tree_id"] = "tree:sha256:foreign"
    else:
        members[1]["binding"]["tree_id"] = "tree:sha256:foreign"
    packet["exhaustion_quorum"] = quorum

    _assert_g050_rejected(packet)


@pytest.mark.parametrize(
    "child_failure",
    ["missing", "wrong", "duplicate", "unverified", "stale", "foreign"],
)
def test_g050_parent_rejects_unverified_stale_or_wrong_child_population(
    child_failure: str,
):
    packet = _g050_completion_packet()
    children = copy.deepcopy(packet["child_goals"])
    if child_failure == "missing":
        children.pop()
    elif child_failure == "wrong":
        children[-1]["goal_id"] = "ASI-G999"
    elif child_failure == "duplicate":
        children[-1]["goal_id"] = children[0]["goal_id"]
    elif child_failure == "unverified":
        children[0]["state"] = "provisionally_complete"
        children[0]["verified"] = False
    elif child_failure == "stale":
        children[0]["proof_requirements"][0]["freshness"] = "stale"
    else:
        children[0]["completion_gate"]["evaluated_evidence"][
            "repository_tree"
        ] = "tree:sha256:foreign"
    packet["child_goals"] = children

    _assert_g050_rejected(packet)


def test_optimizer_projection_preserves_coherent_contract_effects_and_costs():
    task = _task(
        "CONTRACT-PROJECTION",
        acceptance=[
            "The admitted task owns one exact acceptance subset",
            "Every output remains current-tree-bound",
        ],
        effects=["write src/contract.py", "run contract validation"],
        estimated_context_tokens=321,
        estimated_tokens=654,
        estimated_validation_seconds=17,
    )

    result = optimize_task_bundles((task,))
    assert len(result.bundles) == 1
    bundle = result.bundles[0]
    assert bundle.acceptance_subsets == tuple(sorted(task["acceptance"]))
    assert bundle.effect_subsets == tuple(sorted(task["effects"]))
    assert bundle.predicted_paths == tuple(task["predicted_paths"])
    assert bundle.predicted_symbols == tuple(task["predicted_symbols"])
    assert bundle.estimated_context_tokens == 321
    assert bundle.estimated_tokens == 654
    assert bundle.estimated_validation_seconds == 17
    assert len(bundle.task_work_contracts) == 1
    contract = bundle.task_work_contracts[0]
    assert contract["acceptance_subset"] == sorted(task["acceptance"])
    assert contract["effect_subset"] == sorted(task["effects"])
    assert contract["estimated_costs"] == {
        "context_tokens": 321,
        "task_tokens": 654,
        "validation_seconds": 17,
    }
    projection = result.to_dict()["bundles"][0]
    assert projection["task_work_contracts"] == [contract]
    assert projection["work_contract_ids"] == list(bundle.work_contract_ids)


def test_optimizer_preserves_admitted_task_candidate_contract_without_drift():
    candidate = TaskCandidate.from_mapping(
        _task("ADMITTED-CONTRACT", outputs=["src/admitted.py"]),
        validate_identity=False,
    )

    result = optimize_task_bundles((candidate.to_dict(),))
    bundle = result.bundles[0]
    planning_contract = bundle.task_work_contracts[0]

    assert bundle.work_contract_ids == (candidate.work_contract_id,)
    assert planning_contract["canonical_task_cid"] == (
        candidate.canonical_task_cid
    )
    assert planning_contract["canonical_task_key"] == (
        candidate.canonical_task_key
    )
    assert planning_contract["work_contract"] == candidate.work_contract
    assert planning_contract["work_contract_id"] == candidate.work_contract_id
