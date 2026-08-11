"""Focused contract tests for change-propagation task projection (RPR-042)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    AnalyticalTransform,
    ConsumerDisposition,
    ConsumerMigrationObligation,
    GraphNodeRef,
    GraphProvenance,
    ImpactClosureReceipt,
    ImpactCompleteness,
    ImpactConsumer,
    ImpactSCC,
    PropagationAuthorityRoots,
    TransformDisposition,
    TransformKind,
)
from ipfs_accelerate_py.agent_supervisor.objectives.change_propagation_task_source import (
    CHANGE_PROPAGATION_TASK_SOURCE_INTERFACE,
    ChangePropagationTaskProjectionReason,
    ChangePropagationTaskSource,
    deterministic_change_propagation_task_id,
)
from ipfs_accelerate_py.agent_supervisor.planning.change_propagation_plan import (
    ChangePropagationPlanner,
    PlanEvidenceBundle,
    PlanPathSpan,
    PlanResourceBounds,
    PlanValidationCommand,
)
from ipfs_accelerate_py.agent_supervisor.planning.support_behavior_placement import (
    SupportPlacementAction,
    SupportPlacementDecision,
    SupportPlacementDisposition,
)
from ipfs_accelerate_py.agent_supervisor.proof.change_propagation_edit_packet import (
    PropagationEditStepKind,
    materialize_change_propagation_edit_packet,
)
from ipfs_accelerate_py.agent_supervisor.proof.missing_input_synthesis import (
    SynthesisDisposition,
    ValueMappingProof,
)


@pytest.fixture
def roots() -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id="repository:rpr-042",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:rpr-042",
        index_id="index:rpr-042",
        model_id="model:rpr-042",
        config_id="config:rpr-042",
        translator_id="translator:rpr-042",
        toolchain_id="toolchain:rpr-042",
        policy_id="policy:rpr-042",
    )


def _node(path: str, symbol: str) -> GraphNodeRef:
    return GraphNodeRef(
        node_id=f"node:{symbol}",
        kind="function",
        path=path,
        symbol_id=symbol,
        artifact_id=f"blob:{symbol}",
        provenance=GraphProvenance.TRUSTED,
        extractor_id="extractor:ast",
    )


def _obligation(
    roots: PropagationAuthorityRoots,
    *,
    consumer_id: str = "consumer:one",
    path: str = "pkg/caller.py",
    missing: tuple[str, ...] = ("missing:context",),
    behavior: tuple[str, ...] = (),
) -> ConsumerMigrationObligation:
    return ConsumerMigrationObligation(
        roots=roots,
        obligation_id=f"obligation:{consumer_id}",
        consumer_id=consumer_id,
        delta_id="delta:one",
        disposition=ConsumerDisposition.MIGRATE,
        clause_ids=("clause:param-add",),
        node=_node(path, f"symbol:{consumer_id}"),
        proof_refs=("proof:obligation",),
        missing_input_ids=missing,
        behavior_contract_ids=behavior,
        invalidation_refs=("tree:candidate",),
    )


def _consumer(consumer_id: str = "consumer:one", path: str = "pkg/caller.py") -> ImpactConsumer:
    return ImpactConsumer(
        consumer_id=consumer_id,
        node=_node(path, f"symbol:{consumer_id}"),
        depth=1,
        mandatory=True,
        edge_refs=(f"edge:{consumer_id}",),
    )


def _mapping(
    *,
    consumer_id: str = "consumer:one",
    requirement_id: str = "missing:context",
) -> ValueMappingProof:
    return ValueMappingProof(
        requirement_id=requirement_id,
        consumer_id=consumer_id,
        disposition=SynthesisDisposition.UNIQUE_PROVED,
        facet_results=(),
        proved_candidate_ids=("candidate:ctx",),
        refuted_candidate_ids=(),
        expression_ref="expr:ctx",
        type_ref="type:Context",
        repository_id="repository:rpr-042",
        tree_id="tree:candidate",
        toolchain_id="toolchain:rpr-042",
        policy_id="policy:rpr-042",
        reason_codes=("unique_source",),
    )


def _transform(
    roots: PropagationAuthorityRoots,
    *,
    transform_id: str = "transform:add-arg",
    obligation_ids: tuple[str, ...] = ("obligation:consumer:one",),
    path: str = "pkg/caller.py",
    deps: tuple[str, ...] = (),
) -> AnalyticalTransform:
    return AnalyticalTransform(
        roots=roots,
        transform_id=transform_id,
        kind=TransformKind.ADD_ARGUMENT,
        disposition=TransformDisposition.ADMITTED,
        obligation_ids=obligation_ids,
        target_paths=(path,),
        expression_refs=("expr:ctx",),
        proof_refs=("proof:transform",),
        dependency_transform_ids=deps,
    )


def _validation() -> tuple[PlanValidationCommand, ...]:
    return (
        PlanValidationCommand(
            command_id="validate:pytest",
            argv=("python", "-m", "pytest", "-q", "test_propagation.py"),
            required=True,
        ),
    )


def analytical_packet(roots: PropagationAuthorityRoots):
    consumer = _consumer()
    obligation = _obligation(roots)
    evidence = PlanEvidenceBundle(
        roots=roots,
        change_set_id="changeset:one",
        delta_id="delta:one",
        impact_closure=ImpactClosureReceipt(
            roots=roots,
            delta_id="delta:one",
            completeness=ImpactCompleteness.COMPLETE,
            consumers=(consumer,),
            sccs=(),
            frontier_node_ids=(),
            frontier_edge_ids=(),
            validation_refs=("validation:impact",),
            resource_bound_refs=("bound:impact",),
            evidence_refs=("evidence:graph",),
        ),
        obligations=(obligation,),
        value_mapping_proofs=(_mapping(),),
        analytical_transforms=(_transform(roots),),
        placement_decisions=(),
        read_spans=(
            PlanPathSpan(
                path="pkg/caller.py",
                start=0,
                end=40,
                artifact_id="blob:caller",
                before_hash="sha256:caller-before",
            ),
        ),
        write_spans=(
            PlanPathSpan(
                path="pkg/caller.py",
                start=10,
                end=30,
                artifact_id="blob:caller",
                before_hash="sha256:caller-before",
            ),
        ),
        validation_commands=_validation(),
        resource_bounds=PlanResourceBounds(),
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        candidate_set_id="candidate-set:one",
        expected_roots=roots,
    )
    admission = ChangePropagationPlanner().admit(evidence)
    assert admission.admitted
    packet = materialize_change_propagation_edit_packet(
        admission, roots=roots, evidence=evidence
    )
    return packet, admission, evidence


def mixed_packet(roots: PropagationAuthorityRoots):
    c1 = _consumer("consumer:a", "pkg/a.py")
    c2 = _consumer("consumer:b", "pkg/support/context.py")
    o1 = _obligation(roots, consumer_id="consumer:a", path="pkg/a.py")
    o2 = _obligation(
        roots,
        consumer_id="consumer:b",
        path="pkg/support/context.py",
        missing=(),
        behavior=("behavior:SupportContext",),
    )
    evidence = PlanEvidenceBundle(
        roots=roots,
        change_set_id="changeset:mixed",
        delta_id="delta:one",
        impact_closure=ImpactClosureReceipt(
            roots=roots,
            delta_id="delta:one",
            completeness=ImpactCompleteness.COMPLETE,
            consumers=(c1, c2),
            sccs=(),
            frontier_node_ids=(),
            frontier_edge_ids=(),
            validation_refs=("validation:impact",),
            resource_bound_refs=("bound:impact",),
            evidence_refs=("evidence:graph",),
        ),
        obligations=(o1, o2),
        value_mapping_proofs=(_mapping(consumer_id="consumer:a"),),
        analytical_transforms=(
            _transform(
                roots,
                transform_id="transform:a",
                obligation_ids=("obligation:consumer:a",),
                path="pkg/a.py",
            ),
        ),
        placement_decisions=(
            SupportPlacementDecision(
                disposition=SupportPlacementDisposition.ADMITTED,
                roots=roots,
                behavior_id="behavior:SupportContext",
                candidate_set_id="placement-set:one",
                selected_candidate_id="candidate:owner",
                action=SupportPlacementAction.PLACE_NEW,
                target_path="pkg/support/context.py",
                placement_paths=("pkg/support/context.py",),
                reason_codes=("owner_unique",),
                proof_receipt_ids=("proof:placement",),
                evidence_refs=("evidence:arch",),
                eligible_candidate_ids=("candidate:owner",),
                margin=2,
            ),
        ),
        write_spans=(
            PlanPathSpan(
                path="pkg/a.py",
                start=0,
                end=10,
                artifact_id="blob:a",
                before_hash="sha256:a",
            ),
            PlanPathSpan(
                path="pkg/support/context.py",
                start=0,
                end=10,
                artifact_id="blob:support",
                before_hash="sha256:support",
            ),
        ),
        validation_commands=_validation(),
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        expected_roots=roots,
    )
    admission = ChangePropagationPlanner().admit(evidence)
    assert admission.admitted
    packet = materialize_change_propagation_edit_packet(
        admission, roots=roots, evidence=evidence
    )
    return packet, admission, evidence


def scc_packet(roots: PropagationAuthorityRoots):
    c1 = _consumer("consumer:a", "pkg/a.py")
    c2 = _consumer("consumer:b", "pkg/b.py")
    scc = ImpactSCC(scc_id="scc:cycle", member_consumer_ids=("consumer:a", "consumer:b"))
    o1 = _obligation(roots, consumer_id="consumer:a", path="pkg/a.py")
    o2 = _obligation(roots, consumer_id="consumer:b", path="pkg/b.py")
    t1 = _transform(
        roots,
        transform_id="transform:a",
        obligation_ids=("obligation:consumer:a",),
        path="pkg/a.py",
    )
    t2 = _transform(
        roots,
        transform_id="transform:b",
        obligation_ids=("obligation:consumer:b",),
        path="pkg/b.py",
        deps=("transform:a",),
    )
    evidence = PlanEvidenceBundle(
        roots=roots,
        change_set_id="changeset:scc",
        delta_id="delta:one",
        impact_closure=ImpactClosureReceipt(
            roots=roots,
            delta_id="delta:one",
            completeness=ImpactCompleteness.COMPLETE,
            consumers=(c1, c2),
            sccs=(scc,),
            frontier_node_ids=(),
            frontier_edge_ids=(),
            validation_refs=("validation:impact",),
            resource_bound_refs=("bound:impact",),
            evidence_refs=("evidence:graph",),
        ),
        obligations=(o1, o2),
        value_mapping_proofs=(
            _mapping(consumer_id="consumer:a"),
            _mapping(consumer_id="consumer:b"),
        ),
        analytical_transforms=(t1, t2),
        write_spans=(
            PlanPathSpan(
                path="pkg/a.py", start=0, end=10, artifact_id="blob:a", before_hash="h:a"
            ),
            PlanPathSpan(
                path="pkg/b.py", start=0, end=10, artifact_id="blob:b", before_hash="h:b"
            ),
        ),
        validation_commands=_validation(),
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        expected_roots=roots,
    )
    admission = ChangePropagationPlanner().admit(evidence)
    packet = materialize_change_propagation_edit_packet(
        admission, roots=roots, evidence=evidence
    )
    return packet, admission, evidence


def test_projection_is_deterministic_idempotent_and_exact_scope(
    roots: PropagationAuthorityRoots,
) -> None:
    packet, _, _ = analytical_packet(roots)
    source = ChangePropagationTaskSource()
    first = source.project(packet, current_roots=roots)
    second = source.project(packet, roots=roots)

    assert first is second
    assert first.emitted
    assert first.packet_id == packet.packet_id
    assert first.plan_id == packet.plan_id
    assert first.tree_id == roots.candidate_tree_id
    assert first.predicted_files == first.write_scope == packet.permitted_write_paths
    assert first.step_order == packet.step_order
    assert first.scc_group_ids == packet.scc_group_ids
    assert len(first.step_tasks) == len(packet.steps)
    by_packet_step = {step.step_id: step for step in packet.steps}
    for step_task in first.step_tasks:
        step = by_packet_step[step_task.step_id]
        assert step_task.write_paths == step.write_paths
        assert step_task.task_id == deterministic_change_propagation_task_id(
            packet.packet_id, packet.plan_id, step.step_id, roots.candidate_tree_id
        )
        assert step_task.task_record is not None
        assert tuple(step_task.task_record.finding.outputs) == step.write_paths
        assert tuple(step_task.task_record.finding.predicted_files) == step.write_paths
        assert CHANGE_PROPAGATION_TASK_SOURCE_INTERFACE in step_task.task_record.finding.interfaces
    assert tuple(item.step_id for item in first.step_tasks) == packet.step_order
    assert first.projection_id == second.projection_id


def test_prompts_expose_admitted_values_behavior_limits_and_fixed_point(
    roots: PropagationAuthorityRoots,
) -> None:
    packet, _, _ = mixed_packet(roots)
    projection = ChangePropagationTaskSource().project(
        packet, provider_outputs=packet.permitted_write_paths
    )
    assert projection.emitted
    assert projection.provider_step_tasks
    assert projection.analytical_step_tasks

    for step_task in projection.step_tasks:
        prompt = step_task.prompt
        assert packet.plan_id in prompt
        assert packet.packet_id in prompt
        assert step_task.step_id in prompt
        assert packet.fixed_point_obligation_ref in prompt
        assert "Unsupported limits:" in prompt
        assert "Fixed-point obligation:" in prompt
        assert "must not add, modify, rename" in prompt
        assert packet.validation_commands[0] in prompt
        if packet.selected_value_sources:
            assert packet.selected_value_sources[0].candidate_id in prompt or "none" in prompt
        if step_task.invokes_provider:
            assert "MODEL-REQUIRED" in prompt
            assert "llm_router" in prompt
            assert step_task.kind is PropagationEditStepKind.MODEL_REQUIRED
            # Behavior must be visible for model steps.
            assert "behavior:SupportContext" in prompt
        else:
            assert "MUST NOT invoke a provider" in prompt
            assert step_task.kind is not PropagationEditStepKind.MODEL_REQUIRED

    rejected = ChangePropagationTaskSource().project(
        packet, provider_outputs=("pkg/a.py", "pkg/widened.py")
    )
    assert rejected.reason is ChangePropagationTaskProjectionReason.SCOPE_MISMATCH
    assert rejected.implementation_tasks == ()


def test_analytical_steps_never_invoke_a_provider(roots: PropagationAuthorityRoots) -> None:
    packet, _, _ = analytical_packet(roots)
    projection = ChangePropagationTaskSource().project(packet, current_roots=roots)
    assert projection.emitted
    assert projection.provider_step_tasks == ()
    assert projection.analytical_step_tasks
    for step_task in projection.step_tasks:
        assert step_task.invokes_provider is False
        assert step_task.kind is PropagationEditStepKind.ANALYTICAL
        assert "MUST NOT invoke a provider" in step_task.prompt
        assert step_task.task_record is not None
        assert "invokes_provider=False" in step_task.task_record.finding.effects


def test_scc_order_and_dependency_metadata_preserved(roots: PropagationAuthorityRoots) -> None:
    packet, _, _ = scc_packet(roots)
    projection = ChangePropagationTaskSource().project(packet, current_roots=roots)
    assert projection.emitted
    assert projection.step_order == packet.step_order
    assert projection.scc_group_ids == packet.scc_group_ids
    by_step = {item.step_id: item for item in projection.step_tasks}
    for step in packet.steps:
        task = by_step[step.step_id]
        assert task.dependency_step_ids == step.dependency_step_ids
        assert task.scc_group_id == step.scc_group_id
        assert task.task_record is not None
        assert task.task_record.depends_on == task.dependency_task_ids
        for dep_step_id in step.dependency_step_ids:
            assert by_step[dep_step_id].task_id in task.dependency_task_ids


def test_stale_malformed_and_scope_mismatch_emit_no_task(
    roots: PropagationAuthorityRoots,
) -> None:
    packet, _, _ = analytical_packet(roots)
    stale_roots = replace(roots, candidate_tree_id="tree:changed", candidate_overlay_id="overlay:changed")
    stale = ChangePropagationTaskSource().project(packet, current_roots=stale_roots)
    assert stale.reason is ChangePropagationTaskProjectionReason.STALE
    assert stale.implementation_tasks == ()

    malformed = ChangePropagationTaskSource().project({"not": "a packet"})
    assert malformed.reason is ChangePropagationTaskProjectionReason.MALFORMED
    assert malformed.implementation_tasks == ()

    tree_stale = ChangePropagationTaskSource(current_tree_id="tree:other").project(packet)
    assert tree_stale.reason is ChangePropagationTaskProjectionReason.STALE


def test_duplicate_plans_do_not_duplicate_tasks(roots: PropagationAuthorityRoots) -> None:
    packet, _, _ = analytical_packet(roots)
    source = ChangePropagationTaskSource()
    emitted = source.project(packet)
    duplicate = source.project(packet)
    assert duplicate is emitted

    # Same plan identity via re-materialized equal packet still dedupes by plan/tree.
    batch = source.project_many((packet, packet))
    assert batch[0] is emitted
    assert batch[1].reason is ChangePropagationTaskProjectionReason.DUPLICATE
    assert batch[1].implementation_tasks == ()

    # A second projection source starting empty still issues the same task ids.
    other = ChangePropagationTaskSource().project(packet)
    assert other.emitted
    assert [item.task_id for item in other.step_tasks] == [
        item.task_id for item in emitted.step_tasks
    ]
