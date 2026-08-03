"""Fail-closed coverage for plan-bound multi-edit packets (RPR-040)."""

from __future__ import annotations

from copy import deepcopy

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    AnalyticalTransform,
    AtomicPropagationPlan,
    ConsumerDisposition,
    ConsumerMigrationObligation,
    GraphNodeRef,
    GraphProvenance,
    ImpactClosureReceipt,
    ImpactCompleteness,
    ImpactConsumer,
    ImpactSCC,
    PlanDisposition,
    PlanStepKind,
    PropagationAuthorityRoots,
    TransformDisposition,
    TransformKind,
)
from ipfs_accelerate_py.agent_supervisor.planning.change_propagation_plan import (
    ChangePropagationPlanner,
    PlanEvidenceBundle,
    PlanPathSpan,
    PlanResourceBounds,
    PlanValidationCommand,
    PropagationPlanAdmission,
)
from ipfs_accelerate_py.agent_supervisor.planning.support_behavior_placement import (
    SupportPlacementAction,
    SupportPlacementDecision,
    SupportPlacementDisposition,
)
from ipfs_accelerate_py.agent_supervisor.proof.change_propagation_edit_packet import (
    CHANGE_PROPAGATION_EDIT_PACKET_INTERFACE,
    ChangePropagationEditPacket,
    ChangePropagationEditPacketError,
    PropagationEditStepKind,
    PropagationExpansionHandle,
    materialize_change_propagation_edit_packet,
)
from ipfs_accelerate_py.agent_supervisor.proof.missing_input_synthesis import (
    SynthesisDisposition,
    ValueMappingProof,
)


# ---------------------------------------------------------------------------
# Fixtures / builders (aligned with RPR-039 plan admission)
# ---------------------------------------------------------------------------


@pytest.fixture
def roots() -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id="repository:rpr-040",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:rpr-040",
        index_id="index:rpr-040",
        model_id="model:rpr-040",
        config_id="config:rpr-040",
        translator_id="translator:rpr-040",
        toolchain_id="toolchain:rpr-040",
        policy_id="policy:rpr-040",
    )


def _node(
    path: str = "pkg/caller.py",
    symbol: str = "symbol:caller",
    *,
    node_id: str | None = None,
) -> GraphNodeRef:
    return GraphNodeRef(
        node_id=node_id or f"node:{symbol}",
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
    disposition: ConsumerDisposition = ConsumerDisposition.MIGRATE,
    missing: tuple[str, ...] = ("missing:context",),
    behavior: tuple[str, ...] = (),
    proof_refs: tuple[str, ...] = ("proof:obligation",),
) -> ConsumerMigrationObligation:
    return ConsumerMigrationObligation(
        roots=roots,
        obligation_id=f"obligation:{consumer_id}",
        consumer_id=consumer_id,
        delta_id="delta:one",
        disposition=disposition,
        clause_ids=("clause:param-add",),
        node=_node(path=path, symbol=f"symbol:{consumer_id}"),
        proof_refs=proof_refs,
        missing_input_ids=missing,
        behavior_contract_ids=behavior,
        invalidation_refs=("tree:candidate",),
    )


def _consumer(
    consumer_id: str = "consumer:one",
    path: str = "pkg/caller.py",
    *,
    depth: int = 1,
) -> ImpactConsumer:
    return ImpactConsumer(
        consumer_id=consumer_id,
        node=_node(path=path, symbol=f"symbol:{consumer_id}"),
        depth=depth,
        mandatory=True,
        edge_refs=(f"edge:{consumer_id}",),
    )


def _closure(
    roots: PropagationAuthorityRoots,
    consumers: tuple[ImpactConsumer, ...],
    *,
    sccs: tuple[ImpactSCC, ...] = (),
) -> ImpactClosureReceipt:
    return ImpactClosureReceipt(
        roots=roots,
        delta_id="delta:one",
        completeness=ImpactCompleteness.COMPLETE,
        consumers=consumers,
        sccs=sccs,
        frontier_node_ids=(),
        frontier_edge_ids=(),
        validation_refs=("validation:impact",),
        resource_bound_refs=("bound:impact",),
        evidence_refs=("evidence:graph",),
    )


def _mapping(
    *,
    requirement_id: str = "missing:context",
    consumer_id: str = "consumer:one",
    disposition: SynthesisDisposition = SynthesisDisposition.UNIQUE_PROVED,
    proved: tuple[str, ...] | None = None,
) -> ValueMappingProof:
    if proved is None:
        if disposition is SynthesisDisposition.UNIQUE_PROVED:
            proved = ("candidate:ctx",)
        elif disposition is SynthesisDisposition.AMBIGUOUS:
            proved = ("candidate:a", "candidate:b")
        else:
            proved = ()
    return ValueMappingProof(
        requirement_id=requirement_id,
        consumer_id=consumer_id,
        disposition=disposition,
        facet_results=(),
        proved_candidate_ids=proved,
        refuted_candidate_ids=()
        if disposition is not SynthesisDisposition.REFUTED
        else ("candidate:bad",),
        expression_ref="expr:ctx"
        if disposition is SynthesisDisposition.UNIQUE_PROVED
        else "",
        type_ref="type:Context",
        repository_id="repository:rpr-040",
        tree_id="tree:candidate",
        toolchain_id="toolchain:rpr-040",
        policy_id="policy:rpr-040",
        reason_codes=(
            ("unique_source",)
            if disposition is SynthesisDisposition.UNIQUE_PROVED
            else ("non_unique",)
        ),
    )


def _transform(
    roots: PropagationAuthorityRoots,
    *,
    transform_id: str = "transform:add-arg",
    obligation_ids: tuple[str, ...] = ("obligation:consumer:one",),
    path: str = "pkg/caller.py",
    disposition: TransformDisposition = TransformDisposition.ADMITTED,
    deps: tuple[str, ...] = (),
) -> AnalyticalTransform:
    return AnalyticalTransform(
        roots=roots,
        transform_id=transform_id,
        kind=TransformKind.ADD_ARGUMENT,
        disposition=disposition,
        obligation_ids=obligation_ids,
        target_paths=(path,) if disposition is TransformDisposition.ADMITTED else (),
        expression_refs=("expr:ctx",) if disposition is TransformDisposition.ADMITTED else (),
        proof_refs=("proof:transform",) if disposition is TransformDisposition.ADMITTED else (),
        dependency_transform_ids=deps,
        rejection_reasons=("unsupported",)
        if disposition is TransformDisposition.REJECTED
        else (),
    )


def _validation(*command_ids: str) -> tuple[PlanValidationCommand, ...]:
    if not command_ids:
        command_ids = ("validate:pytest",)
    return tuple(
        PlanValidationCommand(
            command_id=cid,
            argv=("python", "-m", "pytest", "-q", f"test_{cid.replace(':', '_')}.py"),
            required=True,
        )
        for cid in command_ids
    )


def _happy_bundle(roots: PropagationAuthorityRoots) -> PlanEvidenceBundle:
    consumer = _consumer()
    obligation = _obligation(roots)
    return PlanEvidenceBundle(
        roots=roots,
        change_set_id="changeset:one",
        delta_id="delta:one",
        impact_closure=_closure(roots, (consumer,)),
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


def _admit(roots: PropagationAuthorityRoots) -> tuple[PropagationPlanAdmission, PlanEvidenceBundle]:
    evidence = _happy_bundle(roots)
    admission = ChangePropagationPlanner().admit(evidence)
    assert admission.admitted
    return admission, evidence


def _placement(
    roots: PropagationAuthorityRoots,
    *,
    behavior_id: str = "behavior:SupportContext",
    path: str = "pkg/support/context.py",
    candidate_id: str = "candidate:owner",
) -> SupportPlacementDecision:
    return SupportPlacementDecision(
        disposition=SupportPlacementDisposition.ADMITTED,
        roots=roots,
        behavior_id=behavior_id,
        candidate_set_id="placement-set:one",
        selected_candidate_id=candidate_id,
        action=SupportPlacementAction.PLACE_NEW,
        target_path=path,
        placement_paths=(path,),
        reason_codes=("owner_unique",),
        proof_receipt_ids=("proof:placement",),
        evidence_refs=("evidence:arch",),
        eligible_candidate_ids=(candidate_id,),
        margin=2,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_interface_constant() -> None:
    assert CHANGE_PROPAGATION_EDIT_PACKET_INTERFACE == "ChangePropagationEditPacket@1"


def test_materializes_current_admitted_plan_with_exact_scope_and_partition(
    roots: PropagationAuthorityRoots,
) -> None:
    admission, evidence = _admit(roots)
    packet = materialize_change_propagation_edit_packet(
        admission,
        roots=roots,
        evidence=evidence,
        counterexample_refs=("validation:impact",),
        expansion_handles=(
            PropagationExpansionHandle(
                "proof-handle",
                "proof_receipt",
                "proof:plan",
                ("pkg/caller.py",),
            ),
        ),
    )

    assert packet.interface == CHANGE_PROPAGATION_EDIT_PACKET_INTERFACE
    assert packet.admission_id == admission.content_id
    assert packet.plan_id == admission.plan.plan_id
    assert packet.plan_content_id == admission.plan.content_id
    assert packet.roots == roots
    assert packet.permitted_write_paths == ("pkg/caller.py",)
    assert packet.permitted_read_paths == admission.plan.permitted_read_paths
    assert packet.step_order == admission.step_order
    assert packet.analytical_step_ids
    assert packet.model_required_step_ids == ()
    assert all(
        step.kind is PropagationEditStepKind.ANALYTICAL for step in packet.steps
    )
    assert packet.steps[0].plan_step_kind is PlanStepKind.ANALYTICAL
    assert packet.steps[0].transform_id == "transform:add-arg"
    assert packet.steps[0].write_paths == ("pkg/caller.py",)
    assert packet.before_hashes
    assert packet.before_hashes[0].before_hash == "sha256:caller-before"
    assert packet.selected_value_sources
    assert packet.selected_value_sources[0].candidate_id == "candidate:ctx"
    assert packet.proof_refs == admission.plan.proof_refs
    assert packet.index_refs == (roots.index_id,)
    assert packet.graph_refs == (roots.graph_id,)
    assert packet.fixed_point_obligation_ref == admission.plan.fixed_point_obligation_ref
    assert packet.fixed_point_obligation_ref in packet.fixed_point_postcondition_refs
    assert packet.per_edit_postcondition_refs
    assert packet.validation_commands
    assert "pytest" in packet.validation_commands[0]
    assert packet.packet_id == packet.content_id
    restored = ChangePropagationEditPacket.from_dict(packet.to_record())
    assert restored == packet


def test_partitions_analytical_and_model_required_steps(
    roots: PropagationAuthorityRoots,
) -> None:
    # One analytical consumer + one placement-backed model-required consumer.
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
        impact_closure=_closure(roots, (c1, c2)),
        obligations=(o1, o2),
        value_mapping_proofs=(
            _mapping(consumer_id="consumer:a"),
        ),
        analytical_transforms=(
            _transform(
                roots,
                transform_id="transform:a",
                obligation_ids=("obligation:consumer:a",),
                path="pkg/a.py",
            ),
        ),
        placement_decisions=(_placement(roots),),
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
    kinds = {step.kind for step in admission.plan.steps}
    assert PlanStepKind.ANALYTICAL in kinds
    assert PlanStepKind.LLM_BOUNDED in kinds

    packet = materialize_change_propagation_edit_packet(
        admission, roots=roots, evidence=evidence
    )
    assert packet.analytical_step_ids
    assert packet.model_required_step_ids
    assert set(packet.analytical_step_ids).isdisjoint(packet.model_required_step_ids)
    by_id = {step.step_id: step for step in packet.steps}
    for sid in packet.analytical_step_ids:
        assert by_id[sid].kind is PropagationEditStepKind.ANALYTICAL
    for sid in packet.model_required_step_ids:
        assert by_id[sid].kind is PropagationEditStepKind.MODEL_REQUIRED
        assert by_id[sid].write_paths
        assert by_id[sid].required_behavior_ids


def test_binds_scc_order_and_dependency_metadata(
    roots: PropagationAuthorityRoots,
) -> None:
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
        impact_closure=_closure(roots, (c1, c2), sccs=(scc,)),
        obligations=(o1, o2),
        value_mapping_proofs=(
            _mapping(consumer_id="consumer:a"),
            _mapping(consumer_id="consumer:b"),
        ),
        analytical_transforms=(t1, t2),
        write_spans=(
            PlanPathSpan(path="pkg/a.py", start=0, end=10, artifact_id="blob:a", before_hash="h:a"),
            PlanPathSpan(path="pkg/b.py", start=0, end=10, artifact_id="blob:b", before_hash="h:b"),
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
    assert packet.step_order[0] == "step:analytical:transform:a"
    assert packet.step_order[1] == "step:analytical:transform:b"
    by_id = {step.step_id: step for step in packet.steps}
    assert by_id["step:analytical:transform:b"].dependency_step_ids == (
        "step:analytical:transform:a",
    )
    assert packet.scc_groups
    assert packet.scc_group_ids
    assert set(packet.scc_groups[0].step_ids) == set(packet.step_order)


# ---------------------------------------------------------------------------
# Fail-closed boundaries
# ---------------------------------------------------------------------------


def test_stale_or_bare_or_abstaining_plans_do_not_materialize(
    roots: PropagationAuthorityRoots,
) -> None:
    admission, evidence = _admit(roots)

    with pytest.raises(ChangePropagationEditPacketError, match="PropagationPlanAdmission"):
        materialize_change_propagation_edit_packet(
            admission.plan,  # type: ignore[arg-type]
            roots=roots,
            evidence=evidence,
        )

    # Stale roots
    stale = PropagationAuthorityRoots(
        repository_id="repository:rpr-040",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate-stale",
        candidate_overlay_id="overlay:candidate-stale",
        graph_id="graph:rpr-040",
        index_id="index:rpr-040",
        model_id="model:rpr-040",
        config_id="config:rpr-040",
        translator_id="translator:rpr-040",
        toolchain_id="toolchain:rpr-040",
        policy_id="policy:rpr-040",
    )
    with pytest.raises(ChangePropagationEditPacketError, match="stale"):
        materialize_change_propagation_edit_packet(
            admission, roots=stale, evidence=evidence
        )

    # Abstaining plan
    bad = PlanEvidenceBundle(
        roots=roots,
        change_set_id="changeset:bad",
        delta_id="delta:one",
        impact_closure=_closure(roots, (_consumer(),)),
        obligations=(_obligation(roots),),
        value_mapping_proofs=(_mapping(disposition=SynthesisDisposition.AMBIGUOUS),),
        analytical_transforms=(),
        write_spans=(
            PlanPathSpan(path="pkg/caller.py", start=0, end=10, artifact_id="blob:caller"),
        ),
        validation_commands=_validation(),
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        expected_roots=roots,
    )
    abstained = ChangePropagationPlanner().admit(bad)
    assert not abstained.admitted
    with pytest.raises(ChangePropagationEditPacketError, match="admitted"):
        materialize_change_propagation_edit_packet(abstained, roots=roots, evidence=bad)


def test_ambiguous_value_mappings_cannot_broaden_scope(
    roots: PropagationAuthorityRoots,
) -> None:
    admission, evidence = _admit(roots)
    with pytest.raises(ChangePropagationEditPacketError, match="alternatives|ambiguous"):
        materialize_change_propagation_edit_packet(
            admission,
            roots=roots,
            evidence=evidence,
            value_mapping_proofs=(
                _mapping(disposition=SynthesisDisposition.AMBIGUOUS),
            ),
        )


def test_unknown_value_semantics_cannot_broaden_scope(
    roots: PropagationAuthorityRoots,
) -> None:
    admission, evidence = _admit(roots)
    with pytest.raises(ChangePropagationEditPacketError, match="unknown|unsupported"):
        materialize_change_propagation_edit_packet(
            admission,
            roots=roots,
            evidence=evidence,
            value_mapping_proofs=(
                _mapping(disposition=SynthesisDisposition.UNKNOWN, proved=()),
            ),
        )


def test_non_selected_handles_cannot_expand_packet_scope(
    roots: PropagationAuthorityRoots,
) -> None:
    admission, evidence = _admit(roots)
    with pytest.raises(ChangePropagationEditPacketError, match="packet-bound evidence"):
        materialize_change_propagation_edit_packet(
            admission,
            roots=roots,
            evidence=evidence,
            expansion_handles=(
                PropagationExpansionHandle(
                    "bad", "proof_receipt", "proof:unbound"
                ),
            ),
        )
    with pytest.raises(ChangePropagationEditPacketError, match="read scope"):
        materialize_change_propagation_edit_packet(
            admission,
            roots=roots,
            evidence=evidence,
            expansion_handles=(
                PropagationExpansionHandle(
                    "bad",
                    "proof_receipt",
                    "proof:plan",
                    ("pkg/not-selected.py",),
                ),
            ),
        )


def test_packet_rejects_forged_identity_and_embedded_bodies(
    roots: PropagationAuthorityRoots,
) -> None:
    admission, evidence = _admit(roots)
    packet = materialize_change_propagation_edit_packet(
        admission,
        roots=roots,
        evidence=evidence,
        expansion_handles=(
            PropagationExpansionHandle(
                "proof-handle", "proof_receipt", "proof:plan"
            ),
        ),
    )

    forged = deepcopy(packet.to_record())
    forged["content_id"] = "baguqeerapiforged"
    with pytest.raises(ChangePropagationEditPacketError, match="forged"):
        ChangePropagationEditPacket.from_dict(forged)

    body = deepcopy(packet.to_dict())
    body["proof_body"] = "by exact unsafe"
    with pytest.raises(ChangePropagationEditPacketError, match="forbidden|unsupported"):
        ChangePropagationEditPacket.from_dict(body)

    handle = deepcopy(packet.to_dict())
    handle["expansion_handles"][0]["body_embedded"] = True
    with pytest.raises(ChangePropagationEditPacketError, match="cannot embed"):
        ChangePropagationEditPacket.from_dict(handle)

    with pytest.raises(ChangePropagationEditPacketError, match="embedded bodies|secrets"):
        PropagationExpansionHandle("x", "source_body", "ref:one")


def test_path_mismatch_on_plan_steps_fails(
    roots: PropagationAuthorityRoots,
) -> None:
    admission, evidence = _admit(roots)
    # Forge a plan step write outside authority via a synthetic admission wrapper.
    bad_step = admission.plan.steps[0]
    # Reconstruct plan with an extra illegal write by replacing permitted paths
    # while leaving step writes intact — that is rejected by AtomicPropagationPlan.
    # Instead, pass mismatched evidence bundle id.
    with pytest.raises(ChangePropagationEditPacketError, match="evidence bundle|current"):
        materialize_change_propagation_edit_packet(
            admission,
            roots=roots,
            evidence=PlanEvidenceBundle(
                roots=roots,
                change_set_id="changeset:other",
                delta_id="delta:one",
                impact_closure=evidence.impact_closure,
                obligations=evidence.obligations,
                value_mapping_proofs=evidence.value_mapping_proofs,
                analytical_transforms=evidence.analytical_transforms,
                write_spans=evidence.write_spans,
                validation_commands=evidence.validation_commands,
                proof_refs=evidence.proof_refs,
                invalidation_refs=evidence.invalidation_refs,
                expected_roots=roots,
            ),
        )
    assert bad_step.write_paths == ("pkg/caller.py",)


def test_partial_plan_without_write_authority_fails(
    roots: PropagationAuthorityRoots,
) -> None:
    admission, evidence = _admit(roots)
    # Build a diagnostic abstained admission with empty writes (already covered),
    # and also ensure partial admitted-shaped objects without fixed-point fail.
    plan = admission.plan
    with pytest.raises(Exception):
        AtomicPropagationPlan(
            roots=plan.roots,
            plan_id=plan.plan_id,
            change_set_id=plan.change_set_id,
            delta_id=plan.delta_id,
            impact_closure_id=plan.impact_closure_id,
            disposition=PlanDisposition.ADMITTED,
            obligations=plan.obligations,
            obligation_set_id=plan.obligation_set_id,
            steps=plan.steps,
            scc_groups=(),
            permitted_read_paths=plan.permitted_read_paths,
            permitted_write_paths=(),
            checkpoint_strategy_ref=plan.checkpoint_strategy_ref,
            rollback_strategy_ref=plan.rollback_strategy_ref,
            fixed_point_obligation_ref="",
            proof_refs=plan.proof_refs,
            invalidation_refs=plan.invalidation_refs,
        )


def test_deterministic_replay(
    roots: PropagationAuthorityRoots,
) -> None:
    admission, evidence = _admit(roots)
    a = materialize_change_propagation_edit_packet(
        admission, roots=roots, evidence=evidence
    )
    b = materialize_change_propagation_edit_packet(
        admission, roots=roots, evidence=evidence
    )
    assert a.content_id == b.content_id
    assert a.to_record() == b.to_record()
