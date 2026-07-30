"""Fail-closed coverage for atomic transitive repair plan admission (RPR-039)."""

from __future__ import annotations

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
    obligation_set_identity,
)
from ipfs_accelerate_py.agent_supervisor.planning.change_propagation_plan import (
    CHANGE_PROPAGATION_PLANNER_INTERFACE,
    PRODUCER_ID,
    ChangePropagationPlanError,
    ChangePropagationPlanner,
    PlanEvidenceBundle,
    PlanPathSpan,
    PlanRejectionReason,
    PlanResourceBounds,
    PlanValidationCommand,
    PropagationPlanAdmission,
    admit_change_propagation_plan,
    plan_set_identity,
)
from ipfs_accelerate_py.agent_supervisor.planning.support_behavior_placement import (
    SupportPlacementAction,
    SupportPlacementDecision,
    SupportPlacementDisposition,
)
from ipfs_accelerate_py.agent_supervisor.proof.missing_input_synthesis import (
    SynthesisDisposition,
    ValueMappingProof,
)


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


@pytest.fixture
def roots() -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id="repository:rpr-039",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:rpr-039",
        index_id="index:rpr-039",
        model_id="model:rpr-039",
        config_id="config:rpr-039",
        translator_id="translator:rpr-039",
        toolchain_id="toolchain:rpr-039",
        policy_id="policy:rpr-039",
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
    mandatory: bool = True,
    depth: int = 1,
) -> ImpactConsumer:
    return ImpactConsumer(
        consumer_id=consumer_id,
        node=_node(path=path, symbol=f"symbol:{consumer_id}"),
        depth=depth,
        mandatory=mandatory,
        edge_refs=(f"edge:{consumer_id}",),
    )


def _closure(
    roots: PropagationAuthorityRoots,
    consumers: tuple[ImpactConsumer, ...],
    *,
    completeness: ImpactCompleteness = ImpactCompleteness.COMPLETE,
    sccs: tuple[ImpactSCC, ...] = (),
    frontier_node_ids: tuple[str, ...] = (),
    frontier_edge_ids: tuple[str, ...] = (),
) -> ImpactClosureReceipt:
    return ImpactClosureReceipt(
        roots=roots,
        delta_id="delta:one",
        completeness=completeness,
        consumers=consumers,
        sccs=sccs,
        frontier_node_ids=frontier_node_ids,
        frontier_edge_ids=frontier_edge_ids,
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
        expression_ref="expr:ctx" if disposition is SynthesisDisposition.UNIQUE_PROVED else "",
        type_ref="type:Context",
        repository_id="repository:rpr-039",
        tree_id="tree:candidate",
        toolchain_id="toolchain:rpr-039",
        policy_id="policy:rpr-039",
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


def _placement(
    roots: PropagationAuthorityRoots,
    *,
    behavior_id: str = "behavior:SupportContext",
    path: str = "pkg/support/context.py",
    candidate_id: str = "candidate:owner",
    disposition: SupportPlacementDisposition = SupportPlacementDisposition.ADMITTED,
) -> SupportPlacementDecision:
    admitted = disposition is SupportPlacementDisposition.ADMITTED
    return SupportPlacementDecision(
        disposition=disposition,
        roots=roots,
        behavior_id=behavior_id,
        candidate_set_id="placement-set:one",
        selected_candidate_id=candidate_id if admitted else "",
        action=SupportPlacementAction.PLACE_NEW if admitted else SupportPlacementAction.NONE,
        target_path=path if admitted else "",
        placement_paths=(path,) if admitted else (),
        reason_codes=("owner_unique",) if admitted else ("ambiguous_site",),
        proof_receipt_ids=("proof:placement",) if admitted else (),
        evidence_refs=("evidence:arch",),
        eligible_candidate_ids=(candidate_id,) if admitted else (),
        margin=2 if admitted else None,
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
        read_spans=(PlanPathSpan(path="pkg/caller.py", start=0, end=40, artifact_id="blob:caller"),),
        write_spans=(PlanPathSpan(path="pkg/caller.py", start=10, end=30, artifact_id="blob:caller"),),
        validation_commands=_validation(),
        resource_bounds=PlanResourceBounds(),
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        candidate_set_id="candidate-set:one",
        expected_roots=roots,
    )


# ---------------------------------------------------------------------------
# Interface / happy path
# ---------------------------------------------------------------------------


def test_interface_constant() -> None:
    assert CHANGE_PROPAGATION_PLANNER_INTERFACE == "ChangePropagationPlanner@1"
    assert ChangePropagationPlanner.INTERFACE == CHANGE_PROPAGATION_PLANNER_INTERFACE
    assert PRODUCER_ID == "change-propagation-plan@1"


def test_admit_happy_path_returns_canonical_atomic_plan(
    roots: PropagationAuthorityRoots,
) -> None:
    evidence = _happy_bundle(roots)
    admission = ChangePropagationPlanner().admit(evidence)

    assert admission.admitted
    assert isinstance(admission, PropagationPlanAdmission)
    assert isinstance(admission.plan, AtomicPropagationPlan)
    assert admission.plan.SCHEMA.endswith("atomic-propagation-plan@1")
    assert admission.disposition is PlanDisposition.ADMITTED
    assert admission.plan.disposition is PlanDisposition.ADMITTED
    assert admission.reason_codes == ()
    assert len(admission.plan.steps) == 1
    step = admission.plan.steps[0]
    assert step.kind is PlanStepKind.ANALYTICAL
    assert step.write_paths == ("pkg/caller.py",)
    assert step.transform_id == "transform:add-arg"
    assert admission.plan.permitted_write_paths == ("pkg/caller.py",)
    assert admission.plan.checkpoint_strategy_ref
    assert admission.plan.rollback_strategy_ref
    assert admission.plan.fixed_point_obligation_ref
    assert admission.plan.proof_refs
    assert admission.validation_command_ids == ("validate:pytest",)
    assert admission.permitted_write_spans
    assert admission.permitted_write_spans[0].path == "pkg/caller.py"
    # Round-trip canonical plan.
    restored = AtomicPropagationPlan.from_dict(admission.plan.to_record())
    assert restored == admission.plan


def test_module_entry_point_matches_planner(roots: PropagationAuthorityRoots) -> None:
    evidence = _happy_bundle(roots)
    a = admit_change_propagation_plan(evidence)
    b = ChangePropagationPlanner().admit(evidence)
    assert a.content_id == b.content_id
    assert a.plan.content_id == b.plan.content_id


def test_deterministic_replay_stable(roots: PropagationAuthorityRoots) -> None:
    evidence = _happy_bundle(roots)
    first = ChangePropagationPlanner().admit(evidence)
    second = ChangePropagationPlanner().admit(evidence)
    assert first.content_id == second.content_id
    assert first.plan.content_id == second.plan.content_id
    assert first.step_order == second.step_order
    assert first.plan.to_record() == second.plan.to_record()


def test_plan_identity_binds_roots_sets_and_toolchain(
    roots: PropagationAuthorityRoots,
) -> None:
    evidence = _happy_bundle(roots)
    admission = ChangePropagationPlanner().admit(evidence)
    plan = admission.plan
    assert plan.roots == roots
    assert plan.change_set_id == "changeset:one"
    assert plan.delta_id == "delta:one"
    assert plan.impact_closure_id == evidence.impact_closure.content_id
    assert plan.obligation_set_id == obligation_set_identity(evidence.obligations)
    # Graph / index / model / translator / toolchain / policy bound via roots.
    assert plan.roots.graph_id == "graph:rpr-039"
    assert plan.roots.index_id == "index:rpr-039"
    assert plan.roots.model_id == "model:rpr-039"
    assert plan.roots.translator_id == "translator:rpr-039"
    assert plan.roots.toolchain_id == "toolchain:rpr-039"
    assert plan.roots.policy_id == "policy:rpr-039"
    assert plan.roots.base_tree_id != plan.roots.candidate_tree_id


# ---------------------------------------------------------------------------
# Multi-consumer / SCC / dependency DAG
# ---------------------------------------------------------------------------


def test_scc_transaction_groups_and_dependency_order(
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
            _mapping(
                requirement_id="missing:context",
                consumer_id="consumer:a",
            ),
            _mapping(
                requirement_id="missing:context",
                consumer_id="consumer:b",
            ),
        ),
        analytical_transforms=(t1, t2),
        write_spans=(
            PlanPathSpan(path="pkg/a.py", start=0, end=10, artifact_id="blob:a"),
            PlanPathSpan(path="pkg/b.py", start=0, end=10, artifact_id="blob:b"),
        ),
        validation_commands=_validation(),
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        expected_roots=roots,
    )
    admission = ChangePropagationPlanner().admit(evidence)
    assert admission.admitted
    assert len(admission.plan.steps) == 2
    # Execution order is carried on the admission; plan.steps may be
    # content-ordered by the canonical record.
    assert admission.step_order[0] == "step:analytical:transform:a"
    assert admission.step_order[1] == "step:analytical:transform:b"
    by_id = {step.step_id: step for step in admission.plan.steps}
    assert by_id["step:analytical:transform:b"].dependency_step_ids == (
        "step:analytical:transform:a",
    )
    assert by_id["step:analytical:transform:a"].dependency_step_ids == ()
    assert len(admission.plan.scc_groups) == 1
    group = admission.plan.scc_groups[0]
    assert group.scc_id == "scc:cycle"
    assert set(group.consumer_ids) == {"consumer:a", "consumer:b"}
    assert set(group.step_ids) == {
        "step:analytical:transform:a",
        "step:analytical:transform:b",
    }


def test_each_mandatory_consumer_has_exactly_one_disposition(
    roots: PropagationAuthorityRoots,
) -> None:
    evidence = _happy_bundle(roots)
    admission = ChangePropagationPlanner().admit(evidence)
    consumers = [item.consumer_id for item in admission.plan.obligations]
    assert len(consumers) == len(set(consumers))
    assert all(
        isinstance(item.disposition, ConsumerDisposition)
        for item in admission.plan.obligations
    )


def test_writes_derive_only_from_authority(roots: PropagationAuthorityRoots) -> None:
    evidence = _happy_bundle(roots)
    admission = ChangePropagationPlanner().admit(evidence)
    step_writes = {path for step in admission.plan.steps for path in step.write_paths}
    assert step_writes.issubset(set(admission.plan.permitted_write_paths))
    assert step_writes.issubset({span.path for span in admission.permitted_write_spans})


# ---------------------------------------------------------------------------
# Abstention cases
# ---------------------------------------------------------------------------


def test_abstain_on_stale_roots(roots: PropagationAuthorityRoots) -> None:
    evidence = _happy_bundle(roots)
    stale = PropagationAuthorityRoots(
        repository_id="repository:rpr-039",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:stale",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:rpr-039",
        index_id="index:rpr-039",
        model_id="model:rpr-039",
        config_id="config:rpr-039",
        translator_id="translator:rpr-039",
        toolchain_id="toolchain:rpr-039",
        policy_id="policy:rpr-039",
    )
    # Rebuild bundle with mismatched expected roots.
    evidence = PlanEvidenceBundle(
        roots=roots,
        change_set_id=evidence.change_set_id,
        delta_id=evidence.delta_id,
        impact_closure=evidence.impact_closure,
        obligations=evidence.obligations,
        value_mapping_proofs=evidence.value_mapping_proofs,
        analytical_transforms=evidence.analytical_transforms,
        write_spans=evidence.write_spans,
        validation_commands=evidence.validation_commands,
        proof_refs=evidence.proof_refs,
        invalidation_refs=evidence.invalidation_refs,
        expected_roots=stale,
    )
    admission = ChangePropagationPlanner().admit(evidence)
    assert not admission.admitted
    assert PlanRejectionReason.STALE_ROOTS.value in admission.reason_codes
    assert admission.plan.disposition is PlanDisposition.ABSTAINED
    assert admission.plan.permitted_write_paths == ()


def test_abstain_on_incomplete_closure_frontier(roots: PropagationAuthorityRoots) -> None:
    consumer = _consumer()
    obligation = _obligation(roots)
    evidence = PlanEvidenceBundle(
        roots=roots,
        change_set_id="changeset:one",
        delta_id="delta:one",
        impact_closure=_closure(
            roots,
            (consumer,),
            completeness=ImpactCompleteness.PARTIAL_WITH_FRONTIER,
            frontier_node_ids=("node:unknown",),
            frontier_edge_ids=("edge:unknown",),
        ),
        obligations=(obligation,),
        value_mapping_proofs=(_mapping(),),
        analytical_transforms=(_transform(roots),),
        write_spans=(
            PlanPathSpan(path="pkg/caller.py", start=0, end=10, artifact_id="blob:caller"),
        ),
        validation_commands=_validation(),
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        expected_roots=roots,
    )
    admission = ChangePropagationPlanner().admit(evidence)
    assert not admission.admitted
    assert (
        PlanRejectionReason.UNRESOLVED_REQUIRED_FRONTIER.value in admission.reason_codes
        or PlanRejectionReason.INCOMPLETE_CLOSURE.value in admission.reason_codes
    )


def test_abstain_on_omitted_consumer(roots: PropagationAuthorityRoots) -> None:
    c1 = _consumer("consumer:one")
    c2 = _consumer("consumer:two", "pkg/other.py")
    evidence = PlanEvidenceBundle(
        roots=roots,
        change_set_id="changeset:one",
        delta_id="delta:one",
        impact_closure=_closure(roots, (c1, c2)),
        obligations=(_obligation(roots, consumer_id="consumer:one"),),
        value_mapping_proofs=(_mapping(),),
        analytical_transforms=(_transform(roots),),
        write_spans=(
            PlanPathSpan(path="pkg/caller.py", start=0, end=10, artifact_id="blob:caller"),
        ),
        validation_commands=_validation(),
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        expected_roots=roots,
    )
    admission = ChangePropagationPlanner().admit(evidence)
    assert not admission.admitted
    assert PlanRejectionReason.OMISSION.value in admission.reason_codes


def test_abstain_on_duplicate_consumer_disposition(
    roots: PropagationAuthorityRoots,
) -> None:
    consumer = _consumer()
    o1 = _obligation(roots, consumer_id="consumer:one")
    # Second obligation same consumer_id via different construction is blocked by
    # unique obligation_id; forge duplicate consumer with distinct obligation ids.
    o2 = ConsumerMigrationObligation(
        roots=roots,
        obligation_id="obligation:consumer:one:dup",
        consumer_id="consumer:one",
        delta_id="delta:one",
        disposition=ConsumerDisposition.COMPATIBLE,
        clause_ids=("clause:param-add",),
        node=_node(),
        proof_refs=("proof:obligation",),
        invalidation_refs=("tree:candidate",),
    )
    evidence = PlanEvidenceBundle(
        roots=roots,
        change_set_id="changeset:one",
        delta_id="delta:one",
        impact_closure=_closure(roots, (consumer,)),
        obligations=(o1, o2),
        value_mapping_proofs=(_mapping(),),
        analytical_transforms=(_transform(roots),),
        write_spans=(
            PlanPathSpan(path="pkg/caller.py", start=0, end=10, artifact_id="blob:caller"),
        ),
        validation_commands=_validation(),
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        expected_roots=roots,
    )
    admission = ChangePropagationPlanner().admit(evidence)
    assert not admission.admitted
    assert PlanRejectionReason.DUPLICATE_DISPOSITION.value in admission.reason_codes


def test_abstain_on_competing_value_mapping(roots: PropagationAuthorityRoots) -> None:
    evidence = PlanEvidenceBundle(
        roots=roots,
        change_set_id="changeset:one",
        delta_id="delta:one",
        impact_closure=_closure(roots, (_consumer(),)),
        obligations=(_obligation(roots),),
        value_mapping_proofs=(
            _mapping(disposition=SynthesisDisposition.AMBIGUOUS),
        ),
        analytical_transforms=(_transform(roots),),
        write_spans=(
            PlanPathSpan(path="pkg/caller.py", start=0, end=10, artifact_id="blob:caller"),
        ),
        validation_commands=_validation(),
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        expected_roots=roots,
    )
    admission = ChangePropagationPlanner().admit(evidence)
    assert not admission.admitted
    assert PlanRejectionReason.COMPETING_MAPPING.value in admission.reason_codes


def test_abstain_on_failed_proof(roots: PropagationAuthorityRoots) -> None:
    evidence = PlanEvidenceBundle(
        roots=roots,
        change_set_id="changeset:one",
        delta_id="delta:one",
        impact_closure=_closure(roots, (_consumer(),)),
        obligations=(_obligation(roots),),
        value_mapping_proofs=(
            _mapping(disposition=SynthesisDisposition.REFUTED),
        ),
        analytical_transforms=(_transform(roots),),
        write_spans=(
            PlanPathSpan(path="pkg/caller.py", start=0, end=10, artifact_id="blob:caller"),
        ),
        validation_commands=_validation(),
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        expected_roots=roots,
    )
    admission = ChangePropagationPlanner().admit(evidence)
    assert not admission.admitted
    assert PlanRejectionReason.FAILED_PROOF.value in admission.reason_codes


def test_abstain_on_competing_placement_sites(roots: PropagationAuthorityRoots) -> None:
    o = _obligation(
        roots,
        missing=(),
        behavior=("behavior:SupportContext",),
    )
    p1 = _placement(roots, path="pkg/support/a.py", candidate_id="candidate:a")
    p2 = _placement(roots, path="pkg/support/b.py", candidate_id="candidate:b")
    evidence = PlanEvidenceBundle(
        roots=roots,
        change_set_id="changeset:one",
        delta_id="delta:one",
        impact_closure=_closure(roots, (_consumer(),)),
        obligations=(o,),
        value_mapping_proofs=(),
        analytical_transforms=(),
        placement_decisions=(p1, p2),
        write_spans=(
            PlanPathSpan(path="pkg/support/a.py", start=0, end=10, artifact_id="blob:a"),
            PlanPathSpan(path="pkg/support/b.py", start=0, end=10, artifact_id="blob:b"),
        ),
        validation_commands=_validation(),
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        expected_roots=roots,
    )
    admission = ChangePropagationPlanner().admit(evidence)
    assert not admission.admitted
    assert PlanRejectionReason.COMPETING_SITE.value in admission.reason_codes


def test_abstain_on_equally_valid_transforms(roots: PropagationAuthorityRoots) -> None:
    t1 = _transform(
        roots,
        transform_id="transform:one",
        path="pkg/caller.py",
    )
    t2 = _transform(
        roots,
        transform_id="transform:two",
        path="pkg/other.py",
    )
    evidence = PlanEvidenceBundle(
        roots=roots,
        change_set_id="changeset:one",
        delta_id="delta:one",
        impact_closure=_closure(roots, (_consumer(),)),
        obligations=(_obligation(roots),),
        value_mapping_proofs=(_mapping(),),
        analytical_transforms=(t1, t2),
        write_spans=(
            PlanPathSpan(path="pkg/caller.py", start=0, end=10, artifact_id="blob:c"),
            PlanPathSpan(path="pkg/other.py", start=0, end=10, artifact_id="blob:o"),
        ),
        validation_commands=_validation(),
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        expected_roots=roots,
    )
    admission = ChangePropagationPlanner().admit(evidence)
    assert not admission.admitted
    assert PlanRejectionReason.EQUALLY_VALID_PLANS.value in admission.reason_codes


def test_abstain_on_forbidden_path(roots: PropagationAuthorityRoots) -> None:
    path = "vendor/third_party/lib.py"
    evidence = PlanEvidenceBundle(
        roots=roots,
        change_set_id="changeset:one",
        delta_id="delta:one",
        impact_closure=_closure(roots, (_consumer(path=path),)),
        obligations=(_obligation(roots, path=path),),
        value_mapping_proofs=(_mapping(),),
        analytical_transforms=(_transform(roots, path=path),),
        write_spans=(PlanPathSpan(path=path, start=0, end=10, artifact_id="blob:v"),),
        validation_commands=_validation(),
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        expected_roots=roots,
    )
    admission = ChangePropagationPlanner().admit(evidence)
    assert not admission.admitted
    assert PlanRejectionReason.FORBIDDEN_PATH.value in admission.reason_codes


def test_abstain_on_missing_write_authority_span(
    roots: PropagationAuthorityRoots,
) -> None:
    # Transform targets pkg/caller.py but write spans only list another path.
    evidence = PlanEvidenceBundle(
        roots=roots,
        change_set_id="changeset:one",
        delta_id="delta:one",
        impact_closure=_closure(roots, (_consumer(),)),
        obligations=(_obligation(roots),),
        value_mapping_proofs=(_mapping(),),
        analytical_transforms=(_transform(roots),),
        write_spans=(
            PlanPathSpan(path="pkg/unrelated.py", start=0, end=10, artifact_id="blob:u"),
        ),
        validation_commands=_validation(),
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        expected_roots=roots,
    )
    admission = ChangePropagationPlanner().admit(evidence)
    assert not admission.admitted
    assert PlanRejectionReason.MISSING_WRITE_AUTHORITY.value in admission.reason_codes


def test_abstain_on_invalid_validation(roots: PropagationAuthorityRoots) -> None:
    with pytest.raises(ChangePropagationPlanError, match="metacharacters"):
        PlanValidationCommand(
            command_id="bad",
            argv=("python", "-c", "import os; os.system('x')"),
        )

    evidence = PlanEvidenceBundle(
        roots=roots,
        change_set_id="changeset:one",
        delta_id="delta:one",
        impact_closure=_closure(roots, (_consumer(),)),
        obligations=(_obligation(roots),),
        value_mapping_proofs=(_mapping(),),
        analytical_transforms=(_transform(roots),),
        write_spans=(
            PlanPathSpan(path="pkg/caller.py", start=0, end=10, artifact_id="blob:c"),
        ),
        validation_commands=(),  # missing required validation
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        expected_roots=roots,
    )
    admission = ChangePropagationPlanner().admit(evidence)
    assert not admission.admitted
    assert PlanRejectionReason.INVALID_VALIDATION.value in admission.reason_codes


def test_abstain_on_cycle_outside_scc(roots: PropagationAuthorityRoots) -> None:
    c1 = _consumer("consumer:a", "pkg/a.py")
    c2 = _consumer("consumer:b", "pkg/b.py")
    # No multi-member SCC grouping the cycle.
    o1 = _obligation(roots, consumer_id="consumer:a", path="pkg/a.py")
    o2 = _obligation(roots, consumer_id="consumer:b", path="pkg/b.py")
    t1 = _transform(
        roots,
        transform_id="transform:a",
        obligation_ids=("obligation:consumer:a",),
        path="pkg/a.py",
        deps=("transform:b",),
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
        change_set_id="changeset:cycle",
        delta_id="delta:one",
        impact_closure=_closure(roots, (c1, c2)),
        obligations=(o1, o2),
        value_mapping_proofs=(
            _mapping(consumer_id="consumer:a"),
            _mapping(consumer_id="consumer:b"),
        ),
        analytical_transforms=(t1, t2),
        write_spans=(
            PlanPathSpan(path="pkg/a.py", start=0, end=10, artifact_id="blob:a"),
            PlanPathSpan(path="pkg/b.py", start=0, end=10, artifact_id="blob:b"),
        ),
        validation_commands=_validation(),
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        expected_roots=roots,
    )
    admission = ChangePropagationPlanner().admit(evidence)
    assert not admission.admitted
    assert PlanRejectionReason.CYCLE_OUTSIDE_SCC.value in admission.reason_codes


def test_abstain_on_resource_bound(roots: PropagationAuthorityRoots) -> None:
    # max_steps=0 rejected at construction
    with pytest.raises(ChangePropagationPlanError, match="at least 1"):
        PlanResourceBounds(max_steps=0)


def test_abstain_on_resource_bound_too_small(roots: PropagationAuthorityRoots) -> None:
    evidence = PlanEvidenceBundle(
        roots=roots,
        change_set_id="changeset:one",
        delta_id="delta:one",
        impact_closure=_closure(roots, (_consumer(),)),
        obligations=(_obligation(roots),),
        value_mapping_proofs=(_mapping(),),
        analytical_transforms=(_transform(roots),),
        write_spans=(
            PlanPathSpan(path="pkg/caller.py", start=0, end=10, artifact_id="blob:c"),
        ),
        validation_commands=_validation(),
        resource_bounds=PlanResourceBounds(max_steps=1, max_write_paths=0),
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        expected_roots=roots,
    )
    admission = ChangePropagationPlanner().admit(evidence)
    assert not admission.admitted
    assert PlanRejectionReason.RESOURCE_BOUND.value in admission.reason_codes


def test_placement_backed_llm_step(roots: PropagationAuthorityRoots) -> None:
    o = _obligation(
        roots,
        missing=(),
        behavior=("behavior:SupportContext",),
    )
    placement = _placement(roots)
    evidence = PlanEvidenceBundle(
        roots=roots,
        change_set_id="changeset:place",
        delta_id="delta:one",
        impact_closure=_closure(roots, (_consumer(),)),
        obligations=(o,),
        value_mapping_proofs=(),
        analytical_transforms=(),
        placement_decisions=(placement,),
        write_spans=(
            PlanPathSpan(
                path="pkg/support/context.py", start=0, end=20, artifact_id="blob:s"
            ),
        ),
        validation_commands=_validation(),
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        expected_roots=roots,
    )
    admission = ChangePropagationPlanner().admit(evidence)
    assert admission.admitted
    assert len(admission.plan.steps) == 1
    assert admission.plan.steps[0].kind is PlanStepKind.LLM_BOUNDED
    assert admission.plan.steps[0].write_paths == ("pkg/support/context.py",)


def test_admission_to_dict_and_plan_set_identity(
    roots: PropagationAuthorityRoots,
) -> None:
    evidence = _happy_bundle(roots)
    admission = ChangePropagationPlanner().admit(evidence)
    payload = admission.to_dict()
    assert payload["disposition"] == "admitted"
    assert payload["content_id"] == admission.content_id
    assert "plan" in payload

    identity = plan_set_identity((admission.plan,))
    assert isinstance(identity, str) and len(identity) > 8


def test_reject_non_bundle_evidence() -> None:
    with pytest.raises(ChangePropagationPlanError, match="PlanEvidenceBundle"):
        ChangePropagationPlanner().admit(object())  # type: ignore[arg-type]


def test_preconditions_postconditions_and_validation_on_steps(
    roots: PropagationAuthorityRoots,
) -> None:
    admission = ChangePropagationPlanner().admit(_happy_bundle(roots))
    step = admission.plan.steps[0]
    assert step.precondition_refs
    assert step.postcondition_refs
    assert step.validation_refs
    assert admission.resource_bounds_ref
