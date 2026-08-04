"""Contract tests for bounded change-propagation records (RPR-022)."""

from __future__ import annotations

import math

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    AnalyticalTransform,
    AtomicPropagationPlan,
    BehaviorEvidencePrecedence,
    BehaviorKind,
    ChangePropagationAuthorityError,
    ChangePropagationBoundsError,
    ChangePropagationError,
    ChangeSetKind,
    CompletionDisposition,
    ConsumerDisposition,
    ConsumerMigrationObligation,
    ContractClauseDelta,
    DeltaDisposition,
    DeltaKind,
    FixedPointReceipt,
    ForgedChangePropagationIdentityError,
    GraphEdgeKind,
    GraphEdgeRef,
    GraphNodeRef,
    GraphProvenance,
    ImpactClosureReceipt,
    ImpactCompleteness,
    ImpactConsumer,
    ImpactSCC,
    MissingInputRequirement,
    PlanDisposition,
    PlanStepKind,
    ProgramChangeSet,
    ProgramContractDelta,
    PropagationAuthorityRoots,
    PropagationCompletionReceipt,
    PropagationPlanStep,
    PropagationSCCGroup,
    PropagationTransaction,
    RequiredBehaviorContract,
    TransactionState,
    TransformDisposition,
    TransformKind,
    ValueCandidate,
    ValueCandidateDisposition,
    ValueCandidateKind,
    obligation_set_identity,
)


@pytest.fixture
def roots() -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id="repository:one",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:one",
        index_id="index:one",
        model_id="model:one",
        config_id="config:one",
        translator_id="translator:one",
        toolchain_id="toolchain:one",
        policy_id="policy:one",
    )


@pytest.fixture
def trusted_node() -> GraphNodeRef:
    return GraphNodeRef(
        node_id="node:caller",
        kind="function",
        path="pkg/caller.py",
        symbol_id="symbol:caller",
        artifact_id="blob:caller",
        provenance=GraphProvenance.TRUSTED,
        extractor_id="extractor:ast",
    )


def _clause() -> ContractClauseDelta:
    return ContractClauseDelta(
        clause_id="clause:param-add",
        kind=DeltaKind.PARAMETER_ADD,
        disposition=DeltaDisposition.BREAKING,
        subject_symbol_id="symbol:process",
        consumer_domain="domain:python-callers",
        before_contract_ref="contract:before",
        after_contract_ref="contract:after",
        reason="third argument required",
    )


def _obligation(
    roots: PropagationAuthorityRoots, node: GraphNodeRef, *, consumer_id: str = "consumer:one"
) -> ConsumerMigrationObligation:
    return ConsumerMigrationObligation(
        roots=roots,
        obligation_id=f"obligation:{consumer_id}",
        consumer_id=consumer_id,
        delta_id="delta:one",
        disposition=ConsumerDisposition.MIGRATE,
        clause_ids=("clause:param-add",),
        node=node,
        proof_refs=("proof:obligation",),
        missing_input_ids=("missing:context",),
        invalidation_refs=("tree:candidate",),
    )


def test_roots_bind_base_candidate_and_shared_identities(roots: PropagationAuthorityRoots) -> None:
    assert roots.base_tree_id == "tree:base"
    assert roots.candidate_tree_id == "tree:candidate"
    assert roots.base_overlay_id != roots.candidate_overlay_id
    assert roots.graph_id == "graph:one"
    assert roots.policy_id == "policy:one"
    assert roots.content_id.startswith("b")
    assert PropagationAuthorityRoots.from_dict(roots.to_record()) == roots


def test_roots_reject_identical_base_and_candidate() -> None:
    with pytest.raises(ChangePropagationAuthorityError, match="must differ"):
        PropagationAuthorityRoots(
            repository_id="repository:one",
            base_forest_id="forest:same",
            base_tree_id="tree:same",
            base_overlay_id="overlay:same",
            candidate_forest_id="forest:same",
            candidate_tree_id="tree:same",
            candidate_overlay_id="overlay:same",
            graph_id="graph:one",
            index_id="index:one",
            model_id="model:one",
            config_id="config:one",
            translator_id="translator:one",
            toolchain_id="toolchain:one",
            policy_id="policy:one",
        )


@pytest.mark.parametrize("bad", ["../escape.py", "/absolute.py", "."])
def test_paths_must_be_repository_relative(bad: str) -> None:
    with pytest.raises(ChangePropagationAuthorityError):
        GraphNodeRef(
            node_id="node:bad",
            kind="function",
            path=bad,
            symbol_id="symbol:bad",
            artifact_id="blob:bad",
            provenance=GraphProvenance.NOMINATED,
        )


def test_forged_identity_rejected(roots: PropagationAuthorityRoots) -> None:
    payload = roots.to_record()
    payload["content_id"] = "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    with pytest.raises(ForgedChangePropagationIdentityError):
        PropagationAuthorityRoots.from_dict(payload)


def test_source_bodies_and_unbounded_floats_rejected(
    roots: PropagationAuthorityRoots, trusted_node: GraphNodeRef
) -> None:
    # Top-level source-body field names are rejected as unsupported / body markers.
    with pytest.raises(ChangePropagationError, match="unsupported fields|source bodies"):
        ProgramChangeSet.from_dict(
            {
                "schema": ProgramChangeSet.SCHEMA,
                "contract_version": 1,
                "roots": roots.to_dict(),
                "kind": ChangeSetKind.REVIEWED_BASE_CANDIDATE.value,
                "producer_id": "producer:diff",
                "changed_paths": ["pkg/a.py"],
                "source_body": "def evil(): pass",
            }
        )
    # Nested body markers inside nested allowed mappings are also rejected.
    poisoned_roots = dict(roots.to_dict())
    poisoned_roots["source_body"] = "def evil(): pass"
    with pytest.raises(ChangePropagationError, match="source bodies"):
        ProgramChangeSet.from_dict(
            {
                "schema": ProgramChangeSet.SCHEMA,
                "contract_version": 1,
                "roots": poisoned_roots,
                "kind": ChangeSetKind.REVIEWED_BASE_CANDIDATE.value,
                "producer_id": "producer:diff",
                "changed_paths": ["pkg/a.py"],
            }
        )

    with pytest.raises(ChangePropagationError):
        GraphEdgeRef(
            edge_id="edge:one",
            kind=GraphEdgeKind.CALL,
            source_node_id="node:a",
            target_node_id="node:b",
            provenance=GraphProvenance.TRUSTED,
            extractor_id="extractor:ast",
            confidence_millipercent=math.inf,  # type: ignore[arg-type]
        )


def test_program_change_set_is_content_addressed(roots: PropagationAuthorityRoots) -> None:
    change_set = ProgramChangeSet(
        roots=roots,
        kind=ChangeSetKind.REVIEWED_BASE_CANDIDATE,
        producer_id="producer:normalized-diff",
        changed_paths=("pkg/process.py", "pkg/caller.py"),
        tombstone_paths=("pkg/legacy.py",),
        span_refs=("span:process",),
        evidence_refs=("evidence:diff",),
    )
    assert change_set.content_id.startswith("b")
    assert ProgramChangeSet.from_dict(change_set.to_record()) == change_set
    with pytest.raises(ChangePropagationError, match="disjoint"):
        ProgramChangeSet(
            roots=roots,
            kind=ChangeSetKind.WORKTREE_DIFF,
            producer_id="producer:x",
            changed_paths=("pkg/a.py",),
            tombstone_paths=("pkg/a.py",),
        )


def test_graph_refs_forbid_trusted_without_extractor_and_authority_promotion() -> None:
    with pytest.raises(ChangePropagationAuthorityError, match="extractor"):
        GraphNodeRef(
            node_id="node:x",
            kind="function",
            path="pkg/x.py",
            symbol_id="symbol:x",
            artifact_id="blob:x",
            provenance=GraphProvenance.TRUSTED,
        )

    with pytest.raises(ChangePropagationAuthorityError, match="full confidence"):
        GraphEdgeRef(
            edge_id="edge:nom",
            kind=GraphEdgeKind.CALL,
            source_node_id="node:a",
            target_node_id="node:b",
            provenance=GraphProvenance.NOMINATED,
            confidence_millipercent=100_000,
        )


def test_program_contract_delta_clauses(roots: PropagationAuthorityRoots) -> None:
    clause = _clause()
    delta = ProgramContractDelta(
        roots=roots,
        change_set_id="changeset:one",
        subject_symbol_id="symbol:process",
        before_contract_ref="contract:before",
        after_contract_ref="contract:after",
        clauses=(clause,),
        evidence_refs=("evidence:extract",),
    )
    assert delta.breaking_clauses == (clause,)
    assert ProgramContractDelta.from_dict(delta.to_record()) == delta

    with pytest.raises(ChangePropagationError, match="subject"):
        ProgramContractDelta(
            roots=roots,
            change_set_id="changeset:one",
            subject_symbol_id="symbol:other",
            before_contract_ref="contract:before",
            after_contract_ref="contract:after",
            clauses=(clause,),
        )


def test_impact_closure_frontier_state_machine(
    roots: PropagationAuthorityRoots, trusted_node: GraphNodeRef
) -> None:
    consumer = ImpactConsumer(
        consumer_id="consumer:one",
        node=trusted_node,
        depth=1,
        mandatory=True,
        edge_refs=("edge:call",),
    )
    scc = ImpactSCC(scc_id="scc:one", member_consumer_ids=("consumer:one",))
    complete = ImpactClosureReceipt(
        roots=roots,
        delta_id="delta:one",
        completeness=ImpactCompleteness.COMPLETE,
        consumers=(consumer,),
        sccs=(scc,),
        validation_refs=("validation:types",),
    )
    assert complete.completeness is ImpactCompleteness.COMPLETE
    assert ImpactClosureReceipt.from_dict(complete.to_record()) == complete

    with pytest.raises(ChangePropagationError, match="open frontier"):
        ImpactClosureReceipt(
            roots=roots,
            delta_id="delta:one",
            completeness=ImpactCompleteness.COMPLETE,
            consumers=(consumer,),
            frontier_node_ids=("node:dynamic",),
        )

    partial = ImpactClosureReceipt(
        roots=roots,
        delta_id="delta:one",
        completeness=ImpactCompleteness.PARTIAL_WITH_FRONTIER,
        consumers=(consumer,),
        frontier_node_ids=("node:plugin",),
        frontier_edge_ids=("edge:reflection",),
    )
    assert partial.frontier_node_ids == ("node:plugin",)

    with pytest.raises(ChangePropagationError, match="explicit frontier"):
        ImpactClosureReceipt(
            roots=roots,
            delta_id="delta:one",
            completeness=ImpactCompleteness.PARTIAL_WITH_FRONTIER,
            consumers=(consumer,),
        )


def test_consumer_obligation_and_missing_input(
    roots: PropagationAuthorityRoots, trusted_node: GraphNodeRef
) -> None:
    obligation = _obligation(roots, trusted_node)
    assert obligation.disposition is ConsumerDisposition.MIGRATE

    missing = MissingInputRequirement(
        roots=roots,
        requirement_id="missing:context",
        obligation_id=obligation.obligation_id,
        clause_id="clause:param-add",
        parameter_name="context",
        type_ref="type:Context",
        nullability="non_null",
        information_content_ref="info:request-context",
        construction_precondition_refs=("pre:available",),
        capability_refs=("cap:context.read",),
        propagation_depth_bound=8,
    )
    assert MissingInputRequirement.from_dict(missing.to_record()) == missing

    with pytest.raises(ChangePropagationError, match="compatible/excluded"):
        ConsumerMigrationObligation(
            roots=roots,
            obligation_id="obligation:compat",
            consumer_id="consumer:compat",
            delta_id="delta:one",
            disposition=ConsumerDisposition.COMPATIBLE,
            clause_ids=("clause:param-add",),
            node=trusted_node,
            missing_input_ids=("missing:x",),
        )

    with pytest.raises(ChangePropagationAuthorityError, match="frontier"):
        ConsumerMigrationObligation(
            roots=roots,
            obligation_id="obligation:front",
            consumer_id="consumer:front",
            delta_id="delta:one",
            disposition=ConsumerDisposition.FRONTIER,
            clause_ids=("clause:param-add",),
            node=trusted_node,
            proof_refs=("proof:forged",),
        )


def test_value_candidate_forbids_authority_promotion(
    roots: PropagationAuthorityRoots, trusted_node: GraphNodeRef
) -> None:
    with pytest.raises(ChangePropagationAuthorityError, match="semantic authority"):
        ValueCandidate(
            roots=roots,
            candidate_id="value:vec",
            requirement_id="missing:context",
            kind=ValueCandidateKind.VECTOR_NOMINATION,
            disposition=ValueCandidateDisposition.NOMINATED,
            source_node=trusted_node,
            expression_ref="expr:ctx",
            type_ref="type:Context",
            semantic_authority=True,
        )

    proved = ValueCandidate(
        roots=roots,
        candidate_id="value:local",
        requirement_id="missing:context",
        kind=ValueCandidateKind.PARAMETER,
        disposition=ValueCandidateDisposition.PROVED,
        source_node=trusted_node,
        expression_ref="expr:ctx",
        type_ref="type:Context",
        semantic_authority=True,
        proof_refs=("proof:value",),
    )
    assert ValueCandidate.from_dict(proved.to_record()) == proved

    with pytest.raises(ChangePropagationError, match="proof refs"):
        ValueCandidate(
            roots=roots,
            candidate_id="value:bad",
            requirement_id="missing:context",
            kind=ValueCandidateKind.LOCAL_NAME,
            disposition=ValueCandidateDisposition.PROVED,
            source_node=trusted_node,
            expression_ref="expr:ctx",
            type_ref="type:Context",
            semantic_authority=True,
        )


def test_required_behavior_rejects_implementation_promotion(
    roots: PropagationAuthorityRoots,
) -> None:
    behavior = RequiredBehaviorContract(
        roots=roots,
        behavior_id="behavior:context",
        kind=BehaviorKind.CLASS,
        subject_symbol_id="symbol:Context",
        evidence_precedence=BehaviorEvidencePrecedence.REVIEWED_IDL,
        field_refs=("field:trace_id",),
        constructor_refs=("ctor:Context",),
        proof_refs=("proof:behavior",),
    )
    assert RequiredBehaviorContract.from_dict(behavior.to_record()) == behavior

    with pytest.raises(ChangePropagationAuthorityError, match="promote"):
        RequiredBehaviorContract(
            roots=roots,
            behavior_id="behavior:bad",
            kind=BehaviorKind.METHOD,
            subject_symbol_id="symbol:x",
            evidence_precedence=BehaviorEvidencePrecedence.CALLER_POSTCONDITION,
            method_refs=("method:x",),
            implementation_hypothesis=True,
        )

    with pytest.raises(ChangePropagationAuthorityError, match="proof authority"):
        RequiredBehaviorContract(
            roots=roots,
            behavior_id="behavior:hyp",
            kind=BehaviorKind.METHOD,
            subject_symbol_id="symbol:x",
            evidence_precedence=BehaviorEvidencePrecedence.IMPLEMENTATION_HYPOTHESIS,
            method_refs=("method:x",),
            implementation_hypothesis=True,
            proof_refs=("proof:illegal",),
        )


def test_analytical_transform_path_authority(roots: PropagationAuthorityRoots) -> None:
    admitted = AnalyticalTransform(
        roots=roots,
        transform_id="transform:add-arg",
        kind=TransformKind.ADD_ARGUMENT,
        disposition=TransformDisposition.ADMITTED,
        obligation_ids=("obligation:consumer:one",),
        target_paths=("pkg/caller.py",),
        expression_refs=("expr:ctx",),
        proof_refs=("proof:transform",),
    )
    assert AnalyticalTransform.from_dict(admitted.to_record()) == admitted

    with pytest.raises(ChangePropagationAuthorityError, match="target path"):
        AnalyticalTransform(
            roots=roots,
            transform_id="transform:abstain",
            kind=TransformKind.ADD_ARGUMENT,
            disposition=TransformDisposition.ABSTAINED,
            obligation_ids=("obligation:consumer:one",),
            target_paths=("pkg/caller.py",),
        )

    with pytest.raises(ChangePropagationError, match="proof refs"):
        AnalyticalTransform(
            roots=roots,
            transform_id="transform:no-proof",
            kind=TransformKind.THREAD_PARAMETER,
            disposition=TransformDisposition.ADMITTED,
            obligation_ids=("obligation:consumer:one",),
            target_paths=("pkg/caller.py",),
        )


def test_plan_requires_complete_consumer_dispositions_and_rejects_forged_set(
    roots: PropagationAuthorityRoots, trusted_node: GraphNodeRef
) -> None:
    obligation = _obligation(roots, trusted_node)
    set_id = obligation_set_identity((obligation,))
    step = PropagationPlanStep(
        step_id="step:analytical",
        kind=PlanStepKind.ANALYTICAL,
        obligation_ids=(obligation.obligation_id,),
        transform_id="transform:add-arg",
        read_paths=("pkg/caller.py",),
        write_paths=("pkg/caller.py",),
        postcondition_refs=("post:arity",),
    )
    plan = AtomicPropagationPlan(
        roots=roots,
        plan_id="plan:one",
        change_set_id="changeset:one",
        delta_id="delta:one",
        impact_closure_id="impact:one",
        disposition=PlanDisposition.ADMITTED,
        obligations=(obligation,),
        obligation_set_id=set_id,
        steps=(step,),
        permitted_read_paths=("pkg/caller.py",),
        permitted_write_paths=("pkg/caller.py",),
        checkpoint_strategy_ref="checkpoint:content-addressed",
        rollback_strategy_ref="rollback:restore-checkpoint",
        fixed_point_obligation_ref="fixed-point:reprove",
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
    )
    assert AtomicPropagationPlan.from_dict(plan.to_record()) == plan

    with pytest.raises(ForgedChangePropagationIdentityError, match="obligation_set_id"):
        AtomicPropagationPlan(
            roots=roots,
            plan_id="plan:forged",
            change_set_id="changeset:one",
            delta_id="delta:one",
            impact_closure_id="impact:one",
            disposition=PlanDisposition.ADMITTED,
            obligations=(obligation,),
            obligation_set_id="obligation-set:forged",
            steps=(step,),
            permitted_read_paths=("pkg/caller.py",),
            permitted_write_paths=("pkg/caller.py",),
            checkpoint_strategy_ref="checkpoint:content-addressed",
            rollback_strategy_ref="rollback:restore-checkpoint",
            fixed_point_obligation_ref="fixed-point:reprove",
            proof_refs=("proof:plan",),
            invalidation_refs=("tree:candidate",),
        )

    # Steps must only name obligations present on the plan; a migrate without a
    # covering step is also rejected when no steps exist.
    other = _obligation(roots, trusted_node, consumer_id="consumer:two")
    other_set = obligation_set_identity((other,))
    with pytest.raises(ChangePropagationError, match="known obligations"):
        AtomicPropagationPlan(
            roots=roots,
            plan_id="plan:partial",
            change_set_id="changeset:one",
            delta_id="delta:one",
            impact_closure_id="impact:one",
            disposition=PlanDisposition.ADMITTED,
            obligations=(other,),
            obligation_set_id=other_set,
            steps=(step,),  # covers a different obligation
            permitted_read_paths=("pkg/caller.py",),
            permitted_write_paths=("pkg/caller.py",),
            checkpoint_strategy_ref="checkpoint:content-addressed",
            rollback_strategy_ref="rollback:restore-checkpoint",
            fixed_point_obligation_ref="fixed-point:reprove",
            proof_refs=("proof:plan",),
            invalidation_refs=("tree:candidate",),
        )
    with pytest.raises(ChangePropagationError, match="require steps|cover every migrate"):
        AtomicPropagationPlan(
            roots=roots,
            plan_id="plan:no-steps",
            change_set_id="changeset:one",
            delta_id="delta:one",
            impact_closure_id="impact:one",
            disposition=PlanDisposition.ADMITTED,
            obligations=(other,),
            obligation_set_id=other_set,
            steps=(),
            permitted_read_paths=("pkg/caller.py",),
            permitted_write_paths=("pkg/caller.py",),
            checkpoint_strategy_ref="checkpoint:content-addressed",
            rollback_strategy_ref="rollback:restore-checkpoint",
            fixed_point_obligation_ref="fixed-point:reprove",
            proof_refs=("proof:plan",),
            invalidation_refs=("tree:candidate",),
        )


def test_plan_scc_group_and_non_admitted_write_ban(
    roots: PropagationAuthorityRoots, trusted_node: GraphNodeRef
) -> None:
    obligation = _obligation(roots, trusted_node)
    set_id = obligation_set_identity((obligation,))
    step = PropagationPlanStep(
        step_id="step:a",
        kind=PlanStepKind.ANALYTICAL,
        obligation_ids=(obligation.obligation_id,),
        transform_id="transform:add-arg",
        write_paths=("pkg/caller.py",),
        scc_group_id="group:scc-one",
    )
    group = PropagationSCCGroup(
        group_id="group:scc-one",
        scc_id="scc:one",
        step_ids=("step:a",),
        consumer_ids=(obligation.consumer_id,),
    )
    plan = AtomicPropagationPlan(
        roots=roots,
        plan_id="plan:scc",
        change_set_id="changeset:one",
        delta_id="delta:one",
        impact_closure_id="impact:one",
        disposition=PlanDisposition.ADMITTED,
        obligations=(obligation,),
        obligation_set_id=set_id,
        steps=(step,),
        scc_groups=(group,),
        permitted_read_paths=("pkg/caller.py",),
        permitted_write_paths=("pkg/caller.py",),
        checkpoint_strategy_ref="checkpoint:content-addressed",
        rollback_strategy_ref="rollback:restore-checkpoint",
        fixed_point_obligation_ref="fixed-point:reprove",
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
    )
    assert plan.scc_groups[0].group_id == "group:scc-one"

    with pytest.raises(ChangePropagationAuthorityError, match="write path"):
        AtomicPropagationPlan(
            roots=roots,
            plan_id="plan:abstain",
            change_set_id="changeset:one",
            delta_id="delta:one",
            impact_closure_id="impact:one",
            disposition=PlanDisposition.ABSTAINED,
            obligations=(obligation,),
            obligation_set_id=set_id,
            steps=(),
            permitted_write_paths=("pkg/caller.py",),
            invalidation_refs=("tree:candidate",),
        )


def test_transaction_state_machine(roots: PropagationAuthorityRoots) -> None:
    pending = PropagationTransaction(
        roots=roots,
        transaction_id="txn:one",
        plan_id="plan:one",
        state=TransactionState.PENDING,
        checkpoint_id="checkpoint:before",
    )
    assert PropagationTransaction.from_dict(pending.to_record()) == pending

    executing = PropagationTransaction(
        roots=roots,
        transaction_id="txn:one",
        plan_id="plan:one",
        state=TransactionState.EXECUTING,
        checkpoint_id="checkpoint:before",
        active_scc_group_id="group:scc-one",
        completed_step_ids=("step:a",),
        lease_id="lease:writer",
    )
    assert executing.lease_id == "lease:writer"

    with pytest.raises(ChangePropagationAuthorityError, match="lease"):
        PropagationTransaction(
            roots=roots,
            transaction_id="txn:bad",
            plan_id="plan:one",
            state=TransactionState.EXECUTING,
            checkpoint_id="checkpoint:before",
        )

    with pytest.raises(ChangePropagationError, match="diagnostic"):
        PropagationTransaction(
            roots=roots,
            transaction_id="txn:rb",
            plan_id="plan:one",
            state=TransactionState.ROLLED_BACK,
            checkpoint_id="checkpoint:before",
        )


def test_completion_requires_fixed_point_receipt(
    roots: PropagationAuthorityRoots, trusted_node: GraphNodeRef
) -> None:
    obligation = _obligation(roots, trusted_node)
    fixed = FixedPointReceipt(
        roots=roots,
        receipt_id="fixed:one",
        plan_id="plan:one",
        iteration_count=2,
        proof_refs=("proof:fixed",),
        validation_refs=("validation:full",),
    )
    assert fixed.is_fixed_point

    complete = PropagationCompletionReceipt(
        roots=roots,
        completion_id="completion:one",
        plan_id="plan:one",
        transaction_id="txn:one",
        disposition=CompletionDisposition.COMPLETE,
        fixed_point_receipt=fixed,
        discharged_obligation_ids=(obligation.obligation_id,),
        proof_refs=("proof:completion",),
        validation_refs=("validation:full",),
        invalidation_refs=("tree:candidate",),
    )
    assert PropagationCompletionReceipt.from_dict(complete.to_record()) == complete

    with pytest.raises(ChangePropagationError, match="fixed-point receipt"):
        PropagationCompletionReceipt(
            roots=roots,
            completion_id="completion:no-fp",
            plan_id="plan:one",
            transaction_id="txn:one",
            disposition=CompletionDisposition.COMPLETE,
            fixed_point_receipt=None,
            discharged_obligation_ids=(obligation.obligation_id,),
            proof_refs=("proof:completion",),
            validation_refs=("validation:full",),
            invalidation_refs=("tree:candidate",),
        )

    residual_fp = FixedPointReceipt(
        roots=roots,
        receipt_id="fixed:residual",
        plan_id="plan:one",
        iteration_count=3,
        residual_consumer_ids=("consumer:second-order",),
        proof_refs=("proof:fixed",),
    )
    with pytest.raises(ChangePropagationError, match="residual-free"):
        PropagationCompletionReceipt(
            roots=roots,
            completion_id="completion:residual",
            plan_id="plan:one",
            transaction_id="txn:one",
            disposition=CompletionDisposition.COMPLETE,
            fixed_point_receipt=residual_fp,
            discharged_obligation_ids=(obligation.obligation_id,),
            proof_refs=("proof:completion",),
            validation_refs=("validation:full",),
            invalidation_refs=("tree:candidate",),
        )

    incomplete = PropagationCompletionReceipt(
        roots=roots,
        completion_id="completion:incomplete",
        plan_id="plan:one",
        transaction_id="txn:one",
        disposition=CompletionDisposition.INCOMPLETE,
        fixed_point_receipt=None,
        unresolved_mandatory_ids=("consumer:missed",),
        invalidation_refs=("tree:candidate",),
    )
    assert incomplete.disposition is CompletionDisposition.INCOMPLETE


def test_round_trip_all_primary_records(
    roots: PropagationAuthorityRoots, trusted_node: GraphNodeRef
) -> None:
    """Every declared schema round-trips through content-addressed serialization."""
    change_set = ProgramChangeSet(
        roots=roots,
        kind=ChangeSetKind.PROPOSED_CONTRACT_CHANGE,
        producer_id="producer:review",
        changed_paths=("pkg/process.py",),
    )
    clause = _clause()
    delta = ProgramContractDelta(
        roots=roots,
        change_set_id=change_set.content_id,
        subject_symbol_id="symbol:process",
        before_contract_ref="contract:before",
        after_contract_ref="contract:after",
        clauses=(clause,),
    )
    edge = GraphEdgeRef(
        edge_id="edge:call",
        kind=GraphEdgeKind.CALL,
        source_node_id=trusted_node.node_id,
        target_node_id="node:process",
        provenance=GraphProvenance.TRUSTED,
        extractor_id="extractor:ast",
    )
    consumer = ImpactConsumer(
        consumer_id="consumer:one",
        node=trusted_node,
        depth=0,
        mandatory=True,
        edge_refs=(edge.edge_id,),
    )
    impact = ImpactClosureReceipt(
        roots=roots,
        delta_id=delta.content_id,
        completeness=ImpactCompleteness.COMPLETE,
        consumers=(consumer,),
    )
    obligation = _obligation(roots, trusted_node)
    missing = MissingInputRequirement(
        roots=roots,
        requirement_id="missing:context",
        obligation_id=obligation.obligation_id,
        clause_id=clause.clause_id,
        parameter_name="context",
        type_ref="type:Context",
        nullability="non_null",
        information_content_ref="info:ctx",
    )
    value = ValueCandidate(
        roots=roots,
        candidate_id="value:param",
        requirement_id=missing.requirement_id,
        kind=ValueCandidateKind.PARAMETER,
        disposition=ValueCandidateDisposition.PROVED,
        source_node=trusted_node,
        expression_ref="expr:ctx",
        type_ref="type:Context",
        semantic_authority=True,
        proof_refs=("proof:value",),
    )
    behavior = RequiredBehaviorContract(
        roots=roots,
        behavior_id="behavior:ctx",
        kind=BehaviorKind.DATA_STRUCTURE,
        subject_symbol_id="symbol:Context",
        evidence_precedence=BehaviorEvidencePrecedence.NORMATIVE_SPEC,
        field_refs=("field:id",),
        proof_refs=("proof:behavior",),
    )
    transform = AnalyticalTransform(
        roots=roots,
        transform_id="transform:add",
        kind=TransformKind.ADD_ARGUMENT,
        disposition=TransformDisposition.ADMITTED,
        obligation_ids=(obligation.obligation_id,),
        target_paths=("pkg/caller.py",),
        proof_refs=("proof:transform",),
    )
    set_id = obligation_set_identity((obligation,))
    step = PropagationPlanStep(
        step_id="step:1",
        kind=PlanStepKind.ANALYTICAL,
        obligation_ids=(obligation.obligation_id,),
        transform_id=transform.transform_id,
        write_paths=("pkg/caller.py",),
    )
    plan = AtomicPropagationPlan(
        roots=roots,
        plan_id="plan:full",
        change_set_id=change_set.content_id,
        delta_id=delta.content_id,
        impact_closure_id=impact.content_id,
        disposition=PlanDisposition.ADMITTED,
        obligations=(obligation,),
        obligation_set_id=set_id,
        steps=(step,),
        permitted_write_paths=("pkg/caller.py",),
        checkpoint_strategy_ref="checkpoint:ca",
        rollback_strategy_ref="rollback:ca",
        fixed_point_obligation_ref="fixed-point:1",
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
    )
    txn = PropagationTransaction(
        roots=roots,
        transaction_id="txn:full",
        plan_id=plan.plan_id,
        state=TransactionState.COMMITTED,
        checkpoint_id="checkpoint:before",
        completed_step_ids=("step:1",),
    )
    fixed = FixedPointReceipt(
        roots=roots,
        receipt_id="fixed:full",
        plan_id=plan.plan_id,
        iteration_count=1,
        proof_refs=("proof:fixed",),
        validation_refs=("validation:full",),
    )
    completion = PropagationCompletionReceipt(
        roots=roots,
        completion_id="completion:full",
        plan_id=plan.plan_id,
        transaction_id=txn.transaction_id,
        disposition=CompletionDisposition.COMPLETE,
        fixed_point_receipt=fixed,
        discharged_obligation_ids=(obligation.obligation_id,),
        proof_refs=("proof:completion",),
        validation_refs=("validation:full",),
        invalidation_refs=("tree:candidate",),
    )

    for record, cls in (
        (change_set, ProgramChangeSet),
        (delta, ProgramContractDelta),
        (trusted_node, GraphNodeRef),
        (edge, GraphEdgeRef),
        (impact, ImpactClosureReceipt),
        (obligation, ConsumerMigrationObligation),
        (missing, MissingInputRequirement),
        (value, ValueCandidate),
        (behavior, RequiredBehaviorContract),
        (transform, AnalyticalTransform),
        (plan, AtomicPropagationPlan),
        (txn, PropagationTransaction),
        (completion, PropagationCompletionReceipt),
    ):
        assert cls.from_dict(record.to_record()) == record
        # No floating-point or source-body fields in serialized payloads.
        assert "source_body" not in record.to_json()
        assert "Infinity" not in record.to_json()


def test_bounds_reject_oversized_identifier_lists(roots: PropagationAuthorityRoots) -> None:
    too_many = tuple(f"id:{i}" for i in range(300))
    with pytest.raises(ChangePropagationBoundsError):
        ProgramChangeSet(
            roots=roots,
            kind=ChangeSetKind.WORKTREE_DIFF,
            producer_id="producer:x",
            changed_paths=("pkg/a.py",),
            evidence_refs=too_many,
        )
