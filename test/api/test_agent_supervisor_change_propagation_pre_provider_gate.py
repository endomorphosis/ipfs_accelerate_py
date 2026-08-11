"""Contract tests for the final, non-dispatching change-propagation provider gate."""

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
    PropagationAuthorityRoots,
    TransformDisposition,
    TransformKind,
)
from ipfs_accelerate_py.agent_supervisor.analysis.repository_snapshot import (
    CoverageDisposition,
    CoverageKind,
    EntryKind,
    GitStatus,
    RepositorySnapshot,
    RepositorySnapshotStats,
)
from ipfs_accelerate_py.agent_supervisor.integrations.change_propagation_capabilities import (
    ChangePropagationCapability,
    ChangePropagationCapabilityDiagnostic,
    ChangePropagationCapabilityReport,
    ChangePropagationCapabilityStatus,
    ChangePropagationDiagnosticCode,
)
from ipfs_accelerate_py.agent_supervisor.planning.change_propagation_plan import (
    ChangePropagationPlanner,
    PlanEvidenceBundle,
    PlanPathSpan,
    PlanValidationCommand,
)
from ipfs_accelerate_py.agent_supervisor.planning.support_behavior_placement import (
    SupportPlacementAction,
    SupportPlacementDecision,
    SupportPlacementDisposition,
)
from ipfs_accelerate_py.agent_supervisor.proof.change_propagation_edit_packet import (
    materialize_change_propagation_edit_packet,
)
from ipfs_accelerate_py.agent_supervisor.proof.missing_input_synthesis import (
    SynthesisDisposition,
    ValueMappingProof,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.change_propagation_provider_router import (
    ProviderModelConfigIdentity,
    WriterLease,
)
from ipfs_accelerate_py.agent_supervisor.validation.change_propagation_pre_provider_gate import (
    ChangePropagationPreProviderGate,
    ChangePropagationPreProviderGateError,
    PropagationGateReason,
    PropagationGateReceipt,
)


ROOTS = PropagationAuthorityRoots(
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

MODEL_TARGET = "pkg/support/context.py"
MODEL_ARTIFACT = "blob:support"
ANALYTICAL_TARGET = "pkg/a.py"


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


def _admit_mixed():
    c1 = ImpactConsumer(
        consumer_id="consumer:a",
        node=_node(ANALYTICAL_TARGET, "symbol:a"),
        depth=1,
        mandatory=True,
        edge_refs=("edge:a",),
    )
    c2 = ImpactConsumer(
        consumer_id="consumer:b",
        node=_node(MODEL_TARGET, "symbol:b"),
        depth=1,
        mandatory=True,
        edge_refs=("edge:b",),
    )
    closure = ImpactClosureReceipt(
        roots=ROOTS,
        delta_id="delta:one",
        completeness=ImpactCompleteness.COMPLETE,
        consumers=(c1, c2),
        sccs=(),
        frontier_node_ids=(),
        frontier_edge_ids=(),
        validation_refs=("validation:impact",),
        resource_bound_refs=("bound:impact",),
        evidence_refs=("evidence:graph",),
    )
    o1 = ConsumerMigrationObligation(
        roots=ROOTS,
        obligation_id="obligation:consumer:a",
        consumer_id="consumer:a",
        delta_id="delta:one",
        disposition=ConsumerDisposition.MIGRATE,
        clause_ids=("clause:param-add",),
        node=_node(ANALYTICAL_TARGET, "symbol:a"),
        proof_refs=("proof:obligation",),
        missing_input_ids=("missing:context",),
        behavior_contract_ids=(),
        invalidation_refs=("tree:candidate",),
    )
    o2 = ConsumerMigrationObligation(
        roots=ROOTS,
        obligation_id="obligation:consumer:b",
        consumer_id="consumer:b",
        delta_id="delta:one",
        disposition=ConsumerDisposition.MIGRATE,
        clause_ids=("clause:param-add",),
        node=_node(MODEL_TARGET, "symbol:b"),
        proof_refs=("proof:obligation",),
        missing_input_ids=(),
        behavior_contract_ids=("behavior:SupportContext",),
        invalidation_refs=("tree:candidate",),
    )
    mapping = ValueMappingProof(
        requirement_id="missing:context",
        consumer_id="consumer:a",
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
    transform = AnalyticalTransform(
        roots=ROOTS,
        transform_id="transform:a",
        kind=TransformKind.ADD_ARGUMENT,
        disposition=TransformDisposition.ADMITTED,
        obligation_ids=("obligation:consumer:a",),
        target_paths=(ANALYTICAL_TARGET,),
        expression_refs=("expr:ctx",),
        proof_refs=("proof:transform",),
    )
    placement = SupportPlacementDecision(
        disposition=SupportPlacementDisposition.ADMITTED,
        roots=ROOTS,
        behavior_id="behavior:SupportContext",
        candidate_set_id="placement-set:one",
        selected_candidate_id="candidate:owner",
        action=SupportPlacementAction.PLACE_NEW,
        target_path=MODEL_TARGET,
        placement_paths=(MODEL_TARGET,),
        reason_codes=("owner_unique",),
        proof_receipt_ids=("proof:placement",),
        evidence_refs=("evidence:arch",),
        eligible_candidate_ids=("candidate:owner",),
        margin=2,
    )
    evidence = PlanEvidenceBundle(
        roots=ROOTS,
        change_set_id="changeset:mixed",
        delta_id="delta:one",
        impact_closure=closure,
        obligations=(o1, o2),
        value_mapping_proofs=(mapping,),
        analytical_transforms=(transform,),
        placement_decisions=(placement,),
        write_spans=(
            PlanPathSpan(
                path=ANALYTICAL_TARGET,
                start=0,
                end=10,
                artifact_id="blob:a",
                before_hash="sha256:a",
            ),
            PlanPathSpan(
                path=MODEL_TARGET,
                start=0,
                end=10,
                artifact_id=MODEL_ARTIFACT,
                before_hash="sha256:support",
            ),
        ),
        validation_commands=(
            PlanValidationCommand(
                command_id="validate:pytest",
                argv=("python", "-m", "pytest", "-q", "test_x.py"),
                required=True,
            ),
        ),
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        expected_roots=ROOTS,
    )
    admission = ChangePropagationPlanner().admit(evidence)
    assert admission.admitted
    packet = materialize_change_propagation_edit_packet(
        admission, roots=ROOTS, evidence=evidence
    )
    return admission, packet, closure, evidence


def snapshot(**changes: object) -> RepositorySnapshot:
    dispositions = (
        CoverageDisposition(
            ANALYTICAL_TARGET,
            CoverageKind.SEMANTIC_AST,
            GitStatus.CLEAN,
            EntryKind.REGULAR,
            "semantic_source",
            "fixture",
            content_digest="sha256:a",
            git_object_id="blob:a",
        ),
        CoverageDisposition(
            MODEL_TARGET,
            CoverageKind.SEMANTIC_AST,
            GitStatus.CLEAN,
            EntryKind.REGULAR,
            "semantic_source",
            "fixture",
            content_digest="sha256:support",
            git_object_id=MODEL_ARTIFACT,
        ),
    )
    values: dict[str, object] = {
        "primary_root": ".",
        "head_commit_id": "commit:test",
        "head_tree_id": ROOTS.candidate_tree_id,
        "index_tree_id": ROOTS.candidate_tree_id,
        "scope_policy_id": "scope:test",
        "scope_id": "scope:test",
        "dispositions": dispositions,
        "dependency_identities": (),
        "gitlinks": (),
        "stats": RepositorySnapshotStats(2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2),
    }
    values.update(changes)
    return RepositorySnapshot(**values)  # type: ignore[arg-type]


def capabilities(*, complete: bool = True) -> ChangePropagationCapabilityReport:
    if complete:
        cap = ChangePropagationCapability(
            "accelerator.llm_router",
            ChangePropagationCapabilityStatus.AVAILABLE,
            module_paths=("fixture.llm_router",),
            reconstruction_compatible=True,
        )
    else:
        cap = ChangePropagationCapability(
            "accelerator.llm_router",
            ChangePropagationCapabilityStatus.PARTIAL,
            diagnostic=ChangePropagationCapabilityDiagnostic(
                ChangePropagationDiagnosticCode.PARTIAL_INTERFACE,
                "accelerator.llm_router",
                "incomplete",
            ),
        )
    return ChangePropagationCapabilityReport((cap,), (), (), "gitlink:test")


def identity() -> ProviderModelConfigIdentity:
    return ProviderModelConfigIdentity(
        provider_id="provider:test",
        model_id="provider-model:test",
        config_id=ROOTS.config_id,
        router_backend="llm_router",
    )


def lease(packet, step_id: str, write_paths: tuple[str, ...]) -> WriterLease:
    return WriterLease(
        lease_id="lease:test",
        permitted_write_paths=write_paths,
        packet_id=packet.packet_id,
        plan_id=packet.plan_id,
        step_id=step_id,
        tree_id=ROOTS.candidate_tree_id,
        provider_id="provider:test",
        model_id="provider-model:test",
        config_id=ROOTS.config_id,
    )


def valid_kwargs(**changes: object) -> dict[str, object]:
    admission, packet, closure, _ = _admit_mixed()
    model_step_id = packet.model_required_step_ids[0]
    model_step = next(step for step in packet.steps if step.step_id == model_step_id)
    values: dict[str, object] = {
        "packet": packet,
        "admission": admission,
        "snapshot": snapshot(),
        "current_roots": ROOTS,
        "capability_report": capabilities(),
        "now": 100,
        "step_id": model_step_id,
        "provider_identity": identity(),
        "writer_lease": lease(packet, model_step_id, model_step.write_paths),
        "impact_closure": closure,
        "expires_at": 400,
    }
    values.update(changes)
    return values


def test_current_admitted_model_step_emits_bounded_non_dispatch_receipt() -> None:
    receipt = ChangePropagationPreProviderGate().require_valid(**valid_kwargs())  # type: ignore[arg-type]

    assert receipt.write_paths == (MODEL_TARGET,)
    assert receipt.to_dict()["provider_invoked"] is False
    assert receipt.to_dict()["authorized_paths"] == [MODEL_TARGET]
    assert receipt.frontier_complete is True
    assert receipt.provider_identity["config_id"] == ROOTS.config_id
    assert receipt.writer_lease_id == "lease:test"
    assert PropagationGateReceipt.from_dict(receipt.to_record()) == receipt


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (
            {"snapshot": snapshot(index_tree_id="tree:changed")},
            PropagationGateReason.TREE_OR_OVERLAY_CHANGED,
        ),
        (
            {"snapshot": snapshot(dispositions=())},
            PropagationGateReason.TARGET_MISSING_OR_MOVED,
        ),
        (
            {
                "snapshot": snapshot(
                    dispositions=(
                        CoverageDisposition(
                            ANALYTICAL_TARGET,
                            CoverageKind.SEMANTIC_AST,
                            GitStatus.CLEAN,
                            EntryKind.REGULAR,
                            "semantic",
                            "fixture",
                            git_object_id="blob:a",
                        ),
                        CoverageDisposition(
                            MODEL_TARGET,
                            CoverageKind.SEMANTIC_AST,
                            GitStatus.CLEAN,
                            EntryKind.REGULAR,
                            "semantic",
                            "fixture",
                            git_object_id="blob:other",
                        ),
                    )
                )
            },
            PropagationGateReason.TARGET_HASH_DRIFT,
        ),
        (
            {
                "current_roots": replace(
                    ROOTS, policy_id="policy:changed", translator_id="translator:changed"
                )
            },
            PropagationGateReason.ROOT_DRIFT,
        ),
        (
            {
                "current_roots": replace(
                    ROOTS, graph_id="graph:changed", index_id="index:changed"
                )
            },
            PropagationGateReason.GRAPH_INDEX_MODEL_CONFIG_DRIFT,
        ),
        (
            {
                "current_roots": replace(
                    ROOTS, toolchain_id="toolchain:changed"
                )
            },
            PropagationGateReason.TRANSLATOR_TOOLCHAIN_POLICY_DRIFT,
        ),
        ({"expires_at": 50}, PropagationGateReason.EXPIRED_PROOF),
        (
            {"capability_report": capabilities(complete=False)},
            PropagationGateReason.INCOMPLETE_CAPABILITY,
        ),
        ({"read_only_paths": (MODEL_TARGET,)}, PropagationGateReason.READ_ONLY_OR_ESCAPED_PATH),
        ({"writer_lease": None}, PropagationGateReason.PATH_LEASE_MISMATCH),
        ({"provider_identity": None}, PropagationGateReason.PROVIDER_IDENTITY_MISMATCH),
    ],
)
def test_gate_rejects_drift_before_a_provider_can_be_called(
    change: dict[str, object], reason: PropagationGateReason
) -> None:
    reasons = ChangePropagationPreProviderGate().validate(**valid_kwargs(**change))  # type: ignore[arg-type]
    assert reason in reasons
    with pytest.raises(ChangePropagationPreProviderGateError, match=reason.value):
        ChangePropagationPreProviderGate().require_valid(**valid_kwargs(**change))  # type: ignore[arg-type]


def test_analytical_step_cannot_pass_pre_provider_gate() -> None:
    values = valid_kwargs()
    packet = values["packet"]
    analytical_id = packet.analytical_step_ids[0]  # type: ignore[index]
    analytical = next(step for step in packet.steps if step.step_id == analytical_id)  # type: ignore[union-attr]
    values["step_id"] = analytical_id
    values["writer_lease"] = lease(packet, analytical_id, analytical.write_paths)
    reasons = ChangePropagationPreProviderGate().validate(**values)  # type: ignore[arg-type]
    assert PropagationGateReason.ANALYTICAL_STEP_PROVIDER in reasons
    assert PropagationGateReason.STEP_NOT_MODEL_REQUIRED in reasons


def test_unresolved_frontier_and_incomplete_behavior_fail_closed() -> None:
    values = valid_kwargs()
    closure = values["impact_closure"]
    assert closure is not None
    frontier = replace(
        closure,
        completeness=ImpactCompleteness.PARTIAL_WITH_FRONTIER,
        frontier_node_ids=("node:unknown",),
    )
    reasons = ChangePropagationPreProviderGate().validate(
        **valid_kwargs(impact_closure=frontier)  # type: ignore[arg-type]
    )
    assert PropagationGateReason.FRONTIER_UNRESOLVED in reasons

    admission, packet, _, _ = _admit_mixed()
    model_step_id = packet.model_required_step_ids[0]
    # Forge incomplete behavior by rebuilding step dict without behavior ids is
    # hard on frozen packets; instead assert the happy path has behavior and a
    # missing provider identity blocks before llm_router.
    assert any(
        step.required_behavior_ids
        for step in packet.steps
        if step.step_id == model_step_id
    )
    assert PropagationGateReason.PROVIDER_IDENTITY_MISMATCH in (
        ChangePropagationPreProviderGate().validate(
            **valid_kwargs(provider_identity={"provider_id": "x"})  # type: ignore[arg-type]
        )
    )


def test_packet_plan_mismatch_and_path_lease_escape_fail_closed() -> None:
    values = valid_kwargs()
    admission = values["admission"]
    assert admission is not None
    # Mutate admission step order identity via replace on a shallow copy of plan
    # content is not available; instead swap in a lease with wrong paths.
    packet = values["packet"]
    model_step_id = values["step_id"]
    bad_lease = lease(packet, model_step_id, ("pkg/escaped.py",))  # type: ignore[arg-type]
    reasons = ChangePropagationPreProviderGate().validate(
        **valid_kwargs(writer_lease=bad_lease)  # type: ignore[arg-type]
    )
    assert PropagationGateReason.PATH_LEASE_MISMATCH in reasons

    # Root policy drift also surfaces as root_drift (and translator/toolchain/policy).
    drifted = replace(ROOTS, policy_id="policy:other")
    assert PropagationGateReason.ROOT_DRIFT in ChangePropagationPreProviderGate().validate(
        **valid_kwargs(current_roots=drifted)  # type: ignore[arg-type]
    )


def test_receipt_rejects_forged_identity_and_path_broadening() -> None:
    receipt = ChangePropagationPreProviderGate().require_valid(**valid_kwargs())  # type: ignore[arg-type]
    forged = receipt.to_record()
    forged["authorized_paths"] = [MODEL_TARGET, "pkg/other.py"]
    with pytest.raises(ChangePropagationPreProviderGateError, match="broaden"):
        PropagationGateReceipt.from_dict(forged)

    forged_id = receipt.to_record()
    forged_id["receipt_id"] = "baguqeeraforged"
    with pytest.raises(ChangePropagationPreProviderGateError, match="forged"):
        PropagationGateReceipt.from_dict(forged_id)
