"""Fail-closed coverage for bounded llm_router change-propagation routes (RPR-041)."""

from __future__ import annotations

import json
from typing import Any, Mapping

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
)
from ipfs_accelerate_py.agent_supervisor.planning.support_behavior_placement import (
    SupportPlacementAction,
    SupportPlacementDecision,
    SupportPlacementDisposition,
)
from ipfs_accelerate_py.agent_supervisor.proof.change_propagation_edit_packet import (
    ChangePropagationEditPacket,
    PropagationEditStepKind,
    materialize_change_propagation_edit_packet,
)
from ipfs_accelerate_py.agent_supervisor.proof.missing_input_synthesis import (
    SynthesisDisposition,
    ValueMappingProof,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.change_propagation_provider_router import (
    CHANGE_PROPAGATION_PROVIDER_ROUTER_INTERFACE,
    MODEL_FORBIDDEN_CHOICES,
    AnalyticalNonSuccessReason,
    ChangePropagationProviderRouter,
    PropagationProviderBounds,
    PropagationProviderEnvelope,
    PropagationProviderReason,
    PropagationProviderRoutingError,
    PropagationRouteStatus,
    ProviderModelConfigIdentity,
    WriterLease,
    assert_proposal_within_lease,
    build_propagation_provider_envelope,
    normalize_analytical_non_success_reason,
    parse_proposal_paths,
    route_change_propagation_step,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.contract_packet_provider_router import (
    ProviderRole,
    RouteStatus,
)


# ---------------------------------------------------------------------------
# Fixtures (aligned with RPR-040 mixed analytical / model-required packet)
# ---------------------------------------------------------------------------


@pytest.fixture
def roots() -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id="repository:rpr-041",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:rpr-041",
        index_id="index:rpr-041",
        model_id="model:rpr-041",
        config_id="config:rpr-041",
        translator_id="translator:rpr-041",
        toolchain_id="toolchain:rpr-041",
        policy_id="policy:rpr-041",
    )


@pytest.fixture
def identity() -> ProviderModelConfigIdentity:
    return ProviderModelConfigIdentity(
        provider_id="provider:grok",
        model_id="model:grok-code",
        config_id="config:propagation-llm",
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
) -> ImpactClosureReceipt:
    return ImpactClosureReceipt(
        roots=roots,
        delta_id="delta:one",
        completeness=ImpactCompleteness.COMPLETE,
        consumers=consumers,
        sccs=(),
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
        proved = (
            ("candidate:ctx",)
            if disposition is SynthesisDisposition.UNIQUE_PROVED
            else ()
        )
    return ValueMappingProof(
        requirement_id=requirement_id,
        consumer_id=consumer_id,
        disposition=disposition,
        facet_results=(),
        proved_candidate_ids=proved,
        refuted_candidate_ids=(),
        expression_ref="expr:ctx"
        if disposition is SynthesisDisposition.UNIQUE_PROVED
        else "",
        type_ref="type:Context",
        repository_id="repository:rpr-041",
        tree_id="tree:candidate",
        toolchain_id="toolchain:rpr-041",
        policy_id="policy:rpr-041",
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
    obligation_ids: tuple[str, ...] = ("obligation:consumer:a",),
    path: str = "pkg/a.py",
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
        dependency_transform_ids=(),
        rejection_reasons=(),
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


def _mixed_packet(roots: PropagationAuthorityRoots) -> ChangePropagationEditPacket:
    """One analytical step + one behavior-complete model-required step."""

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
        value_mapping_proofs=(_mapping(consumer_id="consumer:a"),),
        analytical_transforms=(
            _transform(
                roots,
                transform_id="transform:a",
                obligation_ids=("obligation:consumer:a",),
                path="pkg/a.py",
            ),
        ),
        placement_decisions=(_placement(roots),),
        read_spans=(
            PlanPathSpan(
                path="pkg/a.py",
                start=0,
                end=20,
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
        write_spans=(
            PlanPathSpan(
                path="pkg/a.py",
                start=0,
                end=20,
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
        resource_bounds=PlanResourceBounds(),
        proof_refs=("proof:plan",),
        invalidation_refs=("tree:candidate",),
        expected_roots=roots,
    )
    admission = ChangePropagationPlanner().admit(evidence)
    assert admission.admitted
    packet = materialize_change_propagation_edit_packet(
        admission, roots=roots, evidence=evidence
    )
    assert packet.analytical_step_ids
    assert packet.model_required_step_ids
    return packet


def _model_step_id(packet: ChangePropagationEditPacket) -> str:
    return packet.model_required_step_ids[0]


def _write_path(packet: ChangePropagationEditPacket) -> str:
    step = next(
        s for s in packet.steps if s.step_id == packet.model_required_step_ids[0]
    )
    return step.write_paths[0]


def _accept(proposal, role=None):
    return {"accepted": True, "reason_code": f"admitted:{proposal.role.value}"}


def _patch_for(path: str) -> Mapping[str, Any]:
    return {
        "proposal": {
            "patch": f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n",
            "declared_paths": [path],
        }
    }


# ---------------------------------------------------------------------------
# Analytical reason vocabulary
# ---------------------------------------------------------------------------


def test_interface_constant() -> None:
    assert (
        CHANGE_PROPAGATION_PROVIDER_ROUTER_INTERFACE
        == "ChangePropagationProviderRouter@1"
    )


def test_supported_analytical_reasons_normalize() -> None:
    for reason in AnalyticalNonSuccessReason:
        assert normalize_analytical_non_success_reason(reason) is reason
        assert normalize_analytical_non_success_reason(reason.value) is reason


@pytest.mark.parametrize(
    "blocked",
    [
        "ambiguous",
        "unknown_semantics",
        "missing_behavior",
        "scope_escape",
        "invented_behavior",
        "alternatives",
        "new_dependency",
    ],
)
def test_blocked_analytical_reasons_never_escalate(blocked: str) -> None:
    with pytest.raises(PropagationProviderRoutingError) as exc:
        normalize_analytical_non_success_reason(blocked)
    assert exc.value.reason_code in {
        PropagationProviderReason.ANALYTICAL_REASON_BLOCKED.value,
        PropagationProviderReason.ANALYTICAL_REASON_UNSUPPORTED.value,
    }


def test_missing_analytical_reason_is_required() -> None:
    with pytest.raises(PropagationProviderRoutingError) as exc:
        normalize_analytical_non_success_reason(None)
    assert (
        exc.value.reason_code
        == PropagationProviderReason.ANALYTICAL_REASON_MISSING.value
    )


# ---------------------------------------------------------------------------
# Envelope construction
# ---------------------------------------------------------------------------


def test_envelope_binds_contract_delta_values_behavior_paths_and_forbidden_choices(
    roots: PropagationAuthorityRoots,
    identity: ProviderModelConfigIdentity,
) -> None:
    packet = _mixed_packet(roots)
    step_id = _model_step_id(packet)
    envelope = build_propagation_provider_envelope(
        packet,
        step_id=step_id,
        analytical_non_success_reason=AnalyticalNonSuccessReason.UNSUPPORTED_SYNTAX,
        identity=identity,
        snapshot_id="tree:candidate",
    )

    assert isinstance(envelope, PropagationProviderEnvelope)
    payload = envelope.to_dict()
    assert payload["contract_delta"]["delta_id"] == packet.delta_id
    assert payload["contract_delta"]["plan_id"] == packet.plan_id
    assert payload["required_behavior_ids"]
    assert payload["scope"]["write_paths"]
    assert payload["validation_commands"]
    assert payload["authority"]["model_must_not_choose"] == list(MODEL_FORBIDDEN_CHOICES)
    assert payload["authority"]["repository_write_allowed"] is False
    assert payload["authority"]["completion_authoritative"] is False
    assert payload["body_embedded"] is False
    assert payload["identity"]["router_backend"] == "llm_router"
    assert payload["analytical_non_success_reason"] == "unsupported_syntax"

    provider_input = envelope.provider_input_payload()
    encoded = json.dumps(dict(provider_input), sort_keys=True)
    assert "source_code" not in encoded
    assert "repository_corpus" not in encoded
    assert "proof_body" not in encoded
    assert "value_source" in encoded  # forbidden choice listed
    assert provider_input["authority"]["completion_authoritative"] is False


def test_analytical_step_never_builds_envelope(
    roots: PropagationAuthorityRoots,
    identity: ProviderModelConfigIdentity,
) -> None:
    packet = _mixed_packet(roots)
    analytical = packet.analytical_step_ids[0]
    with pytest.raises(PropagationProviderRoutingError) as exc:
        build_propagation_provider_envelope(
            packet,
            step_id=analytical,
            analytical_non_success_reason=AnalyticalNonSuccessReason.UNSUPPORTED_KIND,
            identity=identity,
        )
    assert exc.value.reason_code == PropagationProviderReason.ANALYTICAL_ONLY.value


# ---------------------------------------------------------------------------
# Scope parsing
# ---------------------------------------------------------------------------


def test_parse_proposal_paths_from_declared_and_diff() -> None:
    path = "pkg/support/context.py"
    assert parse_proposal_paths({"declared_paths": [path]}) == (path,)
    assert parse_proposal_paths(
        {"patch": f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"}
    ) == (path,)
    nested = parse_proposal_paths(_patch_for(path))
    assert nested == (path,)


def test_scope_escape_is_rejected() -> None:
    with pytest.raises(PropagationProviderRoutingError) as exc:
        assert_proposal_within_lease(
            {"declared_paths": ["pkg/other.py"]},
            allowed_write_paths=("pkg/support/context.py",),
        )
    assert exc.value.reason_code == PropagationProviderReason.SCOPE_ESCAPE.value

    with pytest.raises(PropagationProviderRoutingError) as exc:
        parse_proposal_paths({"declared_paths": ["../escape.py"]})
    assert exc.value.reason_code == PropagationProviderReason.SCOPE_ESCAPE.value


# ---------------------------------------------------------------------------
# Happy-path routing through existing contract provider boundary
# ---------------------------------------------------------------------------


def test_routes_only_model_required_step_through_provider_and_writer(
    roots: PropagationAuthorityRoots,
    identity: ProviderModelConfigIdentity,
) -> None:
    packet = _mixed_packet(roots)
    step_id = _model_step_id(packet)
    path = _write_path(packet)
    events: list[str] = []
    writes: list[Any] = []

    def grok(request):
        events.append("grok")
        assert request["role"] == ProviderRole.GROK_IMPLEMENT.value
        # Redacted provider input carries admitted semantics only.
        goal = request["provider_input"]["contract_packet"]["goal"]
        assert goal["step_id"] == step_id
        assert goal["analytical_non_success_reason"] == "unsupported_syntax"
        assert "required_behavior_ids" in goal
        assert request["authority"]["repository_write_allowed"] is False
        return _patch_for(path)

    def codex(request):
        events.append("codex")
        assert request["role"] == ProviderRole.CODEX_REVIEW.value
        return {"decision": "approve", "findings": []}

    def writer(proposal, lease_id):
        events.append("write")
        writes.append((proposal, lease_id))

    lease = WriterLease(
        lease_id="lease:rpr-041:1",
        permitted_write_paths=(path,),
        packet_id=packet.packet_id,
        plan_id=packet.plan_id,
        step_id=step_id,
        tree_id="tree:candidate",
        provider_id=identity.provider_id,
        model_id=identity.model_id,
        config_id=identity.config_id,
    )
    router = ChangePropagationProviderRouter(
        identity=identity,
        grok_provider=grok,
        codex_provider=codex,
        admission_gate=_accept,
        writer=writer,
    )
    result = router.route_step(
        packet,
        step_id=step_id,
        analytical_non_success_reason=AnalyticalNonSuccessReason.UNSUPPORTED_SYNTAX,
        current_snapshot_id="tree:candidate",
        apply=True,
        writer_lease=lease,
    )

    assert result.status is PropagationRouteStatus.SUCCEEDED
    assert result.write_performed is True
    assert result.writer_lease_id == "lease:rpr-041:1"
    assert result.proposal_paths == (path,)
    assert result.receipt is not None
    assert result.receipt.proposal_admitted is True
    assert result.receipt.scope_parsed is True
    assert result.receipt.proposal_trusted is True
    assert result.proof_authoritative is False
    assert result.completion_authoritative is False
    assert events == ["grok", "codex", "write"]
    assert len(writes) == 1
    assert result.provider_execution_receipt is not None
    assert result.implementation_route is not None
    assert result.implementation_route.status is RouteStatus.SUCCEEDED


def test_analytical_step_route_is_skipped_without_provider_call(
    roots: PropagationAuthorityRoots,
    identity: ProviderModelConfigIdentity,
) -> None:
    packet = _mixed_packet(roots)
    calls = 0

    def forbidden(_request):
        nonlocal calls
        calls += 1
        raise AssertionError("provider must not run for analytical steps")

    result = ChangePropagationProviderRouter(
        identity=identity,
        grok_provider=forbidden,
        admission_gate=_accept,
    ).route_step(
        packet,
        step_id=packet.analytical_step_ids[0],
        analytical_non_success_reason=AnalyticalNonSuccessReason.UNSUPPORTED_SYNTAX,
        current_snapshot_id="tree:candidate",
    )
    assert result.status is PropagationRouteStatus.SKIPPED
    assert (
        result.reason_code == PropagationProviderReason.ANALYTICAL_ONLY.value
    )
    assert result.write_performed is False
    assert calls == 0


def test_blocked_analytical_reason_creates_no_write(
    roots: PropagationAuthorityRoots,
    identity: ProviderModelConfigIdentity,
) -> None:
    packet = _mixed_packet(roots)
    writes: list[Any] = []
    result = ChangePropagationProviderRouter(
        identity=identity,
        grok_provider=lambda _r: _patch_for(_write_path(packet)),
        admission_gate=_accept,
        writer=lambda p, l: writes.append((p, l)),
    ).route_step(
        packet,
        step_id=_model_step_id(packet),
        analytical_non_success_reason="scope_escape",
        current_snapshot_id="tree:candidate",
        apply=True,
        writer_lease_id="lease:bad",
    )
    assert result.status is PropagationRouteStatus.REJECTED
    assert result.write_performed is False
    assert writes == []


def test_scope_escape_proposal_is_rejected_with_no_write(
    roots: PropagationAuthorityRoots,
    identity: ProviderModelConfigIdentity,
) -> None:
    packet = _mixed_packet(roots)
    path = _write_path(packet)
    writes: list[Any] = []

    def grok(_request):
        return {
            "proposal": {
                "patch": "diff --git a/pkg/escape.py b/pkg/escape.py\n",
                "declared_paths": ["pkg/escape.py"],
            }
        }

    lease = WriterLease(
        lease_id="lease:scope",
        permitted_write_paths=(path,),
        packet_id=packet.packet_id,
        plan_id=packet.plan_id,
        step_id=_model_step_id(packet),
    )
    result = ChangePropagationProviderRouter(
        identity=identity,
        grok_provider=grok,
        codex_provider=lambda _r: {"decision": "approve"},
        admission_gate=_accept,
        writer=lambda p, l: writes.append((p, l)),
    ).route_step(
        packet,
        step_id=_model_step_id(packet),
        analytical_non_success_reason=AnalyticalNonSuccessReason.COMPLEX_IMPLEMENTATION_REQUIRED,
        current_snapshot_id="tree:candidate",
        apply=True,
        writer_lease=lease,
    )
    assert result.status is PropagationRouteStatus.REJECTED
    assert result.reason_code == PropagationProviderReason.SCOPE_ESCAPE.value
    assert result.write_performed is False
    assert writes == []


def test_timeout_and_unavailable_create_no_write(
    roots: PropagationAuthorityRoots,
    identity: ProviderModelConfigIdentity,
) -> None:
    packet = _mixed_packet(roots)
    path = _write_path(packet)
    writes: list[Any] = []
    lease = WriterLease(
        lease_id="lease:timeout",
        permitted_write_paths=(path,),
        packet_id=packet.packet_id,
        plan_id=packet.plan_id,
        step_id=_model_step_id(packet),
    )

    def boom(_request):
        raise TimeoutError("provider timeout")

    result = ChangePropagationProviderRouter(
        identity=identity,
        grok_provider=boom,
        admission_gate=_accept,
        writer=lambda p, l: writes.append((p, l)),
    ).route_step(
        packet,
        step_id=_model_step_id(packet),
        analytical_non_success_reason=AnalyticalNonSuccessReason.BEHAVIOR_IMPLEMENTATION_GAP,
        current_snapshot_id="tree:candidate",
        apply=True,
        writer_lease=lease,
    )
    assert result.write_performed is False
    assert writes == []
    assert result.status in {
        PropagationRouteStatus.REJECTED,
        PropagationRouteStatus.DEFERRED,
        PropagationRouteStatus.FALLBACK,
    }


def test_malformed_proposal_creates_no_write(
    roots: PropagationAuthorityRoots,
    identity: ProviderModelConfigIdentity,
) -> None:
    packet = _mixed_packet(roots)
    path = _write_path(packet)
    writes: list[Any] = []
    lease = WriterLease(
        lease_id="lease:malformed",
        permitted_write_paths=(path,),
        packet_id=packet.packet_id,
        plan_id=packet.plan_id,
        step_id=_model_step_id(packet),
    )
    result = ChangePropagationProviderRouter(
        identity=identity,
        grok_provider=lambda _r: {"proposal": {"note": "no paths or patch"}},
        admission_gate=_accept,
        writer=lambda p, l: writes.append((p, l)),
    ).route_step(
        packet,
        step_id=_model_step_id(packet),
        analytical_non_success_reason=AnalyticalNonSuccessReason.MISSING_DETERMINISTIC_RENDER,
        current_snapshot_id="tree:candidate",
        apply=True,
        writer_lease=lease,
    )
    assert result.write_performed is False
    assert writes == []
    assert result.status is PropagationRouteStatus.REJECTED


def test_apply_without_lease_never_writes(
    roots: PropagationAuthorityRoots,
    identity: ProviderModelConfigIdentity,
) -> None:
    packet = _mixed_packet(roots)
    path = _write_path(packet)
    writes: list[Any] = []
    result = ChangePropagationProviderRouter(
        identity=identity,
        grok_provider=lambda _r: _patch_for(path),
        codex_provider=lambda _r: {"decision": "approve"},
        admission_gate=_accept,
        writer=lambda p, l: writes.append((p, l)),
    ).route_step(
        packet,
        step_id=_model_step_id(packet),
        analytical_non_success_reason=AnalyticalNonSuccessReason.UNSUPPORTED_KIND,
        current_snapshot_id="tree:candidate",
        apply=True,
    )
    assert result.write_performed is False
    assert writes == []
    assert result.receipt is None or result.receipt.write_performed is False


def test_lease_path_mismatch_rejects_before_provider(
    roots: PropagationAuthorityRoots,
    identity: ProviderModelConfigIdentity,
) -> None:
    packet = _mixed_packet(roots)
    calls = 0

    def forbidden(_request):
        nonlocal calls
        calls += 1
        raise AssertionError("must not call provider")

    lease = WriterLease(
        lease_id="lease:mismatch",
        permitted_write_paths=("pkg/wrong.py",),
        packet_id=packet.packet_id,
        plan_id=packet.plan_id,
        step_id=_model_step_id(packet),
    )
    result = ChangePropagationProviderRouter(
        identity=identity,
        grok_provider=forbidden,
        admission_gate=_accept,
    ).route_step(
        packet,
        step_id=_model_step_id(packet),
        analytical_non_success_reason=AnalyticalNonSuccessReason.UNSUPPORTED_SYNTAX,
        current_snapshot_id="tree:candidate",
        apply=True,
        writer_lease=lease,
    )
    assert result.status is PropagationRouteStatus.REJECTED
    assert (
        result.reason_code == PropagationProviderReason.PATH_LEASE_MISMATCH.value
    )
    assert calls == 0


def test_proposal_untrusted_until_admission(
    roots: PropagationAuthorityRoots,
    identity: ProviderModelConfigIdentity,
) -> None:
    packet = _mixed_packet(roots)
    path = _write_path(packet)
    writes: list[Any] = []
    lease = WriterLease(
        lease_id="lease:untrusted",
        permitted_write_paths=(path,),
        packet_id=packet.packet_id,
        plan_id=packet.plan_id,
        step_id=_model_step_id(packet),
    )
    result = ChangePropagationProviderRouter(
        identity=identity,
        grok_provider=lambda _r: _patch_for(path),
        admission_gate=lambda _p, _r=None: {
            "accepted": False,
            "reason_code": "policy_reject",
        },
        writer=lambda p, l: writes.append((p, l)),
    ).route_step(
        packet,
        step_id=_model_step_id(packet),
        analytical_non_success_reason=AnalyticalNonSuccessReason.UNSUPPORTED_SYNTAX,
        current_snapshot_id="tree:candidate",
        apply=True,
        writer_lease=lease,
    )
    assert result.write_performed is False
    assert writes == []
    assert result.receipt is not None
    assert result.receipt.proposal_trusted is False
    assert result.receipt.proposal_admitted is False


def test_llm_generate_adapter_is_used_as_canonical_backend(
    roots: PropagationAuthorityRoots,
    identity: ProviderModelConfigIdentity,
) -> None:
    packet = _mixed_packet(roots)
    path = _write_path(packet)
    seen: dict[str, Any] = {}

    def generate(prompt: str, config: Mapping[str, Any]) -> str:
        seen["prompt"] = prompt
        seen["config"] = dict(config)
        assert config["provider_id"] == identity.provider_id
        assert config["model_id"] == identity.model_id
        return json.dumps(_patch_for(path))

    result = ChangePropagationProviderRouter(
        identity=identity,
        llm_generate=generate,
        admission_gate=_accept,
    ).route_step(
        packet,
        step_id=_model_step_id(packet),
        analytical_non_success_reason=AnalyticalNonSuccessReason.UNSUPPORTED_SYNTAX,
        current_snapshot_id="tree:candidate",
    )
    assert result.status in {
        PropagationRouteStatus.SUCCEEDED,
        PropagationRouteStatus.FALLBACK,
    }
    assert "prompt" in seen
    assert identity.model_id in json.dumps(seen["config"])
    assert result.write_performed is False  # no apply


def test_functional_facade_and_batch_model_required_only(
    roots: PropagationAuthorityRoots,
    identity: ProviderModelConfigIdentity,
) -> None:
    packet = _mixed_packet(roots)
    step_id = _model_step_id(packet)
    path = _write_path(packet)

    result = route_change_propagation_step(
        packet,
        step_id=step_id,
        analytical_non_success_reason="unsupported_kind",
        identity=identity,
        current_snapshot_id="tree:candidate",
        grok_provider=lambda _r: _patch_for(path),
        codex_provider=lambda _r: {"decision": "approve"},
        admission_gate=_accept,
    )
    assert result.envelope is not None
    assert result.envelope.step_id == step_id
    assert result.receipt is not None

    router = ChangePropagationProviderRouter(
        identity=identity,
        grok_provider=lambda _r: _patch_for(path),
        admission_gate=_accept,
    )
    # Only model-required entries are routed; missing reason rejects that step.
    batch = router.route_model_required_steps(
        packet,
        analytical_non_success_by_step={},
        current_snapshot_id="tree:candidate",
    )
    assert len(batch) == len(packet.model_required_step_ids)
    assert all(
        item.reason_code
        == PropagationProviderReason.ANALYTICAL_REASON_MISSING.value
        for item in batch
    )

    batch_ok = router.route_model_required_steps(
        packet,
        analytical_non_success_by_step={
            step_id: AnalyticalNonSuccessReason.UNSUPPORTED_SYNTAX
        },
        current_snapshot_id="tree:candidate",
    )
    assert len(batch_ok) == 1
    assert batch_ok[0].status in {
        PropagationRouteStatus.SUCCEEDED,
        PropagationRouteStatus.FALLBACK,
    }


def test_provider_cannot_claim_completion_authority(
    roots: PropagationAuthorityRoots,
    identity: ProviderModelConfigIdentity,
) -> None:
    packet = _mixed_packet(roots)
    path = _write_path(packet)
    writes: list[Any] = []
    lease = WriterLease(
        lease_id="lease:authority",
        permitted_write_paths=(path,),
        packet_id=packet.packet_id,
        plan_id=packet.plan_id,
        step_id=_model_step_id(packet),
    )
    result = ChangePropagationProviderRouter(
        identity=identity,
        grok_provider=lambda _r: {
            "proposal": {"patch": f"diff --git a/{path} b/{path}\n", "declared_paths": [path]},
            "completion_authoritative": True,
        },
        admission_gate=_accept,
        writer=lambda p, l: writes.append((p, l)),
    ).route_step(
        packet,
        step_id=_model_step_id(packet),
        analytical_non_success_reason=AnalyticalNonSuccessReason.UNSUPPORTED_SYNTAX,
        current_snapshot_id="tree:candidate",
        apply=True,
        writer_lease=lease,
    )
    assert result.write_performed is False
    assert writes == []
    assert result.completion_authoritative is False
    assert result.proof_authoritative is False


def test_model_required_step_kind_partition(
    roots: PropagationAuthorityRoots,
) -> None:
    packet = _mixed_packet(roots)
    by_id = {step.step_id: step for step in packet.steps}
    for sid in packet.model_required_step_ids:
        assert by_id[sid].kind is PropagationEditStepKind.MODEL_REQUIRED
        assert by_id[sid].plan_step_kind is PlanStepKind.LLM_BOUNDED
        assert by_id[sid].required_behavior_ids
        assert by_id[sid].write_paths


def test_bounds_and_identity_reject_non_llm_router_backend() -> None:
    with pytest.raises(PropagationProviderRoutingError):
        ProviderModelConfigIdentity(
            provider_id="p",
            model_id="m",
            config_id="c",
            router_backend="direct_openai",
        )
    bounds = PropagationProviderBounds(
        max_prompt_tokens=128,
        max_prompt_bytes=4096,
        timeout_seconds=30.0,
        allowed_tools=(),
    )
    assert bounds.to_provider_bounds().timeout_seconds == 30.0
    with pytest.raises(ValueError):
        PropagationProviderBounds(timeout_seconds=0)
