"""Hermetic adversarial evidence tests for CASF-037."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.chaos import (
    CASF_CHAOS_LOCAL_QUALIFICATION_AVAILABLE,
    CASF_CHAOS_REPORT_SCHEMA,
    CASF_CHAOS_TASK_ID,
    ChaosAttack,
    ChaosCapabilityStatus,
    ChaosDiagnosticStatus,
    ChaosDisposition,
    ChaosEvidenceBinding,
    ChaosObservation,
    ChaosProofStatus,
    ChaosReport,
    ChaosReportStatus,
    ChaosVerificationError,
    FederationChaosError,
    FederationChaosIdentity,
    FederationChaosSuite,
    bind_post_merge_validation_evidence,
    build_chaos_observation,
    build_federation_chaos_suite,
    run_closed_federation_chaos_suite,
    run_federation_chaos_suite,
)
from ipfs_accelerate_py.agent_supervisor.federation.contracts import (
    FederationOperation,
    FederationSecretError,
)
from ipfs_accelerate_py.agent_supervisor.federation.control_service import (
    FederationControlAuthorizationError,
)
from ipfs_accelerate_py.agent_supervisor.federation.event_router import (
    BoundedEventRouter,
)
from ipfs_accelerate_py.agent_supervisor.federation.formal import (
    ADVERSARIAL_PROPERTY,
    AdversarialMutation,
    HermeticCheckStatus,
    check_federation_scenario,
)
from ipfs_accelerate_py.agent_supervisor.federation.rebalancing import (
    RebalancingAuthorityError,
)
from ipfs_accelerate_py.agent_supervisor.federation.recovery import (
    RecoveryAuthorityError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.post_merge_validation import (
    build_post_merge_validation_evidence,
)

SOURCE_REVISION = "b8328ec3a9cc066acfb3240d0e4b03d16950f5c7"
SOURCE_TREE = "1e26e0b1c7d7b8df9eafb1d2e7aede6bfea19233"
ROLLBACK_REVISION = "a" * 40
ROLLBACK_TREE = "b" * 40


def _cid(label: str) -> str:
    return content_identity({"fixture": label})


def _identity() -> FederationChaosIdentity:
    return FederationChaosIdentity(
        source_revision=SOURCE_REVISION,
        source_tree=SOURCE_TREE,
        state_schema="casf-control-schema@3",
        generation_id="generation:casf-037",
        federation_id="federation:causal-event-supervisor-v1",
        policy_id="policy:CASF-PLAN-R1",
        policy_revision=1,
        capability_ids=(
            "capability:bounded-local-proof",
            "capability:typed-quack-owner",
        ),
        task_id=CASF_CHAOS_TASK_ID,
        attempt_id="attempt:casf-037-1",
        lease_id="lease:casf-037-1",
        fencing_epoch=8,
        assignment_revision=4,
        worktree_id="worktree:casf-037",
    )


def _observation(
    scenario,
    *,
    disposition: ChaosDisposition = ChaosDisposition.REJECTED,
    **changes: object,
) -> ChaosObservation:
    values: dict[str, object] = {
        "disposition": disposition,
        "evidence_refs": (_cid(f"probe:{scenario.attack.value}"),),
        "reason_code": f"contained:{scenario.attack.value}",
        "unauthorized_effect_observed": False,
        "authority_created": False,
        "completion_created": False,
    }
    values.update(changes)
    return build_chaos_observation(scenario, **values)  # type: ignore[arg-type]


def _observations(suite) -> tuple[ChaosObservation, ...]:
    return tuple(_observation(scenario) for scenario in suite.scenarios)


def _validation_payload(
    identity: FederationChaosIdentity,
    *,
    passed: bool = True,
    returncode: int = 0,
    stale: bool = False,
    target_revision: str | None = None,
    target_tree: str | None = None,
) -> dict[str, object]:
    return build_post_merge_validation_evidence(
        task_id=identity.task_id,
        target_commit=target_revision or identity.source_revision,
        validated_commit=target_revision or identity.source_revision,
        repository_tree_id=target_tree or identity.source_tree,
        validation_result={
            "attempted": True,
            "passed": passed,
            "returncode": returncode,
            "stale": stale,
            "command": "python3 -m pytest -q test/api/causal_federation/test_chaos.py",
        },
    )


def _binding(
    suite,
    *,
    observations: tuple[ChaosObservation, ...] | None = None,
    capability_statuses: tuple[ChaosCapabilityStatus, ...] | None = None,
    proof_statuses: tuple[ChaosProofStatus, ...] = (ChaosProofStatus.PASSED,),
    validation_payload: dict[str, object] | None = None,
    rollback_revision: str = ROLLBACK_REVISION,
    rollback_tree: str = ROLLBACK_TREE,
) -> ChaosEvidenceBinding:
    identity = suite.identity
    statuses = capability_statuses or tuple(
        ChaosCapabilityStatus.QUALIFIED for _item in identity.capability_ids
    )
    capability_receipts = tuple(
        _cid(f"capability:{capability_id}")
        if statuses[index] is ChaosCapabilityStatus.QUALIFIED
        else ""
        for index, capability_id in enumerate(identity.capability_ids)
    )
    proof_properties = tuple(
        f"property:casf-adversarial-safety-{index}" for index in range(len(proof_statuses))
    )
    proof_receipts = tuple(
        _cid(property_id) if proof_statuses[index] is ChaosProofStatus.PASSED else ""
        for index, property_id in enumerate(proof_properties)
    )
    validation = bind_post_merge_validation_evidence(
        validation_payload or _validation_payload(identity),
        identity=identity,
    )
    return ChaosEvidenceBinding(
        suite_id=suite.suite_id,
        validation=validation,
        rollback_revision=rollback_revision,
        rollback_tree=rollback_tree,
        rollback_generation_id="generation:pre-casf-037",
        capability_ids=identity.capability_ids,
        capability_statuses=statuses,
        capability_receipt_ids=capability_receipts,
        proof_property_ids=proof_properties,
        proof_statuses=proof_statuses,
        proof_receipt_ids=proof_receipts,
        observation_ids=(
            tuple(item.observation_id for item in observations) if observations is not None else ()
        ),
    )


def test_exact_report_schema_and_complete_catalog_cover_every_required_surface() -> None:
    first = build_federation_chaos_suite(_identity())
    second = build_federation_chaos_suite(_identity())

    assert CASF_CHAOS_REPORT_SCHEMA == "casf/adversarial-report@1"
    assert first.suite_id == second.suite_id
    assert first.to_dict() == second.to_dict()
    assert len(first.scenarios) == len(ChaosAttack) == 14
    assert tuple(item.attack for item in first.scenarios) == tuple(ChaosAttack)
    assert {
        ChaosAttack.MISSED_CAUSAL_NOTIFICATION,
        ChaosAttack.CAUSAL_INDEPENDENCE_VIOLATION,
        ChaosAttack.STALE_ABSTRACTION_SUPPRESSION,
        ChaosAttack.NON_AUTHORITATIVE_PROMOTION,
    } <= set(ChaosAttack)
    assert first.to_dict()["authority_created"] is False
    assert first.to_dict()["completion_created"] is False


def test_identity_is_exact_tamper_evident_and_secret_rejecting() -> None:
    identity = _identity()
    payload = identity.to_dict()

    assert FederationChaosIdentity.from_dict(payload) == identity
    assert json.loads(json.dumps(payload)) == payload
    payload["fencing_epoch"] = 9
    with pytest.raises(FederationChaosError, match="does not match"):
        FederationChaosIdentity.from_dict(payload)

    payload = identity.to_dict()
    payload["authority_created"] = True
    with pytest.raises(FederationChaosError, match="unknown fields"):
        FederationChaosIdentity.from_dict(payload)

    with pytest.raises(FederationChaosError, match="CASF-037"):
        replace(identity, task_id="CASF-036")
    with pytest.raises(FederationChaosError, match="credential-shaped"):
        replace(identity, capability_ids=("sk-123456789012",))
    with pytest.raises(FederationChaosError, match="not canonical"):
        replace(identity, capability_ids=tuple(reversed(identity.capability_ids)))

    reordered = identity.to_dict()
    reordered["capability_ids"] = list(reversed(reordered["capability_ids"]))
    with pytest.raises(FederationChaosError, match="malformed"):
        FederationChaosIdentity.from_dict(reordered)


def test_suite_rejects_missing_reordered_mutable_or_duplicate_catalogs() -> None:
    suite = build_federation_chaos_suite(_identity())
    with pytest.raises(FederationChaosError, match="canonical"):
        FederationChaosSuite(identity=suite.identity, scenarios=suite.scenarios[:-1])
    with pytest.raises(FederationChaosError, match="canonical"):
        FederationChaosSuite(
            identity=suite.identity,
            scenarios=(suite.scenarios[1], suite.scenarios[0], *suite.scenarios[2:]),
        )
    with pytest.raises(FederationChaosError, match="immutable"):
        FederationChaosSuite(  # type: ignore[arg-type]
            identity=suite.identity,
            scenarios=list(suite.scenarios),
        )


def test_arbitrary_injected_probe_is_diagnostic_and_never_qualifies() -> None:
    suite = build_federation_chaos_suite(_identity())
    report = run_federation_chaos_suite(suite, _observation)

    assert report.status is ChaosDiagnosticStatus.DIAGNOSTIC
    assert report.qualified is False
    assert report.to_dict()["promotion_eligible"] is False


def test_closed_runner_is_truthfully_blocked_pending_upstream_reverification() -> None:
    suite = build_federation_chaos_suite(_identity())
    report = run_closed_federation_chaos_suite(suite, _binding(suite))

    assert CASF_CHAOS_LOCAL_QUALIFICATION_AVAILABLE is False
    assert report.status is ChaosReportStatus.BLOCKED
    assert report.qualified is False
    assert report.promotion_eligible is False
    assert all(item.disposition is ChaosDisposition.BLOCKED for item in report.observations)
    assert report.to_dict()["upstream_reverification_required"] is True
    assert report.to_dict()["local_qualification_available"] is False
    assert ChaosReport.from_dict(report.to_dict()) == report


def test_actual_canonical_authorities_contain_the_full_attack_catalog() -> None:
    """Exercise the real hermetic authorities before producing diagnostics."""

    from ipfs_accelerate_py.agent_supervisor.federation.causal_abstraction import (
        CausalAbstractionAuthorityError,
        refuse_work_suppression,
    )
    from ipfs_accelerate_py.agent_supervisor.federation.scheduler import (
        SchedulerAuthorityError,
        refuse_ducklake_wake_authority,
    )
    from test.api.causal_federation.test_causal_abstraction import _map
    from test.api.causal_federation.test_control_service import (
        Authorizer,
        authorization,
        command,
        service,
    )
    from test.api.causal_federation.test_event_router import event, subscription
    from test.api.causal_federation.test_formal_models import _suite as formal_suite
    from test.api.causal_federation.test_rebalancing import _compile as compile_rebalance
    from test.api.causal_federation.test_rebalancing import _request, _spec
    from test.api.causal_federation.test_recovery import _compile as compile_recovery
    from test.api.causal_federation.test_recovery import _snapshot
    from test.api.causal_federation.test_supervisor_wake import (
        _batch,
        _event,
        _graph,
        _independence,
        _loop,
    )

    formal_mutations = {
        ChaosAttack.STALE_FENCE_MUTATION: AdversarialMutation.STALE_FENCE_MUTATION,
        ChaosAttack.DUPLICATE_AUTHORITATIVE_EFFECT: AdversarialMutation.DUPLICATE_EVENT_EFFECT,
        ChaosAttack.ILLEGAL_LIFECYCLE_TRANSITION: AdversarialMutation.ILLEGAL_LIFECYCLE_TRANSITION,
        ChaosAttack.ORPHAN_CAUSAL_PROPAGATION: AdversarialMutation.ORPHAN_CAUSAL_PROPAGATION,
    }

    def probe(scenario):
        if scenario.attack is ChaosAttack.UNAUTHORIZED_MUTATION:
            control, authorizer_, owner = service()
            with pytest.raises(FederationControlAuthorizationError):
                control.execute(command(operation=FederationOperation.CREATE))
            assert not authorizer_.commands and not owner.calls
        elif scenario.attack is ChaosAttack.CROSS_TENANT_MUTATION:
            value = command()
            authorizer_ = Authorizer()
            authorizer_.override = replace(authorization(value), tenant_id="tenant:other")
            control, _, owner = service(authorizer_=authorizer_)
            with pytest.raises(FederationControlAuthorizationError, match="scope"):
                control.execute(value)
            assert not owner.calls
        elif scenario.attack is ChaosAttack.SECRET_SHAPED_INPUT:
            with pytest.raises(FederationSecretError):
                replace(command(), target_id="federation:sk-123456789012")
        elif scenario.attack in formal_mutations:
            mutation = formal_mutations[scenario.attack]
            receipt = check_federation_scenario(
                formal_suite().scenario(ADVERSARIAL_PROPERTY[mutation]),
                mutation=mutation,
            )
            assert receipt.status is HermeticCheckStatus.COUNTEREXAMPLE
            assert receipt.counterexample is not None
        elif scenario.attack is ChaosAttack.EVENT_STORM:
            router = BoundedEventRouter(maximum_fanout_per_event=2)
            subscriptions = tuple(subscription(f"storm-{number}") for number in range(3))
            for item in subscriptions:
                router.register(item)
            routed = router.route((event(1),), now="2026-08-21T12:00:00Z")
            assert routed.enqueued_deliveries == 2
            assert routed.backpressured_subscriptions == (subscriptions[-1].subscription_id,)
        elif scenario.attack is ChaosAttack.STALE_REBALANCE:
            request = _request()
            with pytest.raises(RebalancingAuthorityError, match="fencing epoch is stale"):
                compile_rebalance(request, _spec("supervisor:idle"), expected_source_fence=1)
        elif scenario.attack is ChaosAttack.MISSED_CAUSAL_NOTIFICATION:
            receipt = _loop().process_batch(_batch((_event(1),)), graph=_graph())
            assert {"supervisor:changed", "supervisor:child"} <= set(receipt.woke_supervisor_ids)
        elif scenario.attack is ChaosAttack.CAUSAL_INDEPENDENCE_VIOLATION:
            receipt = _loop(known_receipts={"supervisor:idle": "receipt:idle"}).process_batch(
                _batch((_event(1),)),
                graph=_graph(independence=(_independence(),)),
            )
            assert receipt.asleep_supervisor_ids == ("supervisor:idle",)
            assert receipt.reused_receipt_refs == ("receipt:idle",)
        elif scenario.attack is ChaosAttack.STALE_ABSTRACTION_SUPPRESSION:
            with pytest.raises(CausalAbstractionAuthorityError, match="stale"):
                refuse_work_suppression(_map(), expected_revision=1, live_revision=2)
        elif scenario.attack is ChaosAttack.NON_AUTHORITATIVE_PROMOTION:
            with pytest.raises(SchedulerAuthorityError, match="DuckLake cannot schedule"):
                refuse_ducklake_wake_authority({"authoritative": True})
        elif scenario.attack is ChaosAttack.CRASH_RECOVERY_REPLAY:
            with pytest.raises(RecoveryAuthorityError, match="process exit cannot complete"):
                compile_recovery(_snapshot(claimed_complete=True))
        else:
            raise AssertionError(f"unmapped chaos attack: {scenario.attack}")
        return _observation(scenario)

    report = run_federation_chaos_suite(
        build_federation_chaos_suite(_identity()),
        probe,
    )
    assert report.status is ChaosDiagnosticStatus.DIAGNOSTIC
    assert {item.attack for item in report.observations} == set(ChaosAttack)
    assert report.qualified is False


@pytest.mark.parametrize(
    "field",
    ("unauthorized_effect_observed", "authority_created", "completion_created"),
)
def test_any_adversarial_escape_fails_closed(field: str) -> None:
    suite = build_federation_chaos_suite(_identity())

    def probe(scenario):
        return _observation(
            scenario,
            **{field: scenario.attack is ChaosAttack.STALE_FENCE_MUTATION},
        )

    with pytest.raises(ChaosVerificationError, match="attack"):
        run_federation_chaos_suite(suite, probe)


def test_missing_capability_and_unavailable_proof_remain_blocked() -> None:
    suite = build_federation_chaos_suite(_identity())
    missing = (
        ChaosCapabilityStatus.MISSING,
        ChaosCapabilityStatus.QUALIFIED,
    )
    capability_report = run_closed_federation_chaos_suite(
        suite,
        _binding(suite, capability_statuses=missing),
    )
    proof_report = run_closed_federation_chaos_suite(
        suite,
        _binding(suite, proof_statuses=(ChaosProofStatus.UNAVAILABLE,)),
    )

    assert capability_report.status is ChaosReportStatus.BLOCKED
    assert proof_report.status is ChaosReportStatus.BLOCKED


def test_bad_probe_bindings_and_probe_failures_cannot_create_a_report() -> None:
    suite = build_federation_chaos_suite(_identity())
    first = _observation(suite.scenarios[0])

    with pytest.raises(ChaosVerificationError, match="does not bind"):
        run_federation_chaos_suite(suite, lambda _scenario: first)
    with pytest.raises(RuntimeError, match="owner unavailable"):
        run_federation_chaos_suite(
            suite,
            lambda _scenario: (_ for _ in ()).throw(RuntimeError("owner unavailable")),
        )


def test_observation_contract_rejects_mutability_duplicates_and_secrets() -> None:
    scenario = build_federation_chaos_suite(_identity()).scenarios[0]
    ordered_refs = tuple(sorted((_cid("one"), _cid("two"))))
    with pytest.raises(FederationChaosError, match="bounds"):
        _observation(scenario, evidence_refs=())
    with pytest.raises(FederationChaosError, match="immutable"):
        _observation(scenario, evidence_refs=[_cid("one")])  # type: ignore[arg-type]
    with pytest.raises(FederationChaosError, match="duplicates"):
        _observation(scenario, evidence_refs=(_cid("one"), _cid("one")))
    with pytest.raises(FederationChaosError, match="not canonical"):
        _observation(scenario, evidence_refs=tuple(reversed(ordered_refs)))
    with pytest.raises(FederationChaosError, match="credential-shaped"):
        _observation(scenario, reason_code="sk-123456789012")
    with pytest.raises(FederationChaosError, match="boolean"):
        _observation(scenario, unauthorized_effect_observed=1)


def test_forged_all_false_observations_cannot_self_qualify() -> None:
    suite = build_federation_chaos_suite(_identity())
    caller_authored = _observations(suite)
    evidence = _binding(suite, observations=caller_authored)

    report = ChaosReport(
        suite=suite,
        evidence=evidence,
        observations=caller_authored,
    )

    assert report.status is ChaosReportStatus.BLOCKED
    assert report.qualified is False
    assert report.to_dict()["local_qualification_available"] is False


def test_report_rejects_mutable_duplicate_or_foreign_observation_population() -> None:
    suite = build_federation_chaos_suite(_identity())
    observations = _observations(suite)
    evidence = _binding(suite, observations=observations)

    with pytest.raises(FederationChaosError, match="immutable"):
        ChaosReport(  # type: ignore[arg-type]
            suite=suite,
            evidence=evidence,
            observations=list(observations),
        )
    duplicate = (observations[0], observations[0], *observations[2:])
    with pytest.raises(ChaosVerificationError, match="does not bind"):
        ChaosReport(suite=suite, evidence=evidence, observations=duplicate)


def test_validation_binding_rejects_wrong_current_tree_and_tampering() -> None:
    identity = _identity()
    wrong = _validation_payload(identity, target_revision="c" * 40)
    with pytest.raises(ChaosVerificationError, match="commit_binding_mismatch"):
        bind_post_merge_validation_evidence(wrong, identity=identity)

    tampered = _validation_payload(identity)
    tampered["returncode"] = 1
    with pytest.raises(ChaosVerificationError, match="receipt_id_mismatch"):
        bind_post_merge_validation_evidence(tampered, identity=identity)


def test_validation_proof_and_rollback_are_content_bound_but_non_authoritative() -> None:
    suite = build_federation_chaos_suite(_identity())
    first = run_closed_federation_chaos_suite(suite, _binding(suite))
    second = run_closed_federation_chaos_suite(
        suite,
        _binding(suite, rollback_tree="d" * 40),
    )

    assert first.report_id != second.report_id
    assert first.evidence.validation.receipt_id
    assert first.evidence.proof_receipt_ids
    assert first.evidence.rollback_revision == ROLLBACK_REVISION
    assert first.to_dict()["promotion_eligible"] is False

    with pytest.raises(FederationChaosError, match="rollback target"):
        run_closed_federation_chaos_suite(
            suite,
            _binding(suite, rollback_revision=SOURCE_REVISION),
        )
    with pytest.raises(FederationChaosError, match="active generation"):
        run_closed_federation_chaos_suite(
            suite,
            replace(
                _binding(suite),
                rollback_generation_id=suite.identity.generation_id,
            ),
        )


def test_wire_report_is_closed_tamper_evident_and_cannot_claim_promotion() -> None:
    suite = build_federation_chaos_suite(_identity())
    report = run_closed_federation_chaos_suite(suite, _binding(suite))
    payload = report.to_dict()

    assert ChaosReport.from_dict(payload) == report

    for field, value in (
        ("status", "qualified"),
        ("authority_created", True),
        ("promotion_eligible", True),
        ("local_qualification_available", True),
    ):
        forged = {**payload, field: value}
        with pytest.raises(FederationChaosError):
            ChaosReport.from_dict(forged)
