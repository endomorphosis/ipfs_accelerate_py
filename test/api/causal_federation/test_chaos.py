"""Hermetic adversarial and chaos qualification for CASF-037."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.federation.chaos import (
    CASF_CHAOS_TASK_ID,
    ChaosAttack,
    ChaosDisposition,
    ChaosObservation,
    ChaosReportStatus,
    ChaosVerificationError,
    FederationChaosError,
    FederationChaosIdentity,
    FederationChaosSuite,
    build_federation_chaos_suite,
    run_federation_chaos_suite,
)
from ipfs_accelerate_py.agent_supervisor.federation.contracts import (
    FederationOperation,
    FederationSecretError,
)
from ipfs_accelerate_py.agent_supervisor.federation.control_service import (
    FederationControlAuthorizationError,
)
from ipfs_accelerate_py.agent_supervisor.federation.event_router import BoundedEventRouter
from ipfs_accelerate_py.agent_supervisor.federation.formal import (
    ADVERSARIAL_PROPERTY,
    AdversarialMutation,
    HermeticCheckStatus,
    check_federation_scenario,
)
from ipfs_accelerate_py.agent_supervisor.federation.rebalancing import RebalancingAuthorityError
from ipfs_accelerate_py.agent_supervisor.federation.recovery import RecoveryAuthorityError


SOURCE_REVISION = "b8328ec3a9cc066acfb3240d0e4b03d16950f5c7"
SOURCE_TREE = "1e26e0b1c7d7b8df9eafb1d2e7aede6bfea19233"


def _identity() -> FederationChaosIdentity:
    return FederationChaosIdentity(
        source_revision=SOURCE_REVISION,
        source_tree=SOURCE_TREE,
        state_schema="casf-control-schema@3",
        generation_id="generation:casf-037",
        federation_id="federation:causal-event-supervisor-v1",
        policy_id="policy:CASF-PLAN-R1",
        capability_ids=("capability:typed-quack-owner", "capability:bounded-local-proof"),
        task_id=CASF_CHAOS_TASK_ID,
        attempt_id="attempt:casf-037-1",
        lease_id="lease:casf-037-1",
        fencing_epoch=8,
        assignment_revision=4,
        worktree_id="worktree:casf-037",
    )


def _safe_observation(
    scenario, *, disposition: ChaosDisposition = ChaosDisposition.REJECTED, **changes: object
) -> ChaosObservation:
    values: dict[str, object] = {
        "scenario_id": scenario.scenario_id,
        "attack": scenario.attack,
        "disposition": disposition,
        "effect_observed": False,
        "authority_created": False,
        "completion_created": False,
        "evidence_refs": (f"evidence:{scenario.attack.value}",),
        "reason_code": f"contained:{scenario.attack.value}",
    }
    values.update(changes)
    return ChaosObservation(**values)  # type: ignore[arg-type]


def test_complete_catalog_covers_every_required_adversarial_surface() -> None:
    first = build_federation_chaos_suite(_identity())
    second = build_federation_chaos_suite(_identity())

    assert first.suite_id == second.suite_id
    assert first.to_dict() == second.to_dict()
    assert {scenario.attack for scenario in first.scenarios} == set(ChaosAttack)
    assert {scenario.domain.value for scenario in first.scenarios} == {
        "authorization",
        "tenancy",
        "secrets",
        "leases_and_fences",
        "events",
        "lifecycle",
        "rebalance",
        "causal_propagation",
        "recovery",
    }
    assert first.to_dict()["authority_created"] is False
    assert first.to_dict()["completion_created"] is False


def test_identity_is_exact_tamper_evident_and_rejects_unknown_fields() -> None:
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


def test_incomplete_or_duplicate_attack_catalog_fails_closed() -> None:
    suite = build_federation_chaos_suite(_identity())
    with pytest.raises(FederationChaosError, match="incomplete"):
        FederationChaosSuite(identity=suite.identity, scenarios=suite.scenarios[:-1])
    with pytest.raises(FederationChaosError, match="repeats"):
        FederationChaosSuite(
            identity=suite.identity,
            scenarios=suite.scenarios[:-1] + (suite.scenarios[0],),
        )


def test_all_attacks_must_be_contained_without_effect_authority_or_completion() -> None:
    suite = build_federation_chaos_suite(_identity())
    report = run_federation_chaos_suite(suite, _safe_observation)

    assert report.status is ChaosReportStatus.QUALIFIED
    assert report.qualified is True
    assert len(report.observations) == len(ChaosAttack)
    assert all(item.disposition is ChaosDisposition.REJECTED for item in report.observations)
    assert all(not item.effect_observed for item in report.observations)
    assert all(not item.authority_created for item in report.observations)
    assert all(not item.completion_created for item in report.observations)
    assert report.to_dict()["authority_created"] is False
    assert report.to_dict()["completion_created"] is False
    assert report.to_dict()["bounded"] is True


def test_actual_canonical_authorities_contain_the_full_attack_catalog() -> None:
    """Tie each abstract attack to the existing closed implementation path."""

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
                formal_suite().scenario(ADVERSARIAL_PROPERTY[mutation]), mutation=mutation
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
        elif scenario.attack is ChaosAttack.CRASH_RECOVERY_REPLAY:
            with pytest.raises(RecoveryAuthorityError, match="process exit cannot complete"):
                compile_recovery(_snapshot(claimed_complete=True))
        else:  # Kept exhaustive as the enum is a closed safety catalog.
            raise AssertionError(f"unmapped chaos attack: {scenario.attack}")
        return _safe_observation(scenario)

    report = run_federation_chaos_suite(build_federation_chaos_suite(_identity()), probe)
    assert report.status is ChaosReportStatus.QUALIFIED
    assert {item.attack for item in report.observations} == set(ChaosAttack)


@pytest.mark.parametrize(
    "field",
    ("effect_observed", "authority_created", "completion_created"),
)
def test_any_adversarial_escape_fails_closed(field: str) -> None:
    suite = build_federation_chaos_suite(_identity())

    def probe(scenario):
        return _safe_observation(scenario, **{field: scenario.attack is ChaosAttack.STALE_FENCE_MUTATION})

    with pytest.raises(ChaosVerificationError, match="attack"):
        run_federation_chaos_suite(suite, probe)


def test_missing_capability_is_blocked_never_reported_as_qualified() -> None:
    suite = build_federation_chaos_suite(_identity())

    def probe(scenario):
        disposition = (
            ChaosDisposition.BLOCKED
            if scenario.attack is ChaosAttack.EVENT_STORM
            else ChaosDisposition.REJECTED
        )
        return _safe_observation(scenario, disposition=disposition)

    report = run_federation_chaos_suite(suite, probe)
    assert report.status is ChaosReportStatus.BLOCKED
    assert report.qualified is False


def test_bad_probe_bindings_and_probe_failures_cannot_create_a_report() -> None:
    suite = build_federation_chaos_suite(_identity())

    def wrong_attack(scenario):
        return _safe_observation(scenario, attack=ChaosAttack.EVENT_STORM)

    with pytest.raises(ChaosVerificationError, match="does not bind"):
        run_federation_chaos_suite(suite, wrong_attack)

    with pytest.raises(RuntimeError, match="owner unavailable"):
        run_federation_chaos_suite(suite, lambda _scenario: (_ for _ in ()).throw(RuntimeError("owner unavailable")))


def test_observation_contract_rejects_unbounded_or_ambiguous_evidence() -> None:
    scenario = build_federation_chaos_suite(_identity()).scenarios[0]
    with pytest.raises(FederationChaosError, match="requires evidence"):
        _safe_observation(scenario, evidence_refs=())
    with pytest.raises(FederationChaosError, match="repeats"):
        _safe_observation(scenario, evidence_refs=("evidence:one", "evidence:one"))
    with pytest.raises(FederationChaosError, match="boolean"):
        _safe_observation(scenario, effect_observed=1)


def test_suite_observation_and_report_wire_records_are_closed_and_tamper_evident() -> None:
    suite = build_federation_chaos_suite(_identity())
    observation = _safe_observation(suite.scenarios[0])
    report = run_federation_chaos_suite(suite, _safe_observation)

    assert FederationChaosSuite.from_dict(suite.to_dict()) == suite
    assert ChaosObservation.from_dict(observation.to_dict()) == observation
    assert type(report).from_dict(report.to_dict()) == report

    payload = report.to_dict()
    payload["authority_created"] = True
    with pytest.raises(FederationChaosError, match="cannot claim authority"):
        type(report).from_dict(payload)
