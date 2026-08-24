"""Fail-closed registry drift, recovery, rollback, and supersession tests."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ProcedureContractError,
    ProcedureDriftReport,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.recovery import (
    DriftDimension,
    DriftDisposition,
    ProcedureDriftMonitor,
    ProcedureDriftObservation,
    ProcedureRecoveryError,
    ProcedureRecoveryPlanner,
    ProcedureRollbackFailure,
    ProcedureRollbackService,
    RegistryRecoveryPlan,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.registry import (
    DRIFT_ACTOR_ID,
    InMemoryProcedureRegistryStore,
    RegistryAuthorizationError,
    RegistryCASError,
    RegistryCASOutcome,
    RegistryLifecycleState,
    RegistryOperation,
)


def _load_registry_helpers():
    path = Path(__file__).with_name("test_registry.py")
    spec = importlib.util.spec_from_file_location("_pcpc026_registry_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load registry test helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_helpers = _load_registry_helpers()


def _promoted_registry():
    procedure, _candidate, certificate, context = _helpers.issue_for()
    store = InMemoryProcedureRegistryStore()
    registry = _helpers.make_registry(context, store)
    registered = _helpers.register_spec(registry, procedure, certificate)
    promoted = _helpers.promote_head(registry, registered, procedure)
    return registry, store, procedure, promoted


def _observation(
    procedure,
    revision_id: str,
    dimension: DriftDimension,
    *,
    suffix: str = "one",
    observer_id: str = "independent-drift-observer@1",
) -> ProcedureDriftObservation:
    return ProcedureDriftObservation(
        procedure_id=procedure.name,
        expected_revision_id=revision_id,
        dimension=dimension,
        observer_id=observer_id,
        expected_cid=f"expected-{dimension.value}-{suffix}",
        observed_cid=f"observed-{dimension.value}-{suffix}",
        evidence_cids=(f"evidence-{dimension.value}-{suffix}",),
        observed_at_ms=101,
    )


def _register_replacement(registry, procedure, promoted):
    replacement = replace(
        _helpers.valid_spec(), version=_helpers.ProcedureVersion(major=2)
    )
    _spec, _candidate, certificate, _context = _helpers.issue_for(replacement)
    registered = _helpers.register_spec(
        registry,
        replacement,
        certificate,
        expected_old_revision_id=promoted.revision.revision_id,
    )
    return replacement, registered


def _promote_replacement(registry, procedure, promoted):
    replacement, registered = _register_replacement(registry, procedure, promoted)
    replacement_promoted = registry.promote(
        procedure_id=procedure.name,
        target_procedure_cid=replacement.content_id,
        expected_old_revision_id=promoted.revision.revision_id,
        rollback_target_revision_id=promoted.revision.revision_id,
        authorization=_helpers.auth(
            RegistryOperation.PROMOTE,
            replacement.content_id,
            expected_old=promoted.revision.revision_id,
            target_revision=registered.revision.revision_id,
        ),
        now_ms=101,
    )
    return replacement, registered, replacement_promoted


@pytest.mark.parametrize("dimension", tuple(DriftDimension))
def test_every_execution_relevant_drift_removes_usability_and_preserves_history(
    dimension: DriftDimension,
) -> None:
    registry, store, procedure, promoted = _promoted_registry()
    original_payload = promoted.revision.to_dict()
    observation = _observation(
        procedure, promoted.revision.revision_id, dimension
    )

    result = ProcedureDriftMonitor(registry).observe(observation)

    expected_disposition = (
        DriftDisposition.REVOKED
        if dimension
        in {
            DriftDimension.AUTHORITY,
            DriftDimension.EFFECT,
            DriftDimension.BOUNDARY,
            DriftDimension.OBSERVED_FAILURE,
        }
        else DriftDisposition.STALE
    )
    expected_state = RegistryLifecycleState(expected_disposition.value)
    assert result.disposition is expected_disposition
    assert result.usable is False
    assert result.mutation.cas.accepted
    assert result.mutation.revision.state is expected_state
    assert result.mutation.revision.actor_id == DRIFT_ACTOR_ID
    assert result.mutation.revision.expected_old_revision_id == promoted.revision.revision_id
    assert registry.lookup_exact(procedure.content_id) is None
    assert registry.choose_version(procedure.name) is None
    assert registry.get_revision(promoted.revision.revision_id).to_dict() == original_payload

    assert isinstance(result.report, ProcedureDriftReport)
    assert result.report.state.value == expected_state.value
    assert result.report.facts["drift_cause"] == dimension.value
    assert result.report.facts["observer_id"] == observation.observer_id
    assert promoted.revision.revision_id in result.report.reference_cids
    assert result.mutation.revision.revision_id in result.report.reference_cids
    event = store.events()[-1]
    assert event["drift_cause"] == dimension.value
    assert event["drift_report_cid"] == observation.content_id


def test_drift_observation_is_canonical_closed_and_independent() -> None:
    registry, _store, procedure, promoted = _promoted_registry()
    observation = _observation(
        procedure, promoted.revision.revision_id, DriftDimension.CONTRACT
    )
    assert ProcedureDriftObservation.from_dict(observation.to_dict()) == observation

    unknown = dict(observation.to_dict())
    unknown["unreviewed_disposition"] = "promoted"
    with pytest.raises(ProcedureContractError, match="unsupported fields"):
        ProcedureDriftObservation.from_dict(unknown)
    with pytest.raises(ProcedureRecoveryError, match="changed value"):
        replace(observation, observed_cid=observation.expected_cid)
    with pytest.raises(ProcedureContractError, match="must not be empty"):
        replace(observation, evidence_cids=())
    with pytest.raises(ProcedureRecoveryError, match="internal drift actor"):
        replace(observation, observer_id=DRIFT_ACTOR_ID)
    with pytest.raises(ProcedureRecoveryError, match="internal drift actor"):
        replace(observation, observer_id=DRIFT_ACTOR_ID.upper())
    with pytest.raises(ProcedureRecoveryError, match="own drift"):
        ProcedureDriftMonitor(registry).observe(
            replace(observation, observer_id=procedure.name)
        )
    assert registry.get(procedure.name, demote_stale=False) == promoted.revision


def test_obsolete_drift_observation_loses_expected_old_cas() -> None:
    registry, _store, procedure, promoted = _promoted_registry()
    history_before = registry.history(procedure.name)
    observation = _observation(
        procedure,
        "obsolete-registry-revision",
        DriftDimension.POLICY,
    )

    with pytest.raises(RegistryCASError, match="expected-old") as caught:
        ProcedureDriftMonitor(registry).observe(observation)

    assert caught.value.cas is not None
    assert caught.value.cas.stale
    assert caught.value.cas.observed_revision_id == promoted.revision.revision_id
    assert registry.history(procedure.name) == history_before
    assert registry.lookup_exact(procedure.content_id) == promoted.revision


def test_repeated_drift_is_noop_but_severe_new_drift_revokes() -> None:
    registry, _store, procedure, promoted = _promoted_registry()
    monitor = ProcedureDriftMonitor(registry)
    stale = monitor.observe(
        _observation(
            procedure, promoted.revision.revision_id, DriftDimension.POLICY
        )
    )
    history_after_stale = registry.history(procedure.name)

    repeated = monitor.observe(
        _observation(
            procedure,
            stale.mutation.revision.revision_id,
            DriftDimension.POLICY,
            suffix="two",
        )
    )
    assert repeated.mutation.cas.outcome is RegistryCASOutcome.NOOP
    assert repeated.report.facts["no_op"] is True
    assert registry.history(procedure.name) == history_after_stale

    revoked = monitor.observe(
        _observation(
            procedure,
            stale.mutation.revision.revision_id,
            DriftDimension.AUTHORITY,
            suffix="three",
        )
    )
    assert revoked.mutation.revision.state is RegistryLifecycleState.REVOKED
    assert len(registry.history(procedure.name)) == len(history_after_stale) + 1
    assert registry.lookup_exact(procedure.content_id) is None


def test_supersession_requires_exact_independent_authorization() -> None:
    registry, _store, procedure, promoted = _promoted_registry()
    expected = promoted.revision.revision_id

    with pytest.raises(RegistryAuthorizationError, match="operation"):
        registry.supersede(
            procedure_id=procedure.name,
            expected_old_revision_id=expected,
            authorization=_helpers.auth(
                RegistryOperation.DEMOTE,
                procedure.content_id,
                expected_old=expected,
                target_revision=expected,
            ),
        )
    with pytest.raises(RegistryAuthorizationError, match="drift actor"):
        registry.supersede(
            procedure_id=procedure.name,
            expected_old_revision_id=expected,
            authorization=_helpers.auth(
                RegistryOperation.SUPERSEDE,
                procedure.content_id,
                expected_old=expected,
                target_revision=expected,
                actor_id=DRIFT_ACTOR_ID,
            ),
        )
    with pytest.raises(RegistryAuthorizationError, match="drift actor"):
        registry.supersede(
            procedure_id=procedure.name,
            expected_old_revision_id=expected,
            authorization=_helpers.auth(
                RegistryOperation.SUPERSEDE,
                procedure.content_id,
                expected_old=expected,
                target_revision=expected,
                actor_id=DRIFT_ACTOR_ID.upper(),
            ),
        )
    with pytest.raises(ProcedureContractError, match="use supersede"):
        registry.advance(
            procedure_id=procedure.name,
            next_state=RegistryLifecycleState.SUPERSEDED,
            expected_old_revision_id=expected,
            authorization=_helpers.auth(
                RegistryOperation.ADVANCE,
                procedure.content_id,
                expected_old=expected,
                target_revision=expected,
            ),
        )
    with pytest.raises(ProcedureContractError, match="stale or degraded"):
        registry.demote(
            procedure_id=procedure.name,
            expected_old_revision_id=expected,
            reason_state=RegistryLifecycleState.SUPERSEDED,
        )
    assert registry.get(procedure.name, demote_stale=False) == promoted.revision


def test_supersession_and_replacement_are_append_only_with_exact_rollback() -> None:
    registry, store, procedure, promoted = _promoted_registry()
    replacement, registered = _register_replacement(registry, procedure, promoted)
    original_payload = promoted.revision.to_dict()
    candidate_payload = registered.revision.to_dict()

    superseded = registry.supersede(
        procedure_id=procedure.name,
        expected_old_revision_id=promoted.revision.revision_id,
        authorization=_helpers.auth(
            RegistryOperation.SUPERSEDE,
            procedure.content_id,
            expected_old=promoted.revision.revision_id,
            target_revision=promoted.revision.revision_id,
        ),
        now_ms=102,
    )
    assert superseded.revision.state is RegistryLifecycleState.SUPERSEDED
    assert superseded.revision.operation is RegistryOperation.SUPERSEDE
    assert superseded.revision.generation > registered.revision.generation
    assert registry.lookup_exact(procedure.content_id) is None
    assert registry.get_revision(promoted.revision.revision_id).to_dict() == original_payload
    assert registry.get_revision(registered.revision.revision_id).to_dict() == candidate_payload
    event = store.events()[-1]
    assert event["superseded_revision_id"] == promoted.revision.revision_id
    assert event["superseding_revision_id"] == superseded.revision.revision_id

    promoted_replacement = registry.promote(
        procedure_id=procedure.name,
        target_procedure_cid=replacement.content_id,
        expected_old_revision_id=superseded.revision.revision_id,
        rollback_target_revision_id=promoted.revision.revision_id,
        authorization=_helpers.auth(
            RegistryOperation.PROMOTE,
            replacement.content_id,
            expected_old=superseded.revision.revision_id,
            target_revision=registered.revision.revision_id,
        ),
        now_ms=103,
    )
    assert promoted_replacement.revision.state is RegistryLifecycleState.PROMOTED
    assert (
        promoted_replacement.revision.rollback_target_revision_id
        == promoted.revision.revision_id
    )
    assert registry.lookup_exact(replacement.content_id) == promoted_replacement.revision
    assert registry.lookup_exact(procedure.content_id) is None
    assert registry.get_revision(promoted.revision.revision_id).to_dict() == original_payload


def test_recovery_plans_and_rolls_back_the_exact_recorded_target() -> None:
    registry, store, procedure, promoted = _promoted_registry()
    replacement, _registered, replacement_promoted = _promote_replacement(
        registry, procedure, promoted
    )
    drifted = ProcedureDriftMonitor(registry).observe(
        _observation(
            procedure,
            replacement_promoted.revision.revision_id,
            DriftDimension.OBSERVED_FAILURE,
        )
    )
    assert registry.lookup_exact(replacement.content_id) is None

    plan = ProcedureRecoveryPlanner(registry).plan(procedure.name)
    assert RegistryRecoveryPlan.from_dict(plan.to_dict()) == plan
    assert plan.expected_head_revision_id == drifted.mutation.revision.revision_id
    assert plan.rollback_target_revision_id == promoted.revision.revision_id
    assert plan.target_procedure_cid == procedure.content_id

    rolled_back = ProcedureRollbackService(registry).rollback(
        plan,
        authorization=_helpers.auth(
            RegistryOperation.ROLLBACK,
            procedure.content_id,
            expected_old=drifted.mutation.revision.revision_id,
            target_revision=promoted.revision.revision_id,
        ),
        now_ms=103,
    )
    assert rolled_back.cas.accepted
    assert rolled_back.revision.state is RegistryLifecycleState.PROMOTED
    assert rolled_back.revision.procedure_cid == procedure.content_id
    assert registry.lookup_exact(procedure.content_id) == rolled_back.revision
    assert registry.lookup_exact(replacement.content_id) is None
    assert not store.quarantined()


def test_missing_or_forged_recovery_target_is_refused_and_quarantined() -> None:
    registry, store, procedure, promoted = _promoted_registry()
    with pytest.raises(ProcedureRecoveryError, match="no recorded rollback target"):
        ProcedureRecoveryPlanner(registry).plan(procedure.name)

    replacement, registered, replacement_promoted = _promote_replacement(
        registry, procedure, promoted
    )
    plan = ProcedureRecoveryPlanner(registry).plan(procedure.name)
    forged = replace(
        plan,
        rollback_target_revision_id=registered.revision.revision_id,
        target_procedure_cid=replacement.content_id,
        target_revision_generation=registered.revision.generation,
    )
    with pytest.raises(ProcedureRollbackFailure) as caught:
        ProcedureRollbackService(registry).rollback(
            forged,
            authorization=_helpers.auth(
                RegistryOperation.ROLLBACK,
                replacement.content_id,
                expected_old=replacement_promoted.revision.revision_id,
                target_revision=registered.revision.revision_id,
            ),
        )
    assert caught.value.reason_code == "recovery_plan_drift"
    quarantine = store.quarantined()[-1]
    assert quarantine["kind"] == "rollback_failure"
    assert quarantine["recovery_plan_cid"] == forged.content_id
    assert registry.get(procedure.name, demote_stale=False) == replacement_promoted.revision


def test_wrong_rollback_authorization_is_quarantined_without_mutation() -> None:
    registry, store, procedure, promoted = _promoted_registry()
    replacement, registered, replacement_promoted = _promote_replacement(
        registry, procedure, promoted
    )
    plan = ProcedureRecoveryPlanner(registry).plan(procedure.name)

    with pytest.raises(ProcedureRollbackFailure) as caught:
        ProcedureRollbackService(registry).rollback(
            plan,
            authorization=_helpers.auth(
                RegistryOperation.ROLLBACK,
                replacement.content_id,
                expected_old=replacement_promoted.revision.revision_id,
                target_revision=registered.revision.revision_id,
            ),
        )
    assert caught.value.reason_code == "authorization_refused"
    assert store.quarantined()[-1]["reason_code"] == "authorization_refused"
    assert registry.get(procedure.name, demote_stale=False) == replacement_promoted.revision
    assert registry.lookup_exact(replacement.content_id) == replacement_promoted.revision


def test_stale_recovery_plan_is_refused_and_quarantined_without_rollback() -> None:
    registry, store, procedure, promoted = _promoted_registry()
    replacement, _registered, replacement_promoted = _promote_replacement(
        registry, procedure, promoted
    )
    plan = ProcedureRecoveryPlanner(registry).plan(procedure.name)
    drifted = ProcedureDriftMonitor(registry).observe(
        _observation(
            procedure,
            replacement_promoted.revision.revision_id,
            DriftDimension.POLICY,
        )
    )

    with pytest.raises(ProcedureRollbackFailure) as caught:
        ProcedureRollbackService(registry).rollback(
            plan,
            authorization=_helpers.auth(
                RegistryOperation.ROLLBACK,
                procedure.content_id,
                expected_old=replacement_promoted.revision.revision_id,
                target_revision=promoted.revision.revision_id,
            ),
        )

    assert caught.value.reason_code == "recovery_plan_drift"
    assert store.quarantined()[-1]["reason_code"] == "recovery_plan_drift"
    assert registry.get(procedure.name, demote_stale=False) == drifted.mutation.revision
    assert registry.lookup_exact(replacement.content_id) is None


def test_failed_post_rollback_verification_revokes_and_quarantines() -> None:
    class FailingVerificationRollbackService(ProcedureRollbackService):
        def _verified_result(self, plan, mutation):
            return False

    registry, store, procedure, promoted = _promoted_registry()
    _replacement, _registered, replacement_promoted = _promote_replacement(
        registry, procedure, promoted
    )
    plan = ProcedureRecoveryPlanner(registry).plan(procedure.name)

    with pytest.raises(ProcedureRollbackFailure) as caught:
        FailingVerificationRollbackService(registry).rollback(
            plan,
            authorization=_helpers.auth(
                RegistryOperation.ROLLBACK,
                procedure.content_id,
                expected_old=replacement_promoted.revision.revision_id,
                target_revision=promoted.revision.revision_id,
            ),
            now_ms=103,
        )
    assert caught.value.reason_code == "rollback_verification_failed"
    current = registry.get(procedure.name, demote_stale=False)
    assert current.state is RegistryLifecycleState.REVOKED
    assert registry.lookup_exact(procedure.content_id) is None
    quarantine = store.quarantined()[-1]
    assert quarantine["committed_revision_id"]
    assert quarantine["containment_revision_id"] == current.revision_id
