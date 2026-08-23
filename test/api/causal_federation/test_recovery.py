"""Hermetic tests for CASF crash recovery, fence CAS, and effect reconciliation."""

from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.events import EventEffectClass
from ipfs_accelerate_py.agent_supervisor.federation.recovery import (
    CrashSnapshot,
    EffectReconciliation,
    FederationRecoveryCoordinator,
    InFlightEffect,
    RecoveryAuthorityError,
    RecoveryError,
    RecoveryStore,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    open_embedded_client,
)
from test.api.causal_federation.test_contracts import sample_binding
from test.api.causal_federation.test_registry import _create
from test.api.causal_federation.test_trigger import sample_policy, sample_request


def _snapshot(**overrides: object) -> CrashSnapshot:
    values: dict[str, object] = {
        "subject_kind": "supervisor",
        "subject_id": "supervisor:crashed",
        "process_birth_id": "birth:crashed",
        "lease_id": "lease:crashed",
        "fencing_epoch": 3,
        "event_cursor": 7,
        "lifecycle_state": "FAILED",
        "preserved_attempt_ids": ("attempt:failed",),
        "checkpoint_ref": "checkpoint:crashed",
        "process_exited": True,
    }
    values.update(overrides)
    return CrashSnapshot(**values)  # type: ignore[arg-type]


def _compile(snapshot: CrashSnapshot, **kwargs: object):
    coordinator = FederationRecoveryCoordinator()
    values: dict[str, object] = {
        "binding": sample_binding(),
        "recovered_subject_id": "supervisor:recovered",
        "recovered_process_birth_id": "birth:recovered",
        "recovered_lease_id": "lease:recovered",
        "expected_fence": snapshot.fencing_epoch,
    }
    values.update(kwargs)
    return coordinator.compile(snapshot, **values)  # type: ignore[arg-type]


def _recover(plan, **kwargs: object):
    coordinator = FederationRecoveryCoordinator()
    values: dict[str, object] = {
        "binding": sample_binding(),
        "expected_fence": plan.previous_fencing_epoch,
    }
    values.update(kwargs)
    return coordinator.recover(plan, **values)  # type: ignore[arg-type]


def test_crash_reconnects_with_fresh_identity_and_replays_durable_cursor() -> None:
    snapshot = _snapshot()
    plan = _compile(snapshot)
    assert plan.recovered_subject_id == "supervisor:recovered"
    assert plan.recovered_process_birth_id == "birth:recovered"
    assert plan.replay_cursor == 7
    assert plan.fencing_epoch == 4
    assert plan.preserved_attempt_ids == ("attempt:failed",)
    assert plan.recovered_lifecycle == "RECOVERING"
    receipt = _recover(plan)
    assert receipt.outcome == "recovered"
    assert receipt.replay_cursor == 7
    assert receipt.fencing_epoch == 4
    assert receipt.preserved_attempt_ids == ("attempt:failed",)
    assert receipt.recovered_subject_id != snapshot.subject_id


def test_stale_fence_fails_closed() -> None:
    snapshot = _snapshot()
    with pytest.raises(RecoveryAuthorityError, match="fencing epoch is stale"):
        _compile(snapshot, expected_fence=1)
    plan = _compile(snapshot)
    with pytest.raises(RecoveryAuthorityError, match="fencing epoch is stale"):
        _recover(plan, expected_fence=1)


def test_unknown_effects_must_reconcile_before_retry() -> None:
    effect = InFlightEffect(
        effect_id="effect:open",
        effect_class=EventEffectClass.READ_ONLY.value,
        attempt_id="attempt:open",
        task_id="task:open",
    )
    snapshot = _snapshot(in_flight_effects=(effect,))
    with pytest.raises(RecoveryAuthorityError, match="unknown effects must reconcile"):
        _compile(snapshot)
    plan = _compile(
        snapshot,
        reconciliations=(
            EffectReconciliation(
                effect_id="effect:open",
                disposition="absent",
                observation_ref="observation:absent",
                evidence_ref="evidence:absent",
            ),
        ),
    )
    assert plan.replay_effect_ids == ("effect:open",)
    assert "attempt:open" in plan.preserved_attempt_ids
    receipt = _recover(plan)
    assert receipt.replay_effect_ids == ("effect:open",)


def test_irreversible_in_flight_cannot_replay() -> None:
    effect = InFlightEffect(
        effect_id="effect:pay",
        effect_class=EventEffectClass.PAYMENT.value,
        attempt_id="attempt:pay",
        task_id="task:pay",
    )
    snapshot = _snapshot(in_flight_effects=(effect,))
    with pytest.raises(RecoveryAuthorityError, match="irreversible"):
        _compile(
            snapshot,
            reconciliations=(
                EffectReconciliation(
                    effect_id="effect:pay",
                    disposition="absent",
                    observation_ref="observation:absent",
                    evidence_ref="evidence:pay",
                ),
            ),
        )
    plan = _compile(
        snapshot,
        reconciliations=(
            EffectReconciliation(
                effect_id="effect:pay",
                disposition="observed",
                observation_ref="observation:paid",
                evidence_ref="evidence:pay",
            ),
        ),
    )
    assert plan.replay_effect_ids == ()
    assert "attempt:pay" in plan.preserved_attempt_ids


def test_process_exit_cannot_complete_federation_work() -> None:
    with pytest.raises(RecoveryAuthorityError, match="process exit cannot complete"):
        _compile(_snapshot(claimed_complete=True))


def test_stopped_identity_cannot_be_revived() -> None:
    with pytest.raises(RecoveryAuthorityError, match="illegal supervisor lifecycle"):
        _compile(_snapshot(lifecycle_state="STOPPED", process_exited=True))


def test_same_process_birth_cannot_recover() -> None:
    with pytest.raises(RecoveryAuthorityError, match="fresh process birth"):
        _compile(_snapshot(), recovered_process_birth_id="birth:crashed")


def test_ducklake_cannot_admit_recovery() -> None:
    with pytest.raises(RecoveryAuthorityError, match="DuckLake cannot admit"):
        _compile(_snapshot(), ducklake_receipt={"recovers": True})
    with pytest.raises(RecoveryAuthorityError, match="DuckLake cannot admit"):
        _compile(_snapshot(), ducklake_receipt={"authoritative": True})


def test_store_rejects_database_path(tmp_path: Path) -> None:
    with pytest.raises(RecoveryError, match="database path"):
        RecoveryStore(tmp_path / "control.duckdb")  # type: ignore[arg-type]


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required for recovery persistence")
def test_store_records_recovery_action_epoch_and_observation(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    report = install_control_plane_schema(database, owner_id="owner:recovery")
    assert report.to_version == 3
    client = open_embedded_client(
        database,
        owner_id="owner:recovery",
        seed_generation=True,
    )
    generation = client.load_generation()
    store = RecoveryStore(client)
    binding = sample_binding(
        control_plane_generation=generation.generation,
        supervisor_population=0,
        causal_graph_revision=1,
    )
    identity, _created = _create(
        store,
        request=sample_request(binding=binding, maximum_supervisors=2, maximum_subagents=2),
        policy=sample_policy(
            binding,
            maximum_supervisors=2,
            maximum_subagents=2,
            maximum_concurrent_subagents=2,
        ),
    )
    effect = InFlightEffect(
        effect_id="effect:open",
        effect_class=EventEffectClass.READ_ONLY.value,
        attempt_id="attempt:open",
        task_id="task:open",
    )
    snapshot = CrashSnapshot(
        subject_kind="supervisor",
        subject_id="supervisor:crashed",
        process_birth_id="birth:crashed",
        lease_id="lease:crashed",
        fencing_epoch=1,
        event_cursor=4,
        lifecycle_state="FAILED",
        in_flight_effects=(effect,),
        preserved_attempt_ids=("attempt:failed",),
        process_exited=True,
    )
    reconciliation = EffectReconciliation(
        effect_id="effect:open",
        disposition="absent",
        observation_ref="observation:absent",
        evidence_ref="evidence:absent",
    )
    coordinator = FederationRecoveryCoordinator()
    plan = coordinator.compile(
        snapshot,
        binding=binding,
        recovered_subject_id="supervisor:recovered",
        recovered_process_birth_id="birth:recovered",
        recovered_lease_id="lease:recovered",
        expected_fence=1,
        reconciliations=(reconciliation,),
    )
    receipt = coordinator.recover(
        plan,
        binding=binding,
        expected_fence=1,
    )
    revision = store.graph_revision(tenant_id=binding.tenant_id, federation_id=identity.record_id)
    store.record_recovery(
        plan,
        receipt,
        federation_id=identity.record_id,
        binding=binding,
        expected_graph_revision=revision,
        idempotency_key="idempotency:recovery",
        reconciliations=(reconciliation,),
    )
    loaded_action = store.load_action(action_id="recovery-action:" + receipt.cid)
    epoch_id = "fencing-epoch:" + content_identity(
        {
            "subject_id": plan.recovered_subject_id,
            "epoch": plan.fencing_epoch,
            "lease_id": plan.recovered_lease_id,
        }
    )
    loaded_epoch = store.load_epoch(
        epoch_id=epoch_id,
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    observation_id = "effect-observation:" + content_identity(
        {
            "effect_id": reconciliation.effect_id,
            "disposition": reconciliation.disposition,
            "observation_ref": reconciliation.observation_ref,
        }
    )
    loaded_observation = store.load_observation(
        observation_id=observation_id,
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    assert loaded_action["action_kind"] == "reconnect"
    assert loaded_action["status"] == "recovered"
    assert loaded_epoch["epoch"] == 2
    assert loaded_epoch["status"] == "active"
    assert loaded_observation["disposition"] == "absent"
