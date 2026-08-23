from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.agent_registry import (
    BoundedSubagentPool,
    effective_concurrency,
)
from ipfs_accelerate_py.agent_supervisor.federation.contracts import (
    FederationAuthorityError,
    FederationBoundsError,
    SubagentInstance,
    SubagentOutcome,
)
from test.api.causal_federation.test_contracts import NOW, sample_binding


class InMemorySubagentStore:
    def __init__(self, *, maximum_active: int = 64) -> None:
        self.instances: dict[str, SubagentInstance] = {}
        self.outcomes: list[SubagentOutcome] = []
        self.registration_calls = 0
        self.reserve_calls = 0
        self.release_calls = 0
        self.maximum_active = maximum_active
        self.active: set[str] = set()

    def register_subagent(self, instance: SubagentInstance) -> SubagentInstance:
        self.registration_calls += 1
        self.instances.setdefault(instance.record_id, instance)
        return self.instances[instance.record_id]

    def get_subagent(
        self,
        subagent_id: str,
        *,
        tenant_id: str,
        federation_id: str,
        supervisor_id: str,
    ) -> SubagentInstance | None:
        instance = self.instances.get(subagent_id)
        if instance is None:
            return None
        if (
            instance.binding.tenant_id != tenant_id
            or instance.federation_id != federation_id
            or instance.supervisor_id != supervisor_id
        ):
            return None
        return instance

    def record_subagent_outcome(self, outcome: SubagentOutcome) -> None:
        if outcome.subagent_id not in self.active:
            raise FederationAuthorityError("outcome lacks an active authoritative slot")
        self.outcomes.append(outcome)

    def reserve_subagent_slot(
        self,
        *,
        subagent_id: str,
        tenant_id: str,
        federation_id: str,
        supervisor_id: str,
        expected_revision: int,
        expected_fencing_epoch: int,
        idempotency_key: str,
    ) -> dict[str, object]:
        del idempotency_key
        instance = self.get_subagent(
            subagent_id,
            tenant_id=tenant_id,
            federation_id=federation_id,
            supervisor_id=supervisor_id,
        )
        if instance is None:
            raise FederationAuthorityError("subagent is absent")
        if (
            instance.revision != expected_revision
            or instance.fencing_epoch != expected_fencing_epoch
            or instance.state != "ADMITTED"
        ):
            raise FederationAuthorityError("subagent reservation authority is stale")
        if subagent_id in self.active or len(self.active) >= self.maximum_active:
            raise FederationAuthorityError("authoritative concurrent slot ceiling reached")
        self.reserve_calls += 1
        self.active.add(subagent_id)
        active = replace(instance, revision=instance.revision + 1, state="ACTIVE")
        self.instances[subagent_id] = active
        return {"subagent": active.to_dict(), "slot_number": len(self.active)}

    def release_subagent_slot(
        self,
        *,
        subagent_id: str,
        tenant_id: str,
        federation_id: str,
        supervisor_id: str,
        expected_revision: int,
        expected_fencing_epoch: int,
        idempotency_key: str,
    ) -> dict[str, object]:
        del idempotency_key
        instance = self.get_subagent(
            subagent_id,
            tenant_id=tenant_id,
            federation_id=federation_id,
            supervisor_id=supervisor_id,
        )
        if instance is None or subagent_id not in self.active:
            raise FederationAuthorityError("subagent has no active slot")
        if (
            instance.revision != expected_revision
            or instance.fencing_epoch != expected_fencing_epoch
            or instance.state != "ACTIVE"
        ):
            raise FederationAuthorityError("subagent release authority is stale")
        self.release_calls += 1
        self.active.remove(subagent_id)
        admitted = replace(instance, revision=instance.revision + 1, state="ADMITTED")
        self.instances[subagent_id] = admitted
        return {"subagent": admitted.to_dict(), "slot_number": 1}


def sample_subagent(index: int) -> SubagentInstance:
    return SubagentInstance(
        record_id=f"subagent:{index}",
        revision=1,
        binding=sample_binding(),
        state="ADMITTED",
        federation_id="federation:test",
        supervisor_id="supervisor:test",
        task_id="",
        lease_id="",
        fencing_epoch=1,
    )


def sample_outcome(
    subagent_id: str,
    *,
    outcome: str = "succeeded",
) -> SubagentOutcome:
    return SubagentOutcome(
        record_id=f"outcome:{subagent_id}",
        revision=1,
        binding=sample_binding(),
        outcome=outcome,
        evidence_refs=("evidence:test",),
        recorded_at=NOW,
        federation_id="federation:test",
        supervisor_id="supervisor:test",
        subagent_id=subagent_id,
        task_id="",
        fencing_epoch=1,
    )


def test_registration_is_logical_and_respects_the_bounded_population() -> None:
    store = InMemorySubagentStore()
    pool = BoundedSubagentPool(
        store,
        maximum_registered=4,
        maximum_active=2,
    )
    worker_calls = 0

    for index in range(4):
        pool.register(sample_subagent(index))

    capacity = pool.capacity()
    assert capacity.registered == 4
    assert capacity.registered_ceiling == 4
    assert capacity.active == 0
    assert capacity.active_ceiling == 2
    assert worker_calls == 0
    assert store.registration_calls == 4

    with pytest.raises(FederationBoundsError):
        pool.register(sample_subagent(4))


def test_duplicate_logical_registration_does_not_consume_another_identity_slot() -> None:
    store = InMemorySubagentStore()
    pool = BoundedSubagentPool(
        store,
        maximum_registered=2,
        maximum_active=1,
    )
    instance = sample_subagent(1)

    assert pool.register(instance) == instance
    assert pool.register(instance) == instance
    assert pool.capacity().registered == 1


def test_small_hermetic_pool_separates_registered_and_active_populations() -> None:
    async def scenario() -> None:
        store = InMemorySubagentStore()
        pool = BoundedSubagentPool(
            store,
            maximum_registered=4,
            maximum_active=2,
        )
        for index in range(4):
            pool.register(sample_subagent(index))

        two_started = asyncio.Event()
        release = asyncio.Event()
        started: list[str] = []

        async def worker(subagent_id: str) -> str:
            started.append(subagent_id)
            if len(started) == 2:
                two_started.set()
            await release.wait()
            return subagent_id

        tasks = [
            asyncio.create_task(
                pool.execute(
                    subagent_id=f"subagent:{index}",
                    worker=lambda index=index: worker(f"subagent:{index}"),
                    outcome_factory=lambda result, error, index=index: sample_outcome(
                        f"subagent:{index}",
                        outcome="succeeded" if error is None and result else "failed",
                    ),
                )
            )
            for index in range(4)
        ]
        await asyncio.wait_for(two_started.wait(), timeout=1)

        capacity = pool.capacity()
        assert capacity.registered == 4
        assert capacity.active == 2
        assert capacity.available_slots == 0

        release.set()
        outcomes = await asyncio.gather(*tasks)
        assert len(outcomes) == 4
        assert len(store.outcomes) == 4
        assert store.reserve_calls == 4
        assert store.release_calls == 4
        assert not store.active
        assert pool.maximum_observed_active == 2
        assert pool.capacity().active == 0

    asyncio.run(scenario())


def test_per_agent_cancellation_prevents_slot_admission() -> None:
    async def scenario() -> None:
        store = InMemorySubagentStore()
        pool = BoundedSubagentPool(
            store,
            maximum_registered=2,
            maximum_active=2,
        )
        pool.register(sample_subagent(1))
        pool.cancel("subagent:1")

        with pytest.raises(asyncio.CancelledError):
            await pool.execute(
                subagent_id="subagent:1",
                worker=lambda: "not-called",
                outcome_factory=lambda result, error: sample_outcome("subagent:1"),
            )

        assert pool.capacity().active == 0
        assert not store.outcomes

    asyncio.run(scenario())


def test_active_cancellation_reaches_worker_and_records_typed_outcome() -> None:
    async def scenario() -> None:
        store = InMemorySubagentStore()
        pool = BoundedSubagentPool(
            store,
            maximum_registered=1,
            maximum_active=1,
        )
        pool.register(sample_subagent(1))
        started = asyncio.Event()
        worker_cancelled = asyncio.Event()

        async def worker() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                worker_cancelled.set()
                raise

        execution = asyncio.create_task(
            pool.execute(
                subagent_id="subagent:1",
                worker=worker,
                outcome_factory=lambda result, error: sample_outcome(
                    "subagent:1",
                    outcome="cancelled" if isinstance(error, asyncio.CancelledError) else "failed",
                ),
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)

        pool.cancel("subagent:1")
        with pytest.raises(asyncio.CancelledError):
            await execution

        assert worker_cancelled.is_set()
        assert [outcome.outcome for outcome in store.outcomes] == ["cancelled"]
        assert store.outcomes[0].subagent_id == "subagent:1"
        assert pool.capacity().active == 0

    asyncio.run(scenario())


def test_one_logical_agent_cannot_hold_two_active_slots() -> None:
    async def scenario() -> None:
        store = InMemorySubagentStore()
        pool = BoundedSubagentPool(
            store,
            maximum_registered=2,
            maximum_active=2,
        )
        pool.register(sample_subagent(1))
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocking_worker() -> str:
            started.set()
            await release.wait()
            return "done"

        first = asyncio.create_task(
            pool.execute(
                subagent_id="subagent:1",
                worker=blocking_worker,
                outcome_factory=lambda result, error: sample_outcome("subagent:1"),
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        second = asyncio.create_task(
            pool.execute(
                subagent_id="subagent:1",
                worker=lambda: "duplicate",
                outcome_factory=lambda result, error: sample_outcome("subagent:1"),
            )
        )
        with pytest.raises(FederationAuthorityError):
            await second

        release.set()
        await first
        assert pool.maximum_observed_active == 1

    asyncio.run(scenario())


def test_independent_pools_cannot_exceed_the_authoritative_slot_ceiling() -> None:
    async def scenario() -> None:
        store = InMemorySubagentStore(maximum_active=1)
        first_pool = BoundedSubagentPool(
            store,
            maximum_registered=1,
            maximum_active=1,
        )
        second_pool = BoundedSubagentPool(
            store,
            maximum_registered=1,
            maximum_active=1,
        )
        first_pool.register(sample_subagent(1))
        second_pool.register(sample_subagent(2))
        started = asyncio.Event()
        release = asyncio.Event()
        second_worker_called = False

        async def first_worker() -> str:
            started.set()
            await release.wait()
            return "first"

        def second_worker() -> str:
            nonlocal second_worker_called
            second_worker_called = True
            return "second"

        first = asyncio.create_task(
            first_pool.execute(
                subagent_id="subagent:1",
                worker=first_worker,
                outcome_factory=lambda result, error: sample_outcome("subagent:1"),
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)

        with pytest.raises(FederationAuthorityError, match="slot ceiling"):
            await second_pool.execute(
                subagent_id="subagent:2",
                worker=second_worker,
                outcome_factory=lambda result, error: sample_outcome("subagent:2"),
            )
        assert second_worker_called is False
        assert store.active == {"subagent:1"}

        release.set()
        await first
        assert not store.active
        assert store.reserve_calls == 1
        assert store.release_calls == 1

    asyncio.run(scenario())


def test_worker_failure_records_bounded_outcome_then_reraises() -> None:
    class WorkerFailure(RuntimeError):
        pass

    async def scenario() -> None:
        store = InMemorySubagentStore()
        pool = BoundedSubagentPool(
            store,
            maximum_registered=1,
            maximum_active=1,
        )
        pool.register(sample_subagent(1))

        def fail() -> None:
            raise WorkerFailure("bounded failure")

        with pytest.raises(WorkerFailure):
            await pool.execute(
                subagent_id="subagent:1",
                worker=fail,
                outcome_factory=lambda result, error: sample_outcome(
                    "subagent:1",
                    outcome="failed" if error is not None else "succeeded",
                ),
            )

        assert [outcome.outcome for outcome in store.outcomes] == ["failed"]
        assert pool.capacity().active == 0

    asyncio.run(scenario())


def test_outcome_factory_cannot_return_untyped_model_output() -> None:
    async def scenario() -> None:
        store = InMemorySubagentStore()
        pool = BoundedSubagentPool(
            store,
            maximum_registered=1,
            maximum_active=1,
        )
        pool.register(sample_subagent(1))

        with pytest.raises(FederationAuthorityError):
            await pool.execute(
                subagent_id="subagent:1",
                worker=lambda: "result",
                outcome_factory=lambda result, error: {"model": "complete"},  # type: ignore[arg-type]
            )

        assert not store.outcomes
        assert store.reserve_calls == 1
        assert store.release_calls == 1
        assert not store.active

    asyncio.run(scenario())


def test_typed_outcome_cannot_cross_subagent_or_fence_scope() -> None:
    async def scenario() -> None:
        store = InMemorySubagentStore()
        pool = BoundedSubagentPool(
            store,
            maximum_registered=1,
            maximum_active=1,
        )
        pool.register(sample_subagent(1))

        with pytest.raises(FederationAuthorityError):
            await pool.execute(
                subagent_id="subagent:1",
                worker=lambda: "result",
                outcome_factory=lambda result, error: sample_outcome("subagent:other"),
            )

        assert not store.outcomes

    asyncio.run(scenario())


def test_frozen_outcome_tampering_cannot_self_authorize_completion() -> None:
    async def scenario() -> None:
        store = InMemorySubagentStore()
        pool = BoundedSubagentPool(store, maximum_registered=1, maximum_active=1)
        pool.register(sample_subagent(1))
        forged = sample_outcome("subagent:1")
        object.__setattr__(forged, "outcome", "completed")

        with pytest.raises(FederationAuthorityError):
            await pool.execute(
                subagent_id="subagent:1",
                worker=lambda: "candidate-result",
                outcome_factory=lambda result, error: forged,
            )

        assert not store.outcomes

    asyncio.run(scenario())


def test_worker_exception_cannot_be_relabelled_as_success() -> None:
    async def scenario() -> None:
        store = InMemorySubagentStore()
        pool = BoundedSubagentPool(store, maximum_registered=1, maximum_active=1)
        pool.register(sample_subagent(1))

        def fail() -> None:
            raise RuntimeError("worker failed")

        with pytest.raises(FederationAuthorityError, match="execution result"):
            await pool.execute(
                subagent_id="subagent:1",
                worker=fail,
                outcome_factory=lambda result, error: sample_outcome("subagent:1"),
            )

        assert not store.outcomes

    asyncio.run(scenario())


def test_outcome_must_bind_the_live_authority_snapshot() -> None:
    async def scenario() -> None:
        store = InMemorySubagentStore()
        pool = BoundedSubagentPool(store, maximum_registered=1, maximum_active=1)
        pool.register(sample_subagent(1))
        wrong_binding = sample_binding(control_plane_generation=2)

        with pytest.raises(FederationAuthorityError, match="live authority"):
            await pool.execute(
                subagent_id="subagent:1",
                worker=lambda: "candidate-result",
                outcome_factory=lambda result, error: SubagentOutcome(
                    record_id="outcome:wrong-binding",
                    revision=1,
                    binding=wrong_binding,
                    outcome="succeeded",
                    evidence_refs=("evidence:test",),
                    recorded_at=NOW,
                    federation_id="federation:test",
                    supervisor_id="supervisor:test",
                    subagent_id="subagent:1",
                    task_id="",
                    fencing_epoch=1,
                ),
            )

        assert not store.outcomes

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"maximum_registered": 0, "maximum_active": 1},
        {"maximum_registered": 4, "maximum_active": 0},
        {"maximum_registered": 4, "maximum_active": 5},
    ],
)
def test_pool_configuration_bounds_fail_closed(kwargs: dict[str, int]) -> None:
    with pytest.raises(FederationBoundsError):
        BoundedSubagentPool(InMemorySubagentStore(), **kwargs)


def test_effective_capacity_is_the_minimum_of_all_fresh_telemetry() -> None:
    assert (
        effective_concurrency(
            policy_ceiling=8,
            host_capacity=7,
            provider_capacity=6,
            proof_capacity=5,
            merge_capacity=4,
            storage_capacity=3,
        )
        == 3
    )


@pytest.mark.parametrize(
    "missing",
    [
        "host_capacity",
        "provider_capacity",
        "proof_capacity",
        "merge_capacity",
        "storage_capacity",
    ],
)
def test_missing_telemetry_grants_no_new_capacity(missing: str) -> None:
    values: dict[str, Any] = {
        "policy_ceiling": 8,
        "host_capacity": 8,
        "provider_capacity": 8,
        "proof_capacity": 8,
        "merge_capacity": 8,
        "storage_capacity": 8,
    }
    values[missing] = None

    assert effective_concurrency(**values) == 0


def test_negative_capacity_is_rejected() -> None:
    with pytest.raises(FederationBoundsError):
        effective_concurrency(
            policy_ceiling=8,
            host_capacity=-1,
            provider_capacity=8,
            proof_capacity=8,
            merge_capacity=8,
            storage_capacity=8,
        )
