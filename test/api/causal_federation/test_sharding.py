"""Hermetic tests for CASF supervisor sharding and specialization."""

from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.events import EventEffectClass
from ipfs_accelerate_py.agent_supervisor.federation.sharding import (
    ShardingAuthorityError,
    ShardingError,
    ShardingStore,
    ShardWork,
    bind_supervisor_specialization,
    compile_supervisor_shards,
    refuse_ducklake_shard_authority,
)
from ipfs_accelerate_py.agent_supervisor.federation.supervisor_registry import (
    compile_registered_shards,
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


def _work(
    task_id: str,
    *,
    effect_class: str = EventEffectClass.READ_ONLY.value,
    goal_id: str = "goal:one",
    symbol: str = "symbol:a",
) -> ShardWork:
    binding = sample_binding()
    return ShardWork(
        task_id=task_id,
        repository_id=binding.repository_ids[0],
        tree_id=binding.repository_tree_ids[0],
        goal_id=goal_id,
        effect_class=effect_class,
        symbol_refs=(symbol,),
    )


def _spec(
    supervisor_id: str,
    *,
    effects: tuple[str, ...] = (EventEffectClass.READ_ONLY.value,),
    goals: tuple[str, ...] = (),
):
    return bind_supervisor_specialization(
        binding=sample_binding(),
        supervisor_id=supervisor_id,
        allowed_goal_refs=goals,
        allowed_effect_classes=effects,
        capability_refs=("capability:test",),
    )


def test_exclusive_writes_have_exactly_one_shard_owner() -> None:
    write = _work("task:write", effect_class=EventEffectClass.AUTHORITATIVE_STATE.value)
    read = _work("task:read", effect_class=EventEffectClass.READ_ONLY.value, goal_id="goal:two")
    writer = _spec(
        "supervisor:writer",
        effects=(EventEffectClass.AUTHORITATIVE_STATE.value,),
    )
    reader = _spec("supervisor:reader")
    plan = compile_supervisor_shards(
        (write, read),
        (writer, reader),
        binding=sample_binding(),
    )
    owners = {task: shard.supervisor_id for shard in plan.shards for task in shard.task_ids}
    assert owners["task:write"] == "supervisor:writer"
    assert owners["task:read"] == "supervisor:reader"
    assert len(owners) == 2


def test_overlapping_exclusive_specializations_fail_closed() -> None:
    write = _work("task:write", effect_class=EventEffectClass.AUTHORITATIVE_STATE.value)
    left = _spec(
        "supervisor:left",
        effects=(EventEffectClass.AUTHORITATIVE_STATE.value,),
    )
    right = _spec(
        "supervisor:right",
        effects=(EventEffectClass.AUTHORITATIVE_STATE.value,),
    )
    with pytest.raises(ShardingAuthorityError, match="two specializations"):
        compile_supervisor_shards((write,), (left, right), binding=sample_binding())


def test_missing_capability_fails_closed() -> None:
    write = _work("task:write", effect_class=EventEffectClass.PAYMENT.value)
    reader = _spec("supervisor:reader")
    with pytest.raises(ShardingAuthorityError, match="no specialized supervisor"):
        compile_supervisor_shards((write,), (reader,), binding=sample_binding())


def test_shared_reads_assign_deterministically_without_double_ownership() -> None:
    read = _work("task:read")
    alpha = _spec("supervisor:alpha")
    beta = _spec("supervisor:beta")
    plan = compile_supervisor_shards((read,), (beta, alpha), binding=sample_binding())
    assert len(plan.shards) == 1
    assert plan.shards[0].supervisor_id == "supervisor:alpha"
    assert plan.shards[0].task_ids == ("task:read",)


def test_goal_ceiling_rejects_out_of_specialization_work() -> None:
    work = _work("task:read", goal_id="goal:other")
    spec = _spec("supervisor:reader", goals=("goal:one",))
    with pytest.raises(ShardingAuthorityError, match="no specialized supervisor"):
        compile_supervisor_shards((work,), (spec,), binding=sample_binding())


def test_unbound_repository_specialization_fails_closed() -> None:
    with pytest.raises(ShardingAuthorityError, match="repository is not bound"):
        bind_supervisor_specialization(
            binding=sample_binding(),
            supervisor_id="supervisor:x",
            allowed_repository_ids=("repo:other",),
            allowed_effect_classes=(EventEffectClass.READ_ONLY.value,),
            capability_refs=("capability:test",),
        )


def test_registry_helper_does_not_open_a_second_authority() -> None:
    plan = compile_registered_shards(
        (_work("task:read"),),
        (_spec("supervisor:reader"),),
        binding=sample_binding(),
    )
    assert plan.shards[0].supervisor_id == "supervisor:reader"


def test_ducklake_cannot_admit_shards() -> None:
    with pytest.raises(ShardingAuthorityError, match="DuckLake cannot admit"):
        refuse_ducklake_shard_authority({"authoritative": True})
    with pytest.raises(ShardingAuthorityError, match="DuckLake cannot admit"):
        compile_supervisor_shards(
            (_work("task:read"),),
            (_spec("supervisor:reader"),),
            binding=sample_binding(),
            ducklake_receipt={"schedules": True},
        )


def test_store_rejects_database_path(tmp_path: Path) -> None:
    with pytest.raises(ShardingError, match="database path"):
        ShardingStore(tmp_path / "control.duckdb")  # type: ignore[arg-type]


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required for shard persistence")
def test_store_records_exact_shard_boundaries_and_assignment(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    report = install_control_plane_schema(database, owner_id="owner:sharding")
    assert report.to_version == 3
    client = open_embedded_client(
        database,
        owner_id="owner:sharding",
        seed_generation=True,
    )
    generation = client.load_generation()
    store = ShardingStore(client)
    binding = sample_binding(
        control_plane_generation=generation.generation,
        supervisor_population=0,
        causal_graph_revision=1,
    )
    identity, _receipt = _create(
        store,
        request=sample_request(binding=binding, maximum_supervisors=2, maximum_subagents=2),
        policy=sample_policy(
            binding,
            maximum_supervisors=2,
            maximum_subagents=2,
            maximum_concurrent_subagents=2,
        ),
    )
    work = ShardWork(
        task_id="task:read",
        repository_id=binding.repository_ids[0],
        tree_id=binding.repository_tree_ids[0],
        goal_id="goal:one",
        effect_class=EventEffectClass.READ_ONLY.value,
        symbol_refs=("symbol:a",),
    )
    spec = bind_supervisor_specialization(
        binding=binding,
        supervisor_id="supervisor:reader",
        allowed_effect_classes=(EventEffectClass.READ_ONLY.value,),
        capability_refs=("capability:test",),
    )
    plan = compile_supervisor_shards((work,), (spec,), binding=binding)
    revision = store.graph_revision(tenant_id=binding.tenant_id, federation_id=identity.record_id)
    store.record_plan(
        plan,
        federation_id=identity.record_id,
        binding=binding,
        expected_graph_revision=revision,
        idempotency_key="idempotency:shard",
    )
    shard = plan.shards[0]
    loaded = store.load_shard(
        shard_id=shard.shard_id,
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    assert loaded["state"] == "active"
    assert loaded["fencing_epoch"] == 1
    assignment = store.load_assignment(
        assignment_id="shard-assignment:"
        + content_identity(
            {
                "shard_id": shard.shard_id,
                "supervisor_id": shard.supervisor_id,
                "revision": shard.revision,
            }
        ),
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    assert assignment["supervisor_id"] == "supervisor:reader"
    assert assignment["state"] == "active"
