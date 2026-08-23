"""Hermetic tests for CASF shard rebalancing and fencing."""

from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.budgets import (
    BudgetDimensionName,
    HierarchicalBudgetLedger,
)
from ipfs_accelerate_py.agent_supervisor.federation.events import EventEffectClass
from ipfs_accelerate_py.agent_supervisor.federation.rebalancing import (
    RebalanceWork,
    RebalancingAuthorityError,
    RebalancingError,
    RebalancingStore,
    ShardRebalancePlanner,
    ShardRebalanceRequest,
)
from ipfs_accelerate_py.agent_supervisor.federation.sharding import (
    ShardWork,
    bind_supervisor_specialization,
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
    *,
    effect_class: str = EventEffectClass.READ_ONLY.value,
    task_id: str = "task:idle",
) -> ShardWork:
    binding = sample_binding()
    return ShardWork(
        task_id=task_id,
        repository_id=binding.repository_ids[0],
        tree_id=binding.repository_tree_ids[0],
        goal_id="goal:one",
        effect_class=effect_class,
        symbol_refs=("symbol:a",),
    )


def _spec(
    supervisor_id: str,
    *,
    effects: tuple[str, ...] = (EventEffectClass.READ_ONLY.value,),
    capabilities: tuple[str, ...] = ("capability:test",),
    max_shards: int = 1,
):
    return bind_supervisor_specialization(
        binding=sample_binding(),
        supervisor_id=supervisor_id,
        allowed_effect_classes=effects,
        capability_refs=capabilities,
        max_shards=max_shards,
    )


def _unit(**overrides: object) -> RebalanceWork:
    values: dict[str, object] = {"work": _work()}
    values.update(overrides)
    return RebalanceWork(**values)  # type: ignore[arg-type]


def _request(**overrides: object) -> ShardRebalanceRequest:
    values: dict[str, object] = {
        "shard_id": "shard:alpha",
        "source_supervisor_id": "supervisor:busy",
        "fencing_epoch": 3,
        "assignment_revision": 2,
        "semantic_root": "semantic:test",
        "units": (_unit(),),
    }
    values.update(overrides)
    return ShardRebalanceRequest(**values)  # type: ignore[arg-type]


def _compile(request: ShardRebalanceRequest, target, **kwargs: object):
    planner = ShardRebalancePlanner()
    binding = sample_binding()
    values: dict[str, object] = {
        "target": target,
        "binding": binding,
        "expected_source_fence": request.fencing_epoch,
        "expected_assignment_revision": request.assignment_revision,
        "current_tree_id": request.units[0].work.tree_id,
        "current_semantic_root": request.semantic_root,
    }
    values.update(kwargs)
    return planner.compile(request, **values)  # type: ignore[arg-type]


def _execute(plan, **kwargs: object):
    planner = ShardRebalancePlanner()
    binding = sample_binding()
    values: dict[str, object] = {
        "binding": binding,
        "current_tree_id": plan.tree_id,
        "current_semantic_root": plan.semantic_root,
        "expected_source_fence": plan.previous_fencing_epoch,
        "expected_assignment_revision": plan.source_revision,
    }
    values.update(kwargs)
    return planner.execute(plan, **values)  # type: ignore[arg-type]


def test_freeze_drain_transfer_fence_and_activate() -> None:
    request = _request(
        units=(
            _unit(
                work=_work(task_id="task:claimed"),
                claimed=True,
                in_flight=True,
                attempt_count=2,
                checkpoint_ref="checkpoint:claimed",
                cursor_ref="cursor:claimed",
            ),
            _unit(work=_work(task_id="task:idle")),
        )
    )
    plan = _compile(request, _spec("supervisor:idle"))
    assert plan.frozen is True
    assert plan.claims_stopped is True
    assert plan.previous_fencing_epoch == 3
    assert plan.fencing_epoch == 4
    assert plan.source_revision == 2
    assert plan.target_revision == 3
    assert plan.transferred_task_ids == ("task:claimed", "task:idle")
    receipt = _execute(plan)
    assert receipt.outcome == "rebalanced"
    assert receipt.activated is True
    assert receipt.owner_supervisor_id == "supervisor:idle"
    assert receipt.fencing_epoch == 4
    assert receipt.transferred_task_ids == ("task:claimed", "task:idle")


def test_identities_checkpoints_and_cursors_are_preserved() -> None:
    request = _request(
        units=(
            _unit(
                work=_work(task_id="task:resume"),
                claimed=True,
                attempt_count=4,
                checkpoint_ref="checkpoint:resume",
                cursor_ref="cursor:resume",
            ),
        )
    )
    plan = _compile(request, _spec("supervisor:idle"))
    receipt = _execute(plan)
    assert receipt.preserved_checkpoint_refs == ("checkpoint:resume",)
    assert receipt.preserved_cursor_refs == ("cursor:resume",)
    assert receipt.preserved_attempt_counts == (4,)
    assert receipt.transferred_task_ids == ("task:resume",)


def test_no_double_shard_ownership_after_activate() -> None:
    request = _request(
        units=(
            _unit(work=_work(task_id="task:a")),
            _unit(work=_work(task_id="task:b")),
        )
    )
    receipt = _execute(_compile(request, _spec("supervisor:idle")))
    assert len(receipt.transferred_task_ids) == len(set(receipt.transferred_task_ids))
    assert receipt.owner_supervisor_id != request.source_supervisor_id
    assert receipt.owner_supervisor_id == "supervisor:idle"


def test_duplicate_task_identities_fail_closed() -> None:
    request = _request(
        units=(
            _unit(work=_work(task_id="task:dup")),
            _unit(work=_work(task_id="task:dup")),
        )
    )
    with pytest.raises(RebalancingAuthorityError, match="more than one owner"):
        _compile(request, _spec("supervisor:idle"))


def test_irreversible_in_flight_cannot_move() -> None:
    request = _request(
        units=(
            _unit(
                work=_work(effect_class=EventEffectClass.PAYMENT.value, task_id="task:pay"),
                in_flight=True,
            ),
        )
    )
    target = _spec("supervisor:idle", effects=(EventEffectClass.PAYMENT.value,))
    with pytest.raises(RebalancingAuthorityError, match="irreversible"):
        _compile(request, target)


def test_stale_fence_and_stale_tree_fail_closed() -> None:
    request = _request()
    target = _spec("supervisor:idle")
    with pytest.raises(RebalancingAuthorityError, match="fencing epoch is stale"):
        _compile(request, target, expected_source_fence=1)
    with pytest.raises(RebalancingAuthorityError, match="current tree identity"):
        _compile(request, target, current_tree_id="tree:other")
    with pytest.raises(RebalancingAuthorityError, match="current semantic state"):
        _compile(request, target, current_semantic_root="semantic:other")
    plan = _compile(request, target)
    with pytest.raises(RebalancingAuthorityError, match="fencing epoch is stale"):
        _execute(plan, expected_source_fence=1)


def test_policy_proof_merge_privacy_and_human_review_cannot_be_bypassed() -> None:
    target = _spec("supervisor:idle")
    for flag in (
        "requires_human_review",
        "requires_privacy_review",
        "requires_proof",
        "requires_merge",
    ):
        with pytest.raises(RebalancingAuthorityError, match="cannot bypass policy"):
            _compile(_request(units=(_unit(**{flag: True}),)), target)


def test_out_of_ceiling_target_cannot_admit_work() -> None:
    request = _request(
        units=(_unit(work=_work(effect_class=EventEffectClass.AUTHORITATIVE_STATE.value)),)
    )
    with pytest.raises(RebalancingAuthorityError, match="cannot admit"):
        _compile(request, _spec("supervisor:idle"))


def test_target_at_shard_ceiling_fails_closed() -> None:
    with pytest.raises(RebalancingAuthorityError, match="cannot admit another shard"):
        _compile(_request(), _spec("supervisor:idle"), target_existing_shard_count=1)


def test_self_rebalance_fails_closed() -> None:
    with pytest.raises(RebalancingAuthorityError, match="cannot rebalance onto itself"):
        _compile(_request(), _spec("supervisor:busy"))


def test_budget_is_conserved_across_rebalance_transfer() -> None:
    ledger = HierarchicalBudgetLedger()
    ledger.open_root(
        account_id="budget:federation",
        owner_id="federation:test",
        dimensions={BudgetDimensionName.INPUT_TOKENS: 400},
    )
    ledger.allocate_child(
        parent_account_id="budget:federation",
        child_account_id="budget:busy",
        child_owner_id="supervisor:busy",
        dimension=BudgetDimensionName.INPUT_TOKENS,
        amount=200,
        expected_parent_revision=1,
    )
    ledger.allocate_child(
        parent_account_id="budget:federation",
        child_account_id="budget:idle",
        child_owner_id="supervisor:idle",
        dimension=BudgetDimensionName.INPUT_TOKENS,
        amount=50,
        expected_parent_revision=2,
    )
    receipt = _execute(
        _compile(_request(), _spec("supervisor:idle")),
        ledger=ledger,
        source_budget_account_id="budget:busy",
        target_budget_account_id="budget:idle",
        budget_dimension=BudgetDimensionName.INPUT_TOKENS,
        budget_amount=40,
        expected_source_budget_revision=1,
        expected_target_budget_revision=1,
    )
    assert receipt.budget_transferred == 40
    assert ledger.account("budget:busy").slice_for(BudgetDimensionName.INPUT_TOKENS).ceiling == 160
    assert ledger.account("budget:idle").slice_for(BudgetDimensionName.INPUT_TOKENS).ceiling == 90
    assert ledger.conserved("budget:federation", BudgetDimensionName.INPUT_TOKENS) is True


def test_rollback_restores_source_and_increments_fence() -> None:
    plan = _compile(_request(), _spec("supervisor:idle"))
    planner = ShardRebalancePlanner()
    receipt = planner.rollback(
        plan,
        binding=sample_binding(),
        current_tree_id=plan.tree_id,
        current_semantic_root=plan.semantic_root,
        expected_source_fence=plan.previous_fencing_epoch,
        expected_assignment_revision=plan.source_revision,
    )
    assert receipt.outcome == "rolled_back"
    assert receipt.activated is False
    assert receipt.owner_supervisor_id == "supervisor:busy"
    assert receipt.fencing_epoch == 4
    assert receipt.transferred_task_ids == ()
    assert receipt.preserved_attempt_counts == (0,)


def test_ducklake_cannot_admit_a_rebalance() -> None:
    with pytest.raises(RebalancingAuthorityError, match="DuckLake cannot admit"):
        _compile(_request(), _spec("supervisor:idle"), ducklake_receipt={"rebalances": True})
    with pytest.raises(RebalancingAuthorityError, match="DuckLake cannot admit"):
        _compile(_request(), _spec("supervisor:idle"), ducklake_receipt={"schedules": True})


def test_store_rejects_database_path(tmp_path: Path) -> None:
    with pytest.raises(RebalancingError, match="database path"):
        RebalancingStore(tmp_path / "control.duckdb")  # type: ignore[arg-type]


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required for rebalance persistence")
def test_store_records_plan_revision_and_receipt(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    report = install_control_plane_schema(database, owner_id="owner:rebalance")
    assert report.to_version == 2
    client = open_embedded_client(
        database,
        owner_id="owner:rebalance",
        seed_generation=True,
    )
    generation = client.load_generation()
    store = RebalancingStore(client)
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
    work = ShardWork(
        task_id="task:idle",
        repository_id=binding.repository_ids[0],
        tree_id=binding.repository_tree_ids[0],
        goal_id="goal:one",
        effect_class=EventEffectClass.READ_ONLY.value,
    )
    request = ShardRebalanceRequest(
        shard_id="shard:alpha",
        source_supervisor_id="supervisor:busy",
        fencing_epoch=1,
        assignment_revision=1,
        semantic_root=binding.semantic_state_roots[0],
        units=(RebalanceWork(work=work),),
    )
    target = bind_supervisor_specialization(
        binding=binding,
        supervisor_id="supervisor:idle",
        allowed_effect_classes=(EventEffectClass.READ_ONLY.value,),
        capability_refs=("capability:test",),
    )
    planner = ShardRebalancePlanner()
    plan = planner.compile(
        request,
        target=target,
        binding=binding,
        expected_source_fence=1,
        expected_assignment_revision=1,
        current_tree_id=work.tree_id,
        current_semantic_root=request.semantic_root,
    )
    receipt = planner.execute(
        plan,
        binding=binding,
        current_tree_id=work.tree_id,
        current_semantic_root=request.semantic_root,
        expected_source_fence=1,
        expected_assignment_revision=1,
    )
    revision = store.graph_revision(tenant_id=binding.tenant_id, federation_id=identity.record_id)
    store.record_rebalance(
        plan,
        receipt,
        federation_id=identity.record_id,
        binding=binding,
        expected_graph_revision=revision,
        idempotency_key="idempotency:rebalance",
    )
    loaded_plan = store.load_plan(
        plan_id=plan.plan_id,
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    loaded_receipt = store.load_receipt(
        receipt_id="rebalance-receipt:" + receipt.cid,
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    loaded_revision = store.load_revision(
        shard_id=plan.shard_id,
        revision=plan.target_revision,
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    assert loaded_plan["state"] == "frozen"
    assert loaded_plan["content_ref"] == plan.cid
    assert loaded_receipt["disposition"] == "rebalanced"
    assert loaded_receipt["content_ref"] == receipt.cid
    assert loaded_revision["fencing_epoch"] == receipt.fencing_epoch
    assert loaded_revision["state"] == "rebalanced"
