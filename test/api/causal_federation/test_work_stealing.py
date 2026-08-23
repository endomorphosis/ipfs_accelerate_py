"""Hermetic tests for CASF virgin-only work stealing."""

from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.budgets import (
    BudgetDimensionName,
    HierarchicalBudgetLedger,
)
from ipfs_accelerate_py.agent_supervisor.federation.events import EventEffectClass
from ipfs_accelerate_py.agent_supervisor.federation.sharding import (
    ShardWork,
    bind_supervisor_specialization,
)
from ipfs_accelerate_py.agent_supervisor.federation.work_stealing import (
    StealCandidate,
    WorkStealingAuthorityError,
    WorkStealingError,
    WorkStealingStore,
    steal_work,
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
):
    return bind_supervisor_specialization(
        binding=sample_binding(),
        supervisor_id=supervisor_id,
        allowed_effect_classes=effects,
        capability_refs=capabilities,
    )


def _candidate(**overrides: object) -> StealCandidate:
    values: dict[str, object] = {
        "work": _work(),
        "source_supervisor_id": "supervisor:busy",
        "fencing_epoch": 3,
        "assignment_revision": 2,
        "semantic_root": "semantic:test",
    }
    values.update(overrides)
    return StealCandidate(**values)  # type: ignore[arg-type]


def _steal(candidate: StealCandidate, thief, **kwargs: object):
    binding = sample_binding()
    values: dict[str, object] = {
        "thief": thief,
        "binding": binding,
        "current_tree_id": candidate.work.tree_id,
        "current_semantic_root": candidate.semantic_root,
        "expected_source_fence": candidate.fencing_epoch,
        "expected_assignment_revision": candidate.assignment_revision,
    }
    values.update(kwargs)
    return steal_work(candidate, **values)  # type: ignore[arg-type]


def test_virgin_in_ceiling_work_is_stolen_with_fence_increment() -> None:
    candidate = _candidate()
    thief = _spec("supervisor:idle")
    receipt = _steal(candidate, thief)
    assert receipt.thief_supervisor_id == "supervisor:idle"
    assert receipt.fencing_epoch == 4
    assert receipt.assignment_revision == 3
    assert receipt.budget_transferred == 0


def test_claimed_or_attempted_work_cannot_be_stolen() -> None:
    thief = _spec("supervisor:idle")
    with pytest.raises(WorkStealingAuthorityError, match="virgin work"):
        _steal(_candidate(claimed=True), thief)
    with pytest.raises(WorkStealingAuthorityError, match="virgin work"):
        _steal(_candidate(attempt_count=1), thief)


def test_active_irreversible_effect_cannot_move() -> None:
    candidate = _candidate(
        work=_work(effect_class=EventEffectClass.PAYMENT.value, task_id="task:pay"),
        in_flight=True,
    )
    thief = _spec("supervisor:idle", effects=(EventEffectClass.PAYMENT.value,))
    with pytest.raises(WorkStealingAuthorityError, match="irreversible"):
        _steal(candidate, thief)


def test_out_of_ceiling_thief_cannot_steal() -> None:
    candidate = _candidate(
        work=_work(effect_class=EventEffectClass.AUTHORITATIVE_STATE.value),
    )
    thief = _spec("supervisor:idle")
    with pytest.raises(WorkStealingAuthorityError, match="cannot admit"):
        _steal(candidate, thief)


def test_stale_fence_and_stale_tree_fail_closed() -> None:
    candidate = _candidate()
    thief = _spec("supervisor:idle")
    with pytest.raises(WorkStealingAuthorityError, match="fencing epoch is stale"):
        _steal(candidate, thief, expected_source_fence=1)
    with pytest.raises(WorkStealingAuthorityError, match="current tree identity"):
        _steal(candidate, thief, current_tree_id="tree:other")
    with pytest.raises(WorkStealingAuthorityError, match="current semantic state"):
        _steal(candidate, thief, current_semantic_root="semantic:other")


def test_policy_proof_merge_privacy_and_human_review_cannot_be_bypassed() -> None:
    thief = _spec("supervisor:idle")
    for flag in (
        "requires_human_review",
        "requires_privacy_review",
        "requires_proof",
        "requires_merge",
    ):
        with pytest.raises(WorkStealingAuthorityError, match="cannot bypass policy"):
            _steal(_candidate(**{flag: True}), thief)


def test_atomic_budget_transfer_accompanies_the_steal() -> None:
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
    candidate = _candidate()
    thief = _spec("supervisor:idle")
    receipt = _steal(
        candidate,
        thief,
        ledger=ledger,
        source_budget_account_id="budget:busy",
        thief_budget_account_id="budget:idle",
        budget_dimension=BudgetDimensionName.INPUT_TOKENS,
        budget_amount=40,
        expected_source_budget_revision=1,
        expected_thief_budget_revision=1,
    )
    assert receipt.budget_transferred == 40
    assert ledger.account("budget:busy").slice_for(BudgetDimensionName.INPUT_TOKENS).ceiling == 160
    assert ledger.account("budget:idle").slice_for(BudgetDimensionName.INPUT_TOKENS).ceiling == 90
    assert ledger.conserved("budget:federation", BudgetDimensionName.INPUT_TOKENS) is True


def test_ducklake_cannot_admit_a_steal() -> None:
    with pytest.raises(WorkStealingAuthorityError, match="DuckLake cannot admit"):
        _steal(
            _candidate(),
            _spec("supervisor:idle"),
            ducklake_receipt={"steals": True},
        )


def test_store_rejects_database_path(tmp_path: Path) -> None:
    with pytest.raises(WorkStealingError, match="database path"):
        WorkStealingStore(tmp_path / "control.duckdb")  # type: ignore[arg-type]


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required for steal persistence")
def test_store_records_work_steal_receipt(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    report = install_control_plane_schema(database, owner_id="owner:steal")
    assert report.to_version == 2
    client = open_embedded_client(
        database,
        owner_id="owner:steal",
        seed_generation=True,
    )
    generation = client.load_generation()
    store = WorkStealingStore(client)
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
        task_id="task:idle",
        repository_id=binding.repository_ids[0],
        tree_id=binding.repository_tree_ids[0],
        goal_id="goal:one",
        effect_class=EventEffectClass.READ_ONLY.value,
    )
    candidate = StealCandidate(
        work=work,
        source_supervisor_id="supervisor:busy",
        fencing_epoch=1,
        assignment_revision=1,
        semantic_root=binding.semantic_state_roots[0],
    )
    thief = bind_supervisor_specialization(
        binding=binding,
        supervisor_id="supervisor:idle",
        allowed_effect_classes=(EventEffectClass.READ_ONLY.value,),
        capability_refs=("capability:test",),
    )
    steal = steal_work(
        candidate,
        thief=thief,
        binding=binding,
        current_tree_id=work.tree_id,
        current_semantic_root=candidate.semantic_root,
        expected_source_fence=1,
        expected_assignment_revision=1,
    )
    revision = store.graph_revision(tenant_id=binding.tenant_id, federation_id=identity.record_id)
    store.record_steal(
        steal,
        federation_id=identity.record_id,
        binding=binding,
        expected_graph_revision=revision,
        idempotency_key="idempotency:steal",
    )
    loaded = store.load_steal(
        receipt_id="federation-receipt:" + steal.cid,
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    assert loaded["receipt_kind"] == "work_steal"
    assert loaded["content_ref"] == steal.cid
