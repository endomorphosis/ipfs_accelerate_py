"""Hermetic tests for CASF conflict-free parallel frontier compilation."""

from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.causal_frontier import (
    IndependenceAdmission,
    compile_frontier,
)
from ipfs_accelerate_py.agent_supervisor.federation.deduplication import (
    IntentIndependenceAdmission,
)
from ipfs_accelerate_py.agent_supervisor.federation.events import EventEffectClass
from ipfs_accelerate_py.agent_supervisor.federation.parallel_frontier import (
    FrontierCapacity,
    ParallelFrontierAuthorityError,
    ParallelFrontierError,
    ParallelFrontierStore,
    bind_parallel_task,
    compile_parallel_frontier,
    refuse_ducklake_parallel_authority,
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
from test.api.causal_federation.test_causal_frontier import _node, _subject
from test.api.causal_federation.test_contracts import sample_binding
from test.api.causal_federation.test_deduplication import _intent
from test.api.causal_federation.test_registry import _create
from test.api.causal_federation.test_trigger import sample_policy, sample_request


def _task(
    *,
    task_id: str,
    targets: tuple[str, ...],
    supervisor_id: str,
    subagent_id: str,
    worktree_id: str,
    lease_id: str,
    effect_class: EventEffectClass = EventEffectClass.READ_ONLY,
    goal_id: str = "goal:test",
    requires_merge_slot: bool = False,
    requires_proof_slot: bool = False,
    binding=None,
    operation: str = "federation.edit",
):
    bound = binding or sample_binding()
    intent = _intent(
        task_id=task_id,
        targets=targets,
        effect_class=effect_class,
        goal_id=goal_id,
        binding=bound,
        operation=operation,
    )
    return bind_parallel_task(
        binding=bound,
        intent=intent,
        supervisor_id=supervisor_id,
        subagent_id=subagent_id,
        worktree_id=worktree_id,
        lease_id=lease_id,
        fencing_epoch=1,
        merge_lane="merge:lane-a",
        validation_plan_ref="validation:test",
        resource_reservation_ref="reservation:resource",
        token_reservation_ref="reservation:token",
        requires_merge_slot=requires_merge_slot,
        requires_proof_slot=requires_proof_slot,
    )


def _independence(*tasks):
    left, right = tasks[0].intent, tasks[1].intent
    return IntentIndependenceAdmission(
        left_intent_cid=left.cid,
        right_intent_cid=right.cid,
        evidence_refs=("evidence:independence",),
        authoritative=True,
    )


def test_admitted_independence_forms_one_parallel_wave() -> None:
    left = _task(
        task_id="task:left",
        targets=("file:a",),
        supervisor_id="supervisor:one",
        subagent_id="subagent:one",
        worktree_id="worktree:one",
        lease_id="lease:one",
    )
    right = _task(
        task_id="task:right",
        targets=("file:b",),
        supervisor_id="supervisor:two",
        subagent_id="subagent:two",
        worktree_id="worktree:two",
        lease_id="lease:two",
    )
    compiled = compile_parallel_frontier(
        (left, right),
        binding=sample_binding(),
        independence=(_independence(left, right),),
    )
    assert compiled.admitted == ("task:left", "task:right")
    assert compiled.serialized == ()
    assert compiled.suppressed == ()
    assert compiled.assignment_refs


def test_duplicates_are_suppressed_to_one_survivor() -> None:
    left = _task(
        task_id="task:left",
        targets=("file:a", "file:b"),
        supervisor_id="supervisor:one",
        subagent_id="subagent:one",
        worktree_id="worktree:one",
        lease_id="lease:one",
    )
    right = _task(
        task_id="task:right",
        targets=("file:b", "file:a"),
        supervisor_id="supervisor:two",
        subagent_id="subagent:two",
        worktree_id="worktree:two",
        lease_id="lease:two",
    )
    compiled = compile_parallel_frontier((left, right), binding=sample_binding())
    assert compiled.admitted == ("task:left",)
    assert compiled.suppressed == ("task:right",)
    assert compiled.serialized == ()


def test_subsumed_work_is_blocked_on_the_covering_task() -> None:
    covering = _task(
        task_id="task:cover",
        targets=("file:a", "file:b", "file:c"),
        supervisor_id="supervisor:one",
        subagent_id="subagent:one",
        worktree_id="worktree:one",
        lease_id="lease:one",
    )
    part = _task(
        task_id="task:part",
        targets=("file:a", "file:b"),
        supervisor_id="supervisor:two",
        subagent_id="subagent:two",
        worktree_id="worktree:two",
        lease_id="lease:two",
    )
    compiled = compile_parallel_frontier((covering, part), binding=sample_binding())
    assert compiled.admitted == ("task:cover",)
    assert compiled.blocked == ("task:part",)


def test_shared_reads_may_run_together() -> None:
    left = _task(
        task_id="task:left",
        targets=("file:a", "file:b"),
        supervisor_id="supervisor:one",
        subagent_id="subagent:one",
        worktree_id="worktree:shared",
        lease_id="lease:one",
        goal_id="goal:one",
    )
    right = _task(
        task_id="task:right",
        targets=("file:b", "file:c"),
        supervisor_id="supervisor:two",
        subagent_id="subagent:two",
        worktree_id="worktree:shared",
        lease_id="lease:two",
        goal_id="goal:two",
    )
    compiled = compile_parallel_frontier((left, right), binding=sample_binding())
    assert compiled.admitted == ("task:left", "task:right")
    assert compiled.serialized == ()


def test_overlapping_writes_serialize() -> None:
    left = _task(
        task_id="task:left",
        targets=("file:a", "file:b"),
        supervisor_id="supervisor:one",
        subagent_id="subagent:one",
        worktree_id="worktree:one",
        lease_id="lease:one",
        effect_class=EventEffectClass.AUTHORITATIVE_STATE,
        goal_id="goal:one",
    )
    right = _task(
        task_id="task:right",
        targets=("file:b", "file:c"),
        supervisor_id="supervisor:two",
        subagent_id="subagent:two",
        worktree_id="worktree:two",
        lease_id="lease:two",
        effect_class=EventEffectClass.AUTHORITATIVE_STATE,
        goal_id="goal:two",
    )
    compiled = compile_parallel_frontier((left, right), binding=sample_binding())
    assert compiled.admitted == ("task:left",)
    assert compiled.serialized == ("task:right",)


def test_disjoint_work_without_independence_serializes() -> None:
    left = _task(
        task_id="task:left",
        targets=("file:a",),
        supervisor_id="supervisor:one",
        subagent_id="subagent:one",
        worktree_id="worktree:one",
        lease_id="lease:one",
    )
    right = _task(
        task_id="task:right",
        targets=("file:b",),
        supervisor_id="supervisor:two",
        subagent_id="subagent:two",
        worktree_id="worktree:two",
        lease_id="lease:two",
    )
    compiled = compile_parallel_frontier((left, right), binding=sample_binding())
    assert compiled.admitted == ("task:left",)
    assert compiled.serialized == ("task:right",)
    assert compiled.admitted != ("task:left", "task:right")


def test_irreversible_effect_is_the_sole_admitted_task() -> None:
    pay = _task(
        task_id="task:pay",
        targets=("file:a",),
        supervisor_id="supervisor:one",
        subagent_id="subagent:one",
        worktree_id="worktree:one",
        lease_id="lease:one",
        effect_class=EventEffectClass.PAYMENT,
        goal_id="goal:pay",
    )
    other = _task(
        task_id="task:other",
        targets=("file:b",),
        supervisor_id="supervisor:two",
        subagent_id="subagent:two",
        worktree_id="worktree:two",
        lease_id="lease:two",
    )
    compiled = compile_parallel_frontier(
        (pay, other),
        binding=sample_binding(),
        independence=(_independence(pay, other),),
    )
    assert compiled.admitted == ("task:pay",)
    assert compiled.serialized == ("task:other",)


def test_merge_slot_capacity_reduces_the_wave() -> None:
    left = _task(
        task_id="task:left",
        targets=("file:a",),
        supervisor_id="supervisor:one",
        subagent_id="subagent:one",
        worktree_id="worktree:one",
        lease_id="lease:one",
        requires_merge_slot=True,
    )
    right = _task(
        task_id="task:right",
        targets=("file:b",),
        supervisor_id="supervisor:two",
        subagent_id="subagent:two",
        worktree_id="worktree:two",
        lease_id="lease:two",
        requires_merge_slot=True,
    )
    compiled = compile_parallel_frontier(
        (left, right),
        binding=sample_binding(),
        independence=(_independence(left, right),),
        capacity=FrontierCapacity(merge_slots=1, proof_slots=2, max_parallel=8),
    )
    assert compiled.admitted == ("task:left",)
    assert compiled.serialized == ("task:right",)
    assert compiled.merge_order == ("task:left", "task:right")


def test_do_not_wake_supervisors_stay_asleep() -> None:
    binding = sample_binding()
    changed = _node("node:changed", "symbol:changed")
    independent = _node("node:independent", "symbol:independent")
    causal = compile_frontier(
        event_id="event:change",
        binding=binding,
        graph_revision=1,
        nodes=(changed, independent),
        edges=(),
        changed_fact_refs=("node:changed",),
        subjects=(
            _subject("supervisor:one", "node:changed"),
            _subject("supervisor:idle", "node:independent"),
        ),
        independence=(
            IndependenceAdmission(
                subject=_subject("supervisor:idle", "node:independent"),
                evidence_refs=("evidence:independence",),
                authoritative=True,
            ),
        ),
    )
    awake = _task(
        task_id="task:awake",
        targets=("file:a",),
        supervisor_id="supervisor:one",
        subagent_id="subagent:one",
        worktree_id="worktree:one",
        lease_id="lease:one",
        binding=binding,
    )
    idle = _task(
        task_id="task:idle",
        targets=("file:b",),
        supervisor_id="supervisor:idle",
        subagent_id="subagent:two",
        worktree_id="worktree:two",
        lease_id="lease:two",
        binding=binding,
    )
    compiled = compile_parallel_frontier(
        (awake, idle),
        binding=binding,
        causal_frontier=causal,
        independence=(_independence(awake, idle),),
    )
    assert compiled.admitted == ("task:awake",)
    assert compiled.asleep == ("task:idle",)
    assert "task:idle" not in compiled.admitted
    assert "task:idle" not in compiled.serialized


def test_force_parallel_fails_closed() -> None:
    task = _task(
        task_id="task:left",
        targets=("file:a",),
        supervisor_id="supervisor:one",
        subagent_id="subagent:one",
        worktree_id="worktree:one",
        lease_id="lease:one",
    )
    with pytest.raises(ParallelFrontierAuthorityError, match="cannot be forced"):
        compile_parallel_frontier(
            (task,),
            binding=sample_binding(),
            force_parallel=("task:left",),
        )


def test_missing_lease_fails_closed() -> None:
    intent = _intent(task_id="task:left", targets=("file:a",))
    with pytest.raises(Exception, match="lease_id"):
        bind_parallel_task(
            binding=sample_binding(),
            intent=intent,
            supervisor_id="supervisor:one",
            subagent_id="subagent:one",
            worktree_id="worktree:one",
            lease_id="",
            fencing_epoch=1,
            merge_lane="merge:lane-a",
            validation_plan_ref="validation:test",
            resource_reservation_ref="reservation:resource",
            token_reservation_ref="reservation:token",
        )


def test_validation_plan_must_match_intent() -> None:
    intent = _intent(task_id="task:left", targets=("file:a",))
    with pytest.raises(ParallelFrontierAuthorityError, match="validation plan"):
        bind_parallel_task(
            binding=sample_binding(),
            intent=intent,
            supervisor_id="supervisor:one",
            subagent_id="subagent:one",
            worktree_id="worktree:one",
            lease_id="lease:one",
            fencing_epoch=1,
            merge_lane="merge:lane-a",
            validation_plan_ref="validation:other",
            resource_reservation_ref="reservation:resource",
            token_reservation_ref="reservation:token",
        )


def test_same_lease_cannot_admit_two_tasks() -> None:
    left = _task(
        task_id="task:left",
        targets=("file:a",),
        supervisor_id="supervisor:one",
        subagent_id="subagent:one",
        worktree_id="worktree:one",
        lease_id="lease:shared",
    )
    right = _task(
        task_id="task:right",
        targets=("file:b",),
        supervisor_id="supervisor:two",
        subagent_id="subagent:two",
        worktree_id="worktree:two",
        lease_id="lease:shared",
    )
    compiled = compile_parallel_frontier(
        (left, right),
        binding=sample_binding(),
        independence=(_independence(left, right),),
    )
    assert compiled.admitted == ("task:left",)
    assert compiled.serialized == ("task:right",)


def test_ducklake_cannot_admit_parallel_frontier() -> None:
    with pytest.raises(ParallelFrontierAuthorityError, match="DuckLake cannot admit"):
        refuse_ducklake_parallel_authority({"parallelizes": True})
    task = _task(
        task_id="task:left",
        targets=("file:a",),
        supervisor_id="supervisor:one",
        subagent_id="subagent:one",
        worktree_id="worktree:one",
        lease_id="lease:one",
    )
    with pytest.raises(Exception, match="DuckLake cannot"):
        compile_parallel_frontier(
            (task,),
            binding=sample_binding(),
            ducklake_receipt={"authoritative": True},
        )


def test_store_rejects_database_path(tmp_path: Path) -> None:
    with pytest.raises(ParallelFrontierError, match="database path"):
        ParallelFrontierStore(tmp_path / "control.duckdb")  # type: ignore[arg-type]


@pytest.mark.skipif(
    not duckdb_available(), reason="DuckDB required for parallel frontier persistence"
)
def test_store_records_admitted_and_serialized_bindings(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    report = install_control_plane_schema(database, owner_id="owner:parallel-frontier")
    assert report.to_version == 3
    client = open_embedded_client(
        database,
        owner_id="owner:parallel-frontier",
        seed_generation=True,
    )
    generation = client.load_generation()
    store = ParallelFrontierStore(client)
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
    left = _task(
        task_id="task:left",
        targets=("file:a",),
        supervisor_id="supervisor:one",
        subagent_id="subagent:one",
        worktree_id="worktree:one",
        lease_id="lease:one",
        binding=binding,
        requires_merge_slot=True,
    )
    right = _task(
        task_id="task:right",
        targets=("file:b",),
        supervisor_id="supervisor:two",
        subagent_id="subagent:two",
        worktree_id="worktree:two",
        lease_id="lease:two",
        binding=binding,
        requires_merge_slot=True,
    )
    compiled = compile_parallel_frontier(
        (left, right),
        binding=binding,
        independence=(_independence(left, right),),
        capacity=FrontierCapacity(merge_slots=1),
    )
    revision = store.graph_revision(tenant_id=binding.tenant_id, federation_id=identity.record_id)
    store.record_frontier(
        compiled,
        federation_id=identity.record_id,
        binding=binding,
        expected_graph_revision=revision,
        idempotency_key="idempotency:frontier",
        tasks=(left, right),
        event_watermark=3,
    )
    loaded = store.load_frontier(
        receipt_id="federation-receipt:" + compiled.cid,
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    assert loaded["receipt_kind"] == "parallel_frontier"
    assert loaded["content_ref"] == compiled.cid
    admitted_binding = store.load_task_binding(
        binding_id="task-binding:"
        + content_identity(
            {
                "wave_id": compiled.wave_id,
                "task_id": "task:left",
                "status": "admitted_parallel",
            }
        ),
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    assert admitted_binding["status"] == "admitted_parallel"
    serialized_binding = store.load_task_binding(
        binding_id="task-binding:"
        + content_identity(
            {
                "wave_id": compiled.wave_id,
                "task_id": "task:right",
                "status": "serialized",
            }
        ),
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    assert serialized_binding["status"] == "serialized"
