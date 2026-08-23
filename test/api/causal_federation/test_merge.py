"""Hermetic tests for CASF isolated worktrees, merge train, and proof gates."""

from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.events import EventEffectClass
from ipfs_accelerate_py.agent_supervisor.federation.merge import (
    FederationMergeCoordinator,
    MergeAuthorityError,
    MergeCandidate,
    MergeError,
    MergeStore,
    bind_worktree,
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

CAPABILITIES = ("capability:merge", "capability:test")


def _worktree(
    worktree_id: str = "worktree:alpha",
    *,
    isolated: bool = True,
    exclusive: bool = True,
    fencing_epoch: int = 1,
):
    return bind_worktree(
        binding=sample_binding(),
        worktree_id=worktree_id,
        owner_session_id="session:owner",
        head_commit_id="commit:base",
        branch_name="branch:source",
        fencing_epoch=fencing_epoch,
        isolated=isolated,
        exclusive=exclusive,
    )


def _candidate(**overrides: object) -> MergeCandidate:
    values: dict[str, object] = {
        "task_id": "task:one",
        "worktree": _worktree(),
        "merge_lane": "merge-lane:default",
        "source_branch": "branch:source",
        "target_branch": "branch:target",
        "effect_class": EventEffectClass.AUTHORITATIVE_STATE.value,
        "fencing_epoch": 1,
        "proof_status": "proved",
        "test_status": "passed",
    }
    values.update(overrides)
    return MergeCandidate(**values)  # type: ignore[arg-type]


def _compile(candidates, **kwargs: object):
    coordinator = FederationMergeCoordinator()
    values: dict[str, object] = {
        "binding": sample_binding(),
        "capability_refs": CAPABILITIES,
        "expected_fence": 1,
    }
    values.update(kwargs)
    return coordinator.compile(candidates, **values)  # type: ignore[arg-type]


def _release(train, task_id: str, **kwargs: object):
    coordinator = FederationMergeCoordinator()
    values: dict[str, object] = {
        "binding": sample_binding(),
        "expected_fence": train.fencing_epoch,
        "result_commit_id": "commit:merged",
    }
    values.update(kwargs)
    return coordinator.release(train, task_id, **values)  # type: ignore[arg-type]


def test_isolated_worktrees_reject_exclusive_overlap() -> None:
    shared = _worktree("worktree:shared")
    left = _candidate(task_id="task:left", worktree=shared)
    right = _candidate(task_id="task:right", worktree=shared)
    with pytest.raises(MergeAuthorityError, match="isolated worktrees cannot overlap"):
        _compile((left, right))


def test_shared_reads_may_share_a_worktree() -> None:
    shared = _worktree("worktree:reads", isolated=False, exclusive=False)
    left = _candidate(
        task_id="task:left",
        worktree=shared,
        effect_class=EventEffectClass.READ_ONLY.value,
    )
    right = _candidate(
        task_id="task:right",
        worktree=shared,
        effect_class=EventEffectClass.READ_ONLY.value,
    )
    train = _compile((right, left))
    assert train.merge_order == ("task:left", "task:right")
    assert train.worktrees[0].worktree_id == "worktree:reads"


def test_exclusive_work_requires_an_isolated_worktree() -> None:
    with pytest.raises(MergeAuthorityError, match="isolated worktree"):
        bind_worktree(
            binding=sample_binding(),
            worktree_id="worktree:open",
            owner_session_id="session:owner",
            head_commit_id="commit:base",
            branch_name="branch:source",
            isolated=False,
            exclusive=True,
        )


def test_filesystem_worktree_path_fails_closed() -> None:
    with pytest.raises(MergeAuthorityError, match="filesystem paths"):
        bind_worktree(
            binding=sample_binding(),
            worktree_id="/tmp/worktree",
            owner_session_id="session:owner",
            head_commit_id="commit:base",
            branch_name="branch:source",
        )


def test_explicit_merge_order_follows_predecessors() -> None:
    first = _candidate(
        task_id="task:first",
        worktree=_worktree("worktree:first"),
    )
    second = _candidate(
        task_id="task:second",
        worktree=_worktree("worktree:second"),
        predecessor_task_ids=("task:first",),
    )
    train = _compile((second, first))
    assert train.merge_order == ("task:first", "task:second")
    assert tuple(item.ordinal for item in train.entries) == (1, 2)
    head = _release(train, "task:first")
    assert head.outcome == "merged"
    assert head.fencing_epoch == 2
    tail = _release(train, "task:second", merged_task_ids=("task:first",))
    assert tail.task_id == "task:second"


def test_out_of_order_release_fails_closed() -> None:
    first = _candidate(task_id="task:first", worktree=_worktree("worktree:first"))
    second = _candidate(
        task_id="task:second",
        worktree=_worktree("worktree:second"),
        predecessor_task_ids=("task:first",),
    )
    train = _compile((first, second))
    with pytest.raises(MergeAuthorityError, match="explicit merge order"):
        _release(train, "task:second")


def test_cycle_in_predecessors_fails_closed() -> None:
    left = _candidate(
        task_id="task:left",
        worktree=_worktree("worktree:left"),
        predecessor_task_ids=("task:right",),
    )
    right = _candidate(
        task_id="task:right",
        worktree=_worktree("worktree:right"),
        predecessor_task_ids=("task:left",),
    )
    with pytest.raises(MergeAuthorityError, match="cycle"):
        _compile((left, right))


def test_proof_release_gate_requires_proved_and_passed() -> None:
    train = _compile((_candidate(proof_status="open", test_status="pending"),))
    with pytest.raises(MergeAuthorityError, match="proof-release gate"):
        _release(train, "task:one")
    ready = _compile((_candidate(),))
    receipt = _release(ready, "task:one")
    assert receipt.outcome == "merged"
    assert receipt.result_commit_id == "commit:merged"


def test_missing_seal_fails_closed_when_required() -> None:
    train = _compile((_candidate(requires_seal=True, sealed=False),))
    with pytest.raises(MergeAuthorityError, match="current seal"):
        _release(train, "task:one")


def test_stale_fence_fails_closed() -> None:
    with pytest.raises(MergeAuthorityError, match="fencing epoch is stale"):
        _compile((_candidate(),), expected_fence=9)
    train = _compile((_candidate(),))
    with pytest.raises(MergeAuthorityError, match="fencing epoch is stale"):
        _release(train, "task:one", expected_fence=9)


def test_missing_merge_capability_fails_closed() -> None:
    with pytest.raises(MergeAuthorityError, match="merge capability is missing"):
        _compile((_candidate(),), capability_refs=("capability:test",))


def test_ducklake_cannot_admit_a_merge() -> None:
    with pytest.raises(MergeAuthorityError, match="DuckLake cannot admit"):
        _compile((_candidate(),), ducklake_receipt={"merges": True})
    with pytest.raises(MergeAuthorityError, match="DuckLake cannot admit"):
        _compile((_candidate(),), ducklake_receipt={"authoritative": True})


def test_store_rejects_database_path(tmp_path: Path) -> None:
    with pytest.raises(MergeError, match="database path"):
        MergeStore(tmp_path / "control.duckdb")  # type: ignore[arg-type]


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required for merge persistence")
def test_store_records_worktree_queue_and_attempt(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    report = install_control_plane_schema(database, owner_id="owner:merge")
    assert report.to_version == 2
    client = open_embedded_client(
        database,
        owner_id="owner:merge",
        seed_generation=True,
    )
    generation = client.load_generation()
    store = MergeStore(client)
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
    coordinator = FederationMergeCoordinator()
    worktree = coordinator.bind_worktree(
        binding=binding,
        worktree_id="worktree:alpha",
        owner_session_id="session:owner",
        head_commit_id="commit:base",
        branch_name="branch:source",
        fencing_epoch=1,
    )
    candidate = MergeCandidate(
        task_id="task:one",
        worktree=worktree,
        merge_lane="merge-lane:default",
        source_branch="branch:source",
        target_branch="branch:target",
        effect_class=EventEffectClass.AUTHORITATIVE_STATE.value,
        fencing_epoch=1,
        proof_status="proved",
        test_status="passed",
    )
    train = coordinator.compile(
        (candidate,),
        binding=binding,
        capability_refs=CAPABILITIES,
        expected_fence=1,
    )
    receipt = coordinator.release(
        train,
        "task:one",
        binding=binding,
        expected_fence=1,
        result_commit_id="commit:merged",
    )
    revision = store.graph_revision(tenant_id=binding.tenant_id, federation_id=identity.record_id)
    store.record_train(
        train,
        federation_id=identity.record_id,
        binding=binding,
        expected_graph_revision=revision,
        idempotency_key="idempotency:merge-train",
    )
    revision = store.graph_revision(tenant_id=binding.tenant_id, federation_id=identity.record_id)
    store.record_attempt(
        receipt,
        federation_id=identity.record_id,
        binding=binding,
        expected_graph_revision=revision,
        idempotency_key="idempotency:merge-attempt",
    )
    loaded_worktree = store.load_worktree(worktree_id=worktree.worktree_id)
    loaded_entry = store.load_entry(entry_id=train.entries[0].entry_id)
    loaded_attempt = store.load_attempt(attempt_id="merge-attempt:" + receipt.cid)
    assert loaded_worktree["status"] == "isolated"
    assert loaded_worktree["fence_epoch"] == 1
    assert loaded_entry["status"] == "queued"
    assert loaded_entry["ordinal"] == 1
    assert loaded_attempt["status"] == "merged"
    assert loaded_attempt["result_commit_id"] == "commit:merged"
