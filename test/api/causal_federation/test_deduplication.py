"""Hermetic tests for CASF duplicate-work and task-subsumption detection."""

from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.deduplication import (
    DeduplicationAuthorityError,
    DeduplicationError,
    DeduplicationStore,
    IntentDisposition,
    IntentIndependenceAdmission,
    bind_task_intent,
    classify_intent_pair,
    classify_intents,
    refuse_ducklake_dedup_authority,
)
from ipfs_accelerate_py.agent_supervisor.federation.events import EventEffectClass
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


def _intent(
    *,
    task_id: str,
    targets: tuple[str, ...],
    operation: str = "federation.edit",
    effect_class: EventEffectClass = EventEffectClass.READ_ONLY,
    goal_id: str = "goal:test",
    tree_id: str = "",
    binding=None,
):
    return bind_task_intent(
        binding=binding or sample_binding(),
        goal_id=goal_id,
        operation=operation,
        targets=targets,
        acceptance_ref="acceptance:test",
        effect_class=effect_class,
        validation_ref="validation:test",
        subgoal_id="subgoal:test",
        task_id=task_id,
        tree_id=tree_id,
    )


def test_exact_duplicates_share_one_canonical_result() -> None:
    left = _intent(task_id="task:left", targets=("file:a", "file:b"))
    right = _intent(task_id="task:right", targets=("file:b", "file:a"))
    relation = classify_intent_pair(left, right)
    assert relation.disposition is IntentDisposition.DUPLICATE
    assert relation.resolution_kind == "share_result"
    assert relation.canonical_task_id == "task:" + left.cid
    report = classify_intents((left, right))
    assert report.duplicate_pairs == (relation.cid,)
    assert dict(report.canonical_task_ids)["task:left"] == relation.canonical_task_id
    assert dict(report.canonical_task_ids)["task:right"] == relation.canonical_task_id


def test_strict_subset_is_subsumed_and_depends_on_covering_task() -> None:
    covering = _intent(task_id="task:cover", targets=("file:a", "file:b", "file:c"))
    part = _intent(task_id="task:part", targets=("file:a", "file:b"))
    relation = classify_intent_pair(covering, part)
    assert relation.disposition is IntentDisposition.SUBSUMED
    assert relation.resolution_kind == "depend"
    assert relation.covering_task_id == "task:cover"


def test_read_overlap_gets_explicit_boundaries() -> None:
    left = _intent(task_id="task:left", targets=("file:a", "file:b"), goal_id="goal:one")
    right = _intent(task_id="task:right", targets=("file:b", "file:c"), goal_id="goal:two")
    relation = classify_intent_pair(left, right)
    assert relation.disposition is IntentDisposition.OVERLAP
    assert relation.resolution_kind == "bound"
    assert relation.boundary_refs == ("file:b",)


def test_overlapping_writes_serialize() -> None:
    left = _intent(
        task_id="task:left",
        targets=("file:a", "file:b"),
        effect_class=EventEffectClass.AUTHORITATIVE_STATE,
        goal_id="goal:one",
    )
    right = _intent(
        task_id="task:right",
        targets=("file:b", "file:c"),
        effect_class=EventEffectClass.AUTHORITATIVE_STATE,
        goal_id="goal:two",
    )
    relation = classify_intent_pair(left, right)
    assert relation.disposition is IntentDisposition.CONFLICT
    assert relation.resolution_kind == "serialize"


def test_disjoint_targets_without_independence_serialize() -> None:
    left = _intent(task_id="task:left", targets=("file:a",))
    right = _intent(task_id="task:right", targets=("file:b",))
    relation = classify_intent_pair(left, right)
    assert relation.disposition is IntentDisposition.CONFLICT
    assert relation.evidence_refs == ("dedup:unknown-serializes",)
    report = classify_intents((left, right))
    assert report.parallel_pairs == ()
    assert report.serial_pairs == (relation.cid,)


def test_admitted_independence_may_run_concurrently() -> None:
    left = _intent(task_id="task:left", targets=("file:a",))
    right = _intent(task_id="task:right", targets=("file:b",))
    admission = IntentIndependenceAdmission(
        left_intent_cid=left.cid,
        right_intent_cid=right.cid,
        evidence_refs=("evidence:independence",),
        authoritative=True,
    )
    relation = classify_intent_pair(left, right, independence=admission)
    assert relation.disposition is IntentDisposition.INDEPENDENT
    assert relation.resolution_kind == "admit_parallel"
    report = classify_intents((left, right), independence=(admission,))
    assert report.parallel_pairs == (relation.cid,)
    assert report.serial_pairs == ()


def test_retrieval_nomination_cannot_prove_independence() -> None:
    left = _intent(task_id="task:left", targets=("file:a",))
    right = _intent(task_id="task:right", targets=("file:b",))
    with pytest.raises(DeduplicationAuthorityError, match="cannot prove independence"):
        IntentIndependenceAdmission(
            left_intent_cid=left.cid,
            right_intent_cid=right.cid,
            evidence_refs=("evidence:vector",),
            authoritative=False,
        )


def test_independence_cannot_suppress_overlapping_targets() -> None:
    left = _intent(task_id="task:left", targets=("file:a", "file:b"))
    right = _intent(
        task_id="task:right",
        targets=("file:b", "file:c"),
        goal_id="goal:other",
    )
    admission = IntentIndependenceAdmission(
        left_intent_cid=left.cid,
        right_intent_cid=right.cid,
        evidence_refs=("evidence:independence",),
        authoritative=True,
    )
    with pytest.raises(DeduplicationAuthorityError, match="cannot suppress overlapping"):
        classify_intent_pair(left, right, independence=admission)


def test_model_independence_cannot_admit_parallel_via_nomination_flag() -> None:
    left = _intent(task_id="task:left", targets=("file:a",))
    right = _intent(task_id="task:right", targets=("file:b",))
    from ipfs_accelerate_py.agent_supervisor.federation.deduplication import IntentRelation

    with pytest.raises(DeduplicationAuthorityError, match="nomination-only independence"):
        IntentRelation(
            left_task_id=left.task_id,
            right_task_id=right.task_id,
            left_intent_cid=left.cid,
            right_intent_cid=right.cid,
            disposition=IntentDisposition.INDEPENDENT,
            resolution_kind="admit_parallel",
            canonical_task_id=left.task_id,
            evidence_refs=("evidence:model",),
            nomination_only=True,
        )


def test_different_trees_are_not_duplicates() -> None:
    binding = sample_binding(
        repository_ids=("repo:test", "repo:other"),
        repository_tree_ids=("tree:test", "tree:other"),
    )
    left = _intent(task_id="task:left", targets=("file:a",), binding=binding)
    right = bind_task_intent(
        binding=binding,
        goal_id="goal:test",
        operation="federation.edit",
        targets=("file:a",),
        acceptance_ref="acceptance:test",
        effect_class=EventEffectClass.READ_ONLY,
        validation_ref="validation:test",
        subgoal_id="subgoal:test",
        task_id="task:right",
        tree_id="tree:other",
        repository_id="repo:other",
    )
    relation = classify_intent_pair(left, right)
    assert relation.disposition is not IntentDisposition.DUPLICATE
    assert left.cid != right.cid


def test_unbound_tree_fails_closed() -> None:
    with pytest.raises(DeduplicationAuthorityError, match="tree identity mismatches"):
        bind_task_intent(
            binding=sample_binding(),
            goal_id="goal:test",
            operation="federation.edit",
            targets=("file:a",),
            acceptance_ref="acceptance:test",
            effect_class=EventEffectClass.READ_ONLY,
            validation_ref="validation:test",
            task_id="task:bad",
            tree_id="tree:other",
        )


def test_unknown_effect_class_fails_closed() -> None:
    with pytest.raises(Exception, match="effect_class is not closed"):
        bind_task_intent(
            binding=sample_binding(),
            goal_id="goal:test",
            operation="federation.edit",
            targets=("file:a",),
            acceptance_ref="acceptance:test",
            effect_class="vibes",
            validation_ref="validation:test",
            task_id="task:bad",
        )


def test_empty_targets_fail_closed() -> None:
    with pytest.raises(Exception, match="must not be empty"):
        _intent(task_id="task:empty", targets=())


def test_ducklake_cannot_admit_dedup_authority() -> None:
    with pytest.raises(DeduplicationAuthorityError, match="DuckLake cannot admit"):
        refuse_ducklake_dedup_authority({"authoritative": True})
    left = _intent(task_id="task:left", targets=("file:a",))
    right = _intent(task_id="task:right", targets=("file:a",))
    with pytest.raises(DeduplicationAuthorityError, match="DuckLake cannot admit"):
        classify_intent_pair(left, right, ducklake_receipt={"deduplicates": True})


def test_pair_classification_is_commutative_for_duplicates_and_conflicts() -> None:
    left = _intent(task_id="task:left", targets=("file:a", "file:b"))
    right = _intent(task_id="task:right", targets=("file:b", "file:a"))
    assert classify_intent_pair(left, right).cid == classify_intent_pair(right, left).cid
    write_left = _intent(
        task_id="task:w1",
        targets=("file:a", "file:b"),
        effect_class=EventEffectClass.AUTHORITATIVE_STATE,
        goal_id="goal:one",
    )
    write_right = _intent(
        task_id="task:w2",
        targets=("file:b", "file:c"),
        effect_class=EventEffectClass.AUTHORITATIVE_STATE,
        goal_id="goal:two",
    )
    assert (
        classify_intent_pair(write_left, write_right).disposition
        is classify_intent_pair(write_right, write_left).disposition
    )


def test_store_rejects_database_path(tmp_path: Path) -> None:
    with pytest.raises(DeduplicationError, match="database path"):
        DeduplicationStore(tmp_path / "control.duckdb")  # type: ignore[arg-type]


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required for deduplication persistence")
def test_store_records_duplicate_and_conflict_resolutions(tmp_path: Path) -> None:
    database = tmp_path / "control.duckdb"
    report = install_control_plane_schema(database, owner_id="owner:deduplication")
    assert report.to_version == 3
    client = open_embedded_client(
        database,
        owner_id="owner:deduplication",
        seed_generation=True,
    )
    generation = client.load_generation()
    store = DeduplicationStore(client)
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
    left = _intent(task_id="task:left", targets=("file:a",), binding=binding)
    right = _intent(task_id="task:right", targets=("file:a",), binding=binding)
    duplicate = classify_intent_pair(left, right)
    writer = _intent(
        task_id="task:writer",
        targets=("file:a", "file:b"),
        effect_class=EventEffectClass.AUTHORITATIVE_STATE,
        goal_id="goal:write",
        binding=binding,
    )
    conflict = classify_intent_pair(left, writer)
    revision = store.graph_revision(tenant_id=binding.tenant_id, federation_id=identity.record_id)
    revision = store.record_relation(
        duplicate,
        federation_id=identity.record_id,
        binding=binding,
        expected_graph_revision=revision,
        idempotency_key="idempotency:duplicate",
        effect_class=EventEffectClass.READ_ONLY,
    ).graph_revision
    store.record_relation(
        conflict,
        federation_id=identity.record_id,
        binding=binding,
        expected_graph_revision=revision,
        idempotency_key="idempotency:conflict",
        effect_class=EventEffectClass.AUTHORITATIVE_STATE,
    )
    loaded_duplicate = store.load_conflict(
        conflict_id="task-conflict:" + duplicate.cid,
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    loaded_conflict = store.load_conflict(
        conflict_id="task-conflict:" + conflict.cid,
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    loaded_resolution = store.load_resolution(
        resolution_id="task-resolution:" + duplicate.cid,
        tenant_id=binding.tenant_id,
        federation_id=identity.record_id,
    )
    assert loaded_duplicate["conflict_kind"] == "duplicate"
    assert loaded_conflict["conflict_kind"] == "conflict"
    assert loaded_resolution["resolution_kind"] == "share_result"
