"""Tests for append-only plan revision store and dual Markdown/DuckDB apply (PDR-031)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import (
    CompletionAuthority,
    DeltaEffectClass,
    LifecycleState,
    MergeStrategyKind,
    PlanAuthorityRoots,
    PlanCompletionRule,
    PlanConflictContract,
    PlanDelta,
    PlanDeltaItem,
    PlanDeltaOperation,
    PlanLeaseContract,
    PlanMergeStrategy,
    PlanOrigin,
    PlanPopulationDigest,
    PlanProviderContract,
    PlanResourceContract,
    PlanRetryContract,
    PlanRevision,
    PlanRevisionLifecycleError,
    PlanValidationNode,
    PlanWorktreeContract,
    PopulationKind,
    plan_revision_cid,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_task_source import (
    DuckDBTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.markdown_task_source import (
    MarkdownTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.plan_revision_store import (
    PLAN_REVISION_STORE_INTERFACE,
    PlanRevisionApplyRequest,
    PlanRevisionApplyState,
    PlanRevisionStore,
    PlanRevisionStoreConflictError,
    PlanRevisionStoreQuarantinedError,
    PlanRevisionStoreStaleError,
    open_plan_revision_store,
)

duckdb = pytest.importorskip("duckdb")


def _cid(name: str) -> str:
    return plan_revision_cid({"fixture": name})


def _roots(**changes: object) -> PlanAuthorityRoots:
    values: dict[str, object] = {
        "repository_id": "repository:sha256:test",
        "repository_root_cid": _cid("repo-root"),
        "dirty_worktree_root": _cid("dirty"),
        "task_source_id": "task-source:markdown:board",
        "task_source_revision": _cid("ts-rev-1"),
        "policy_root": _cid("policy"),
        "intent_ir_root": _cid("intent"),
        "legal_ir_root": _cid("legal"),
        "security_ir_root": _cid("security"),
        "program_root": _cid("program"),
        "capability_catalog_root": _cid("capability"),
        "provider_catalog_root": _cid("provider-catalog"),
        "usage_policy_root": _cid("usage"),
        "configuration_root": _cid("config"),
    }
    values.update(changes)
    return PlanAuthorityRoots(**values)


def _population(kind: PopulationKind, *members: str) -> PlanPopulationDigest:
    return PlanPopulationDigest(kind=kind, member_cids=members)


def _revision(**changes: object) -> PlanRevision:
    values: dict[str, object] = {
        "plan_root_cid": _cid("plan-root-1"),
        "semantic_revision": 1,
        "parent_plan_root": "",
        "origin": PlanOrigin.CREATE,
        "roots": _roots(),
        "request_cid": _cid("create-request"),
        "delta_cid": "",
        "scan_receipt_cid": _cid("scan"),
        "query_plan_cid": _cid("query"),
        "evidence_bundle_cid": _cid("evidence"),
        "admission_receipt_cid": _cid("admission"),
        "execution_plan_cid": _cid("exec-plan"),
        "goal_population": _population(PopulationKind.RETAINED, _cid("goal-1")),
        "task_population": _population(PopulationKind.RETAINED, _cid("task-1")),
        "added_population": _population(
            PopulationKind.ADDED, _cid("goal-1"), _cid("task-1")
        ),
        "superseded_population": _population(PopulationKind.SUPERSEDED),
        "retained_population": _population(PopulationKind.RETAINED),
        "deferred_population": _population(PopulationKind.DEFERRED),
        "claimed_population": _population(PopulationKind.CLAIMED),
        "completed_population": _population(PopulationKind.COMPLETED),
        "blocked_population": _population(PopulationKind.BLOCKED),
        "resource_contract": PlanResourceContract(),
        "provider_contract": PlanProviderContract(),
        "lease_contract": PlanLeaseContract(),
        "retry_contract": PlanRetryContract(),
        "worktree_contract": PlanWorktreeContract(),
        "merge_strategy": PlanMergeStrategy(kind=MergeStrategyKind.SERIAL),
        "conflict_contract": PlanConflictContract(
            predicted_files=(
                "ipfs_accelerate_py/agent_supervisor/task_sources/plan_revision_store.py",
            ),
        ),
        "completion_rule": PlanCompletionRule(
            authority=CompletionAuthority.VALIDATION_GATE,
        ),
        "validation_dag": (
            PlanValidationNode(
                validation_key="validation:pytest",
                argv=("python", "-m", "pytest", "-q"),
            ),
        ),
        "event_cursor": _cid("cursor-0"),
    }
    values.update(changes)
    return PlanRevision(**values)


def _delta_item(**changes: object) -> PlanDeltaItem:
    values: dict[str, object] = {
        "item_key": "delta:add-task",
        "operation": PlanDeltaOperation.ADD_TASK,
        "target_cid": "",
        "expected_target_lifecycle": LifecycleState.PROPOSED,
        "expected_target_spec_revision": "",
        "before_digest": "",
        "after_record_cid": _cid("new-task"),
        "effect_class": DeltaEffectClass.MATERIALIZABLE_NOW,
        "rationale": "Add a successor task.",
        "expected_effects": ("append-task",),
    }
    values.update(changes)
    return PlanDeltaItem(**values)


def _delta(**changes: object) -> PlanDelta:
    values: dict[str, object] = {
        "base_plan_root": _cid("plan-root-1"),
        "base_plan_revision": 1,
        "request_cid": _cid("steer-request"),
        "roots": _roots(),
        "items": (_delta_item(),),
        "expected_effects": ("append-task",),
        "deferred_item_keys": (),
        "claimed_population_digest": _cid("claimed-pop"),
        "accepted_population_digest": _cid("accepted-pop"),
        "scan_receipt_cid": _cid("scan"),
        "evidence_bundle_cid": _cid("evidence"),
        "admission_receipt_cid": _cid("admission"),
    }
    values.update(changes)
    return PlanDelta(**values)


def _canonical_fixture():
    from test.api.test_agent_supervisor_task_source_e2e import _canonical_fixture as fixture

    return fixture()


def test_interface_constant_and_open_store(tmp_path: Path) -> None:
    assert PLAN_REVISION_STORE_INTERFACE == "PlanRevisionStore@1"
    store = open_plan_revision_store(tmp_path / "revisions")
    assert store.get_active() is None
    assert store.list_revision_cids() == ()


def test_journal_intent_reobserves_roots_and_is_durable(tmp_path: Path) -> None:
    store = PlanRevisionStore(tmp_path / "store")
    revision = _revision()
    intent = store.journal_intent(
        PlanRevisionApplyRequest(
            revision=revision,
            observed_roots=revision.roots,
            idempotency_key="idem:create-1",
            expected_effects=("materialize-revision-1",),
        )
    )
    assert intent.intent_cid
    assert intent.state is PlanRevisionApplyState.INTENT_JOURNALED
    assert intent.expected_effects == ("materialize-revision-1",)
    reloaded = store.load_intent(intent.intent_cid)
    assert reloaded.to_dict() == intent.to_dict()
    continuation = store.load_continuation("idem:create-1")
    assert continuation is not None
    assert continuation["intent_cid"] == intent.intent_cid

    stale_roots = replace(revision.roots, dirty_worktree_root=_cid("dirty-stale"))
    with pytest.raises(PlanRevisionStoreStaleError):
        store.journal_intent(
            PlanRevisionApplyRequest(
                revision=revision,
                observed_roots=stale_roots,
                idempotency_key="idem:stale",
            )
        )


def test_apply_create_markdown_and_duckdb_round_trip_and_exact_cids(
    tmp_path: Path,
) -> None:
    graph, admission, aliases, tree_id = _canonical_fixture()
    markdown = MarkdownTaskSource(
        tmp_path / "tasks.md",
        root=tmp_path,
        task_prefix="FIX",
        board_namespace="fixture",
    )
    duck = DuckDBTaskSource(tmp_path / "tasks.duckdb")
    store = PlanRevisionStore(tmp_path / "store")
    revision = _revision(
        plan_root_cid=admission.plan_root_cid,
        roots=_roots(
            dirty_worktree_root=tree_id,
            task_source_id="task-source:both:fixture",
        ),
        task_population=_population(
            PopulationKind.RETAINED, *sorted(admission.task_cids)
        ),
        added_population=_population(
            PopulationKind.ADDED, *sorted(admission.task_cids)
        ),
    )
    receipt = store.apply(
        PlanRevisionApplyRequest(
            revision=revision,
            observed_roots=revision.roots,
            idempotency_key="idem:create-dual",
            expected_effects=("materialize-revision-1",),
            admission=admission,
            goal_graph=graph,
            aliases=aliases,
            markdown_source=markdown,
            duckdb_source=duck,
            repository_tree_id=tree_id,
        )
    )
    assert receipt.committed
    assert receipt.state is PlanRevisionApplyState.COMMITTED
    assert receipt.markdown_projection_cid
    assert receipt.duckdb_projection_cid
    assert markdown.plan_revision_projection_cid() == receipt.markdown_projection_cid
    assert duck.plan_revision_projection_cid() == receipt.duckdb_projection_cid
    active = store.get_active()
    assert active is not None
    assert active.plan_root_cid == revision.plan_root_cid
    assert active.revision_cid == revision.revision_cid
    loaded = store.load_revision(active.revision_cid)
    assert loaded.revision_cid == revision.revision_cid
    assert store.list_revision_cids() == (revision.revision_cid,)


def test_idempotent_replay_from_continuation_not_process_dict(
    tmp_path: Path,
) -> None:
    graph, admission, aliases, tree_id = _canonical_fixture()
    markdown = MarkdownTaskSource(
        tmp_path / "tasks.md",
        root=tmp_path,
        task_prefix="FIX",
        board_namespace="fixture",
    )
    duck = DuckDBTaskSource(tmp_path / "tasks.duckdb")
    store = PlanRevisionStore(tmp_path / "store")
    revision = _revision(
        plan_root_cid=admission.plan_root_cid,
        roots=_roots(dirty_worktree_root=tree_id),
        task_population=_population(
            PopulationKind.RETAINED, *sorted(admission.task_cids)
        ),
        added_population=_population(
            PopulationKind.ADDED, *sorted(admission.task_cids)
        ),
    )
    request = PlanRevisionApplyRequest(
        revision=revision,
        observed_roots=revision.roots,
        idempotency_key="idem:replay",
        expected_effects=("materialize-revision-1",),
        admission=admission,
        goal_graph=graph,
        aliases=aliases,
        markdown_source=markdown,
        duckdb_source=duck,
        repository_tree_id=tree_id,
    )
    first = store.apply(request)
    # New store instance must reload continuation from CAS/store files.
    reopened = PlanRevisionStore(tmp_path / "store", recover=False)
    second = reopened.apply(request)
    assert second.state is PlanRevisionApplyState.REPLAYED
    assert second.receipt_cid == first.receipt_cid
    assert second.resumed is True
    assert markdown.plan_revision_projection_cid() == first.markdown_projection_cid
    assert duck.plan_revision_projection_cid() == first.duckdb_projection_cid


def test_steer_appends_history_never_edits_claimed_specs(tmp_path: Path) -> None:
    store = PlanRevisionStore(tmp_path / "store")
    base = _revision(
        claimed_population=_population(PopulationKind.CLAIMED, _cid("task-running")),
        task_population=_population(
            PopulationKind.RETAINED, _cid("task-1"), _cid("task-running")
        ),
    )
    create_receipt = store.apply(
        PlanRevisionApplyRequest(
            revision=base,
            observed_roots=base.roots,
            idempotency_key="idem:base",
            expected_effects=("create",),
        )
    )
    assert create_receipt.committed

    deferred_key = "delta:deferred-successor"
    delta = _delta(
        base_plan_root=base.plan_root_cid,
        items=(
            _delta_item(
                item_key="delta:safe-successor",
                operation=PlanDeltaOperation.ADD_TASK,
                target_cid=_cid("task-running"),
                expected_target_lifecycle=LifecycleState.CLAIMED,
                after_record_cid=_cid("successor-task"),
                effect_class=DeltaEffectClass.DEFERRED,
            ),
            _delta_item(
                item_key=deferred_key,
                operation=PlanDeltaOperation.ADD_TASK,
                target_cid=_cid("task-running"),
                expected_target_lifecycle=LifecycleState.RUNNING,
                after_record_cid=_cid("later-task"),
                effect_class=DeltaEffectClass.DEFERRED,
            ),
        ),
        deferred_item_keys=(deferred_key,),
        expected_effects=("append-task", "defer-successor"),
    )
    child = _revision(
        plan_root_cid=_cid("plan-root-2"),
        semantic_revision=2,
        parent_plan_root=base.plan_root_cid,
        origin=PlanOrigin.STEER,
        roots=base.roots,
        request_cid=_cid("steer-request"),
        delta_cid=delta.delta_cid,
        claimed_population=base.claimed_population,
        task_population=_population(
            PopulationKind.RETAINED,
            _cid("task-1"),
            _cid("task-running"),
            _cid("successor-task"),
        ),
        added_population=_population(PopulationKind.ADDED, _cid("successor-task")),
        retained_population=_population(
            PopulationKind.RETAINED, _cid("task-1"), _cid("task-running")
        ),
        deferred_population=_population(PopulationKind.DEFERRED, deferred_key),
    )
    steer_receipt = store.apply(
        PlanRevisionApplyRequest(
            revision=child,
            observed_roots=child.roots,
            idempotency_key="idem:steer",
            expected_effects=delta.expected_effects,
            delta=delta,
            expected_active_plan_root=base.plan_root_cid,
            expected_active_revision_cid=base.revision_cid,
            base_event_cursor=create_receipt.event_cursor,
        )
    )
    assert steer_receipt.committed
    assert deferred_key in steer_receipt.deferred_item_keys
    assert store.list_revision_cids() == (base.revision_cid, child.revision_cid)
    supersessions = store.list_supersessions()
    assert any(row.get("kind") == "plan_revision" for row in supersessions)
    # History of the claimed task remains in the claimed population digest.
    loaded = store.load_revision(child.revision_cid)
    assert _cid("task-running") in loaded.claimed_population.member_cids

    with pytest.raises((PlanRevisionLifecycleError, PlanRevisionStoreConflictError)):
        bad_delta = _delta(
            base_plan_root=child.plan_root_cid,
            items=(
                _delta_item(
                    item_key="delta:mutate-claimed",
                    operation=PlanDeltaOperation.SUPERSEDE_UNSTARTED_TASK,
                    target_cid=_cid("task-running"),
                    expected_target_lifecycle=LifecycleState.CLAIMED,
                    before_digest=_cid("before"),
                    after_record_cid=_cid("after"),
                ),
            ),
        )
        store.apply(
            PlanRevisionApplyRequest(
                revision=_revision(
                    plan_root_cid=_cid("plan-root-bad"),
                    semantic_revision=3,
                    parent_plan_root=child.plan_root_cid,
                    origin=PlanOrigin.STEER,
                    roots=child.roots,
                    delta_cid=bad_delta.delta_cid,
                    claimed_population=child.claimed_population,
                    task_population=child.task_population,
                ),
                observed_roots=child.roots,
                idempotency_key="idem:bad-steer",
                delta=bad_delta,
                expected_active_plan_root=child.plan_root_cid,
            )
        )


def test_deferred_successors_activate_only_when_preconditions_true(
    tmp_path: Path,
) -> None:
    store = PlanRevisionStore(tmp_path / "store")
    deferred_key = "delta:wait-for-claim"
    revision = _revision(
        deferred_population=_population(PopulationKind.DEFERRED, deferred_key),
    )
    store.apply(
        PlanRevisionApplyRequest(
            revision=revision,
            observed_roots=revision.roots,
            idempotency_key="idem:deferred",
            expected_effects=("create-with-deferred",),
        )
    )
    active = store.get_active()
    assert active is not None
    assert deferred_key in active.deferred_item_keys

    none = store.activate_deferred(
        (deferred_key,),
        preconditions_satisfied={deferred_key: False},
    )
    assert none == ()
    assert deferred_key in store.get_active().deferred_item_keys  # type: ignore[union-attr]

    activated = store.activate_deferred(
        (deferred_key,),
        preconditions_satisfied={deferred_key: True},
    )
    assert activated == (deferred_key,)
    assert deferred_key not in store.get_active().deferred_item_keys  # type: ignore[union-attr]
    events = store.list_events()
    assert any(
        row.get("event_type") == "deferred_activated" for row in events
    )


def test_crash_after_prepare_restores_prior_active_projection(
    tmp_path: Path,
) -> None:
    store = PlanRevisionStore(tmp_path / "store")
    base = _revision()
    first = store.apply(
        PlanRevisionApplyRequest(
            revision=base,
            observed_roots=base.roots,
            idempotency_key="idem:first",
            expected_effects=("create",),
        )
    )
    assert first.committed
    prior_active = store.get_active()
    assert prior_active is not None

    child = _revision(
        plan_root_cid=_cid("plan-root-2"),
        semantic_revision=2,
        parent_plan_root=base.plan_root_cid,
        origin=PlanOrigin.STEER,
        delta_cid=_delta().delta_cid,
        roots=base.roots,
        task_population=base.task_population,
        added_population=_population(PopulationKind.ADDED, _cid("task-2")),
    )

    def _fault(point: str) -> None:
        if point == "after_prepare":
            raise RuntimeError("injected-crash-after-prepare")

    with pytest.raises(PlanRevisionStoreConflictError, match="injected-crash"):
        store.apply(
            PlanRevisionApplyRequest(
                revision=child,
                observed_roots=child.roots,
                idempotency_key="idem:crash",
                expected_effects=("steer",),
                delta=_delta(base_plan_root=base.plan_root_cid),
                expected_active_plan_root=base.plan_root_cid,
                fault_injector=_fault,
            )
        )
    # Prior active projection restored; failed child revision history retained
    # in CAS/index when append happened, but active pointer is prior.
    active = store.get_active()
    assert active is not None
    assert active.plan_root_cid == base.plan_root_cid


def test_recover_reloads_continuation_from_store_not_process_memory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    store = PlanRevisionStore(root, recover=False)
    revision = _revision()
    store.journal_intent(
        PlanRevisionApplyRequest(
            revision=revision,
            observed_roots=revision.roots,
            idempotency_key="idem:recover",
            expected_effects=("create",),
        )
    )
    intent_cid = store.load_continuation("idem:recover")["intent_cid"]  # type: ignore[index]
    store.put_continuation(
        "idem:recover",
        {
            "phase": PlanRevisionApplyState.PREPARED.value,
            "intent_cid": intent_cid,
            "revision_cid": revision.revision_cid,
            "plan_root_cid": revision.plan_root_cid,
        },
    )
    # New process: empty in-memory state, recover from durable files only.
    reopened = PlanRevisionStore(root, recover=False)
    recovered = reopened.recover()
    assert recovered
    continuation = reopened.load_continuation("idem:recover")
    assert continuation is not None
    assert continuation["phase"] == PlanRevisionApplyState.RESTORED.value


def test_split_brain_quarantine_blocks_further_apply(tmp_path: Path) -> None:
    store = PlanRevisionStore(tmp_path / "store")
    base = _revision()
    store.apply(
        PlanRevisionApplyRequest(
            revision=base,
            observed_roots=base.roots,
            idempotency_key="idem:base-q",
            expected_effects=("create",),
        )
    )

    class _DisagreeMarkdown:
        path = tmp_path / "ghost.md"

        def apply_plan_revision(self, **_kwargs):
            return {"projection_cid": "md-cid-a"}

        def plan_revision_projection_cid(self):
            return "md-cid-a"

        def compare_plan_revision_parity(self, _other):
            return {"valid": False, "mismatches": ("projection",)}

    class _DisagreeDuck:
        database_path = tmp_path / "ghost.duckdb"

        def apply_plan_revision(self, **_kwargs):
            return {"projection_cid": "db-cid-b"}

        def plan_revision_projection_cid(self):
            return "db-cid-b"

    child = _revision(
        plan_root_cid=_cid("plan-root-split"),
        semantic_revision=2,
        parent_plan_root=base.plan_root_cid,
        origin=PlanOrigin.STEER,
        delta_cid=_delta().delta_cid,
        roots=base.roots,
        task_population=base.task_population,
    )
    with pytest.raises(PlanRevisionStoreQuarantinedError):
        store.apply(
            PlanRevisionApplyRequest(
                revision=child,
                observed_roots=child.roots,
                idempotency_key="idem:split",
                expected_effects=("steer",),
                delta=_delta(base_plan_root=base.plan_root_cid),
                markdown_source=_DisagreeMarkdown(),
                duckdb_source=_DisagreeDuck(),
                expected_active_plan_root=base.plan_root_cid,
            )
        )
    assert store.is_quarantined()
    with pytest.raises(PlanRevisionStoreQuarantinedError):
        store.apply(
            PlanRevisionApplyRequest(
                revision=child,
                observed_roots=child.roots,
                idempotency_key="idem:after-quarantine",
                expected_effects=("steer",),
                delta=_delta(base_plan_root=base.plan_root_cid),
            )
        )
    events = store.list_events()
    assert any(
        row.get("event_type") == "split_brain_quarantined" for row in events
    )


def test_markdown_pending_reloads_from_durable_continuation(
    tmp_path: Path,
) -> None:
    graph, admission, aliases, tree_id = _canonical_fixture()
    markdown = MarkdownTaskSource(
        tmp_path / "tasks.md",
        root=tmp_path,
        task_prefix="FIX",
        board_namespace="fixture",
    )
    store = PlanRevisionStore(tmp_path / "store")
    projection = markdown.project(admission, aliases=aliases)
    preview = projection.preview("")
    markdown._persist_pending_preview(  # noqa: SLF001 - intentional boundary test
        projection,
        preview,
        store_continuation=store,
        idempotency_key="idem:md-pending",
    )
    # Drop process dictionary; reload from durable store/file.
    markdown._pending.clear()  # noqa: SLF001
    reloaded = markdown._load_pending_preview(  # noqa: SLF001
        projection,
        store_continuation=store,
        idempotency_key="idem:md-pending",
    )
    assert reloaded is not None
    assert reloaded.preview_id == preview.preview_id
    assert reloaded.candidate_board_revision == preview.candidate_board_revision


def test_append_only_records_and_events_survive_reopen(tmp_path: Path) -> None:
    store = PlanRevisionStore(tmp_path / "store")
    revision = _revision()
    store.apply(
        PlanRevisionApplyRequest(
            revision=revision,
            observed_roots=revision.roots,
            idempotency_key="idem:history",
            expected_effects=("create",),
            records={"goal:v1": {"goal_cid": _cid("goal-1"), "title": "Goal"}},
        )
    )
    events_before = store.list_events()
    assert any(row.get("event_type") == "revision_appended" for row in events_before)
    reopened = PlanRevisionStore(tmp_path / "store")
    assert reopened.list_revision_cids() == (revision.revision_cid,)
    assert len(reopened.list_events()) >= len(events_before)
    cas_goal = None
    for path in (tmp_path / "store" / "cas").iterdir():
        payload = path.read_text(encoding="utf-8")
        if "goal:v1" in payload:
            cas_goal = payload
            break
    assert cas_goal is not None
