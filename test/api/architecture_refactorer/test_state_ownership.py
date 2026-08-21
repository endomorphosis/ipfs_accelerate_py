"""Hermetic PCAR-014 state-ownership model tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.contracts import (
    ArchitectureContractError,
    Confidence,
    SourceFactIdentity,
    SourceSpan,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.state_ownership import (
    CACHE_IS_NOT_AUTHORITY,
    CAN_AUTHORIZE_CHANGES,
    CAN_CREATE_INDEFINITE_DUAL_OWNERSHIP,
    CAN_GRANT_AUTHORITY,
    CLOSED_MIGRATION_PHASES,
    CLOSED_SEMANTIC_FACTS,
    CLOSED_STATE_CONFLICTS,
    CLOSED_STATE_DISPOSITIONS,
    CLOSED_STORE_KINDS,
    CURRENT_TREE_STORE_BINDINGS,
    DERIVED_SEMANTIC_FACTS,
    DUCKDB_CONTROL_STORE_ID,
    DUCKLAKE_IS_NOT_AUTHORITY,
    DUAL_WRITE_BOUND,
    EFFECT_CLASS,
    EXTRACTOR_IDENTITY,
    MARKDOWN_IS_NOT_AUTHORITY,
    MUTABLE_SEMANTIC_FACTS,
    PROJECTION_IS_NOT_AUTHORITY,
    REQUIRED_STORE_KINDS,
    STATE_OWNERSHIP_EVIDENCE,
    STATE_OWNERSHIP_SCHEMA,
    STATE_OWNERSHIP_VERSION,
    TASK_ID,
    UNKNOWN_OWNER_ACCEPTANCE_PROHIBITED,
    BOUNDED_MIGRATION_PHASES,
    MigrationPhase,
    SemanticFactKind,
    StateConflictKind,
    StateDisposition,
    StateItem,
    StateMigrationPhase,
    StateMigrationPlan,
    StateNomination,
    StateOwnershipAuthorityError,
    StateOwnershipError,
    StateOwnershipModel,
    StoreKind,
    accept_unknown_owner,
    build_state_ownership_model,
    classify_current_tree_state_ownership,
    classify_state_ownership,
    closed_migration_phases,
    detect_state_conflicts,
    physical_store_id,
    plan_state_migration,
    refuse_authority_grant,
    refuse_indefinite_dual_authority,
    refuse_state_authorization,
    semantic_fact_for,
    state_items_from_inventory,
)
from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

_TREE = "pcar-014-fixture-tree"
_FRESHNESS = "pcar-014-fixture"
_ROOT = Path(__file__).resolve().parents[3]
_INVENTORY = (
    _ROOT
    / "docs/architecture/architecture_refactorer_inventory"
    / "current_state_stores.json"
)
_BASELINE = (
    _ROOT
    / "docs/architecture/architecture_refactorer_inventory"
    / "state_store_baseline.json"
)


def _span(path: str, start: int, end: int | None = None) -> SourceSpan:
    return SourceSpan(path, start, start if end is None else end)


def _fact(
    path: str,
    start: int,
    *,
    confidence: Confidence = Confidence.EXACT,
    end: int | None = None,
) -> SourceFactIdentity:
    return SourceFactIdentity(
        extractor_identity="pcar-014-fixture",
        span=_span(path, start, end),
        confidence=confidence,
        freshness=_FRESHNESS,
        repository_tree=_TREE,
    )


def _nomination(
    kind: StoreKind,
    path: str,
    disposition: StateDisposition,
    *,
    start: int = 1,
    store_locator: str = "",
    nominated_owner: str = "",
    uncertainty: str = "",
    tables: tuple[str, ...] = (),
    rebuildable: bool | None = None,
    item_id: str = "",
) -> StateNomination:
    return StateNomination(
        kind=kind,
        path=path,
        disposition=disposition,
        provenance=_fact(path if not path.startswith("{") else "ipfs_accelerate_py/x.py", start),
        item_id=item_id,
        store_locator=store_locator or path,
        nominated_owner=nominated_owner,
        uncertainty=uncertainty,
        tables=tables,
        rebuildable=rebuildable,
    )


def _current() -> StateOwnershipModel:
    return classify_current_tree_state_ownership()


def test_closed_store_class_and_kind_vocabulary() -> None:
    assert STATE_OWNERSHIP_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/state-ownership-model@1"
    )
    assert STATE_OWNERSHIP_VERSION == 1
    assert STATE_OWNERSHIP_EVIDENCE == "pcar/state-ownership-model@1"
    assert EXTRACTOR_IDENTITY == "pcar-014-state-ownership-model"
    assert TASK_ID == "PCAR-014"
    assert EFFECT_CLASS == "read_only_analysis_and_immutable_migration_proposals"
    assert CAN_AUTHORIZE_CHANGES is False
    assert CAN_GRANT_AUTHORITY is False
    assert CAN_CREATE_INDEFINITE_DUAL_OWNERSHIP is False
    assert MARKDOWN_IS_NOT_AUTHORITY is True
    assert DUCKLAKE_IS_NOT_AUTHORITY is True
    assert PROJECTION_IS_NOT_AUTHORITY is True
    assert CACHE_IS_NOT_AUTHORITY is True
    assert UNKNOWN_OWNER_ACCEPTANCE_PROHIBITED is True
    assert tuple(item.value for item in StateDisposition) == (
        "authoritative",
        "materialized_projection",
        "cache",
        "historical_event",
        "fixture",
        "legacy",
        "unknown",
    )
    assert CLOSED_STATE_DISPOSITIONS == {item.value for item in StateDisposition}
    assert tuple(item.value for item in REQUIRED_STORE_KINDS) == (
        "DuckDB tables",
        "JSON files",
        "Markdown task boards",
        "in-memory registries",
        "event logs",
        "cache namespaces",
        "worktree metadata",
        "lease records",
        "provider state",
        "goal state",
        "task state",
        "completion state",
        "receipt state",
    )
    assert CLOSED_STORE_KINDS == {item.value for item in StoreKind}
    assert CLOSED_MIGRATION_PHASES == {item.value for item in MigrationPhase}
    assert tuple(item.value for item in BOUNDED_MIGRATION_PHASES) == (
        "snapshot",
        "dual_read_shadow",
        "dual_write",
        "cutover",
        "validation",
        "read_only_legacy",
        "retirement",
    )
    assert "unknown_conflict" in CLOSED_STATE_CONFLICTS
    assert "multiple_authoritative_stores" in CLOSED_STATE_CONFLICTS
    assert CLOSED_SEMANTIC_FACTS == {item.value for item in SemanticFactKind}
    with pytest.raises(ValueError):
        StateDisposition("dashboard")
    with pytest.raises(ValueError):
        StoreKind("sqlite files")
    with pytest.raises(ValueError):
        MigrationPhase("indefinite_dual_write")
    with pytest.raises(ValueError):
        StateConflictKind("ignore")


def test_all_store_classes_are_classified_on_the_current_tree() -> None:
    model = _current()
    assert model.covers_required_kinds is True
    assert tuple(item.kind for item in CURRENT_TREE_STORE_BINDINGS) == REQUIRED_STORE_KINDS
    kinds = [item.kind for item in model.items]
    assert kinds == list(REQUIRED_STORE_KINDS)
    dispositions = {item.disposition for item in model.items}
    assert StateDisposition.AUTHORITATIVE in dispositions
    assert StateDisposition.MATERIALIZED_PROJECTION in dispositions
    assert StateDisposition.CACHE in dispositions
    assert StateDisposition.HISTORICAL_EVENT in dispositions
    assert StateDisposition.UNKNOWN in dispositions
    fixture_item = _nomination(
        StoreKind.JSON_FILES,
        "test/api/architecture_refactorer/fixtures/goals.json",
        StateDisposition.FIXTURE,
        item_id="fixture-goals",
        nominated_owner=DUCKDB_CONTROL_STORE_ID,
    ).to_item()
    legacy_item = _nomination(
        StoreKind.EVENT_LOGS,
        "state/events.jsonl",
        StateDisposition.LEGACY,
        item_id="legacy-jsonl-events",
        store_locator="state/events.jsonl",
    ).to_item()
    assert fixture_item.disposition is StateDisposition.FIXTURE
    assert legacy_item.disposition is StateDisposition.LEGACY
    assert {item.value for item in StateDisposition} == CLOSED_STATE_DISPOSITIONS
    for item in model.items:
        assert item.disposition.value in CLOSED_STATE_DISPOSITIONS
        assert item.kind in REQUIRED_STORE_KINDS
        assert item.semantic_fact is semantic_fact_for(item.kind)


def test_current_tree_source_bindings_match_inventory_and_source() -> None:
    inventory = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    baseline = json.loads(_BASELINE.read_text(encoding="utf-8"))
    assert set(baseline["closed_dispositions"]) == CLOSED_STATE_DISPOSITIONS
    assert [item.value for item in REQUIRED_STORE_KINDS] == baseline["required_kinds"]
    assert inventory["authority"] is False
    nominated_kinds = [item["kind"] for item in inventory["stores"]]
    assert nominated_kinds == [item.value for item in REQUIRED_STORE_KINDS]
    for binding in CURRENT_TREE_STORE_BINDINGS:
        path = _ROOT / binding.path
        assert path.is_file(), binding.path
        lines = path.read_text(encoding="utf-8").splitlines()
        assert 1 <= binding.start_line <= binding.end_line <= len(lines)
    nominations = state_items_from_inventory(
        inventory, repository_tree=_TREE, freshness=_FRESHNESS
    )
    assert [item.kind for item in nominations] == list(REQUIRED_STORE_KINDS)
    receipt = next(item for item in nominations if item.kind is StoreKind.RECEIPT_STATE)
    assert receipt.store_locator.endswith("#completion_receipts")
    assert physical_store_id(receipt.store_locator) == DUCKDB_CONTROL_STORE_ID


def test_one_authoritative_store_per_mutable_fact() -> None:
    model = _current()
    assert model.one_authoritative_store is True
    resolved = (
        SemanticFactKind.GOAL_STATE,
        SemanticFactKind.TASK_STATE,
        SemanticFactKind.LEASE_RECORDS,
        SemanticFactKind.WORKTREE_METADATA,
        SemanticFactKind.COMPLETION_STATE,
        SemanticFactKind.RECEIPT_STATE,
    )
    for fact in resolved:
        assert model.conflicts_for(fact) == ()
        assert model.authoritative_store_for(fact) == DUCKDB_CONTROL_STORE_ID
        owners = model.authoritative_items(fact)
        assert owners
        assert {item.physical_store_id for item in owners} == {DUCKDB_CONTROL_STORE_ID}
    duckdb = model.items_of(StoreKind.DUCKDB_TABLES)[0]
    assert duckdb.disposition is StateDisposition.AUTHORITATIVE
    assert duckdb.physical_store_id == DUCKDB_CONTROL_STORE_ID
    assert "goals" in duckdb.tables
    assert model.authoritative_store_for(SemanticFactKind.GOAL_STATE) == (
        model.authoritative_store_for(SemanticFactKind.TASK_STATE)
    )


def test_two_physical_stores_for_one_fact_is_a_hard_conflict() -> None:
    model = _current()
    extra = _nomination(
        StoreKind.GOAL_STATE,
        "docs/architecture/agent_supervisor_architecture_refactorer.todo.md",
        StateDisposition.AUTHORITATIVE,
        item_id="markdown-goals",
        store_locator="docs/architecture/agent_supervisor_architecture_refactorer.todo.md",
    ).to_item()
    conflicted = classify_state_ownership(
        (*model.items, extra),
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    conflicts = conflicted.conflicts_for(SemanticFactKind.GOAL_STATE)
    assert any(
        item.kind is StateConflictKind.MULTIPLE_AUTHORITATIVE_STORES for item in conflicts
    )
    assert any(
        item.kind is StateConflictKind.MARKDOWN_CLAIMED_AUTHORITY for item in conflicts
    )
    with pytest.raises(StateOwnershipError, match="no unique authoritative store"):
        conflicted.authoritative_store_for(SemanticFactKind.GOAL_STATE)
    same_store = _nomination(
        StoreKind.GOAL_STATE,
        "ipfs_accelerate_py/agent_supervisor/task_sources/sql/0001_control_plane.sql",
        StateDisposition.AUTHORITATIVE,
        start=407,
        item_id="goals-alias",
        store_locator="{state_root}/control.duckdb#goals",
        nominated_owner=DUCKDB_CONTROL_STORE_ID,
        tables=("goals",),
    ).to_item()
    aliased = classify_state_ownership(
        (*model.items, same_store),
        repository_tree=model.repository_tree,
        freshness=model.freshness,
        migrations=model.migrations,
    )
    assert aliased.conflicts_for(SemanticFactKind.GOAL_STATE) == ()
    assert aliased.authoritative_store_for(SemanticFactKind.GOAL_STATE) == (
        DUCKDB_CONTROL_STORE_ID
    )


def test_unknown_conflict_is_never_an_accepted_owner() -> None:
    model = _current()
    registry = model.items_of(StoreKind.IN_MEMORY_REGISTRIES)[0]
    provider = model.items_of(StoreKind.PROVIDER_STATE)[0]
    assert model.fails_closed is True
    assert registry.disposition is StateDisposition.UNKNOWN
    assert provider.disposition is StateDisposition.UNKNOWN
    assert registry.uncertainty
    assert provider.uncertainty
    registry_conflicts = model.conflicts_for(SemanticFactKind.REGISTRY_MEMBERSHIP)
    provider_conflicts = model.conflicts_for(SemanticFactKind.PROVIDER_STATE)
    assert any(item.kind is StateConflictKind.UNKNOWN_CONFLICT for item in registry_conflicts)
    assert any(item.kind is StateConflictKind.UNKNOWN_OWNER for item in registry_conflicts)
    assert any(item.kind is StateConflictKind.UNKNOWN_CONFLICT for item in provider_conflicts)
    assert any(item.kind is StateConflictKind.UNKNOWN_OWNER for item in provider_conflicts)
    with pytest.raises(StateOwnershipError, match="no unique authoritative store"):
        model.authoritative_store_for(SemanticFactKind.PROVIDER_STATE)
    with pytest.raises(StateOwnershipError, match="unknown-owner acceptance"):
        accept_unknown_owner(registry)
    unknown_only = classify_state_ownership(
        (
            _nomination(
                StoreKind.TASK_STATE,
                "ipfs_accelerate_py/agent_supervisor/todo_daemon/registry.py",
                StateDisposition.UNKNOWN,
                start=84,
                uncertainty="dynamic_registry_membership",
            ),
        ),
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    record = unknown_only.items_for_fact(SemanticFactKind.TASK_STATE)[0]
    assert record.is_unknown is True
    assert unknown_only.fails_closed is True
    assert any(
        item.kind is StateConflictKind.UNKNOWN_CONFLICT
        for item in unknown_only.conflicts_for(SemanticFactKind.TASK_STATE)
    )


def test_unknown_beside_an_authoritative_store_is_a_conflicting_disposition() -> None:
    model = _current()
    unknown_goals = _nomination(
        StoreKind.GOAL_STATE,
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/registry.py",
        StateDisposition.UNKNOWN,
        start=84,
        item_id="unknown-goals",
        uncertainty="in-memory-goals",
    ).to_item()
    conflicted = classify_state_ownership(
        (*model.items, unknown_goals),
        repository_tree=model.repository_tree,
        freshness=model.freshness,
    )
    kinds = {item.kind for item in conflicted.conflicts_for(SemanticFactKind.GOAL_STATE)}
    assert StateConflictKind.UNKNOWN_CONFLICT in kinds
    assert StateConflictKind.CONFLICTING_DISPOSITION in kinds
    with pytest.raises(StateOwnershipError, match="no unique authoritative store"):
        conflicted.authoritative_store_for(SemanticFactKind.GOAL_STATE)


def test_bounded_migration_closes_dual_read_and_write() -> None:
    phases = closed_migration_phases()
    assert tuple(item.phase for item in phases) == BOUNDED_MIGRATION_PHASES
    dual = [item for item in phases if item.dual_authority]
    assert [item.phase for item in dual] == [
        MigrationPhase.DUAL_READ_SHADOW,
        MigrationPhase.DUAL_WRITE,
    ]
    assert all(item.dual_authority_bounded for item in dual)
    assert all(item.grants_authority is False for item in phases)
    assert phases[-1].phase is MigrationPhase.RETIREMENT
    assert phases[-1].terminal is True
    assert phases[-1].dual_authority is False
    plan = plan_state_migration(
        SemanticFactKind.PROVIDER_STATE,
        "path:ipfs_accelerate_py/agent_supervisor/control/provider_attempt_store.py",
        DUCKDB_CONTROL_STORE_ID,
    )
    assert plan.ends_dual_authority is True
    assert plan.can_grant_authority is False
    assert plan.can_create_indefinite_dual_ownership is False
    assert plan.dual_write_bound == DUAL_WRITE_BOUND
    assert plan.terminal_phase.phase is MigrationPhase.RETIREMENT
    model = _current()
    provider_plan = model.migration_for(SemanticFactKind.PROVIDER_STATE)
    registry_plan = model.migration_for(SemanticFactKind.REGISTRY_MEMBERSHIP)
    assert provider_plan is not None
    assert registry_plan is not None
    assert provider_plan.target_store_id == DUCKDB_CONTROL_STORE_ID
    assert registry_plan.target_store_id == DUCKDB_CONTROL_STORE_ID
    assert provider_plan.phases == phases
    with pytest.raises(StateOwnershipError, match="closed snapshot-to-retirement"):
        StateMigrationPlan(
            plan_id="open-dual-write",
            semantic_fact=SemanticFactKind.PROVIDER_STATE,
            source_store_id="path:provider",
            target_store_id=DUCKDB_CONTROL_STORE_ID,
            phases=phases[:3],
        )
    with pytest.raises(StateOwnershipError, match="cannot grant authority"):
        StateMigrationPlan(
            plan_id="grant",
            semantic_fact=SemanticFactKind.PROVIDER_STATE,
            source_store_id="path:provider",
            target_store_id=DUCKDB_CONTROL_STORE_ID,
            phases=phases,
            can_grant_authority=True,
        )
    with pytest.raises(StateOwnershipError, match="indefinite dual"):
        StateMigrationPlan(
            plan_id="indefinite",
            semantic_fact=SemanticFactKind.PROVIDER_STATE,
            source_store_id="path:provider",
            target_store_id=DUCKDB_CONTROL_STORE_ID,
            phases=phases,
            can_create_indefinite_dual_ownership=True,
        )
    with pytest.raises(StateOwnershipError, match="must end dual authority"):
        StateMigrationPlan(
            plan_id="no-end",
            semantic_fact=SemanticFactKind.PROVIDER_STATE,
            source_store_id="path:provider",
            target_store_id=DUCKDB_CONTROL_STORE_ID,
            phases=phases,
            ends_dual_authority=False,
        )
    with pytest.raises(StateOwnershipError, match="Markdown"):
        plan_state_migration(
            SemanticFactKind.TASK_STATE,
            DUCKDB_CONTROL_STORE_ID,
            "markdown:docs/architecture/agent_supervisor_architecture_refactorer.todo.md",
        )
    with pytest.raises(StateOwnershipError, match="DuckLake"):
        plan_state_migration(
            SemanticFactKind.TASK_STATE,
            DUCKDB_CONTROL_STORE_ID,
            "ducklake",
        )
    with pytest.raises(StateOwnershipError, match="formally bounded"):
        StateMigrationPhase(
            phase=MigrationPhase.DUAL_WRITE,
            dual_authority=True,
            dual_authority_bounded=False,
            writes_legacy=True,
            writes_target=True,
        )


def test_rebuildable_projections_and_caches_are_not_authority() -> None:
    model = _current()
    projections = model.rebuildable_projections()
    kinds = {item.kind for item in projections}
    assert StoreKind.JSON_FILES in kinds
    assert StoreKind.MARKDOWN_TASK_BOARDS in kinds
    assert StoreKind.CACHE_NAMESPACES in kinds
    for item in projections:
        assert item.rebuildable is True
        assert item.disposition in {
            StateDisposition.MATERIALIZED_PROJECTION,
            StateDisposition.CACHE,
        }
        assert item.is_authoritative is False
        assert item.rebuild_from_store_id == DUCKDB_CONTROL_STORE_ID
        assert item.semantic_fact in DERIVED_SEMANTIC_FACTS
    markdown = model.items_of(StoreKind.MARKDOWN_TASK_BOARDS)[0]
    assert markdown.disposition is StateDisposition.MATERIALIZED_PROJECTION
    cache = model.items_of(StoreKind.CACHE_NAMESPACES)[0]
    assert cache.disposition is StateDisposition.CACHE
    events = model.items_of(StoreKind.EVENT_LOGS)[0]
    assert events.disposition is StateDisposition.HISTORICAL_EVENT
    assert events.is_authoritative is False
    markdown_owner = _nomination(
        StoreKind.MARKDOWN_TASK_BOARDS,
        "docs/architecture/agent_supervisor_architecture_refactorer.todo.md",
        StateDisposition.AUTHORITATIVE,
        start=108,
    ).to_item()
    cache_owner = _nomination(
        StoreKind.CACHE_NAMESPACES,
        "ipfs_accelerate_py/agent_supervisor/analysis/analysis_cache.py",
        StateDisposition.AUTHORITATIVE,
        start=846,
        item_id="cache-as-owner",
    ).to_item()
    stale_projection = _nomination(
        StoreKind.JSON_FILES,
        "docs/architecture/architecture_refactorer_inventory/state_store_baseline.json",
        StateDisposition.MATERIALIZED_PROJECTION,
        nominated_owner=DUCKDB_CONTROL_STORE_ID,
        rebuildable=False,
        item_id="stale-json",
    ).to_item()
    ducklake = _nomination(
        StoreKind.TASK_STATE,
        "ipfs_accelerate_py/agent_supervisor/task_sources/sql/0001_control_plane.sql",
        StateDisposition.AUTHORITATIVE,
        start=470,
        item_id="ducklake-tasks",
        store_locator="ducklake://history/tasks",
    ).to_item()
    findings = detect_state_conflicts(
        (markdown_owner, cache_owner, stale_projection, ducklake)
    )
    kinds = {item.kind for item in findings}
    assert StateConflictKind.MARKDOWN_CLAIMED_AUTHORITY in kinds
    assert StateConflictKind.CACHE_CLAIMED_AUTHORITY in kinds
    assert StateConflictKind.NON_REBUILDABLE_PROJECTION in kinds
    assert StateConflictKind.DUCKLAKE_CLAIMED_AUTHORITY in kinds
    fixture_only = classify_state_ownership(
        (
            _nomination(
                StoreKind.TASK_STATE,
                "test/api/architecture_refactorer/fixtures/tasks.json",
                StateDisposition.FIXTURE,
            ),
        ),
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    assert any(
        item.kind is StateConflictKind.FIXTURE_CLAIMED_AUTHORITY
        for item in fixture_only.conflicts_for(SemanticFactKind.TASK_STATE)
    )
    legacy_only = classify_state_ownership(
        (
            _nomination(
                StoreKind.LEASE_RECORDS,
                "state/leases.jsonl",
                StateDisposition.LEGACY,
            ),
        ),
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    assert any(
        item.kind is StateConflictKind.LEGACY_CLAIMED_AUTHORITY
        for item in legacy_only.conflicts_for(SemanticFactKind.LEASE_RECORDS)
    )


def test_model_round_trip_and_unknown_fields_fail_closed() -> None:
    model = classify_current_tree_state_ownership(
        repository_tree=_TREE, freshness=_FRESHNESS
    )
    payload = model.to_dict()
    assert payload["schema"] == STATE_OWNERSHIP_SCHEMA
    assert payload["version"] == STATE_OWNERSHIP_VERSION
    assert payload["can_authorize_changes"] is False
    assert payload["can_grant_authority"] is False
    identity = cid_for_dag_json({key: value for key, value in payload.items() if key != "content_identity"})
    assert payload["content_identity"] == identity
    validate_cid(payload["content_identity"], codecs=("dag-json",))
    restored = StateOwnershipModel.from_mapping(payload)
    assert restored == model
    assert StateOwnershipModel.from_json(model.to_json()) == model
    assert build_state_ownership_model(model.items, repository_tree=_TREE, freshness=_FRESHNESS).items == (
        classify_state_ownership(model.items, repository_tree=_TREE, freshness=_FRESHNESS).items
    )
    with pytest.raises(StateOwnershipError, match="unknown state-ownership field"):
        StateOwnershipModel.from_mapping({**payload, "extra": True})
    item_payload = model.items[0].to_dict()
    with pytest.raises(StateOwnershipError, match="unknown state-ownership field"):
        StateItem.from_mapping({**item_payload, "owner_hint": "x"})
    with pytest.raises(StateOwnershipError, match="missing state-ownership field"):
        StateItem.from_mapping({key: value for key, value in item_payload.items() if key != "kind"})
    with pytest.raises(StateOwnershipError, match="content identity"):
        StateItem(
            item_id="baguqeeraaf2trsznudx7wxyyocgpkaoqf5smketqgebile3aobqxbcvdddbq",
            kind=StoreKind.GOAL_STATE,
            path="ipfs_accelerate_py/agent_supervisor/task_sources/sql/0001_control_plane.sql",
            disposition=StateDisposition.AUTHORITATIVE,
            provenance=_fact(
                "ipfs_accelerate_py/agent_supervisor/task_sources/sql/0001_control_plane.sql",
                407,
            ),
        )


def test_model_cannot_authorize_grant_or_dual_own() -> None:
    model = _current()
    assert model.can_authorize_changes is False
    assert model.can_grant_authority is False
    assert model.can_create_indefinite_dual_ownership is False
    with pytest.raises(StateOwnershipAuthorityError, match="cannot authorize"):
        model.authorize_change("migrate")
    with pytest.raises(StateOwnershipAuthorityError, match="cannot grant"):
        model.grant_authority("duckdb")
    with pytest.raises(StateOwnershipAuthorityError, match="indefinite dual"):
        model.create_indefinite_dual_ownership()
    with pytest.raises(StateOwnershipAuthorityError, match="cannot authorize"):
        refuse_state_authorization("write")
    with pytest.raises(StateOwnershipAuthorityError, match="cannot grant"):
        refuse_authority_grant("grant")
    with pytest.raises(StateOwnershipAuthorityError, match="indefinite dual"):
        refuse_indefinite_dual_authority()
    with pytest.raises(StateOwnershipError, match="cannot authorize changes"):
        StateOwnershipModel(
            repository_tree=_TREE,
            freshness=_FRESHNESS,
            items=model.items,
            can_authorize_changes=True,
        )


def test_historical_events_are_not_mutable_authority() -> None:
    events = _nomination(
        StoreKind.EVENT_LOGS,
        "ipfs_accelerate_py/agent_supervisor/task_sources/sql/0001_control_plane.sql",
        StateDisposition.AUTHORITATIVE,
        start=827,
        store_locator="{state_root}/control.duckdb#domain_events",
        tables=("domain_events",),
    ).to_item()
    findings = detect_state_conflicts((events,))
    assert any(
        item.kind is StateConflictKind.EVENT_CLAIMED_MUTABLE_AUTHORITY for item in findings
    )
    model = _current()
    historical = model.items_of(StoreKind.EVENT_LOGS)[0]
    assert historical.disposition is StateDisposition.HISTORICAL_EVENT
    assert historical.semantic_fact not in MUTABLE_SEMANTIC_FACTS


def test_missing_required_kinds_fail_closed() -> None:
    model = classify_state_ownership(
        (
            _nomination(
                StoreKind.GOAL_STATE,
                "ipfs_accelerate_py/agent_supervisor/task_sources/sql/0001_control_plane.sql",
                StateDisposition.AUTHORITATIVE,
                start=407,
                store_locator="{state_root}/control.duckdb#goals",
                nominated_owner=DUCKDB_CONTROL_STORE_ID,
                tables=("goals",),
            ),
        ),
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    assert model.covers_required_kinds is False
    assert any(item.kind is StateConflictKind.MISSING_KIND for item in model.conflicts)
    assert model.fails_closed is True
    assert model.one_authoritative_store is True
    assert model.authoritative_store_for(SemanticFactKind.GOAL_STATE) == DUCKDB_CONTROL_STORE_ID


def test_contract_errors_are_state_ownership_errors() -> None:
    assert issubclass(StateOwnershipError, ArchitectureContractError)
    with pytest.raises(StateOwnershipError, match="unsupported ArchitectureIR"):
        StateItem(
            item_id="bad-disposition",
            kind=StoreKind.GOAL_STATE,
            path="ipfs_accelerate_py/agent_supervisor/task_sources/sql/0001_control_plane.sql",
            disposition="not-a-disposition",  # type: ignore[arg-type]
            provenance=_fact(
                "ipfs_accelerate_py/agent_supervisor/task_sources/sql/0001_control_plane.sql",
                407,
            ),
        )
