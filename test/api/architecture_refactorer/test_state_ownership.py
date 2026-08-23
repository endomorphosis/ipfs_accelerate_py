"""Hermetic PCAR-014 state-ownership model tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.architecture_ir import (
    ArchitectureEdge,
    ArchitectureIR,
    ArchitectureNode,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.contracts import (
    Confidence,
    EdgeKind,
    NodeKind,
    SourceFactIdentity,
    SourceSpan,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.state_ownership import (
    CACHE_CAN_BE_AUTHORITATIVE,
    CLOSED_CONFLICT_KINDS,
    CLOSED_DISPOSITIONS,
    CLOSED_MIGRATION_PHASES,
    CLOSED_STORE_KINDS,
    CONTENT_IDENTITY_IS_NOT_AUTHORITY,
    CURRENT_STATE_BINDINGS,
    DASHBOARD_CAN_BE_AUTHORITATIVE,
    DEFAULT_FRESHNESS,
    DUCKDB_QUACK_OWNER,
    EFFECT_CLASS,
    EXTRACTOR_IDENTITY,
    FACT_ANALYSIS_CACHE,
    FACT_COMPLETION_RECORD,
    FACT_CONTROL_PLANE_STORE,
    FACT_DAEMON_REGISTRY,
    FACT_DOMAIN_EVENT,
    FACT_GOAL_RECORD,
    FACT_JSON_INVENTORY,
    FACT_LEASE_RECORD,
    FACT_PROVIDER_INVOCATION,
    FACT_RECEIPT_RECORD,
    FACT_TASK_RECORD,
    FACT_WORKTREE_BINDING,
    INDEFINITE_DUAL_AUTHORITY_PROHIBITED,
    MARKDOWN_CAN_BE_AUTHORITATIVE,
    MODEL_CAN_CREATE_DUAL_AUTHORITY,
    MODEL_CAN_GRANT_AUTHORITY,
    MODEL_CAN_MUTATE_STORES,
    PROHIBITED_AUTHORITATIVE_KINDS,
    PROJECTION_CAN_BE_AUTHORITATIVE,
    REBUILDABLE_DISPOSITIONS,
    REQUIRED_DISPOSITIONS,
    REQUIRED_MIGRATION_PHASES,
    REQUIRED_MUTABLE_FACTS,
    REQUIRED_STORE_KINDS,
    STATE_OWNERSHIP_EVIDENCE,
    STATE_OWNERSHIP_SCHEMA,
    STATE_OWNERSHIP_VERSION,
    TASK_ID,
    UNKNOWN_OWNER_ACCEPTED,
    MigrationPhase,
    StateConflictKind,
    StateDisposition,
    StateItem,
    StateMigrationPhase,
    StateMigrationPlan,
    StateOwnershipAuthorityError,
    StateOwnershipError,
    StateOwnershipModel,
    StoreKind,
    build_state_ownership_model,
    classify_state_item,
    detect_state_conflicts,
    items_from_inventory,
    plan_bounded_migration,
    refuse_authority_grant,
    refuse_dual_authority,
    refuse_markdown_authority,
    refuse_store_mutation,
)
from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

_TREE = "a698da9e4b54e2929adacb613bc61ba3e72eed58"
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


def _item(
    item_id: str,
    kind: StoreKind,
    fact_id: str,
    path: str,
    disposition: StateDisposition,
    *,
    start: int = 1,
    rebuildable: bool | None = None,
    writable: bool | None = None,
    nominated_owner: str = DUCKDB_QUACK_OWNER,
    tables: tuple[str, ...] = (),
    uncertainty: str = "",
    confidence: Confidence = Confidence.EXACT,
    source_path: str | None = None,
) -> StateItem:
    if rebuildable is None:
        rebuildable = disposition in REBUILDABLE_DISPOSITIONS
    if writable is None:
        writable = disposition is StateDisposition.AUTHORITATIVE
    return StateItem(
        item_id=item_id,
        kind=kind,
        fact_id=fact_id,
        path=path,
        disposition=disposition,
        nominated_owner=nominated_owner,
        provenance=_fact(source_path or "pkg/store.py", start, confidence=confidence),
        rebuildable=rebuildable,
        writable=writable,
        tables=tables,
        uncertainty=uncertainty,
    )


def _node(
    node_id: str,
    kind: NodeKind,
    path: str,
    start: int,
    *,
    confidence: Confidence = Confidence.EXACT,
) -> ArchitectureNode:
    return ArchitectureNode(
        node_id=node_id,
        kind=kind,
        provenance=_fact(path, start, confidence=confidence),
    )


def _edge(
    edge_id: str,
    kind: EdgeKind,
    source: str,
    target: str,
    path: str,
    start: int,
) -> ArchitectureEdge:
    return ArchitectureEdge(
        edge_id=edge_id,
        kind=kind,
        source=source,
        target=target,
        provenance=_fact(path, start),
    )


def _graph(
    nodes: tuple[ArchitectureNode, ...],
    edges: tuple[ArchitectureEdge, ...] = (),
) -> ArchitectureIR:
    return ArchitectureIR.from_parts(
        repository_tree=_TREE,
        freshness=_FRESHNESS,
        nodes=nodes,
        edges=edges,
    )


def _model(**kwargs: object) -> StateOwnershipModel:
    defaults: dict[str, object] = {
        "repository_tree": _TREE,
        "freshness": _FRESHNESS,
    }
    defaults.update(kwargs)
    return build_state_ownership_model(**defaults)  # type: ignore[arg-type]


def test_closed_store_classes_dispositions_and_evidence_pins() -> None:
    assert STATE_OWNERSHIP_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/state-ownership-model@1"
    )
    assert STATE_OWNERSHIP_SCHEMA.endswith("state-ownership-model@1")
    assert STATE_OWNERSHIP_VERSION == 1
    assert STATE_OWNERSHIP_EVIDENCE == "pcar/state-ownership-model@1"
    assert EXTRACTOR_IDENTITY == "pcar-014-state-ownership-model"
    assert TASK_ID == "PCAR-014"
    assert DEFAULT_FRESHNESS == "pcar-014-state-ownership"
    assert EFFECT_CLASS == "read_only_analysis"
    assert MODEL_CAN_MUTATE_STORES is False
    assert MODEL_CAN_GRANT_AUTHORITY is False
    assert MODEL_CAN_CREATE_DUAL_AUTHORITY is False
    assert MARKDOWN_CAN_BE_AUTHORITATIVE is False
    assert DASHBOARD_CAN_BE_AUTHORITATIVE is False
    assert PROJECTION_CAN_BE_AUTHORITATIVE is False
    assert CACHE_CAN_BE_AUTHORITATIVE is False
    assert UNKNOWN_OWNER_ACCEPTED is False
    assert INDEFINITE_DUAL_AUTHORITY_PROHIBITED is True
    assert CONTENT_IDENTITY_IS_NOT_AUTHORITY is True
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
    assert tuple(item.value for item in REQUIRED_DISPOSITIONS) == (
        "authoritative",
        "materialized_projection",
        "cache",
        "historical_event",
        "fixture",
        "legacy",
        "unknown",
    )
    assert CLOSED_DISPOSITIONS == {item.value for item in StateDisposition}
    baseline = json.loads(_BASELINE.read_text(encoding="utf-8"))
    assert set(baseline["required_kinds"]) == CLOSED_STORE_KINDS
    assert set(baseline["closed_dispositions"]) == CLOSED_DISPOSITIONS
    assert tuple(item.value for item in REQUIRED_MIGRATION_PHASES) == (
        "snapshot",
        "dual_read_shadow",
        "controlled_dual_write",
        "cutover",
        "validation",
        "read_only_legacy",
        "retirement",
    )
    assert CLOSED_MIGRATION_PHASES == {item.value for item in MigrationPhase}
    assert "unknown_owner" in CLOSED_CONFLICT_KINDS
    assert "multiple_authoritative_stores" in CLOSED_CONFLICT_KINDS
    assert StoreKind.MARKDOWN_TASK_BOARDS in PROHIBITED_AUTHORITATIVE_KINDS
    assert StoreKind.CACHE_NAMESPACES in PROHIBITED_AUTHORITATIVE_KINDS
    with pytest.raises(ValueError):
        StoreKind("sqlite sidecar")
    with pytest.raises(ValueError):
        StateDisposition("shared")
    with pytest.raises(ValueError):
        MigrationPhase("forever_dual_write")
    with pytest.raises(ValueError):
        StateConflictKind("ignore")


def test_all_store_classes_are_classified_with_provenance() -> None:
    model = _model()
    assert model.covers_required_store_kinds is True
    covered = {item.kind for item in model.items}
    assert covered == set(REQUIRED_STORE_KINDS)
    by_kind = {item.kind: item for item in model.items}
    assert by_kind[StoreKind.DUCKDB_TABLES].disposition is StateDisposition.AUTHORITATIVE
    assert (
        by_kind[StoreKind.JSON_FILES].disposition
        is StateDisposition.MATERIALIZED_PROJECTION
    )
    assert (
        by_kind[StoreKind.MARKDOWN_TASK_BOARDS].disposition
        is StateDisposition.MATERIALIZED_PROJECTION
    )
    assert by_kind[StoreKind.IN_MEMORY_REGISTRIES].disposition is StateDisposition.FIXTURE
    assert by_kind[StoreKind.EVENT_LOGS].disposition is StateDisposition.HISTORICAL_EVENT
    assert by_kind[StoreKind.CACHE_NAMESPACES].disposition is StateDisposition.CACHE
    assert by_kind[StoreKind.WORKTREE_METADATA].is_authoritative is True
    assert by_kind[StoreKind.LEASE_RECORDS].is_authoritative is True
    assert any(
        item.kind is StoreKind.PROVIDER_STATE and item.is_authoritative
        for item in model.items
    )
    assert by_kind[StoreKind.GOAL_STATE].is_authoritative is True
    assert by_kind[StoreKind.TASK_STATE].is_authoritative is True
    assert by_kind[StoreKind.COMPLETION_STATE].is_authoritative is True
    assert by_kind[StoreKind.RECEIPT_STATE].is_authoritative is True
    for item in model.items:
        assert item.disposition in StateDisposition
        assert item.provenance.extractor_identity
        assert item.provenance.span.path
        assert item.provenance.repository_tree == _TREE
        assert item.provenance.freshness == _FRESHNESS
        classified = classify_state_item(
            item, repository_tree=_TREE, freshness=_FRESHNESS
        )
        assert classified == item
    assert model.covers_required_dispositions is True
    assert model.unknown_ownership_count == 0


def test_one_authoritative_store_per_mutable_fact() -> None:
    model = _model()
    assert model.one_authoritative_store_holds is True
    assert model.fails_closed is False
    for fact_id in REQUIRED_MUTABLE_FACTS:
        owner = model.authoritative_owner(fact_id)
        assert owner.is_authoritative is True
        assert owner.writable is True
        assert owner.rebuildable is False
        assert owner.nominated_owner == DUCKDB_QUACK_OWNER
        assert owner.path.startswith("{state_root}/control.duckdb")
        peers = [item for item in model.items_for_fact(fact_id) if item.is_authoritative]
        assert len(peers) == 1
    task_items = model.items_for_fact(FACT_TASK_RECORD)
    assert {item.disposition for item in task_items} == {
        StateDisposition.AUTHORITATIVE,
        StateDisposition.MATERIALIZED_PROJECTION,
    }
    provider_items = model.items_for_fact(FACT_PROVIDER_INVOCATION)
    assert {item.disposition for item in provider_items} == {
        StateDisposition.AUTHORITATIVE,
        StateDisposition.LEGACY,
    }
    duplicate = _item(
        "goals-sidecar",
        StoreKind.GOAL_STATE,
        FACT_GOAL_RECORD,
        "state/goals.json",
        StateDisposition.AUTHORITATIVE,
        start=4,
        source_path="pkg/goals_sidecar.py",
    )
    conflicted = _model(items=(*model.items, duplicate), migrations=model.migrations)
    assert conflicted.fails_closed is True
    assert conflicted.one_authoritative_store_holds is False
    kinds = {item.kind for item in conflicted.conflicts_for(FACT_GOAL_RECORD)}
    assert StateConflictKind.MULTIPLE_AUTHORITATIVE_STORES in kinds
    with pytest.raises(StateOwnershipError, match="exactly one authoritative store"):
        conflicted.authoritative_owner(FACT_GOAL_RECORD)


def test_unknown_owner_is_a_hard_conflict_even_beside_authority() -> None:
    model = _model()
    unknown = _item(
        "goals-unknown",
        StoreKind.GOAL_STATE,
        FACT_GOAL_RECORD,
        "pkg/mystery.py",
        StateDisposition.UNKNOWN,
        start=9,
        source_path="pkg/mystery.py",
    )
    conflicted = _model(items=(*model.items, unknown), migrations=model.migrations)
    assert conflicted.fails_closed is True
    assert conflicted.unknown_ownership_count == 1
    kinds = {item.kind for item in conflicted.conflicts_for(FACT_GOAL_RECORD)}
    assert StateConflictKind.UNKNOWN_OWNER in kinds
    assert StateConflictKind.UNKNOWN_PRODUCTION_OWNER in kinds
    assert StateConflictKind.CONFLICTING_DISPOSITION in kinds
    with pytest.raises(StateOwnershipError, match="exactly one authoritative store"):
        conflicted.authoritative_owner(FACT_GOAL_RECORD)
    orphan = _item(
        "orphan-unknown",
        StoreKind.PROVIDER_STATE,
        "provider.orphan",
        "pkg/orphan.py",
        StateDisposition.UNKNOWN,
        start=3,
        source_path="pkg/orphan.py",
    )
    only_unknown = _model(items=(*model.items, orphan), migrations=model.migrations)
    assert only_unknown.fails_closed is True
    assert any(
        item.kind is StateConflictKind.UNKNOWN_OWNER
        for item in only_unknown.conflicts_for("provider.orphan")
    )


def test_inventory_unknown_nominations_fail_closed() -> None:
    inventory = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    model = _model(inventory=inventory)
    assert model.covers_required_store_kinds is True
    unknown_kinds = {
        item.kind
        for item in model.items
        if item.disposition is StateDisposition.UNKNOWN
    }
    assert StoreKind.IN_MEMORY_REGISTRIES in unknown_kinds
    assert StoreKind.PROVIDER_STATE in unknown_kinds
    assert model.fails_closed is True
    assert model.unknown_ownership_count >= 2
    assert any(
        item.kind is StateConflictKind.UNKNOWN_OWNER for item in model.conflicts
    )
    parsed = items_from_inventory(
        inventory, repository_tree=_TREE, freshness=_FRESHNESS
    )
    assert {item.kind for item in parsed} == set(REQUIRED_STORE_KINDS)
    with pytest.raises(StateOwnershipError, match="unknown state-ownership field"):
        items_from_inventory(
            {"stores": [{**inventory["stores"][0], "hidden": True}]},
            repository_tree=_TREE,
            freshness=_FRESHNESS,
        )


def test_bounded_migration_phases_end_dual_read_write() -> None:
    plan = plan_bounded_migration(
        plan_id="migrate-provider-attempt-store",
        fact_id=FACT_PROVIDER_INVOCATION,
        source_item_id="provider-attempt-store",
        target_item_id="provider-invocations",
    )
    assert tuple(item.phase for item in plan.phases) == REQUIRED_MIGRATION_PHASES
    assert plan.dual_write_window() == (MigrationPhase.CONTROLLED_DUAL_WRITE,)
    assert plan.ends_dual_read_write is True
    assert plan.grants_authority is False
    assert plan.indefinite_dual_authority is False
    assert plan.phases[0].phase is MigrationPhase.SNAPSHOT
    assert plan.phases[2].dual_write is True
    assert plan.phases[2].dual_read is True
    assert plan.phases[3].dual_write is False
    assert plan.phases[3].source_writable is False
    assert plan.phases[-1].phase is MigrationPhase.RETIREMENT
    assert plan.phases[-1].retired is True
    assert plan.phases[-1].dual_read is False
    model = _model()
    assert model.migrations == (plan,)
    payload = plan.to_dict()
    restored = StateMigrationPlan.from_mapping(payload)
    assert restored == plan
    claimed = payload.pop("content_identity")
    validate_cid(claimed, codecs=("dag-json",))
    assert claimed == cid_for_dag_json(payload)
    snapshot_only = [
        StateMigrationPhase.for_phase(MigrationPhase.SNAPSHOT),
        StateMigrationPhase.for_phase(MigrationPhase.DUAL_READ_SHADOW),
    ]
    with pytest.raises(StateOwnershipError, match="missing migration phase"):
        StateMigrationPlan(
            plan_id="partial",
            fact_id=FACT_PROVIDER_INVOCATION,
            source_item_id="provider-attempt-store",
            target_item_id="provider-invocations",
            phases=tuple(snapshot_only),
        )
    with pytest.raises(StateOwnershipError, match="closed bounded sequence"):
        StateMigrationPlan(
            plan_id="reversed",
            fact_id=FACT_PROVIDER_INVOCATION,
            source_item_id="provider-attempt-store",
            target_item_id="provider-invocations",
            phases=tuple(
                StateMigrationPhase.for_phase(phase)
                for phase in reversed(REQUIRED_MIGRATION_PHASES)
            ),
        )
    with pytest.raises(StateOwnershipError, match="end dual-read/write"):
        StateMigrationPlan(
            plan_id="open",
            fact_id=FACT_PROVIDER_INVOCATION,
            source_item_id="provider-attempt-store",
            target_item_id="provider-invocations",
            phases=tuple(
                StateMigrationPhase.for_phase(phase)
                for phase in REQUIRED_MIGRATION_PHASES
            ),
            ends_dual_read_write=False,
        )
    with pytest.raises(StateOwnershipError, match="indefinite dual authority"):
        StateMigrationPlan(
            plan_id="indefinite",
            fact_id=FACT_PROVIDER_INVOCATION,
            source_item_id="provider-attempt-store",
            target_item_id="provider-invocations",
            phases=tuple(
                StateMigrationPhase.for_phase(phase)
                for phase in REQUIRED_MIGRATION_PHASES
            ),
            indefinite_dual_authority=True,
        )
    with pytest.raises(StateOwnershipAuthorityError, match="cannot grant authority"):
        StateMigrationPlan(
            plan_id="grant",
            fact_id=FACT_PROVIDER_INVOCATION,
            source_item_id="provider-attempt-store",
            target_item_id="provider-invocations",
            phases=tuple(
                StateMigrationPhase.for_phase(phase)
                for phase in REQUIRED_MIGRATION_PHASES
            ),
            grants_authority=True,
        )
    with pytest.raises(StateOwnershipError, match="not bounded"):
        StateMigrationPhase(
            phase=MigrationPhase.RETIREMENT,
            dual_read=True,
            dual_write=True,
            source_writable=True,
            target_writable=True,
            legacy_read_only=False,
            retired=False,
        )
    without_plan = _model(migrations=())
    assert any(
        item.kind is StateConflictKind.LEGACY_WITHOUT_CUTOVER
        for item in without_plan.conflicts
    )


def test_rebuildable_projections_and_caches_are_not_authoritative() -> None:
    model = _model()
    projections = [
        item
        for item in model.items
        if item.disposition in REBUILDABLE_DISPOSITIONS
    ]
    assert projections
    for item in projections:
        assert item.rebuildable is True
        assert item.writable is False
        assert item.is_authoritative is False
    json_item = model.items_for(StoreKind.JSON_FILES)[0]
    markdown_item = model.items_for(StoreKind.MARKDOWN_TASK_BOARDS)[0]
    cache_item = model.items_for(StoreKind.CACHE_NAMESPACES)[0]
    events = model.items_for(StoreKind.EVENT_LOGS)[0]
    registry = model.items_for(StoreKind.IN_MEMORY_REGISTRIES)[0]
    assert json_item.fact_id == FACT_JSON_INVENTORY
    assert markdown_item.fact_id == FACT_TASK_RECORD
    assert cache_item.fact_id == FACT_ANALYSIS_CACHE
    assert json_item.rebuildable is True
    assert markdown_item.rebuildable is True
    assert cache_item.rebuildable is True
    assert events.disposition is StateDisposition.HISTORICAL_EVENT
    assert events.writable is False
    assert events.rebuildable is False
    assert registry.disposition is StateDisposition.FIXTURE
    assert registry.fact_id == FACT_DAEMON_REGISTRY
    with pytest.raises(StateOwnershipError, match="Markdown cannot be"):
        _item(
            "md-auth",
            StoreKind.MARKDOWN_TASK_BOARDS,
            FACT_TASK_RECORD,
            "docs/board.md",
            StateDisposition.AUTHORITATIVE,
            rebuildable=False,
            writable=True,
            source_path="docs/board.md",
        )
    with pytest.raises(StateOwnershipError, match="cache cannot be"):
        _item(
            "cache-auth",
            StoreKind.CACHE_NAMESPACES,
            FACT_ANALYSIS_CACHE,
            "pkg/cache.py",
            StateDisposition.AUTHORITATIVE,
            rebuildable=False,
            writable=True,
            source_path="pkg/cache.py",
        )
    with pytest.raises(StateOwnershipError, match="JSON projections cannot be"):
        _item(
            "json-auth",
            StoreKind.JSON_FILES,
            FACT_JSON_INVENTORY,
            "docs/inventory.json",
            StateDisposition.AUTHORITATIVE,
            rebuildable=False,
            writable=True,
            source_path="docs/inventory.json",
        )
    with pytest.raises(StateOwnershipError, match="dashboard cannot be"):
        _item(
            "dash-auth",
            StoreKind.TASK_STATE,
            FACT_TASK_RECORD,
            "templates/enhanced_dashboard.html",
            StateDisposition.AUTHORITATIVE,
            rebuildable=False,
            writable=True,
            source_path="templates/enhanced_dashboard.html",
        )
    with pytest.raises(StateOwnershipError, match="must be rebuildable"):
        _item(
            "stale-proj",
            StoreKind.JSON_FILES,
            FACT_JSON_INVENTORY,
            "docs/inventory",
            StateDisposition.MATERIALIZED_PROJECTION,
            rebuildable=False,
            writable=False,
            source_path="docs/inventory.json",
        )
    with pytest.raises(StateOwnershipError, match="cannot accept production writes"):
        _item(
            "write-cache",
            StoreKind.CACHE_NAMESPACES,
            FACT_ANALYSIS_CACHE,
            "pkg/cache.py",
            StateDisposition.CACHE,
            rebuildable=True,
            writable=True,
            source_path="pkg/cache.py",
        )
    with pytest.raises(StateOwnershipError, match="historical events cannot be"):
        _item(
            "event-auth",
            StoreKind.EVENT_LOGS,
            FACT_DOMAIN_EVENT,
            "{state_root}/control.duckdb#domain_events",
            StateDisposition.AUTHORITATIVE,
            rebuildable=False,
            writable=True,
            source_path="pkg/events.py",
        )
    with pytest.raises(StateOwnershipError, match="in-memory registries cannot be"):
        _item(
            "reg-auth",
            StoreKind.IN_MEMORY_REGISTRIES,
            FACT_DAEMON_REGISTRY,
            "pkg/registry.py",
            StateDisposition.AUTHORITATIVE,
            rebuildable=False,
            writable=True,
            source_path="pkg/registry.py",
        )


def test_current_tree_source_bindings_exist() -> None:
    kinds = {item.kind for item in CURRENT_STATE_BINDINGS}
    assert kinds == set(REQUIRED_STORE_KINDS)
    facts = {item.fact_id for item in CURRENT_STATE_BINDINGS}
    assert set(REQUIRED_MUTABLE_FACTS) <= facts
    assert FACT_CONTROL_PLANE_STORE in facts
    assert FACT_LEASE_RECORD in facts
    assert FACT_WORKTREE_BINDING in facts
    assert FACT_COMPLETION_RECORD in facts
    assert FACT_RECEIPT_RECORD in facts
    assert FACT_DOMAIN_EVENT in facts
    for binding in CURRENT_STATE_BINDINGS:
        path = _ROOT / binding.source_path
        assert path.is_file(), binding.source_path
        text = path.read_text(encoding="utf-8")
        assert binding.nominated_symbol in text
        lines = text.splitlines()
        assert 1 <= binding.start_line <= binding.end_line <= len(lines)


def test_architecture_ir_state_nodes_require_one_owner() -> None:
    owned = _graph(
        (
            _node("n-owner", NodeKind.AUTHORITY, "pkg/store.py", 1),
            _node("n-state", NodeKind.STATE, "pkg/store.py", 4),
        ),
        (
            _edge(
                "e-persist",
                EdgeKind.PERSISTS,
                "n-owner",
                "n-state",
                "pkg/store.py",
                4,
            ),
        ),
    )
    clean = _model(architecture=owned)
    assert not any(
        item.kind is StateConflictKind.GRAPH_OWNER_AMBIGUITY
        for item in clean.conflicts
    )
    assert clean.architecture_ir_identity == owned.content_identity
    ambiguous = _graph(
        (
            _node("n-owner-a", NodeKind.AUTHORITY, "pkg/a.py", 1),
            _node("n-owner-b", NodeKind.AUTHORITY, "pkg/b.py", 1),
            _node("n-state", NodeKind.STATE, "pkg/store.py", 4),
        ),
        (
            _edge("e-a", EdgeKind.WRITES, "n-owner-a", "n-state", "pkg/a.py", 2),
            _edge("e-b", EdgeKind.MUTATES, "n-owner-b", "n-state", "pkg/b.py", 2),
        ),
    )
    conflicted = _model(architecture=ambiguous)
    assert conflicted.fails_closed is True
    assert any(
        item.kind is StateConflictKind.GRAPH_OWNER_AMBIGUITY
        for item in conflicted.conflicts
    )
    missing = _graph((_node("n-state", NodeKind.STATE, "pkg/store.py", 4),))
    missing_model = _model(architecture=missing)
    assert any(
        "no persist/write owner" in item.message for item in missing_model.conflicts
    )


def test_round_trip_and_canonical_identity() -> None:
    model = _model()
    payload = model.to_dict()
    restored = StateOwnershipModel.from_mapping(payload)
    assert restored == model
    assert restored.to_dict() == payload
    assert StateOwnershipModel.from_json(model.to_json()) == model
    claimed = payload.pop("content_identity")
    validate_cid(claimed, codecs=("dag-json",))
    assert claimed == cid_for_dag_json(payload)
    assert claimed == model.content_identity
    assert not claimed.startswith("sha256:")
    assert restored.schema == STATE_OWNERSHIP_SCHEMA
    assert restored.version == STATE_OWNERSHIP_VERSION
    assert restored.can_mutate_stores is False
    assert restored.can_grant_authority is False
    assert restored.can_create_dual_authority is False
    reordered = StateOwnershipModel(
        repository_tree=model.repository_tree,
        freshness=model.freshness,
        items=tuple(reversed(model.items)),
        conflicts=tuple(reversed(model.conflicts)),
        migrations=tuple(reversed(model.migrations)),
        architecture_ir_identity=model.architecture_ir_identity,
    )
    assert reordered.content_identity == model.content_identity
    item_payload = model.items[0].to_dict()
    assert StateItem.from_mapping(item_payload) == model.items[0]


def test_unknown_fields_and_identity_mismatch_are_rejected() -> None:
    model = _model()
    with pytest.raises(StateOwnershipError, match="unknown state-ownership field"):
        StateOwnershipModel.from_mapping({**model.to_dict(), "unexpected": True})
    item_payload = model.items[0].to_dict()
    with pytest.raises(StateOwnershipError, match="unknown state-ownership field"):
        StateItem.from_mapping({**item_payload, "extra": 1})
    plan_payload = model.migrations[0].to_dict()
    with pytest.raises(StateOwnershipError, match="unknown state-ownership field"):
        StateMigrationPlan.from_mapping({**plan_payload, "hidden": True})
    broken = dict(model.to_dict())
    mismatched = {key: value for key, value in broken.items() if key != "content_identity"}
    mismatched["freshness"] = "pcar-014-forged"
    broken["content_identity"] = cid_for_dag_json(mismatched)
    with pytest.raises(StateOwnershipError, match="content identity mismatch"):
        StateOwnershipModel.from_mapping(broken)
    with pytest.raises(StateOwnershipError, match="content identity is not inferred"):
        _item(
            "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            StoreKind.GOAL_STATE,
            FACT_GOAL_RECORD,
            "{state_root}/control.duckdb#goals",
            StateDisposition.AUTHORITATIVE,
            source_path="pkg/goals.py",
        )
    with pytest.raises(StateOwnershipError, match="heuristic or opaque"):
        _item(
            "opaque-goals",
            StoreKind.GOAL_STATE,
            FACT_GOAL_RECORD,
            "{state_root}/control.duckdb#goals",
            StateDisposition.AUTHORITATIVE,
            confidence=Confidence.OPAQUE,
            source_path="pkg/goals.py",
        )


def test_model_cannot_mutate_grant_or_create_dual_authority() -> None:
    model = _model()
    with pytest.raises(StateOwnershipAuthorityError, match="cannot mutate stores"):
        model.mutate_store("goals")
    with pytest.raises(StateOwnershipAuthorityError, match="cannot grant authority"):
        model.grant_authority(FACT_TASK_RECORD)
    with pytest.raises(StateOwnershipAuthorityError, match="cannot create dual"):
        model.create_dual_authority(FACT_TASK_RECORD)
    with pytest.raises(StateOwnershipAuthorityError, match="Markdown cannot be"):
        model.authorize_markdown("todo.md")
    with pytest.raises(StateOwnershipAuthorityError, match="cannot mutate stores"):
        refuse_store_mutation("write")
    with pytest.raises(StateOwnershipAuthorityError, match="cannot grant authority"):
        refuse_authority_grant("promote")
    with pytest.raises(StateOwnershipAuthorityError, match="cannot create dual"):
        refuse_dual_authority("shadow-forever")
    with pytest.raises(StateOwnershipAuthorityError, match="Markdown cannot be"):
        refuse_markdown_authority("board")
    with pytest.raises(StateOwnershipAuthorityError, match="cannot mutate stores"):
        StateOwnershipModel(
            repository_tree=_TREE,
            freshness=_FRESHNESS,
            items=model.items,
            conflicts=model.conflicts,
            migrations=model.migrations,
            can_mutate_stores=True,
        )
    with pytest.raises(StateOwnershipAuthorityError, match="cannot grant authority"):
        StateOwnershipModel(
            repository_tree=_TREE,
            freshness=_FRESHNESS,
            items=model.items,
            conflicts=model.conflicts,
            migrations=model.migrations,
            can_grant_authority=True,
        )
    with pytest.raises(StateOwnershipAuthorityError, match="cannot create dual"):
        StateOwnershipModel(
            repository_tree=_TREE,
            freshness=_FRESHNESS,
            items=model.items,
            conflicts=model.conflicts,
            migrations=model.migrations,
            can_create_dual_authority=True,
        )


def test_missing_required_store_class_and_fact_are_conflicts() -> None:
    model = _model()
    without_cache = tuple(
        item for item in model.items if item.kind is not StoreKind.CACHE_NAMESPACES
    )
    missing_class = _model(items=without_cache, migrations=model.migrations)
    assert any(
        item.kind is StateConflictKind.UNCLASSIFIED_STORE
        for item in missing_class.conflicts
    )
    without_goals = tuple(
        item for item in model.items if item.fact_id != FACT_GOAL_RECORD
    )
    missing_fact = _model(items=without_goals, migrations=model.migrations)
    assert any(
        item.kind is StateConflictKind.MISSING_AUTHORITATIVE_STORE
        and item.fact_id == FACT_GOAL_RECORD
        for item in missing_fact.conflicts
    )
    detected = detect_state_conflicts(without_goals, model.migrations)
    assert any(
        item.kind is StateConflictKind.MISSING_AUTHORITATIVE_STORE
        for item in detected
    )


def test_legacy_without_authoritative_owner_fails_closed() -> None:
    legacy_only = _item(
        "sidecar",
        StoreKind.PROVIDER_STATE,
        "provider.sidecar",
        "pkg/sidecar.py",
        StateDisposition.LEGACY,
        source_path="pkg/sidecar.py",
    )
    model = _model()
    conflicted = _model(items=(*model.items, legacy_only), migrations=model.migrations)
    kinds = {item.kind for item in conflicted.conflicts_for("provider.sidecar")}
    assert StateConflictKind.MISSING_AUTHORITATIVE_STORE in kinds
    assert StateConflictKind.LEGACY_WITHOUT_CUTOVER in kinds
