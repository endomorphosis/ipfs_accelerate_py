"""Hermetic PCAR-015 legacy/simulation reachability inventory tests."""

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
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.legacy_paths import (
    CLOSED_CONFLICT_KINDS,
    CLOSED_DYNAMIC_MECHANISMS,
    CLOSED_ORIGIN_TAINTS,
    CLOSED_PATH_KINDS,
    CLOSED_PRODUCTION_PREDICATES,
    CLOSED_REACHABILITY,
    CLOSED_SIDE_EFFECTS,
    COMPACT_INVENTORY_SCHEMA,
    CONTENT_IDENTITY_IS_NOT_AUTHORITY,
    CURRENT_LEGACY_BINDINGS,
    DEAD_CLASSIFICATION_POLICY,
    DEFAULT_FRESHNESS,
    DYNAMIC_UNCERTAINTY_BLOCKS_DEAD,
    EFFECT_CLASS,
    EXTRACTOR_IDENTITY,
    INVENTORY_AUTHORITY,
    INVENTORY_CAN_AUTHORIZE_DELETION,
    INVENTORY_CAN_GRANT_PRODUCTION_AUTHORITY,
    INVENTORY_CAN_PROMOTE_FAKE_TO_LIVE,
    LEGACY_INVENTORY_EVIDENCE,
    LEGACY_INVENTORY_SCHEMA,
    LEGACY_INVENTORY_VERSION,
    PRODUCTION_FLOW_INVARIANT,
    QUARANTINED_ORIGINS,
    REQUIRED_PATH_KINDS,
    REQUIRED_PRODUCTION_PREDICATES,
    REQUIRED_REACHABILITY,
    SEALED_REPOSITORY_TREE,
    STATIC_REACHABILITY_PROVES_DEAD,
    TAINTED_ORIGINS,
    TASK_ID,
    DynamicMechanism,
    DynamicReachabilityRecord,
    LegacyConflictKind,
    LegacyPathAuthorityError,
    LegacyPathError,
    LegacyPathInventory,
    LegacyPathRecord,
    OriginTaint,
    PathKind,
    ProductionPredicate,
    ReachabilityDisposition,
    ReachabilityMethod,
    SideEffectKind,
    blocked_production_predicates,
    build_legacy_path_inventory,
    classify_legacy_path,
    classify_reachability,
    compact_inventory_payload,
    current_legacy_paths,
    join_origin_taint,
    origin_may_satisfy_production_predicate,
    paths_from_inventory,
    preserve_origin_taint,
    refuse_dead_from_static_only,
    refuse_deletion,
    refuse_fake_to_live_promotion,
    refuse_production_authority,
    scan_sources_without_import,
    trace_entrypoint_reachability,
)
from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

_TREE = "a698da9e4b54e2929adacb613bc61ba3e72eed58"
_FRESHNESS = "pcar-015-fixture"
_ROOT = Path(__file__).resolve().parents[3]
_INVENTORY = (
    _ROOT
    / "docs/architecture/architecture_refactorer_inventory"
    / "legacy_simulation_inventory.json"
)
_BASELINE = (
    _ROOT
    / "docs/architecture/architecture_refactorer_inventory"
    / "legacy_simulation_baseline.json"
)

_BOOM = """raise RuntimeError("imported")

def surviving() -> int:
    return 1
"""

_ENTRY = """def main() -> object:
    from pkg.mock_worker import MockWorker
    return MockWorker().test_hardware()
"""

_MOCK_WORKER = """class MockWorker:
    def test_hardware(self) -> dict[str, bool]:
        return {"cuda": True}
"""

_ORPHAN = """def unused() -> int:
    return 1
"""

_DYNAMIC = """import importlib

def load(name: str) -> object:
    return importlib.import_module(name)
"""

_COMPAT = """def old_route() -> str:
    from pkg.legacy_router import historical_route
    return historical_route()
"""

_LEGACY_ROUTER = """def historical_route() -> str:
    return "compat"
"""

_TEST_ONLY = """from pkg.mock_worker import MockWorker

def test_worker() -> None:
    assert MockWorker().test_hardware()["cuda"] is True
"""

_EAGER_EFFECT = """import os
print("loading")
open("state.json", "w").write("x")

def use() -> None:
    return None
"""

_LAZY_CLEAN = """def use() -> None:
    from pkg.mock_worker import MockWorker
    return MockWorker()
"""


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
        extractor_identity="pcar-015-fixture",
        span=_span(path, start, end),
        confidence=confidence,
        freshness=_FRESHNESS,
        repository_tree=_TREE,
    )


def _path(
    path_id: str,
    kind: PathKind,
    path: str,
    symbol: str,
    origin: OriginTaint,
    reachability: ReachabilityDisposition,
    *,
    start: int = 1,
    uncertainty: str = "",
    mechanisms: tuple[DynamicMechanism, ...] = (),
    confidence: Confidence = Confidence.EXACT,
) -> LegacyPathRecord:
    if reachability is ReachabilityDisposition.UNKNOWN and not uncertainty:
        uncertainty = "dynamic_loading_unresolved"
    if reachability is ReachabilityDisposition.UNKNOWN and not mechanisms:
        mechanisms = (DynamicMechanism.UNKNOWN,)
    return LegacyPathRecord(
        path_id=path_id,
        kind=kind,
        path=path,
        symbol=symbol,
        origin_taint=origin,
        reachability=reachability,
        provenance=_fact(path, start, confidence=confidence),
        dynamic_mechanisms=mechanisms,
        uncertainty=uncertainty,
    )


def _all_kind_paths() -> tuple[LegacyPathRecord, ...]:
    return (
        _path(
            "mock-worker",
            PathKind.MOCK_WORKERS,
            "pkg/mock_worker.py",
            "MockWorker",
            OriginTaint.MOCK,
            ReachabilityDisposition.PRODUCTION_REACHABLE,
        ),
        _path(
            "mock-inference",
            PathKind.MOCK_INFERENCE_HANDLERS,
            "pkg/ops.py",
            "_create_mock_handler",
            OriginTaint.MOCK,
            ReachabilityDisposition.PRODUCTION_REACHABLE,
            start=4,
        ),
        _path(
            "sim-hw",
            PathKind.SIMULATED_HARDWARE,
            "pkg/hardware.py",
            "create_mock_hardware",
            OriginTaint.SIMULATION,
            ReachabilityDisposition.TEST_ONLY,
        ),
        _path(
            "fake-cid",
            PathKind.FAKE_OR_COMPATIBILITY_CIDS,
            "pkg/cids.py",
            "random_cid",
            OriginTaint.COMPATIBILITY,
            ReachabilityDisposition.COMPATIBILITY_ONLY,
        ),
        _path(
            "fixture-provider",
            PathKind.FIXTURE_PROVIDERS,
            "test/fixtures/provider.py",
            "FixtureProvider",
            OriginTaint.FIXTURE,
            ReachabilityDisposition.TEST_ONLY,
        ),
        _path(
            "fallback-success",
            PathKind.FALLBACK_SUCCESS_PATHS,
            "pkg/fallback.py",
            "overall_passed",
            OriginTaint.SIMULATION,
            ReachabilityDisposition.PRODUCTION_REACHABLE,
        ),
        _path(
            "deprecated-coord",
            PathKind.DEPRECATED_COORDINATORS,
            "pkg/coordinator.py",
            "LegacyCoordinator",
            OriginTaint.COMPATIBILITY,
            ReachabilityDisposition.COMPATIBILITY_ONLY,
        ),
        _path(
            "legacy-endpoints",
            PathKind.LEGACY_ENDPOINT_REGISTRIES,
            "pkg/endpoints.py",
            "ENDPOINTS",
            OriginTaint.COMPATIBILITY,
            ReachabilityDisposition.COMPATIBILITY_ONLY,
        ),
        _path(
            "historical-router",
            PathKind.HISTORICAL_PROVIDER_ROUTERS,
            "pkg/legacy_router.py",
            "historical_route",
            OriginTaint.COMPATIBILITY,
            ReachabilityDisposition.COMPATIBILITY_ONLY,
        ),
        _path(
            "dynamic-loader",
            PathKind.HISTORICAL_PROVIDER_ROUTERS,
            "pkg/loader.py",
            "load",
            OriginTaint.UNKNOWN,
            ReachabilityDisposition.UNKNOWN,
            uncertainty="dynamic_import_module_target_unbound",
            mechanisms=(DynamicMechanism.IMPORTLIB_IMPORT_MODULE,),
        ),
    )


def _inventory(**kwargs: object) -> LegacyPathInventory:
    defaults: dict[str, object] = {
        "repository_tree": _TREE,
        "freshness": _FRESHNESS,
        "paths": _all_kind_paths(),
    }
    defaults.update(kwargs)
    return build_legacy_path_inventory(**defaults)  # type: ignore[arg-type]


def _sources() -> dict[str, str]:
    return {
        "pkg/cli.py": _ENTRY,
        "pkg/mock_worker.py": _MOCK_WORKER,
        "pkg/orphan.py": _ORPHAN,
        "pkg/loader.py": _DYNAMIC,
        "pkg/compat.py": _COMPAT,
        "pkg/legacy_router.py": _LEGACY_ROUTER,
        "pkg/boom.py": _BOOM,
        "pkg/eager.py": _EAGER_EFFECT,
        "pkg/lazy.py": _LAZY_CLEAN,
        "test/test_ops.py": _TEST_ONLY,
    }


def test_closed_vocabularies_and_evidence_pins() -> None:
    assert LEGACY_INVENTORY_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/legacy-simulation-inventory@1"
    )
    assert LEGACY_INVENTORY_SCHEMA.endswith("legacy-simulation-inventory@1")
    assert LEGACY_INVENTORY_VERSION == 1
    assert LEGACY_INVENTORY_EVIDENCE == "pcar/legacy-simulation-inventory@1"
    assert EXTRACTOR_IDENTITY == "pcar-015-legacy-path-inventory"
    assert TASK_ID == "PCAR-015"
    assert DEFAULT_FRESHNESS == "pcar-015-legacy-simulation"
    assert EFFECT_CLASS == "read_only_analysis"
    assert INVENTORY_AUTHORITY is False
    assert INVENTORY_CAN_AUTHORIZE_DELETION is False
    assert INVENTORY_CAN_PROMOTE_FAKE_TO_LIVE is False
    assert INVENTORY_CAN_GRANT_PRODUCTION_AUTHORITY is False
    assert STATIC_REACHABILITY_PROVES_DEAD is False
    assert DYNAMIC_UNCERTAINTY_BLOCKS_DEAD is True
    assert CONTENT_IDENTITY_IS_NOT_AUTHORITY is True
    assert tuple(item.value for item in REQUIRED_PATH_KINDS) == (
        "mock workers",
        "mock inference handlers",
        "simulated hardware",
        "fake or compatibility CIDs",
        "fixture providers",
        "fallback success paths",
        "deprecated coordinators",
        "legacy endpoint registries",
        "historical provider routers",
    )
    assert CLOSED_PATH_KINDS == {item.value for item in PathKind}
    assert tuple(item.value for item in REQUIRED_REACHABILITY) == (
        "production_reachable",
        "test_only",
        "compatibility_only",
        "dead",
        "unknown",
    )
    assert CLOSED_REACHABILITY == {item.value for item in ReachabilityDisposition}
    assert CLOSED_ORIGIN_TAINTS == {
        "production",
        "compatibility",
        "fixture",
        "simulation",
        "mock",
        "unknown",
    }
    assert tuple(item.value for item in REQUIRED_PRODUCTION_PREDICATES) == (
        "production_capability",
        "execution_success",
        "proof",
        "completion",
        "release",
    )
    assert CLOSED_PRODUCTION_PREDICATES == {
        item.value for item in ProductionPredicate
    }
    assert OriginTaint.MOCK in QUARANTINED_ORIGINS
    assert OriginTaint.FIXTURE in TAINTED_ORIGINS
    assert OriginTaint.PRODUCTION not in TAINTED_ORIGINS
    assert "importlib.import_module" in CLOSED_DYNAMIC_MECHANISMS
    assert "none" in CLOSED_SIDE_EFFECTS
    assert "dead_with_dynamic_loading" in CLOSED_CONFLICT_KINDS
    baseline = json.loads(_BASELINE.read_text(encoding="utf-8"))
    assert set(baseline["required_categories"]) == CLOSED_PATH_KINDS
    assert set(baseline["closed_reachability"]) == CLOSED_REACHABILITY
    assert baseline["production_flow_invariant"] == PRODUCTION_FLOW_INVARIANT
    with pytest.raises(ValueError):
        PathKind("live plugins")
    with pytest.raises(ValueError):
        ReachabilityDisposition("maybe")
    with pytest.raises(ValueError):
        OriginTaint("trusted")
    with pytest.raises(ValueError):
        ProductionPredicate("ok")


def test_all_required_path_types_have_source_identity_and_reachability() -> None:
    inventory = _inventory()
    assert inventory.covers_required_path_kinds is True
    covered = {item.kind for item in inventory.paths}
    assert covered >= set(REQUIRED_PATH_KINDS)
    for kind in REQUIRED_PATH_KINDS:
        records = inventory.paths_for(kind)
        assert records
        for item in records:
            assert item.provenance.extractor_identity
            assert item.provenance.span.path
            assert item.provenance.span.start_line >= 1
            assert item.provenance.repository_tree == _TREE
            assert item.provenance.freshness == _FRESHNESS
            assert item.reachability in ReachabilityDisposition
            assert item.origin_taint in OriginTaint
            assert item.production_authority is False
            classified = classify_legacy_path(
                item, repository_tree=_TREE, freshness=_FRESHNESS
            )
            assert classified.path_id == item.path_id
    worker = inventory.path_for("mock-worker")
    assert worker.origin_taint is OriginTaint.MOCK
    assert worker.reachability is ReachabilityDisposition.PRODUCTION_REACHABLE
    assert worker.may_satisfy_production_predicate is False
    assert worker.is_quarantined is True


def test_explicit_reachability_dispositions_include_unknown_and_dead() -> None:
    traces = trace_entrypoint_reachability(
        _sources(),
        repository_tree=_TREE,
        freshness=_FRESHNESS,
        entrypoints=("pkg/cli.py", "pkg/compat.py", "test/test_ops.py"),
        entrypoint_kinds={
            "pkg/cli.py": "production",
            "pkg/compat.py": "compatibility",
            "test/test_ops.py": "test",
        },
        origin_by_symbol={
            "MockWorker": OriginTaint.MOCK,
            "historical_route": OriginTaint.COMPATIBILITY,
        },
        dead_when_unreferenced=True,
    )
    by_target = {item.target: item for item in traces}
    mock = by_target["pkg.mock_worker.MockWorker"]
    assert mock.disposition is ReachabilityDisposition.PRODUCTION_REACHABLE
    assert mock.origin_taint is OriginTaint.MOCK
    assert mock.method in {
        ReachabilityMethod.STATIC_IMPORT,
        ReachabilityMethod.ENTRYPOINT,
    }
    historical = by_target["pkg.legacy_router.historical_route"]
    assert historical.disposition is ReachabilityDisposition.COMPATIBILITY_ONLY
    loader = by_target["pkg.loader.load"]
    assert loader.disposition is ReachabilityDisposition.UNKNOWN
    assert loader.method is ReachabilityMethod.DYNAMIC_UNKNOWN
    orphan = by_target["pkg.orphan.unused"]
    assert orphan.disposition is ReachabilityDisposition.DEAD
    assert orphan.method is ReachabilityMethod.UNREFERENCED_NO_DYNAMIC
    dispositions = {item.disposition for item in traces}
    assert ReachabilityDisposition.PRODUCTION_REACHABLE in dispositions
    assert ReachabilityDisposition.COMPATIBILITY_ONLY in dispositions
    assert ReachabilityDisposition.UNKNOWN in dispositions
    assert ReachabilityDisposition.DEAD in dispositions


def test_dynamic_uncertainty_blocks_dead_classification() -> None:
    dynamic = (
        DynamicReachabilityRecord(
            path="pkg/loader.py",
            symbol="importlib.import_module",
            mechanism=DynamicMechanism.IMPORTLIB_IMPORT_MODULE,
            provenance=_fact("pkg/loader.py", 4, confidence=Confidence.OPAQUE),
            uncertainty="dynamic_import_module_target_unbound",
        ),
    )
    assert (
        classify_reachability(
            static_from_production=False,
            static_from_test=False,
            static_from_compatibility=False,
            dynamic_records=dynamic,
            unreferenced=True,
            dead_when_unreferenced=True,
        )
        is ReachabilityDisposition.UNKNOWN
    )
    with pytest.raises(LegacyPathError, match="unknown, not dead"):
        _path(
            "dead-dynamic",
            PathKind.HISTORICAL_PROVIDER_ROUTERS,
            "pkg/loader.py",
            "load",
            OriginTaint.UNKNOWN,
            ReachabilityDisposition.DEAD,
            mechanisms=(DynamicMechanism.IMPORTLIB_IMPORT_MODULE,),
            uncertainty="should-not-matter",
        )
    with pytest.raises(LegacyPathError, match="static reachability alone"):
        refuse_dead_from_static_only("orphan")
    assert STATIC_REACHABILITY_PROVES_DEAD is False
    assert DYNAMIC_UNCERTAINTY_BLOCKS_DEAD is True


def test_origin_taint_is_preserved_and_cannot_satisfy_production() -> None:
    joined = preserve_origin_taint(
        OriginTaint.MOCK, OriginTaint.PRODUCTION, OriginTaint.COMPATIBILITY
    )
    assert joined is OriginTaint.MOCK
    assert join_origin_taint(OriginTaint.PRODUCTION, OriginTaint.FIXTURE) is (
        OriginTaint.FIXTURE
    )
    for origin in QUARANTINED_ORIGINS:
        assert origin_may_satisfy_production_predicate(origin) is False
        blocked = blocked_production_predicates(origin)
        assert set(blocked) == CLOSED_PRODUCTION_PREDICATES
        for predicate in REQUIRED_PRODUCTION_PREDICATES:
            assert origin_may_satisfy_production_predicate(origin, predicate) is False
    assert origin_may_satisfy_production_predicate(OriginTaint.PRODUCTION) is True
    inventory = _inventory()
    mock = inventory.path_for("mock-worker")
    taint = next(
        item
        for item in inventory.taint_records
        if item.symbol == "MockWorker"
    )
    assert taint.preserved is True
    assert taint.origin is OriginTaint.MOCK
    assert taint.predicates_blocked == blocked_production_predicates(OriginTaint.MOCK)
    assert mock.may_satisfy_production_predicate is False
    fake = inventory.path_for("fake-cid")
    assert fake.origin_taint is OriginTaint.COMPATIBILITY
    assert fake.may_satisfy_production_predicate is False


def test_side_effect_free_scan_never_imports_inspected_modules() -> None:
    dynamic, scans, imports = scan_sources_without_import(
        {
            "pkg/boom.py": _BOOM,
            "pkg/eager.py": _EAGER_EFFECT,
            "pkg/lazy.py": _LAZY_CLEAN,
            "pkg/loader.py": _DYNAMIC,
        },
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    boom = next(item for item in scans if item.path == "pkg/boom.py")
    assert SideEffectKind.EXCEPTION in boom.effects
    assert boom.side_effect_free is False
    eager = next(item for item in scans if item.path == "pkg/eager.py")
    assert (
        SideEffectKind.FILESYSTEM in eager.effects
        or SideEffectKind.MUTATION in eager.effects
    )
    lazy = next(item for item in scans if item.path == "pkg/lazy.py")
    assert lazy.side_effect_free is True
    assert "pkg.mock_worker" in imports["pkg/lazy.py"]
    assert any(
        item.mechanism is DynamicMechanism.IMPORTLIB_IMPORT_MODULE
        for item in dynamic
    )
    inventory = build_legacy_path_inventory(
        paths=_all_kind_paths(),
        sources={"pkg/boom.py": _BOOM, "pkg/loader.py": _DYNAMIC},
        repository_tree=_TREE,
        freshness=_FRESHNESS,
        dead_when_unreferenced=True,
    )
    assert inventory.side_effect_scans
    assert any(item.path == "pkg/boom.py" for item in inventory.side_effect_scans)
    assert inventory.unknown_dynamic_count >= 1


def test_entrypoint_reachability_fixtures_bind_production_mock_flow() -> None:
    inventory = build_legacy_path_inventory(
        paths=_all_kind_paths(),
        sources=_sources(),
        entrypoints=("pkg/cli.py",),
        entrypoint_kinds={"pkg/cli.py": "production"},
        origin_by_symbol={"MockWorker": OriginTaint.MOCK},
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    mock_traces = [
        item
        for item in inventory.traces
        if item.target.endswith("MockWorker")
    ]
    assert mock_traces
    assert all(
        item.disposition is ReachabilityDisposition.PRODUCTION_REACHABLE
        for item in mock_traces
    )
    assert all(item.origin_taint is OriginTaint.MOCK for item in mock_traces)
    assert all(
        origin_may_satisfy_production_predicate(item.origin_taint) is False
        for item in mock_traces
    )
    hops = mock_traces[0].hops
    assert "pkg/cli.py" in hops
    assert "pkg/mock_worker.py" in hops


def test_dynamic_loader_counterexamples_remain_unknown_not_dead() -> None:
    traces = trace_entrypoint_reachability(
        {
            "pkg/cli.py": "def main() -> None:\n    return None\n",
            "pkg/loader.py": _DYNAMIC,
        },
        repository_tree=_TREE,
        freshness=_FRESHNESS,
        entrypoints=("pkg/cli.py",),
        entrypoint_kinds={"pkg/cli.py": "production"},
        dead_when_unreferenced=True,
    )
    loader = next(item for item in traces if item.target.endswith("load"))
    assert loader.disposition is ReachabilityDisposition.UNKNOWN
    assert loader.method is ReachabilityMethod.DYNAMIC_UNKNOWN
    current = build_legacy_path_inventory(repository_tree=_TREE, freshness=_FRESHNESS)
    unknown = [
        item
        for item in current.paths
        if item.reachability is ReachabilityDisposition.UNKNOWN
    ]
    assert unknown
    assert all(item.uncertainty for item in unknown)
    assert all(item.dynamic_mechanisms for item in unknown)
    assert all(
        item.reachability is not ReachabilityDisposition.DEAD for item in unknown
    )
    assert current.dynamic_records
    assert all(item.blocking is True for item in current.dynamic_records)


def test_inventory_never_grants_production_authority_or_deletion() -> None:
    inventory = _inventory()
    assert inventory.authority is False
    assert inventory.can_authorize_deletion is False
    assert inventory.can_promote_fake_to_live is False
    assert inventory.can_grant_production_authority is False
    with pytest.raises(LegacyPathAuthorityError, match="cannot authorize deletion"):
        inventory.authorize_deletion("mock-worker")
    with pytest.raises(LegacyPathAuthorityError, match="cannot promote fake-to-live"):
        inventory.promote_fake_to_live("fake-cid")
    with pytest.raises(
        LegacyPathAuthorityError, match="cannot grant production authority"
    ):
        inventory.grant_production_authority("mock-worker")
    with pytest.raises(LegacyPathAuthorityError, match="cannot authorize deletion"):
        refuse_deletion("mock-worker")
    with pytest.raises(LegacyPathAuthorityError, match="cannot promote fake-to-live"):
        refuse_fake_to_live_promotion("fake-cid")
    with pytest.raises(
        LegacyPathAuthorityError, match="cannot grant production authority"
    ):
        refuse_production_authority("mock-worker")
    with pytest.raises(
        LegacyPathAuthorityError, match="cannot grant production authority"
    ):
        LegacyPathRecord(
            path_id="granted",
            kind=PathKind.MOCK_WORKERS,
            path="pkg/mock_worker.py",
            symbol="MockWorker",
            origin_taint=OriginTaint.MOCK,
            reachability=ReachabilityDisposition.PRODUCTION_REACHABLE,
            provenance=_fact("pkg/mock_worker.py", 1),
            production_authority=True,
        )
    with pytest.raises(LegacyPathError, match="cannot have production origin"):
        _path(
            "live-mock",
            PathKind.MOCK_WORKERS,
            "pkg/mock_worker.py",
            "MockWorker",
            OriginTaint.PRODUCTION,
            ReachabilityDisposition.PRODUCTION_REACHABLE,
        )


def test_round_trip_and_canonical_identity() -> None:
    inventory = _inventory()
    payload = inventory.to_dict()
    restored = LegacyPathInventory.from_mapping(payload)
    assert restored == inventory
    assert restored.to_dict() == payload
    assert LegacyPathInventory.from_json(inventory.to_json()) == inventory
    claimed = payload.pop("content_identity")
    validate_cid(claimed, codecs=("dag-json",))
    assert claimed == cid_for_dag_json(payload)
    assert claimed == inventory.content_identity
    assert not claimed.startswith("sha256:")
    reordered = LegacyPathInventory(
        repository_tree=inventory.repository_tree,
        freshness=inventory.freshness,
        paths=tuple(reversed(inventory.paths)),
        dynamic_records=tuple(reversed(inventory.dynamic_records)),
        taint_records=tuple(reversed(inventory.taint_records)),
        traces=tuple(reversed(inventory.traces)),
        side_effect_scans=tuple(reversed(inventory.side_effect_scans)),
        conflicts=tuple(reversed(inventory.conflicts)),
        architecture_ir_identity=inventory.architecture_ir_identity,
    )
    assert reordered.content_identity == inventory.content_identity
    item_payload = inventory.paths[0].to_dict()
    assert LegacyPathRecord.from_mapping(item_payload) == inventory.paths[0]


def test_unknown_fields_and_identity_mismatch_are_rejected() -> None:
    inventory = _inventory()
    with pytest.raises(LegacyPathError, match="unknown legacy-path field"):
        LegacyPathInventory.from_mapping({**inventory.to_dict(), "unexpected": True})
    item_payload = inventory.paths[0].to_dict()
    with pytest.raises(LegacyPathError, match="unknown legacy-path field"):
        LegacyPathRecord.from_mapping({**item_payload, "extra": 1})
    payload = inventory.to_dict()
    forged = {key: value for key, value in payload.items() if key != "content_identity"}
    forged["freshness"] = "pcar-015-forged"
    payload["content_identity"] = cid_for_dag_json(forged)
    with pytest.raises(LegacyPathError, match="content identity mismatch"):
        LegacyPathInventory.from_mapping(payload)
    with pytest.raises(LegacyPathError, match="content identity is not inferred"):
        _path(
            "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            PathKind.MOCK_WORKERS,
            "pkg/mock_worker.py",
            "MockWorker",
            OriginTaint.MOCK,
            ReachabilityDisposition.PRODUCTION_REACHABLE,
        )


def test_current_tree_bindings_exist() -> None:
    kinds = {item.kind for item in CURRENT_LEGACY_BINDINGS}
    assert kinds == set(REQUIRED_PATH_KINDS)
    reach = {item.reachability for item in CURRENT_LEGACY_BINDINGS}
    assert ReachabilityDisposition.PRODUCTION_REACHABLE in reach
    assert ReachabilityDisposition.TEST_ONLY in reach
    assert ReachabilityDisposition.COMPATIBILITY_ONLY in reach
    assert ReachabilityDisposition.UNKNOWN in reach
    origins = {item.origin_taint for item in CURRENT_LEGACY_BINDINGS}
    assert OriginTaint.MOCK in origins
    assert OriginTaint.SIMULATION in origins
    assert OriginTaint.FIXTURE in origins
    assert OriginTaint.COMPATIBILITY in origins
    for binding in CURRENT_LEGACY_BINDINGS:
        path = _ROOT / binding.source_path
        assert path.is_file(), binding.source_path
        text = path.read_text(encoding="utf-8")
        assert binding.nominated_symbol in text
        lines = text.splitlines()
        assert 1 <= binding.start_line <= binding.end_line <= len(lines)
    current = build_legacy_path_inventory(repository_tree=_TREE, freshness=_FRESHNESS)
    assert current.covers_required_path_kinds is True
    assert current.covers_required_reachability is True
    assert current.unknown_dynamic_count >= 1
    assert all(item.production_authority is False for item in current.paths)
    quarantined = [item for item in current.paths if item.is_quarantined]
    assert quarantined
    assert all(item.may_satisfy_production_predicate is False for item in quarantined)


def test_compact_inventory_matches_baseline_and_source_spans() -> None:
    raw = _INVENTORY.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert raw == json.dumps(payload, indent=2, sort_keys=True) + "\n"
    assert payload["schema"] == COMPACT_INVENTORY_SCHEMA
    assert payload["authority"] is False
    assert payload["task_id"] == TASK_ID
    assert payload["repository_tree"] == SEALED_REPOSITORY_TREE
    assert payload["production_flow_invariant"] == PRODUCTION_FLOW_INVARIANT
    assert payload["dead_classification_policy"] == DEAD_CLASSIFICATION_POLICY
    assert payload["inspection"]["method"] == "static_and_hermetic_dynamic_reachability"
    assert set(payload["required_categories"]) == CLOSED_PATH_KINDS
    assert payload["closed_reachability"] == [
        item.value for item in REQUIRED_REACHABILITY
    ]
    kinds = {item["kind"] for item in payload["paths"]}
    assert kinds == CLOSED_PATH_KINDS
    reach = {item["reachability"] for item in payload["paths"]}
    assert "production_reachable" in reach
    assert "test_only" in reach
    assert "compatibility_only" in reach
    assert "unknown" in reach
    for item in payload["paths"]:
        assert item["present"] is True
        assert item["production_authority"] is False
        span = item["source_span"]
        path = _ROOT / span["path"]
        assert path.is_file(), span["path"]
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        assert 1 <= span["start_line"] <= span["end_line"] <= len(lines)
        assert item["nominated_symbol"] in path.read_text(encoding="utf-8")
        if item["reachability"] == "unknown":
            assert item["uncertainty"]
            assert item["dynamic_uncertainty"] is True
    current = build_legacy_path_inventory(repository_tree=SEALED_REPOSITORY_TREE)
    compact = compact_inventory_payload(
        current.paths, repository_tree=SEALED_REPOSITORY_TREE
    )
    assert json.dumps(compact, indent=2, sort_keys=True) + "\n" == raw
    parsed = paths_from_inventory(
        payload, repository_tree=_TREE, freshness=_FRESHNESS
    )
    assert {item.kind for item in parsed} == set(REQUIRED_PATH_KINDS)
    with pytest.raises(LegacyPathError, match="unknown legacy-path field"):
        paths_from_inventory(
            {"paths": [{**payload["paths"][0], "hidden": True}]},
            repository_tree=_TREE,
            freshness=_FRESHNESS,
        )


def test_architecture_ir_simulation_and_compatibility_nodes() -> None:
    fact = _fact("pkg/sim.py", 4)
    graph = ArchitectureIR.from_parts(
        repository_tree=_TREE,
        freshness=_FRESHNESS,
        nodes=(
            ArchitectureNode("n:simulation:pkg.sim.probe", NodeKind.SIMULATION, fact),
            ArchitectureNode(
                "n:compatibility:pkg.compat.router",
                NodeKind.COMPATIBILITY,
                _fact("pkg/compat.py", 1),
            ),
            ArchitectureNode(
                "n:symbol:pkg.fallback.ok",
                NodeKind.SYMBOL,
                _fact("pkg/fallback.py", 2),
            ),
        ),
        edges=(
            ArchitectureEdge(
                "e:fallbacks:probe:ok",
                EdgeKind.FALLBACKS_TO,
                "n:simulation:pkg.sim.probe",
                "n:symbol:pkg.fallback.ok",
                fact,
            ),
        ),
    )
    inventory = build_legacy_path_inventory(
        paths=_all_kind_paths(),
        architecture=graph,
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    assert inventory.architecture_ir_identity == graph.content_identity
    sim = next(
        item for item in inventory.paths if item.symbol == "pkg.sim.probe"
    )
    assert sim.origin_taint is OriginTaint.SIMULATION
    assert sim.reachability is ReachabilityDisposition.UNKNOWN
    assert sim.may_satisfy_production_predicate is False
    compat = next(
        item for item in inventory.paths if item.symbol == "pkg.compat.router"
    )
    assert compat.origin_taint is OriginTaint.COMPATIBILITY
    fallback = next(
        item for item in inventory.paths if item.path_id.startswith("ir-fallback-")
    )
    assert fallback.kind is PathKind.FALLBACK_SUCCESS_PATHS
    assert fallback.origin_taint is OriginTaint.SIMULATION


def test_missing_required_path_kind_fails_closed() -> None:
    incomplete = _all_kind_paths()[:-2]
    inventory = build_legacy_path_inventory(
        paths=incomplete,
        repository_tree=_TREE,
        freshness=_FRESHNESS,
    )
    assert inventory.fails_closed is True
    assert any(
        item.kind is LegacyConflictKind.MISSING_REQUIRED_PATH_KIND
        for item in inventory.conflicts
    )
    assert inventory.covers_required_path_kinds is False


def test_current_paths_match_bindings() -> None:
    paths = current_legacy_paths(repository_tree=_TREE, freshness=_FRESHNESS)
    assert {item.path_id for item in paths} == {
        item.path_id for item in CURRENT_LEGACY_BINDINGS
    }
    by_id = {item.path_id: item for item in paths}
    worker = by_id["mock-worker-accelerate"]
    assert worker.kind is PathKind.MOCK_WORKERS
    assert worker.reachability is ReachabilityDisposition.PRODUCTION_REACHABLE
    assert worker.origin_taint is OriginTaint.MOCK
    fallback = by_id["fallback-hardware-success"]
    assert fallback.kind is PathKind.FALLBACK_SUCCESS_PATHS
    assert fallback.origin_taint is OriginTaint.SIMULATION
    loader = by_id["mock-ipfs-dynamic-loader"]
    assert loader.reachability is ReachabilityDisposition.UNKNOWN
    assert loader.uncertainty
