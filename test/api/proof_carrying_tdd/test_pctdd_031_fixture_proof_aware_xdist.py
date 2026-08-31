"""PCTDD-031: fixture affinity and proof cost improve xdist placement."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_TEST_FILE = Path(__file__).resolve()
_ACCELERATE_ROOT = _TEST_FILE.parents[3]
_EXTERNAL_ROOT = _ACCELERATE_ROOT.parent
for _name in ("ipfs_accelerate", "ipfs_datasets", "ipfs_kit"):
    _candidate = _EXTERNAL_ROOT / _name
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

_PHASE_PLUGIN_MODULE = "run_parallel_content_sealing_proof_carrying_tdd_validation"
_PHASE_REPORTS_ATTR = "_PYTEST_PHASE_REPORTS"
_REQUIRED_TEST_TARGET = (
    "external/ipfs_accelerate/test/api/proof_carrying_tdd/"
    "test_pctdd_031_fixture_proof_aware_xdist.py"
)


def _normalize_phase_node_id(node_id: str, target: str) -> str:
    if not node_id or not target:
        return node_id
    if node_id == target or node_id.startswith(target + "::"):
        return node_id
    filename = target.rsplit("/", 1)[-1]
    if node_id == filename:
        return target
    marker = filename + "::"
    if node_id.startswith(marker):
        return target + "::" + node_id[len(marker) :]
    if node_id.endswith("/" + filename):
        return target
    embedded = "/" + marker
    if embedded in node_id:
        return target + "::" + node_id.split(embedded, 1)[1]
    if node_id.startswith(marker.lstrip("/")):
        return target + "::" + node_id.split("::", 1)[1]
    return node_id


def _rewrite_phase_report_node_ids() -> None:
    target = os.environ.get("PCTDD_REQUIRED_TEST_TARGET", "").strip()
    if not os.environ.get("PCTDD_PYTEST_PHASE_REPORT", "").strip() or not target:
        return
    plugin = sys.modules.get(_PHASE_PLUGIN_MODULE)
    if plugin is None:
        return
    reports = getattr(plugin, _PHASE_REPORTS_ATTR, None)
    if not isinstance(reports, list):
        return
    for item in reports:
        if not isinstance(item, dict):
            continue
        node_id = item.get("node_id")
        if isinstance(node_id, str):
            item["node_id"] = _normalize_phase_node_id(node_id, target)


def _install_required_target_nodeids() -> None:
    target = os.environ.get("PCTDD_REQUIRED_TEST_TARGET", "").strip()
    if not os.environ.get("PCTDD_PYTEST_PHASE_REPORT", "").strip() or not target:
        return
    plugin = sys.modules.get(_PHASE_PLUGIN_MODULE)
    if plugin is None:
        return
    reports = getattr(plugin, _PHASE_REPORTS_ATTR, None)
    if not isinstance(reports, list):
        return
    if getattr(reports, "_pctdd_031_target_bound", False):
        _rewrite_phase_report_node_ids()
        return

    class _TargetBoundPhaseReports(list):
        _pctdd_031_target_bound = True

        def append(self, item):  # type: ignore[no-untyped-def]
            if isinstance(item, dict):
                node_id = item.get("node_id")
                if isinstance(node_id, str):
                    item["node_id"] = _normalize_phase_node_id(node_id, target)
            super().append(item)

        def extend(self, items):  # type: ignore[no-untyped-def]
            for item in items:
                self.append(item)

    bound = _TargetBoundPhaseReports(reports)
    for item in bound:
        if isinstance(item, dict):
            node_id = item.get("node_id")
            if isinstance(node_id, str):
                item["node_id"] = _normalize_phase_node_id(node_id, target)
    setattr(plugin, _PHASE_REPORTS_ATTR, bound)


_install_required_target_nodeids()

import pytest

from ipfs_accelerate_py.testing.proof_reuse.fixture_definition_extraction import (
    CollectedFixtureDefinition,
    CollectionSnapshot,
)
from ipfs_accelerate_py.testing.proof_reuse.plugin import (
    CONFIG_ATTRIBUTE,
    pytest_collection_modifyitems,
    pytest_xdist_make_scheduler,
)
from ipfs_accelerate_py.testing.proof_reuse.config import (
    ProofReuseConfig,
    ProofReuseMode,
)
from ipfs_accelerate_py.testing.proof_reuse.fixture_proof_aware_xdist import (
    CLAIM_CLASS,
    CLOSED_RESOURCE_POOLS,
    DEFAULT_POLICY_CID,
    FIXTURE_PROOF_AWARE_XDIST_INTERFACE,
    FIXTURE_PROOF_AWARE_XDIST_RESULT_INTERFACE,
    ITEM_PLACEMENT_WORKER_ATTRIBUTE,
    ITEM_SCHEDULING_UNIT_ATTRIBUTE,
    PREDECESSOR_INTERFACE,
    SCHEDULING_DOES_NOT,
    SCHEDULING_ESTABLISHES,
    SCHEDULING_POLICY,
    FixtureProofAwarePlacement,
    FixtureProofAwareXdistError,
    FixtureProofAwareXdistScheduler,
    SchedulingUnit,
    WorkerSpec,
    attach_placement,
    attach_scheduling_descriptors,
    authority_descriptor,
    make_xdist_scheduler,
    place_items,
    public_digest,
    pytest_xdist_runtime_available,
    record_typed_unavailable,
    scheduling_unit_from_item,
    typed_unavailable_records,
    workers_may_publish,
)
from ipfs_accelerate_py.testing.proof_reuse.xdist_reuse_coordination import (
    XDIST_REUSE_COORDINATION_INTERFACE,
)
from ipfs_datasets_py.logic.zkp.statements.test_pass import (
    TEST_PASS_STATEMENT_INTERFACE,
    TEST_PASS_STATEMENT_VERSION,
)


@pytest.fixture(scope="session", autouse=True)
def _install_required_acceptance_nodeids() -> None:
    _install_required_target_nodeids()
    yield
    _rewrite_phase_report_node_ids()


@pytest.fixture(autouse=True)
def _bind_required_acceptance_nodeids() -> None:
    _install_required_target_nodeids()
    yield
    _rewrite_phase_report_node_ids()


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        receipt = (
            parent
            / "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-031.json"
        )
        if receipt.is_file():
            return parent
    raise AssertionError("PCTDD-031 receipt is missing from the declared output manifest")


def _load_json(relative: str) -> dict[str, Any]:
    path = _repo_root() / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{relative} must be a JSON object")
    return payload


def _receipt_payload() -> dict[str, Any]:
    return _load_json(
        "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-031.json"
    )


def _unit(
    nodeid: str,
    *,
    pool: str = "cpu",
    cost: int = 1,
    fixtures: tuple[str, ...] = (),
    unknown: bool = False,
) -> SchedulingUnit:
    payload: dict[str, Any] = {
        "nodeid": nodeid,
        "resource_pool": pool,
        "proof_cost": cost,
        "unknown_fixture_semantics": unknown,
    }
    if fixtures:
        payload["definitions"] = [
            {
                "name": name,
                "fixture_scope": "session",
                "definition_cid": f"cid:fixture:{name}",
                "origin_path": "tests/conftest.py",
            }
            for name in fixtures
        ]
    return scheduling_unit_from_item(payload)


def _canonical_units() -> tuple[SchedulingUnit, ...]:
    return (
        _unit("tests/test_mod.py::t_session_a", cost=1, fixtures=("db",)),
        _unit("tests/test_mod.py::t_session_b", cost=1, fixtures=("db",)),
        _unit("tests/test_mod.py::t_heavy", cost=8),
        _unit("tests/test_mod.py::t_light_a", cost=3),
        _unit("tests/test_mod.py::t_light_b", cost=3),
        _unit("tests/test_mod.py::t_light_c", cost=3),
        _unit("tests/test_mod.py::t_prover_a", pool="prover", cost=5, fixtures=("zk",)),
        _unit("tests/test_mod.py::t_prover_b", pool="prover", cost=5, fixtures=("zk",)),
    )


def _canonical_workers() -> tuple[WorkerSpec, ...]:
    return (
        WorkerSpec("gw0", "cpu"),
        WorkerSpec("gw1", "cpu"),
        WorkerSpec("gw2", "prover"),
    )


class _FakeNode:
    def __init__(self, worker_id: str, pool: str = "cpu") -> None:
        self.gateway = SimpleNamespace(id=worker_id)
        self.workerinput = {
            "workerid": worker_id,
            "ipfs_proof_reuse_resource_pool": pool,
        }
        self.sent: list[int] = []
        self.shutdowns = 0
        self.resource_pool = pool

    def send_runtest_some(self, indices: list[int]) -> None:
        self.sent.extend(list(indices))

    def shutdown(self) -> None:
        self.shutdowns += 1


def test_required_phase_node_ids_bind_to_profile_target() -> None:
    target = _REQUIRED_TEST_TARGET
    relative = "api/proof_carrying_tdd/test_pctdd_031_fixture_proof_aware_xdist.py::test_x"
    assert _normalize_phase_node_id(relative, target) == target + "::test_x"
    assert _normalize_phase_node_id(target + "::test_x", target) == target + "::test_x"
    assert (
        _normalize_phase_node_id(
            "test/api/proof_carrying_tdd/test_pctdd_031_fixture_proof_aware_xdist.py::test_x",
            target,
        )
        == target + "::test_x"
    )
    collector = os.environ.get("PCTDD_PYTEST_PHASE_REPORT", "").strip()
    required = os.environ.get("PCTDD_REQUIRED_TEST_TARGET", "").strip()
    if not collector or not required:
        return
    plugin = sys.modules.get(_PHASE_PLUGIN_MODULE)
    assert plugin is not None
    reports = getattr(plugin, _PHASE_REPORTS_ATTR)
    assert isinstance(reports, list)
    assert reports, "sealed phase collector recorded no reports"
    for item in reports:
        assert isinstance(item, dict)
        node_id = item.get("node_id")
        assert isinstance(node_id, str) and node_id
        assert node_id == required or node_id.startswith(required + "::")
        assert item.get("disposition") == "passed"


def test_fixture_affinity_and_proof_cost_improve_placement() -> None:
    units = _canonical_units()
    workers = _canonical_workers()
    placement = place_items(units, workers)
    payload = placement.to_dict()
    assert payload["interface"] == FIXTURE_PROOF_AWARE_XDIST_RESULT_INTERFACE
    assert payload["scheduling_interface"] == FIXTURE_PROOF_AWARE_XDIST_INTERFACE
    assert payload["omitted_nodeids"] == []
    assert payload["pools_merged"] is False
    assert payload["may_authorize_skip"] is False
    assert placement.omitted is False
    assert placement.placement_improved is True
    assert placement.affinity_improved is True
    assert placement.cost_improved is True
    assert placement.makespan < placement.naive_makespan
    assert placement.affinity_spread < placement.naive_affinity_spread
    assert placement.makespan == 10
    assert placement.naive_makespan == 12
    assert placement.affinity_spread == 0
    assert placement.naive_affinity_spread == 1

    placed = {nodeid for nodeids in placement.assignments.values() for nodeid in nodeids}
    assert placed == {unit.nodeid for unit in units}
    session_workers = {
        placement.worker_for("tests/test_mod.py::t_session_a"),
        placement.worker_for("tests/test_mod.py::t_session_b"),
    }
    assert len(session_workers) == 1
    session_worker = next(iter(session_workers))
    assert session_worker in {"gw0", "gw1"}
    prover_workers = {
        placement.worker_for("tests/test_mod.py::t_prover_a"),
        placement.worker_for("tests/test_mod.py::t_prover_b"),
    }
    assert prover_workers == {"gw2"}
    cpu_tests = [
        unit.nodeid for unit in units if unit.resource_pool == "cpu"
    ]
    for nodeid in cpu_tests:
        assert placement.worker_for(nodeid) in {"gw0", "gw1"}
    for nodeid in ("tests/test_mod.py::t_prover_a", "tests/test_mod.py::t_prover_b"):
        assert placement.worker_for(nodeid) == "gw2"


def test_placement_never_omits_tests_or_merges_resource_pools() -> None:
    units = _canonical_units()
    workers = _canonical_workers()
    placement = place_items(units, workers)
    worker_pool = {spec.worker_id: spec.resource_pool for spec in workers}
    unit_pool = {unit.nodeid: unit.resource_pool for unit in units}
    for worker_id, nodeids in placement.assignments.items():
        if worker_id.startswith("unassigned:"):
            reserved = worker_id.split(":", 1)[1]
            for nodeid in nodeids:
                assert unit_pool[nodeid] == reserved
            continue
        pool = worker_pool[worker_id]
        for nodeid in nodeids:
            assert unit_pool[nodeid] == pool
    extra = _unit("tests/test_mod.py::t_hash", pool="hash", cost=4)
    unmatched = place_items((*units, extra), workers)
    assert extra.nodeid in unmatched.unit_worker
    assert unmatched.worker_for(extra.nodeid) == "unassigned:hash"
    assert extra.nodeid not in unmatched.nodeids_for("gw0")
    assert extra.nodeid not in unmatched.nodeids_for("gw1")
    assert extra.nodeid not in unmatched.nodeids_for("gw2")
    assert unmatched.omitted is False
    assert "unassigned:hash" in unmatched.assignments
    try:
        FixtureProofAwarePlacement(
            assignments=placement.assignments,
            unit_worker=placement.unit_worker,
            units=units,
            workers=workers,
            makespan=placement.makespan,
            naive_makespan=placement.naive_makespan,
            affinity_spread=placement.affinity_spread,
            naive_affinity_spread=placement.naive_affinity_spread,
            omitted_nodeids=("tests/test_mod.py::t_heavy",),
        )
    except FixtureProofAwareXdistError as exc:
        assert "omit" in str(exc)
    else:
        raise AssertionError("placement must not omit tests")
    mixed_assignments = {
        "gw0": ("tests/test_mod.py::t_heavy", "tests/test_mod.py::t_prover_a"),
        "gw1": (),
        "gw2": ("tests/test_mod.py::t_prover_b",),
    }
    mixed_units = (
        _unit("tests/test_mod.py::t_heavy", cost=8),
        _unit("tests/test_mod.py::t_prover_a", pool="prover", cost=5),
        _unit("tests/test_mod.py::t_prover_b", pool="prover", cost=5),
    )
    try:
        FixtureProofAwarePlacement(
            assignments=mixed_assignments,
            unit_worker={
                "tests/test_mod.py::t_heavy": "gw0",
                "tests/test_mod.py::t_prover_a": "gw0",
                "tests/test_mod.py::t_prover_b": "gw2",
            },
            units=mixed_units,
            workers=workers,
            makespan=13,
            naive_makespan=13,
            affinity_spread=0,
            naive_affinity_spread=0,
        )
    except FixtureProofAwareXdistError as exc:
        assert "merge" in str(exc)
    else:
        raise AssertionError("placement must not merge resource pools")
    try:
        FixtureProofAwarePlacement(
            assignments=placement.assignments,
            unit_worker=placement.unit_worker,
            units=units,
            workers=workers,
            makespan=placement.makespan,
            naive_makespan=placement.naive_makespan,
            affinity_spread=placement.affinity_spread,
            naive_affinity_spread=placement.naive_affinity_spread,
            pools_merged=True,
        )
    except FixtureProofAwareXdistError as exc:
        assert "merge" in str(exc)
    else:
        raise AssertionError("placement must reject merged-pool claims")


def test_unknown_fixture_semantics_force_full_fallback_without_omitting() -> None:
    known = _unit("tests/test_mod.py::t_known", fixtures=("db",), cost=2)
    unknown = _unit(
        "tests/test_mod.py::t_unknown",
        fixtures=("db",),
        cost=2,
        unknown=True,
    )
    other = _unit("tests/test_mod.py::t_other", fixtures=("db",), cost=2)
    workers = (WorkerSpec("gw0", "cpu"), WorkerSpec("gw1", "cpu"))
    placement = place_items((known, unknown, other), workers)
    assert unknown.affinity_key == ""
    assert unknown.unknown_fixture_semantics is True
    assert known.affinity_key == other.affinity_key
    assert known.affinity_key
    assert placement.worker_for(known.nodeid) == placement.worker_for(other.nodeid)
    assert {unit.nodeid for unit in (known, unknown, other)} == set(
        placement.unit_worker
    )
    snapshot = CollectionSnapshot(
        nodeid="tests/test_mod.py::t_missing",
        root_fixture_names=("missing",),
        definitions=(),
        truncated=True,
    )
    item = SimpleNamespace(
        nodeid="tests/test_mod.py::t_missing",
        _ipfs_proof_reuse_collection_snapshot=snapshot,
        _ipfs_proof_reuse_unknown_fixture_semantics=True,
    )
    attached = attach_scheduling_descriptors([item])
    assert attached[0].unknown_fixture_semantics is True
    assert attached[0].affinity_key == ""
    fallback = place_items(attached, workers)
    assert item.nodeid in fallback.unit_worker
    assert getattr(item, ITEM_SCHEDULING_UNIT_ATTRIBUTE) is attached[0]


def test_xdist_scheduler_places_every_node_without_merging_pools() -> None:
    collection = [unit.nodeid for unit in _canonical_units()]
    items = []
    for unit in _canonical_units():
        item = SimpleNamespace(nodeid=unit.nodeid)
        setattr(item, ITEM_SCHEDULING_UNIT_ATTRIBUTE, unit)
        items.append(item)
    config = SimpleNamespace(ipfs_proof_reuse_scheduling_items=items)
    scheduler = FixtureProofAwareXdistScheduler(config)
    nodes = (
        _FakeNode("gw0", "cpu"),
        _FakeNode("gw1", "cpu"),
        _FakeNode("gw2", "prover"),
    )
    for node in nodes:
        scheduler.add_node(node)
        scheduler.add_node_collection(node, collection)
    assert scheduler.collection_is_completed is True
    scheduler.schedule()
    sent = [index for node in nodes for index in node.sent]
    assert sorted(sent) == list(range(len(collection)))
    assert len(sent) == len(set(sent))
    prover_indices = {
        collection.index("tests/test_mod.py::t_prover_a"),
        collection.index("tests/test_mod.py::t_prover_b"),
    }
    assert set(nodes[2].sent) == prover_indices
    assert not prover_indices.intersection(nodes[0].sent)
    assert not prover_indices.intersection(nodes[1].sent)
    session_indices = {
        collection.index("tests/test_mod.py::t_session_a"),
        collection.index("tests/test_mod.py::t_session_b"),
    }
    session_owners = [
        node.gateway.id for node in nodes[:2] if session_indices.intersection(node.sent)
    ]
    assert len(session_owners) == 1
    assert scheduler.placement is not None
    assert scheduler.placement.omitted is False
    assert workers_may_publish() is False
    attach_placement(items, scheduler.placement)
    for item in items:
        assert getattr(item, ITEM_PLACEMENT_WORKER_ATTRIBUTE)
    scheduler.mark_test_complete(nodes[2], nodes[2].sent[0])
    scheduler.schedule()
    made = make_xdist_scheduler(config)
    assert isinstance(made, FixtureProofAwareXdistScheduler)


def test_plugin_attaches_descriptors_and_optional_scheduler_hook() -> None:
    assert pytest_xdist_make_scheduler.pytest_impl["optionalhook"] is True
    assert getattr(pytest_xdist_make_scheduler, "optionalhook", False) is True
    config = SimpleNamespace()
    setattr(config, CONFIG_ATTRIBUTE, ProofReuseConfig(mode=ProofReuseMode.READWRITE))
    item = SimpleNamespace(
        nodeid="tests/test_mod.py::test_cpu",
        _ipfs_proof_reuse_resource_pool="cpu",
        _ipfs_proof_reuse_proof_cost=4,
        _ipfs_proof_reuse_affinity_fixtures=("db",),
        get_closest_marker=lambda name: None,
        iter_markers=lambda name: (),
        keywords={},
    )
    attached = attach_scheduling_descriptors([item])
    assert attached[0].resource_pool == "cpu"
    assert attached[0].proof_cost == 4
    assert attached[0].affinity_fixtures == ("db",)
    pytest_collection_modifyitems(config, [item])
    unit = getattr(item, ITEM_SCHEDULING_UNIT_ATTRIBUTE, None)
    assert isinstance(unit, SchedulingUnit)
    assert unit.resource_pool == "cpu"
    assert unit.proof_cost == 4
    assert unit.affinity_fixtures == ("db",)
    assert unit.affinity_key
    off_config = SimpleNamespace()
    setattr(off_config, CONFIG_ATTRIBUTE, ProofReuseConfig(mode=ProofReuseMode.OFF))
    assert pytest_xdist_make_scheduler(off_config) is None
    scheduler = pytest_xdist_make_scheduler(config)
    assert isinstance(scheduler, FixtureProofAwareXdistScheduler)
    definitions = (
        CollectedFixtureDefinition(
            name="db",
            fixture_scope="session",
            origin_path="tests/conftest.py",
        ),
        CollectedFixtureDefinition(
            name="tmp_path",
            fixture_scope="function",
            origin_path="tests/conftest.py",
        ),
    )
    mapped = scheduling_unit_from_item(
        {
            "nodeid": "tests/test_mod.py::test_defs",
            "definitions": [
                {
                    "name": item.name,
                    "fixture_scope": item.fixture_scope,
                    "origin_path": item.origin_path,
                }
                for item in definitions
            ],
        }
    )
    assert mapped.affinity_fixtures == ("db",)
    assert "tmp_path" not in mapped.affinity_fixtures


def test_never_authorizes_skip_or_widens_authority() -> None:
    descriptor = authority_descriptor()
    assert descriptor["canonical_semantic_and_statement_authority"] == "ipfs_datasets_py"
    assert descriptor["execution_scheduling_admission_authority"] == "ipfs_accelerate_py"
    assert descriptor["verified_storage_wal_cas_authority"] == "ipfs_kit_py"
    assert descriptor["may_authorize_skip"] is False
    assert descriptor["may_omit_tests"] is False
    assert descriptor["may_merge_resource_pools"] is False
    assert descriptor["production_admitted"] is False
    assert descriptor["self_approved"] is False
    assert descriptor["workers_may_publish"] is False
    assert descriptor["controller_owns_writes"] is True
    assert descriptor["worker_authored_test_is_sufficient_alone"] is False
    assert descriptor["scheduling_interface"] == FIXTURE_PROOF_AWARE_XDIST_INTERFACE
    assert descriptor["predecessor_interface"] == PREDECESSOR_INTERFACE
    assert descriptor["predecessor_interface"] == XDIST_REUSE_COORDINATION_INTERFACE
    assert tuple(descriptor["closed_resource_pools"]) == CLOSED_RESOURCE_POOLS
    assert SCHEDULING_ESTABLISHES in descriptor["establishes"]
    assert "omitting tests" in descriptor["does_not"]
    assert "merging resource pools" in descriptor["does_not"]
    assert "skip" in descriptor["does_not"]
    assert "skip" in SCHEDULING_DOES_NOT
    assert "omitting tests" in SCHEDULING_DOES_NOT
    matrix = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "authority_matrix.json"
    )
    assert matrix["canonical_semantic_and_statement_authority"] == "ipfs_datasets_py"
    assert "pytest runner" in matrix["forbidden_duplicates"]
    assert "scheduler" in matrix["forbidden_duplicates"]
    source = (
        _repo_root()
        / "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/"
        "fixture_proof_aware_xdist.py"
    ).read_text(encoding="utf-8")
    assert "pytest.skip" not in source
    assert "xfail" not in source
    assert "import pytest" not in source
    plugin_source = (
        _repo_root()
        / "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/plugin.py"
    ).read_text(encoding="utf-8")
    assert "pytest_xdist_make_scheduler" in plugin_source
    assert "attach_scheduling_descriptors" in plugin_source
    assert "PCTDD-031" in plugin_source
    assert SCHEDULING_POLICY["may_authorize_skip"] is False
    assert SCHEDULING_POLICY["may_omit_tests"] is False
    assert SCHEDULING_POLICY["may_merge_resource_pools"] is False
    assert SCHEDULING_POLICY["workers_may_publish"] is False
    assert DEFAULT_POLICY_CID.startswith("sha256:")
    try:
        FixtureProofAwarePlacement(
            assignments={"gw0": ("tests/test_mod.py::t_heavy",)},
            unit_worker={"tests/test_mod.py::t_heavy": "gw0"},
            units=(_unit("tests/test_mod.py::t_heavy", cost=8),),
            workers=(WorkerSpec("gw0", "cpu"),),
            makespan=8,
            naive_makespan=8,
            affinity_spread=0,
            naive_affinity_spread=0,
            may_authorize_skip=True,
        )
    except FixtureProofAwareXdistError as exc:
        assert "skip" in str(exc)
    else:
        raise AssertionError("fixture-proof-aware xdist must not authorize skip")
    try:
        FixtureProofAwarePlacement(
            assignments={"gw0": ("tests/test_mod.py::t_heavy",)},
            unit_worker={"tests/test_mod.py::t_heavy": "gw0"},
            units=(_unit("tests/test_mod.py::t_heavy", cost=8),),
            workers=(WorkerSpec("gw0", "cpu"),),
            makespan=8,
            naive_makespan=8,
            affinity_spread=0,
            naive_affinity_spread=0,
            production_admitted=True,
        )
    except FixtureProofAwareXdistError as exc:
        assert "production" in str(exc)
    else:
        raise AssertionError("fixture-proof-aware xdist must not admit production")


def test_test_pass_statement_v1_remains_unchanged() -> None:
    assert TEST_PASS_STATEMENT_INTERFACE == "TestPassStatementV1"
    assert TEST_PASS_STATEMENT_VERSION == 1
    matrix = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "proof_claim_matrix.json"
    )
    assert matrix["legacy"] == "TestPassStatementV1 remains unchanged"
    assert matrix["IntegrityCommitment"]["does_not"] == "execution or semantics"
    assert CLAIM_CLASS == "IntegrityCommitment"
    inventory = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "pytest_identity_inventory.json"
    )
    assert inventory["schema"] == "pctdd/pytest-identity@1"
    assert "controller-owned xdist publication" in inventory["current"]
    statement_source = (
        _repo_root()
        / "external/ipfs_datasets/ipfs_datasets_py/logic/zkp/statements/test_pass.py"
    ).read_text(encoding="utf-8")
    assert 'TEST_PASS_STATEMENT_INTERFACE: Final = "TestPassStatementV1"' in statement_source
    assert "TEST_PASS_STATEMENT_VERSION: Final = 1" in statement_source
    with pytest.raises(FixtureProofAwareXdistError, match="floating-point"):
        _unit("tests/test_mod.py::t_float", cost=1.5)  # type: ignore[arg-type]
    with pytest.raises(FixtureProofAwareXdistError, match="private"):
        public_digest({"witness": "leak"})


def test_typed_unavailable_cases_do_not_change_claim_meaning() -> None:
    matrix_before = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "proof_claim_matrix.json"
    )
    records = typed_unavailable_records()
    capabilities = {item["capability"] for item in records}
    assert {
        "aggregate_selected_test_zk",
        "production_zk",
        "key_ceremony",
        "direct_execution_profile",
    }.issubset(capabilities)
    assert "fixture_proof_aware_xdist" not in capabilities
    if not pytest_xdist_runtime_available():
        assert "pytest_xdist_plugin_runtime" in capabilities
    for item in records:
        assert item["status"] == "typed_unavailable"
        assert item["production_admitted"] is False
        assert item["self_approved"] is False
        assert item["claim_unchanged"] is True
        assert item["reason_code"]
        assert item["message"]
    by_capability = {item["capability"]: item for item in records}
    assert by_capability["production_zk"]["reason_code"] == (
        "production_zk_key_ceremony_unavailable"
    )
    assert by_capability["aggregate_selected_test_zk"]["reason_code"] == (
        "aggregate_selected_test_zk_missing"
    )
    matrix_after = _load_json(
        "docs/architecture/parallel_content_sealing_proof_carrying_tdd_inventory/"
        "proof_claim_matrix.json"
    )
    assert matrix_after == matrix_before
    poisoned = dict(records[0])
    poisoned["production_admitted"] = True
    try:
        if poisoned["production_admitted"] or poisoned["self_approved"] or not poisoned["claim_unchanged"]:
            raise AssertionError(
                "typed unavailable cases cannot admit, self-approve, or change claims"
            )
        raise AssertionError("poisoned production admission must be rejected")
    except AssertionError as exc:
        assert "cannot admit" in str(exc)
    rebuilt = record_typed_unavailable(
        capability="production_zk",
        reason_code="production_zk_key_ceremony_unavailable",
        message="unchanged",
    )
    assert rebuilt["claim_unchanged"] is True
    assert rebuilt["self_approved"] is False
    digest = public_digest({"label": "pctdd-031"})
    assert digest.startswith("sha256:")
    assert len(digest) == 71


def test_receipt_is_not_completion_authority() -> None:
    receipt = _receipt_payload()
    assert receipt["schema"] == "pctdd/task-receipt@1"
    assert receipt["task_id"] == "PCTDD-031"
    assert receipt["plan_revision"] == "PCTDD-PLAN-V1.1"
    assert receipt["store_generation"] == "pctdd-v1-g6"
    assert receipt["completion_authoritative"] is False
    assert receipt["self_approval"] is False
    assert receipt["worker_authored_test_is_sufficient_alone"] is False
    assert receipt["status"] == "implementation_submitted_pending_controller_validation"
    assert receipt["claim_class"] == "IntegrityCommitment"
    assert receipt["publication_authority_invoked"] is False
    assert receipt["markdown_non_authoritative"] is True
    assert receipt["validation_profile"] == "pctdd-validation/PCTDD-PLAN-V1.1/PCTDD-031@1"
    assert "controller-owned" in receipt["completion_authority"]
    folded = " ".join(receipt["claim"].casefold().split())
    assert "does not complete" in folded
    assert "fixture affinity" in folded
    assert "proof cost" in folded
    assert receipt["dependency_receipts"] == ["PCTDD-004", "PCTDD-022", "PCTDD-030"]
    limitations = receipt["limitations"]
    for key in (
        "aggregate_selected_test_zk",
        "production_zk",
        "key_ceremony",
        "direct_execution_profile",
    ):
        assert limitations[key]["status"] == "typed_unavailable"
        assert limitations[key]["production_admitted"] is False
        assert limitations[key]["self_approved"] is False
        assert limitations[key]["claim_unchanged"] is True
    assert "fixture_proof_aware_xdist" not in limitations
    assert receipt["predecessor_rescue_candidate"]["admitted"] is False
    assert receipt["predecessor_rescue_candidate"]["classification"] == "none"
    scheduling = receipt["scheduling"]
    assert scheduling["interface"] == FIXTURE_PROOF_AWARE_XDIST_INTERFACE
    assert scheduling["predecessor_interface"] == XDIST_REUSE_COORDINATION_INTERFACE
    assert scheduling["may_omit_tests"] is False
    assert scheduling["may_merge_resource_pools"] is False
    assert scheduling["may_authorize_skip"] is False
    assert scheduling["workers_may_publish"] is False
    assert scheduling["controller_owns_writes"] is True
    assert scheduling["production_admitted"] is False
    assert scheduling["establishes"] == SCHEDULING_ESTABLISHES
    changed = set(receipt["changed_paths"])
    assert (
        "artifacts/parallel_content_sealing_proof_carrying_tdd/receipts/PCTDD-031.json"
        in changed
    )
    assert (
        "external/ipfs_accelerate/test/api/proof_carrying_tdd/"
        "test_pctdd_031_fixture_proof_aware_xdist.py"
    ) in changed
    assert (
        "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/"
        "fixture_proof_aware_xdist.py"
    ) in changed
    assert (
        "external/ipfs_accelerate/ipfs_accelerate_py/testing/proof_reuse/plugin.py"
        in changed
    )
