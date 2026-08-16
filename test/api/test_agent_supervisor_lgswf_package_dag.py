"""LGSWF-002: package DAG inventory and cross-authority interface freeze.

Hermetic static checks over the frozen inventory artifacts:

* ``package_dag.json`` records the actual domain-package import graph, SCCs,
  upward imports, and a remediation map (no package moves).
* ``authority_map.json`` freezes one owner per semantic/operational concern and
  forbids new duplicate authorities.
* ``interface_freeze.json`` names one existing canonical interface (or an
  evidence-backed gap) for every required integration boundary.

The static import-graph recomputation is the authority for graph root equality;
LLM inventory text is not.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INVENTORY = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "logic_governed_semantic_work_fabric_inventory"
)
PACKAGE_DAG_PATH = INVENTORY / "package_dag.json"
AUTHORITY_MAP_PATH = INVENTORY / "authority_map.json"
INTERFACE_FREEZE_PATH = INVENTORY / "interface_freeze.json"
AGENT_SUPERVISOR_ROOT = REPO_ROOT / "ipfs_accelerate_py" / "agent_supervisor"

DOMAIN_PACKAGES = (
    "core",
    "control",
    "task_sources",
    "context",
    "analysis",
    "proof",
    "objectives",
    "planning",
    "prompt",
    "validation",
    "merge",
    "rescue",
    "runtime",
    "self_improvement",
    "integrations",
    "todo_daemon",
)

REQUIRED_BOUNDARY_IDS = (
    "datasets-to-accelerator",
    "context-pack",
    "completion-authority",
    "invalidation",
    "proof-consumer-boundary",
    "objective-plan",
    "resource-claim",
    "result-completion-work",
)

INTENDED_LAYER = {
    "core": 0,
    "control": 1,
    "task_sources": 1,
    "context": 1,
    "analysis": 1,
    "proof": 1,
    "objectives": 2,
    "planning": 2,
    "prompt": 2,
    "validation": 2,
    "merge": 3,
    "rescue": 3,
    "runtime": 3,
    "self_improvement": 3,
    "todo_daemon": 4,
    "integrations": 4,
}


def _load_json(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"missing inventory artifact: {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_of(module: str, scope: set[str]) -> str | None:
    prefix = "ipfs_accelerate_py.agent_supervisor."
    if not module.startswith(prefix):
        return None
    top = module[len(prefix) :].split(".")[0]
    return top if top in scope else None


def _collect_domain_edges() -> tuple[list[tuple[str, str]], dict[str, set[str]]]:
    """AST-static package edges among AGENT_SUPERVISOR_DOMAIN_PACKAGES."""

    scope = set(DOMAIN_PACKAGES)
    edges: dict[str, set[str]] = defaultdict(set)

    for pkg in DOMAIN_PACKAGES:
        root = AGENT_SUPERVISOR_ROOT / pkg
        assert root.is_dir(), f"missing domain package tree: {root}"
        for path in sorted(root.rglob("*.py")):
            try:
                tree = ast.parse(
                    path.read_text(encoding="utf-8", errors="replace"),
                    filename=str(path),
                )
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                targets: list[str] = []
                if isinstance(node, ast.Import):
                    targets = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level and node.level > 0:
                        mod_parts = list(
                            path.relative_to(AGENT_SUPERVISOR_ROOT)
                            .with_suffix("")
                            .parts
                        )
                        if mod_parts and mod_parts[-1] == "__init__":
                            mod_parts = mod_parts[:-1]
                        base = mod_parts[: max(0, len(mod_parts) - node.level)]
                        if node.module:
                            full = ".".join(
                                [
                                    "ipfs_accelerate_py",
                                    "agent_supervisor",
                                    *base,
                                    *node.module.split("."),
                                ]
                            )
                        else:
                            full = ".".join(
                                [
                                    "ipfs_accelerate_py",
                                    "agent_supervisor",
                                    *base,
                                ]
                            )
                        targets = [full]
                    elif node.module:
                        targets = [node.module]
                for target in targets:
                    dst = _package_of(target, scope)
                    if dst and dst != pkg:
                        edges[pkg].add(dst)

    edge_list = sorted((src, dst) for src, dsts in edges.items() for dst in dsts)
    adj = {pkg: set(edges.get(pkg, ())) for pkg in DOMAIN_PACKAGES}
    return edge_list, adj


def _tarjan_sccs(nodes: list[str], adj: dict[str, set[str]]) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    onstack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    components: list[list[str]] = []

    def strongconnect(vertex: str) -> None:
        nonlocal index
        indices[vertex] = index
        lowlink[vertex] = index
        index += 1
        stack.append(vertex)
        onstack.add(vertex)
        for successor in sorted(adj.get(vertex, ())):
            if successor not in indices:
                strongconnect(successor)
                lowlink[vertex] = min(lowlink[vertex], lowlink[successor])
            elif successor in onstack:
                lowlink[vertex] = min(lowlink[vertex], indices[successor])
        if lowlink[vertex] == indices[vertex]:
            component: list[str] = []
            while True:
                member = stack.pop()
                onstack.remove(member)
                component.append(member)
                if member == vertex:
                    break
            components.append(sorted(component))

    for node in sorted(nodes):
        if node not in indices:
            strongconnect(node)
    return components


def _graph_root_sha256(edge_list: list[tuple[str, str]]) -> str:
    payload = {
        "nodes": list(DOMAIN_PACKAGES),
        "edges": [[src, dst] for src, dst in edge_list],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@pytest.fixture(scope="module")
def package_dag() -> dict[str, Any]:
    return _load_json(PACKAGE_DAG_PATH)


@pytest.fixture(scope="module")
def authority_map() -> dict[str, Any]:
    return _load_json(AUTHORITY_MAP_PATH)


@pytest.fixture(scope="module")
def interface_freeze() -> dict[str, Any]:
    return _load_json(INTERFACE_FREEZE_PATH)


@pytest.fixture(scope="module")
def recomputed_graph() -> tuple[list[tuple[str, str]], dict[str, set[str]], str]:
    edge_list, adj = _collect_domain_edges()
    return edge_list, adj, _graph_root_sha256(edge_list)


def test_inventory_artifacts_exist_and_are_objects() -> None:
    for path in (PACKAGE_DAG_PATH, AUTHORITY_MAP_PATH, INTERFACE_FREEZE_PATH):
        data = _load_json(path)
        assert data["task_id"] == "LGSWF-002"
        assert data["goal_id"] == "LGSWF-G010"
        assert data["board_namespace"] == "logic-governed-semantic-work-fabric-v1"
        assert data["plan_revision"] == "LGSWF-PLAN-R1"


def test_package_dag_schema_and_contract_identity(package_dag: dict[str, Any]) -> None:
    assert package_dag["schema"] == "lgswf/package-dependency-dag@1"
    assert package_dag["evidence_id"] == "lgswf/package-dag@1"
    assert package_dag["interface_id"] == "PackageDependencyDAG@1"
    assert package_dag["analysis"]["method"] == "ast_static_imports"
    assert (
        package_dag["analysis"]["intended_direction"]
        == "bottom_up_higher_may_import_lower"
    )
    assert package_dag["acceptance"]["satisfied"] is True


def test_recomputed_graph_root_matches_freeze(
    package_dag: dict[str, Any],
    recomputed_graph: tuple[list[tuple[str, str]], dict[str, set[str]], str],
) -> None:
    edge_list, _adj, root = recomputed_graph
    assert package_dag["graph_root_sha256"] == root
    assert package_dag["summary"]["domain_package_count"] == len(DOMAIN_PACKAGES)
    assert package_dag["summary"]["domain_edge_count"] == len(edge_list)
    assert package_dag["summary"]["domain_edge_count"] > 0

    frozen_edges = {
        (edge["from_package"], edge["to_package"]) for edge in package_dag["edges"]
    }
    assert frozen_edges == set(edge_list)


def test_domain_scc_is_recorded_with_witness_and_remediation(
    package_dag: dict[str, Any],
    recomputed_graph: tuple[list[tuple[str, str]], dict[str, set[str]], str],
) -> None:
    _edge_list, adj, _root = recomputed_graph
    multi = [comp for comp in _tarjan_sccs(list(DOMAIN_PACKAGES), adj) if len(comp) > 1]
    assert multi, "expected the known multi-package domain SCC"
    largest = max(multi, key=len)
    assert package_dag["summary"]["largest_scc_size"] == len(largest)
    assert package_dag["summary"]["multi_member_scc_count"] == len(multi)
    assert package_dag["summary"]["acyclic"] is False

    sccs = package_dag["sccs"]
    assert sccs
    recorded = {tuple(item["members"]) for item in sccs}
    assert tuple(largest) in recorded
    for item in sccs:
        assert item["size"] == len(item["members"])
        assert item["disposition"] == "inventoried_no_package_move"
        assert item["cycle_witness_path"]
        assert len(item["cycle_witness_path"]) >= 2
        assert item["remediation_summary"]


def test_upward_imports_have_complete_remediation_map(
    package_dag: dict[str, Any],
    recomputed_graph: tuple[list[tuple[str, str]], dict[str, set[str]], str],
) -> None:
    edge_list, _adj, _root = recomputed_graph
    expected_upward = sorted(
        (src, dst)
        for src, dst in edge_list
        if INTENDED_LAYER[src] < INTENDED_LAYER[dst]
    )
    frozen_upward = sorted(
        (item["from_package"], item["to_package"])
        for item in package_dag["upward_imports"]
    )
    assert frozen_upward == expected_upward
    assert package_dag["summary"]["upward_import_count"] == len(expected_upward)
    assert expected_upward, "fixture expects documented upward violations"

    remediation_edges = {
        item["edge"] for item in package_dag["remediation_map"]
    }
    for src, dst in expected_upward:
        edge_key = f"{src}->{dst}"
        assert edge_key in remediation_edges
    for item in package_dag["remediation_map"]:
        assert item["authority_change_allowed"] is False
        assert item["status"] == "planned_no_move_until_evidence"
        assert item["strategy"]
        assert item["detail"]
        assert item["witness_files"]


def test_compatibility_boundaries_classify_reverse_imports(
    package_dag: dict[str, Any],
) -> None:
    compat = package_dag["compatibility_boundaries"]
    reverse = compat["datasets_to_accelerator_reverse_imports"]
    assert package_dag["summary"]["datasets_to_accelerator_import_file_count"] == len(
        reverse
    )
    assert reverse, "datasets reverse imports of accelerator must be inventoried"
    for item in reverse:
        assert item["authority"] == "none_not_semantic_authority"
        assert item["classification"]
        assert item["path"].startswith("ipfs_datasets_py/")
        assert (REPO_ROOT / item["path"]).is_file()
    assert "not semantic authority" in compat["rule"].lower() or (
        "semantic meaning" in compat["rule"]
    )


def test_authority_map_forbids_duplicate_semantic_and_operational_authority(
    authority_map: dict[str, Any],
) -> None:
    assert authority_map["schema"] == "lgswf/authority-map@1"
    assert authority_map["interface_id"] == "AuthorityMap@1"
    policy = authority_map["conflict_policy"]
    assert policy["admission"] == "fail_closed"
    assert policy["duplicate_semantic_authority_allowed"] is False
    assert policy["duplicate_operational_authority_allowed"] is False
    assert policy["compatibility_facade_grants_authority"] is False
    assert policy["new_semantic_index_allowed"] is False
    assert policy["new_capsule_compiler_allowed"] is False
    assert policy["new_plan_store_allowed"] is False
    assert policy["new_daemon_framework_allowed"] is False
    assert policy["new_mcpplusplus_profile_allowed"] is False
    assert policy["operational_fields_in_semantic_state_root_allowed"] is False

    owners = authority_map["canonical_owners"]
    assert owners["semantic_identity_and_roots"]["owner"] == "ipfs_datasets_py"
    assert owners["operational_coordination"]["owner"] == "ipfs_accelerate_py"
    assert owners["semantic_identity_and_roots"]["role"] == "semantic_authority"
    assert owners["operational_coordination"]["role"] == "operational_authority"

    guarantees = authority_map["no_new_authority_guarantees"]
    assert any("semantic index" in item.lower() for item in guarantees)
    assert any("capsule compiler" in item.lower() for item in guarantees)
    assert any("operational fields" in item.lower() for item in guarantees)
    assert authority_map["acceptance"]["satisfied"] is True


def test_authority_map_classifies_facades_as_non_authority(
    authority_map: dict[str, Any],
) -> None:
    facades = authority_map["facade_classifications"]
    assert facades
    for facade in facades:
        if facade.get("classification") in {
            "compatibility_facade",
            "composition_facade",
            "dag_inversion_non_authority",
        }:
            assert facade.get("authority") in (False, None) or facade[
                "authority"
            ] is False
    semantic_state = authority_map["package_authority"][
        "ipfs_accelerate_py.agent_supervisor.semantic_state"
    ]
    assert semantic_state["may_redefine_semantic_identity"] is False
    assert "datasets_adapter.py" in semantic_state["canonical_adapter"]


def test_interface_freeze_covers_required_boundaries(
    interface_freeze: dict[str, Any],
) -> None:
    assert interface_freeze["schema"] == "lgswf/integration-interface-freeze@1"
    assert interface_freeze["interface_id"] == "IntegrationInterfaceFreeze@1"
    assert interface_freeze["freeze_policy"]["admission"] == "fail_closed"
    assert (
        interface_freeze["freeze_policy"]["compatibility_facade_is_not_canonical"]
        is True
    )

    required = set(interface_freeze["required_boundary_ids"])
    assert required == set(REQUIRED_BOUNDARY_IDS)
    by_id = {item["boundary_id"]: item for item in interface_freeze["boundaries"]}
    assert required.issubset(by_id)

    for boundary_id in REQUIRED_BOUNDARY_IDS:
        boundary = by_id[boundary_id]
        assert boundary["status"] == "frozen"
        assert boundary["canonical_interface"]
        assert boundary["source_exists"] is True
        assert boundary["duplicate_authority_forbidden"] is True
        assert boundary["new_parallel_interface_allowed"] is False
        assert boundary["owner_repository"] in {
            "ipfs_datasets_py",
            "ipfs_accelerate_py",
        }
        source = REPO_ROOT / boundary["source_path"]
        assert source.is_file()
        assert boundary["source_sha256"] == _file_sha256(source)
        text = source.read_text(encoding="utf-8", errors="replace")
        for symbol in boundary["symbols"]:
            assert symbol in text, (
                f"{boundary_id}: symbol {symbol!r} missing from {boundary['source_path']}"
            )


def test_interface_freeze_has_no_duplicate_canonical_names(
    interface_freeze: dict[str, Any],
) -> None:
    seen: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []
    for boundary in interface_freeze["boundaries"]:
        key = boundary["canonical_interface"]
        parts = key.split(" / ") if " / " in key else [key]
        for part in parts:
            if part in seen:
                duplicates.append(
                    {
                        "interface": part,
                        "a": seen[part],
                        "b": boundary["boundary_id"],
                    }
                )
            else:
                seen[part] = boundary["boundary_id"]
    assert duplicates == []
    assert interface_freeze["cross_checks"]["duplicate_canonical_interface_names"] == []
    assert interface_freeze["cross_checks"]["all_boundary_sources_exist"] is True


def test_interface_freeze_cross_links_package_dag_root(
    package_dag: dict[str, Any],
    interface_freeze: dict[str, Any],
) -> None:
    assert (
        interface_freeze["cross_checks"]["package_dag_graph_root_sha256"]
        == package_dag["graph_root_sha256"]
    )
    assert interface_freeze["acceptance"]["satisfied"] is True


def test_no_new_duplicate_authority_introduced_by_freeze(
    authority_map: dict[str, Any],
    interface_freeze: dict[str, Any],
) -> None:
    """Acceptance: freeze reuses existing interfaces; does not mint parallel authorities."""

    semantic_owner = authority_map["canonical_owners"]["semantic_identity_and_roots"][
        "owner"
    ]
    operational_owner = authority_map["canonical_owners"]["operational_coordination"][
        "owner"
    ]
    assert semantic_owner != operational_owner

    semantic_boundaries = {
        "datasets-semantic-state-root",
        "invalidation",
        "invalidation-extension",
        "proof-consumer-boundary",
        "callable-contract",
        "program-contract",
    }
    operational_boundaries = {
        "context-pack",
        "context-pack-record",
        "objective-plan",
        "resource-claim",
        "resource-scheduling",
        "result-completion-work",
        "completion-authority",
        "conflict-graph",
    }
    by_id = {item["boundary_id"]: item for item in interface_freeze["boundaries"]}
    for boundary_id in semantic_boundaries:
        assert by_id[boundary_id]["owner_repository"] == "ipfs_datasets_py"

    # Adapter freezes the datasets producer contract from the accelerate consumer path.
    adapter = by_id["datasets-to-accelerator"]
    assert adapter["owner_repository"] == "ipfs_datasets_py"
    assert adapter["consumer_repository"] == "ipfs_accelerate_py"
    assert "datasets_adapter.py" in adapter["source_path"]
    assert adapter["canonical_interface"] == "SemanticStateProvider@1"

    for boundary_id in operational_boundaries:
        assert by_id[boundary_id]["owner_repository"] == "ipfs_accelerate_py"

    for boundary in interface_freeze["boundaries"]:
        assert boundary["new_parallel_interface_allowed"] is False
        assert boundary["duplicate_authority_forbidden"] is True


def test_semantic_state_root_gap_is_explicit(interface_freeze: dict[str, Any]) -> None:
    by_id = {item["boundary_id"]: item for item in interface_freeze["boundaries"]}
    root = by_id["datasets-semantic-state-root"]
    assert root["gap"]
    assert "unavailable" in root["gap"].lower() or "LGSWF-003" in root["gap"]
    gaps = interface_freeze["cross_checks"]["gaps"]
    assert any(item["boundary_id"] == "datasets-semantic-state-root" for item in gaps)


def test_import_smoke_frozen_adapter_and_domain_constants() -> None:
    """Import-smoke: sealed adapter pins and domain package constant remain loadable."""

    from ipfs_accelerate_py.agent_supervisor import AGENT_SUPERVISOR_DOMAIN_PACKAGES
    from ipfs_accelerate_py.agent_supervisor.semantic_state import datasets_adapter

    assert tuple(AGENT_SUPERVISOR_DOMAIN_PACKAGES) == DOMAIN_PACKAGES
    assert datasets_adapter.PROVIDER_CONTRACT == "SemanticStateProvider@1"
    assert datasets_adapter.EXPECTED_PRODUCER_INTERFACE == "SemanticStateProducer@1"
    assert datasets_adapter.EXPECTED_STATE_VIEW_INTERFACE == "SemanticStateView@1"
    assert datasets_adapter.ADAPTER_ID == "ipfs-datasets-semantic-state-adapter@1"


def test_graph_root_is_deterministic(
    recomputed_graph: tuple[list[tuple[str, str]], dict[str, set[str]], str],
) -> None:
    edge_list, _adj, root = recomputed_graph
    again_edges, _again_adj = _collect_domain_edges()
    assert again_edges == edge_list
    assert _graph_root_sha256(again_edges) == root


def test_package_dag_edge_witnesses_exist(package_dag: dict[str, Any]) -> None:
    for edge in package_dag["edges"]:
        assert edge["witness_files"], edge
        for relative in edge["witness_files"]:
            assert (REPO_ROOT / relative).is_file(), relative
        if not edge["allowed_by_intended_bottom_up"]:
            assert edge["from_layer"] < edge["to_layer"]
