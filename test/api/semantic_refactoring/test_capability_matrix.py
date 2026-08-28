"""Independent current-tree probes for SPAR-001 VerifiedCapabilityMatrix."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[3]
MATRIX_PATH = (
    ROOT
    / "docs/architecture/semantic_preserving_autonomous_remodularization_inventory"
    / "verified_capability_matrix.json"
)
INVENTORY = ROOT / "docs/architecture/semantic_preserving_autonomous_remodularization_inventory"
SCHEMA = "spar/verified-capability-matrix@1"
INTERFACE = "VerifiedCapabilityMatrix@1"
PROGRAM = "semantic-preserving-autonomous-remodularization-v1"
TASK_ID = "SPAR-001"

ENTRY_KEYS = {
    "authority",
    "authority_is_completion",
    "blocks_dependent_work",
    "classification",
    "concern",
    "disposition",
    "duplicate_authority_forbidden",
    "evidence_class",
    "evidence_status",
    "gap_statement",
    "id",
    "interface_concern",
    "notes",
    "overlap_item",
    "owner",
    "paths",
    "planned_paths",
    "probe_method",
    "required",
    "required_symbols",
    "tests",
    "tool_observations",
}
EVIDENCE_STATUSES = {
    "gap_confirmed",
    "typed_unavailable",
    "verified_present",
    "verified_present_advisory",
    "verified_present_non_authoritative",
}
ACCEPTANCE_SUBSET = (
    "exact-current-tree",
    "declared-effects",
    "independent-validation",
    "rollback",
    "authority-separation",
    "no-safety-floor-regression",
)
WRITE_SCOPE = (
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/verified_capability_matrix.json",
    "test/api/semantic_refactoring/test_capability_matrix.py",
)
PROTECTED_PATHS = (
    ".gitignore",
    "benchmarks/agent_supervisor/semantic_refactoring/preregistration.json",
    "config/agent_supervisor_semantic_preserving_remodularization_scheduler.json",
    "config/semantic_preserving_autonomous_remodularization_dependencies.seal.json",
    "docs/architecture/SEMANTIC_PRESERVING_AUTONOMOUS_REMODULARIZATION_PLAN.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization.objectives.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization.todo.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/authority_matrix.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/benchmark_preregistration.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/dynamic_python_risk_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/identity_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/interface_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/overlap_gap_matrix.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/repository_baseline.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/rollout_baseline.json",
    "scripts/materialize_semantic_preserving_remodularization_program.py",
    "scripts/ops/agent_supervisor/semantic_preserving_remodularization.py",
    "scripts/validate_semantic_preserving_remodularization_board.py",
    "scripts/validate_semantic_preserving_remodularization_dependencies.py",
    "test/api/semantic_refactoring/test_bootstrap_controls.py",
)
AUTHORITY_OWNERS = {
    "formal semantic authority": "ipfs_datasets_py",
    "storage and retrieval authority": "ipfs_kit_py",
    "operational refactoring authority": "ipfs_accelerate_py",
    "live operational authority": "DuckDB/DatabaseTaskSource@1 via Quack",
    "non-authoritative rebuildable projection": "DuckLake",
    "advisory=false authority": "existing vector/hybrid indexes",
    "proposal-only": "current model router",
}
HAPPENS_BEFORE_EDGE_NAMES = frozenset(
    {
        "happens_before",
        "happens-before",
        "initialization_order",
        "initialization-order",
    }
)
LANDING_SEARCH_ROOTS = (
    ROOT / "ipfs_accelerate_py/agent_supervisor",
    ROOT / "ipfs_datasets_py/ipfs_datasets_py",
    ROOT / "ipfs_kit_py/ipfs_kit_py",
)


class VerifiedCapabilityMatrix:
    """Closed SPAR-001 matrix contract reconstructed from current-tree probes."""

    schema = SCHEMA
    interface = INTERFACE
    path = MATRIX_PATH

    @classmethod
    def load(cls) -> dict[str, Any]:
        return json.loads(cls.path.read_text(encoding="utf-8"))


def _load_inventory(name: str) -> dict[str, Any]:
    return json.loads((INVENTORY / name).read_text(encoding="utf-8"))


def _sorted_unique_strings(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(item, str) and item for item in value)
        and value == sorted(set(value))
    )


def _posix(path: str) -> Path:
    candidate = ROOT / path
    if path.startswith("/") or ".." in Path(path).parts:
        raise AssertionError(f"path is not a repository-relative POSIX path: {path}")
    return candidate


def _iter_python_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix == ".py":
            yield path
        return
    if path.is_dir():
        yield from sorted(item for item in path.rglob("*.py") if item.is_file())


def probe_ast_names(path: Path) -> set[str]:
    """Collect current-tree definition and import names without executing modules."""

    names: set[str] = set()
    for file_path in _iter_python_files(path):
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.asname or alias.name.split(".", 1)[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    names.add(alias.asname or alias.name)
    return names


def probe_enum_string_values(path: Path, class_names: Iterable[str]) -> set[str]:
    wanted = set(class_names)
    values: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in wanted:
            for statement in node.body:
                if isinstance(statement, ast.Assign) and isinstance(
                    statement.value, ast.Constant
                ):
                    if isinstance(statement.value.value, str):
                        values.add(statement.value.value)
                        for target in statement.targets:
                            if isinstance(target, ast.Name):
                                values.add(target.id.lower())
    return values


def probe_assign_bool(path: Path, name: str) -> bool | None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                value = node.value.value
                return value if isinstance(value, bool) else None
    return None


def probe_module_spec(module: str) -> str:
    """Sealed-environment tool observation. Never infers usability from source."""

    return "present" if importlib.util.find_spec(module) is not None else "typed_unavailable"


def _symbol_name(symbol: str) -> str:
    return symbol.split("@", 1)[0]


def _landed_class_names() -> set[str]:
    names: set[str] = set()
    for root in LANDING_SEARCH_ROOTS:
        if not root.exists():
            continue
        for file_path in root.rglob("*.py"):
            if "semantic_refactoring" not in file_path.parts:
                continue
            tree = ast.parse(file_path.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    names.add(node.name)
    return names


def probe_current_tree() -> dict[str, Any]:
    """Reconstruct capability facts from the current tree, not from the matrix."""

    interface_inventory = _load_inventory("interface_inventory.json")
    overlap = _load_inventory("overlap_gap_matrix.json")
    identity = _load_inventory("identity_inventory.json")
    present_interfaces = {
        row["concern"]: (_posix(row["path"])).exists()
        for row in interface_inventory["interfaces"]
    }
    edge_values = probe_enum_string_values(
        ROOT / "ipfs_accelerate_py/agent_supervisor/analysis/program_graph.py",
        ("ProgramEdgeKind", "_SnapshotProgramEdgeKind"),
    )
    operator_path = (
        ROOT
        / "ipfs_accelerate_py/agent_supervisor/architecture_refactorer/refactor_operators.py"
    )
    return {
        "interface_paths_present": present_interfaces,
        "semantic_refactoring_packages_present": {
            "ipfs_datasets_py/ipfs_datasets_py/semantic_refactoring": (
                ROOT / "ipfs_datasets_py/ipfs_datasets_py/semantic_refactoring"
            ).exists(),
            "ipfs_accelerate_py/agent_supervisor/semantic_refactoring": (
                ROOT / "ipfs_accelerate_py/agent_supervisor/semantic_refactoring"
            ).exists(),
            "ipfs_kit_py/ipfs_kit_py/semantic_refactoring": (
                ROOT / "ipfs_kit_py/ipfs_kit_py/semantic_refactoring"
            ).exists(),
        },
        "happens_before_edges": sorted(HAPPENS_BEFORE_EDGE_NAMES & edge_values),
        "operator_can_authorize_execution": probe_assign_bool(
            operator_path, "OPERATOR_CAN_AUTHORIZE_EXECUTION"
        ),
        "libcst": probe_module_spec("libcst"),
        "parso": probe_module_spec("parso"),
        "asttokens": probe_module_spec("asttokens"),
        "landed_refactoring_classes": sorted(_landed_class_names()),
        "declared_capsule_types": list(identity["capsule_types"]),
        "overlap_reuse": list(overlap["current_tree_reuse"]),
        "overlap_gaps": list(overlap["gaps"]),
    }


@pytest.fixture(scope="module")
def matrix() -> dict[str, Any]:
    return VerifiedCapabilityMatrix.load()


@pytest.fixture(scope="module")
def capabilities(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = matrix["capabilities"]
    assert isinstance(rows, list) and rows
    return rows


@pytest.fixture(scope="module")
def probes() -> dict[str, Any]:
    return probe_current_tree()


def test_matrix_schema_is_canonical(matrix: Mapping[str, Any]) -> None:
    raw = MATRIX_PATH.read_bytes()
    parsed = json.loads(raw.decode("utf-8"))
    assert raw.decode("utf-8") == json.dumps(parsed, indent=2, sort_keys=True) + "\n"
    assert matrix["schema"] == SCHEMA
    assert matrix["interface"] == INTERFACE
    assert matrix["program"] == PROGRAM
    assert matrix["task_id"] == TASK_ID
    assert tuple(matrix["acceptance_subset"]) == ACCEPTANCE_SUBSET
    assert set(matrix["evidence_status_vocabulary"]) == EVIDENCE_STATUSES
    assert matrix["authority_rules"]["completion_authoritative"] is False
    assert matrix["authority_rules"]["worker_self_approval"] is False
    assert matrix["authority_rules"]["vector_similarity_is_authority"] is False
    assert matrix["authority_rules"]["ducklake_is_authority"] is False


def test_declared_effects_are_owned_paths_only(matrix: Mapping[str, Any]) -> None:
    assert tuple(matrix["declared_effects"]["write_scope"]) == WRITE_SCOPE
    for path in WRITE_SCOPE:
        assert _posix(path).is_file()
    protected = set(PROTECTED_PATHS)
    assert protected.isdisjoint(matrix["declared_effects"]["write_scope"])
    prohibited = matrix["declared_effects"]["prohibited"]
    assert any("sealed SPAR controls" in item for item in prohibited)
    assert any("Self-approval" in item for item in prohibited)


def test_protected_controls_remain_in_tree() -> None:
    for relative in PROTECTED_PATHS:
        path = _posix(relative)
        assert path.exists(), relative
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))


def test_rollback_rejects_without_advancing_roots(matrix: Mapping[str, Any]) -> None:
    rollback = matrix["rollback"]
    assert rollback["advance_accepted_roots"] is False
    assert rollback["source_mutation"] is False
    assert rollback["retains_negative_evidence"] is True
    policy = rollback["policy"].lower()
    assert "reject" in policy
    assert "preimages" in policy or "worktree" in policy
    assert "do not advance accepted roots" in policy


def test_safety_floors_do_not_regress(matrix: Mapping[str, Any]) -> None:
    preregistration = _load_inventory("benchmark_preregistration.json")
    floors = matrix["zero_safety_floors"]
    assert floors == preregistration["zero_safety_floors"]
    assert floors and all(value == 0 for value in floors.values())
    assert matrix["failed_efficiency_target"] == preregistration["failed_efficiency_target"]
    assert "safety floors remain unchanged" in matrix["failed_efficiency_target"]


def test_source_binding_matches_repository_baseline(matrix: Mapping[str, Any]) -> None:
    baseline = _load_inventory("repository_baseline.json")
    by_repo = {row["repository"]: row for row in baseline["authoritative_repositories"]}
    binding = matrix["source_binding"]
    assert binding["accelerator"]["commit"] == by_repo["ipfs_accelerate_py"]["head"]
    assert binding["accelerator"]["tree"] == by_repo["ipfs_accelerate_py"]["tree"]
    assert binding["datasets"]["commit"] == by_repo["ipfs_datasets_py"]["head"]
    assert binding["datasets"]["tree"] == by_repo["ipfs_datasets_py"]["tree"]
    assert binding["kit"]["commit"] == by_repo["ipfs_kit_py"]["head"]
    assert binding["kit"]["tree"] == by_repo["ipfs_kit_py"]["tree"]
    assert matrix["observed_tree"] == _load_inventory("interface_inventory.json")["observed_tree"]


def test_every_interface_inventory_row_has_an_owner_and_status(
    capabilities: list[dict[str, Any]],
) -> None:
    interfaces = _load_inventory("interface_inventory.json")["interfaces"]
    by_concern = {
        row["interface_concern"]: row for row in capabilities if row["interface_concern"]
    }
    assert set(by_concern) == {row["concern"] for row in interfaces}
    for interface in interfaces:
        row = by_concern[interface["concern"]]
        assert row["disposition"] == interface["disposition"]
        assert interface["path"] in row["paths"]
        assert row["evidence_status"] in EVIDENCE_STATUSES
        assert row["duplicate_authority_forbidden"] is True


def test_overlap_and_gap_rows_are_classified(
    matrix: Mapping[str, Any],
    capabilities: list[dict[str, Any]],
) -> None:
    overlap = _load_inventory("overlap_gap_matrix.json")
    by_id = {row["id"]: row for row in capabilities}
    overlap_items = {row["item"]: row["capability_ids"] for row in matrix["overlaps"]}
    assert set(overlap_items) == set(overlap["current_tree_reuse"])
    for item, capability_ids in overlap_items.items():
        assert capability_ids == sorted(set(capability_ids))
        for capability_id in capability_ids:
            assert by_id[capability_id]["overlap_item"] == item
    gap_statements = {row["statement"]: row for row in matrix["gaps"]}
    assert list(gap_statements) == list(overlap["gaps"])
    for statement, row in gap_statements.items():
        capability = by_id[row["capability_id"]]
        assert capability["classification"] == "confirmed_gap"
        assert capability["evidence_status"] == "gap_confirmed"
        assert capability["gap_statement"] == statement
        assert row["evidence_status"] == "gap_confirmed"


def test_authority_separation_has_exactly_one_owner(
    capabilities: list[dict[str, Any]],
) -> None:
    authority = _load_inventory("authority_matrix.json")
    expected_owners = {row["authority"]: row["owner"] for row in authority["rules"]}
    assert expected_owners == AUTHORITY_OWNERS
    for row in capabilities:
        assert set(row) == ENTRY_KEYS
        assert row["owner"] == expected_owners[row["authority"]]
        assert row["duplicate_authority_forbidden"] is True
        if row["authority"] == "advisory=false authority":
            assert row["evidence_status"] == "verified_present_advisory"
            assert row["evidence_class"] == "vector_candidate"
        if row["authority"] == "non-authoritative rebuildable projection":
            assert row["evidence_status"] == "verified_present_non_authoritative"
        if row["authority"] == "live operational authority":
            assert row["authority_is_completion"] is True
        else:
            assert row["authority_is_completion"] is False
    semantic_owners = {
        row["owner"] for row in capabilities if row["authority"] == "formal semantic authority"
    }
    storage_owners = {
        row["owner"] for row in capabilities if row["authority"] == "storage and retrieval authority"
    }
    assert semantic_owners == {"ipfs_datasets_py"}
    assert storage_owners == {"ipfs_kit_py"}


def test_independent_probes_agree_with_matrix(
    capabilities: list[dict[str, Any]],
    probes: Mapping[str, Any],
) -> None:
    assert all(probes["interface_paths_present"].values())
    assert not any(probes["semantic_refactoring_packages_present"].values())
    by_id = {row["id"]: row for row in capabilities}
    assert by_id.keys() == {row["id"] for row in capabilities}
    assert [row["id"] for row in capabilities] == sorted(by_id)

    for row in capabilities:
        present_paths = [path for path in row["paths"] if _posix(path).exists()]
        missing_paths = [path for path in row["paths"] if not _posix(path).exists()]
        missing_planned = [path for path in row["planned_paths"] if not _posix(path).exists()]
        extra_planned = [path for path in row["planned_paths"] if _posix(path).exists()]
        names: set[str] = set()
        for path in present_paths:
            names |= probe_ast_names(_posix(path))
        for test_path in row["tests"]:
            assert _posix(test_path).is_file(), test_path

        if row["classification"] == "confirmed_gap":
            assert row["evidence_status"] == "gap_confirmed"
            assert extra_planned == []
            assert missing_planned == row["planned_paths"]
            if row["id"] != "pcar_source_mutation_executor":
                for symbol in row["required_symbols"]:
                    assert _symbol_name(symbol) not in names
        else:
            assert missing_paths == []
            assert present_paths == row["paths"]
            for symbol in row["required_symbols"]:
                assert _symbol_name(symbol) in names

    assert by_id["vector_retrieval"]["evidence_status"] == "verified_present_advisory"
    assert by_id["ducklake"]["evidence_status"] == "verified_present_non_authoritative"
    assert probes["operator_can_authorize_execution"] is False
    assert by_id["pcar_source_mutation_executor"]["evidence_status"] == "gap_confirmed"


def test_capsule_family_gap_is_independently_confirmed(probes: Mapping[str, Any]) -> None:
    identity = _load_inventory("identity_inventory.json")
    landed = set(probes["landed_refactoring_classes"])
    for capsule in identity["capsule_types"]:
        assert _symbol_name(capsule) not in landed
    assert probes["semantic_refactoring_packages_present"][
        "ipfs_datasets_py/ipfs_datasets_py/semantic_refactoring"
    ] is False
    semantic_names = probe_ast_names(
        ROOT / "ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/semantic_state"
    )
    assert "SemanticCapsule" in semantic_names
    assert "FunctionSemanticCapsule" not in semantic_names


def test_happens_before_gap_is_independently_confirmed(probes: Mapping[str, Any]) -> None:
    assert probes["happens_before_edges"] == []
    edge_values = probe_enum_string_values(
        ROOT / "ipfs_accelerate_py/agent_supervisor/analysis/program_graph.py",
        ("ProgramEdgeKind", "_SnapshotProgramEdgeKind"),
    )
    assert "imports" in edge_values
    assert "state_flow" in edge_values
    assert not HAPPENS_BEFORE_EDGE_NAMES & edge_values


def test_pcar_operators_cannot_authorize_execution() -> None:
    path = (
        ROOT
        / "ipfs_accelerate_py/agent_supervisor/architecture_refactorer/refactor_operators.py"
    )
    assert probe_assign_bool(path, "OPERATOR_CAN_AUTHORIZE_EXECUTION") is False
    assert probe_assign_bool(path, "OPERATOR_CAN_REDUCE_GATES") is False
    assert probe_assign_bool(path, "OPERATOR_CAN_SELF_PROMOTE") is False
    assert (ROOT / "ipfs_accelerate_py/agent_supervisor/semantic_refactoring").exists() is False


def test_cst_and_libcst_are_typed_against_the_validation_environment(
    capabilities: list[dict[str, Any]],
    probes: Mapping[str, Any],
) -> None:
    row = next(item for item in capabilities if item["id"] == "cst_codemod")
    observations = {item["module"]: item["status"] for item in row["tool_observations"]}
    assert observations["libcst"] == probes["libcst"] == "typed_unavailable"
    assert observations["parso"] == "present_parse_helper"
    assert observations["asttokens"] == "present_parse_helper"
    assert probes["parso"] == "present"
    assert probes["asttokens"] == "present"
    assert probes["libcst"] != "present"
    assert row["evidence_status"] == "gap_confirmed"
    assert "must not be claimed usable" in row["notes"]


def test_predecessors_are_not_completion_authority(matrix: Mapping[str, Any]) -> None:
    overlap = _load_inventory("overlap_gap_matrix.json")
    recorded = {(row["program"], row["branch"], row["historical_revision"]) for row in matrix["predecessors"]}
    expected = {
        (row["program"], row["branch"], row["historical_revision"])
        for row in overlap["predecessors"]
    }
    assert recorded == expected
    for row in matrix["predecessors"]:
        assert row["current_tree_verification_required"] is True
        assert row["historical_branch_is_completion"] is False


def test_evidence_classes_stay_separate(
    matrix: Mapping[str, Any],
    capabilities: list[dict[str, Any]],
) -> None:
    allowed = set(matrix["evidence_classes"])
    observed = {row["evidence_class"] for row in capabilities}
    assert observed <= allowed
    assert "exact_static_fact" in observed
    assert "vector_candidate" in observed
    assert "runtime_observation" in observed
    vector_rows = [row for row in capabilities if row["evidence_class"] == "vector_candidate"]
    assert vector_rows and all(
        row["authority"] == "advisory=false authority" for row in vector_rows
    )
    assert all(row["required"] is True for row in capabilities)
    assert all(row["blocks_dependent_work"] is False for row in capabilities)


def test_independent_validation_does_not_import_the_matrix_to_probe(probes: Mapping[str, Any]) -> None:
    assert "capabilities" not in probes
    assert probes["libcst"] == probe_module_spec("libcst")
    assert MATRIX_PATH.is_file()
    # Reconstruction remains possible after discarding the JSON payload.
    reconstructed = probe_current_tree()
    assert reconstructed["interface_paths_present"] == probes["interface_paths_present"]
    assert reconstructed["happens_before_edges"] == probes["happens_before_edges"]
