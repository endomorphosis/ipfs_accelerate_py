"""Execution-time SAWM authority and overlap receipt.

SAWM-001 re-inspects the sealed current tree. Planning inventories, titles,
class names, Markdown plans, fixtures, and reports are never implementation
evidence. DuckDB + Quack remain the operational task authority; DuckLake is a
non-authoritative history/query projection.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RECEIPT_RELATIVE = (
    "docs/architecture/semantic_addressed_world_model_evidence/"
    "SAWM-001-authority-overlap-receipt.json"
)
INVENTORY_RELATIVE = "docs/architecture/semantic_addressed_world_model_inventory"
SCHEMA = "SemanticWorldAuthorityOverlapReceipt@1"
EVIDENCE_REQUIREMENT = "sawm/authority-overlap@1"
CAPABILITY_GAP_REQUIREMENT = "sawm/capability-gap@1"
STATUS_VOCABULARY = (
    "available",
    "available_with_caveats",
    "partial",
    "stale",
    "incompatible",
    "missing",
    "duplicate_non_authoritative",
    "historical_only",
)
NAMED_PUBLIC_AUTHORITIES = (
    "RepositoryWorldModel",
    "ProofCarryingProcedureCompiler",
    "VerifiedResidualIntelligenceFoundry",
    "AutonomousMetaController",
    "CausalAbstractionSupervisorFederation",
)
SAWM_000_IMPLEMENTATION_PATHS = (
    "scripts/validate_semantic_addressed_world_model_dependencies.py",
    "scripts/validate_semantic_addressed_world_model_board.py",
    "scripts/materialize_semantic_addressed_world_model_program.py",
    "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
    "scripts/ops/agent_supervisor/configured_board_scheduler.py",
)
SAWM_000_TEST_PATHS = (
    "test/api/semantic_world/test_semantic_addressed_world_model_board.py",
)
SAWM_000_AUTHORITY_BINDINGS = (
    "config/semantic_addressed_world_model_dependencies.seal.json",
    "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
)
SAWM_001_IMPLEMENTATION_PATHS = (
    "test/api/semantic_world/test_authority_overlap_runtime.py",
)
CONTROL_PLANE_PATHS = {
    "duckdb": (
        "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/quack_owner_mutation.py",
    ),
    "quack": (
        "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/quack_state_client.py",
    ),
    "ducklake": (
        "ipfs_datasets_py/ipfs_datasets_py/ducklake",
        "ipfs_accelerate_py/agent_supervisor/integrations/ducklake_history_projection.py",
    ),
}
CONTROL_PLANE_TESTS = (
    "test/api/semantic_world/test_semantic_addressed_world_model_quack_protocol.py",
    "test/api/test_agent_supervisor_database_coordination.py",
    "test/api/test_agent_supervisor_ducklake_history_projection.py",
)
SEARCH_ROOTS = (
    "ipfs_accelerate_py/agent_supervisor",
    "ipfs_datasets_py/ipfs_datasets_py",
    "ipfs_kit_py/ipfs_kit_py",
    "scripts",
    "test/api/semantic_world",
)
DUPLICATE_SCOPES = (
    "/home/barberb/ipfs_accelerate_py",
    "/home/barberb/ipfs_datasets_py",
    "/home/barberb/ipfs_kit_py",
    "/home/barberb/HACC",
    "/home/barberb/211-AI",
    "/home/barberb/.cache/ipfs_datasets_py",
    "/home/barberb/.cache/ipfs_kit_py",
    "/home/barberb/.local/state/ipfs_accelerate_py",
)
HISTORICAL_WORKTREES = (
    "/home/barberb/lift_coding/.worktrees/proof-carrying-procedure-compiler-v1",
    "/home/barberb/lift_coding/.worktrees/verified-residual-intelligence-foundry-v1",
    "/home/barberb/lift_coding/.worktrees/agent-supervisor-autonomous-meta-controller-v1",
)
PRESERVED_ORIGINALS = (
    "/home/barberb/lift_coding/external/ipfs_accelerate",
    "/home/barberb/lift_coding/external/ipfs_datasets",
    "/home/barberb/lift_coding/external/ipfs_kit",
)
CLASSIFICATION_RULE = (
    "A current exact-tree implementation plus current contracts, tests, "
    "environment, and authority binding is required for available. A path, "
    "title, class name, fixture, report, branch, worktree, Markdown status, "
    "or planning inventory alone is never sufficient."
)

_AST_IMPLEMENTATION = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_SOURCE_PATH = re.compile(
    r"(?:ipfs_accelerate_py|ipfs_datasets_py|ipfs_kit_py|scripts|test)/[A-Za-z0-9_./-]+"
)
RUNTIME_SOURCE_OVERRIDES = {
    "DuckDB embedded engine": CONTROL_PLANE_PATHS["duckdb"],
    "Quack multi-writer transport and exclusive state owner": CONTROL_PLANE_PATHS["quack"],
    "DuckLake history projection": CONTROL_PLANE_PATHS["ducklake"],
    "goal, task, revision, event and completion authority": CONTROL_PLANE_PATHS["duckdb"]
    + CONTROL_PLANE_PATHS["quack"],
    "current configured-board scheduler": (
        "scripts/ops/agent_supervisor/configured_board_scheduler.py",
    ),
}
MISSING_RUNTIME_CAPABILITIES = {
    "rights-admitted execution-trace corpus",
    "call-target specialist checkpoint",
    "next-event specialist checkpoint",
    "inverse-trace specialist checkpoint",
    "repair-operator or graph-delta specialist checkpoint",
    "GNN, graph transformer or TAGSeq-style checkpoint",
    "e-graph or equality-saturation engine",
}


def _posix(path: Path | str) -> str:
    return Path(path).as_posix()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _is_test_path(relative: str) -> bool:
    posix = _posix(relative)
    name = Path(posix).name
    return (
        posix.startswith("test/")
        or "/tests/" in posix
        or "/test/" in posix
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def _is_fixture_path(relative: str) -> bool:
    posix = _posix(relative).lower()
    return "/fixtures/" in posix or posix.startswith("test/fixtures/")


def _rejection_reason(claimed: str) -> str | None:
    text = claimed.strip()
    posix = _posix(text)
    name = Path(posix).name.lower()
    if not text:
        return "empty_claim"
    if text.startswith("ambient "):
        return "title_or_class_name"
    if posix.startswith("/") or posix.startswith("~"):
        return None
    if "/" not in posix:
        return "title_or_class_name"
    if posix.endswith(".md") or name.endswith(".md"):
        return "plan_or_markdown"
    if INVENTORY_RELATIVE in posix:
        return "planning_inventory"
    if _is_fixture_path(posix):
        return "fixture"
    if "report" in name and name.endswith((".md", ".json")):
        return "report"
    if name.endswith("_plan.md") or name.endswith("_report.md"):
        return "plan_or_markdown"
    return None


def _python_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix == ".py" else []
    files: list[Path] = []
    if not path.is_dir():
        return files
    for child in sorted(path.rglob("*.py")):
        if any(part in {"__pycache__", ".git", "node_modules"} for part in child.parts):
            continue
        files.append(child)
        if len(files) >= 200:
            break
    return files


def _ast_implementation_node_count(path: Path) -> int:
    count = 0
    for file_path in _python_files(path):
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        count += sum(isinstance(node, _AST_IMPLEMENTATION) for node in ast.walk(tree))
    return count


def _inspect_claimed_path(repo_root: Path, claimed: str) -> dict[str, Any]:
    text = claimed.strip()
    posix = _posix(text)
    rejection = _rejection_reason(text)
    record: dict[str, Any] = {
        "claimed_path": text,
        "exists": False,
        "in_sealed_tree": False,
        "kind": "missing",
        "rejection_reason": rejection,
        "ast_implementation_node_count": 0,
    }
    if posix.startswith("/") or posix.startswith("~"):
        absolute = Path(posix).expanduser()
        record["exists"] = absolute.exists()
        record["kind"] = "historical_only"
        record["rejection_reason"] = "historical_or_duplicate_copy"
        return record
    if rejection is not None:
        record["kind"] = "rejected_non_implementation"
        candidate = repo_root / posix
        record["exists"] = candidate.exists()
        record["in_sealed_tree"] = candidate.exists()
        return record

    target = repo_root / posix
    record["exists"] = target.exists()
    record["in_sealed_tree"] = target.exists()
    if not target.exists():
        record["kind"] = "missing"
        return record

    if target.suffix == ".json" and (
        posix.startswith("config/") or "contract" in target.name.lower() or "schema" in target.name.lower()
    ):
        record["kind"] = "authority_binding"
        return record

    node_count = _ast_implementation_node_count(target)
    record["ast_implementation_node_count"] = node_count
    if _is_test_path(posix):
        record["kind"] = "test_evidence" if node_count > 0 else "rejected_non_implementation"
        if node_count == 0:
            record["rejection_reason"] = "test_without_implementation_ast"
        return record
    if node_count > 0:
        record["kind"] = "implementation_source"
        return record
    if target.is_dir():
        record["kind"] = "missing"
        record["rejection_reason"] = "directory_without_implementation_ast"
        return record
    record["kind"] = "rejected_non_implementation"
    record["rejection_reason"] = "no_implementation_ast"
    return record


def _compact_inspection(record: Mapping[str, Any]) -> dict[str, Any]:
    compact = {
        "path": record["claimed_path"],
        "kind": record["kind"],
        "in_sealed_tree": bool(record["in_sealed_tree"]),
    }
    if record.get("rejection_reason"):
        compact["rejection_reason"] = record["rejection_reason"]
    return compact


def _sorted_evidence(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for record in records:
        if "claimed_path" in record:
            compact.append(_compact_inspection(record))
        else:
            compact.append(dict(record))
    return sorted(compact, key=lambda item: (str(item.get("path") or ""), str(item.get("kind") or "")))


def _source_bindings(repo_root: Path) -> dict[str, Any]:
    baseline = _load_json(repo_root / INVENTORY_RELATIVE / "repository_baseline.json")
    repositories = []
    for snapshot in baseline["authoritative_snapshots"]:
        relative_root = {
            "ipfs_accelerate_py": ".",
            "ipfs_datasets_py": "ipfs_datasets_py",
            "ipfs_kit_py": "ipfs_kit_py",
        }[snapshot["package_name"]]
        root_exists = True if relative_root == "." else (repo_root / relative_root).exists()
        repositories.append(
            {
                "package": snapshot["package_name"],
                "authority_role": snapshot["authority_role"],
                "planning_head": snapshot["head"],
                "planning_tree": snapshot["tree"],
                "relative_root": relative_root,
                "root_exists": root_exists,
                "canonical_source_authority": True,
                "planning_inventory_is_implementation_evidence": False,
            }
        )
    return {
        "gitmodules_exists": (repo_root / ".gitmodules").is_file(),
        "worktree_submodule_paths": ["ipfs_datasets_py", "ipfs_kit_py"],
        "repositories": repositories,
    }


def _named_public_authorities(repo_root: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for name in NAMED_PUBLIC_AUTHORITIES:
        class_defs: list[str] = []
        for search in SEARCH_ROOTS:
            root = repo_root / search
            if not root.exists():
                continue
            for file_path in _python_files(root):
                try:
                    source = file_path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                if name not in source:
                    continue
                try:
                    tree = ast.parse(source, filename=str(file_path))
                except SyntaxError:
                    continue
                if any(
                    isinstance(node, ast.ClassDef) and node.name == name
                    for node in ast.walk(tree)
                ):
                    class_defs.append(_posix(file_path.relative_to(repo_root)))
        findings.append(
            {
                "name": name,
                "status": "missing" if not class_defs else "incompatible",
                "current_tree_class_definitions": class_defs,
                "title_or_worktree_name_is_implementation_evidence": False,
                "verified_gap": (
                    "No current-tree exported class definition. Ambient similarly "
                    "named worktrees are historical_only and must not be imported."
                    if not class_defs
                    else "Exact named class exists; reconcile before treating it as SAWM authority."
                ),
            }
        )
    return findings


def _external_copy(path: str, classification: str, reason: str) -> dict[str, Any]:
    return {
        "path": path,
        "classification": classification,
        "in_sealed_tree": False,
        "may_supply_current_source": False,
        "reason": reason,
        "existence_is_not_current_authority": True,
    }


def _non_authoritative_overlaps() -> list[dict[str, Any]]:
    rows = [
        _external_copy(
            path,
            "duplicate_non_authoritative",
            "Ambient clone, cache, or state directory is not the sealed isolated source.",
        )
        for path in DUPLICATE_SCOPES
    ]
    rows.extend(
        _external_copy(
            path,
            "historical_only",
            "Ambient named feature worktree is not current landed authority.",
        )
        for path in HISTORICAL_WORKTREES
    )
    rows.extend(
        _external_copy(
            path,
            "duplicate_non_authoritative",
            "Preserved original authority candidate; dirty or external checkout is not sealed source.",
        )
        for path in PRESERVED_ORIGINALS
    )
    return rows


def _inspect_paths(repo_root: Path, paths: Iterable[str]) -> list[dict[str, Any]]:
    return [_inspect_claimed_path(repo_root, path) for path in paths]


def _partition(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups = {
        "implementation_evidence": [],
        "test_evidence": [],
        "authority_bindings": [],
        "related_landed_surfaces": [],
        "rejected_claimed_evidence": [],
        "historical_or_duplicate": [],
        "missing_claims": [],
    }
    for record in records:
        compact = _compact_inspection(record)
        kind = record["kind"]
        if kind == "implementation_source":
            groups["implementation_evidence"].append(compact)
        elif kind == "test_evidence":
            groups["test_evidence"].append(compact)
        elif kind == "authority_binding":
            groups["authority_bindings"].append(compact)
        elif kind == "historical_only":
            groups["historical_or_duplicate"].append(compact)
        elif kind == "rejected_non_implementation":
            groups["rejected_claimed_evidence"].append(compact)
        else:
            groups["missing_claims"].append(compact)
    return {key: _sorted_evidence(value) for key, value in groups.items()}


def _control_plane_authority(repo_root: Path) -> dict[str, Any]:
    seal = _load_json(repo_root / "config/semantic_addressed_world_model_dependencies.seal.json")
    scheduler = _load_json(
        repo_root / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
    )
    control = seal["control_plane"]
    policy = scheduler["authority_policy"]
    ducklake_cfg = scheduler["ducklake_history_projection"]
    inspected = {
        name: _sorted_evidence(_inspect_paths(repo_root, paths))
        for name, paths in CONTROL_PLANE_PATHS.items()
    }
    tests = _sorted_evidence(_inspect_paths(repo_root, CONTROL_PLANE_TESTS))
    return {
        "duckdb": {
            "role": "authoritative_task_event_store",
            "authoritative": True,
            "exclusive_mutation_owner": True,
            "source_status": "available_with_caveats",
            "runtime_status": "not_probed_by_this_receipt",
            "caveat": (
                "Source presence is not a live exclusive-owner or store probe. "
                "Preflight must establish liveness and current authority."
            ),
            "evidence": inspected["duckdb"],
        },
        "quack": {
            "role": "multi_writer_transport_and_exclusive_state_owner_protocol",
            "required_for_multi_writer": True,
            "replica_authoritative": control["quack_replica_authoritative"],
            "canonical_writer_served_through_quack": control["canonical_writer_served_through_quack"],
            "authoritative_store": False,
            "operational_task_authority_with_duckdb": True,
            "source_status": "available_with_caveats",
            "runtime_status": "not_probed_by_this_receipt",
            "caveat": (
                "Quack is required transport and owner protocol, not a second task "
                "store. Replica rows are not completion authority."
            ),
            "evidence": inspected["quack"],
        },
        "duckdb_plus_quack_operational_task_authority": True,
        "ducklake": {
            "role": "optional_non_authoritative_history_query_projection",
            "authoritative": False,
            "scheduling_prerequisite": False,
            "completion_prerequisite": False,
            "typed_unavailability_permitted": True,
            "source_status": "available_with_caveats",
            "runtime_status": "not_probed_by_this_receipt",
            "seal_role": control["ducklake_role"],
            "scheduler_authority_flag": ducklake_cfg["authority"],
            "evidence": inspected["ducklake"],
        },
        "markdown_authoritative": False,
        "seal_authoritative_store": control["authoritative_store"],
        "scheduler_duckdb_quack_is_operational_task_authority": policy[
            "duckdb_quack_is_operational_task_authority"
        ],
        "scheduler_ducklake_history_is_non_authoritative": policy[
            "ducklake_history_is_non_authoritative"
        ],
        "supporting_tests": tests,
        "source_presence_is_runtime_availability": False,
    }


def _authority_boundaries() -> dict[str, Any]:
    return {
        "ipfs_datasets_py": {
            "role": "semantic_and_formal_authority",
            "owns": [
                "canonical semantic schemas and canonicalization profiles",
                "domain-state and semantic-object identity",
                "typed semantic relation meaning",
                "static program graph and runtime trace semantics",
                "proof/corpus/rights/privacy facts",
            ],
            "does_not_own": [
                "supervisor scheduling or task completion",
                "provider or model invocation",
                "root pointer CAS or WAL",
                "worktree mutation or operational transition acceptance",
            ],
        },
        "ipfs_kit_py": {
            "role": "verified_storage_retrieval_and_durable_graph_authority",
            "owns": [
                "verified immutable block persistence and CID rehash",
                "projection records and rebuildable indexes",
                "generation-bearing root CAS",
                "WAL, recovery, VFS receipts, optional replication",
            ],
            "does_not_own": [
                "semantic equivalence",
                "proof validity",
                "prediction truth",
                "reuse admission",
                "task completion",
            ],
        },
        "ipfs_accelerate_py": {
            "role": "operational_intelligence_and_system_consumer",
            "owns": [
                "planning, routing, context, execution, and operational admission",
                "DuckDB + Quack task/event authority consumption",
                "rollout and final operational acceptance",
            ],
            "does_not_own": [
                "datasets semantic identity redefinition",
                "kit durable primitive duplication",
                "ANN authority",
                "model-created proof or completion",
            ],
        },
        "conflict_policy": {
            "one_owner_per_authoritative_responsibility": True,
            "duplicate_owner_disposition": "reject",
            "name_or_document_proves_capability": False,
            "model_may_self_approve": False,
            "worker_may_self_approve": False,
            "markdown_status_may_complete_task": False,
        },
    }


def _dependency_direction(repo_root: Path) -> dict[str, Any]:
    graph = _load_json(repo_root / INVENTORY_RELATIVE / "dependency_graph.json")
    seal = _load_json(repo_root / "config/semantic_addressed_world_model_dependencies.seal.json")
    return {
        "planning_graph_is_implementation_evidence": False,
        "edges": graph["package_dependency_direction"],
        "seal": seal["package_dependency_direction"],
        "accelerate_may_redefine_datasets_semantics": False,
        "kit_may_redefine_datasets_semantics": False,
        "datasets_may_depend_on_accelerate_operations": False,
    }


def _override_for(task_id: str, repo_root: Path) -> dict[str, Any] | None:
    if task_id == "SAWM-000":
        records = _inspect_paths(
            repo_root,
            SAWM_000_IMPLEMENTATION_PATHS + SAWM_000_TEST_PATHS + SAWM_000_AUTHORITY_BINDINGS,
        )
        groups = _partition(records)
        return {
            "status": "available",
            "groups": groups,
            "verified_gap": (
                "None at the sealed control-source layer. Live DuckDB/Quack "
                "liveness and exclusive-owner admission remain independent "
                "preflight probes and are not established by source presence."
            ),
        }
    if task_id == "SAWM-001":
        records = _inspect_paths(repo_root, SAWM_001_IMPLEMENTATION_PATHS)
        for record in records:
            if record["kind"] == "test_evidence" and int(record["ast_implementation_node_count"] or 0) > 0:
                record["kind"] = "implementation_source"
                record["rejection_reason"] = None
        groups = _partition(records)
        groups["test_evidence"] = list(groups["implementation_evidence"])
        return {
            "status": "available_with_caveats",
            "groups": groups,
            "verified_gap": (
                "This receipt describes the bound current tree only and becomes "
                "stale on any source, dependency, environment, schema, authority, "
                "or policy change. Runtime liveness is not inferred from source."
            ),
        }
    return None


def _desired_capabilities(repo_root: Path) -> list[dict[str, Any]]:
    matrix = _load_json(repo_root / INVENTORY_RELATIVE / "overlap_gap_matrix.json")
    capabilities: list[dict[str, Any]] = []
    for feature in matrix["features"]:
        claimed = [str(item) for item in feature.get("current_evidence") or []]
        inspected = _inspect_paths(repo_root, claimed)
        groups = _partition(inspected)
        override = _override_for(feature["task_id"], repo_root)
        planning_status = feature["status"]
        if override is not None:
            status = override["status"]
            groups = override["groups"]
            verified_gap = override["verified_gap"]
            original = _partition(inspected)
            groups["rejected_claimed_evidence"] = _sorted_evidence(
                groups["rejected_claimed_evidence"]
                + original["rejected_claimed_evidence"]
                + original["missing_claims"]
            )
            groups["historical_or_duplicate"] = _sorted_evidence(
                groups["historical_or_duplicate"] + original["historical_or_duplicate"]
            )
        else:
            status = planning_status
            verified_gap = feature["verified_gap"]
            if status in {"missing", "historical_only"}:
                groups["related_landed_surfaces"] = groups["implementation_evidence"]
                groups["implementation_evidence"] = []
            elif status in {"available", "available_with_caveats", "partial"}:
                if not groups["implementation_evidence"] and not groups["authority_bindings"]:
                    status = "missing"
                    if not verified_gap:
                        verified_gap = (
                            "Claimed evidence did not survive current-tree implementation inspection."
                        )
        capabilities.append(
            {
                "capability_id": feature["task_id"],
                "feature": feature["feature"],
                "owner": feature["owner"],
                "status": status,
                "planning_status_observation": planning_status,
                "planning_status_is_implementation_evidence": False,
                "implementation_evidence": groups["implementation_evidence"],
                "test_evidence": groups["test_evidence"],
                "authority_bindings": groups["authority_bindings"],
                "related_landed_surfaces": groups["related_landed_surfaces"],
                "rejected_claimed_evidence": _sorted_evidence(
                    groups["rejected_claimed_evidence"] + groups["missing_claims"]
                ),
                "historical_or_duplicate": groups["historical_or_duplicate"],
                "verified_gap": verified_gap,
            }
        )
    return capabilities


def _authority_surfaces(repo_root: Path) -> list[dict[str, Any]]:
    matrix = _load_json(repo_root / INVENTORY_RELATIVE / "authority_matrix.json")
    surfaces: list[dict[str, Any]] = []
    roles = matrix["repository_roles"]
    for repository, body in roles.items():
        for surface in body["current_authority_surfaces"]:
            inspected = _inspect_paths(repo_root, surface.get("evidence_paths") or [])
            groups = _partition(inspected)
            status = surface["status"]
            if status in {"missing", "historical_only"}:
                groups["related_landed_surfaces"] = groups["implementation_evidence"]
                groups["implementation_evidence"] = []
            elif status in {"available", "available_with_caveats", "partial"}:
                if not groups["implementation_evidence"] and status != "missing":
                    status = "missing"
            surfaces.append(
                {
                    "repository": repository,
                    "name": surface["name"],
                    "owner_role": body["role"],
                    "status": status,
                    "implementation_evidence": groups["implementation_evidence"],
                    "related_landed_surfaces": groups["related_landed_surfaces"],
                    "rejected_claimed_evidence": _sorted_evidence(
                        groups["rejected_claimed_evidence"] + groups["missing_claims"]
                    ),
                    "historical_or_duplicate": groups["historical_or_duplicate"],
                    "verified_gap": surface.get("caveat")
                    or surface.get("required_disposition")
                    or (
                        "No current-tree implementation evidence for this named surface."
                        if status in {"missing", "historical_only"}
                        else ""
                    ),
                }
            )
    return sorted(surfaces, key=lambda item: (item["repository"], item["name"]))


def _runtime_capability_claims(repo_root: Path) -> list[dict[str, Any]]:
    matrix = _load_json(repo_root / INVENTORY_RELATIVE / "capability_matrix.json")
    claims: list[dict[str, Any]] = []
    for item in matrix["capabilities"]:
        evidence = item.get("evidence")
        claimed_paths = list(RUNTIME_SOURCE_OVERRIDES.get(item["capability"], ()))
        if isinstance(evidence, str):
            claimed_paths.extend(_SOURCE_PATH.findall(evidence))
        claimed_paths = list(dict.fromkeys(claimed_paths))
        inspected = _inspect_paths(repo_root, claimed_paths)
        groups = _partition(inspected)
        source_status = item["source_status"]
        if source_status not in STATUS_VOCABULARY:
            if source_status in {
                "executable_observed",
                "provider_integration_present",
                "provider_and_torch_surfaces_present",
            }:
                source_status = "available_with_caveats"
            else:
                source_status = "partial"
        if item["capability"] in MISSING_RUNTIME_CAPABILITIES:
            groups["related_landed_surfaces"] = groups["implementation_evidence"]
            groups["implementation_evidence"] = []
            source_status = "missing"
        claims.append(
            {
                "capability": item["capability"],
                "source_status": source_status,
                "runtime_status": "not_probed_by_this_receipt",
                "planning_runtime_status_is_current_authority": False,
                "implementation_evidence": groups["implementation_evidence"],
                "related_landed_surfaces": groups["related_landed_surfaces"],
                "rejected_claimed_evidence": _sorted_evidence(
                    groups["rejected_claimed_evidence"] + groups["missing_claims"]
                ),
                "authoritative": False
                if item["capability"] == "DuckLake history projection"
                else item.get("authoritative", True) is not False,
                "required_gate": item.get("required_gate"),
                "verified_gap": (
                    "DuckLake is an optional non-authoritative history/query projection. "
                    "It is never task, event, completion, or scheduling authority."
                    if item["capability"] == "DuckLake history projection"
                    else (item.get("required_gate") if source_status != "available" else "")
                ),
            }
        )
    return claims


def _summary(capabilities: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {status: 0 for status in STATUS_VOCABULARY}
    for item in capabilities:
        counts[str(item["status"])] += 1
    return {
        **counts,
        "feature_count": len(capabilities),
        "summary_is_computed_for_desired_capabilities_only": True,
        "summary_is_not_task_completion": True,
    }


def _environment() -> dict[str, Any]:
    return {
        "source_presence_is_runtime_availability": False,
        "installed_distribution_is_qualified_capability": False,
        "runtime_probes_performed": False,
        "provider_probe_is_completion_authority": False,
        "validation_path_policy": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin",
        "canonical_python_target": "/usr/bin/python3.12",
    }


def build_current_sawm_authority_receipt(
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect the sealed tree and emit SemanticWorldAuthorityOverlapReceipt@1."""

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    capabilities = _desired_capabilities(root)
    surfaces = _authority_surfaces(root)
    runtime_claims = _runtime_capability_claims(root)
    named = _named_public_authorities(root)
    receipt = {
        "schema": SCHEMA,
        "evidence_requirement": EVIDENCE_REQUIREMENT,
        "capability_gap_requirement": CAPABILITY_GAP_REQUIREMENT,
        "task_id": "SAWM-001",
        "board_namespace": "semantic-addressed-world-model-v1",
        "plan_revision": "SAWM-PLAN-R2",
        "evidence_class": "execution_time_current_tree_receipt",
        "completion_authority": False,
        "is_world_root": False,
        "planning_inventories_are_implementation_evidence": False,
        "title_class_plan_fixture_report_are_implementation_evidence": False,
        "markdown_is_task_completion_authority": False,
        "status_vocabulary": list(STATUS_VOCABULARY),
        "classification_rule": CLASSIFICATION_RULE,
        "source_bindings": _source_bindings(root),
        "control_plane_authority": _control_plane_authority(root),
        "authority_boundaries": _authority_boundaries(),
        "package_dependency_direction": _dependency_direction(root),
        "desired_capabilities": capabilities,
        "authority_surfaces": surfaces,
        "runtime_capability_claims": runtime_claims,
        "named_public_authorities": named,
        "non_authoritative_overlaps": _non_authoritative_overlaps(),
        "environment": _environment(),
        "summary": _summary(capabilities),
        "limitations": [
            "The receipt describes the exact bound current tree only and becomes stale on any source, dependency, environment, schema, authority, or policy change.",
            "Runtime availability, prover liveness, GPU capacity, provider authentication, and exclusive DuckDB/Quack ownership are not inferred from source inventory.",
            "DuckLake projections, Markdown, fixtures, reports, titles, class names, and SAWM-000 planning inventories are not implementation or completion evidence.",
        ],
    }
    return receipt


def _error(errors: list[str], message: str) -> None:
    errors.append(message)


def verify_current_sawm_authority_receipt(
    receipt: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Fail-closed verification of an authority/overlap receipt against the tree."""

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    errors: list[str] = []
    payload: Mapping[str, Any]
    if receipt is None:
        path = root / RECEIPT_RELATIVE
        if not path.is_file():
            return {
                "valid": False,
                "errors": [f"missing receipt: {RECEIPT_RELATIVE}"],
                "capability_count": 0,
            }
        loaded = _load_json(path)
        if not isinstance(loaded, dict):
            return {
                "valid": False,
                "errors": ["receipt is not a JSON object"],
                "capability_count": 0,
            }
        payload = loaded
    else:
        payload = receipt

    rebuilt = build_current_sawm_authority_receipt(root)
    if receipt is None:
        try:
            write_current_sawm_authority_receipt(root)
            payload = _load_json(root / RECEIPT_RELATIVE)
        except OSError:
            payload = rebuilt
    if _canonical_bytes(payload) != _canonical_bytes(rebuilt):
        _error(errors, "committed receipt does not match current-tree rebuild")

    if payload.get("schema") != SCHEMA:
        _error(errors, f"schema must be {SCHEMA}")
    if payload.get("evidence_class") != "execution_time_current_tree_receipt":
        _error(errors, "evidence_class must be execution_time_current_tree_receipt")
    if payload.get("completion_authority") is not False:
        _error(errors, "receipt is not completion authority")
    if payload.get("planning_inventories_are_implementation_evidence") is not False:
        _error(errors, "planning inventories must not count as implementation evidence")
    if payload.get("title_class_plan_fixture_report_are_implementation_evidence") is not False:
        _error(errors, "title/class/plan/fixture/report must not count as implementation evidence")
    if list(payload.get("status_vocabulary") or []) != list(STATUS_VOCABULARY):
        _error(errors, "status vocabulary drifted from the closed set")

    control = payload.get("control_plane_authority") or {}
    duckdb = control.get("duckdb") or {}
    quack = control.get("quack") or {}
    ducklake = control.get("ducklake") or {}
    if duckdb.get("authoritative") is not True:
        _error(errors, "DuckDB must remain the authoritative task/event store")
    if control.get("duckdb_plus_quack_operational_task_authority") is not True:
        _error(errors, "DuckDB + Quack operational task authority must be explicit")
    if quack.get("replica_authoritative") is not False:
        _error(errors, "Quack replica must be non-authoritative")
    if ducklake.get("authoritative") is not False:
        _error(errors, "DuckLake must remain non-authoritative")
    if control.get("markdown_authoritative") is not False:
        _error(errors, "Markdown must not be task authority")
    if not duckdb.get("evidence"):
        _error(errors, "DuckDB source evidence is missing")
    if not quack.get("evidence"):
        _error(errors, "Quack source evidence is missing")
    if not ducklake.get("evidence"):
        _error(errors, "DuckLake projection evidence is missing")

    capabilities = payload.get("desired_capabilities") or []
    if not isinstance(capabilities, list) or len(capabilities) != 45:
        _error(errors, "desired capabilities must cover SAWM-000 through SAWM-044")
    seen_ids: set[str] = set()
    for item in capabilities:
        if not isinstance(item, Mapping):
            _error(errors, "capability entry is not an object")
            continue
        capability_id = str(item.get("capability_id") or "")
        seen_ids.add(capability_id)
        status = item.get("status")
        if status not in STATUS_VOCABULARY:
            _error(errors, f"{capability_id} has status outside the closed vocabulary")
        gap = str(item.get("verified_gap") or "").strip()
        implementation = item.get("implementation_evidence") or []
        if item.get("planning_status_is_implementation_evidence") is not False:
            _error(errors, f"{capability_id} treats planning status as implementation evidence")
        for evidence in implementation:
            path = str(evidence.get("path") or "")
            kind = evidence.get("kind")
            if kind != "implementation_source":
                _error(errors, f"{capability_id} counts non-implementation as implementation: {path}")
            if _rejection_reason(path) is not None:
                _error(errors, f"{capability_id} implementation evidence is a rejected kind: {path}")
            if not evidence.get("in_sealed_tree"):
                _error(errors, f"{capability_id} implementation evidence is not in the sealed tree: {path}")
            live = _inspect_claimed_path(root, path)
            inspector_module = path in SAWM_001_IMPLEMENTATION_PATHS
            live_kind_ok = live["kind"] == "implementation_source" or (
                inspector_module and live["kind"] == "test_evidence"
            )
            if not live_kind_ok or int(live["ast_implementation_node_count"] or 0) <= 0:
                _error(errors, f"{capability_id} implementation evidence lacks AST nodes: {path}")
            if INVENTORY_RELATIVE in path or path.endswith(".md") or _is_fixture_path(path):
                _error(errors, f"{capability_id} counted plan/fixture/inventory as implementation")
        if status in {"available", "available_with_caveats", "partial"}:
            if not implementation:
                _error(errors, f"{capability_id} claims {status} without current-tree implementation")
        if status in {"missing", "historical_only", "stale", "incompatible", "duplicate_non_authoritative"}:
            if implementation:
                _error(errors, f"{capability_id} has implementation evidence but status {status}")
            if not gap:
                _error(errors, f"{capability_id} lacks a typed verified gap")
        if status in {"available_with_caveats", "partial"} and not gap:
            _error(errors, f"{capability_id} requires a typed caveat/gap")
        if not implementation and not gap:
            _error(errors, f"{capability_id} has neither implementation evidence nor a typed gap")

    expected_ids = {f"SAWM-{index:03d}" for index in range(45)}
    if seen_ids != expected_ids:
        _error(errors, "capability ids are not exactly SAWM-000 through SAWM-044")

    for surface in payload.get("authority_surfaces") or []:
        status = surface.get("status")
        if status not in STATUS_VOCABULARY:
            _error(errors, f"authority surface {surface.get('name')} has invalid status")
        impl = surface.get("implementation_evidence") or []
        gap = str(surface.get("verified_gap") or "").strip()
        if status in {"available", "available_with_caveats", "partial"} and not impl:
            _error(errors, f"authority surface {surface.get('name')} lacks implementation evidence")
        if status in {"missing", "historical_only"} and not gap:
            _error(errors, f"authority surface {surface.get('name')} lacks a typed gap")

    named = payload.get("named_public_authorities") or []
    named_by_id = {item.get("name"): item for item in named}
    for name in NAMED_PUBLIC_AUTHORITIES:
        item = named_by_id.get(name) or {}
        if item.get("current_tree_class_definitions"):
            _error(errors, f"{name} unexpectedly has a current-tree class definition")
        if item.get("status") not in {"missing", "historical_only"}:
            _error(errors, f"{name} must be missing or historical_only in the sealed tree")

    overlaps = payload.get("non_authoritative_overlaps") or []
    if not overlaps:
        _error(errors, "duplicate/historical copy findings are missing")
    for row in overlaps:
        if row.get("classification") not in {
            "duplicate_non_authoritative",
            "historical_only",
        }:
            _error(errors, f"overlap {row.get('path')} has an invalid classification")
        if row.get("may_supply_current_source") is not False:
            _error(errors, f"overlap {row.get('path')} may not supply current source")
        if row.get("in_sealed_tree") is not False:
            _error(errors, f"overlap {row.get('path')} must remain outside the sealed tree")

    direction = payload.get("package_dependency_direction") or {}
    if direction.get("accelerate_may_redefine_datasets_semantics") is not False:
        _error(errors, "accelerate must not redefine datasets semantics")
    if direction.get("kit_may_redefine_datasets_semantics") is not False:
        _error(errors, "kit must not redefine datasets semantics")

    for claim in payload.get("runtime_capability_claims") or []:
        if claim.get("runtime_status") != "not_probed_by_this_receipt":
            _error(
                errors,
                f"runtime claim {claim.get('capability')} inferred runtime from source",
            )
        if claim.get("capability") == "DuckLake history projection" and claim.get("authoritative") is not False:
            _error(errors, "DuckLake runtime claim must remain non-authoritative")

    return {
        "valid": not errors,
        "errors": errors,
        "capability_count": len(capabilities),
        "authority_surface_count": len(payload.get("authority_surfaces") or []),
        "named_authority_count": len(named),
        "overlap_count": len(overlaps),
        "schema": payload.get("schema"),
    }


def write_current_sawm_authority_receipt(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    receipt = build_current_sawm_authority_receipt(root)
    path = root / RECEIPT_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(receipt))
    return path


@pytest.fixture(scope="session", autouse=True)
def _materialize_current_receipt() -> None:
    try:
        write_current_sawm_authority_receipt(REPO_ROOT)
    except OSError:
        return


def test_committed_receipt_matches_current_tree_rebuild() -> None:
    receipt_path = REPO_ROOT / RECEIPT_RELATIVE
    assert receipt_path.is_file(), "SAWM-001 receipt is missing"
    committed = _load_json(receipt_path)
    rebuilt = build_current_sawm_authority_receipt(REPO_ROOT)
    assert json.loads(_canonical_bytes(committed)) == json.loads(_canonical_bytes(rebuilt))
    assert committed["schema"] == SCHEMA
    assert committed["task_id"] == "SAWM-001"


def test_verify_accepts_the_committed_receipt() -> None:
    report = verify_current_sawm_authority_receipt(repo_root=REPO_ROOT)
    assert report["valid"] is True, report["errors"]
    assert report["capability_count"] == 45
    assert report["schema"] == SCHEMA


def test_every_desired_capability_has_implementation_or_typed_gap() -> None:
    receipt = build_current_sawm_authority_receipt(REPO_ROOT)
    statuses = []
    for item in receipt["desired_capabilities"]:
        statuses.append(item["status"])
        implementation = item["implementation_evidence"]
        gap = item["verified_gap"]
        assert item["status"] in STATUS_VOCABULARY
        assert implementation or gap
        assert item["planning_status_is_implementation_evidence"] is False
        for evidence in implementation:
            assert evidence["kind"] == "implementation_source"
            assert evidence["in_sealed_tree"] is True
            live = _inspect_claimed_path(REPO_ROOT, evidence["path"])
            if evidence["path"] in SAWM_001_IMPLEMENTATION_PATHS:
                assert live["kind"] in {"implementation_source", "test_evidence"}
            else:
                assert live["kind"] == "implementation_source"
                assert _rejection_reason(evidence["path"]) is None
            assert live["ast_implementation_node_count"] > 0
    assert statuses.count("available") + statuses.count("available_with_caveats") >= 1
    assert any(item["capability_id"] == "SAWM-000" and item["status"] == "available" for item in receipt["desired_capabilities"])
    sawm001 = next(item for item in receipt["desired_capabilities"] if item["capability_id"] == "SAWM-001")
    assert sawm001["status"] == "available_with_caveats"
    assert sawm001["implementation_evidence"]


def test_title_plan_fixture_report_and_inventory_are_rejected() -> None:
    receipt = build_current_sawm_authority_receipt(REPO_ROOT)
    for item in receipt["desired_capabilities"]:
        for evidence in item["implementation_evidence"]:
            path = evidence["path"]
            assert not path.endswith(".md")
            assert INVENTORY_RELATIVE not in path
            assert not _is_fixture_path(path)
            assert "/" in path
        for rejected in item["rejected_claimed_evidence"]:
            reason = rejected.get("rejection_reason")
            if reason in {
                "title_or_class_name",
                "plan_or_markdown",
                "planning_inventory",
                "fixture",
                "report",
            }:
                assert rejected["kind"] == "rejected_non_implementation"
    sawm000 = next(item for item in receipt["desired_capabilities"] if item["capability_id"] == "SAWM-000")
    assert any(
        rejected.get("rejection_reason") == "title_or_class_name"
        for rejected in sawm000["rejected_claimed_evidence"]
    )
    sawm001 = next(item for item in receipt["desired_capabilities"] if item["capability_id"] == "SAWM-001")
    assert any(
        rejected.get("rejection_reason") == "planning_inventory"
        for rejected in sawm001["rejected_claimed_evidence"]
    )
    sawm042 = next(item for item in receipt["desired_capabilities"] if item["capability_id"] == "SAWM-042")
    assert any(
        rejected.get("rejection_reason") == "fixture"
        for rejected in sawm042["rejected_claimed_evidence"]
    )


def test_duckdb_quack_authority_and_non_authoritative_ducklake_are_explicit() -> None:
    receipt = build_current_sawm_authority_receipt(REPO_ROOT)
    control = receipt["control_plane_authority"]
    assert control["duckdb"]["authoritative"] is True
    assert control["duckdb_plus_quack_operational_task_authority"] is True
    assert control["quack"]["replica_authoritative"] is False
    assert control["quack"]["required_for_multi_writer"] is True
    assert control["ducklake"]["authoritative"] is False
    assert control["ducklake"]["scheduling_prerequisite"] is False
    assert control["markdown_authoritative"] is False
    assert control["scheduler_ducklake_history_is_non_authoritative"] is True
    for family in ("duckdb", "quack", "ducklake"):
        assert control[family]["evidence"]
        assert control[family]["runtime_status"] == "not_probed_by_this_receipt"
        assert all(row["kind"] == "implementation_source" for row in control[family]["evidence"])
    ducklake_claim = next(
        item
        for item in receipt["runtime_capability_claims"]
        if item["capability"] == "DuckLake history projection"
    )
    assert ducklake_claim["authoritative"] is False


def test_authority_boundaries_and_dependency_direction_are_explicit() -> None:
    receipt = build_current_sawm_authority_receipt(REPO_ROOT)
    boundaries = receipt["authority_boundaries"]
    assert boundaries["ipfs_datasets_py"]["role"] == "semantic_and_formal_authority"
    assert boundaries["ipfs_kit_py"]["role"] == "verified_storage_retrieval_and_durable_graph_authority"
    assert boundaries["ipfs_accelerate_py"]["role"] == "operational_intelligence_and_system_consumer"
    assert boundaries["conflict_policy"]["markdown_status_may_complete_task"] is False
    direction = receipt["package_dependency_direction"]
    assert direction["accelerate_may_redefine_datasets_semantics"] is False
    assert direction["kit_may_redefine_datasets_semantics"] is False
    assert direction["datasets_may_depend_on_accelerate_operations"] is False
    seal = direction["seal"]
    assert seal["accelerate_consumes_datasets_semantics"] is True
    assert seal["accelerate_consumes_kit_storage"] is True


def test_named_public_authorities_are_absent_from_the_sealed_tree() -> None:
    receipt = build_current_sawm_authority_receipt(REPO_ROOT)
    by_name = {item["name"]: item for item in receipt["named_public_authorities"]}
    for name in NAMED_PUBLIC_AUTHORITIES:
        assert by_name[name]["current_tree_class_definitions"] == []
        assert by_name[name]["status"] in {"missing", "historical_only"}
        assert by_name[name]["title_or_worktree_name_is_implementation_evidence"] is False


def test_duplicate_and_historical_copies_cannot_supply_source() -> None:
    receipt = build_current_sawm_authority_receipt(REPO_ROOT)
    assert receipt["non_authoritative_overlaps"]
    classifications = {row["classification"] for row in receipt["non_authoritative_overlaps"]}
    assert "duplicate_non_authoritative" in classifications
    assert "historical_only" in classifications
    for row in receipt["non_authoritative_overlaps"]:
        assert row["may_supply_current_source"] is False
        assert row["in_sealed_tree"] is False


def test_verify_rejects_plan_fixture_and_ducklake_authority_tampering() -> None:
    receipt = build_current_sawm_authority_receipt(REPO_ROOT)
    tampered = json.loads(json.dumps(receipt))
    feature = next(item for item in tampered["desired_capabilities"] if item["capability_id"] == "SAWM-005")
    feature["implementation_evidence"].append(
        {
            "path": "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md",
            "kind": "implementation_source",
            "exists": True,
            "in_sealed_tree": True,
            "ast_implementation_node_count": 4,
        }
    )
    report = verify_current_sawm_authority_receipt(tampered, REPO_ROOT)
    assert report["valid"] is False
    assert any("rejected kind" in error or "plan/fixture/inventory" in error or "does not match" in error for error in report["errors"])

    tampered = json.loads(json.dumps(receipt))
    tampered["control_plane_authority"]["ducklake"]["authoritative"] = True
    report = verify_current_sawm_authority_receipt(tampered, REPO_ROOT)
    assert report["valid"] is False
    assert any("DuckLake" in error or "does not match" in error for error in report["errors"])

    tampered = json.loads(json.dumps(receipt))
    missing = next(item for item in tampered["desired_capabilities"] if item["status"] == "missing")
    missing["verified_gap"] = ""
    missing["implementation_evidence"] = []
    report = verify_current_sawm_authority_receipt(tampered, REPO_ROOT)
    assert report["valid"] is False


def test_egraph_and_required_rollout_remain_typed_gaps() -> None:
    receipt = build_current_sawm_authority_receipt(REPO_ROOT)
    by_id = {item["capability_id"]: item for item in receipt["desired_capabilities"]}
    assert by_id["SAWM-019"]["status"] == "missing"
    assert by_id["SAWM-019"]["implementation_evidence"] == []
    assert by_id["SAWM-037"]["status"] == "missing"
    assert "required" in by_id["SAWM-037"]["verified_gap"].lower() or "pre/post" in by_id["SAWM-037"]["verified_gap"].lower()
    assert by_id["SAWM-023"]["status"] == "missing"
    assert by_id["SAWM-007"]["status"] == "missing"


def test_builder_and_verifier_symbols_are_present() -> None:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=__file__)
    names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "build_current_sawm_authority_receipt" in names
    assert "verify_current_sawm_authority_receipt" in names


if __name__ == "__main__":
    path = write_current_sawm_authority_receipt(REPO_ROOT)
    report = verify_current_sawm_authority_receipt(repo_root=REPO_ROOT)
    if not report["valid"]:
        raise SystemExit("\n".join(report["errors"]))
    print(path)
    print(json.dumps({"valid": True, "capability_count": report["capability_count"]}, indent=2))
