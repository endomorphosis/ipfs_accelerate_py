"""Exact-tree prerequisite inventory contract tests for PCPC-000/001."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[3]
INVENTORY = ROOT / "docs/architecture/procedure_compiler_inventory"
START_COMMIT = "bbf7f68799072c2b81f7d96eac91f2df3c4b3952"
START_TREE = "a698da9e4b54e2929adacb613bc61ba3e72eed58"
ALLOWED = {"available", "available_with_caveats", "incompatible", "stale", "missing"}
SCRIPT = ROOT / "scripts/materialize_agent_supervisor_procedure_compiler_program.py"
REQUIRED_BINDINGS = {
    "source_bindings",
    "symbol_bindings",
    "interface_bindings",
    "schema_bindings",
    "package_bindings",
    "submodule_bindings",
    "test_producer_bindings",
}


def _object(name: str) -> dict[str, object]:
    value = json.loads((INVENTORY / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_materializer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pcpc_inventory_probe_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_binds_exact_committed_tree_and_releases() -> None:
    baseline = _object("baseline.json")
    repository = baseline["repository"]
    assert isinstance(repository, dict)
    assert repository["commit"] == START_COMMIT
    assert repository["tree"] == START_TREE
    assert repository["origin_main_commit"] == START_COMMIT
    assert baseline["package"] == {"name": "ipfs_accelerate_py", "version": "0.0.45"}
    assert baseline["schema"].endswith("procedure-compiler-baseline@4")
    assert baseline["package_bindings"] == [
        {
            "binding_id": "package:ipfs_accelerate_py@0.0.45",
            "manifest_blob_id": "179882cc1254039920c5a1ab755f383dcb70842d",
            "manifest_path": "pyproject.toml",
            "name": "ipfs_accelerate_py",
            "version": "0.0.45",
        }
    ]
    assert baseline["sibling_gitlinks"] == {
        "ipfs_accelerate_py/mcplusplus": "5ac0ab162f420264fd224073a5df3f2d7c054ae3",
        "ipfs_datasets_py": "480a1666f144ad606fcb3cacb66e59775f28d0d1",
        "ipfs_kit_py": "2564aea1ae35061f2165872aff91e8a40801ab7e",
    }
    sibling_rows = baseline["sibling_release_bindings"]
    assert isinstance(sibling_rows, list) and len(sibling_rows) == 3
    assert all(
        isinstance(row, dict)
        and row["binding_id"].startswith("gitlink:")
        and len(row["gitlink_commit"]) == 40
        for row in sibling_rows
    )
    assert baseline["excluded_sources"] == [
        "planning_document_status",
        "receipt_shaped_unadmitted_json",
        "task_board_status",
        "uncommitted_working_tree_overlays",
    ]
    assert baseline["test_evidence"]


def test_baseline_binds_every_exact_test_producer_and_expected_failure() -> None:
    baseline = _object("baseline.json")
    producers = baseline["test_producers"]
    assert isinstance(producers, list) and len(producers) == 18
    by_id = {str(row["producer_id"]): row for row in producers}
    assert len(by_id) == len(producers)
    module = _load_materializer()
    assert set(by_id) == module.REQUIRED_PREREQUISITE_PRODUCER_IDS
    for producer in producers:
        assert producer["command"][:6] == [
            "python",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
        ]
        assert producer["simulated"] is False
        assert producer["source_bindings"]
        for source in producer["source_bindings"]:
            if source.get("baseline_absent") is True:
                assert "blob_id" not in source
                assert len(source["current_blob_id"]) == 40
            else:
                assert len(source["blob_id"]) == 40
                if "current_blob_id" in source:
                    assert len(source["current_blob_id"]) == 40
            assert source["path"] in producer["command"]
        expected = producer["expected"]
        assert expected["collected"] == expected["passed"] + expected["failed"]
        if expected["returncode"]:
            assert producer["expected_failure"]
            assert producer["expected_failure"]["required_output_fragments"]
    assert by_id["TP-ADAPTIVE-PLANNER-IMPORT"]["expected"] == {
        "collected": 0,
        "errors": 1,
        "failed": 0,
        "passed": 0,
        "returncode": 2,
    }
    assert by_id["TP-ADAPTIVE-PLANNER-IMPORT"]["expected_failure"] == {
        "reason_code": "adaptive_planner_import_undefined_hammer_trace_schema",
        "required_output_fragments": [
            "NameError: name 'HAMMER_TRACE_SCHEMA' is not defined"
        ],
        "signature": "NameError: name 'HAMMER_TRACE_SCHEMA' is not defined",
    }
    assert by_id["TP-DELTA-RETRY"]["expected"]["failed"] == 1
    assert by_id["TP-DEFAULT-PROVIDER-ROUTE"]["expected"]["failed"] == 21
    assert by_id["TP-WORKTREE-LIFECYCLE"]["expected"] == {
        "collected": 51,
        "errors": 0,
        "failed": 2,
        "passed": 49,
        "returncode": 1,
    }
    assert by_id["TP-WORKTREE-LIFECYCLE"]["source_bindings"] == [
        {
            "blob_id": "4cf5c39ff1e9dfc97f533e1d036e1b9256d15e52",
            "current_blob_id": "472daae1adaf2d41bdb09df087dbf410bef8420c",
            "path": "test/api/test_agent_supervisor_worktree_lifecycle.py",
        }
    ]
    assert by_id["TP-FENCE-REGISTRY-QUEUE"]["expected"]["failed"] == 7


def test_prerequisites_have_one_closed_honest_disposition() -> None:
    inventory = _object("prerequisites.json")
    rows = inventory["dispositions"]
    assert isinstance(rows, list)
    by_name = {str(row["authority"]): row for row in rows}
    assert len(by_name) == len(rows)
    assert {str(row["status"]) for row in rows} <= ALLOWED
    assert inventory["baseline_commit"] == START_COMMIT
    assert inventory["baseline_tree"] == START_TREE
    assert inventory["schema"].endswith("procedure-compiler-prerequisite-inventory@3")
    for row in rows:
        assert REQUIRED_BINDINGS <= set(row)
        assert row["package_bindings"] == ["package:ipfs_accelerate_py@0.0.45"]
        if row["status"] == "missing":
            assert not row["source_bindings"]
            assert not row["symbol_bindings"]
            assert not row["interface_bindings"]
            assert not row["schema_bindings"]
            assert not row["test_producer_bindings"]
            assert row["blocker"]
            assert row["negative_probes"]
        else:
            assert row["source_bindings"]
            assert row["symbol_bindings"]
            assert row["interface_bindings"]
            assert row["schema_bindings"]
            assert row["test_producer_bindings"]
            for binding in row["source_bindings"]:
                if binding.get("baseline_absent") is True:
                    assert "blob_id" not in binding
                    assert len(binding["current_blob_id"]) == 40
                else:
                    assert len(binding["blob_id"]) == 40
                    if "current_blob_id" in binding:
                        assert len(binding["current_blob_id"]) == 40
    assert by_name["AdaptivePlanner"]["status"] == "incompatible"
    assert (
        by_name["AdaptivePlanner"]["blocker"]
        == "adaptive_planner_import_undefined_hammer_trace_schema"
    )
    catalog_binding = next(
        binding
        for binding in by_name["AdaptivePlanner"]["source_bindings"]
        if binding["path"].endswith("mcp_contract_catalog.py")
    )
    assert catalog_binding == {
        "baseline_absent": True,
        "current_blob_id": "986656ec49b66cf2a7abbbbedb3151f33bc817ec",
        "path": "ipfs_accelerate_py/agent_supervisor/analysis/mcp_contract_catalog.py",
    }
    assert by_name["AutonomousMetaController"]["status"] == "missing"
    assert by_name["AdversarialAssuranceEngine"]["status"] == "available_with_caveats"
    assert by_name["IncrementalProofSealer"]["status"] == "available_with_caveats"


def test_deterministic_per_authority_probe_binds_current_tree() -> None:
    module = _load_materializer()
    probe = module._probe_prerequisite_inventory()
    inventory = _object("prerequisites.json")
    assert probe["authority_count"] == len(inventory["dispositions"])
    assert probe["test_producer_count"] >= 18
    assert probe["source_drift_permitted"] is False
    assert probe["simulated"] is False
    assert module._stored_prerequisite_probe_is_intact(
        probe,
        head=module._git("rev-parse", "HEAD"),
        tree=module._git("rev-parse", "HEAD^{tree}"),
    )


def test_prerequisite_validation_rejects_missing_per_authority_binding() -> None:
    module = _load_materializer()
    baseline = _object("baseline.json")
    inventory = copy.deepcopy(_object("prerequisites.json"))
    inventory["dispositions"][0].pop("schema_bindings")
    with pytest.raises(module.MaterializationError, match="schema_bindings binding is missing"):
        module._validate_prerequisite_payloads(baseline, inventory)


def test_prerequisite_validation_rejects_current_source_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_materializer()
    baseline = _object("baseline.json")
    inventory = _object("prerequisites.json")
    original = module._object_id

    def drifted_object_id(revision: str, path: str) -> str:
        if revision == "HEAD" and path.endswith("semantic_state/harness.py"):
            return "f" * 40
        return original(revision, path)

    monkeypatch.setattr(module, "_object_id", drifted_object_id)
    with pytest.raises(module.MaterializationError, match="current-tree source drift"):
        module._validate_prerequisite_payloads(baseline, inventory)


def test_prerequisite_probe_identity_rejects_inventory_digest_edit() -> None:
    module = _load_materializer()
    probe = module._probe_prerequisite_inventory()
    probe["prerequisites_sha256"] = "0" * 64
    assert not module._stored_prerequisite_probe_is_intact(
        probe,
        head=module._git("rev-parse", "HEAD"),
        tree=module._git("rev-parse", "HEAD^{tree}"),
    )


def test_inventory_does_not_claim_runtime_completion() -> None:
    for name in ("baseline.json", "prerequisites.json"):
        text = (INVENTORY / name).read_text(encoding="utf-8").lower()
        assert "production ready" not in text
        assert "board complete" not in text
        assert "all tasks done" not in text
