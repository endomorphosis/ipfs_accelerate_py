from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "docs/architecture/external_agent_autonomous_execution_fabric"
SOURCE_PATH = CAMPAIGN / "source_reconciliation_manifest.json"
STACK_PATH = CAMPAIGN / "stack_compatibility_manifest.json"
BOARD_PATH = CAMPAIGN / "task_board.json"
REPORT_PATH = CAMPAIGN / "reconciliation_report.md"
RECEIPT_PATH = CAMPAIGN / "receipts/source_reconciliation_verification.json"
EXPECTED_REPOSITORIES = (
    "ipfs_accelerate_py",
    "ipfs_datasets_py",
    "ipfs_kit_py",
    "Mcp-Plus-Plus",
)
EXPECTED_CLASSIFICATIONS = {
    "ipfs_accelerate_py": [
        "partially_merged_high_conflict",
        "partial_unqualified",
        "partially_merged_superseded",
        "potential_duplicate_contract_family",
    ],
    "ipfs_datasets_py": [
        "content_integrated_without_ancestry",
        "stale_proof_reuse_restoration",
        "isolated_two_file_semantic_contract_candidate",
        "alternate_conflict_heavy_snapshot",
    ],
    "ipfs_kit_py": [
        "incompatible_stale_proof_api",
        "simplistic_subprocess_wrappers_not_handoff_adapters",
        "superseded_partial_CAS",
    ],
    "Mcp-Plus-Plus": [],
}
REQUIRED_CLASSIFICATION_FIELDS = (
    "branch",
    "head",
    "files_changed",
    "schemas_changed",
    "public_apis_changed",
    "tests",
    "dependencies",
    "superseded",
    "conflict_risk",
    "classification",
    "disposition",
)
FALSE_POLICY = (
    "force_push",
    "history_rewrite",
    "discard_unmerged_work",
    "squash_provenance",
    "overwrite_dirty_worktrees",
    "branch_names_are_qualification",
    "historical_test_claims_are_current",
)
ZERO_PRESERVATION = (
    "refs_deleted",
    "worktrees_deleted",
    "force_pushes",
    "history_rewrites",
    "dirty_overlays_overwritten",
    "remote_pushes",
)


def _canonical_cid(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _source() -> dict:
    return _load(SOURCE_PATH)


def test_frozen_manifest_schema_and_forest_root_reproduce() -> None:
    source = _source()
    assert source["schema"] == "SourceReconciliationManifest@1"
    payload = source["source_forest_payload"]
    assert payload["schema"] == "ExternalAgentSourceForest@1"
    assert set(payload["repositories"]) == set(EXPECTED_REPOSITORIES)
    assert _canonical_cid(payload) == source["source_forest_root"]
    assert source["source_forest_root"] == (
        "sha256:ed543c10f6aa90e093c8ae8b8866934e0cc1614e1be49ddcdc5dd7a2ce8565fa"
    )


def test_board_and_stack_bind_the_same_unmutated_inputs() -> None:
    source = _source()
    board = _load(BOARD_PATH)
    stack = _load(STACK_PATH)
    assert board["source_reconciliation_manifest_cid"] == _canonical_cid(source)
    assert board["source_forest_root"] == source["source_forest_root"]
    assert board["stack_compatibility_manifest_cid"] == _canonical_cid(stack)
    selected = source["selected_integration_roots"]
    roots = stack["integration_roots"]
    assert set(selected) == set(EXPECTED_REPOSITORIES)
    assert set(roots) == set(EXPECTED_REPOSITORIES)
    for name in EXPECTED_REPOSITORIES:
        for field in ("commit", "tree", "integration_branch"):
            assert selected[name][field] == roots[name][field]


def test_policy_and_preservation_forbid_destructive_source_ops() -> None:
    source = _source()
    policy = source["policy"]
    for key in FALSE_POLICY:
        assert policy[key] is False
    preservation = source["preservation_receipt"]
    for key in ZERO_PRESERVATION:
        assert preservation[key] == 0


def test_four_repository_classifications_match_frozen_manifest() -> None:
    source = _source()
    repositories = source["repositories"]
    assert set(repositories) == set(EXPECTED_REPOSITORIES)
    for name, expected in EXPECTED_CLASSIFICATIONS.items():
        unmerged = repositories[name].get("relevant_unmerged") or []
        observed = [item["classification"] for item in unmerged]
        assert observed == expected
        for item in unmerged:
            for field in REQUIRED_CLASSIFICATION_FIELDS:
                assert field in item
            files_changed = item["files_changed"]
            assert files_changed["complete"] is True
            assert files_changed["sha256"]
            assert item["safe_to_cherry_pick"] is False


def test_human_report_projects_the_same_roots_and_decisions() -> None:
    source = _source()
    report = REPORT_PATH.read_text(encoding="utf-8")
    assert "source_reconciliation_manifest.json" in report
    for name in EXPECTED_REPOSITORIES:
        selected = source["selected_integration_roots"][name]
        assert selected["commit"] in report
        assert selected["tree"] in report
    assert source["source_forest_root"] in report
    assert "two-parent UI/UX-IR merge" in report
    assert "backend roles without changing schema" in report
    datasets = source["repositories"]["ipfs_datasets_py"]
    stale = [
        item
        for item in datasets["relevant_unmerged"]
        if item["classification"] == "alternate_conflict_heavy_snapshot"
    ]
    assert stale and "do not merge wholesale" in stale[0]["disposition"]
    proof = [
        item
        for item in datasets["relevant_unmerged"]
        if item["classification"] == "stale_proof_reuse_restoration"
    ]
    assert proof and "never blind cherry-pick" in proof[0]["disposition"]


def test_cross_repository_authority_split_is_recorded() -> None:
    source = _source()
    decisions = source["cross_repository_decisions"]
    assert "DuckDB transactional records" in decisions["mutable_coordination"]
    assert "one fenced authenticated Quack file owner" in decisions["mutable_coordination"]
    assert "never current claim, lease, fence or merge authority" in decisions[
        "immutable_history"
    ]
    mcp = source["repositories"]["Mcp-Plus-Plus"]
    assert "Profile G" in mcp["contract_reuse"]
    assert "new MCP++ profile" in mcp["prohibited_new_contracts"]
    assert "not the DuckDB/Quack authority" in mcp["qualification_limit"]


def test_verification_receipt_reproduces_without_mutating_board_inputs() -> None:
    source_before = SOURCE_PATH.read_bytes()
    board_before = BOARD_PATH.read_bytes()
    stack_before = STACK_PATH.read_bytes()
    report_before = REPORT_PATH.read_bytes()
    source = json.loads(source_before.decode("utf-8"))
    board = json.loads(board_before.decode("utf-8"))
    stack = json.loads(stack_before.decode("utf-8"))
    receipt = {
        "schema": "SourceReconciliationVerification@1",
        "task_id": "EAAEF-001",
        "campaign": "ExternalAgentAutonomousExecutionFabric",
        "decision": "verified_independent_reproduction",
        "inputs_unmutated": True,
        "source_reconciliation_manifest_cid": _canonical_cid(source),
        "stack_compatibility_manifest_cid": _canonical_cid(stack),
        "source_forest_root": source["source_forest_root"],
        "board_source_forest_root": board["source_forest_root"],
        "report_sha256": "sha256:" + hashlib.sha256(report_before).hexdigest(),
        "selected_integration_roots": {
            name: {
                "commit": rec["commit"],
                "tree": rec["tree"],
                "integration_branch": rec["integration_branch"],
                "decision": rec["decision"],
            }
            for name, rec in source["selected_integration_roots"].items()
        },
        "classifications": {
            name: [
                {
                    "branch": item["branch"],
                    "classification": item["classification"],
                    "disposition": item["disposition"],
                    "files_changed_sha256": item["files_changed"]["sha256"],
                }
                for item in (source["repositories"][name].get("relevant_unmerged") or [])
            ]
            for name in EXPECTED_REPOSITORIES
        },
        "policy": source["policy"],
        "preservation_receipt": source["preservation_receipt"],
        "board_inputs": [
            str(SOURCE_PATH.relative_to(ROOT)),
            str(STACK_PATH.relative_to(ROOT)),
            str(BOARD_PATH.relative_to(ROOT)),
            str(REPORT_PATH.relative_to(ROOT)),
        ],
    }
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stored = _load(RECEIPT_PATH)
    assert stored["decision"] == "verified_independent_reproduction"
    assert stored["source_forest_root"] == source["source_forest_root"]
    assert stored["source_reconciliation_manifest_cid"] == board[
        "source_reconciliation_manifest_cid"
    ]
    assert SOURCE_PATH.read_bytes() == source_before
    assert BOARD_PATH.read_bytes() == board_before
    assert STACK_PATH.read_bytes() == stack_before
    assert REPORT_PATH.read_bytes() == report_before
