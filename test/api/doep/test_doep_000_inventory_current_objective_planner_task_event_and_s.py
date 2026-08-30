"""Independent current-tree tests for DOEP-000 authority inventory.

A worker or model assertion alone is insufficient. This module verifies the
plan-bound declared outputs, authority inventory coverage, and candidate
receipt schema required by DOEP-PLAN-V5.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = ACCELERATE_ROOT / "ipfs_accelerate_py" / "agent_supervisor"

ADR_PATH = (
    ACCELERATE_ROOT
    / "docs"
    / "architecture"
    / "agent_supervisor"
    / "DOEP_AUTHORITY_ADR.md"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "outputs"
    / "DOEP-000.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "receipts"
    / "DOEP-000.json"
)

OWNER_RELATIVE_OUTPUTS = (
    "docs/architecture/agent_supervisor/DOEP_AUTHORITY_ADR.md",
    "test/api/doep/test_doep_000_inventory_current_objective_planner_task_event_and_s.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-000.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-000.json",
)

REQUIRED_AUTHORITIES = ("objective", "planner", "task", "event", "state")

BASE_REPOSITORIES = {
    "ipfs_accelerate_py": {
        "commit": "87715e9295626e7918f7fc8a7b1a1531ab04208f",
        "tree": "1c9a399cc7a599d5904e5be2ae58c6be3650cff7",
    },
    "ipfs_datasets_py": {
        "commit": "3668b8857a9aa7b1a3c847be12725b5cd057d2e7",
        "tree": "456e09b51d6a07a3a5873436df24054768195320",
    },
    "ipfs_kit_py": {
        "commit": "b6c65ba732733d7e33852713ba18aa3b12235668",
        "tree": "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2",
    },
    "lift_coding": {
        "commit": "bb8869ed72eb7002434345d9969efee729c4f7f6",
        "tree": "99e85bfe584b7688ffbeff86da1e612dd6893a42",
    },
}

AUTHORITY_ANCHORS = {
    "objective": (
        "objectives/objective_tracker.py",
        "objectives/objective_graph.py",
        "objectives/goal_completion.py",
        "entrypoints/intent_service.py",
        "entrypoints/plan_materializer.py",
    ),
    "planner": (
        "planning/formal_plan_compiler.py",
        "planning/formal_plan_validator.py",
        "planning/formal_replanner.py",
        "planning/adaptive_planner.py",
    ),
    "task": (
        "task_sources/database_task_source.py",
        "task_sources/intent_repository.py",
        "runtime/quack_state_server.py",
    ),
    "event": (
        "runtime/database_event_log.py",
        "runtime/event_log.py",
        "runtime/runtime_cas.py",
    ),
    "state": (
        "runtime/quack_state_server.py",
        "task_sources/intent_repository.py",
        "runtime/runtime_cas.py",
    ),
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"{path} root must be an object"
    return payload


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        path = ACCELERATE_ROOT / relative
        assert path.is_file(), f"missing declared output: {relative}"
    assert ADR_PATH.is_file()
    assert TEST_PATH.is_file()
    assert OUTPUT_PATH.is_file()
    assert RECEIPT_PATH.is_file()


def test_adr_inventories_five_authorities_and_reuse_rule() -> None:
    text = ADR_PATH.read_text(encoding="utf-8")
    assert "DOEP-PLAN-V5" in text
    assert "DOEP-000" in text
    assert "Accepted" in text
    for authority in REQUIRED_AUTHORITIES:
        assert authority in text.lower()
    lowered = text.lower()
    for needle in (
        "DatabaseTaskSource@1",
        "QuackStateServer@1",
        "DatabaseEventLog@1",
        "SupervisorIntentService",
        "FormalPlanCompiler",
        "AdaptivePlanner",
        "DuckLake",
        "ContextPacker.V01_PRODUCTION_AUTHORITY",
    ):
        assert needle in text, f"ADR missing required inventory claim: {needle}"
    for needle in (
        "no competing subsystem",
        "deferred hazards",
        "ducklake",
    ):
        assert needle in lowered, f"ADR missing required inventory claim: {needle}"


def test_authority_source_anchors_exist_on_current_tree() -> None:
    for authority, relatives in AUTHORITY_ANCHORS.items():
        for relative in relatives:
            path = PACKAGE_ROOT / relative
            assert path.is_file(), f"{authority} anchor missing: {relative}"


def test_interface_constants_present_in_source() -> None:
    checks = {
        PACKAGE_ROOT / "task_sources" / "database_task_source.py": "DatabaseTaskSource@1",
        PACKAGE_ROOT / "runtime" / "quack_state_server.py": "QuackStateServer@1",
        PACKAGE_ROOT / "runtime" / "database_event_log.py": "DatabaseEventLog@1",
        PACKAGE_ROOT
        / "semantic_state"
        / "context_pack.py": "V01_PRODUCTION_AUTHORITY: ClassVar[bool] = False",
    }
    for path, needle in checks.items():
        assert path.is_file(), f"missing source file: {path}"
        assert needle in path.read_text(encoding="utf-8")


def test_output_manifest_contract() -> None:
    manifest = _load_json(OUTPUT_PATH)
    assert manifest["schema"] == "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"
    assert manifest["task_id"] == "DOEP-000"
    assert manifest["plan_revision"] == "DOEP-PLAN-V5"
    assert (
        manifest["plan_cid"]
        == "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
    )
    assert (
        manifest["board_namespace"]
        == "agent-supervisor-direct-objective-and-event-driven-planning-v1"
    )
    assert manifest["completion_authoritative"] is False
    assert manifest["worker_completion_insufficient"] is True
    assert manifest["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert list(manifest["declared_outputs"]) == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES

    authorities = manifest["authorities"]
    assert isinstance(authorities, dict)
    assert set(authorities) == set(REQUIRED_AUTHORITIES)
    for name in REQUIRED_AUTHORITIES:
        entry = authorities[name]
        assert isinstance(entry, dict)
        assert entry["disposition"] == "reuse"
        assert isinstance(entry["paths"], list) and entry["paths"]
        assert isinstance(entry["interfaces"], list) and entry["interfaces"]

    hazards = manifest["deferred_hazards"]
    assert isinstance(hazards, list) and len(hazards) >= 4
    hazard_ids = {item["id"] for item in hazards}
    assert {
        "legacy_contextpack_candidates",
        "noncanonical_in_memory_event_helpers",
        "duplicate_adapters",
        "retention_tombstone_gaps",
    } <= hazard_ids


def test_candidate_receipt_contract() -> None:
    receipt = _load_json(RECEIPT_PATH)
    assert receipt["schema"] == "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"
    assert receipt["task_id"] == "DOEP-000"
    assert receipt["plan_revision"] == "DOEP-PLAN-V5"
    assert (
        receipt["board_namespace"]
        == "agent-supervisor-direct-objective-and-event-driven-planning-v1"
    )
    assert receipt["candidate_status"] == "implemented"
    assert receipt["completion_authoritative"] is False
    assert receipt["worker_completion_insufficient"] is True
    assert list(receipt["changed_paths"]) == list(OWNER_RELATIVE_OUTPUTS)
    assert list(receipt["write_scope"]) == list(OWNER_RELATIVE_OUTPUTS)
    assert list(receipt["expected_outputs"]) == list(OWNER_RELATIVE_OUTPUTS)

    outputs_present = receipt["outputs_present"]
    assert isinstance(outputs_present, dict)
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert outputs_present.get(relative) is True

    evidence = receipt["required_evidence"]
    assert isinstance(evidence, dict)
    assert evidence["source_commit_tree_gitlinks"] == BASE_REPOSITORIES
    assert isinstance(evidence["changed_path_digest"], str)
    assert evidence["changed_path_digest"].startswith("sha256:")
    assert evidence["test_proof_results"]["validation_command"] == [
        "python3",
        "-m",
        "pytest",
        "test/api/doep/test_doep_000_inventory_current_objective_planner_task_event_and_s.py",
        "-q",
    ]
    assert isinstance(evidence["limitations"], list) and evidence["limitations"]
    assert evidence["verifier_admission"] == "pending_independent_fenced_supervisor"
    assert evidence["receipt_cid"]["status"] == "unavailable"

    authority = receipt["authority"]
    assert authority["ducklake_authority"] is False
    assert authority["model_output_is_completion_authority"] is False
    assert authority["canonical_semantic_identity"] == "ipfs_datasets_py"
    assert authority["exact_bytes_cid_storage"] == "ipfs_kit_py"
    assert authority["operational_admission"] == "ipfs_accelerate_py"

    assert receipt["supervisor_acceptance"]["state"] == (
        "pending_independent_fenced_supervisor"
    )
    assert receipt["supervisor_acceptance"]["completion_authoritative"] is False


def test_changed_path_digest_matches_current_files() -> None:
    """Digest non-receipt declared outputs; the receipt embeds those digests."""
    receipt = _load_json(RECEIPT_PATH)
    digested_paths = [
        relative
        for relative in OWNER_RELATIVE_OUTPUTS
        if not relative.endswith("/receipts/DOEP-000.json")
    ]
    digests = {
        relative: _sha256_file(ACCELERATE_ROOT / relative) for relative in digested_paths
    }
    canonical = json.dumps(digests, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    expected = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert receipt["required_evidence"]["changed_path_digest"] == expected
    assert receipt["path_digests"] == digests
    assert set(digests) == set(digested_paths)


def test_output_manifest_and_receipt_agree() -> None:
    manifest = _load_json(OUTPUT_PATH)
    receipt = _load_json(RECEIPT_PATH)
    assert manifest["task_id"] == receipt["task_id"] == "DOEP-000"
    assert manifest["plan_revision"] == receipt["plan_revision"]
    assert manifest["board_namespace"] == receipt["board_namespace"]
    assert manifest["declared_outputs"] == receipt["expected_outputs"]
    assert manifest["base_repositories"] == receipt["required_evidence"][
        "source_commit_tree_gitlinks"
    ]
    assert set(manifest["authorities"]) == set(REQUIRED_AUTHORITIES)
