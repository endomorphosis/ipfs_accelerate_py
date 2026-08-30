"""Independent current-tree tests for DOEP-002 direct-state-write inventory.

A worker or model assertion alone is insufficient. This module verifies the
plan-bound declared outputs, inventory coverage, source anchors, and candidate
receipt schema required by DOEP-PLAN-V5.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = ACCELERATE_ROOT / "ipfs_accelerate_py" / "agent_supervisor"

INVENTORY_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "inventory"
    / "direct_state_writes.json"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "outputs"
    / "DOEP-002.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "receipts"
    / "DOEP-002.json"
)

OWNER_RELATIVE_OUTPUTS = (
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/inventory/direct_state_writes.json",
    "test/api/doep/test_doep_002_inventory_direct_state_write_and_controller_bypasses.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-002.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-002.json",
)

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

PLAN_CID = "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
TASK_CID = "sha256:a7f00461b76661ed402eb94cb98284e2f270c92d4d1803c40f3013b719c256b0"
BOARD_NS = "agent-supervisor-direct-objective-and-event-driven-planning-v1"

REQUIRED_DEFERRED_HAZARD_IDS = {
    "legacy_contextpack_candidates",
    "noncanonical_in_memory_event_helpers",
    "duplicate_adapters",
    "retention_tombstone_gaps",
}

REQUIRED_DIRECT_DUCKDB_WRITER_IDS = {
    "artifact-store-json-duckdb",
    "bundle-index-duckdb-sidecar",
    "duckdb-state-primitives",
    "duckdb-task-source",
    "formal-verification-cache-duckdb",
    "lease-coordination-duckdb",
    "legacy-landed-review-duckdb",
    "merge-queue-duckdb",
    "merge-resolver-duckdb",
    "proof-carrying-workflow-duckdb",
    "proof-scheduler-duckdb",
    "prompt-workflow-duckdb-materialization",
    "prover-evidence-duckdb",
}

SOURCE_ANCHORS = {
    "ipfs_accelerate_py/agent_supervisor/control/control_plane.py": (
        "class SupervisorControlService",
    ),
    "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py": (
        "QuackStateServer@1",
        "class QuackStateServer",
    ),
    "ipfs_accelerate_py/agent_supervisor/task_sources/database_task_source.py": (
        "DatabaseTaskSource@1",
    ),
    "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py": (
        "DuckDBTaskSource",
    ),
    "ipfs_accelerate_py/agent_supervisor/entrypoints/plan_materializer.py": (
        "DuckDBTaskSource",
        "materialize",
    ),
    "ipfs_accelerate_py/agent_supervisor/entrypoints/facade.py": (
        "class Supervisor",
    ),
    "ipfs_accelerate_py/agent_supervisor/entrypoints/cli.py": (
        "register_supervisor_cli",
    ),
    "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_repository.py": (
        "RepositoryAuthorityMode",
        "never falls back",
    ),
    "ipfs_accelerate_py/agent_supervisor/semantic_state/context_pack.py": (
        "V01_PRODUCTION_AUTHORITY: ClassVar[bool] = False",
    ),
    "ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/prompt_entrypoints.py": (
        "PROMPT_LIFECYCLE_TOOLS",
    ),
    "ipfs_accelerate_py/agent_supervisor/architecture_refactorer/duplicate_authority.py": (
        "CONTROL_BYPASS",
        "COMPATIBILITY_BYPASS",
    ),
    "docs/architecture/agent_supervisor/FOR_AGENTS.md": (
        "Do not bypass `SupervisorControlService`",
    ),
}


def _load_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert isinstance(payload, dict), f"{path} root must be an object"
    assert raw == json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", (
        f"{path} must be canonical JSON"
    )
    return payload


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        path = ACCELERATE_ROOT / relative
        assert path.is_file(), f"missing declared output: {relative}"
        assert path.stat().st_size > 0
    assert INVENTORY_PATH.is_file()
    assert TEST_PATH.is_file()
    assert OUTPUT_PATH.is_file()
    assert RECEIPT_PATH.is_file()


def test_inventory_contracts_and_coverage() -> None:
    inventory = _load_json(INVENTORY_PATH)
    assert inventory["schema"] == (
        "ipfs_accelerate_py/agent-supervisor/doep-direct-state-write-inventory@1"
    )
    assert inventory["task_id"] == "DOEP-002"
    assert inventory["plan_revision"] == "DOEP-PLAN-V5"
    assert inventory["plan_cid"] == PLAN_CID
    assert inventory["task_cid"] == TASK_CID
    assert inventory["board_namespace"] == BOARD_NS
    assert inventory["goal_id"] == "DOEP-G010.S1"
    assert inventory["completion_authoritative"] is False
    assert inventory["worker_completion_insufficient"] is True
    assert inventory["no_competing_subsystem_created"] is True
    assert inventory["authority"] is False
    assert inventory["promotion_eligible"] is False
    assert inventory["base_repositories"] == BASE_REPOSITORIES
    assert list(inventory["declared_outputs"]) == list(OWNER_RELATIVE_OUTPUTS)

    canonical = inventory["canonical_paths"]
    assert isinstance(canonical, list) and len(canonical) >= 5
    canonical_ids = {item["id"] for item in canonical}
    assert {"CP-001", "CP-004", "CP-006", "CP-007"} <= canonical_ids
    for item in canonical:
        assert item["kind"] == "canonical_path"
        assert item["disposition"] in {"allowed_owner", "compatibility_adapter"}
        assert item["exists_on_disk"] is True
        assert (ACCELERATE_ROOT / item["path"]).is_file()

    write_path = inventory["canonical_write_path"]
    assert "SupervisorControlService" in write_path["summary"]
    assert "QuackStateServer@1" in write_path["summary"]
    assert "control.duckdb" in write_path["summary"]

    direct = inventory["direct_state_writes"]
    assert isinstance(direct, list) and len(direct) >= 8
    for item in direct:
        assert item["kind"] == "direct_state_write"
        assert item["disposition"] == "record_only"
        assert item["campaign_expansion"] is False
        assert item["exists_on_disk"] is True
        assert (ACCELERATE_ROOT / item["path"]).is_file()
    direct_ids = {item["id"] for item in direct}
    assert {"DSW-001", "DSW-002", "DSW-003", "DSW-007"} <= direct_ids

    bypasses = inventory["controller_bypasses"]
    assert isinstance(bypasses, list) and len(bypasses) >= 5
    for item in bypasses:
        assert item["kind"] == "controller_bypass"
        assert item["disposition"] in {"record_only", "forbidden_if_used"}
        assert item["campaign_expansion"] is False
        assert item["exists_on_disk"] is True
        assert (ACCELERATE_ROOT / item["path"]).is_file()
    bypass_ids = {item["id"] for item in bypasses}
    assert {"CB-001", "CB-002", "CB-003", "CB-007"} <= bypass_ids

    hazards = inventory["deferred_hazards"]
    assert isinstance(hazards, list)
    hazard_ids = {item["id"] for item in hazards}
    assert REQUIRED_DEFERRED_HAZARD_IDS <= hazard_ids
    for item in hazards:
        assert item["kind"] == "deferred_hazard"
        assert item["campaign_expansion"] is False

    writers = inventory["direct_duckdb_writers"]
    assert isinstance(writers, list)
    writer_ids = {item["sink_id"] for item in writers}
    assert writer_ids == REQUIRED_DIRECT_DUCKDB_WRITER_IDS
    assert len(writers) == 13
    for item in writers:
        assert (ACCELERATE_ROOT / item["path"]).is_file()

    bypass_ref = inventory["bypass_detection_reference"]
    assert bypass_ref["module"].endswith("duplicate_authority.py")
    assert set(bypass_ref["kinds"]) == {"compatibility_bypass", "control_bypass"}
    assert (ACCELERATE_ROOT / bypass_ref["module"]).is_file()

    negatives = inventory["negative_assertions"]
    assert isinstance(negatives, list) and len(negatives) >= 4
    joined = " ".join(negatives).lower()
    assert "completion" in joined
    assert "quack" in joined
    assert "competing" in joined


def test_source_anchors_exist_on_current_tree() -> None:
    for relative, needles in SOURCE_ANCHORS.items():
        path = ACCELERATE_ROOT / relative
        assert path.is_file(), f"missing source anchor: {relative}"
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            assert needle in text, f"{relative} missing {needle!r}"


def test_interface_constants_present_in_source() -> None:
    checks = {
        PACKAGE_ROOT / "runtime" / "quack_state_server.py": "QuackStateServer@1",
        PACKAGE_ROOT / "task_sources" / "database_task_source.py": "DatabaseTaskSource@1",
        PACKAGE_ROOT / "runtime" / "database_event_log.py": "DatabaseEventLog@1",
        PACKAGE_ROOT
        / "semantic_state"
        / "context_pack.py": "V01_PRODUCTION_AUTHORITY: ClassVar[bool] = False",
        PACKAGE_ROOT / "control" / "control_plane.py": "class SupervisorControlService",
    }
    for path, needle in checks.items():
        assert path.is_file(), f"missing source file: {path}"
        assert needle in path.read_text(encoding="utf-8")


def test_output_manifest_contract() -> None:
    manifest = _load_json(OUTPUT_PATH)
    assert manifest["schema"] == "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"
    assert manifest["task_id"] == "DOEP-002"
    assert manifest["plan_revision"] == "DOEP-PLAN-V5"
    assert manifest["plan_cid"] == PLAN_CID
    assert manifest["task_cid"] == TASK_CID
    assert manifest["board_namespace"] == BOARD_NS
    assert manifest["completion_authoritative"] is False
    assert manifest["worker_completion_insufficient"] is True
    assert manifest["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert list(manifest["declared_outputs"]) == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["title"] == (
        "Inventory direct-state-write and controller bypasses"
    )
    assert manifest["validation_profile"] == "doep-validation/DOEP-PLAN-V5/DOEP-002@1"

    inventory_summary = manifest["inventory_summary"]
    assert inventory_summary["direct_state_write_count"] >= 8
    assert inventory_summary["controller_bypass_count"] >= 5
    assert inventory_summary["direct_duckdb_writer_count"] == 13
    assert inventory_summary["deferred_hazard_count"] >= 4
    assert inventory_summary["canonical_path_count"] >= 5


def test_candidate_receipt_contract() -> None:
    receipt = _load_json(RECEIPT_PATH)
    assert receipt["schema"] == "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"
    assert receipt["task_id"] == "DOEP-002"
    assert receipt["plan_revision"] == "DOEP-PLAN-V5"
    assert receipt["plan_cid"] == PLAN_CID
    assert receipt["task_cid"] == TASK_CID
    assert receipt["board_namespace"] == BOARD_NS
    assert receipt["candidate_status"] == "implemented"
    assert receipt["completion_authoritative"] is False
    assert receipt["worker_completion_insufficient"] is True
    assert receipt["no_competing_subsystem_created"] is True
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
        "test/api/doep/test_doep_002_inventory_direct_state_write_and_controller_bypasses.py",
        "-q",
    ]
    assert evidence["test_proof_results"]["proof_selection"] == "verified_empty"
    assert isinstance(evidence["limitations"], list) and evidence["limitations"]
    assert evidence["verifier_admission"] == "pending_independent_fenced_supervisor"
    assert evidence["receipt_cid"]["status"] == "unavailable"

    authority = receipt["authority"]
    assert authority["ducklake_authority"] is False
    assert authority["model_output_is_completion_authority"] is False
    assert authority["canonical_semantic_identity"] == "ipfs_datasets_py"
    assert authority["exact_bytes_cid_storage"] == "ipfs_kit_py"
    assert authority["operational_admission"] == "ipfs_accelerate_py"
    assert "direct_state_writes" in authority["inventoried"]
    assert "controller_bypasses" in authority["inventoried"]

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
        if not relative.endswith("/receipts/DOEP-002.json")
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
    inventory = _load_json(INVENTORY_PATH)
    assert manifest["task_id"] == receipt["task_id"] == inventory["task_id"] == "DOEP-002"
    assert manifest["plan_revision"] == receipt["plan_revision"] == inventory["plan_revision"]
    assert manifest["board_namespace"] == receipt["board_namespace"]
    assert manifest["declared_outputs"] == receipt["expected_outputs"]
    assert manifest["base_repositories"] == receipt["required_evidence"][
        "source_commit_tree_gitlinks"
    ]
    assert manifest["inventory_summary"]["direct_duckdb_writer_count"] == len(
        inventory["direct_duckdb_writers"]
    )
    assert manifest["inventory_summary"]["deferred_hazard_count"] == len(
        inventory["deferred_hazards"]
    )


def test_no_competing_subsystem_and_no_promotion_claim() -> None:
    inventory = _load_json(INVENTORY_PATH)
    manifest = _load_json(OUTPUT_PATH)
    receipt = _load_json(RECEIPT_PATH)
    assert inventory["no_competing_subsystem_created"] is True
    assert manifest["no_competing_subsystem_created"] is True
    assert receipt["no_competing_subsystem_created"] is True
    assert inventory["promotion_eligible"] is False
    assert manifest.get("promotion_eligible", False) is False
    assert receipt.get("promotion_eligible", False) is False
    assert inventory["completion_authoritative"] is False
    assert all(item.get("campaign_expansion") is False for item in inventory["deferred_hazards"])
