"""Independent current-tree tests for DOEP-003 repository contract seal.

A worker or model assertion alone is insufficient. This module verifies the
plan-bound declared outputs, sealed source-forest commits/trees, cross-
repository ownership contracts, and candidate receipt schema required by
DOEP-PLAN-V5.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
SUPERPROJECT_ROOT = ACCELERATE_ROOT.parent.parent

SEAL_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "inventory"
    / "repository_contract_seal.json"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "outputs"
    / "DOEP-003.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "receipts"
    / "DOEP-003.json"
)

OWNER_RELATIVE_OUTPUTS = (
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/inventory/repository_contract_seal.json",
    "test/api/doep/test_doep_003_seal_current_cross_repository_contracts_and_commits.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-003.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-003.json",
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

GITLINK_PATHS = {
    "ipfs_accelerate_py": "external/ipfs_accelerate",
    "ipfs_datasets_py": "external/ipfs_datasets",
    "ipfs_kit_py": "external/ipfs_kit",
}

OWNERSHIP = {
    "canonical_semantic_identity": "ipfs_datasets_py",
    "ducklake_authority": False,
    "exact_bytes_cid_storage": "ipfs_kit_py",
    "model_output_is_completion_authority": False,
    "operational_admission": "ipfs_accelerate_py",
}

ACCELERATE_DEPENDENCY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/scoped-project-dependency-preflight@4"
)
DATASETS_DEPENDENCY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/scoped-project-dependency-preflight@3"
)


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
    assert SEAL_PATH.is_file()
    assert TEST_PATH.is_file()
    assert OUTPUT_PATH.is_file()
    assert RECEIPT_PATH.is_file()


def test_repository_contract_seal_binds_source_forest_and_ownership() -> None:
    seal = _load_json(SEAL_PATH)
    assert seal["schema"] == (
        "ipfs_accelerate_py/agent-supervisor/doep-repository-contract-seal@1"
    )
    assert seal["seal_status"] == "sealed"
    assert seal["task_id"] == "DOEP-003"
    assert seal["plan_revision"] == "DOEP-PLAN-V5"
    assert (
        seal["plan_cid"]
        == "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
    )
    assert (
        seal["board_namespace"]
        == "agent-supervisor-direct-objective-and-event-driven-planning-v1"
    )
    assert seal["plan_epoch"] == 1
    assert (
        seal["task_cid"]
        == "sha256:27ae152374f415aad20e7396c1ba88ebb63440b98f667cbf59a26eb79a987ead"
    )
    assert seal["completion_authoritative"] is False
    assert seal["worker_completion_insufficient"] is True
    assert seal["no_competing_subsystem_created"] is True
    assert seal["branch_names_are_navigation_only"] is True

    forest = seal["immutable_source_forest"]
    assert isinstance(forest, dict)
    assert set(forest) == set(BASE_REPOSITORIES)
    for name, expected in BASE_REPOSITORIES.items():
        entry = forest[name]
        assert entry["commit"] == expected["commit"]
        assert entry["tree"] == expected["tree"]
        assert isinstance(entry["role"], str) and entry["role"]
        assert isinstance(entry["superproject_path"], str) and entry["superproject_path"]

    assert seal["superproject_gitlink_paths"] == GITLINK_PATHS

    ownership = seal["cross_repository_ownership"]
    for key, value in OWNERSHIP.items():
        assert ownership[key] == value
    roles = ownership["roles"]
    assert set(roles) >= {
        "ipfs_accelerate_py",
        "ipfs_datasets_py",
        "ipfs_kit_py",
        "lift_coding",
    }

    contracts = seal["dependency_contracts"]
    assert contracts["ipfs_accelerate_py"]["schema"] == ACCELERATE_DEPENDENCY_SCHEMA
    assert contracts["ipfs_datasets_py"]["schema"] == DATASETS_DEPENDENCY_SCHEMA
    assert "ipfs_kit_py" in contracts
    assert isinstance(seal["non_goals"], list) and len(seal["non_goals"]) >= 3


def test_dependency_contract_schemas_present_on_current_tree() -> None:
    accelerate_pyproject = (ACCELERATE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert ACCELERATE_DEPENDENCY_SCHEMA in accelerate_pyproject
    assert (
        "test/api/doep/test_doep_003_seal_current_cross_repository_contracts_and_commits.py"
        in accelerate_pyproject
    )

    datasets_root = SUPERPROJECT_ROOT / "external" / "ipfs_datasets"
    datasets_pyproject = (datasets_root / "pyproject.toml").read_text(encoding="utf-8")
    assert DATASETS_DEPENDENCY_SCHEMA in datasets_pyproject

    kit_root = SUPERPROJECT_ROOT / "external" / "ipfs_kit"
    assert kit_root.is_dir(), "ipfs_kit nested repository missing from superproject"


def test_sealed_commits_resolvable_when_local_objects_present() -> None:
    """When the sealed objects are present locally, commit/tree pairs must match."""
    import subprocess

    mapping = {
        "ipfs_accelerate_py": ACCELERATE_ROOT,
        "ipfs_datasets_py": SUPERPROJECT_ROOT / "external" / "ipfs_datasets",
        "ipfs_kit_py": SUPERPROJECT_ROOT / "external" / "ipfs_kit",
        "lift_coding": SUPERPROJECT_ROOT,
    }
    for name, repo_path in mapping.items():
        expected = BASE_REPOSITORIES[name]
        commit = expected["commit"]
        probe = subprocess.run(
            ["git", "-C", str(repo_path), "cat-file", "-t", commit],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            continue
        tree = subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-parse", f"{commit}^{{tree}}"],
            text=True,
        ).strip()
        assert tree == expected["tree"], f"{name} sealed tree mismatch for {commit}"


def test_output_manifest_contract() -> None:
    manifest = _load_json(OUTPUT_PATH)
    assert manifest["schema"] == "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"
    assert manifest["task_id"] == "DOEP-003"
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
    assert (
        manifest["task_cid"]
        == "sha256:27ae152374f415aad20e7396c1ba88ebb63440b98f667cbf59a26eb79a987ead"
    )
    assert manifest["validation_profile"] == "doep-validation/DOEP-PLAN-V5/DOEP-003@1"

    ownership = manifest["cross_repository_ownership"]
    for key, value in OWNERSHIP.items():
        assert ownership[key] == value

    seal_summary = manifest["repository_contract_seal"]
    assert seal_summary["path"] == OWNER_RELATIVE_OUTPUTS[0]
    assert seal_summary["schema"] == (
        "ipfs_accelerate_py/agent-supervisor/doep-repository-contract-seal@1"
    )
    assert seal_summary["seal_status"] == "sealed"
    assert seal_summary["immutable_source_forest"] == {
        name: {"commit": values["commit"], "tree": values["tree"]}
        for name, values in BASE_REPOSITORIES.items()
    }


def test_candidate_receipt_contract() -> None:
    receipt = _load_json(RECEIPT_PATH)
    assert receipt["schema"] == "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"
    assert receipt["task_id"] == "DOEP-003"
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
        "test/api/doep/test_doep_003_seal_current_cross_repository_contracts_and_commits.py",
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
    assert authority["repository_contract_sealed"] is True

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
        if not relative.endswith("/receipts/DOEP-003.json")
    ]
    digests = {
        relative: _sha256_file(ACCELERATE_ROOT / relative) for relative in digested_paths
    }
    canonical = json.dumps(digests, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    expected = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert receipt["required_evidence"]["changed_path_digest"] == expected
    assert receipt["path_digests"] == digests
    assert set(digests) == set(digested_paths)


def test_seal_manifest_and_receipt_agree() -> None:
    seal = _load_json(SEAL_PATH)
    manifest = _load_json(OUTPUT_PATH)
    receipt = _load_json(RECEIPT_PATH)
    assert seal["task_id"] == manifest["task_id"] == receipt["task_id"] == "DOEP-003"
    assert seal["plan_revision"] == manifest["plan_revision"] == receipt["plan_revision"]
    assert (
        seal["board_namespace"]
        == manifest["board_namespace"]
        == receipt["board_namespace"]
    )
    assert list(manifest["declared_outputs"]) == list(receipt["expected_outputs"])
    assert manifest["base_repositories"] == receipt["required_evidence"][
        "source_commit_tree_gitlinks"
    ]
    for name, expected in BASE_REPOSITORIES.items():
        forest_entry = seal["immutable_source_forest"][name]
        assert forest_entry["commit"] == expected["commit"]
        assert forest_entry["tree"] == expected["tree"]
        assert manifest["base_repositories"][name] == expected
