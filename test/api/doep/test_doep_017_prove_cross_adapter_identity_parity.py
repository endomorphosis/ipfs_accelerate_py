"""Independent current-tree checks for DOEP-017 cross-adapter identity parity.

A worker or model assertion alone is insufficient. This module verifies the
plan-bound declared outputs and re-runs the primary cross-adapter identity
parity proof against the current tree.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
PRIMARY_PATH = ACCELERATE_ROOT / "test/api/doep/test_cross_adapter_identity.py"
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "outputs"
    / "DOEP-017.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "receipts"
    / "DOEP-017.json"
)

OWNER_RELATIVE_OUTPUTS = (
    "test/api/doep/test_cross_adapter_identity.py",
    "test/api/doep/test_doep_017_prove_cross_adapter_identity_parity.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-017.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-017.json",
)
TASK_CID = "sha256:6678d7998da7148d60dad27153be7ceb24debedbc2e7ca7835ed488d1c82485a"
PLAN_CID = "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_primary_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "doep017_cross_adapter_identity",
        PRIMARY_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_primary_module_proves_cross_adapter_identity_parity() -> None:
    primary = _load_primary_module()
    intent = primary._minimal_intent()
    receipts = primary.assert_cross_adapter_identity_parity(intent=intent)
    assert set(receipts) == set(primary.ADAPTER_TRANSPORTS)
    canonical = receipts["canonical"]
    for transport in ("python", "cli", "mcp", "mcpplusplus"):
        assert receipts[transport] == canonical
    surfaces = primary.adapter_surfaces_delegate_to_canonical_service()
    assert surfaces["interface"] == primary.CROSS_ADAPTER_IDENTITY_INTERFACE
    assert surfaces["submission_delegate"] == primary.CANONICAL_SUBMISSION_DELEGATE
    assert surfaces["canonical_service"] == "SupervisorIntentService@1"
    assert "class CompetingObjectiveService" not in PRIMARY_PATH.read_text(encoding="utf-8")


def test_manifest_and_candidate_receipt_bind_the_exact_current_tree_outputs() -> None:
    manifest = _load_json(OUTPUT_PATH)
    receipt = _load_json(RECEIPT_PATH)

    assert manifest["schema"] == "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"
    assert receipt["schema"] == "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"
    assert manifest["task_id"] == receipt["task_id"] == "DOEP-017"
    assert manifest["task_cid"] == receipt["task_cid"] == TASK_CID
    assert manifest["plan_cid"] == receipt["plan_cid"] == PLAN_CID
    assert manifest["plan_revision"] == receipt["plan_revision"] == "DOEP-PLAN-V5"
    assert (
        manifest["board_namespace"]
        == receipt["board_namespace"]
        == "agent-supervisor-direct-objective-and-event-driven-planning-v1"
    )
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert receipt["base_repositories"] == BASE_REPOSITORIES
    assert tuple(manifest["declared_outputs"]) == OWNER_RELATIVE_OUTPUTS
    assert tuple(receipt["expected_outputs"]) == OWNER_RELATIVE_OUTPUTS
    assert tuple(receipt["changed_paths"]) == OWNER_RELATIVE_OUTPUTS
    assert tuple(receipt["write_scope"]) == OWNER_RELATIVE_OUTPUTS
    assert receipt["outputs_present"] == {item: True for item in OWNER_RELATIVE_OUTPUTS}

    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["no_competing_subsystem_created"] is True
    assert manifest["completion_authoritative"] is False
    assert manifest["worker_completion_insufficient"] is True
    assert receipt["completion_authoritative"] is False
    assert receipt["worker_completion_insufficient"] is True
    assert receipt["no_competing_subsystem_created"] is True
    assert receipt["candidate_status"] == "implemented"

    parity = manifest["cross_adapter_identity_parity"]
    assert parity["identifier"] == "CrossAdapterObjectiveIdentityParity@1"
    assert parity["primary_module"] == (
        "test.api.doep.test_cross_adapter_identity"
    )
    assert parity["submission_delegate"].endswith("intent_service.submit_objective")
    assert parity["transports"] == [
        "canonical",
        "python",
        "cli",
        "mcp",
        "mcpplusplus",
    ]
    assert parity["identity_fields"] == [
        "receipt_id",
        "intent_id",
        "intent_sha256",
        "objective_id",
        "objective_cid",
        "objective_revision_cid",
    ]
    assert parity["competing_subsystem_created"] is False
    assert parity["completion_authority"] is False
    assert parity["callers_supply_authoritative_policy"] is False

    digests = receipt["path_digests"]
    assert digests[OWNER_RELATIVE_OUTPUTS[0]] == _sha256_file(PRIMARY_PATH)
    assert digests[OWNER_RELATIVE_OUTPUTS[1]] == _sha256_file(TEST_PATH)
    assert digests[OWNER_RELATIVE_OUTPUTS[2]] == _sha256_file(OUTPUT_PATH)
    assert OWNER_RELATIVE_OUTPUTS[3] not in digests

    evidence = receipt["required_evidence"]
    assert evidence["source_commit_tree_gitlinks"] == BASE_REPOSITORIES
    assert evidence["test_proof_results"]["validation_command"] == [
        "python3",
        "-m",
        "pytest",
        "test/api/doep/test_doep_017_prove_cross_adapter_identity_parity.py",
        "-q",
    ]
    assert evidence["verifier_admission"] == "pending_independent_fenced_supervisor"
    assert "Worker or model assertion alone is insufficient." in evidence["limitations"]
    assert receipt["supervisor_acceptance"]["completion_authoritative"] is False
    assert (
        receipt["supervisor_acceptance"]["state"]
        == "pending_independent_fenced_supervisor"
    )
    assert manifest["validation_profile"] == "doep-validation/DOEP-PLAN-V5/DOEP-017@1"
