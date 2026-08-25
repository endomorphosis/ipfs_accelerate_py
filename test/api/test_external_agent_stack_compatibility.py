from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "docs/architecture/external_agent_autonomous_execution_fabric"
R1_PATH = CAMPAIGN / "stack_compatibility_manifest.json"
PROPOSAL_PATH = CAMPAIGN / "proposals/stack_compatibility_manifest.r2.json"
RECEIPT_PATH = CAMPAIGN / "receipts/stack_compatibility_verification.json"
BOARD_PATH = CAMPAIGN / "task_board.json"
WORKER_IMAGE_RECEIPT = CAMPAIGN / "receipts/host_admission/worker_image.json"
EXPECTED_REPOSITORIES = (
    "ipfs_accelerate_py",
    "ipfs_datasets_py",
    "ipfs_kit_py",
    "Mcp-Plus-Plus",
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


def test_r1_compatibility_input_is_not_overwritten() -> None:
    r1_bytes = R1_PATH.read_bytes()
    board = _load(BOARD_PATH)
    r1 = json.loads(r1_bytes.decode("utf-8"))
    assert r1["schema"] == "StackCompatibilityManifest@1"
    assert board["stack_compatibility_manifest_cid"] == _canonical_cid(r1)
    proposal = _load(PROPOSAL_PATH)
    assert proposal["overwrite_r1_in_place"] is False
    assert proposal["r1_manifest_cid"] == board["stack_compatibility_manifest_cid"]
    assert proposal["r1_input_path"] == str(R1_PATH.relative_to(ROOT))
    assert PROPOSAL_PATH.resolve() != R1_PATH.resolve()
    assert R1_PATH.read_bytes() == r1_bytes


def test_proposal_binds_roots_packages_and_admitted_oci() -> None:
    r1 = _load(R1_PATH)
    proposal = _load(PROPOSAL_PATH)
    image = _load(WORKER_IMAGE_RECEIPT)
    assert proposal["schema"] == "StackCompatibilityProposal@1"
    assert proposal["live_launch_allowed"] is False
    assert proposal["plan_r2_admitted"] is False
    assert proposal["configured_board_launch"] is False
    assert set(proposal["planning_integration_roots"]) == set(EXPECTED_REPOSITORIES)
    for name in EXPECTED_REPOSITORIES:
        for field in ("commit", "tree", "integration_branch"):
            assert proposal["planning_integration_roots"][name][field] == r1[
                "integration_roots"
            ][name][field]
    assert proposal["package_compatibility"] == r1["package_compatibility"]
    oci = proposal["admitted_bootstrap_oci"]
    assert oci["image_digest"] == image["evidence"]["image_digest"]
    assert oci["sbom_digest"] == image["evidence"]["sbom_digest"]
    assert oci["docker_socket_mounted"] is False
    assert oci["live_dispatch_claimed"] is False
    overlay = proposal["working_overlay"]
    assert overlay["clean_commit"] is False
    assert overlay["classification"] == "task_owned_working_tree_overlay"


def test_stack_compatibility_verification_receipt_is_current() -> None:
    r1_bytes = R1_PATH.read_bytes()
    proposal_before = PROPOSAL_PATH.read_bytes()
    r1 = json.loads(r1_bytes.decode("utf-8"))
    proposal = json.loads(proposal_before.decode("utf-8"))
    board = _load(BOARD_PATH)
    receipt = {
        "schema": "StackCompatibilityVerification@1",
        "task_id": "EAAEF-006",
        "decision": "proposal_bound_r1_unmutated",
        "r1_manifest_cid": _canonical_cid(r1),
        "proposal_cid": _canonical_cid(proposal),
        "board_stack_compatibility_manifest_cid": board[
            "stack_compatibility_manifest_cid"
        ],
        "r1_overwritten": False,
        "live_launch_allowed": False,
        "plan_r2_admitted": False,
        "admitted_bootstrap_oci": proposal["admitted_bootstrap_oci"]["image_digest"],
        "source_forest_root": proposal["source_forest_root"],
    }
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stored = _load(RECEIPT_PATH)
    assert stored["r1_overwritten"] is False
    assert stored["r1_manifest_cid"] == board["stack_compatibility_manifest_cid"]
    assert stored["live_launch_allowed"] is False
    assert stored["plan_r2_admitted"] is False
    assert R1_PATH.read_bytes() == r1_bytes
    assert PROPOSAL_PATH.read_bytes() == proposal_before
