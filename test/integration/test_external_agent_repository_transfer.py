"""EAAEF-024: transfer modes reconstruct declared state or typed-refuse."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
KIT_ROOT = REPO_ROOT / "ipfs_kit_py"
if str(KIT_ROOT) not in sys.path:
    sys.path.insert(0, str(KIT_ROOT))

from ipfs_kit_py.repository_transfer.bundle import (  # noqa: E402
    TransferError,
    admit_transfer,
)

RECEIPT = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "external_agent_autonomous_execution_fabric"
    / "receipts"
    / "repository_transfer.json"
)
ARTIFACT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-offline-qualification-artifact@1"
)
PRODUCER_ARGV = (
    "python3",
    "-m",
    "pytest",
    "-q",
    "test/integration/test_external_agent_repository_transfer.py",
)
RECEIPT_FIELDS = {
    "artifact_cid",
    "evidence_mode",
    "external_repository_contacted",
    "host_path_typed_refusal_validated",
    "live_runtime_invoked",
    "managed_alias_contract_validated",
    "producer_argv",
    "producer_source_cid",
    "production_qualification_claimed",
    "qualification_scope",
    "qualification_status",
    "repository_transfer_performed",
    "schema",
    "task_completion_claimed",
    "task_id",
    "user_checkout_mutated",
}


def _producer_source_cid() -> str:
    return "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _validate_receipt(payload: dict[str, object]) -> None:
    assert set(payload) == RECEIPT_FIELDS
    assert payload["schema"] == ARTIFACT_SCHEMA
    assert payload["task_id"] == "EAAEF-024"
    assert payload["evidence_mode"] == "contract_fail_closed"
    assert payload["qualification_scope"] == "offline_repository_transfer_contract_only"
    assert payload["qualification_status"] == "not_live_qualified"
    assert payload["task_completion_claimed"] is False
    assert payload["production_qualification_claimed"] is False
    assert payload["live_runtime_invoked"] is False
    assert payload["repository_transfer_performed"] is False
    assert payload["external_repository_contacted"] is False
    assert payload["user_checkout_mutated"] is False
    assert payload["producer_argv"] == list(PRODUCER_ARGV)
    assert payload["producer_source_cid"] == _producer_source_cid()
    unsealed = dict(payload)
    artifact_cid = unsealed.pop("artifact_cid")
    assert artifact_cid == content_identity(unsealed)


def _write_receipt(payload: dict[str, object]) -> dict[str, object]:
    sealed = {
        **payload,
        "producer_argv": list(PRODUCER_ARGV),
        "producer_source_cid": _producer_source_cid(),
    }
    sealed["artifact_cid"] = content_identity(sealed)
    _validate_receipt(sealed)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(
        json.dumps(sealed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sealed


def test_admitted_modes_do_not_mutate_user_checkout(tmp_path) -> None:
    before = list(tmp_path.iterdir())
    req = admit_transfer(mode="managed_alias", locator="repos/core", alias="core")
    assert req.mode == "managed_alias"
    assert list(tmp_path.iterdir()) == before


def test_host_path_is_typed_refusal() -> None:
    with pytest.raises(TransferError, match="host paths"):
        admit_transfer(mode="git_bundle", locator="/home/user/src.git")


def test_write_offline_repository_transfer_receipt(tmp_path: Path) -> None:
    before = tuple(tmp_path.iterdir())
    request = admit_transfer(
        mode="managed_alias",
        locator="repos/core",
        alias="core",
    )
    assert request.mode == "managed_alias"
    assert tuple(tmp_path.iterdir()) == before
    with pytest.raises(TransferError, match="host paths"):
        admit_transfer(mode="git_bundle", locator="/home/user/src.git")

    receipt = _write_receipt(
        {
            "schema": ARTIFACT_SCHEMA,
            "task_id": "EAAEF-024",
            "evidence_mode": "contract_fail_closed",
            "qualification_scope": "offline_repository_transfer_contract_only",
            "qualification_status": "not_live_qualified",
            "task_completion_claimed": False,
            "production_qualification_claimed": False,
            "live_runtime_invoked": False,
            "repository_transfer_performed": False,
            "external_repository_contacted": False,
            "user_checkout_mutated": False,
            "managed_alias_contract_validated": True,
            "host_path_typed_refusal_validated": True,
        }
    )
    _validate_receipt(receipt)
