"""FVT-G232 AuthorizationReplacementExternalApproval@1 fail-closed surface.

External legal/IP/security/deployment approval may only be observed. Agents
cannot author, forge, self-approve, or weaken the envelope.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
APPROVAL_TOOL = (
    REPO_ROOT
    / "tools"
    / "logic"
    / "certify_authorization_replacement_external_approval.py"
)
APPROVAL_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_authorization_replacement_approval.json"
)
INTERFACE = "AuthorizationReplacementExternalApproval@1"
GOAL_ID = "FVT-G232"


def _load(path: Path, name: str):
    for candidate in (REPO_ROOT, REPO_ROOT / "ipfs_datasets_py"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def approval_mod():
    return _load(APPROVAL_TOOL, "fvt_g232_external_approval")


def test_expected_outputs_and_constants_exist(approval_mod) -> None:
    assert APPROVAL_TOOL.is_file()
    assert APPROVAL_PATH.is_file()
    assert Path(__file__).is_file()
    assert approval_mod.INTERFACE == INTERFACE
    assert approval_mod.GOAL_ID == GOAL_ID
    assert (
        approval_mod.DEFAULT_RELATIVE.as_posix()
        == "docs/architecture/formal_verification_authorization_replacement_approval.json"
    )


def test_checked_envelope_is_pending_and_not_complete(approval_mod) -> None:
    payload = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
    assert payload["interface"] == INTERFACE
    assert payload["goal_id"] == GOAL_ID
    assert payload["approval_complete"] is False
    assert payload["legal_approval_complete"] is False
    assert payload["status"] == "external_approval_pending"
    assert payload["self_approval_forbidden"] is True
    assert payload["agent_authored_approval_forbidden"] is True
    result = approval_mod.validate_external_approval(payload)
    assert result["valid"] is False
    assert result["approval_complete"] is False
    assert result["failures"]


def test_observe_repository_envelope_fails_closed(approval_mod) -> None:
    result = approval_mod.observe_external_approval(REPO_ROOT)
    assert result["present"] is True
    assert result["approval_complete"] is False
    assert result["valid"] is False


def test_forged_complete_claim_without_signers_fails(approval_mod) -> None:
    payload = approval_mod.build_pending_external_approval()
    payload["status"] = "external_approval_complete"
    payload["approval_complete"] = True
    payload["legal_approval_complete"] = True
    payload["security_approval_complete"] = True
    payload["deployment_owner_approval_complete"] = True
    payload.pop("receipt_digest_sha256", None)
    payload["receipt_digest_sha256"] = approval_mod.content_digest(
        {
            key: value
            for key, value in payload.items()
            if key != "receipt_digest_sha256"
        }
    )
    result = approval_mod.validate_external_approval(payload)
    assert result["valid"] is False
    assert result["approval_complete"] is False
    assert any("signer" in item or "binding_missing" in item for item in result["failures"])


def test_agent_identity_signers_are_rejected(approval_mod) -> None:
    payload = approval_mod.build_pending_external_approval()
    payload["status"] = "external_approval_complete"
    payload["approval_complete"] = True
    payload["legal_approval_complete"] = True
    payload["security_approval_complete"] = True
    payload["deployment_owner_approval_complete"] = True
    payload["binding"] = {
        field: (
            "deadbeef" * 8
            if "digest" in field or field.endswith("_tree") or field.endswith("commit")
            else (
                ["linux-aarch64"]
                if field == "supported_platforms"
                else (
                    "2027-01-01T00:00:00Z"
                    if field == "expiry_or_review_at"
                    else f"value:{field}"
                )
            )
        )
        for field in approval_mod.REQUIRED_BINDING_FIELDS
    }
    payload["binding"]["implementation_commit"] = "a" * 40
    payload["binding"]["implementation_tree"] = "b" * 40
    payload["signers"] = {
        role: {
            "identity": f"automation-agent-{role}",
            "signed": True,
            "signature_digest_sha256": "c" * 64,
        }
        for role in approval_mod.REQUIRED_SIGNERS
    }
    payload.pop("receipt_digest_sha256", None)
    payload["receipt_digest_sha256"] = approval_mod.content_digest(
        {
            key: value
            for key, value in payload.items()
            if key != "receipt_digest_sha256"
        }
    )
    result = approval_mod.validate_external_approval(payload)
    assert result["valid"] is False
    assert "agent_identity_cannot_sign_external_approval" in result["failures"]


def test_post_remediation_observes_g232_without_authoring(
    approval_mod,
) -> None:
    cert_path = (
        REPO_ROOT
        / "tools"
        / "logic"
        / "certify_formal_verification_toolchains.py"
    )
    certifier = _load(cert_path, "fvt_g232_post_remediation_certifier")
    assert (
        certifier._external_authorization_replacement_approval_complete(REPO_ROOT)
        is False
    )
    # Optimistic reseal of the production replacement receipt cannot unlock G232.
    production_path = (
        REPO_ROOT
        / "docs"
        / "architecture"
        / "formal_verification_production_authorization_replacement_receipt.json"
    )
    production = json.loads(production_path.read_text(encoding="utf-8"))
    assert production.get("legal_approval_complete") is False
