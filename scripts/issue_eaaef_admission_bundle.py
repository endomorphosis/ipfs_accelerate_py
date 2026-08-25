#!/usr/bin/env python3
"""Independently sign the current EAAEF-191 admission or no-go bundle.

Uses the trusted local-operator profile and host lifecycle root. The
prospective supervisor does not sign. The bundle may be admitted when 182-190
are independently admitted; configured-board-launch still does not start here.
"""

from __future__ import annotations

import base64
import json
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ipfs_accelerate_py.agent_supervisor.control.profile_authority import (
    ed25519_did_key,
    lifecycle_root_identity_did,
)
from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
    BUNDLE_SIGNATURES_PATH,
    BUNDLE_SIGNATURES_SCHEMA,
    RECEIPT_DIR,
    RECEIPT_FILES,
    TRUSTED_OPERATOR_DIDS,
    TRUSTED_SECURITY_REVIEWER_DIDS,
    admission_bundle_review_payload,
    admission_bundle_target_decision,
    cid,
    collect_and_write,
    load_admission_bundle_signatures,
    verify_admission_bundle_receipt,
)

OPERATOR_KEY = (
    Path.home()
    / ".ipfs_accelerate"
    / "agent_supervisor"
    / "local_profile"
    / "local_dev_profile.key"
)
LIFECYCLE_KEY = (
    Path.home()
    / ".local"
    / "state"
    / "ipfs_accelerate_py"
    / "local-profile-root-registry"
    / "lifecycle_root_ed25519.key"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sign(key_path: Path, payload: object) -> tuple[str, str]:
    key = Ed25519PrivateKey.from_private_bytes(key_path.read_bytes())
    did = ed25519_did_key(key.public_key())
    signature = base64.b64encode(key.sign(_canonical(payload))).decode("ascii")
    return did, signature


def issue() -> dict[str, str]:
    collect_and_write()
    child_receipts = {
        task_id: json.loads(
            (RECEIPT_DIR / filename).read_text(encoding="utf-8")
        )
        for task_id, filename in RECEIPT_FILES.items()
        if task_id != "EAAEF-191"
    }
    unsigned_bundle = json.loads(
        (RECEIPT_DIR / RECEIPT_FILES["EAAEF-191"]).read_text(encoding="utf-8")
    )
    evidence = unsigned_bundle.get("evidence") or {}
    child_decisions = {
        task_id: str(receipt.get("decision") or "")
        for task_id, receipt in child_receipts.items()
    }
    child_receipt_cids = {
        task_id: str(receipt.get("receipt_cid") or "")
        for task_id, receipt in child_receipts.items()
    }
    open_host_gates = [
        str(item) for item in evidence.get("inventory_open_host_gated") or ()
    ]
    bootstrap_cid = str(evidence.get("bootstrap_admission_statement_cid") or "")
    materialization_cid = str(evidence.get("materialization_receipt_cid") or "")
    decision = admission_bundle_target_decision(
        child_decisions=child_decisions,
        bootstrap_admission_statement_cid=bootstrap_cid,
        materialization_receipt_cid=materialization_cid,
    )
    review = admission_bundle_review_payload(
        child_decisions=child_decisions,
        child_receipt_cids=child_receipt_cids,
        decision=decision,
        launch_plan_allowed=False,
        source_head=str(unsigned_bundle.get("source_head") or ""),
        source_tree=str(unsigned_bundle.get("source_tree") or ""),
        board_namespace=str(unsigned_bundle.get("board_namespace") or ""),
        board_cid=str(unsigned_bundle.get("board_cid") or ""),
        bootstrap_admission_statement_cid=bootstrap_cid,
        materialization_receipt_cid=materialization_cid,
        inventory_open_host_gated=open_host_gates,
    )
    operator_did, operator_signature = _sign(OPERATOR_KEY, review)
    if operator_did not in TRUSTED_OPERATOR_DIDS:
        raise RuntimeError(f"operator DID is not trusted: {operator_did}")
    reviewer_payload = {
        **review,
        "operator_did": operator_did,
        "operator_signature": operator_signature,
    }
    reviewer_did, reviewer_signature = _sign(LIFECYCLE_KEY, reviewer_payload)
    if reviewer_did not in TRUSTED_SECURITY_REVIEWER_DIDS:
        raise RuntimeError(f"lifecycle root DID is not trusted: {reviewer_did}")
    if reviewer_did != lifecycle_root_identity_did():
        raise RuntimeError("lifecycle root DID drifted")
    artifact = {
        "schema": BUNDLE_SIGNATURES_SCHEMA,
        "operator_did": operator_did,
        "operator_signature": operator_signature,
        "security_reviewer_did": reviewer_did,
        "security_reviewer_signature": reviewer_signature,
        "payload_sha256": cid(review),
        "supervisor_signed": False,
        "configured_board_launch": False,
        "decision": decision,
    }
    BUNDLE_SIGNATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if BUNDLE_SIGNATURES_PATH.exists():
        os.chmod(BUNDLE_SIGNATURES_PATH, stat.S_IRUSR | stat.S_IWUSR)
        BUNDLE_SIGNATURES_PATH.unlink()
    BUNDLE_SIGNATURES_PATH.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(BUNDLE_SIGNATURES_PATH, stat.S_IRUSR)
    # Do not collect a second time here.  The DuckDB/Quack qualification
    # carries observation timing, so a new probe has a different receipt CID
    # and would invalidate the signatures that bind the exact first capture.
    verified_signatures = load_admission_bundle_signatures(
        child_decisions=child_decisions,
        child_receipt_cids=child_receipt_cids,
        decision=decision,
        launch_plan_allowed=False,
        source_head=str(unsigned_bundle.get("source_head") or ""),
        source_tree=str(unsigned_bundle.get("source_tree") or ""),
        board_namespace=str(unsigned_bundle.get("board_namespace") or ""),
        board_cid=str(unsigned_bundle.get("board_cid") or ""),
        bootstrap_admission_statement_cid=bootstrap_cid,
        materialization_receipt_cid=materialization_cid,
        inventory_open_host_gated=open_host_gates,
        signatures_path=BUNDLE_SIGNATURES_PATH,
    )
    if not all(verified_signatures.values()):
        raise RuntimeError("new admission-bundle signatures did not verify")
    bundle_evidence = {
        **evidence,
        **verified_signatures,
        "child_receipt_cids": child_receipt_cids,
        "independent_signature_present": True,
    }
    bundle = {
        **unsigned_bundle,
        "decision": decision,
        "evidence": bundle_evidence,
    }
    bundle.pop("receipt_cid", None)
    bundle["receipt_cid"] = cid(bundle)
    bundle_path = RECEIPT_DIR / RECEIPT_FILES["EAAEF-191"]
    bundle_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if bundle.get("decision") != decision:
        raise RuntimeError(
            f"signed bundle decision drifted: expected {decision} got {bundle.get('decision')}"
        )
    if not bundle["evidence"].get("independent_signature_present"):
        raise RuntimeError("signed bundle did not verify on collect")
    if bundle["evidence"].get("launch_plan_allowed") is True:
        raise RuntimeError("issuer must not start live launch")
    if bundle.get("process_started") is True or bundle.get("supervisor_process_started") is True:
        raise RuntimeError("issuer started a supervisor")
    verification = verify_admission_bundle_receipt(
        receipt_dir=RECEIPT_DIR,
        expected_source_head=str(bundle.get("source_head") or ""),
        expected_source_tree=str(bundle.get("source_tree") or ""),
        expected_board_namespace=str(bundle.get("board_namespace") or ""),
        expected_board_cid=str(bundle.get("board_cid") or ""),
    )
    expected_blockers = (
        []
        if decision == "admitted"
        else ["EAAEF-191 closed admission preconditions are not admitted"]
    )
    if (
        verification.get("decision") != decision
        or verification.get("target_decision") != decision
        or verification.get("blockers") != expected_blockers
    ):
        raise RuntimeError(
            "final signed bundle did not verify against the captured receipts: "
            + json.dumps(verification, sort_keys=True)
        )
    return {
        "operator_did": operator_did,
        "security_reviewer_did": reviewer_did,
        "payload_sha256": artifact["payload_sha256"],
        "signatures_path": str(BUNDLE_SIGNATURES_PATH.relative_to(ROOT)),
        "decision": decision,
        "independent_signature_present": "true",
        "configured_board_launch": "false",
        "collection": json.dumps(
            {**child_decisions, "EAAEF-191": decision}, sort_keys=True
        ),
    }


def main() -> int:
    print(json.dumps(issue(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
