#!/usr/bin/env python3
"""Independently sign the current EAAEF-191 admission or no-go bundle.

Uses the trusted local-operator profile and host lifecycle root. The
prospective supervisor does not sign. Live launch remains fail-closed while
child S-epic artifacts are typed_missing.
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
    cid,
    collect_and_write,
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
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sign(key_path: Path, payload: object) -> tuple[str, str]:
    key = Ed25519PrivateKey.from_private_bytes(key_path.read_bytes())
    did = ed25519_did_key(key.public_key())
    signature = base64.b64encode(key.sign(_canonical(payload))).decode("ascii")
    return did, signature


def issue() -> dict[str, str]:
    collection = collect_and_write()
    child_decisions = {
        task_id: str(collection["decisions"][task_id])
        for task_id in RECEIPT_FILES
        if task_id != "EAAEF-191"
    }
    review = admission_bundle_review_payload(
        child_decisions=child_decisions,
        decision="no_go",
        launch_plan_allowed=False,
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
        "decision": "no_go",
    }
    BUNDLE_SIGNATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    BUNDLE_SIGNATURES_PATH.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(BUNDLE_SIGNATURES_PATH, stat.S_IRUSR)
    refreshed = collect_and_write()
    bundle = json.loads(
        (RECEIPT_DIR / RECEIPT_FILES["EAAEF-191"]).read_text(encoding="utf-8")
    )
    if bundle.get("decision") != "no_go":
        raise RuntimeError("signed bundle must remain no_go until children admit")
    if not bundle["evidence"].get("independent_signature_present"):
        raise RuntimeError("signed bundle did not verify on collect")
    if bundle["evidence"].get("launch_plan_allowed") is True:
        raise RuntimeError("signed no-go bundle must not allow live launch")
    return {
        "operator_did": operator_did,
        "security_reviewer_did": reviewer_did,
        "payload_sha256": artifact["payload_sha256"],
        "signatures_path": str(BUNDLE_SIGNATURES_PATH.relative_to(ROOT)),
        "decision": "no_go",
        "independent_signature_present": "true",
        "configured_board_launch": "false",
        "collection": json.dumps(refreshed["decisions"], sort_keys=True),
    }


def main() -> int:
    print(json.dumps(issue(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
