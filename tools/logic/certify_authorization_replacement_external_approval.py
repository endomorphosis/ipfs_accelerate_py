#!/usr/bin/env python3
"""FVT-G232 external legal/IP/security approval envelope (observe-only).

``AuthorizationReplacementExternalApproval@1``

This surface never authors or forges approval.  Implementation agents may only:
* validate a supplied, public-safe, signed envelope, or
* emit a pending fail-closed assessment when no valid approval is bound.

Authorized external reviewers must supply the signed envelope that binds the
exact implementation tree, license inventory, clean-room record, residual
risks, expiry, scope, and revocation mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping

SCHEMA_VERSION: Final = "authorization-replacement-external-approval/v1"
INTERFACE: Final = "AuthorizationReplacementExternalApproval@1"
GOAL_ID: Final = "FVT-G232"
TASK_ID: Final = "FVT-099B"
PROGRAM: Final = "formal-verification-tactician/authorization-replacement-approval"
DEFAULT_RELATIVE: Final = Path(
    "docs/architecture/formal_verification_authorization_replacement_approval.json"
)
REQUIRED_BINDING_FIELDS: Final[tuple[str, ...]] = (
    "implementation_commit",
    "implementation_tree",
    "dependency_license_inventory_digest_sha256",
    "clean_room_record_digest_sha256",
    "patent_ip_disposition",
    "threat_model_digest_sha256",
    "semantic_receipt_digest_sha256",
    "adversarial_receipt_digest_sha256",
    "supported_platforms",
    "operational_controls",
    "residual_risks",
    "expiry_or_review_at",
    "approval_scope",
    "revocation_mechanism",
)
REQUIRED_SIGNERS: Final[tuple[str, ...]] = (
    "legal_ip_reviewer",
    "security_reviewer",
    "deployment_owner",
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ISO_Z = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


def content_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text(value: Any) -> str:
    return str(value or "").strip()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def build_pending_external_approval(
    *,
    observed_at: str | None = None,
    reason: str = "signed_external_approval_envelope_not_bound",
) -> dict[str, Any]:
    """Emit a public-safe pending envelope (never marks approval complete)."""

    timestamp = observed_at or datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "interface": INTERFACE,
        "goal_id": GOAL_ID,
        "task_id": TASK_ID,
        "program": PROGRAM,
        "observed_at": timestamp,
        "status": "external_approval_pending",
        "approval_complete": False,
        "legal_approval_complete": False,
        "security_approval_complete": False,
        "deployment_owner_approval_complete": False,
        "self_approval_forbidden": True,
        "agent_authored_approval_forbidden": True,
        "binding": {
            field: None for field in REQUIRED_BINDING_FIELDS
        },
        "signers": {
            role: {
                "identity": None,
                "signed": False,
                "signature_digest_sha256": None,
            }
            for role in REQUIRED_SIGNERS
        },
        "blockers": sorted(
            {
                reason,
                "legal_ip_review_missing",
                "security_review_missing",
                "deployment_owner_approval_missing",
                "signed_current_tree_envelope_missing",
            }
        ),
        "policy": {
            "completion_authority": "external",
            "implementation_agents_may_validate_only": True,
            "implementation_agents_may_not_author_or_forge": True,
            "missing_expired_wrong_tree_self_approved_unsigned_ambiguous_or_revoked_fail_closed": True,
        },
        "disclosures": {
            "does_not_complete_fvt_g219": True,
            "does_not_claim_microsoft_secpal_authority": True,
            "does_not_embed_privileged_review_material": True,
        },
    }
    payload["receipt_digest_sha256"] = content_digest(
        {key: value for key, value in payload.items() if key != "receipt_digest_sha256"}
    )
    return payload


def validate_external_approval(
    payload: Mapping[str, Any] | None,
    *,
    expected_implementation_commit: str | None = None,
    expected_implementation_tree: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fail-closed validation of a supplied G232 envelope."""

    failures: list[str] = []
    if not isinstance(payload, Mapping) or not payload:
        return {
            "valid": False,
            "approval_complete": False,
            "failures": ["approval_envelope_missing"],
            "status": "external_approval_pending",
        }

    if payload.get("schema_version") != SCHEMA_VERSION:
        failures.append("schema_version_mismatch")
    if payload.get("interface") != INTERFACE:
        failures.append("interface_mismatch")
    if payload.get("goal_id") != GOAL_ID:
        failures.append("goal_id_mismatch")

    declared = _text(payload.get("receipt_digest_sha256")).lower().removeprefix(
        "sha256:"
    )
    computed = content_digest(
        {
            key: value
            for key, value in payload.items()
            if key != "receipt_digest_sha256"
        }
    )
    if not _HEX64.fullmatch(declared) or declared != computed:
        failures.append("receipt_digest_mismatch")

    if payload.get("self_approval_forbidden") is not True:
        failures.append("self_approval_not_forbidden")
    if payload.get("agent_authored_approval_forbidden") is not True:
        failures.append("agent_authored_approval_not_forbidden")

    binding = payload.get("binding")
    binding = binding if isinstance(binding, Mapping) else {}
    for field in REQUIRED_BINDING_FIELDS:
        value = binding.get(field)
        if value in (None, "", (), []):
            failures.append(f"binding_missing:{field}")

    commit = _text(binding.get("implementation_commit"))
    tree = _text(binding.get("implementation_tree"))
    if expected_implementation_commit and commit != expected_implementation_commit:
        failures.append("implementation_commit_mismatch")
    if expected_implementation_tree and tree != expected_implementation_tree:
        failures.append("implementation_tree_mismatch")

    signers = payload.get("signers")
    signers = signers if isinstance(signers, Mapping) else {}
    for role in REQUIRED_SIGNERS:
        entry = signers.get(role)
        entry = entry if isinstance(entry, Mapping) else {}
        if not _text(entry.get("identity")):
            failures.append(f"signer_identity_missing:{role}")
        if entry.get("signed") is not True:
            failures.append(f"signer_not_signed:{role}")
        sig = _text(entry.get("signature_digest_sha256")).lower().removeprefix(
            "sha256:"
        )
        if not _HEX64.fullmatch(sig):
            failures.append(f"signer_signature_invalid:{role}")

    # Self-approval / agent identity heuristics.
    signer_identities = {
        _text((signers.get(role) or {}).get("identity")).lower()
        for role in REQUIRED_SIGNERS
    }
    if any(
        token in identity
        for identity in signer_identities
        for token in ("agent", "supervisor", "bot", "automation")
    ):
        failures.append("agent_identity_cannot_sign_external_approval")
    if len({identity for identity in signer_identities if identity}) < 3:
        failures.append("distinct_external_signers_required")

    expiry_raw = _text(binding.get("expiry_or_review_at"))
    if expiry_raw and not _ISO_Z.match(expiry_raw):
        failures.append("expiry_timestamp_unreadable")
    elif expiry_raw:
        try:
            expiry = datetime.fromisoformat(expiry_raw.replace("Z", "+00:00"))
            clock = now or datetime.now(timezone.utc)
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            if expiry <= clock:
                failures.append("approval_expired")
        except ValueError:
            failures.append("expiry_timestamp_unreadable")

    if payload.get("status") == "revoked" or payload.get("revoked") is True:
        failures.append("approval_revoked")

    approval_complete = bool(
        not failures
        and payload.get("approval_complete") is True
        and payload.get("legal_approval_complete") is True
        and payload.get("security_approval_complete") is True
        and payload.get("deployment_owner_approval_complete") is True
        and payload.get("status") == "external_approval_complete"
    )
    if payload.get("approval_complete") is True and not approval_complete:
        failures.append("approval_complete_claim_not_supported_by_evidence")

    return {
        "valid": not failures and approval_complete,
        "approval_complete": approval_complete,
        "failures": sorted(set(failures)),
        "status": (
            "external_approval_complete"
            if approval_complete
            else str(payload.get("status") or "external_approval_pending")
        ),
        "goal_id": GOAL_ID,
        "interface": INTERFACE,
    }


def observe_external_approval(
    repo_root: Path,
    *,
    expected_implementation_commit: str | None = None,
    expected_implementation_tree: str | None = None,
) -> dict[str, Any]:
    """Load the repository approval envelope and validate fail-closed."""

    path = repo_root / DEFAULT_RELATIVE
    payload = _load_json(path)
    result = validate_external_approval(
        payload,
        expected_implementation_commit=expected_implementation_commit,
        expected_implementation_tree=expected_implementation_tree,
    )
    result["path"] = DEFAULT_RELATIVE.as_posix()
    result["present"] = path.is_file()
    result["payload"] = payload
    return result


def write_pending_external_approval(
    repo_root: Path,
    *,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Write the pending envelope path (never marks complete)."""

    payload = build_pending_external_approval(observed_at=observed_at)
    path = repo_root / DEFAULT_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Observe or emit the FVT-G232 AuthorizationReplacementExternalApproval "
            "envelope (never forges approval)."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--write-pending",
        action="store_true",
        help="Write a pending fail-closed envelope at the evidence path",
    )
    parser.add_argument(
        "--observe",
        action="store_true",
        help="Validate the current repository envelope and print the result",
    )
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    if args.write_pending:
        payload = write_pending_external_approval(root)
        print(f"wrote {root / DEFAULT_RELATIVE}")
        print(
            "status=",
            payload["status"],
            "approval_complete=",
            payload["approval_complete"],
            sep="",
        )
    if args.observe or not args.write_pending:
        result = observe_external_approval(root)
        print(json.dumps(
            {
                key: value
                for key, value in result.items()
                if key != "payload"
            },
            indent=2,
            sort_keys=True,
        ))
        return 0 if result.get("approval_complete") is True else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
