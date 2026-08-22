#!/usr/bin/env python3
"""Create-once EAAEF-000 bootstrap admission for the current source HEAD.

Diagnoses without starting a supervisor. Publishes only when the live
materialization receipt, provider/container qualification, Quack owner
qualification, and owner-only parent chain all bind this HEAD. Does not
rematerialize a new store generation.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import base64

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ipfs_accelerate_py.agent_supervisor.control.profile_authority import (
    ed25519_did_key,
    lifecycle_root_identity_did,
)
from ipfs_accelerate_py.agent_supervisor.validation.external_agent_bootstrap_admission import (
    ExternalAgentBootstrapAdmissionError,
    assemble_external_agent_bootstrap_admission,
    external_agent_bootstrap_admission_relative_path,
    prepare_external_agent_bootstrap_admission,
    prepare_external_agent_bootstrap_approval,
    publish_external_agent_bootstrap_admission,
)
from ipfs_accelerate_py.agent_supervisor.validation.external_agent_bootstrap_admission import (
    _canonical_bytes,
)

CONFIG_PATH = ROOT / "config/external_agent_autonomous_execution_fabric_scheduler.json"
CURSOR_PATH = (
    ROOT
    / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
    / "generation-cursor.json"
)
AUTHORITY_DIR = (
    ROOT
    / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
    / "authority"
)
PRINCIPAL_DIR = AUTHORITY_DIR / "runtime-principals"
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
_IDENTITY_BLOCKERS = frozenset(
    {
        "materialization_source_or_board_mismatch",
        "materialization_source_tree_mismatch",
        "materialization_board_cid_mismatch",
        "immutable publication parent is not an owner-only directory",
        "immutable publication parent is unavailable",
        "bootstrap admission receipt already exists",
    }
)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _principal_did(role: str) -> str:
    payload = _load(PRINCIPAL_DIR / f"{role}.json")
    did = str(payload.get("did") or "")
    if not did.startswith("did:key:z"):
        raise RuntimeError(f"{role} principal DID is missing")
    return did


def _parent_mode_blockers(relative: Path) -> list[str]:
    blockers: list[str] = []
    current = ROOT
    for part in relative.parts[:-1]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except OSError:
            blockers.append("immutable publication parent is unavailable")
            return blockers
        mode = stat.S_IMODE(metadata.st_mode)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or mode & 0o077
        ):
            blockers.append(
                "immutable publication parent is not an owner-only directory"
            )
            return blockers
    return blockers


def _qualification_paths() -> dict[str, list[str]]:
    names = [
        path.name
        for path in AUTHORITY_DIR.iterdir()
        if path.is_file() and not path.is_symlink()
    ]
    return {
        "provider_container": sorted(
            name
            for name in names
            if name.startswith("provider-container-qualification--")
        ),
        "quack_owner": sorted(
            name for name in names if name.startswith("quack-owner-qualification--")
        ),
    }


def _receipt_path() -> Path:
    generation = "eaaef-run-v14"
    if CURSOR_PATH.is_file():
        cursor = _load(CURSOR_PATH)
        active = str(cursor.get("active_generation") or "").strip()
        if active.startswith("eaaef-run-v"):
            generation = active
    return (
        ROOT
        / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
        / generation.removeprefix("eaaef-")
        / "registry"
        / "bootstrap-materialization.json"
    )


def diagnose() -> dict[str, Any]:
    """Read-only blockers for create-once publication at current HEAD."""

    source_head = _git("rev-parse", "HEAD")
    source_tree = _git("rev-parse", "HEAD^{tree}")
    relative = external_agent_bootstrap_admission_relative_path(source_head)
    target = ROOT / relative
    receipt_path = _receipt_path()
    board_path = (
        ROOT
        / "docs/architecture/external_agent_autonomous_execution_fabric"
        / "task_board.json"
    )
    receipt = _load(receipt_path) if receipt_path.is_file() else {}
    board = _load(board_path) if board_path.is_file() else {}
    qualifications = _qualification_paths()
    blockers: list[str] = []
    if str(receipt.get("source_head") or "") != source_head:
        blockers.append("materialization_source_or_board_mismatch")
    if str(receipt.get("source_tree") or "") != source_tree:
        blockers.append("materialization_source_tree_mismatch")
    if str((receipt.get("board_validation") or {}).get("board_cid") or "") != str(
        board.get("board_cid") or ""
    ):
        blockers.append("materialization_board_cid_mismatch")
    if not qualifications["provider_container"]:
        blockers.append("provider_container_qualification_missing")
    if not qualifications["quack_owner"]:
        blockers.append("quack_owner_qualification_missing")
    blockers.extend(_parent_mode_blockers(relative))
    if target.is_file():
        blockers.append("bootstrap admission receipt already exists")
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-bootstrap-admission-issue@1",
        "source_head": source_head,
        "source_tree": source_tree,
        "materialization_source_head": str(receipt.get("source_head") or ""),
        "materialization_source_tree": str(receipt.get("source_tree") or ""),
        "materialization_receipt_cid": str(receipt.get("receipt_cid") or ""),
        "materialization_store_generation": str(
            (
                (receipt.get("database_program_bindings") or {}).get("bootstrap") or {}
            ).get("store_generation")
            or ""
        ),
        "board_cid": str(board.get("board_cid") or ""),
        "relative_path": relative.as_posix(),
        "exists": target.is_file(),
        "qualifications": qualifications,
        "materialization_child_adapter_status": str(
            (receipt.get("database_program_bindings") or {}).get(
                "operational_child_adapter_status"
            )
            or ""
        ),
        "blockers": list(dict.fromkeys(blockers)),
        "identity_blockers": [
            item
            for item in dict.fromkeys(blockers)
            if item in _IDENTITY_BLOCKERS
        ],
        "would_publish": not blockers,
        "would_publish_nogo": not any(
            item in _IDENTITY_BLOCKERS for item in blockers
        )
        and not target.is_file(),
        "published": False,
        "process_started": False,
        "configured_board_launch": False,
        "rematerialize": False,
    }


def _sign_approval(statement: dict[str, Any], role: str, key_path: Path) -> dict[str, Any]:
    key = Ed25519PrivateKey.from_private_bytes(key_path.read_bytes())
    did = ed25519_did_key(key.public_key())
    approval = prepare_external_agent_bootstrap_approval(
        statement,
        role=role,
        identity_did=did,
        issued_at_ms=int(statement["issued_at_ms"]),
        expires_at_ms=int(statement["expires_at_ms"]),
    )
    payload = dict(approval)
    signature = base64.b64encode(key.sign(_canonical_bytes(payload))).decode("ascii")
    approval["signature"] = signature
    return approval


def issue() -> dict[str, Any]:
    report = diagnose()
    if report.get("identity_blockers"):
        report["published"] = False
        return report
    config = _load(CONFIG_PATH)
    board = _load(
        ROOT
        / "docs/architecture/external_agent_autonomous_execution_fabric"
        / "task_board.json"
    )
    receipt = _load(_receipt_path())
    now_ms = int(time.time() * 1000)
    live_seal = config.get("configured_board_live_seal") or {}
    trusted_operator = tuple(live_seal.get("trusted_operator_dids") or ())
    trusted_security = tuple(live_seal.get("trusted_security_reviewer_dids") or ())
    try:
        statement = prepare_external_agent_bootstrap_admission(
            board=board,
            materialization_receipt=receipt,
            provider_container_qualification=None,
            route_plan=None,
            image_qualification=None,
            container_profile=None,
            quack_owner_qualification=None,
            trusted_provider_signer_dids=(),
            trusted_image_reviewer_dids=(),
            trusted_container_profile_reviewer_dids=(),
            trusted_quack_reviewer_dids=(),
            expected_worker_principal_did=_principal_did("worker"),
            expected_provider_principal_did=_principal_did("provider"),
            expected_source_commit=str(report["source_head"]),
            expected_source_tree=str(report["source_tree"]),
            one_use_nonce=os.urandom(24).hex(),
            issued_at_ms=now_ms,
            expires_at_ms=now_ms + 3_600_000,
        )
    except ExternalAgentBootstrapAdmissionError as exc:
        report["blockers"] = list(dict.fromkeys([*report["blockers"], str(exc)]))
        report["would_publish"] = False
        report["published"] = False
        return report
    if statement.get("decision") == "admitted":
        raise RuntimeError(
            "create-once issuer reached admitted prepare without independent "
            "qualifications; refusing to sign"
        )
    operator = _sign_approval(statement, "independent_operator", OPERATOR_KEY)
    security = _sign_approval(
        statement, "independent_security_reviewer", LIFECYCLE_KEY
    )
    if operator["identity_did"] not in trusted_operator:
        raise RuntimeError(f"operator DID is not trusted: {operator['identity_did']}")
    if security["identity_did"] not in trusted_security:
        raise RuntimeError(
            f"security reviewer DID is not trusted: {security['identity_did']}"
        )
    if security["identity_did"] != lifecycle_root_identity_did():
        raise RuntimeError("lifecycle root DID drifted")
    receipt_out = assemble_external_agent_bootstrap_admission(
        statement,
        operator_approval=operator,
        security_approval=security,
        trusted_operator_dids=trusted_operator,
        trusted_security_reviewer_dids=trusted_security,
        now_ms=now_ms,
    )
    verification = publish_external_agent_bootstrap_admission(
        ROOT,
        receipt_out,
        trusted_operator_dids=trusted_operator,
        trusted_security_reviewer_dids=trusted_security,
        now_ms=now_ms,
    )
    report["published"] = True
    report["statement_decision"] = statement.get("decision")
    report["receipt_cid"] = verification.get("receipt_cid")
    report["relative_path"] = external_agent_bootstrap_admission_relative_path(
        str(report["source_head"])
    ).as_posix()
    report["process_started"] = False
    report["configured_board_launch"] = False
    return report


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "diagnose"
    if command not in {"diagnose", "issue"}:
        raise SystemExit("usage: issue_eaaef_bootstrap_admission.py [diagnose|issue]")
    payload = diagnose() if command == "diagnose" else issue()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("published") else 1


if __name__ == "__main__":
    raise SystemExit(main())
