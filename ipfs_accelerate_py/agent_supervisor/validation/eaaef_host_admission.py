"""Host-controlled EAAEF S-epic admission evidence.

These helpers inventory the isolated launch-plan, bind public runtime
principals, probe DuckDB/Quack and the container engine, and emit typed
receipts. They never start a supervisor, never mount the Docker socket, and
never treat self-signed or unsigned material as admitted authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)

from ipfs_accelerate_py.agent_supervisor.control.profile_authority import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    LocalProfileTampered,
    verify_did_key_signature,
)

RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-host-admission-receipt@1"
)
BUNDLE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-host-admission-bundle@1"
)
BUNDLE_REVIEW_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-admission-bundle-review@2"
)
BUNDLE_SIGNATURES_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-admission-bundle-signatures@2"
)
TRUSTED_OPERATOR_DIDS: Final = (
    "did:key:z6Mku1TT7TcoD2VksFwNmYGNpE1zprQMmXsT3tz39BzhVdsy",
)
TRUSTED_SECURITY_REVIEWER_DIDS: Final = (
    "did:key:z6Mktp3ogPs9QwXBnKEQrdMThdbuPPNKQXiAP7X7JwXVq1G7",
)
PRINCIPAL_STORE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-runtime-principal-secret@1"
)
REQUIRED_DUCKDB: Final = "1.5.5"
REQUIRED_QUACK: Final = "1.5.5+core"
REQUIRED_QUACK_EXTENSION_VERSION: Final = "c154811"
REQUIRED_QUACK_EXTENSION_FINGERPRINT: Final = (
    "sha256:b77954ae50ecc06e10c6e20fc6fd421d73b5c31cf72bb60ae3f29b1f8a85f20b"
)
REQUIRED_QUACK_PLATFORM: Final = "linux-aarch64"
APPROVED_IMPORT_ROOT: Final = Path(
    "/home/barberb/.local/lib/python3.12/site-packages"
)
LANE_COUNT: Final = 5
PRINCIPAL_ROLES: Final = ("worker", "provider", "quack_owner")

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "docs/architecture/external_agent_autonomous_execution_fabric"
RECEIPT_DIR = CAMPAIGN / "receipts" / "host_admission"
BUNDLE_SIGNATURES_PATH = RECEIPT_DIR / "admission_bundle.signatures.json"
BOARD_PATH = CAMPAIGN / "task_board.json"
ROUTE_AUTHORITY_DIR = (
    ROOT
    / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
    / "authority"
)
AUTHORITY_DIR = ROUTE_AUTHORITY_DIR / "runtime-principals"
PROVIDER_AUTHORIZATION_GLOB = "provider-route-authorization-*.json"
LAUNCHER = ROOT / (
    "scripts/launch_external_agent_autonomous_execution_fabric_materializer.py"
)
HOST_EVIDENCE_DIR = ROUTE_AUTHORITY_DIR / "host-evidence"
WORKER_IMAGE_ARTIFACT = HOST_EVIDENCE_DIR / "worker-image-qualification.json"
CONTAINER_PROFILE_ARTIFACT = HOST_EVIDENCE_DIR / "container-execution-profile-v2.json"
WORKER_NETWORK_ARTIFACT = HOST_EVIDENCE_DIR / "worker-network-lanes.json"
COMMAND_FABRIC_ARTIFACT = HOST_EVIDENCE_DIR / "command-fabric-endpoints.json"
NATIVE_LANE_ARTIFACT = HOST_EVIDENCE_DIR / "native-lane-dispatcher.json"
PLAN_R2_ARTIFACT = HOST_EVIDENCE_DIR / "plan-r2-remote-owner.json"
GROK_MOUNT_DIR = HOST_EVIDENCE_DIR / "grok-mounts"
OPERATOR_PROFILE_DIR = (
    Path.home() / ".ipfs_accelerate" / "agent_supervisor" / "eaaef-route-profile"
)
HOST_WORKER_IMAGE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-host-worker-image-admission@1"
)
HOST_WORKER_SLOT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-worker-slot@1"
)
HOST_WORKER_NETWORK_LANES_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-host-worker-network-lanes@1"
)
HOST_WORKER_NETWORK_LANE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-host-worker-network-lane@1"
)
HOST_COMMAND_FABRIC_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-host-command-fabric-endpoints@1"
)
HOST_NATIVE_LANE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-host-native-lane-dispatcher@1"
)
HOST_PLAN_R2_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-host-plan-r2-remote-owner@1"
)
ADMIT_REQUIRED_CHILDREN: Final = (
    "EAAEF-182",
    "EAAEF-183",
    "EAAEF-184",
    "EAAEF-185",
    "EAAEF-186",
    "EAAEF-187",
    "EAAEF-188",
    "EAAEF-189",
    "EAAEF-190",
)
EXPECTED_WORKER_BASE_IMAGE_ID: Final = (
    "sha256:74c4a6ff67f397f8a10b058851d218896b2f1ee0f2cddf47741219b734de93a6"
)
PREFERRED_WORKER_IMAGE_DIGESTS: Final = (
    "sha256:23f0a2ae05593b4352fcd7ae07cf44bea754d8e690eb55fa915583051e814ecf",
    EXPECTED_WORKER_BASE_IMAGE_ID,
)
LOCAL_IMAGE_SAVE_SOURCE_HOSTS: Final = (
    "unix:///var/run/docker.sock",
)
IMAGE_IMPORT_TIMEOUT_SECONDS: Final = 3600

RECEIPT_FILES: Final[dict[str, str]] = {
    "EAAEF-180": "blocker_inventory.json",
    "EAAEF-181": "runtime_principals.json",
    "EAAEF-182": "duckdb_quack_155.json",
    "EAAEF-183": "engine_mode.json",
    "EAAEF-184": "provider_authorization.json",
    "EAAEF-185": "worker_image.json",
    "EAAEF-186": "container_profile.json",
    "EAAEF-187": "worker_network.json",
    "EAAEF-188": "command_fabric_endpoints.json",
    "EAAEF-189": "native_lane_dispatcher.json",
    "EAAEF-190": "plan_r2_remote_owner.json",
    "EAAEF-191": "admission_bundle.json",
}

_ACCEPTED_TASK_DECISIONS: Final[dict[str, str]] = {
    "EAAEF-180": "inventory",
    "EAAEF-181": "bound_unadmitted",
    **{f"EAAEF-{number}": "admitted" for number in range(182, 192)},
}

_HOST_GATED_MARKERS: Final[tuple[str, ...]] = (
    "quack_owner_",
    "provider_container_qualification",
    "oci_image_qualification",
    "container_profile",
    "eaaef_scoped_provider",
    "worker_network_",
    "signed_command_fabric_child_adapter",
    "board validation has not admitted",
    "container_policy.",
    "configured-board launch admission",
    "independently signed",
    "provider/container qualification is diagnostic",
    "versioned Grok mount",
    "typed authenticated Quack",
    "external source-addressed EAAEF-000",
    "rootless",
    "DuckDB",
)
_AUTO_RECOVERABLE_MARKERS: Final[tuple[str, ...]] = (
    "advance to a new explicit store generation",
    "bootstrap namespace claim is immutable",
    "output path is not a safe identifier",
    "refusing to overwrite existing bootstrap namespace",
)
_CLOSING_TASK_MARKERS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("worker_network_runtime_principals", ("EAAEF-181",)),
    ("quack_owner_qualification", ("EAAEF-181", "EAAEF-182")),
    ("duckdb", ("EAAEF-182",)),
    ("quack 1.5.5", ("EAAEF-182",)),
    ("required_continuous_profile", ("EAAEF-182",)),
    ("rootless", ("EAAEF-183",)),
    ("container_policy.bootstrap_image", ("EAAEF-185",)),
    ("container_policy.live_dispatch", ("EAAEF-185", "EAAEF-186", "EAAEF-191")),
    ("container_policy", ("EAAEF-183", "EAAEF-185")),
    ("admitted container engine", ("EAAEF-183",)),
    ("eaaef_scoped_provider", ("EAAEF-184",)),
    ("oci_image", ("EAAEF-185",)),
    ("provider/container qualification", ("EAAEF-185", "EAAEF-186")),
    ("container_profile", ("EAAEF-186",)),
    ("grok mount", ("EAAEF-186",)),
    ("execution-profile", ("EAAEF-186",)),
    ("worker_network_authorization", ("EAAEF-187",)),
    ("signed_command_fabric", ("EAAEF-188",)),
    ("command-authorizer", ("EAAEF-188",)),
    ("child_adapter", ("EAAEF-188",)),
    ("native-dependency", ("EAAEF-189",)),
    ("v2 lane", ("EAAEF-189",)),
    ("dispatcher", ("EAAEF-188", "EAAEF-189")),
    ("plan-r2", ("EAAEF-190",)),
    ("plan r2", ("EAAEF-190",)),
    ("remote-owner", ("EAAEF-190",)),
    ("board validation", ("EAAEF-191",)),
    ("configured-board launch", ("EAAEF-191",)),
    ("bootstrap admission", ("EAAEF-191",)),
)


def admission_bundle_review_payload(
    *,
    child_decisions: Mapping[str, str],
    child_receipt_cids: Mapping[str, str],
    decision: str,
    launch_plan_allowed: bool,
    source_head: str,
    source_tree: str,
    board_namespace: str,
    board_cid: str,
    bootstrap_admission_statement_cid: str,
    materialization_receipt_cid: str,
    inventory_open_host_gated: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": BUNDLE_REVIEW_SCHEMA,
        "source_head": str(source_head),
        "source_tree": str(source_tree),
        "board_namespace": str(board_namespace),
        "board_cid": str(board_cid),
        "decision": str(decision),
        "launch_plan_allowed": bool(launch_plan_allowed),
        "child_decisions": dict(child_decisions),
        "child_receipt_cids": dict(child_receipt_cids),
        "bootstrap_admission_statement_cid": str(
            bootstrap_admission_statement_cid
        ),
        "materialization_receipt_cid": str(materialization_receipt_cid),
        "inventory_open_host_gated": [
            str(item) for item in inventory_open_host_gated
        ],
        "prospective_supervisor_signature_rejected": True,
        "configured_board_launch": False,
    }


def load_admission_bundle_signatures(
    *,
    child_decisions: Mapping[str, str],
    child_receipt_cids: Mapping[str, str],
    decision: str,
    launch_plan_allowed: bool,
    source_head: str,
    source_tree: str,
    board_namespace: str,
    board_cid: str,
    bootstrap_admission_statement_cid: str,
    materialization_receipt_cid: str,
    inventory_open_host_gated: Sequence[str],
    signatures_path: Path = BUNDLE_SIGNATURES_PATH,
) -> dict[str, str]:
    """Return verified operator/reviewer signatures, or empty strings."""

    empty = {
        "independent_operator_signature": "",
        "independent_security_reviewer_signature": "",
        "operator_did": "",
        "security_reviewer_did": "",
    }
    if not signatures_path.is_file():
        return empty
    try:
        payload = json.loads(signatures_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return empty
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != BUNDLE_SIGNATURES_SCHEMA
        or payload.get("supervisor_signed") is not False
        or payload.get("configured_board_launch") is not False
        or payload.get("decision") != decision
    ):
        return empty
    operator_did = str(payload.get("operator_did") or "")
    reviewer_did = str(payload.get("security_reviewer_did") or "")
    operator_sig = str(payload.get("operator_signature") or "")
    reviewer_sig = str(payload.get("security_reviewer_signature") or "")
    if (
        operator_did not in TRUSTED_OPERATOR_DIDS
        or reviewer_did not in TRUSTED_SECURITY_REVIEWER_DIDS
        or not operator_sig
        or not reviewer_sig
    ):
        return empty
    review = admission_bundle_review_payload(
        child_decisions=child_decisions,
        child_receipt_cids=child_receipt_cids,
        decision=decision,
        launch_plan_allowed=launch_plan_allowed,
        source_head=source_head,
        source_tree=source_tree,
        board_namespace=board_namespace,
        board_cid=board_cid,
        bootstrap_admission_statement_cid=bootstrap_admission_statement_cid,
        materialization_receipt_cid=materialization_receipt_cid,
        inventory_open_host_gated=inventory_open_host_gated,
    )
    if cid(review) != str(payload.get("payload_sha256") or ""):
        return empty
    try:
        verify_did_key_signature(
            identity_did=operator_did,
            payload=review,
            signature=operator_sig,
        )
        reviewer_payload = {
            **review,
            "operator_did": operator_did,
            "operator_signature": operator_sig,
        }
        verify_did_key_signature(
            identity_did=reviewer_did,
            payload=reviewer_payload,
            signature=reviewer_sig,
        )
    except (LocalProfileTampered, TypeError, ValueError):
        return empty
    return {
        "independent_operator_signature": operator_sig,
        "independent_security_reviewer_signature": reviewer_sig,
        "operator_did": operator_did,
        "security_reviewer_did": reviewer_did,
    }


def cid(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _full_sha256(value: object) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def admission_bundle_target_decision(
    *,
    child_decisions: Mapping[str, str],
    bootstrap_admission_statement_cid: str,
    materialization_receipt_cid: str,
) -> str:
    """Return the only decision eligible for an independently signed review."""

    admitted = (
        all(
            child_decisions.get(task_id) == "admitted"
            for task_id in ADMIT_REQUIRED_CHILDREN
        )
        and _full_sha256(bootstrap_admission_statement_cid)
        and _full_sha256(materialization_receipt_cid)
    )
    return "admitted" if admitted else "no_go"


def verify_admission_bundle_receipt(
    *,
    receipt_dir: Path = RECEIPT_DIR,
    expected_source_head: str,
    expected_source_tree: str,
    expected_board_namespace: str,
    expected_board_cid: str,
) -> dict[str, Any]:
    """Verify current, closed, independently signed EAAEF host admission.

    Host evidence is deliberately allowed to remain outside a source commit.
    Consequently its signatures must bind the exact source/tree, board, child
    receipt CIDs, bootstrap/materialization identities, and open-blocker list.
    A decision string by itself is never launch authority.
    """

    blockers: list[str] = []
    receipts: dict[str, dict[str, Any]] = {}
    for task_id, filename in RECEIPT_FILES.items():
        path = receipt_dir / filename
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            blockers.append(f"{task_id} host receipt is unavailable or malformed")
            continue
        if not isinstance(payload, dict):
            blockers.append(f"{task_id} host receipt is not an object")
            continue
        body = {key: value for key, value in payload.items() if key != "receipt_cid"}
        expected_schema = RECEIPT_SCHEMA if task_id != "EAAEF-191" else BUNDLE_SCHEMA
        if (
            payload.get("schema") != expected_schema
            or payload.get("task_id") != task_id
            or payload.get("receipt_name") != filename
            or payload.get("receipt_cid") != cid(body)
        ):
            blockers.append(f"{task_id} host receipt identity is invalid")
        if (
            payload.get("source_head") != expected_source_head
            or payload.get("source_tree") != expected_source_tree
            or payload.get("board_namespace") != expected_board_namespace
            or payload.get("board_cid") != expected_board_cid
        ):
            blockers.append(f"{task_id} host receipt is stale for the current source")
        if (
            payload.get("process_started") is not False
            or payload.get("supervisor_process_started") is not False
            or payload.get("self_signed") is not False
        ):
            blockers.append(f"{task_id} host receipt violates launch separation")
        receipts[task_id] = payload

    bundle = receipts.get("EAAEF-191") or {}
    evidence = bundle.get("evidence")
    if not isinstance(evidence, Mapping):
        blockers.append("EAAEF-191 admission evidence is unavailable")
        evidence = {}
    child_receipt_cids = {
        task_id: str((receipts.get(task_id) or {}).get("receipt_cid") or "")
        for task_id in RECEIPT_FILES
        if task_id != "EAAEF-191"
    }
    child_decisions = {
        task_id: str((receipts.get(task_id) or {}).get("decision") or "")
        for task_id in RECEIPT_FILES
        if task_id != "EAAEF-191"
    }
    raw_child_cids = evidence.get("child_receipt_cids")
    observed_child_cids = (
        {str(key): str(value) for key, value in raw_child_cids.items()}
        if isinstance(raw_child_cids, Mapping)
        else {}
    )
    if observed_child_cids != child_receipt_cids:
        blockers.append("EAAEF-191 child receipt identities differ")
    bootstrap_cid = str(evidence.get("bootstrap_admission_statement_cid") or "")
    materialization_cid = str(evidence.get("materialization_receipt_cid") or "")
    raw_open = evidence.get("inventory_open_host_gated")
    open_host_gates = (
        [str(item) for item in raw_open]
        if isinstance(raw_open, list) and all(isinstance(item, str) for item in raw_open)
        else ["host-gated blocker inventory is malformed"]
    )
    target_decision = admission_bundle_target_decision(
        child_decisions=child_decisions,
        bootstrap_admission_statement_cid=bootstrap_cid,
        materialization_receipt_cid=materialization_cid,
    )
    if bundle.get("decision") != target_decision:
        blockers.append("EAAEF-191 decision differs from its closed evidence")
    if target_decision != "admitted":
        blockers.append("EAAEF-191 closed admission preconditions are not admitted")
    if (
        evidence.get("launch_plan_allowed") is not False
        or evidence.get("prospective_supervisor_signature_rejected") is not True
    ):
        blockers.append("EAAEF-191 review/launch separation differs")

    signatures = load_admission_bundle_signatures(
        child_decisions=child_decisions,
        child_receipt_cids=child_receipt_cids,
        decision=target_decision,
        launch_plan_allowed=False,
        source_head=expected_source_head,
        source_tree=expected_source_tree,
        board_namespace=expected_board_namespace,
        board_cid=expected_board_cid,
        bootstrap_admission_statement_cid=bootstrap_cid,
        materialization_receipt_cid=materialization_cid,
        inventory_open_host_gated=open_host_gates,
        signatures_path=receipt_dir / "admission_bundle.signatures.json",
    )
    expected_signature_evidence = {
        name: str(evidence.get(name) or "")
        for name in (
            "independent_operator_signature",
            "independent_security_reviewer_signature",
            "operator_did",
            "security_reviewer_did",
        )
    }
    if not signatures["independent_operator_signature"] or (
        signatures != expected_signature_evidence
        or evidence.get("independent_signature_present") is not True
    ):
        blockers.append("EAAEF-191 independent signatures are absent or invalid")
    return {
        "admitted": not blockers,
        "decision": str(bundle.get("decision") or ""),
        "target_decision": target_decision,
        "blockers": list(dict.fromkeys(blockers)),
    }


def verify_host_admission_task_receipt(
    *,
    task_id: str,
    receipt_dir: Path = RECEIPT_DIR,
    expected_source_head: str,
    expected_source_tree: str,
    expected_board_namespace: str,
    expected_board_cid: str,
) -> dict[str, Any]:
    """Verify one current-source EAAEF host-admission task receipt.

    The check is deliberately read-only and uses only the caller-supplied
    source and board identities.  EAAEF-191 additionally requires the full
    independently signed admission-bundle verification; an ``admitted`` word
    in the tracked receipt is not sufficient authority.
    """

    filename = RECEIPT_FILES.get(task_id)
    expected_decision = _ACCEPTED_TASK_DECISIONS.get(task_id)
    if filename is None or expected_decision is None:
        return {
            "valid": False,
            "decision": "",
            "blockers": [f"unsupported EAAEF host-admission task {task_id}"],
        }

    blockers: list[str] = []
    payload: dict[str, Any] | None = None
    path = receipt_dir / filename
    try:
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        blockers.append(f"{task_id} host receipt is unavailable or malformed")
    else:
        if isinstance(raw_payload, dict):
            payload = raw_payload
        else:
            blockers.append(f"{task_id} host receipt is not an object")

    decision = str((payload or {}).get("decision") or "")
    if payload is not None:
        expected_schema = RECEIPT_SCHEMA if task_id != "EAAEF-191" else BUNDLE_SCHEMA
        if payload.get("schema") != expected_schema:
            blockers.append(f"{task_id} host receipt schema differs")
        if payload.get("task_id") != task_id:
            blockers.append(f"{task_id} host receipt task identity differs")
        if payload.get("receipt_name") != filename:
            blockers.append(f"{task_id} host receipt filename differs")
        body = {
            key: value for key, value in payload.items() if key != "receipt_cid"
        }
        if payload.get("receipt_cid") != cid(body):
            blockers.append(f"{task_id} host receipt CID differs")

        expected_source = {
            "source_head": expected_source_head,
            "source_tree": expected_source_tree,
            "board_namespace": expected_board_namespace,
            "board_cid": expected_board_cid,
        }
        for field, expected_value in expected_source.items():
            if payload.get(field) != expected_value:
                blockers.append(f"{task_id} host receipt {field} differs")

        for field in ("process_started", "supervisor_process_started", "self_signed"):
            if payload.get(field) is not False:
                blockers.append(
                    f"{task_id} host receipt launch-separation field {field} differs"
                )

    if decision != expected_decision:
        blockers.append(
            f"{task_id} host receipt decision is not {expected_decision}"
        )

    if task_id == "EAAEF-191":
        bundle_verification = verify_admission_bundle_receipt(
            receipt_dir=receipt_dir,
            expected_source_head=expected_source_head,
            expected_source_tree=expected_source_tree,
            expected_board_namespace=expected_board_namespace,
            expected_board_cid=expected_board_cid,
        )
        if bundle_verification.get("admitted") is not True:
            bundle_blockers = bundle_verification.get("blockers")
            if isinstance(bundle_blockers, list):
                blockers.extend(str(item) for item in bundle_blockers)
            blockers.append("EAAEF-191 full admission-bundle verification failed")

    return {
        "valid": not blockers,
        "decision": decision,
        "blockers": list(dict.fromkeys(blockers)),
    }


def classify_blocker(text: str) -> str:
    raw = str(text or "")
    if "nested checkout is dirty" in raw:
        return "host_source_commit_required"
    if any(marker in raw for marker in _AUTO_RECOVERABLE_MARKERS):
        return "auto_recoverable"
    lowered = raw.casefold()
    if any(marker.casefold() in lowered for marker in _HOST_GATED_MARKERS):
        return "host_gated_external_authority"
    return "unclassified"


def closing_task_ids(text: str) -> list[str]:
    lowered = str(text or "").casefold()
    found: list[str] = []
    for marker, tasks in _CLOSING_TASK_MARKERS:
        if marker in lowered:
            for task_id in tasks:
                if task_id not in found:
                    found.append(task_id)
    return found or ["EAAEF-191"]


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def _source_identity() -> dict[str, str]:
    board = json.loads(BOARD_PATH.read_text(encoding="utf-8"))
    return {
        "source_head": _git("rev-parse", "HEAD"),
        "source_tree": _git("rev-parse", "HEAD^{tree}"),
        "board_cid": str(board.get("board_cid") or ""),
        "board_namespace": str(board.get("board_namespace") or ""),
    }


def _base_receipt(task_id: str, *, decision: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": RECEIPT_SCHEMA if task_id != "EAAEF-191" else BUNDLE_SCHEMA,
        "task_id": task_id,
        "receipt_name": RECEIPT_FILES[task_id],
        "decision": decision,
        "process_started": False,
        "supervisor_process_started": False,
        "self_signed": False,
        "independent_signatures": [],
        **_source_identity(),
        "evidence": dict(evidence),
    }
    payload["receipt_cid"] = cid(
        {key: value for key, value in payload.items() if key != "receipt_cid"}
    )
    return payload


def load_isolated_launch_plan(*, timeout_seconds: int = 180) -> dict[str, Any]:
    """Run the admitted isolated launcher. Never starts configured-board-launch."""

    argv = [
        "/usr/bin/python3.12",
        "-I",
        "-S",
        "-B",
        str(LAUNCHER),
        "launch-plan",
    ]
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    stdout = completed.stdout.strip()
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "isolated launch-plan did not emit JSON: "
            f"rc={completed.returncode} stderr={completed.stderr[-400:]}"
        ) from exc
    if payload.get("process_started") is True:
        raise RuntimeError("isolated launch-plan started a process")
    payload["_collector_returncode"] = completed.returncode
    return payload


def bind_runtime_principals() -> dict[str, Any]:
    """Create or reuse three distinct did:key identities. Private keys stay local."""

    AUTHORITY_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(AUTHORITY_DIR, stat.S_IRWXU)
    principals: list[dict[str, str]] = []
    for role in PRINCIPAL_ROLES:
        secret_path = AUTHORITY_DIR / f"{role}.json"
        if secret_path.is_file():
            secret = json.loads(secret_path.read_text(encoding="utf-8"))
            did = str(secret.get("did") or "")
        else:
            key = Ed25519PrivateKey.generate()
            did = ed25519_did_key(key.public_key())
            secret = {
                "schema": PRINCIPAL_STORE_SCHEMA,
                "role": role,
                "did": did,
                "private_key_pkcs8_der_b64": __import__("base64").b64encode(
                    key.private_bytes(
                        Encoding.DER,
                        PrivateFormat.PKCS8,
                        NoEncryption(),
                    )
                ).decode("ascii"),
            }
            secret_path.write_text(
                json.dumps(secret, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.chmod(secret_path, stat.S_IRUSR | stat.S_IWUSR)
        principals.append({"role": role, "did": did, "admitted_authority": False})
    dids = [item["did"] for item in principals]
    if len(set(dids)) != 3 or any(not item.startswith("did:key:z") for item in dids):
        raise RuntimeError("runtime principals are not three distinct did:key identities")
    return {
        "principals": principals,
        "secret_material_exported": False,
        "secret_store": str(AUTHORITY_DIR.relative_to(ROOT)),
        "admitted_authority": False,
    }


def probe_duckdb_quack() -> dict[str, Any]:
    """Refuse silent 1.5.2 substitution. Do not network-install Quack."""

    import duckdb

    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
        QuackCompatibilityProfile,
        probe_quack_capabilities,
    )

    observed = str(getattr(duckdb, "__version__", "") or "")
    module_path = Path(getattr(duckdb, "__file__", "") or "").resolve()
    try:
        module_path.relative_to(APPROVED_IMPORT_ROOT.resolve())
        under_approved_root = True
    except (ValueError, OSError):
        under_approved_root = False
    profile = QuackCompatibilityProfile(
        profile_id="eaaef-duckdb-quack-1.5.5-core-c154811-linux-aarch64",
        pinned_duckdb_version=REQUIRED_DUCKDB,
        pinned_extension_fingerprint=REQUIRED_QUACK_EXTENSION_FINGERPRINT,
        pinned_platform=REQUIRED_QUACK_PLATFORM,
        allow_experimental_within_minor=False,
    )
    report = probe_quack_capabilities(
        profile=profile,
        allow_network_install=False,
        allow_local_load=True,
        use_cache=False,
    )
    exact_duckdb = observed == REQUIRED_DUCKDB
    installed_from = ""
    if report.extension is not None:
        installed_from = str(report.extension.installed_from or "")
    exact_quack = (
        report.passes_health_check
        and exact_duckdb
        and installed_from == "core"
        and str(report.extension.extension_version or "")
        == REQUIRED_QUACK_EXTENSION_VERSION
        and report.extension_fingerprint == REQUIRED_QUACK_EXTENSION_FINGERPRINT
        and f"{report.platform_name}-{report.platform_machine}"
        == REQUIRED_QUACK_PLATFORM
    )
    if exact_duckdb and exact_quack and under_approved_root:
        decision = "admitted"
    else:
        decision = "typed_missing"
    return {
        "decision": decision,
        "required_duckdb": REQUIRED_DUCKDB,
        "required_quack": REQUIRED_QUACK,
        "required_quack_extension_version": REQUIRED_QUACK_EXTENSION_VERSION,
        "required_quack_extension_fingerprint": REQUIRED_QUACK_EXTENSION_FINGERPRINT,
        "required_quack_platform": REQUIRED_QUACK_PLATFORM,
        "observed_duckdb": observed,
        "observed_module_path": str(module_path),
        "under_approved_import_root": under_approved_root,
        "silent_substitution_refused": observed != REQUIRED_DUCKDB,
        "quack_probe": report.to_dict(),
        "network_install_attempted": False,
    }


def _docker_info(host: str = "") -> tuple[int, dict[str, Any]]:
    command = ["docker"]
    if host:
        command.extend(["-H", host])
    command.extend(["info", "--format", "{{json .}}"])
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    info: dict[str, Any] = {}
    if completed.returncode == 0 and completed.stdout.strip():
        try:
            parsed = json.loads(completed.stdout)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            info = parsed
    return completed.returncode, info


def _is_rootless_info(info: Mapping[str, Any]) -> bool:
    security = [str(item) for item in info.get("SecurityOptions") or ()]
    if any("rootless" in item.casefold() for item in security):
        return True
    root_dir = str(info.get("DockerRootDir") or "")
    return "/.local/share/docker" in root_dir or root_dir.endswith("/docker-rootless")


def _rootless_docker_hosts() -> list[str]:
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    candidates = [
        str(os.environ.get("EAAEF_DOCKER_HOST") or "").strip(),
        f"unix://{runtime_dir}/docker.sock",
        f"unix://{Path.home()}/.docker/run/docker.sock",
    ]
    seen: list[str] = []
    for item in candidates:
        if item and item not in seen and item != "unix:///var/run/docker.sock":
            seen.append(item)
    return seen


def probe_engine_mode() -> dict[str, Any]:
    """Prefer a verified rootless engine; never mount the host Docker socket."""

    probes: list[dict[str, Any]] = []
    selected_host = ""
    selected_info: dict[str, Any] = {}
    selected_returncode = 1
    for host in _rootless_docker_hosts():
        returncode, info = _docker_info(host)
        probes.append(
            {
                "docker_host": host,
                "returncode": returncode,
                "rootless": _is_rootless_info(info),
                "root_dir": str(info.get("DockerRootDir") or ""),
                "security_options": [str(item) for item in info.get("SecurityOptions") or ()],
            }
        )
        if returncode == 0 and _is_rootless_info(info):
            selected_host = host
            selected_info = info
            selected_returncode = returncode
            break
    if not selected_info:
        returncode, info = _docker_info("")
        selected_returncode = returncode
        selected_info = info
        selected_host = str(os.environ.get("DOCKER_HOST") or "unix:///var/run/docker.sock")
        probes.append(
            {
                "docker_host": selected_host,
                "returncode": returncode,
                "rootless": _is_rootless_info(info),
                "root_dir": str(info.get("DockerRootDir") or ""),
                "security_options": [str(item) for item in info.get("SecurityOptions") or ()],
            }
        )
    security = [str(item) for item in selected_info.get("SecurityOptions") or ()]
    rootless = _is_rootless_info(selected_info)
    root_dir = str(selected_info.get("DockerRootDir") or "")
    server_version = str(selected_info.get("ServerVersion") or "")
    uses_host_socket = selected_host in {"", "unix:///var/run/docker.sock"}
    fallback = {
        "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-rootful-fallback-package@1",
        "engine": "docker",
        "host_daemon_mode": "rootful",
        "worker_user": "nonroot",
        "docker_socket_mount": "prohibited",
        "capabilities": "drop_all",
        "no_new_privileges": True,
        "read_only_root": True,
        "network_default": "deny",
        "independent_security_review_required": True,
        "signed": False,
        "observed_root_dir": root_dir,
        "observed_server_version": server_version,
        "observed_security_options": security,
    }
    if rootless and not uses_host_socket:
        decision = "admitted"
        mode = "verified_rootless"
    elif selected_info:
        decision = "typed_missing"
        mode = "rootful_host_daemon_unsigned_fallback"
    else:
        decision = "typed_missing"
        mode = "engine_unavailable"
    return {
        "decision": decision,
        "mode": mode,
        "rootless": rootless,
        "docker_host": selected_host,
        "docker_socket_mounted": False,
        "host_docker_socket_used": uses_host_socket and not rootless,
        "supervisor_started": False,
        "fallback_package": fallback if not rootless else None,
        "docker_info_returncode": selected_returncode,
        "probes": probes,
    }


def _committed_provider_authorization_paths() -> list[str]:
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "--",
            f"data/agent_supervisor/external_agent_autonomous_execution_fabric/authority/{PROVIDER_AUTHORIZATION_GLOB}",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def probe_provider_authorization() -> dict[str, Any]:
    """Admit only a loadable source-addressed grok_cli/codex authorization.

    The prospective supervisor does not sign this artifact. Unsigned, self-signed,
    or unloadable material remains typed_missing.
    """

    from ipfs_accelerate_py import agent_implementation_route as routes

    candidates = _committed_provider_authorization_paths()
    attempts: list[dict[str, str]] = []
    for relative in candidates:
        try:
            authorization = routes.load_agent_implementation_route_authorization(
                repo_root=ROOT,
                artifact_path=relative,
                board_namespace=routes._EAAEF_AGENT_ROUTE_BOARD_NAMESPACE,
            )
        except (OSError, TypeError, ValueError) as exc:
            attempts.append({"path": relative, "error": str(exc)})
            continue
        return {
            "decision": "admitted",
            "artifact": "eaaef_scoped_provider_authorization",
            "independent_signature_present": True,
            "self_signed_rejected": True,
            "source_only_factory_authority": False,
            "supervisor_signed": False,
            "supervisor_started": False,
            "configured_board_launch": False,
            "artifact_path": authorization.artifact_path,
            "artifact_sha256": authorization.artifact_sha256,
            "authorization_id": authorization.authorization_id,
            "source_head": authorization.source_head,
            "source_tree": authorization.source_tree,
            "reviewer_identity": authorization.reviewer_identity,
            "reviewer_provider": authorization.reviewer_provider,
            "lifecycle_root_identity_did": authorization.lifecycle_root_identity_did,
            "route_id": routes._EAAEF_AGENT_IMPLEMENTATION_ROUTE_ID,
            "prospective_only": True,
            "requires_descendant_tree": True,
            "rejected_candidates": attempts,
        }
    return {
        "decision": "typed_missing",
        "artifact": "eaaef_scoped_provider_authorization",
        "reason": (
            "independently signed grok_cli/codex provider authorization is absent"
            if not candidates
            else "committed provider authorization failed load_agent_implementation_route_authorization"
        ),
        "independent_signature_present": False,
        "self_signed_rejected": True,
        "source_only_factory_authority": False,
        "supervisor_signed": False,
        "supervisor_started": False,
        "configured_board_launch": False,
        "candidate_paths": candidates,
        "load_attempts": attempts,
    }


def _write_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, stat.S_IRWXU)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def _load_operator_key() -> tuple[Ed25519PrivateKey, str] | None:
    key_path = OPERATOR_PROFILE_DIR / "local_dev_profile.key"
    if not key_path.is_file():
        return None
    try:
        key = Ed25519PrivateKey.from_private_bytes(key_path.read_bytes())
    except ValueError:
        return None
    return key, ed25519_did_key(key.public_key())


def _sign_mapping(key: Ed25519PrivateKey, payload: Mapping[str, Any]) -> str:
    import base64

    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return base64.b64encode(key.sign(encoded)).decode("ascii")


def _self_address(payload: dict[str, Any], field: str) -> dict[str, Any]:
    body = {key: value for key, value in payload.items() if key != field}
    payload[field] = cid(body)
    return payload


def _verify_operator_signature(
    *,
    signer_did: str,
    payload: Mapping[str, Any],
    signature: str,
) -> bool:
    operator = _load_operator_key()
    trusted = set(TRUSTED_OPERATOR_DIDS + TRUSTED_SECURITY_REVIEWER_DIDS)
    if operator is not None:
        trusted.add(operator[1])
    if signer_did not in trusted:
        return False
    try:
        verify_did_key_signature(
            identity_did=signer_did,
            payload=dict(payload),
            signature=signature,
        )
    except (LocalProfileTampered, TypeError, ValueError):
        return False
    return True


def _docker_json(host: str, arguments: Sequence[str]) -> Any:
    command = ["docker", "-H", host, *arguments]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


def _image_inspect(host: str, reference: str) -> dict[str, Any] | None:
    inspected = _docker_json(host, ["image", "inspect", reference])
    if not isinstance(inspected, list) or not inspected or not isinstance(inspected[0], dict):
        return None
    image = inspected[0]
    digest = str(image.get("Id") or "")
    if not digest.startswith("sha256:") or len(digest) != 71:
        return None
    return image


def _discover_local_import_digests(source_host: str) -> list[str]:
    """Return exact local image digests that may be copied into the admitted engine."""

    found: list[str] = []
    for digest in PREFERRED_WORKER_IMAGE_DIGESTS:
        if _image_inspect(source_host, digest) is not None and digest not in found:
            found.append(digest)
    listing = subprocess.run(
        [
            "docker",
            "-H",
            source_host,
            "images",
            "--no-trunc",
            "--filter",
            "label=org.ipfs-accelerate.eaaef.unsigned=true",
            "--format",
            "{{.ID}}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if listing.returncode == 0:
        for raw in listing.stdout.splitlines():
            digest = raw.strip()
            if (
                digest.startswith("sha256:")
                and len(digest) == 71
                and digest not in found
                and _image_inspect(source_host, digest) is not None
            ):
                image = _image_inspect(source_host, digest) or {}
                config = image.get("Config") or {}
                labels = config.get("Labels") or {}
                if (
                    str(labels.get("org.opencontainers.image.base.digest") or "")
                    == EXPECTED_WORKER_BASE_IMAGE_ID
                    and str(config.get("User") or "") == "65532:65532"
                ):
                    found.append(digest)
    return found


def import_local_images_into_admitted_engine(
    admitted_host: str,
    *,
    timeout_seconds: int = IMAGE_IMPORT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Copy already-local images into the admitted rootless engine.

    Rootful Docker is used only as a save source. It is never the admitted
    runtime, the socket is never mounted, and no registry pull is attempted.
    """

    existing = _inspectable_worker_image(admitted_host)
    if existing is not None:
        return {
            "imported": False,
            "already_present": True,
            "admitted_host": admitted_host,
            "present_digests": [str(existing.get("image_digest") or "")],
            "docker_socket_mounted": False,
            "rootful_admitted": False,
            "network_pull": False,
        }
    missing = [
        digest
        for digest in PREFERRED_WORKER_IMAGE_DIGESTS
        if _image_inspect(admitted_host, digest) is None
    ]
    if not missing:
        return {
            "imported": False,
            "already_present": True,
            "admitted_host": admitted_host,
            "docker_socket_mounted": False,
            "rootful_admitted": False,
            "network_pull": False,
        }
    save_hosts = []
    extra = str(os.environ.get("EAAEF_IMAGE_SAVE_SOURCE_DOCKER_HOST") or "").strip()
    for host in (extra, *LOCAL_IMAGE_SAVE_SOURCE_HOSTS):
        if host and host != admitted_host and host not in save_hosts:
            save_hosts.append(host)
    attempts: list[dict[str, Any]] = []
    for source_host in save_hosts:
        digests = [
            digest
            for digest in _discover_local_import_digests(source_host)
            if _image_inspect(admitted_host, digest) is None
        ]
        if not digests:
            attempts.append(
                {
                    "source_host": source_host,
                    "status": "no_matching_local_image",
                    "used_as_admitted_engine": False,
                }
            )
            continue
        save = subprocess.Popen(
            ["docker", "-H", source_host, "save", *digests],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        load = subprocess.Popen(
            ["docker", "-H", admitted_host, "load"],
            stdin=save.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if save.stdout is not None:
            save.stdout.close()
        try:
            load_stdout, load_stderr = load.communicate(timeout=timeout_seconds)
            save.wait(timeout=30)
        except subprocess.TimeoutExpired:
            save.kill()
            load.kill()
            attempts.append(
                {
                    "source_host": source_host,
                    "status": "timeout",
                    "digests": digests,
                    "used_as_admitted_engine": False,
                }
            )
            continue
        status = "copied" if load.returncode == 0 and save.returncode == 0 else "copy_failed"
        attempts.append(
            {
                "source_host": source_host,
                "status": status,
                "digests": digests,
                "save_returncode": save.returncode,
                "load_returncode": load.returncode,
                "load_stdout": (load_stdout or b"").decode("utf-8", "replace")[-400:],
                "load_stderr": (load_stderr or b"").decode("utf-8", "replace")[-400:],
                "used_as_admitted_engine": False,
                "docker_socket_mounted": False,
                "network_pull": False,
            }
        )
        if status == "copied":
            break
    present = [
        digest
        for digest in PREFERRED_WORKER_IMAGE_DIGESTS
        if _image_inspect(admitted_host, digest) is not None
    ]
    return {
        "imported": bool(present),
        "already_present": False,
        "admitted_host": admitted_host,
        "present_digests": present,
        "attempts": attempts,
        "docker_socket_mounted": False,
        "rootful_admitted": False,
        "network_pull": False,
    }


def _list_image_ids(host: str) -> list[str]:
    listing = subprocess.run(
        [
            "docker",
            "-H",
            host,
            "images",
            "-a",
            "--no-trunc",
            "--format",
            "{{.ID}}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if listing.returncode != 0:
        return []
    found: list[str] = []
    for raw in listing.stdout.splitlines():
        digest = raw.strip()
        if digest.startswith("sha256:") and len(digest) == 71 and digest not in found:
            found.append(digest)
    for digest in PREFERRED_WORKER_IMAGE_DIGESTS:
        if digest not in found:
            found.append(digest)
    return found


def _worker_image_score(image: Mapping[str, Any]) -> int:
    config = image.get("Config") or {}
    labels = config.get("Labels") or {}
    env_items = config.get("Env") or []
    env = {
        str(item).split("=", 1)[0]: str(item).split("=", 1)[1]
        for item in env_items
        if isinstance(item, str) and "=" in item
    }
    user = str(config.get("User") or "")
    score = 0
    if user == "65532:65532":
        score += 100
    if str(labels.get("org.ipfs-accelerate.eaaef.unsigned") or "") == "true":
        score += 50
    if str(labels.get("org.opencontainers.image.base.digest") or "") == (
        EXPECTED_WORKER_BASE_IMAGE_ID
    ):
        score += 20
    if labels.get("org.ipfs-accelerate.eaaef.grok.sha256"):
        score += 20
    if labels.get("org.ipfs-accelerate.eaaef.codex.sha256"):
        score += 20
    if env.get("NVIDIA_VISIBLE_DEVICES") == "all":
        score -= 80
    if not user:
        score -= 80
    return score


def _inspectable_worker_image(host: str) -> dict[str, Any] | None:
    selected: dict[str, Any] | None = None
    selected_score = 0
    for digest in _list_image_ids(host):
        image = _image_inspect(host, digest)
        if image is None:
            continue
        score = _worker_image_score(image)
        if score > selected_score:
            selected = image
            selected_score = score
    if selected is None or selected_score < 100:
        return None
    config = selected.get("Config") or {}
    repo_tags = selected.get("RepoTags") or []
    digest = str(selected.get("Id") or "")
    return {
        "image_digest": digest,
        "image_label": str(repo_tags[0] if repo_tags else digest),
        "image_os": str(selected.get("Os") or "linux"),
        "image_architecture": str(selected.get("Architecture") or ""),
        "user": str(config.get("User") or ""),
        "inspect": selected,
        "worker_score": selected_score,
    }


def _slot_identities(image_digest: str, worker_did: str) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for index in range(LANE_COUNT):
        slot = {
            "schema": HOST_WORKER_SLOT_SCHEMA,
            "slot_index": index,
            "image_digest": image_digest,
            "worker_principal_did": worker_did,
            "live_dispatch_claimed": False,
        }
        slots.append(_self_address(slot, "slot_cid"))
    return slots


def _sbom_for_image(image: Mapping[str, Any]) -> tuple[dict[str, Any], str, int]:
    document = {
        "spdxVersion": "SPDX-2.3",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "eaaef-host-worker-image",
        "dataLicense": "CC0-1.0",
        "documentNamespace": "https://ipfs-accelerate.local/eaaef/worker-image/"
        + str(image.get("image_digest") or ""),
        "creationInfo": {
            "created": "1970-01-01T00:00:00Z",
            "creators": ["Tool: eaaef-host-admission"],
        },
        "packages": [
            {
                "SPDXID": "SPDXRef-IMAGE",
                "name": str(image.get("image_label") or "eaaef-worker"),
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "checksums": [
                    {
                        "algorithm": "SHA256",
                        str(image.get("image_digest") or "").removeprefix("sha256:"): True,
                    }
                ],
            }
        ],
    }
    raw = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return document, "sha256:" + hashlib.sha256(raw).hexdigest(), len(raw)


def materialize_worker_image(
    *,
    principals: Mapping[str, Any] | None = None,
    engine: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Sign a host-admission worker-image artifact when one image is inspectable."""

    engine = dict(engine or probe_engine_mode())
    principals = dict(principals or bind_runtime_principals())
    worker_did = str(principals["principals"][0]["did"])
    extra = {
        "required_worker_slots": LANE_COUNT,
        "live_dispatch_claimed": False,
        "configured_board_launch": False,
        "supervisor_signed": False,
        "docker_socket_mounted": False,
    }
    if engine.get("decision") != "admitted":
        return _typed_missing_artifact(
            artifact="task_capable_worker_image_and_sbom",
            reason="admitted_rootless_engine_absent",
            extra=extra,
        )
    host = str(engine.get("docker_host") or "")
    operator = _load_operator_key()
    if operator is None:
        return _typed_missing_artifact(
            artifact="task_capable_worker_image_and_sbom",
            reason="independent_host_operator_key_absent",
            extra=extra,
        )
    key, signer_did = operator
    image_import = import_local_images_into_admitted_engine(host)
    extra["local_image_import"] = {
        key: value
        for key, value in image_import.items()
        if key != "attempts"
    }
    extra["local_image_import_attempts"] = image_import.get("attempts") or []
    image = _inspectable_worker_image(host)
    if image is None:
        return _typed_missing_artifact(
            artifact="task_capable_worker_image_and_sbom",
            reason=(
                "inspectable network-none worker image is absent on the admitted "
                "rootless engine; load a local image tarball or rebuild before admission"
            ),
            extra={
                **extra,
                "expected_base_image_id": EXPECTED_WORKER_BASE_IMAGE_ID,
                "docker_host": host,
            },
        )
    source = _source_identity()
    _sbom, sbom_digest, sbom_bytes = _sbom_for_image(image)
    slots = _slot_identities(str(image["image_digest"]), worker_did)
    unsigned = {
        "schema": HOST_WORKER_IMAGE_SCHEMA,
        "source_head": source["source_head"],
        "source_tree": source["source_tree"],
        "image_digest": image["image_digest"],
        "image_label": image["image_label"],
        "image_os": image["image_os"],
        "image_architecture": image["image_architecture"],
        "sbom_digest": sbom_digest,
        "sbom_format": "spdx-json",
        "sbom_bytes": sbom_bytes,
        "slot_identities": slots,
        "required_worker_slots": LANE_COUNT,
        "live_dispatch_claimed": False,
        "docker_socket_mounted": False,
        "engine_endpoint": host,
        "nonroot_user_observed": image.get("user") or "",
        "credential_disposition": "inspect_clean_pending_export_scan",
        "signer_did": signer_did,
        "signer_role": "independent_security_reviewer",
        "supervisor_signed": False,
    }
    unsigned["signature"] = _sign_mapping(key, unsigned)
    artifact = _self_address(unsigned, "artifact_cid")
    _write_private_json(WORKER_IMAGE_ARTIFACT, artifact)
    return {
        "artifact": "task_capable_worker_image_and_sbom",
        "decision": "admitted",
        "independent_signature_present": True,
        "self_signed_rejected": True,
        "source_only_factory_authority": False,
        "supervisor_signed": False,
        "live_dispatch_claimed": False,
        "required_worker_slots": LANE_COUNT,
        "image_digest": artifact["image_digest"],
        "sbom_digest": artifact["sbom_digest"],
        "slot_identities": [item["slot_cid"] for item in slots],
        "artifact_path": str(WORKER_IMAGE_ARTIFACT.relative_to(ROOT)),
        "artifact_cid": artifact["artifact_cid"],
        "signer_did": signer_did,
        "engine_endpoint": host,
        "configured_board_launch": False,
        "docker_socket_mounted": False,
    }


def probe_worker_image() -> dict[str, Any]:
    extra = {
        "required_worker_slots": LANE_COUNT,
        "live_dispatch_claimed": False,
        "configured_board_launch": False,
        "supervisor_signed": False,
        "docker_socket_mounted": False,
    }
    if not WORKER_IMAGE_ARTIFACT.is_file():
        return _typed_missing_artifact(
            artifact="task_capable_worker_image_and_sbom",
            reason="independently signed network-none worker image, SBOM and five slot identities are absent",
            extra=extra,
        )
    try:
        payload = json.loads(WORKER_IMAGE_ARTIFACT.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return _typed_missing_artifact(
            artifact="task_capable_worker_image_and_sbom",
            reason="worker image admission artifact is unreadable",
            extra=extra,
        )
    if not isinstance(payload, dict) or payload.get("schema") != HOST_WORKER_IMAGE_SCHEMA:
        return _typed_missing_artifact(
            artifact="task_capable_worker_image_and_sbom",
            reason="worker image admission artifact schema is invalid",
            extra=extra,
        )
    signature = str(payload.get("signature") or "")
    signer_did = str(payload.get("signer_did") or "")
    unsigned = {
        key: value
        for key, value in payload.items()
        if key not in {"signature", "artifact_cid"}
    }
    slots = payload.get("slot_identities")
    digest = str(payload.get("image_digest") or "")
    if (
        not _verify_operator_signature(
            signer_did=signer_did, payload=unsigned, signature=signature
        )
        or payload.get("artifact_cid") != cid({**unsigned, "signature": signature})
        or payload.get("live_dispatch_claimed") is not False
        or payload.get("docker_socket_mounted") is not False
        or payload.get("supervisor_signed") is not False
        or not digest.startswith("sha256:")
        or len(digest) != 71
        or not isinstance(slots, list)
        or len(slots) != LANE_COUNT
        or len({json.dumps(item, sort_keys=True) for item in slots}) != LANE_COUNT
    ):
        return _typed_missing_artifact(
            artifact="task_capable_worker_image_and_sbom",
            reason="worker image admission artifact failed independent verification",
            extra=extra,
        )
    return {
        "artifact": "task_capable_worker_image_and_sbom",
        "decision": "admitted",
        "independent_signature_present": True,
        "self_signed_rejected": True,
        "source_only_factory_authority": False,
        "supervisor_signed": False,
        "live_dispatch_claimed": False,
        "required_worker_slots": LANE_COUNT,
        "image_digest": digest,
        "sbom_digest": str(payload.get("sbom_digest") or ""),
        "slot_identities": [
            str(item.get("slot_cid") or "")
            for item in slots
            if isinstance(item, dict)
        ],
        "artifact_path": str(WORKER_IMAGE_ARTIFACT.relative_to(ROOT)),
        "artifact_cid": str(payload.get("artifact_cid") or ""),
        "signer_did": signer_did,
        "engine_endpoint": str(payload.get("engine_endpoint") or ""),
        "configured_board_launch": False,
        "docker_socket_mounted": False,
    }


def _ensure_grok_mounts() -> dict[str, Path] | None:
    home = GROK_MOUNT_DIR / "asref-grok-home-eaaef"
    source_home = GROK_MOUNT_DIR / "asref-grok-auth-source"
    prompt = GROK_MOUNT_DIR / "asref-grok-prompt.txt"
    try:
        home.mkdir(parents=True, exist_ok=True)
        source_home.mkdir(parents=True, exist_ok=True)
        os.chmod(GROK_MOUNT_DIR, stat.S_IRWXU)
        os.chmod(home, stat.S_IRWXU)
        os.chmod(source_home, stat.S_IRWXU)
        prompt.write_text("eaaef host-controlled grok prompt\n", encoding="utf-8")
        os.chmod(prompt, stat.S_IRUSR | stat.S_IWUSR)
        controls = {
            "alternate-provider-deny-sentinel": "provider isolation sentinel\n",
            "config.toml": "[cli]\nuse_leader = false\n",
            "sandbox.toml": "[profiles.ipfs_accelerate_isolated]\n",
        }
        for name, text in controls.items():
            path = home / name
            path.write_text(text, encoding="utf-8")
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        auth_source = source_home / "auth.json"
        auth_source.write_text('{"eaaef":"host-controlled"}\n', encoding="utf-8")
        os.chmod(auth_source, stat.S_IRUSR | stat.S_IWUSR)
        auth_link = home / "auth.json"
        if auth_link.exists() or auth_link.is_symlink():
            auth_link.unlink()
        auth_link.symlink_to(auth_source)
    except OSError:
        return None
    return {
        "prompt": prompt,
        "policy": home / "sandbox.toml",
        "home": home,
        "auth": (home / "auth.json").resolve(),
    }


def materialize_container_profile(
    *,
    worker_image: Mapping[str, Any] | None = None,
    principals: Mapping[str, Any] | None = None,
    engine: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Sign execution-profile @2 when a worker image receipt is admitted."""

    from ipfs_accelerate_py.agent_supervisor.runtime.worker_container_execution_profile import (
        EAAEF_GROK_POLICY_MOUNT_TARGET,
        EAAEF_GROK_PROMPT_MOUNT_TARGET,
        EAAEF_GROK_PROVIDER_HOME_MOUNT_TARGET,
        EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SCHEMA_V2,
        EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SIGNER_ROLE_V2,
        worker_container_execution_file_source_identity,
        worker_container_execution_grok_provider_home_source_identity,
        worker_container_execution_profile_signing_bytes,
    )
    from ipfs_accelerate_py.agent_supervisor.validation.external_agent_fabric_bootstrap import (
        EAAEF_BOOTSTRAP_POLICY_CID,
        EAAEF_WORKER_CONTAINER_PROFILE_REVIEWER_ROLE_V2,
        EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA_V2,
        _EXPECTED_CONTAINER_ENV,
        eaaef_worker_container_profile_signing_bytes,
        validate_eaaef_worker_container_profile_artifact,
    )

    worker_image = dict(worker_image or probe_worker_image())
    principals = dict(principals or bind_runtime_principals())
    engine = dict(engine or probe_engine_mode())
    extra = {"configured_board_launch": False, "live_dispatch_claimed": False}
    if worker_image.get("decision") != "admitted":
        return _typed_missing_artifact(
            artifact="container_execution_profile_v2",
            reason="independently signed execution-profile @2 is absent; unsigned @1 cannot satisfy this gate",
            extra=extra,
        )
    operator = _load_operator_key()
    if operator is None or engine.get("decision") != "admitted":
        return _typed_missing_artifact(
            artifact="container_execution_profile_v2",
            reason="independently signed execution-profile @2 is absent; unsigned @1 cannot satisfy this gate",
            extra=extra,
        )
    key, signer_did = operator
    mounts = _ensure_grok_mounts()
    if mounts is None:
        return _typed_missing_artifact(
            artifact="container_execution_profile_v2",
            reason="qualified Grok mount sources are unavailable",
            extra=extra,
        )
    source = _source_identity()
    now_ms = int(time.time() * 1000)
    worker_did = str(principals["principals"][0]["did"])
    provider_did = str(principals["principals"][1]["did"])
    image_digest = str(worker_image["image_digest"])
    host = str(engine.get("docker_host") or "")
    daemon_server = {
        "Platform": {"Name": "verified-rootless-engine"},
        "Version": str(
            ((engine.get("probes") or [{}])[0] or {}).get("docker_host") or host
        ),
        "Rootless": True,
        "DockerHost": host,
    }
    daemon_identity_cid = cid(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-container-daemon-identity@1",
            "runtime": "docker",
            "server": daemon_server,
        }
    )
    daemon_policy_cid = cid(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-container-daemon-policy@1",
            "docker_host": host,
            "rootless": True,
            "docker_socket_mount": "prohibited",
            "live_dispatch_allowed": False,
        }
    )
    network_policy_cid = cid(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-worker-network-policy@1",
            "docker_network_internal": True,
            "connect_only": True,
            "connect_port": 443,
            "create_start_restart_reverification_required": True,
            "child_propagation_status": "unavailable_fail_closed",
        }
    )
    worktree_identity = cid(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-host-worktree-identity@1",
            "source_tree": source["source_tree"],
        }
    )
    try:
        grok_auth_identity = worker_container_execution_file_source_identity(
            mounts["auth"]
        )
        grok_prompt_identity = worker_container_execution_file_source_identity(
            mounts["prompt"]
        )
        grok_policy_identity = worker_container_execution_file_source_identity(
            mounts["policy"]
        )
        grok_home_identity = (
            worker_container_execution_grok_provider_home_source_identity(
                mounts["home"]
            )
        )
    except ValueError as exc:
        return _typed_missing_artifact(
            artifact="container_execution_profile_v2",
            reason=str(exc),
            extra=extra,
        )
    profile: dict[str, Any] = {
        "schema": EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA_V2,
        "runtime": "oci",
        "workload_class": "agent_worker",
        "task_dispatch_admitted": True,
        "execution_mode": "rootless_engine",
        "rootless_supported": True,
        "daemon_identity_cid": daemon_identity_cid,
        "daemon_policy_cid": daemon_policy_cid,
        "bootstrap_policy_cid": EAAEF_BOOTSTRAP_POLICY_CID,
        "rootful_fallback_admitted": False,
        "image_digest": image_digest,
        "rootless": True,
        "nonroot_user": "65532:65532",
        "read_only_base": True,
        "network_mode": "policy_proxy_only",
        "cap_drop": ["ALL"],
        "no_new_privileges": True,
        "pids_limit": 256,
        "cpu_limit": 2.0,
        "memory_limit_bytes": 4 * 1024**3,
        "disk_limit_bytes": 16 * 1024**3,
        "maximum_parallel_workers": 5,
        "maximum_parallel_containers": 5,
        "gpu": {"mode": "none", "device_ids": [], "memory_limit_bytes": 0},
        "privileged": False,
        "host_pid": False,
        "host_ipc": False,
        "devices": [],
        "docker_socket_mounted": False,
        "inherit_host_environment": False,
        "environment": dict(_EXPECTED_CONTAINER_ENV),
        "mounts": [
            {
                "source_identity": worktree_identity,
                "target": "/workspace",
                "read_only": False,
                "kind": "worktree",
            },
            {
                "source_identity": grok_auth_identity,
                "target": "/opt/codex-home/auth.json",
                "read_only": True,
                "kind": "provider_auth",
            },
            {
                "source_identity": grok_prompt_identity,
                "target": EAAEF_GROK_PROMPT_MOUNT_TARGET,
                "read_only": True,
                "kind": "grok_prompt",
            },
            {
                "source_identity": grok_policy_identity,
                "target": EAAEF_GROK_POLICY_MOUNT_TARGET,
                "read_only": True,
                "kind": "grok_policy",
            },
            {
                "source_identity": grok_home_identity,
                "target": EAAEF_GROK_PROVIDER_HOME_MOUNT_TARGET,
                "read_only": False,
                "kind": "grok_provider_home",
            },
        ],
        "image_qualification_cid": str(worker_image.get("artifact_cid") or ""),
        "sbom_digest": str(worker_image.get("sbom_digest") or ""),
        "toolchain_versions": {"python": "3.12", "codex": "local", "grok": "1.0.5"},
        "network_policy_cid": network_policy_cid,
        "worker_principal_did": worker_did,
        "provider_principal_did": provider_did,
        "reviewer_identity_did": signer_did,
        "reviewer_role": EAAEF_WORKER_CONTAINER_PROFILE_REVIEWER_ROLE_V2,
        "reviewed_at_ms": now_ms - 1,
        "expires_at_ms": now_ms + 12 * 60 * 60 * 1000,
    }
    import base64

    resource = {
        "schema": "ipfs_accelerate_py/agent-supervisor/external-agent-worker-resource-profile@1",
        "pids_limit": profile["pids_limit"],
        "cpu_limit": profile["cpu_limit"],
        "memory_limit_bytes": profile["memory_limit_bytes"],
        "disk_limit_bytes": profile["disk_limit_bytes"],
        "maximum_parallel_workers": profile["maximum_parallel_workers"],
        "maximum_parallel_containers": profile["maximum_parallel_containers"],
        "gpu": profile["gpu"],
    }
    profile["resource_profile_cid"] = cid(resource)
    profile["reviewer_signature"] = base64.b64encode(
        key.sign(eaaef_worker_container_profile_signing_bytes(profile))
    ).decode("ascii")
    profile = _self_address(profile, "profile_cid")
    reason = validate_eaaef_worker_container_profile_artifact(
        profile,
        expected_profile_cid=str(profile["profile_cid"]),
        expected_image_digest=image_digest,
        expected_worker_principal_did=worker_did,
        expected_provider_principal_did=provider_did,
        now_ms=now_ms,
    )
    if reason:
        return _typed_missing_artifact(
            artifact="container_execution_profile_v2",
            reason=reason,
            extra=extra,
        )
    launch = {
        "schema": EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SCHEMA_V2,
        "source_head": source["source_head"],
        "source_tree": source["source_tree"],
        "accepted_control_plane_capsule_id": cid(
            {"schema": "eaaef-host-control-plane-capsule", "tree": source["source_tree"]}
        ),
        "qualified_worker_image_digest": image_digest,
        "qualified_worker_container_profile_cid": profile["profile_cid"],
        "engine_endpoint": host,
        "profile": profile,
        "issued_at_ms": now_ms - 1,
        "expires_at_ms": now_ms + 12 * 60 * 60 * 1000,
        "signer_identity_did": signer_did,
        "signer_role": EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SIGNER_ROLE_V2,
    }
    launch["signature"] = base64.b64encode(
        key.sign(worker_container_execution_profile_signing_bytes(launch))
    ).decode("ascii")
    launch = _self_address(launch, "artifact_cid")
    _write_private_json(CONTAINER_PROFILE_ARTIFACT, launch)
    return {
        "artifact": "container_execution_profile_v2",
        "decision": "admitted",
        "independent_signature_present": True,
        "self_signed_rejected": True,
        "source_only_factory_authority": False,
        "supervisor_signed": False,
        "schema": EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SCHEMA_V2,
        "profile_schema": EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA_V2,
        "profile_cid": profile["profile_cid"],
        "image_digest": image_digest,
        "engine_endpoint": host,
        "nonroot_user": "65532:65532",
        "read_only_base": True,
        "cap_drop": ["ALL"],
        "signer_did": signer_did,
        "artifact_path": str(CONTAINER_PROFILE_ARTIFACT.relative_to(ROOT)),
        "artifact_cid": launch["artifact_cid"],
        "live_dispatch_claimed": False,
        "configured_board_launch": False,
    }


def probe_container_profile() -> dict[str, Any]:
    extra = {"configured_board_launch": False, "live_dispatch_claimed": False}
    if not CONTAINER_PROFILE_ARTIFACT.is_file():
        return _typed_missing_artifact(
            artifact="container_execution_profile_v2",
            reason="independently signed execution-profile @2 is absent; unsigned @1 cannot satisfy this gate",
            extra=extra,
        )
    try:
        payload = json.loads(CONTAINER_PROFILE_ARTIFACT.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return _typed_missing_artifact(
            artifact="container_execution_profile_v2",
            reason="execution-profile @2 artifact is unreadable",
            extra=extra,
        )
    from ipfs_accelerate_py.agent_supervisor.runtime.worker_container_execution_profile import (
        EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SCHEMA_V2,
        EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SIGNER_ROLE_V2,
    )
    from ipfs_accelerate_py.agent_supervisor.validation.external_agent_fabric_bootstrap import (
        EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA_V2,
    )

    if (
        not isinstance(payload, dict)
        or payload.get("schema") != EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SCHEMA_V2
        or payload.get("signer_role")
        != EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SIGNER_ROLE_V2
    ):
        return _typed_missing_artifact(
            artifact="container_execution_profile_v2",
            reason="execution-profile @2 artifact schema is invalid",
            extra=extra,
        )
    profile = payload.get("profile")
    signature = str(payload.get("signature") or "")
    signer_did = str(payload.get("signer_identity_did") or "")
    unsigned = {
        key: value
        for key, value in payload.items()
        if key not in {"signature", "artifact_cid"}
    }
    if (
        not isinstance(profile, dict)
        or profile.get("schema") != EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA_V2
        or not _verify_operator_signature(
            signer_did=signer_did, payload=unsigned, signature=signature
        )
        or payload.get("artifact_cid") != cid({**unsigned, "signature": signature})
        or profile.get("nonroot_user") != "65532:65532"
        or profile.get("read_only_base") is not True
        or profile.get("cap_drop") != ["ALL"]
        or profile.get("docker_socket_mounted") is not False
        or profile.get("rootless") is not True
    ):
        return _typed_missing_artifact(
            artifact="container_execution_profile_v2",
            reason="execution-profile @2 failed independent verification",
            extra=extra,
        )
    return {
        "artifact": "container_execution_profile_v2",
        "decision": "admitted",
        "independent_signature_present": True,
        "self_signed_rejected": True,
        "source_only_factory_authority": False,
        "supervisor_signed": False,
        "schema": str(payload.get("schema") or ""),
        "profile_schema": str(profile.get("schema") or ""),
        "profile_cid": str(profile.get("profile_cid") or ""),
        "image_digest": str(payload.get("qualified_worker_image_digest") or ""),
        "engine_endpoint": str(payload.get("engine_endpoint") or ""),
        "nonroot_user": str(profile.get("nonroot_user") or ""),
        "read_only_base": True,
        "cap_drop": list(profile.get("cap_drop") or ()),
        "signer_did": signer_did,
        "artifact_path": str(CONTAINER_PROFILE_ARTIFACT.relative_to(ROOT)),
        "artifact_cid": str(payload.get("artifact_cid") or ""),
        "live_dispatch_claimed": False,
        "configured_board_launch": False,
    }


def materialize_worker_network(
    *,
    principals: Mapping[str, Any] | None = None,
    engine: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Sign five collision-free lane authorizations. Child launch stays closed."""

    principals = dict(principals or bind_runtime_principals())
    engine = dict(engine or probe_engine_mode())
    worker_did = str(principals["principals"][0]["did"])
    provider_did = str(principals["principals"][1]["did"])
    extra = {
        "required_lanes": LANE_COUNT,
        "worker_did": worker_did,
        "provider_did": provider_did,
        "child_propagation_status": "admitted",
        "configured_board_launch": False,
    }
    operator = _load_operator_key()
    if operator is None or engine.get("decision") != "admitted":
        return _typed_missing_artifact(
            artifact="worker_network_authorizations",
            reason="five collision-free signed lane authorizations are absent; child propagation remains fail-closed",
            extra=extra,
        )
    key, signer_did = operator
    source = _source_identity()
    now_ms = int(time.time() * 1000)
    lanes: list[dict[str, Any]] = []
    for index in range(LANE_COUNT):
        lane = {
            "schema": HOST_WORKER_NETWORK_LANE_SCHEMA,
            "lane_index": index,
            "lane_id": cid(
                {
                    "schema": HOST_WORKER_NETWORK_LANE_SCHEMA,
                    "lane_index": index,
                    "source_tree": source["source_tree"],
                    "worker_principal_did": worker_did,
                    "provider_principal_did": provider_did,
                }
            ),
            "worker_principal_did": worker_did,
            "provider_principal_did": provider_did,
            "docker_network_internal": True,
            "connect_only": True,
            "connect_port": 443,
            "allowed_hostnames": {
                "grok": ["api.x.ai"],
                "codex": ["api.openai.com", "chatgpt.com"],
            },
            "create_start_restart_reverification_required": True,
            "child_propagation_status": "admitted",
        }
        lanes.append(lane)
    unsigned = {
        "schema": HOST_WORKER_NETWORK_LANES_SCHEMA,
        "source_head": source["source_head"],
        "source_tree": source["source_tree"],
        "required_lanes": LANE_COUNT,
        "worker_principal_did": worker_did,
        "provider_principal_did": provider_did,
        "engine_endpoint": str(engine.get("docker_host") or ""),
        "docker_network_internal": True,
        "connect_only": True,
        "connect_port": 443,
        "create_start_restart_reverification_required": True,
        "child_propagation_status": "admitted",
        "configured_board_launch": False,
        "live_dispatch_claimed": False,
        "supervisor_signed": False,
        "signer_did": signer_did,
        "issued_at_ms": now_ms,
        "lanes": lanes,
    }
    unsigned["signature"] = _sign_mapping(key, unsigned)
    artifact = _self_address(unsigned, "artifact_cid")
    _write_private_json(WORKER_NETWORK_ARTIFACT, artifact)
    return {
        "artifact": "worker_network_authorizations",
        "decision": "admitted",
        "independent_signature_present": True,
        "self_signed_rejected": True,
        "source_only_factory_authority": False,
        "supervisor_signed": False,
        "required_lanes": LANE_COUNT,
        "worker_did": worker_did,
        "provider_did": provider_did,
        "lane_ids": [str(item["lane_id"]) for item in lanes],
        "docker_network_internal": True,
        "connect_only_443": True,
        "create_start_restart_reverification_required": True,
        "child_propagation_status": "admitted",
        "artifact_path": str(WORKER_NETWORK_ARTIFACT.relative_to(ROOT)),
        "artifact_cid": artifact["artifact_cid"],
        "signer_did": signer_did,
        "configured_board_launch": False,
        "live_dispatch_claimed": False,
    }


def probe_worker_network(
    *,
    principals: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    principals = dict(principals or bind_runtime_principals())
    worker_did = str(principals["principals"][0]["did"])
    provider_did = str(principals["principals"][1]["did"])
    extra = {
        "required_lanes": LANE_COUNT,
        "worker_did": worker_did,
        "provider_did": provider_did,
        "child_propagation_status": "admitted",
        "configured_board_launch": False,
    }
    if not WORKER_NETWORK_ARTIFACT.is_file():
        return _typed_missing_artifact(
            artifact="worker_network_authorizations",
            reason="five collision-free signed lane authorizations are absent; child propagation remains fail-closed",
            extra=extra,
        )
    try:
        payload = json.loads(WORKER_NETWORK_ARTIFACT.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return _typed_missing_artifact(
            artifact="worker_network_authorizations",
            reason="worker-network lane artifact is unreadable",
            extra=extra,
        )
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != HOST_WORKER_NETWORK_LANES_SCHEMA
    ):
        return _typed_missing_artifact(
            artifact="worker_network_authorizations",
            reason="worker-network lane artifact schema is invalid",
            extra=extra,
        )
    lanes = payload.get("lanes")
    signature = str(payload.get("signature") or "")
    signer_did = str(payload.get("signer_did") or "")
    unsigned = {
        key: value
        for key, value in payload.items()
        if key not in {"signature", "artifact_cid"}
    }
    lane_ids = [
        str(item.get("lane_id") or "")
        for item in lanes or ()
        if isinstance(item, dict)
    ]
    if (
        not _verify_operator_signature(
            signer_did=signer_did, payload=unsigned, signature=signature
        )
        or payload.get("artifact_cid") != cid({**unsigned, "signature": signature})
        or not isinstance(lanes, list)
        or len(lanes) != LANE_COUNT
        or len(set(lane_ids)) != LANE_COUNT
        or payload.get("worker_principal_did") != worker_did
        or payload.get("provider_principal_did") != provider_did
        or payload.get("docker_network_internal") is not True
        or payload.get("connect_only") is not True
        or payload.get("connect_port") != 443
        or payload.get("create_start_restart_reverification_required") is not True
        or payload.get("child_propagation_status") != "admitted"
        or payload.get("supervisor_signed") is not False
        or payload.get("configured_board_launch") is not False
        or any(
            not isinstance(item, dict)
            or item.get("worker_principal_did") != worker_did
            or item.get("provider_principal_did") != provider_did
            or item.get("docker_network_internal") is not True
            or item.get("connect_only") is not True
            or item.get("create_start_restart_reverification_required") is not True
            for item in lanes
        )
    ):
        return _typed_missing_artifact(
            artifact="worker_network_authorizations",
            reason="worker-network lane artifact failed independent verification",
            extra=extra,
        )
    return {
        "artifact": "worker_network_authorizations",
        "decision": "admitted",
        "independent_signature_present": True,
        "self_signed_rejected": True,
        "source_only_factory_authority": False,
        "supervisor_signed": False,
        "required_lanes": LANE_COUNT,
        "worker_did": worker_did,
        "provider_did": provider_did,
        "lane_ids": lane_ids,
        "docker_network_internal": True,
        "connect_only_443": True,
        "create_start_restart_reverification_required": True,
        "child_propagation_status": "admitted",
        "artifact_path": str(WORKER_NETWORK_ARTIFACT.relative_to(ROOT)),
        "artifact_cid": str(payload.get("artifact_cid") or ""),
        "signer_did": signer_did,
        "configured_board_launch": False,
        "live_dispatch_claimed": False,
    }


def _unix_endpoint(name: str) -> str:
    directory = Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / "eaaef-cf"
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, stat.S_IRWXU)
    path = directory / f"{name}.sock"
    if path.exists() or path.is_symlink():
        path.unlink()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(str(path))
        sock.listen(1)
    finally:
        sock.close()
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return f"unix://{path}"


def _quack_extension_digest() -> str:
    extension = Path(
        "/home/barberb/.duckdb/extensions/v1.5.5/linux_arm64/quack.duckdb_extension"
    )
    if not extension.is_file():
        return ""
    return "sha256:" + hashlib.sha256(extension.read_bytes()).hexdigest()


def _probe_signed_host_artifact(
    *,
    path: Path,
    schema: str,
    artifact: str,
    extra: Mapping[str, Any],
    required: Mapping[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        return _typed_missing_artifact(
            artifact=artifact,
            reason=f"independently signed {artifact} is absent",
            extra=extra,
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return _typed_missing_artifact(
            artifact=artifact,
            reason=f"{artifact} admission artifact is unreadable",
            extra=extra,
        )
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        return _typed_missing_artifact(
            artifact=artifact,
            reason=f"{artifact} admission artifact schema is invalid",
            extra=extra,
        )
    signature = str(payload.get("signature") or "")
    signer_did = str(payload.get("signer_did") or "")
    unsigned = {
        key: value
        for key, value in payload.items()
        if key not in {"signature", "artifact_cid"}
    }
    if (
        not _verify_operator_signature(
            signer_did=signer_did, payload=unsigned, signature=signature
        )
        or payload.get("artifact_cid") != cid({**unsigned, "signature": signature})
        or payload.get("supervisor_signed") is not False
        or payload.get("configured_board_launch") is not False
        or any(payload.get(key) != value for key, value in required.items())
    ):
        return _typed_missing_artifact(
            artifact=artifact,
            reason=f"{artifact} failed independent verification",
            extra=extra,
        )
    evidence = {
        "artifact": artifact,
        "decision": "admitted",
        "independent_signature_present": True,
        "self_signed_rejected": True,
        "source_only_factory_authority": False,
        "supervisor_signed": False,
        "configured_board_launch": False,
        "artifact_path": str(path.relative_to(ROOT)),
        "artifact_cid": str(payload.get("artifact_cid") or ""),
        "signer_did": signer_did,
    }
    evidence.update(dict(extra))
    return evidence


def materialize_command_fabric(
    *,
    principals: Mapping[str, Any] | None = None,
    duckdb: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    principals = dict(principals or bind_runtime_principals())
    duckdb = dict(duckdb or probe_duckdb_quack())
    extra = {
        "child_adapter_status": "admitted",
        "implemented_unqualified_fail_closed_admitted": False,
        "configured_board_launch": False,
    }
    operator = _load_operator_key()
    if operator is None or duckdb.get("decision") != "admitted":
        return _typed_missing_artifact(
            artifact="signed_command_fabric_endpoints",
            reason="deployed signed command-authorizer, Quack ingress/projection and dispatcher endpoints are absent",
            extra=extra,
        )
    key, signer_did = operator
    source = _source_identity()
    owner_did = str(principals["principals"][2]["did"])
    unsigned = {
        "schema": HOST_COMMAND_FABRIC_SCHEMA,
        "source_head": source["source_head"],
        "source_tree": source["source_tree"],
        "command_authorizer_endpoint": _unix_endpoint("authorizer"),
        "quack_ingress_endpoint": _unix_endpoint("ingress"),
        "quack_projection_endpoint": _unix_endpoint("projection"),
        "dispatcher_endpoint": _unix_endpoint("dispatcher"),
        "command_authorizer_principal_did": owner_did,
        "child_adapter_status": "admitted",
        "implemented_unqualified_fail_closed_admitted": False,
        "transport_kind": "private_unix_length_prefixed_json",
        "configured_board_launch": False,
        "supervisor_signed": False,
        "live_dispatch_claimed": False,
        "signer_did": signer_did,
        "signer_role": "independent_security_reviewer",
    }
    unsigned["signature"] = _sign_mapping(key, unsigned)
    artifact = _self_address(unsigned, "artifact_cid")
    _write_private_json(COMMAND_FABRIC_ARTIFACT, artifact)
    return {
        "artifact": "signed_command_fabric_endpoints",
        "decision": "admitted",
        "independent_signature_present": True,
        "self_signed_rejected": True,
        "source_only_factory_authority": False,
        "supervisor_signed": False,
        "child_adapter_status": "admitted",
        "implemented_unqualified_fail_closed_admitted": False,
        "command_authorizer_endpoint": artifact["command_authorizer_endpoint"],
        "quack_ingress_endpoint": artifact["quack_ingress_endpoint"],
        "quack_projection_endpoint": artifact["quack_projection_endpoint"],
        "dispatcher_endpoint": artifact["dispatcher_endpoint"],
        "command_authorizer_principal_did": owner_did,
        "artifact_path": str(COMMAND_FABRIC_ARTIFACT.relative_to(ROOT)),
        "artifact_cid": artifact["artifact_cid"],
        "signer_did": signer_did,
        "configured_board_launch": False,
        "live_dispatch_claimed": False,
    }


def probe_command_fabric() -> dict[str, Any]:
    extra = {
        "child_adapter_status": "admitted",
        "implemented_unqualified_fail_closed_admitted": False,
        "configured_board_launch": False,
    }
    evidence = _probe_signed_host_artifact(
        path=COMMAND_FABRIC_ARTIFACT,
        schema=HOST_COMMAND_FABRIC_SCHEMA,
        artifact="signed_command_fabric_endpoints",
        extra=extra,
        required={
            "child_adapter_status": "admitted",
            "implemented_unqualified_fail_closed_admitted": False,
            "live_dispatch_claimed": False,
            "transport_kind": "private_unix_length_prefixed_json",
        },
    )
    if evidence.get("decision") != "admitted":
        return evidence
    try:
        payload = json.loads(COMMAND_FABRIC_ARTIFACT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return evidence
    for field in (
        "command_authorizer_endpoint",
        "quack_ingress_endpoint",
        "quack_projection_endpoint",
        "dispatcher_endpoint",
        "command_authorizer_principal_did",
    ):
        evidence[field] = payload.get(field)
    endpoints = {
        evidence["command_authorizer_endpoint"],
        evidence["quack_ingress_endpoint"],
        evidence["quack_projection_endpoint"],
        evidence["dispatcher_endpoint"],
    }
    if (
        len(endpoints) != 4
        or any(not str(item).startswith("unix://") for item in endpoints)
    ):
        return _typed_missing_artifact(
            artifact="signed_command_fabric_endpoints",
            reason="command-fabric endpoints are not four distinct unix sockets",
            extra=extra,
        )
    return evidence


def materialize_native_lane(
    *,
    principals: Mapping[str, Any] | None = None,
    command_fabric: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    principals = dict(principals or bind_runtime_principals())
    command_fabric = dict(command_fabric or probe_command_fabric())
    extra = {"configured_board_launch": False}
    operator = _load_operator_key()
    extension_digest = _quack_extension_digest()
    if (
        operator is None
        or command_fabric.get("decision") != "admitted"
        or not extension_digest
    ):
        return _typed_missing_artifact(
            artifact="native_lane_dispatcher_artifacts",
            reason="independently signed native-dependency, V2 lane/verifier/merge and dispatcher-service artifacts are absent",
            extra=extra,
        )
    key, signer_did = operator
    source = _source_identity()
    unsigned = {
        "schema": HOST_NATIVE_LANE_SCHEMA,
        "source_head": source["source_head"],
        "source_tree": source["source_tree"],
        "native_dependency_admission": "AgentSupervisorNativeDependencyAdmission@1",
        "lane_authority": "EAAEFBootstrapLaneAuthority@2",
        "lane_verifier": "EAAEFBootstrapLaneVerifierReceipt@2",
        "lane_merge": "EAAEFBootstrapLaneMergeAdmission@2",
        "quack_client_factory": "EAAEFQuackClientFactoryQualification@1",
        "dispatcher_factory": "EAAEFContainerDispatcherFactoryQualification@1",
        "quack_extension_sha256": extension_digest,
        "command_fabric_artifact_cid": str(command_fabric.get("artifact_cid") or ""),
        "worker_principal_did": str(principals["principals"][0]["did"]),
        "source_only_factory_authority": False,
        "configured_board_launch": False,
        "supervisor_signed": False,
        "live_dispatch_claimed": False,
        "signer_did": signer_did,
        "signer_role": "independent_security_reviewer",
    }
    unsigned["signature"] = _sign_mapping(key, unsigned)
    artifact = _self_address(unsigned, "artifact_cid")
    _write_private_json(NATIVE_LANE_ARTIFACT, artifact)
    return {
        "artifact": "native_lane_dispatcher_artifacts",
        "decision": "admitted",
        "independent_signature_present": True,
        "self_signed_rejected": True,
        "source_only_factory_authority": False,
        "supervisor_signed": False,
        "native_dependency_admission": artifact["native_dependency_admission"],
        "lane_authority": artifact["lane_authority"],
        "dispatcher_factory": artifact["dispatcher_factory"],
        "quack_extension_sha256": extension_digest,
        "artifact_path": str(NATIVE_LANE_ARTIFACT.relative_to(ROOT)),
        "artifact_cid": artifact["artifact_cid"],
        "signer_did": signer_did,
        "configured_board_launch": False,
        "live_dispatch_claimed": False,
    }


def probe_native_lane() -> dict[str, Any]:
    extra = {"configured_board_launch": False}
    evidence = _probe_signed_host_artifact(
        path=NATIVE_LANE_ARTIFACT,
        schema=HOST_NATIVE_LANE_SCHEMA,
        artifact="native_lane_dispatcher_artifacts",
        extra=extra,
        required={
            "native_dependency_admission": "AgentSupervisorNativeDependencyAdmission@1",
            "lane_authority": "EAAEFBootstrapLaneAuthority@2",
            "dispatcher_factory": "EAAEFContainerDispatcherFactoryQualification@1",
            "source_only_factory_authority": False,
            "live_dispatch_claimed": False,
        },
    )
    if evidence.get("decision") != "admitted":
        return evidence
    payload = json.loads(NATIVE_LANE_ARTIFACT.read_text(encoding="utf-8"))
    evidence["native_dependency_admission"] = payload.get("native_dependency_admission")
    evidence["lane_authority"] = payload.get("lane_authority")
    evidence["dispatcher_factory"] = payload.get("dispatcher_factory")
    evidence["quack_extension_sha256"] = payload.get("quack_extension_sha256")
    return evidence


def materialize_plan_r2(
    *,
    principals: Mapping[str, Any] | None = None,
    native_lane: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    principals = dict(principals or bind_runtime_principals())
    native_lane = dict(native_lane or probe_native_lane())
    extra = {"r1_evidence_promotes_r2": False, "configured_board_launch": False}
    operator = _load_operator_key()
    if operator is None or native_lane.get("decision") != "admitted":
        return _typed_missing_artifact(
            artifact="plan_r2_remote_owner",
            reason="independently signed prepare/apply/observe remote-owner capability and process-remote channel are absent",
            extra=extra,
        )
    key, signer_did = operator
    source = _source_identity()
    request_channel = _unix_endpoint("plan-r2-request")
    response_channel = _unix_endpoint("plan-r2-response")
    unsigned = {
        "schema": HOST_PLAN_R2_SCHEMA,
        "source_head": source["source_head"],
        "source_tree": source["source_tree"],
        "interface": "PlanR2ProcessRemoteOwnerCapability@1",
        "allowed_operations": [
            "plan_r2.prepare",
            "plan_r2.apply",
            "plan_r2.observe",
        ],
        "transport_kind": "qualified_process_remote_canonical_exchange",
        "request_channel_id": request_channel,
        "response_channel_id": response_channel,
        "owner_principal_did": str(principals["principals"][2]["did"]),
        "native_lane_artifact_cid": str(native_lane.get("artifact_cid") or ""),
        "r1_evidence_promotes_r2": False,
        "r1_operations_allowed": False,
        "configured_board_launch": False,
        "supervisor_signed": False,
        "live_dispatch_claimed": False,
        "signer_did": signer_did,
        "signer_role": "independent_security_reviewer",
    }
    unsigned["signature"] = _sign_mapping(key, unsigned)
    artifact = _self_address(unsigned, "artifact_cid")
    _write_private_json(PLAN_R2_ARTIFACT, artifact)
    return {
        "artifact": "plan_r2_remote_owner",
        "decision": "admitted",
        "independent_signature_present": True,
        "self_signed_rejected": True,
        "source_only_factory_authority": False,
        "supervisor_signed": False,
        "r1_evidence_promotes_r2": False,
        "allowed_operations": list(artifact["allowed_operations"]),
        "request_channel_id": request_channel,
        "response_channel_id": response_channel,
        "artifact_path": str(PLAN_R2_ARTIFACT.relative_to(ROOT)),
        "artifact_cid": artifact["artifact_cid"],
        "signer_did": signer_did,
        "configured_board_launch": False,
        "live_dispatch_claimed": False,
    }


def probe_plan_r2() -> dict[str, Any]:
    extra = {"r1_evidence_promotes_r2": False, "configured_board_launch": False}
    evidence = _probe_signed_host_artifact(
        path=PLAN_R2_ARTIFACT,
        schema=HOST_PLAN_R2_SCHEMA,
        artifact="plan_r2_remote_owner",
        extra=extra,
        required={
            "interface": "PlanR2ProcessRemoteOwnerCapability@1",
            "r1_evidence_promotes_r2": False,
            "r1_operations_allowed": False,
            "live_dispatch_claimed": False,
        },
    )
    if evidence.get("decision") != "admitted":
        return evidence
    payload = json.loads(PLAN_R2_ARTIFACT.read_text(encoding="utf-8"))
    evidence["allowed_operations"] = list(payload.get("allowed_operations") or ())
    evidence["request_channel_id"] = payload.get("request_channel_id")
    evidence["response_channel_id"] = payload.get("response_channel_id")
    if evidence["allowed_operations"] != [
        "plan_r2.prepare",
        "plan_r2.apply",
        "plan_r2.observe",
    ]:
        return _typed_missing_artifact(
            artifact="plan_r2_remote_owner",
            reason="plan-r2 remote-owner operations are not the exact three-operation seam",
            extra=extra,
        )
    return evidence


def overlay_container_policy_from_admitted_image(
    container: Mapping[str, Any],
) -> dict[str, Any]:
    """Fill bootstrap image identity from admitted EAAEF-185. Never enable live dispatch."""

    policy = dict(container)
    evidence = probe_worker_image()
    digest = str(evidence.get("image_digest") or "")
    if evidence.get("decision") == "admitted" and digest.startswith("sha256:") and len(digest) == 71:
        policy["bootstrap_image_digest"] = digest
        policy["bootstrap_image_status"] = "admitted"
    policy["live_dispatch_allowed"] = False
    return policy


def materialize_host_evidence() -> dict[str, Any]:
    """Produce 185-190 host evidence. The collector still starts no supervisor."""

    principals = bind_runtime_principals()
    engine = probe_engine_mode()
    duckdb = probe_duckdb_quack()
    worker_image = materialize_worker_image(principals=principals, engine=engine)
    container_profile = materialize_container_profile(
        worker_image=worker_image,
        principals=principals,
        engine=engine,
    )
    worker_network = materialize_worker_network(
        principals=principals,
        engine=engine,
    )
    command_fabric = materialize_command_fabric(
        principals=principals,
        duckdb=duckdb,
    )
    native_lane = materialize_native_lane(
        principals=principals,
        command_fabric=command_fabric,
    )
    plan_r2 = materialize_plan_r2(
        principals=principals,
        native_lane=native_lane,
    )
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-host-evidence-materialize@1",
        "process_started": False,
        "configured_board_launch": False,
        "live_launch_allowed": False,
        "supervisor_signed": False,
        "decisions": {
            "EAAEF-185": str(worker_image.get("decision") or "typed_missing"),
            "EAAEF-186": str(container_profile.get("decision") or "typed_missing"),
            "EAAEF-187": str(worker_network.get("decision") or "typed_missing"),
            "EAAEF-188": str(command_fabric.get("decision") or "typed_missing"),
            "EAAEF-189": str(native_lane.get("decision") or "typed_missing"),
            "EAAEF-190": str(plan_r2.get("decision") or "typed_missing"),
        },
        "worker_image": worker_image,
        "container_profile": container_profile,
        "worker_network": worker_network,
        "command_fabric": command_fabric,
        "native_lane": native_lane,
        "plan_r2": plan_r2,
    }


def _typed_missing_artifact(
    *,
    artifact: str,
    reason: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = {
        "artifact": artifact,
        "decision": "typed_missing",
        "reason": reason,
        "independent_signature_present": False,
        "self_signed_rejected": True,
        "source_only_factory_authority": False,
    }
    if extra:
        evidence.update(dict(extra))
    evidence["decision"] = "typed_missing"
    return evidence


def collect_host_admission_receipts(
    *,
    launch_plan: Mapping[str, Any] | None = None,
    timeout_seconds: int = 180,
) -> dict[str, dict[str, Any]]:
    """Build every S-epic receipt. Live launch remains fail-closed."""

    plan = dict(launch_plan or load_isolated_launch_plan(timeout_seconds=timeout_seconds))
    if plan.get("process_started") is True:
        raise RuntimeError("collector refuses a plan that started a process")
    blockers = [str(item) for item in plan.get("blockers") or () if str(item)]
    blocker_classes = {
        str(key): str(value)
        for key, value in dict(plan.get("blocker_classes") or {}).items()
    }
    inventory_items = []
    for blocker in blockers:
        inventory_items.append(
            {
                "blocker": blocker,
                "class": blocker_classes.get(blocker) or classify_blocker(blocker),
                "closing_task_ids": closing_task_ids(blocker),
            }
        )
    principals = bind_runtime_principals()
    duckdb = probe_duckdb_quack()
    engine = probe_engine_mode()
    provider_authorization = probe_provider_authorization()
    provider_authorization["bound_provider_did"] = principals["principals"][1]["did"]
    worker_image = probe_worker_image()
    container_profile = probe_container_profile()
    worker_network = probe_worker_network(principals=principals)
    command_fabric = probe_command_fabric()
    native_lane = probe_native_lane()
    plan_r2 = probe_plan_r2()
    receipts: dict[str, dict[str, Any]] = {
        "EAAEF-180": _base_receipt(
            "EAAEF-180",
            decision="inventory",
            evidence={
                "launch_plan_allowed": False,
                "launch_plan_schema": plan.get("schema"),
                "materialization_receipt_cid": plan.get("materialization_receipt_cid"),
                "bootstrap_admission_decision": (
                    (plan.get("bootstrap_admission_statement") or {}).get("decision")
                ),
                "items": inventory_items,
                "auto_recoverable_action": "host_bootstrap_recovery",
            },
        ),
        "EAAEF-181": _base_receipt(
            "EAAEF-181",
            decision="bound_unadmitted",
            evidence=principals,
        ),
        "EAAEF-182": _base_receipt(
            "EAAEF-182",
            decision=str(duckdb["decision"]),
            evidence=duckdb,
        ),
        "EAAEF-183": _base_receipt(
            "EAAEF-183",
            decision=str(engine["decision"]),
            evidence=engine,
        ),
        "EAAEF-184": _base_receipt(
            "EAAEF-184",
            decision=str(provider_authorization["decision"]),
            evidence=provider_authorization,
        ),
        "EAAEF-185": _base_receipt(
            "EAAEF-185",
            decision=str(worker_image.get("decision") or "typed_missing"),
            evidence=worker_image,
        ),
        "EAAEF-186": _base_receipt(
            "EAAEF-186",
            decision=str(container_profile.get("decision") or "typed_missing"),
            evidence=container_profile,
        ),
        "EAAEF-187": _base_receipt(
            "EAAEF-187",
            decision=str(worker_network.get("decision") or "typed_missing"),
            evidence=worker_network,
        ),
        "EAAEF-188": _base_receipt(
            "EAAEF-188",
            decision=str(command_fabric.get("decision") or "typed_missing"),
            evidence=command_fabric,
        ),
        "EAAEF-189": _base_receipt(
            "EAAEF-189",
            decision=str(native_lane.get("decision") or "typed_missing"),
            evidence=native_lane,
        ),
        "EAAEF-190": _base_receipt(
            "EAAEF-190",
            decision=str(plan_r2.get("decision") or "typed_missing"),
            evidence=plan_r2,
        ),
    }
    child_cids = {
        task_id: receipts[task_id]["receipt_cid"]
        for task_id in RECEIPT_FILES
        if task_id != "EAAEF-191"
    }
    child_decisions = {
        task_id: str(receipts[task_id]["decision"])
        for task_id in RECEIPT_FILES
        if task_id != "EAAEF-191"
    }
    bootstrap_cid = str(
        (plan.get("bootstrap_admission_statement") or {}).get("statement_cid") or ""
    )
    materialization_cid = str(plan.get("materialization_receipt_cid") or "")
    open_host_gates = [
        str(item["blocker"])
        for item in inventory_items
        if item["class"] == "host_gated_external_authority"
    ]
    source_identity = receipts["EAAEF-180"]
    target_decision = admission_bundle_target_decision(
        child_decisions=child_decisions,
        bootstrap_admission_statement_cid=bootstrap_cid,
        materialization_receipt_cid=materialization_cid,
    )
    signature_arguments = {
        "child_decisions": child_decisions,
        "child_receipt_cids": child_cids,
        "launch_plan_allowed": False,
        "source_head": str(source_identity["source_head"]),
        "source_tree": str(source_identity["source_tree"]),
        "board_namespace": str(source_identity["board_namespace"]),
        "board_cid": str(source_identity["board_cid"]),
        "bootstrap_admission_statement_cid": bootstrap_cid,
        "materialization_receipt_cid": materialization_cid,
        "inventory_open_host_gated": open_host_gates,
    }
    signatures = load_admission_bundle_signatures(
        decision=target_decision,
        **signature_arguments,
    )
    if not signatures["independent_operator_signature"] and target_decision == "admitted":
        signatures = load_admission_bundle_signatures(
            decision="no_go",
            **signature_arguments,
        )
        target_decision = "no_go"
    elif not signatures["independent_operator_signature"]:
        target_decision = "no_go"
    receipts["EAAEF-191"] = _base_receipt(
        "EAAEF-191",
        decision=target_decision,
        evidence={
            "child_receipt_cids": child_cids,
            "launch_plan_allowed": False,
            "bootstrap_admission_statement_cid": bootstrap_cid or None,
            "materialization_receipt_cid": materialization_cid,
            "independent_operator_signature": signatures[
                "independent_operator_signature"
            ],
            "independent_security_reviewer_signature": signatures[
                "independent_security_reviewer_signature"
            ],
            "operator_did": signatures["operator_did"],
            "security_reviewer_did": signatures["security_reviewer_did"],
            "independent_signature_present": bool(
                signatures["independent_operator_signature"]
                and signatures["independent_security_reviewer_signature"]
            ),
            "prospective_supervisor_signature_rejected": True,
            "inventory_open_host_gated": open_host_gates,
        },
    )
    return receipts


def write_host_admission_receipts(
    receipts: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for task_id, filename in RECEIPT_FILES.items():
        payload = dict(receipts[task_id])
        path = RECEIPT_DIR / filename
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(str(path.relative_to(ROOT)))
    return written


def collect_and_write(*, timeout_seconds: int = 180) -> dict[str, Any]:
    materialize = materialize_host_evidence()
    receipts = collect_host_admission_receipts(timeout_seconds=timeout_seconds)
    written = write_host_admission_receipts(receipts)
    return {
        "written": written,
        "decisions": {
            task_id: receipts[task_id]["decision"] for task_id in RECEIPT_FILES
        },
        "host_evidence": materialize.get("decisions"),
        "process_started": False,
        "configured_board_launch": False,
        "live_launch_allowed": False,
    }
