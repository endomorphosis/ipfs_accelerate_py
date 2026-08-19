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
import stat
import subprocess
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

RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-host-admission-receipt@1"
)
BUNDLE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-host-admission-bundle@1"
)
PRINCIPAL_STORE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-runtime-principal-secret@1"
)
REQUIRED_DUCKDB: Final = "1.5.5"
REQUIRED_QUACK: Final = "1.5.5+core"
APPROVED_IMPORT_ROOT: Final = Path(
    "/home/barberb/.local/lib/python3.12/site-packages"
)
LANE_COUNT: Final = 5
PRINCIPAL_ROLES: Final = ("worker", "provider", "quack_owner")

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "docs/architecture/external_agent_autonomous_execution_fabric"
RECEIPT_DIR = CAMPAIGN / "receipts" / "host_admission"
BOARD_PATH = CAMPAIGN / "task_board.json"
AUTHORITY_DIR = (
    ROOT
    / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
    / "authority"
    / "runtime-principals"
)
LAUNCHER = ROOT / (
    "scripts/launch_external_agent_autonomous_execution_fabric_materializer.py"
)

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
    ("container_policy", ("EAAEF-183",)),
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


def cid(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


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
        probe_quack_capabilities,
    )

    observed = str(getattr(duckdb, "__version__", "") or "")
    module_path = Path(getattr(duckdb, "__file__", "") or "").resolve()
    try:
        module_path.relative_to(APPROVED_IMPORT_ROOT.resolve())
        under_approved_root = True
    except (ValueError, OSError):
        under_approved_root = False
    report = probe_quack_capabilities(
        allow_network_install=False,
        allow_local_load=True,
        use_cache=False,
    )
    exact_duckdb = observed == REQUIRED_DUCKDB
    exact_quack = (
        report.passes_health_check
        and "1.5.5" in str(report.extension_fingerprint or report.duckdb_version or "")
    )
    if exact_duckdb and exact_quack and under_approved_root:
        decision = "admitted"
    else:
        decision = "typed_missing"
    return {
        "decision": decision,
        "required_duckdb": REQUIRED_DUCKDB,
        "required_quack": REQUIRED_QUACK,
        "observed_duckdb": observed,
        "observed_module_path": str(module_path),
        "under_approved_import_root": under_approved_root,
        "silent_substitution_refused": observed != REQUIRED_DUCKDB,
        "quack_probe": report.to_dict(),
        "network_install_attempted": False,
    }


def probe_engine_mode() -> dict[str, Any]:
    """Record rootless presence or an unsigned rootful fallback package."""

    completed = subprocess.run(
        ["docker", "info", "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    info: dict[str, Any] = {}
    if completed.returncode == 0 and completed.stdout.strip():
        try:
            info = json.loads(completed.stdout)
        except json.JSONDecodeError:
            info = {}
    security = [str(item) for item in info.get("SecurityOptions") or ()]
    rootless = any("rootless" in item.casefold() for item in security)
    root_dir = str(info.get("DockerRootDir") or "")
    server_version = str(info.get("ServerVersion") or "")
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
    if rootless:
        decision = "admitted"
        mode = "verified_rootless"
    elif info:
        decision = "typed_missing"
        mode = "rootful_host_daemon_unsigned_fallback"
    else:
        decision = "typed_missing"
        mode = "engine_unavailable"
    return {
        "decision": decision,
        "mode": mode,
        "rootless": rootless,
        "docker_socket_mounted": False,
        "supervisor_started": False,
        "fallback_package": fallback if not rootless else None,
        "docker_info_returncode": completed.returncode,
    }


def _typed_missing_artifact(
    *,
    artifact: str,
    reason: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = {
        "artifact": artifact,
        "reason": reason,
        "independent_signature_present": False,
        "self_signed_rejected": True,
        "source_only_factory_authority": False,
    }
    if extra:
        evidence.update(dict(extra))
    return evidence


def collect_host_admission_receipts(
    *,
    launch_plan: Mapping[str, Any] | None = None,
    timeout_seconds: int = 180,
) -> dict[str, dict[str, Any]]:
    """Build every S-epic receipt. Live launch remains fail-closed."""

    plan = dict(launch_plan or load_isolated_launch_plan(timeout_seconds=timeout_seconds))
    if plan.get("process_started") is True or plan.get("allowed") is True:
        raise RuntimeError("collector refuses a live-launch-allowed plan")
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
            decision="typed_missing",
            evidence=_typed_missing_artifact(
                artifact="eaaef_scoped_provider_authorization",
                reason="independently signed grok_cli/codex provider authorization is absent",
                extra={"bound_provider_did": principals["principals"][1]["did"]},
            ),
        ),
        "EAAEF-185": _base_receipt(
            "EAAEF-185",
            decision="typed_missing",
            evidence=_typed_missing_artifact(
                artifact="task_capable_worker_image_and_sbom",
                reason="independently signed network-none worker image, SBOM and five slot identities are absent",
                extra={"required_worker_slots": LANE_COUNT, "live_dispatch_claimed": False},
            ),
        ),
        "EAAEF-186": _base_receipt(
            "EAAEF-186",
            decision="typed_missing",
            evidence=_typed_missing_artifact(
                artifact="container_execution_profile_v2",
                reason="independently signed execution-profile @2 is absent; unsigned @1 cannot satisfy this gate",
            ),
        ),
        "EAAEF-187": _base_receipt(
            "EAAEF-187",
            decision="typed_missing",
            evidence=_typed_missing_artifact(
                artifact="worker_network_authorizations",
                reason="five collision-free signed lane authorizations are absent; child propagation remains fail-closed",
                extra={
                    "required_lanes": LANE_COUNT,
                    "worker_did": principals["principals"][0]["did"],
                    "provider_did": principals["principals"][1]["did"],
                    "child_propagation_status": "unavailable_fail_closed",
                },
            ),
        ),
        "EAAEF-188": _base_receipt(
            "EAAEF-188",
            decision="typed_missing",
            evidence=_typed_missing_artifact(
                artifact="signed_command_fabric_endpoints",
                reason="deployed signed command-authorizer, Quack ingress/projection and dispatcher endpoints are absent",
                extra={
                    "child_adapter_status": "implemented_unqualified_fail_closed",
                    "implemented_unqualified_fail_closed_admitted": False,
                },
            ),
        ),
        "EAAEF-189": _base_receipt(
            "EAAEF-189",
            decision="typed_missing",
            evidence=_typed_missing_artifact(
                artifact="native_lane_dispatcher_artifacts",
                reason="independently signed native-dependency, V2 lane/verifier/merge and dispatcher-service artifacts are absent",
            ),
        ),
        "EAAEF-190": _base_receipt(
            "EAAEF-190",
            decision="typed_missing",
            evidence=_typed_missing_artifact(
                artifact="plan_r2_remote_owner",
                reason="independently signed prepare/apply/observe remote-owner capability and process-remote channel are absent",
                extra={"r1_evidence_promotes_r2": False},
            ),
        ),
    }
    child_cids = {
        task_id: receipts[task_id]["receipt_cid"]
        for task_id in RECEIPT_FILES
        if task_id != "EAAEF-191"
    }
    receipts["EAAEF-191"] = _base_receipt(
        "EAAEF-191",
        decision="no_go",
        evidence={
            "child_receipt_cids": child_cids,
            "launch_plan_allowed": False,
            "bootstrap_admission_statement_cid": (
                (plan.get("bootstrap_admission_statement") or {}).get("statement_cid")
            ),
            "materialization_receipt_cid": plan.get("materialization_receipt_cid"),
            "independent_operator_signature": "",
            "independent_security_reviewer_signature": "",
            "prospective_supervisor_signature_rejected": True,
            "inventory_open_host_gated": [
                item["blocker"]
                for item in inventory_items
                if item["class"] == "host_gated_external_authority"
            ],
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
    receipts = collect_host_admission_receipts(timeout_seconds=timeout_seconds)
    written = write_host_admission_receipts(receipts)
    return {
        "written": written,
        "decisions": {
            task_id: receipts[task_id]["decision"] for task_id in RECEIPT_FILES
        },
        "process_started": False,
    }
