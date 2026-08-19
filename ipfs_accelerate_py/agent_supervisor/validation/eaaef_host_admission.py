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
    installed_from = ""
    if report.extension is not None:
        installed_from = str(report.extension.installed_from or "")
    exact_quack = (
        report.passes_health_check
        and exact_duckdb
        and installed_from == "core"
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
    provider_authorization = probe_provider_authorization()
    provider_authorization["bound_provider_did"] = principals["principals"][1]["did"]
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
