"""Effect-free admission preflight for the EAAEF bootstrap supervisor.

This module deliberately does not launch a process, create a container, open a
control-plane database, or mint authority.  It joins evidence that must already
exist and returns deterministic reason codes.  The configured-board
multi-supervisor capsule gate remains a separate mandatory gate for that launch
path; this preflight can admit only the direct, single-supervisor bootstrap.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from ipfs_accelerate_py.agent_implementation_route import (
    AgentImplementationRouteAuthorization,
    AgentImplementationRoutePlan,
    resolve_agent_implementation_route,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    LocalProfileTampered,
    verify_did_key_signature,
)

EAAEF_BOARD_NAMESPACE: Final = "external-agent-autonomous-execution-fabric-v1"
EAAEF_BOARD_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "external-agent-autonomous-execution-fabric-board@1"
)
EAAEF_MATERIALIZATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "external-agent-autonomous-execution-fabric-materialization@1"
)
EAAEF_BOOTSTRAP_PREFLIGHT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "external-agent-autonomous-execution-fabric-bootstrap-preflight@1"
)
EAAEF_CONTAINER_IMAGE_QUALIFICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "external-agent-container-image-qualification@1"
)
EAAEF_CONTAINER_PROFILE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "external-agent-bootstrap-container-profile@1"
)
EAAEF_DIRECT_LAUNCH_MODE: Final = "direct_single_supervisor"
EAAEF_PRIMARY_MODEL_ID: Final = "grok-4.6"

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}")
_ALLOWED_SBOM_FORMATS: Final = frozenset(
    {"spdx-json", "cyclonedx-json"}
)
_EXPECTED_CONTAINER_ENV: Final = {
    "BASH_ENV": "",
    "CODEX_HOME": "/opt/codex-home",
    "ENV": "",
    "HOME": "/opt/codex-home",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/opt/ipfs-task-tools/bin:/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": "/opt/ipfs-validation-site-packages",
    "TERM": "dumb",
}
_CONTAINER_PROFILE_FIELDS: Final = frozenset(
    {
        "schema",
        "runtime",
        "image_digest",
        "rootless",
        "nonroot_user",
        "read_only_base",
        "network_mode",
        "cap_drop",
        "no_new_privileges",
        "pids_limit",
        "cpu_limit",
        "memory_limit_bytes",
        "disk_limit_bytes",
        "gpu",
        "privileged",
        "host_pid",
        "host_ipc",
        "devices",
        "docker_socket_mounted",
        "inherit_host_environment",
        "environment",
        "mounts",
        "profile_cid",
    }
)
_IMAGE_QUALIFICATION_FIELDS: Final = frozenset(
    {
        "schema",
        "image_digest",
        "image_label",
        "image_os",
        "image_architecture",
        "sbom_digest",
        "sbom_format",
        "sbom_bytes",
        "toolchain_versions",
        "rootless_verified",
        "reviewer_identity_did",
        "reviewer_role",
        "verified_at_ms",
        "expires_at_ms",
        "reviewer_signature",
        "qualification_cid",
    }
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _cid(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _self_addressed(value: Mapping[str, Any], field: str) -> bool:
    identity = value.get(field)
    body = dict(value)
    body.pop(field, None)
    return isinstance(identity, str) and identity == _cid(body)


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _positive_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value > 0
    )


def _validate_board(board: object) -> str:
    if not isinstance(board, Mapping):
        return "board_missing_or_invalid"
    identity = board.get("board_cid")
    body = dict(board)
    body.pop("board_cid", None)
    if (
        board.get("schema") != EAAEF_BOARD_SCHEMA
        or board.get("board_namespace") != EAAEF_BOARD_NAMESPACE
        or board.get("parent_objective")
        != "ExternalAgentAutonomousExecutionFabric"
        or not isinstance(board.get("goals"), list)
        or not isinstance(board.get("tasks"), list)
        or not isinstance(identity, str)
        or identity != _cid(body)
    ):
        return "board_missing_or_invalid"
    return ""


def _validate_materialization(
    receipt: object,
    *,
    board: Mapping[str, Any],
) -> str:
    if not isinstance(receipt, Mapping):
        return "materialization_receipt_missing_or_invalid"
    board_validation = receipt.get("board_validation")
    source_generation = receipt.get("source_generation")
    if (
        receipt.get("schema") != EAAEF_MATERIALIZATION_SCHEMA
        or not _self_addressed(receipt, "receipt_cid")
        or receipt.get("authority_mode") != "embedded"
        or receipt.get("maximum_writer_processes") != 1
        or receipt.get("continuous_quack_authority") is not False
        or receipt.get("ducklake_authority") is not False
        or receipt.get("process_started") is not False
        or not isinstance(board_validation, Mapping)
        or board_validation.get("valid") is not True
        or board_validation.get("board_cid") != board.get("board_cid")
        or board_validation.get("source_forest_root")
        != board.get("source_forest_root")
        or not isinstance(source_generation, Mapping)
        or not _SHA256.fullmatch(
            str(source_generation.get("source_generation_cid") or "")
        )
        or not _GIT_OBJECT.fullmatch(str(receipt.get("source_head") or ""))
        or not _GIT_OBJECT.fullmatch(str(receipt.get("source_tree") or ""))
        or not _SHA256.fullmatch(str(receipt.get("population_cid") or ""))
        or not _SHA256.fullmatch(str(receipt.get("plan_root_cid") or ""))
    ):
        return "materialization_receipt_missing_or_invalid"
    source_body = dict(source_generation)
    source_identity = str(source_body.pop("source_generation_cid", ""))
    if source_identity != _cid(source_body):
        return "materialization_source_generation_invalid"
    return ""


def _validate_route(route: object) -> str:
    if not isinstance(route, AgentImplementationRoutePlan):
        return "eaaef_scoped_provider_authorization_missing"
    authorization = route.authorization
    if not isinstance(authorization, AgentImplementationRouteAuthorization):
        return "eaaef_scoped_provider_authorization_missing"
    if (
        authorization.board_namespace != EAAEF_BOARD_NAMESPACE
        or not authorization.artifact_path.startswith(
            "data/agent_supervisor/external_agent_autonomous_execution_fabric/"
        )
        or route.primary_model_id != EAAEF_PRIMARY_MODEL_ID
        or route.primary_provider_id != "grok_cli"
        or route.fallback_provider_id != "codex"
        or route.fallback_model_id != "gpt-5.6-terra"
        or route.fallback_reasoning_effort != "high"
    ):
        return "eaaef_scoped_provider_authorization_invalid"
    try:
        resolved = resolve_agent_implementation_route(
            **route.as_dict(),
            authorization=authorization,
        )
    except ValueError:
        return "eaaef_scoped_provider_authorization_invalid"
    if (
        resolved.authorization != authorization
        or resolved.route_id != route.route_id
        or resolved.as_dict() != route.as_dict()
    ):
        return "eaaef_scoped_provider_authorization_invalid"
    return ""


def _image_signed_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("reviewer_signature", None)
    body.pop("qualification_cid", None)
    return body


def _validate_image_qualification(
    value: object,
    *,
    trusted_reviewer_dids: frozenset[str],
    now_ms: int,
) -> str:
    if not isinstance(value, Mapping):
        return "oci_image_qualification_missing"
    reviewer = str(value.get("reviewer_identity_did") or "")
    toolchains = value.get("toolchain_versions")
    if (
        set(value) != _IMAGE_QUALIFICATION_FIELDS
        or value.get("schema") != EAAEF_CONTAINER_IMAGE_QUALIFICATION_SCHEMA
        or not _SHA256.fullmatch(str(value.get("image_digest") or ""))
        or not str(value.get("image_label") or "")
        or value.get("image_os") != "linux"
        or not str(value.get("image_architecture") or "")
        or not _SHA256.fullmatch(str(value.get("sbom_digest") or ""))
        or value.get("sbom_format") not in _ALLOWED_SBOM_FORMATS
        or not _positive_integer(value.get("sbom_bytes"))
        or not isinstance(toolchains, Mapping)
        or not toolchains
        or any(not str(key) or not str(item) for key, item in toolchains.items())
        or value.get("rootless_verified") is not True
        or reviewer not in trusted_reviewer_dids
        or value.get("reviewer_role") != "independent_security_reviewer"
        or not _positive_integer(value.get("verified_at_ms"))
        or not _positive_integer(value.get("expires_at_ms"))
        or int(value.get("verified_at_ms") or 0) > now_ms
        or now_ms > int(value.get("expires_at_ms") or 0)
        or not _self_addressed(value, "qualification_cid")
    ):
        return "oci_image_qualification_invalid"
    signature = value.get("reviewer_signature")
    if not isinstance(signature, str) or not signature:
        return "oci_image_qualification_invalid"
    try:
        verify_did_key_signature(
            identity_did=reviewer,
            payload=_image_signed_payload(value),
            signature=signature,
        )
    except (LocalProfileTampered, ValueError):
        return "oci_image_qualification_invalid"
    return ""


def _validate_mounts(value: object) -> bool:
    if not isinstance(value, list) or not value:
        return False
    writable_targets: list[str] = []
    targets: set[str] = set()
    for mount in value:
        if not isinstance(mount, Mapping) or set(mount) != {
            "source_identity",
            "target",
            "read_only",
            "kind",
        }:
            return False
        source = str(mount.get("source_identity") or "")
        target = str(mount.get("target") or "")
        lowered = f"{source} {target}".lower()
        if (
            not _SHA256.fullmatch(source)
            or not target.startswith("/")
            or ".." in target.split("/")
            or target in targets
            or mount.get("kind") not in {"worktree", "provider_auth", "secret"}
            or not isinstance(mount.get("read_only"), bool)
            or "docker.sock" in lowered
            or "/run/docker" in lowered
            or "/var/run/docker" in lowered
        ):
            return False
        targets.add(target)
        if mount.get("read_only") is False:
            writable_targets.append(target)
    return writable_targets == ["/workspace"]


def _validate_container_profile(
    value: object,
    *,
    image_qualification: Mapping[str, Any] | None,
) -> str:
    if not isinstance(value, Mapping):
        return "container_profile_missing"
    gpu = value.get("gpu")
    user = str(value.get("nonroot_user") or "")
    image_digest = str(value.get("image_digest") or "")
    if (
        set(value) != _CONTAINER_PROFILE_FIELDS
        or value.get("schema") != EAAEF_CONTAINER_PROFILE_SCHEMA
        or value.get("runtime") not in {"docker", "oci"}
        or not _SHA256.fullmatch(image_digest)
        or not isinstance(image_qualification, Mapping)
        or image_digest != image_qualification.get("image_digest")
        or value.get("rootless") is not True
        or re.fullmatch(r"[1-9][0-9]*:[1-9][0-9]*", user) is None
        or value.get("read_only_base") is not True
        or value.get("network_mode") != "none"
        or value.get("cap_drop") != ["ALL"]
        or value.get("no_new_privileges") is not True
        or not _positive_integer(value.get("pids_limit"))
        or not _positive_number(value.get("cpu_limit"))
        or not _positive_integer(value.get("memory_limit_bytes"))
        or not _positive_integer(value.get("disk_limit_bytes"))
        or not isinstance(gpu, Mapping)
        or set(gpu) != {"mode", "device_ids", "memory_limit_bytes"}
        or gpu.get("mode") != "none"
        or gpu.get("device_ids") != []
        or gpu.get("memory_limit_bytes") != 0
        or value.get("privileged") is not False
        or value.get("host_pid") is not False
        or value.get("host_ipc") is not False
        or value.get("devices") != []
        or value.get("docker_socket_mounted") is not False
        or value.get("inherit_host_environment") is not False
        or value.get("environment") != _EXPECTED_CONTAINER_ENV
        or not _validate_mounts(value.get("mounts"))
        or not _self_addressed(value, "profile_cid")
    ):
        return "container_profile_invalid"
    return ""


@dataclass(frozen=True, slots=True)
class ExternalAgentFabricBootstrapPreflight:
    """A portable, body-free decision; never an authority grant itself."""

    allowed: bool
    blockers: tuple[str, ...]
    board_cid: str
    materialization_receipt_cid: str
    route_id: str
    image_digest: str
    container_profile_cid: str
    launch_mode: str

    def as_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema": EAAEF_BOOTSTRAP_PREFLIGHT_SCHEMA,
            "allowed": self.allowed,
            "blockers": list(self.blockers),
            "board_cid": self.board_cid,
            "materialization_receipt_cid": self.materialization_receipt_cid,
            "route_id": self.route_id,
            "image_digest": self.image_digest,
            "container_profile_cid": self.container_profile_cid,
            "launch_mode": self.launch_mode,
            "configured_board_capsule_gate_bypassed": False,
            "authority_mutated": False,
            "process_started": False,
        }
        body["preflight_cid"] = _cid(body)
        return body


def evaluate_external_agent_fabric_bootstrap_preflight(
    *,
    board: object,
    materialization_receipt: object,
    route_plan: object,
    image_qualification: object,
    container_profile: object,
    trusted_image_reviewer_dids: Sequence[str] = (),
    now_ms: int,
    launch_mode: str = EAAEF_DIRECT_LAUNCH_MODE,
    configured_board_capsule_gate_bypassed: bool = False,
) -> ExternalAgentFabricBootstrapPreflight:
    """Join immutable bootstrap evidence without performing any authority effect."""

    blockers: list[str] = []
    board_mapping = board if isinstance(board, Mapping) else {}
    receipt_mapping = (
        materialization_receipt
        if isinstance(materialization_receipt, Mapping)
        else {}
    )
    image_mapping = (
        image_qualification if isinstance(image_qualification, Mapping) else None
    )
    profile_mapping = (
        container_profile if isinstance(container_profile, Mapping) else {}
    )

    if launch_mode != EAAEF_DIRECT_LAUNCH_MODE:
        blockers.append("direct_single_supervisor_launch_mode_required")
    if configured_board_capsule_gate_bypassed:
        blockers.append("configured_board_capsule_gate_bypass_prohibited")
    if not _positive_integer(now_ms):
        blockers.append("preflight_time_invalid")

    board_reason = _validate_board(board)
    if board_reason:
        blockers.append(board_reason)
    elif isinstance(board, Mapping):
        receipt_reason = _validate_materialization(
            materialization_receipt,
            board=board,
        )
        if receipt_reason:
            blockers.append(receipt_reason)

    route_reason = _validate_route(route_plan)
    if route_reason:
        blockers.append(route_reason)

    trusted = frozenset(
        value for value in trusted_image_reviewer_dids if isinstance(value, str)
    )
    image_reason = _validate_image_qualification(
        image_qualification,
        trusted_reviewer_dids=trusted,
        now_ms=now_ms if _positive_integer(now_ms) else 0,
    )
    if image_reason:
        blockers.append(image_reason)

    profile_reason = _validate_container_profile(
        container_profile,
        image_qualification=image_mapping,
    )
    if profile_reason:
        blockers.append(profile_reason)

    return ExternalAgentFabricBootstrapPreflight(
        allowed=not blockers,
        blockers=tuple(blockers),
        board_cid=str(board_mapping.get("board_cid") or ""),
        materialization_receipt_cid=str(
            receipt_mapping.get("receipt_cid") or ""
        ),
        route_id=(
            route_plan.route_id
            if isinstance(route_plan, AgentImplementationRoutePlan)
            else ""
        ),
        image_digest=str(
            (image_mapping or {}).get("image_digest") or ""
        ),
        container_profile_cid=str(profile_mapping.get("profile_cid") or ""),
        launch_mode=launch_mode,
    )


__all__ = (
    "EAAEF_BOARD_NAMESPACE",
    "EAAEF_BOOTSTRAP_PREFLIGHT_SCHEMA",
    "EAAEF_CONTAINER_IMAGE_QUALIFICATION_SCHEMA",
    "EAAEF_CONTAINER_PROFILE_SCHEMA",
    "EAAEF_DIRECT_LAUNCH_MODE",
    "ExternalAgentFabricBootstrapPreflight",
    "evaluate_external_agent_fabric_bootstrap_preflight",
)
