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
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from ipfs_accelerate_py.agent_implementation_route import (
    AgentImplementationInvocationBinding,
    AgentImplementationRouteAuthorization,
    AgentImplementationRoutePlan,
    agent_implementation_route_review_payload,
    eaaef_agent_route_authorization_path,
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
    "external-agent-autonomous-execution-fabric-materialization@2"
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
EAAEF_WORKER_IMAGE_QUALIFICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "external-agent-worker-image-qualification@1"
)
EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "external-agent-worker-container-profile@1"
)
EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA_V2: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "external-agent-worker-container-profile@2"
)
EAAEF_WORKER_CONTAINER_PROFILE_REVIEWER_ROLE_V2: Final = (
    "independent_grok_container_security_reviewer"
)
EAAEF_PROVIDER_CONTAINER_QUALIFICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-provider-container-qualification@1"
)
EAAEF_PROVIDER_CONTAINER_VERIFICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-provider-container-qualification-verification@1"
)
EAAEF_REPOSITORY_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-bootstrap-repository-binding@1"
)
EAAEF_PROVIDER_BUDGET_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-bootstrap-provider-budget-binding@1"
)
EAAEF_DIRECT_LAUNCH_MODE: Final = "direct_single_supervisor"
EAAEF_PRIMARY_MODEL_ID: Final = "grok-4.6"
EAAEF_BOOTSTRAP_TASK_ID: Final = "EAAEF-000"
EAAEF_QUALIFICATION_SIGNER_ROLE: Final = (
    "independent_bootstrap_admission_reviewer"
)
EAAEF_REQUIRED_ROUTE_EFFECTS: Final = (
    "edit",
    "isolated_worktree",
    "test",
)
EAAEF_ROUTE_AUTHORIZATION_PATH_PREFIX: Final = (
    "data/agent_supervisor/external_agent_autonomous_execution_fabric/"
    "authority/provider-route-authorization-"
)
EAAEF_ROUTE_ID: Final = (
    "agent-supervisor-eaaef-v1-grok46-terra56-high-auth-or-hard-quota-v1"
)
EAAEF_INVOCATION_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.provider-fallback-invocation@2"
)
EAAEF_BOOTSTRAP_POLICY_CID: Final = (
    "sha256:dc6eca2b7f0c4838fddc680160387541eaf11229188e1e53af0a2e66e8031534"
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_GIT_OBJECT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_ALLOWED_SBOM_FORMATS: Final = frozenset(
    {"spdx-json", "cyclonedx-json"}
)
_MAX_PIDS: Final = 4096
_MAX_CPU: Final = 64.0
_MAX_MEMORY_BYTES: Final = 256 * 1024**3
_MAX_DISK_BYTES: Final = 2 * 1024**4
_MAX_QUALIFICATION_LIFETIME_MS: Final = 24 * 60 * 60 * 1000
_MAX_INVOCATION_LIFETIME_MS: Final = 5 * 60 * 1000
_MAX_PARALLEL_WORKERS: Final = 5
_MAX_PARALLEL_CONTAINERS: Final = 5
_ADMITTED_ROOTFUL_FALLBACK_POLICY_CIDS: Final = frozenset()
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
        "workload_class",
        "task_dispatch_admitted",
        "execution_mode",
        "rootless_supported",
        "daemon_identity_cid",
        "daemon_policy_cid",
        "bootstrap_policy_cid",
        "rootful_fallback_admitted",
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
        "maximum_parallel_workers",
        "maximum_parallel_containers",
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
        "workload_class",
        "task_dispatch_verified",
        "execution_mode",
        "rootless_supported",
        "rootless_verified",
        "nonroot_hardening_verified",
        "daemon_identity_cid",
        "daemon_policy_cid",
        "reviewer_identity_did",
        "reviewer_role",
        "verified_at_ms",
        "expires_at_ms",
        "reviewer_signature",
        "qualification_cid",
    }
)
_WORKER_IMAGE_QUALIFICATION_FIELDS: Final = _IMAGE_QUALIFICATION_FIELDS | {
    "credential_disposition",
    "credential_disposition_evidence_cid",
    "reproducible_build_evidence_cid",
    "reproducible_build_count",
    "network_policy_cid",
}
_WORKER_CONTAINER_PROFILE_FIELDS: Final = _CONTAINER_PROFILE_FIELDS | {
    "image_qualification_cid",
    "sbom_digest",
    "toolchain_versions",
    "network_policy_cid",
    "resource_profile_cid",
    "worker_principal_did",
    "provider_principal_did",
    "reviewer_identity_did",
    "reviewer_role",
    "reviewed_at_ms",
    "expires_at_ms",
    "reviewer_signature",
}
_PROVIDER_CONTAINER_QUALIFICATION_FIELDS: Final = frozenset(
    {
        "schema",
        "board_cid",
        "source_head",
        "source_tree",
        "source_generation_cid",
        "materialization_receipt_cid",
        "control_projection_root",
        "coordination_projection_root",
        "execution_projection_root",
        "control_plane_schema_version",
        "route_id",
        "route_authorization_id",
        "route_authorization_sha256",
        "route_repository_cid",
        "route_baseline_commit",
        "route_effects",
        "route_budget_cid",
        "route_resource_cid",
        "image_qualification_cid",
        "image_digest",
        "sbom_digest",
        "container_profile_cid",
        "execution_mode",
        "daemon_identity_cid",
        "daemon_policy_cid",
        "workload_class",
        "task_dispatch_admitted",
        "worker_principal_did",
        "provider_principal_did",
        "maximum_parallel_workers",
        "maximum_parallel_containers",
        "admitted_at_ms",
        "expires_at_ms",
        "signer_identity_did",
        "signer_role",
        "signer_signature",
        "receipt_cid",
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


def _nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
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
    bounds = authorization.authority_bounds
    if (
        authorization.board_namespace != EAAEF_BOARD_NAMESPACE
        or authorization.artifact_path
        != eaaef_agent_route_authorization_path(authorization.source_tree)
        or route.route_id != EAAEF_ROUTE_ID
        or route.primary_model_id != EAAEF_PRIMARY_MODEL_ID
        or route.primary_provider_id != "grok_cli"
        or route.fallback_provider_id != "codex"
        or route.fallback_model_id != "gpt-5.6-terra"
        or route.fallback_reasoning_effort != "high"
        or authorization.authorization_kind != "explicit_operator_override"
        or not _SHA256.fullmatch(authorization.artifact_sha256)
        or not _GIT_OBJECT.fullmatch(authorization.source_head)
        or not _GIT_OBJECT.fullmatch(authorization.source_tree)
        or not authorization.reviewer_identity.startswith("did:key:z")
        or authorization.reviewer_provider != "local_operator"
        or not authorization.reviewer_signature
        or authorization.fallback_implementer_identity != "codex"
        or bounds is None
    ):
        return "eaaef_scoped_provider_authorization_invalid"
    assert bounds is not None
    artifact_route = {
        "route_id": route.route_id,
        "primary_provider_id": route.primary_provider_id,
        "primary_model_id": route.primary_model_id,
        "fallback_provider_id": route.fallback_provider_id,
        "fallback_model_id": route.fallback_model_id,
        "fallback_reasoning_effort": route.fallback_reasoning_effort,
        "allowed_trigger_classes": [
            "grok_authentication_unavailable",
            "grok_hard_quota_exhausted",
        ],
    }
    identity_body = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "provider-fallback-policy-authorization@2"
        ),
        "board_namespace": authorization.board_namespace,
        "artifact_path": authorization.artifact_path,
        "artifact_sha256": authorization.artifact_sha256,
        "authorization_kind": authorization.authorization_kind,
        "source_head": authorization.source_head,
        "source_tree": authorization.source_tree,
        "reviewer_identity": authorization.reviewer_identity,
        "reviewer_provider": authorization.reviewer_provider,
        "reviewer_signature": authorization.reviewer_signature,
        "reviewer_profile_id": authorization.reviewer_profile_id,
        "reviewer_profile_content_id": (
            authorization.reviewer_profile_content_id
        ),
        "reviewer_lifecycle_anchor_id": (
            authorization.reviewer_lifecycle_anchor_id
        ),
        "reviewer_lifecycle_generation": (
            authorization.reviewer_lifecycle_generation
        ),
        "reviewer_witness_path": authorization.reviewer_witness_path,
        "reviewer_witness_sha256": authorization.reviewer_witness_sha256,
        "lifecycle_root_identity_did": (
            authorization.lifecycle_root_identity_did
        ),
        "lifecycle_witness_nonce": authorization.lifecycle_witness_nonce,
        "lifecycle_root_pin_path": authorization.lifecycle_root_pin_path,
        "lifecycle_root_pin_sha256": (
            authorization.lifecycle_root_pin_sha256
        ),
        "authorized_at_ms": authorization.authorized_at_ms,
        "fallback_implementer_identity": (
            authorization.fallback_implementer_identity
        ),
        "authority_bounds": bounds.as_dict(),
    }
    if authorization.authorization_id != _cid(
        {**identity_body, "authorization_id": ""}
    ):
        return "eaaef_scoped_provider_authorization_invalid"
    review_payload = agent_implementation_route_review_payload(
        board_namespace=authorization.board_namespace,
        authorization_kind=authorization.authorization_kind,
        source_head=authorization.source_head,
        source_tree=authorization.source_tree,
        route=artifact_route,
        authority_bounds=bounds.as_dict(),
        reviewer_identity=authorization.reviewer_identity,
        reviewer_provider=authorization.reviewer_provider,
        reviewer_profile_id=authorization.reviewer_profile_id,
        reviewer_profile_content_id=(
            authorization.reviewer_profile_content_id
        ),
        reviewer_lifecycle_anchor_id=(
            authorization.reviewer_lifecycle_anchor_id
        ),
        reviewer_lifecycle_generation=(
            authorization.reviewer_lifecycle_generation
        ),
        reviewer_witness_path=authorization.reviewer_witness_path,
        reviewer_witness_sha256=authorization.reviewer_witness_sha256,
        lifecycle_root_identity_did=(
            authorization.lifecycle_root_identity_did
        ),
        lifecycle_witness_nonce=authorization.lifecycle_witness_nonce,
        lifecycle_root_pin_path=authorization.lifecycle_root_pin_path,
        lifecycle_root_pin_sha256=(
            authorization.lifecycle_root_pin_sha256
        ),
        authorized_at_ms=authorization.authorized_at_ms,
        fallback_implementer_identity=(
            authorization.fallback_implementer_identity
        ),
    )
    try:
        verify_did_key_signature(
            identity_did=authorization.reviewer_identity,
            payload=review_payload,
            signature=authorization.reviewer_signature,
        )
    except (LocalProfileTampered, ValueError):
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


def _validate_invocation_binding(
    route: object,
    *,
    now_ms: int,
) -> str:
    if not isinstance(route, AgentImplementationRoutePlan):
        return ""
    invocation = route.invocation_binding
    authorization = route.authorization
    bounds = (
        authorization.authority_bounds
        if isinstance(authorization, AgentImplementationRouteAuthorization)
        else None
    )
    if (
        not isinstance(invocation, AgentImplementationInvocationBinding)
        or not isinstance(authorization, AgentImplementationRouteAuthorization)
        or bounds is None
        or invocation.schema != EAAEF_INVOCATION_BINDING_SCHEMA
        or invocation.task_id != EAAEF_BOOTSTRAP_TASK_ID
        or invocation.attempt < 1
        or invocation.profile_lifecycle_generation < 1
        or not _positive_integer(invocation.issued_at_ms)
        or not _positive_integer(invocation.expires_at_ms)
        or invocation.issued_at_ms > now_ms
        or now_ms >= invocation.expires_at_ms
        or invocation.expires_at_ms - invocation.issued_at_ms
        > _MAX_INVOCATION_LIFETIME_MS
        or not invocation.profile_identity_did.startswith("did:key:z")
        or invocation.profile_identity_did != invocation.reviewer_identity
        or invocation.reviewer_identity != authorization.reviewer_identity
        or invocation.reviewer_provider != authorization.reviewer_provider
        or invocation.repository_cid != bounds.repository_cid
        or invocation.baseline_commit != bounds.baseline_commit
        or invocation.effects != bounds.effects
        or invocation.budget_cid != bounds.budget_cid
        or invocation.resource_cid != bounds.resource_cid
        or invocation.authority_cid != bounds.authority_cid
        or invocation.route_id != route.route_id
        or invocation.primary_provider_id != route.primary_provider_id
        or invocation.primary_model_id != route.primary_model_id
        or invocation.fallback_provider_id != route.fallback_provider_id
        or invocation.fallback_model_id != route.fallback_model_id
        or invocation.fallback_reasoning_effort
        != route.fallback_reasoning_effort
        or invocation.fallback_implementer_identity
        != route.fallback_implementer_identity
        or any(
            not _SHA256.fullmatch(value)
            for value in (
                invocation.invocation_id,
                invocation.logical_attempt_id,
                invocation.task_revision_cid,
                invocation.prompt_cid,
                invocation.worktree_id,
                invocation.scope_cid,
                invocation.provider_attempt_store_identity,
            )
        )
        or not invocation.workspace_path.startswith("/")
        or ".." in invocation.workspace_path.split("/")
        or not invocation.reviewer_signature
    ):
        return "eaaef_provider_invocation_binding_missing_or_invalid"
    try:
        verify_did_key_signature(
            identity_did=invocation.reviewer_identity,
            payload=invocation.signed_payload(),
            signature=invocation.reviewer_signature,
        )
    except (LocalProfileTampered, ValueError):
        return "eaaef_provider_invocation_binding_missing_or_invalid"
    return ""


def _image_signed_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("reviewer_signature", None)
    body.pop("qualification_cid", None)
    return body


def eaaef_worker_image_qualification_signing_bytes(
    value: Mapping[str, Any],
) -> bytes:
    """Return canonical bytes for an independently reviewed worker image."""

    return _canonical_bytes(_image_signed_payload(value))


def _validate_image_qualification(
    value: object,
    *,
    trusted_reviewer_dids: frozenset[str],
    now_ms: int,
) -> str:
    if not isinstance(value, Mapping):
        return "oci_image_qualification_missing"
    is_worker = (
        value.get("schema") == EAAEF_WORKER_IMAGE_QUALIFICATION_SCHEMA
    )
    expected_fields = (
        _WORKER_IMAGE_QUALIFICATION_FIELDS
        if is_worker
        else _IMAGE_QUALIFICATION_FIELDS
    )
    reviewer = str(value.get("reviewer_identity_did") or "")
    toolchains = value.get("toolchain_versions")
    workload_class = value.get("workload_class")
    execution_mode = value.get("execution_mode")
    mode_valid = bool(
        (
            execution_mode == "rootless_engine"
            and value.get("rootless_supported") is True
            and value.get("rootless_verified") is True
        )
        or (
            execution_mode == "rootful_daemon_nonroot_worker"
            and value.get("rootless_supported") is False
            and value.get("rootless_verified") is False
        )
    )
    workload_valid = bool(
        (
            not is_worker
            and workload_class == "bootstrap_diagnostic_only"
            and value.get("task_dispatch_verified") is False
        )
        or (
            is_worker
            and workload_class == "agent_worker"
            and value.get("task_dispatch_verified") is True
            and value.get("credential_disposition")
            == "clean_no_credentials"
            and _SHA256.fullmatch(
                str(
                    value.get("credential_disposition_evidence_cid") or ""
                )
            )
            and _SHA256.fullmatch(
                str(value.get("reproducible_build_evidence_cid") or "")
            )
            and isinstance(value.get("reproducible_build_count"), int)
            and not isinstance(value.get("reproducible_build_count"), bool)
            and int(value.get("reproducible_build_count") or 0) >= 2
            and _SHA256.fullmatch(
                str(value.get("network_policy_cid") or "")
            )
        )
    )
    if (
        set(value) != expected_fields
        or value.get("schema")
        not in {
            EAAEF_CONTAINER_IMAGE_QUALIFICATION_SCHEMA,
            EAAEF_WORKER_IMAGE_QUALIFICATION_SCHEMA,
        }
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
        or not workload_valid
        or not mode_valid
        or value.get("nonroot_hardening_verified") is not True
        or not _SHA256.fullmatch(str(value.get("daemon_identity_cid") or ""))
        or not _SHA256.fullmatch(str(value.get("daemon_policy_cid") or ""))
        or reviewer not in trusted_reviewer_dids
        or value.get("reviewer_role") != "independent_security_reviewer"
        or not _positive_integer(value.get("verified_at_ms"))
        or not _positive_integer(value.get("expires_at_ms"))
        or int(value.get("verified_at_ms") or 0) > now_ms
        or now_ms >= int(value.get("expires_at_ms") or 0)
        or (
            is_worker
            and int(value.get("expires_at_ms") or 0)
            - int(value.get("verified_at_ms") or 0)
            > _MAX_QUALIFICATION_LIFETIME_MS
        )
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
        if mount.get("kind") == "worktree" and (
            target != "/workspace" or mount.get("read_only") is not False
        ):
            return False
        if mount.get("kind") == "provider_auth" and (
            target != "/opt/codex-home/auth.json"
            or mount.get("read_only") is not True
        ):
            return False
        if mount.get("kind") == "secret" and (
            not target.startswith("/run/secrets/")
            or mount.get("read_only") is not True
        ):
            return False
        targets.add(target)
        if mount.get("read_only") is False:
            writable_targets.append(target)
    return writable_targets == ["/workspace"]


def _validate_worker_mounts(value: object, *, schema: object) -> bool:
    """Validate one worker schema without reinterpreting its predecessor."""

    if schema == EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA:
        return _validate_mounts(value)
    if (
        schema != EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA_V2
        or not isinstance(value, list)
        or not value
    ):
        return False
    required = {
        "worktree": ("/workspace", False),
        "provider_auth": ("/opt/codex-home/auth.json", True),
        "grok_prompt": ("/run/eaaef/grok/prompt.txt", True),
        "grok_policy": ("/opt/codex-home/sandbox.toml", True),
        "grok_provider_home": ("/opt/codex-home", False),
    }
    observed: dict[str, tuple[str, bool]] = {}
    targets: set[str] = set()
    writable_targets: set[str] = set()
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
        kind = str(mount.get("kind") or "")
        read_only = mount.get("read_only")
        lowered = f"{source} {target}".lower()
        if (
            not _SHA256.fullmatch(source)
            or not target.startswith("/")
            or ".." in target.split("/")
            or target in targets
            or not isinstance(read_only, bool)
            or "docker.sock" in lowered
            or "/run/docker" in lowered
            or "/var/run/docker" in lowered
        ):
            return False
        expected = required.get(kind)
        if kind == "secret":
            if not target.startswith("/run/secrets/") or read_only is not True:
                return False
        elif expected is None or (target, read_only) != expected:
            return False
        elif kind in observed:
            return False
        else:
            observed[kind] = (target, read_only)
        targets.add(target)
        if read_only is False:
            writable_targets.add(target)
    return (
        all(observed.get(kind) == expected for kind, expected in required.items())
        and writable_targets == {"/workspace", "/opt/codex-home"}
    )


def _validate_diagnostic_container_profile(
    value: object,
    *,
    image_qualification: Mapping[str, Any] | None,
) -> str:
    if not isinstance(value, Mapping):
        return "container_profile_missing"
    gpu = value.get("gpu")
    user = str(value.get("nonroot_user") or "")
    image_digest = str(value.get("image_digest") or "")
    execution_mode = value.get("execution_mode")
    workload_class = value.get("workload_class")
    mounts = value.get("mounts")
    maximum_parallel_workers = value.get("maximum_parallel_workers")
    maximum_parallel_containers = value.get("maximum_parallel_containers")
    image_mode = (
        image_qualification.get("execution_mode")
        if isinstance(image_qualification, Mapping)
        else None
    )
    rootless_mode_valid = bool(
        execution_mode == "rootless_engine"
        and value.get("rootless") is True
        and value.get("rootless_supported") is True
        and value.get("rootful_fallback_admitted") is False
    )
    rootful_mode_valid = bool(
        execution_mode == "rootful_daemon_nonroot_worker"
        and value.get("rootless") is False
        and value.get("rootless_supported") is False
        and value.get("rootful_fallback_admitted") is True
        and value.get("bootstrap_policy_cid")
        in _ADMITTED_ROOTFUL_FALLBACK_POLICY_CIDS
    )
    diagnostic_profile_valid = bool(
        workload_class == "bootstrap_diagnostic_only"
        and value.get("task_dispatch_admitted") is False
        and value.get("network_mode") == "none"
        and maximum_parallel_workers == 0
        and _positive_integer(maximum_parallel_containers)
        and isinstance(mounts, list)
        and all(
            isinstance(mount, Mapping) and mount.get("kind") == "worktree"
            for mount in mounts
        )
    )
    if (
        set(value) != _CONTAINER_PROFILE_FIELDS
        or value.get("schema") != EAAEF_CONTAINER_PROFILE_SCHEMA
        or value.get("runtime") not in {"docker", "oci"}
        or execution_mode != image_mode
        or not (rootless_mode_valid or rootful_mode_valid)
        or not diagnostic_profile_valid
        or workload_class
        != (
            image_qualification.get("workload_class")
            if isinstance(image_qualification, Mapping)
            else None
        )
        or value.get("task_dispatch_admitted")
        != (
            image_qualification.get("task_dispatch_verified")
            if isinstance(image_qualification, Mapping)
            else None
        )
        or value.get("rootless_supported")
        != (
            image_qualification.get("rootless_supported")
            if isinstance(image_qualification, Mapping)
            else None
        )
        or value.get("daemon_identity_cid")
        != (
            image_qualification.get("daemon_identity_cid")
            if isinstance(image_qualification, Mapping)
            else None
        )
        or value.get("daemon_policy_cid")
        != (
            image_qualification.get("daemon_policy_cid")
            if isinstance(image_qualification, Mapping)
            else None
        )
        or not _SHA256.fullmatch(str(value.get("daemon_identity_cid") or ""))
        or not _SHA256.fullmatch(str(value.get("daemon_policy_cid") or ""))
        or value.get("bootstrap_policy_cid") != EAAEF_BOOTSTRAP_POLICY_CID
        or not _SHA256.fullmatch(image_digest)
        or not isinstance(image_qualification, Mapping)
        or image_digest != image_qualification.get("image_digest")
        or re.fullmatch(r"[1-9][0-9]*:[1-9][0-9]*", user) is None
        or value.get("read_only_base") is not True
        or value.get("cap_drop") != ["ALL"]
        or value.get("no_new_privileges") is not True
        or not _positive_integer(value.get("pids_limit"))
        or int(value.get("pids_limit") or 0) > _MAX_PIDS
        or not _positive_number(value.get("cpu_limit"))
        or float(value.get("cpu_limit") or 0) > _MAX_CPU
        or not _positive_integer(value.get("memory_limit_bytes"))
        or int(value.get("memory_limit_bytes") or 0) > _MAX_MEMORY_BYTES
        or not _positive_integer(value.get("disk_limit_bytes"))
        or int(value.get("disk_limit_bytes") or 0) > _MAX_DISK_BYTES
        or not _nonnegative_integer(maximum_parallel_workers)
        or int(maximum_parallel_workers or 0) > _MAX_PARALLEL_WORKERS
        or not _positive_integer(maximum_parallel_containers)
        or int(maximum_parallel_containers or 0) > _MAX_PARALLEL_CONTAINERS
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
        or not _validate_mounts(mounts)
        or not _self_addressed(value, "profile_cid")
    ):
        return "container_profile_invalid"
    return ""


def _worker_profile_signed_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("reviewer_signature", None)
    body.pop("profile_cid", None)
    return body


def eaaef_worker_container_profile_signing_bytes(
    value: Mapping[str, Any],
) -> bytes:
    """Return canonical bytes for an independently reviewed worker profile."""

    return _canonical_bytes(_worker_profile_signed_payload(value))


def _validate_worker_container_profile(
    value: Mapping[str, Any],
    *,
    image_qualification: Mapping[str, Any] | None,
    trusted_reviewer_dids: frozenset[str],
    expected_worker_principal_did: str,
    expected_provider_principal_did: str,
    now_ms: int,
) -> str:
    image = image_qualification or {}
    reviewer = str(value.get("reviewer_identity_did") or "")
    image_reviewer = str(image.get("reviewer_identity_did") or "")
    worker_principal = str(value.get("worker_principal_did") or "")
    provider_principal = str(value.get("provider_principal_did") or "")
    reviewed_at_ms = value.get("reviewed_at_ms")
    expires_at_ms = value.get("expires_at_ms")
    expected_reviewer_role = (
        EAAEF_WORKER_CONTAINER_PROFILE_REVIEWER_ROLE_V2
        if value.get("schema") == EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA_V2
        else "independent_container_security_reviewer"
    )
    resource_body = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "external-agent-worker-resource-profile@1"
        ),
        "pids_limit": value.get("pids_limit"),
        "cpu_limit": value.get("cpu_limit"),
        "memory_limit_bytes": value.get("memory_limit_bytes"),
        "disk_limit_bytes": value.get("disk_limit_bytes"),
        "maximum_parallel_workers": value.get("maximum_parallel_workers"),
        "maximum_parallel_containers": value.get(
            "maximum_parallel_containers"
        ),
        "gpu": value.get("gpu"),
    }
    # Reuse all invariant runtime hardening checks, but project the worker
    # contract into the deliberately narrower diagnostic field set only for
    # those common invariants.  Workload and dispatch are checked below.
    common = {
        key: item
        for key, item in value.items()
        if key in _CONTAINER_PROFILE_FIELDS
    }
    common["schema"] = EAAEF_CONTAINER_PROFILE_SCHEMA
    common["workload_class"] = "bootstrap_diagnostic_only"
    common["task_dispatch_admitted"] = False
    common["network_mode"] = "none"
    common["maximum_parallel_workers"] = 0
    common["maximum_parallel_containers"] = 1
    # The bootstrap diagnostic contract deliberately permits only its
    # worktree mount.  A dispatch-admitted worker profile may additionally
    # carry the separately validated read-only provider-auth/secret mounts;
    # omit those only from this narrowed common-invariant projection.
    common["mounts"] = [
        dict(mount)
        for mount in value.get("mounts", [])
        if isinstance(mount, Mapping) and mount.get("kind") == "worktree"
    ]
    common["profile_cid"] = _cid(
        {key: item for key, item in common.items() if key != "profile_cid"}
    )
    common_image = dict(image)
    common_image["workload_class"] = "bootstrap_diagnostic_only"
    common_image["task_dispatch_verified"] = False
    common_image["execution_mode"] = value.get("execution_mode")
    common_image["rootless_supported"] = value.get("rootless_supported")
    common_image["daemon_identity_cid"] = value.get("daemon_identity_cid")
    common_image["daemon_policy_cid"] = value.get("daemon_policy_cid")
    if _validate_diagnostic_container_profile(
        common,
        image_qualification=common_image,
    ):
        return "container_profile_invalid"
    if (
        set(value) != _WORKER_CONTAINER_PROFILE_FIELDS
        or value.get("schema")
        not in {
            EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA,
            EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA_V2,
        }
        or image.get("schema") != EAAEF_WORKER_IMAGE_QUALIFICATION_SCHEMA
        or value.get("workload_class") != "agent_worker"
        or value.get("task_dispatch_admitted") is not True
        or value.get("network_mode") != "policy_proxy_only"
        or value.get("network_policy_cid")
        != image.get("network_policy_cid")
        or value.get("image_qualification_cid")
        != image.get("qualification_cid")
        or value.get("image_digest") != image.get("image_digest")
        or value.get("sbom_digest") != image.get("sbom_digest")
        or value.get("toolchain_versions") != image.get("toolchain_versions")
        or value.get("resource_profile_cid") != _cid(resource_body)
        or not _positive_integer(value.get("maximum_parallel_workers"))
        or not _positive_integer(value.get("maximum_parallel_containers"))
        or int(value.get("maximum_parallel_workers") or 0)
        > _MAX_PARALLEL_WORKERS
        or int(value.get("maximum_parallel_containers") or 0)
        > _MAX_PARALLEL_CONTAINERS
        or int(value.get("maximum_parallel_workers") or 0)
        > int(value.get("maximum_parallel_containers") or 0)
        or not _validate_worker_mounts(
            value.get("mounts"),
            schema=value.get("schema"),
        )
        or worker_principal != expected_worker_principal_did
        or provider_principal != expected_provider_principal_did
        or not worker_principal.startswith("did:key:z")
        or not provider_principal.startswith("did:key:z")
        or worker_principal == provider_principal
        or reviewer not in trusted_reviewer_dids
        or reviewer in {image_reviewer, worker_principal, provider_principal}
        or value.get("reviewer_role") != expected_reviewer_role
        or not _positive_integer(reviewed_at_ms)
        or not _positive_integer(expires_at_ms)
        or int(reviewed_at_ms or 0) > now_ms
        or now_ms >= int(expires_at_ms or 0)
        or int(expires_at_ms or 0) - int(reviewed_at_ms or 0)
        > _MAX_QUALIFICATION_LIFETIME_MS
        or not _self_addressed(value, "profile_cid")
    ):
        return "container_profile_invalid"
    signature = value.get("reviewer_signature")
    try:
        if not isinstance(signature, str) or not signature:
            raise ValueError("signature missing")
        verify_did_key_signature(
            identity_did=reviewer,
            payload=_worker_profile_signed_payload(value),
            signature=signature,
        )
    except (LocalProfileTampered, ValueError):
        return "container_profile_invalid"
    return ""


def _validate_container_profile(
    value: object,
    *,
    image_qualification: Mapping[str, Any] | None,
    trusted_reviewer_dids: frozenset[str] = frozenset(),
    expected_worker_principal_did: str = "",
    expected_provider_principal_did: str = "",
    now_ms: int = 0,
) -> str:
    if (
        isinstance(value, Mapping)
        and value.get("schema")
        in {
            EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA,
            EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA_V2,
        }
    ):
        return _validate_worker_container_profile(
            value,
            image_qualification=image_qualification,
            trusted_reviewer_dids=trusted_reviewer_dids,
            expected_worker_principal_did=expected_worker_principal_did,
            expected_provider_principal_did=expected_provider_principal_did,
            now_ms=now_ms,
        )
    return _validate_diagnostic_container_profile(
        value,
        image_qualification=image_qualification,
    )


def validate_eaaef_worker_container_profile_artifact(
    value: object,
    *,
    expected_profile_cid: str,
    expected_image_digest: str,
    expected_worker_principal_did: str,
    expected_provider_principal_did: str,
    now_ms: int,
) -> str:
    """Revalidate a launch-time worker profile against admitted identities.

    The bootstrap verifier validates the full image/profile/qualification
    chain.  At launch time the configured-board capsule carries the resulting
    exact profile and image CIDs.  This narrower verifier intentionally does
    not recreate that admission ceremony: it verifies the complete profile's
    self-address, independent signature, hardening semantics, freshness and
    exact capsule-bound identities.  The frozen profile CID is the trust
    anchor; accepting the reviewer DID merely from an otherwise unbound
    artifact would not be sufficient.
    """

    if not isinstance(value, Mapping):
        return "container_profile_invalid"
    profile = dict(value)
    reviewer = str(profile.get("reviewer_identity_did") or "")
    image_reviewer = "did:key:zaccepted-image-qualification-reviewer"
    if reviewer == image_reviewer:
        image_reviewer += "-independent"
    image_projection = {
        "schema": EAAEF_WORKER_IMAGE_QUALIFICATION_SCHEMA,
        "qualification_cid": profile.get("image_qualification_cid"),
        "image_digest": profile.get("image_digest"),
        "sbom_digest": profile.get("sbom_digest"),
        "toolchain_versions": profile.get("toolchain_versions"),
        "network_policy_cid": profile.get("network_policy_cid"),
        "reviewer_identity_did": image_reviewer,
    }
    if (
        not expected_profile_cid
        or profile.get("profile_cid") != expected_profile_cid
        or profile.get("image_digest") != expected_image_digest
    ):
        return "container_profile_invalid"
    return _validate_worker_container_profile(
        profile,
        image_qualification=image_projection,
        trusted_reviewer_dids=frozenset({reviewer}),
        expected_worker_principal_did=expected_worker_principal_did,
        expected_provider_principal_did=expected_provider_principal_did,
        now_ms=now_ms,
    )


def _bootstrap_task(board: Mapping[str, Any]) -> Mapping[str, Any] | None:
    tasks = board.get("tasks")
    if not isinstance(tasks, list):
        return None
    matches = [
        task
        for task in tasks
        if isinstance(task, Mapping)
        and task.get("stable_task_id") == EAAEF_BOOTSTRAP_TASK_ID
    ]
    return matches[0] if len(matches) == 1 else None


def _projection_root(
    materialization_receipt: Mapping[str, Any],
    field: str,
) -> str:
    projection = materialization_receipt.get(field)
    if not isinstance(projection, Mapping):
        return ""
    identity = str(projection.get("projection_root") or "")
    return identity if _SHA256.fullmatch(identity) else ""


def eaaef_repository_binding_cid(
    *,
    board: Mapping[str, Any],
    materialization_receipt: Mapping[str, Any],
) -> str:
    """Bind provider authority to the exact materialized repository state."""

    source_generation = materialization_receipt.get("source_generation")
    source_generation_cid = (
        str(source_generation.get("source_generation_cid") or "")
        if isinstance(source_generation, Mapping)
        else ""
    )
    return _cid(
        {
            "schema": EAAEF_REPOSITORY_BINDING_SCHEMA,
            "board_cid": str(board.get("board_cid") or ""),
            "source_head": str(
                materialization_receipt.get("source_head") or ""
            ),
            "source_tree": str(
                materialization_receipt.get("source_tree") or ""
            ),
            "source_generation_cid": source_generation_cid,
            "materialization_receipt_cid": str(
                materialization_receipt.get("receipt_cid") or ""
            ),
        }
    )


def eaaef_provider_budget_binding_cid(
    *,
    board: Mapping[str, Any],
) -> str:
    """Bind the route budget to the one board-declared bootstrap task."""

    task = _bootstrap_task(board)
    if task is None:
        return ""
    resource_request = task.get("resource_request")
    if (
        not isinstance(resource_request, Mapping)
        or not _SHA256.fullmatch(str(task.get("task_spec_cid") or ""))
        or not _SHA256.fullmatch(str(task.get("idempotency_key") or ""))
        or not str(task.get("provider_policy") or "")
        or not str(task.get("model_route") or "")
    ):
        return ""
    return _cid(
        {
            "schema": EAAEF_PROVIDER_BUDGET_BINDING_SCHEMA,
            "board_cid": str(board.get("board_cid") or ""),
            "task_id": EAAEF_BOOTSTRAP_TASK_ID,
            "task_spec_cid": str(task.get("task_spec_cid")),
            "idempotency_key": str(task.get("idempotency_key")),
            "provider_policy": str(task.get("provider_policy")),
            "model_route": str(task.get("model_route")),
            "resource_request": dict(resource_request),
        }
    )


def _provider_container_qualification_body(
    *,
    board: Mapping[str, Any],
    materialization_receipt: Mapping[str, Any],
    route_plan: AgentImplementationRoutePlan,
    image_qualification: Mapping[str, Any],
    container_profile: Mapping[str, Any],
    worker_principal_did: str,
    provider_principal_did: str,
    admitted_at_ms: int,
    expires_at_ms: int,
    signer_identity_did: str,
) -> dict[str, Any]:
    authorization = route_plan.authorization
    if not isinstance(authorization, AgentImplementationRouteAuthorization):
        raise ValueError("EAAEF route authorization is unavailable")
    bounds = authorization.authority_bounds
    if bounds is None:
        raise ValueError("EAAEF route authority bounds are unavailable")
    invocation = route_plan.invocation_binding
    if not isinstance(invocation, AgentImplementationInvocationBinding):
        raise ValueError("EAAEF provider invocation binding is unavailable")
    source_generation = materialization_receipt.get("source_generation")
    task = _bootstrap_task(board)
    if not isinstance(source_generation, Mapping) or task is None:
        raise ValueError("EAAEF source or task binding is unavailable")
    return {
        "schema": EAAEF_PROVIDER_CONTAINER_QUALIFICATION_SCHEMA,
        "board_cid": str(board.get("board_cid") or ""),
        "source_head": str(materialization_receipt.get("source_head") or ""),
        "source_tree": str(materialization_receipt.get("source_tree") or ""),
        "source_generation_cid": str(
            source_generation.get("source_generation_cid") or ""
        ),
        "materialization_receipt_cid": str(
            materialization_receipt.get("receipt_cid") or ""
        ),
        "control_projection_root": _projection_root(
            materialization_receipt, "control_projection"
        ),
        "coordination_projection_root": _projection_root(
            materialization_receipt, "coordination_projection"
        ),
        "execution_projection_root": _projection_root(
            materialization_receipt, "execution_projection"
        ),
        "control_plane_schema_version": str(
            task.get("source_control_plane_schema_version") or ""
        ),
        "route_id": route_plan.route_id,
        "route_authorization_id": authorization.authorization_id,
        "route_authorization_sha256": authorization.artifact_sha256,
        "route_repository_cid": bounds.repository_cid,
        "route_baseline_commit": bounds.baseline_commit,
        "route_effects": list(bounds.effects),
        "route_budget_cid": bounds.budget_cid,
        "route_resource_cid": bounds.resource_cid,
        "image_qualification_cid": str(
            image_qualification.get("qualification_cid") or ""
        ),
        "image_digest": str(image_qualification.get("image_digest") or ""),
        "sbom_digest": str(image_qualification.get("sbom_digest") or ""),
        "container_profile_cid": str(
            container_profile.get("profile_cid") or ""
        ),
        "execution_mode": str(container_profile.get("execution_mode") or ""),
        "daemon_identity_cid": str(
            container_profile.get("daemon_identity_cid") or ""
        ),
        "daemon_policy_cid": str(
            container_profile.get("daemon_policy_cid") or ""
        ),
        "workload_class": str(container_profile.get("workload_class") or ""),
        "task_dispatch_admitted": container_profile.get(
            "task_dispatch_admitted"
        ),
        "worker_principal_did": worker_principal_did,
        "provider_principal_did": provider_principal_did,
        "maximum_parallel_workers": container_profile.get(
            "maximum_parallel_workers"
        ),
        "maximum_parallel_containers": container_profile.get(
            "maximum_parallel_containers"
        ),
        "admitted_at_ms": admitted_at_ms,
        "expires_at_ms": expires_at_ms,
        "signer_identity_did": signer_identity_did,
        "signer_role": EAAEF_QUALIFICATION_SIGNER_ROLE,
    }


def eaaef_provider_container_qualification_signing_bytes(
    value: Mapping[str, Any],
) -> bytes:
    """Return the exact bytes an external qualification principal signs."""

    body = dict(value)
    body.pop("signer_signature", None)
    body.pop("receipt_cid", None)
    return _canonical_bytes(body)


def prepare_eaaef_provider_container_qualification(
    *,
    board: Mapping[str, Any],
    materialization_receipt: Mapping[str, Any],
    route_plan: AgentImplementationRoutePlan,
    image_qualification: Mapping[str, Any],
    container_profile: Mapping[str, Any],
    worker_principal_did: str,
    provider_principal_did: str,
    signer_identity_did: str,
    admitted_at_ms: int,
    expires_at_ms: int,
    now_ms: int,
    trusted_image_reviewer_dids: Sequence[str],
    trusted_container_profile_reviewer_dids: Sequence[str] = (),
) -> dict[str, Any]:
    """Prepare, but never sign, a closed provider/container review payload."""

    reasons = [
        reason
        for reason in (
            _validate_board(board),
            _validate_materialization(materialization_receipt, board=board),
            _validate_route(route_plan),
            _validate_invocation_binding(route_plan, now_ms=now_ms),
            _validate_image_qualification(
                image_qualification,
                trusted_reviewer_dids=frozenset(trusted_image_reviewer_dids),
                now_ms=now_ms,
            ),
            _validate_container_profile(
                container_profile,
                image_qualification=image_qualification,
                trusted_reviewer_dids=frozenset(
                    trusted_container_profile_reviewer_dids
                ),
                expected_worker_principal_did=worker_principal_did,
                expected_provider_principal_did=provider_principal_did,
                now_ms=now_ms,
            ),
        )
        if reason
    ]
    authorization = route_plan.authorization
    bounds = (
        authorization.authority_bounds
        if isinstance(authorization, AgentImplementationRouteAuthorization)
        else None
    )
    expected_repository_cid = eaaef_repository_binding_cid(
        board=board,
        materialization_receipt=materialization_receipt,
    )
    expected_budget_cid = eaaef_provider_budget_binding_cid(board=board)
    task = _bootstrap_task(board)
    image_reviewer = str(image_qualification.get("reviewer_identity_did") or "")
    profile_reviewer = str(container_profile.get("reviewer_identity_did") or "")
    invocation = route_plan.invocation_binding
    lifecycle_reviewer = (
        invocation.reviewer_identity
        if isinstance(invocation, AgentImplementationInvocationBinding)
        else ""
    )
    if (
        not _positive_integer(now_ms)
        or not _positive_integer(admitted_at_ms)
        or not _positive_integer(expires_at_ms)
        or admitted_at_ms > now_ms
        or now_ms >= expires_at_ms
        or expires_at_ms - admitted_at_ms > _MAX_QUALIFICATION_LIFETIME_MS
    ):
        reasons.append("provider_container_qualification_time_invalid")
    if (
        not signer_identity_did.startswith("did:key:z")
        or signer_identity_did == image_reviewer
        or signer_identity_did == profile_reviewer
        or signer_identity_did in {worker_principal_did, provider_principal_did}
        or (
            isinstance(authorization, AgentImplementationRouteAuthorization)
            and (
                image_reviewer == authorization.reviewer_identity
                or profile_reviewer == authorization.reviewer_identity
                or signer_identity_did == authorization.reviewer_identity
            )
        )
    ):
        reasons.append("provider_container_reviewer_not_independent")
    if (
        not worker_principal_did.startswith("did:key:z")
        or not provider_principal_did.startswith("did:key:z")
        or worker_principal_did == provider_principal_did
        or lifecycle_reviewer in {worker_principal_did, provider_principal_did}
        or image_reviewer in {worker_principal_did, provider_principal_did}
    ):
        reasons.append("provider_container_service_principals_invalid")
    if (
        bounds is None
        or task is None
        or not expected_budget_cid
        or not all(
            _projection_root(materialization_receipt, field)
            for field in (
                "control_projection",
                "coordination_projection",
                "execution_projection",
            )
        )
        or bounds.repository_cid != expected_repository_cid
        or bounds.baseline_commit
        != materialization_receipt.get("source_head")
        or authorization is None
        or authorization.source_head
        != materialization_receipt.get("source_head")
        or authorization.source_tree
        != materialization_receipt.get("source_tree")
        or bounds.effects != EAAEF_REQUIRED_ROUTE_EFFECTS
        or bounds.budget_cid != expected_budget_cid
        or bounds.resource_cid != container_profile.get("profile_cid")
        or not str(task.get("source_control_plane_schema_version") or "")
    ):
        reasons.append("provider_container_binding_mismatch")
    if reasons:
        raise ValueError(
            "provider/container qualification cannot be prepared: "
            + ", ".join(dict.fromkeys(reasons))
        )
    return _provider_container_qualification_body(
        board=board,
        materialization_receipt=materialization_receipt,
        route_plan=route_plan,
        image_qualification=image_qualification,
        container_profile=container_profile,
        worker_principal_did=worker_principal_did,
        provider_principal_did=provider_principal_did,
        admitted_at_ms=admitted_at_ms,
        expires_at_ms=expires_at_ms,
        signer_identity_did=signer_identity_did,
    )


def seal_eaaef_provider_container_qualification(
    *,
    prepared_payload: Mapping[str, Any],
    signer_signature: str,
) -> dict[str, Any]:
    """Attach an externally produced signature; no key or authority is minted."""

    expected = _PROVIDER_CONTAINER_QUALIFICATION_FIELDS - {
        "signer_signature",
        "receipt_cid",
    }
    if set(prepared_payload) != expected or not signer_signature:
        raise ValueError("provider/container prepared payload is invalid")
    sealed = {**dict(prepared_payload), "signer_signature": signer_signature}
    sealed["receipt_cid"] = _cid(sealed)
    return sealed


@dataclass(frozen=True, slots=True)
class EAAEFProviderContainerQualificationVerification:
    """Portable verification result; it is evidence, never launch authority."""

    valid: bool
    blockers: tuple[str, ...]
    qualification_cid: str
    board_cid: str
    materialization_receipt_cid: str
    route_id: str
    image_qualification_cid: str
    image_digest: str
    container_profile_cid: str
    workload_class: str
    task_dispatch_admitted: bool
    worker_principal_did: str
    provider_principal_did: str
    maximum_parallel_workers: int
    maximum_parallel_containers: int

    @property
    def authority_mutated(self) -> bool:
        return False

    @property
    def process_started(self) -> bool:
        return False

    @property
    def verifier_cid(self) -> str:
        return str(self.as_dict()["verifier_cid"])

    def as_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema": EAAEF_PROVIDER_CONTAINER_VERIFICATION_SCHEMA,
            "valid": self.valid,
            "blockers": list(self.blockers),
            "qualification_cid": self.qualification_cid,
            "board_cid": self.board_cid,
            "materialization_receipt_cid": self.materialization_receipt_cid,
            "route_id": self.route_id,
            "image_qualification_cid": self.image_qualification_cid,
            "image_digest": self.image_digest,
            "container_profile_cid": self.container_profile_cid,
            "workload_class": self.workload_class,
            "task_dispatch_admitted": self.task_dispatch_admitted,
            "worker_principal_did": self.worker_principal_did,
            "provider_principal_did": self.provider_principal_did,
            "maximum_parallel_workers": self.maximum_parallel_workers,
            "maximum_parallel_containers": self.maximum_parallel_containers,
            "authority_mutated": False,
            "process_started": False,
        }
        body["verifier_cid"] = _cid(body)
        return body


def verify_eaaef_provider_container_qualification(
    *,
    qualification: object,
    board: Mapping[str, Any],
    materialization_receipt: Mapping[str, Any],
    route_plan: AgentImplementationRoutePlan | None,
    image_qualification: object,
    container_profile: object,
    trusted_qualification_signer_dids: Sequence[str],
    trusted_image_reviewer_dids: Sequence[str],
    trusted_container_profile_reviewer_dids: Sequence[str] = (),
    expected_worker_principal_did: str,
    expected_provider_principal_did: str,
    now_ms: int,
) -> EAAEFProviderContainerQualificationVerification:
    """Verify every constituent and exact cross-binding without side effects."""

    blockers: list[str] = []
    image = image_qualification if isinstance(image_qualification, Mapping) else {}
    profile = container_profile if isinstance(container_profile, Mapping) else {}
    route: object = route_plan
    for reason in (
        _validate_board(board),
        _validate_materialization(materialization_receipt, board=board),
        _validate_route(route),
        _validate_invocation_binding(
            route,
            now_ms=now_ms if _positive_integer(now_ms) else 0,
        ),
        _validate_image_qualification(
            image_qualification,
            trusted_reviewer_dids=frozenset(trusted_image_reviewer_dids),
            now_ms=now_ms if _positive_integer(now_ms) else 0,
        ),
        _validate_container_profile(
            container_profile,
            image_qualification=image if image else None,
            trusted_reviewer_dids=frozenset(
                trusted_container_profile_reviewer_dids
            ),
            expected_worker_principal_did=expected_worker_principal_did,
            expected_provider_principal_did=expected_provider_principal_did,
            now_ms=now_ms if _positive_integer(now_ms) else 0,
        ),
    ):
        if reason:
            blockers.append(reason)
    if not _positive_integer(now_ms):
        blockers.append("provider_container_qualification_time_invalid")

    value = qualification if isinstance(qualification, Mapping) else {}
    if not value:
        blockers.append("provider_container_qualification_missing")
    elif (
        set(value) != _PROVIDER_CONTAINER_QUALIFICATION_FIELDS
        or value.get("schema") != EAAEF_PROVIDER_CONTAINER_QUALIFICATION_SCHEMA
        or not _self_addressed(value, "receipt_cid")
    ):
        blockers.append("provider_container_qualification_invalid")
    else:
        admitted_at_ms = value.get("admitted_at_ms")
        expires_at_ms = value.get("expires_at_ms")
        signer = str(value.get("signer_identity_did") or "")
        image_reviewer = str(image.get("reviewer_identity_did") or "")
        profile_reviewer = str(profile.get("reviewer_identity_did") or "")
        safe_now_ms = now_ms if _positive_integer(now_ms) else 0
        if (
            not _positive_integer(admitted_at_ms)
            or not _positive_integer(expires_at_ms)
            or int(admitted_at_ms) > safe_now_ms
            or safe_now_ms >= int(expires_at_ms)
            or int(expires_at_ms) - int(admitted_at_ms)
            > _MAX_QUALIFICATION_LIFETIME_MS
        ):
            blockers.append("provider_container_qualification_time_invalid")
        authorization = (
            route_plan.authorization
            if isinstance(route_plan, AgentImplementationRoutePlan)
            else None
        )
        invocation = (
            route_plan.invocation_binding
            if isinstance(route_plan, AgentImplementationRoutePlan)
            else None
        )
        lifecycle_reviewer = (
            invocation.reviewer_identity
            if isinstance(invocation, AgentImplementationInvocationBinding)
            else ""
        )
        worker_principal = str(value.get("worker_principal_did") or "")
        provider_principal = str(value.get("provider_principal_did") or "")
        if (
            signer not in frozenset(trusted_qualification_signer_dids)
            or value.get("signer_role") != EAAEF_QUALIFICATION_SIGNER_ROLE
        ):
            blockers.append("provider_container_signer_not_trusted")
        if (
            not signer.startswith("did:key:z")
            or signer == image_reviewer
            or signer == profile_reviewer
            or signer in {worker_principal, provider_principal}
            or (
                isinstance(
                    authorization, AgentImplementationRouteAuthorization
                )
                and (
                    image_reviewer == authorization.reviewer_identity
                    or profile_reviewer == authorization.reviewer_identity
                    or signer == authorization.reviewer_identity
                )
            )
        ):
            blockers.append("provider_container_reviewer_not_independent")
        if (
            not expected_worker_principal_did.startswith("did:key:z")
            or not expected_provider_principal_did.startswith("did:key:z")
            or worker_principal != expected_worker_principal_did
            or provider_principal != expected_provider_principal_did
            or worker_principal == provider_principal
            or lifecycle_reviewer in {worker_principal, provider_principal}
            or image_reviewer in {worker_principal, provider_principal}
        ):
            blockers.append("provider_container_service_principals_invalid")
        signature = value.get("signer_signature")
        try:
            if not isinstance(signature, str) or not signature:
                raise ValueError("signature missing")
            verify_did_key_signature(
                identity_did=signer,
                payload={
                    key: item
                    for key, item in value.items()
                    if key not in {"signer_signature", "receipt_cid"}
                },
                signature=signature,
            )
        except (LocalProfileTampered, ValueError):
            blockers.append("provider_container_signature_invalid")

        if (
            isinstance(route_plan, AgentImplementationRoutePlan)
            and isinstance(image_qualification, Mapping)
            and isinstance(container_profile, Mapping)
        ):
            try:
                expected = _provider_container_qualification_body(
                    board=board,
                    materialization_receipt=materialization_receipt,
                    route_plan=route_plan,
                    image_qualification=image_qualification,
                    container_profile=container_profile,
                    worker_principal_did=worker_principal,
                    provider_principal_did=provider_principal,
                    admitted_at_ms=int(admitted_at_ms),
                    expires_at_ms=int(expires_at_ms),
                    signer_identity_did=signer,
                )
            except (TypeError, ValueError):
                expected = {}
            observed = {
                key: item
                for key, item in value.items()
                if key not in {"signer_signature", "receipt_cid"}
            }
            authorization = route_plan.authorization
            bounds = (
                authorization.authority_bounds
                if isinstance(
                    authorization, AgentImplementationRouteAuthorization
                )
                else None
            )
            if (
                not expected
                or observed != expected
                or bounds is None
                or bounds.repository_cid
                != eaaef_repository_binding_cid(
                    board=board,
                    materialization_receipt=materialization_receipt,
                )
                or bounds.baseline_commit
                != materialization_receipt.get("source_head")
                or authorization.source_head
                != materialization_receipt.get("source_head")
                or authorization.source_tree
                != materialization_receipt.get("source_tree")
                or bounds.effects != EAAEF_REQUIRED_ROUTE_EFFECTS
                or bounds.budget_cid
                != eaaef_provider_budget_binding_cid(board=board)
                or bounds.resource_cid != profile.get("profile_cid")
            ):
                blockers.append("provider_container_binding_mismatch")
        else:
            blockers.append("provider_container_binding_mismatch")

    unique_blockers = tuple(dict.fromkeys(blockers))
    return EAAEFProviderContainerQualificationVerification(
        valid=not unique_blockers,
        blockers=unique_blockers,
        qualification_cid=str(value.get("receipt_cid") or ""),
        board_cid=str(board.get("board_cid") or ""),
        materialization_receipt_cid=str(
            materialization_receipt.get("receipt_cid") or ""
        ),
        route_id=(
            route_plan.route_id
            if isinstance(route_plan, AgentImplementationRoutePlan)
            else ""
        ),
        image_qualification_cid=str(image.get("qualification_cid") or ""),
        image_digest=str(image.get("image_digest") or ""),
        container_profile_cid=str(profile.get("profile_cid") or ""),
        workload_class=str(profile.get("workload_class") or ""),
        task_dispatch_admitted=(
            profile.get("task_dispatch_admitted") is True
        ),
        worker_principal_did=str(value.get("worker_principal_did") or ""),
        provider_principal_did=str(value.get("provider_principal_did") or ""),
        maximum_parallel_workers=(
            int(profile.get("maximum_parallel_workers"))
            if _nonnegative_integer(profile.get("maximum_parallel_workers"))
            and int(profile.get("maximum_parallel_workers"))
            <= _MAX_PARALLEL_WORKERS
            else 0
        ),
        maximum_parallel_containers=(
            int(profile.get("maximum_parallel_containers"))
            if _positive_integer(profile.get("maximum_parallel_containers"))
            and int(profile.get("maximum_parallel_containers"))
            <= _MAX_PARALLEL_CONTAINERS
            else 0
        ),
    )


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
    provider_container_qualification_cid: str
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
            "provider_container_qualification_cid": (
                self.provider_container_qualification_cid
            ),
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
    trusted_container_profile_reviewer_dids: Sequence[str] = (),
    now_ms: int,
    expected_worker_principal_did: str = "",
    expected_provider_principal_did: str = "",
    provider_container_qualification: object = None,
    trusted_qualification_signer_dids: Sequence[str] = (),
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
    verification = verify_eaaef_provider_container_qualification(
        qualification=provider_container_qualification,
        board=board_mapping,
        materialization_receipt=receipt_mapping,
        route_plan=(
            route_plan
            if isinstance(route_plan, AgentImplementationRoutePlan)
            else None
        ),
        image_qualification=image_qualification,
        container_profile=container_profile,
        trusted_qualification_signer_dids=(
            trusted_qualification_signer_dids
        ),
        trusted_image_reviewer_dids=trusted_image_reviewer_dids,
        trusted_container_profile_reviewer_dids=(
            trusted_container_profile_reviewer_dids
        ),
        expected_worker_principal_did=expected_worker_principal_did,
        expected_provider_principal_did=expected_provider_principal_did,
        now_ms=now_ms,
    )
    blockers.extend(verification.blockers)
    if (
        verification.workload_class != "agent_worker"
        or verification.task_dispatch_admitted is not True
        or verification.maximum_parallel_workers < 1
    ):
        blockers.append("provider_task_dispatch_not_admitted")

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
        provider_container_qualification_cid=(
            verification.qualification_cid
        ),
        launch_mode=launch_mode,
    )


__all__ = (
    "EAAEF_BOARD_NAMESPACE",
    "EAAEF_BOOTSTRAP_PREFLIGHT_SCHEMA",
    "EAAEF_CONTAINER_IMAGE_QUALIFICATION_SCHEMA",
    "EAAEF_CONTAINER_PROFILE_SCHEMA",
    "EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA",
    "EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA_V2",
    "EAAEF_WORKER_CONTAINER_PROFILE_REVIEWER_ROLE_V2",
    "EAAEF_WORKER_IMAGE_QUALIFICATION_SCHEMA",
    "EAAEF_DIRECT_LAUNCH_MODE",
    "EAAEF_PROVIDER_CONTAINER_QUALIFICATION_SCHEMA",
    "EAAEF_PROVIDER_CONTAINER_VERIFICATION_SCHEMA",
    "EAAEF_QUALIFICATION_SIGNER_ROLE",
    "EAAEF_REQUIRED_ROUTE_EFFECTS",
    "EAAEFProviderContainerQualificationVerification",
    "ExternalAgentFabricBootstrapPreflight",
    "eaaef_provider_budget_binding_cid",
    "eaaef_provider_container_qualification_signing_bytes",
    "eaaef_repository_binding_cid",
    "eaaef_worker_container_profile_signing_bytes",
    "eaaef_worker_image_qualification_signing_bytes",
    "evaluate_external_agent_fabric_bootstrap_preflight",
    "prepare_eaaef_provider_container_qualification",
    "seal_eaaef_provider_container_qualification",
    "validate_eaaef_worker_container_profile_artifact",
    "verify_eaaef_provider_container_qualification",
)
