"""Content-bound propagation for EAAEF worker-network authority.

The configured-board launch capsule chooses the worker and provider
principals.  A later, reviewer-signed invocation chooses the private profile
root from which one deterministic authorization artifact is read.  This
module carries those two independent authorities through supervisor and
provider child boundaries without accepting an authorization path from a
caller.

It deliberately does not create networks, issue authorizations, sign
artifacts, or launch processes.  Missing or stale artifacts remain a typed
pre-effect failure in :mod:`worker_network`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from .worker_network import (
    PROVIDER_HOSTNAME_ALLOWLISTS,
    WorkerNetworkAuthorization,
    load_worker_network_authorization,
    worker_network_authorization_relative_path,
)

EAAEF_WORKER_NETWORK_LAUNCH_AUTHORITY_FLAG: Final = (
    "--eaaef-worker-network-launch-authority-json"
)
EAAEF_WORKER_NETWORK_ATTEMPT_AUTHORITY_FLAG: Final = (
    "--eaaef-worker-network-attempt-authority-json"
)
EAAEF_WORKER_NETWORK_LAUNCH_AUTHORITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-worker-network-launch-authority@1"
)
EAAEF_WORKER_NETWORK_ATTEMPT_AUTHORITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-worker-network-attempt-authority@1"
)
EAAEF_BOARD_NAMESPACE: Final = (
    "external-agent-autonomous-execution-fabric-v1"
)

_CID = re.compile(r"sha256:[0-9a-f]{64}")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}")
_MAX_AUTHORITY_BYTES = 64 * 1024
_LAUNCH_FIELDS = frozenset(
    {
        "schema",
        "board_namespace",
        "configured_board_capsule_cid",
        "live_verification_cid",
        "source_head",
        "source_tree",
        "accepted_control_plane_capsule_id",
        "accepted_control_plane_pin_cid",
        "worker_principal_did",
        "provider_principal_did",
        "qualified_worker_image_digest",
        "qualified_worker_container_profile_cid",
        "worker_network_authorization_policy",
        "authority_cid",
    }
)
_ATTEMPT_FIELDS = frozenset(
    {
        "schema",
        "launch_authority",
        "invocation_binding_id",
        "invocation_id",
        "logical_attempt_id",
        "task_id",
        "worktree_id",
        "route_id",
        "control_plane_capsule_id",
        "workspace_cid",
        "providers",
        "authority_cid",
    }
)
_PROVIDER_FIELDS = frozenset(
    {"provider", "artifact_cid", "artifact_relative_path"}
)
_POLICY_FIELDS = frozenset(
    {
        "schema",
        "authorization_schema",
        "verifier_interface",
        "artifact_path_authority",
        "artifact_relative_path_template",
        "dynamic_caller_path_allowed",
        "expected_artifact_cid_required",
        "expected_worker_principal_did_required",
        "expected_provider_principal_did_required",
        "control_plane_capsule_binding_required",
        "task_plan_source_worktree_effect_binding_required",
        "container_and_lease_binding_required",
        "create_start_restart_reverification_required",
        "supported_providers",
        "child_propagation_status",
    }
)
_POLICY_SEMANTICS: Final = {
    "schema": (
        "ipfs_accelerate_py/agent-supervisor/"
        "eaaef-worker-network-dispatch-policy@1"
    ),
    "authorization_schema": (
        "ipfs_accelerate_py/eaaef-worker-network-authorization@1"
    ),
    "verifier_interface": "verify_worker_network_authorization@1",
    "artifact_path_authority": "verified_invocation_profile_dir",
    "artifact_relative_path_template": (
        "network-authorizations/<sha256(invocation_id)>/<provider>.json"
    ),
    "dynamic_caller_path_allowed": False,
    "expected_artifact_cid_required": True,
    "expected_worker_principal_did_required": True,
    "expected_provider_principal_did_required": True,
    "control_plane_capsule_binding_required": True,
    "task_plan_source_worktree_effect_binding_required": True,
    "container_and_lease_binding_required": True,
    "create_start_restart_reverification_required": True,
    "supported_providers": ["codex", "grok"],
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate worker-network authority key: {key}")
        result[key] = value
    return result


def _parse_json_object(raw: str, *, noun: str) -> dict[str, Any]:
    encoded = str(raw or "").encode("utf-8")
    if not encoded or len(encoded) > _MAX_AUTHORITY_BYTES:
        raise ValueError(f"{noun} is absent or oversized")
    try:
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{noun} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{noun} is not an object")
    return value


def _pin_dict(pin: object) -> dict[str, Any]:
    value = pin.as_dict() if hasattr(pin, "as_dict") else pin
    if not isinstance(value, Mapping):
        raise ValueError("accepted control-plane pin is not canonical")
    result = dict(value)
    required = {
        "schema",
        "runner_path",
        "runner_sha256",
        "capsule_root",
        "capsule_id",
        "source_head",
        "source_tree",
        "archive_sha256",
    }
    if set(result) != required or any(
        not isinstance(item, str) or not item for item in result.values()
    ):
        raise ValueError("accepted control-plane pin is not canonical")
    return result


def _validate_policy(
    value: object,
    *,
    require_admitted: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _POLICY_FIELDS:
        raise ValueError("worker-network dispatch policy shape is invalid")
    policy = dict(value)
    if any(policy.get(key) != expected for key, expected in _POLICY_SEMANTICS.items()):
        raise ValueError("worker-network dispatch policy semantics drifted")
    status = policy.get("child_propagation_status")
    if status not in {"unavailable_fail_closed", "admitted"}:
        raise ValueError("worker-network child propagation status is invalid")
    if require_admitted and status != "admitted":
        raise ValueError("worker-network child propagation is not admitted")
    return policy


def parse_worker_network_launch_authority(
    value: str | Mapping[str, Any],
    *,
    accepted_control_plane_pin: object | None = None,
    require_admitted: bool = True,
) -> dict[str, Any]:
    """Parse one path-free capsule-derived launch authority."""

    authority = (
        _parse_json_object(value, noun="worker-network launch authority")
        if isinstance(value, str)
        else dict(value)
        if isinstance(value, Mapping)
        else None
    )
    if authority is None or set(authority) != _LAUNCH_FIELDS:
        raise ValueError("worker-network launch authority shape is invalid")
    body = {key: item for key, item in authority.items() if key != "authority_cid"}
    worker_did = authority.get("worker_principal_did")
    provider_did = authority.get("provider_principal_did")
    if (
        authority.get("schema") != EAAEF_WORKER_NETWORK_LAUNCH_AUTHORITY_SCHEMA
        or authority.get("board_namespace") != EAAEF_BOARD_NAMESPACE
        or any(
            _CID.fullmatch(str(authority.get(name) or "")) is None
            for name in (
                "configured_board_capsule_cid",
                "live_verification_cid",
                "accepted_control_plane_capsule_id",
                "accepted_control_plane_pin_cid",
                "qualified_worker_image_digest",
                "qualified_worker_container_profile_cid",
                "authority_cid",
            )
        )
        or _GIT_OBJECT.fullmatch(str(authority.get("source_head") or "")) is None
        or _GIT_OBJECT.fullmatch(str(authority.get("source_tree") or "")) is None
        or not isinstance(worker_did, str)
        or not worker_did.startswith("did:key:z")
        or not isinstance(provider_did, str)
        or not provider_did.startswith("did:key:z")
        or worker_did == provider_did
        or authority.get("authority_cid") != _content_id(body)
    ):
        raise ValueError("worker-network launch authority binding is invalid")
    authority["worker_network_authorization_policy"] = _validate_policy(
        authority.get("worker_network_authorization_policy"),
        require_admitted=require_admitted,
    )
    if accepted_control_plane_pin is not None:
        pin = _pin_dict(accepted_control_plane_pin)
        if (
            authority["accepted_control_plane_capsule_id"] != pin["capsule_id"]
            or authority["accepted_control_plane_pin_cid"] != _content_id(pin)
            or authority["source_head"] != pin["source_head"]
            or authority["source_tree"] != pin["source_tree"]
        ):
            raise ValueError(
                "worker-network launch authority differs from the accepted control plane"
            )
    return authority


def build_worker_network_launch_authority(
    live_verification: Mapping[str, Any],
    *,
    accepted_control_plane_pin: object,
    require_admitted: bool = True,
) -> dict[str, Any]:
    """Project only path-free worker authority from one verified live seal."""

    pin = _pin_dict(accepted_control_plane_pin)
    body: dict[str, Any] = {
        "schema": EAAEF_WORKER_NETWORK_LAUNCH_AUTHORITY_SCHEMA,
        "board_namespace": EAAEF_BOARD_NAMESPACE,
        "configured_board_capsule_cid": str(
            live_verification.get("configured_board_capsule_cid") or ""
        ),
        "live_verification_cid": str(
            live_verification.get("verification_cid") or ""
        ),
        "source_head": str(live_verification.get("source_head") or ""),
        "source_tree": str(live_verification.get("source_tree") or ""),
        "accepted_control_plane_capsule_id": pin["capsule_id"],
        "accepted_control_plane_pin_cid": _content_id(pin),
        "worker_principal_did": str(
            live_verification.get("provider_worker_principal_did") or ""
        ),
        "provider_principal_did": str(
            live_verification.get("provider_principal_did") or ""
        ),
        "qualified_worker_image_digest": str(
            live_verification.get("qualified_worker_image_digest") or ""
        ),
        "qualified_worker_container_profile_cid": str(
            live_verification.get("qualified_worker_container_profile_cid") or ""
        ),
        "worker_network_authorization_policy": dict(
            live_verification.get("worker_network_authorization_policy") or {}
        ),
    }
    authority = {**body, "authority_cid": _content_id(body)}
    return parse_worker_network_launch_authority(
        authority,
        accepted_control_plane_pin=pin,
        require_admitted=require_admitted,
    )


def canonical_worker_network_launch_authority_json(
    authority: str | Mapping[str, Any],
    *,
    accepted_control_plane_pin: object | None = None,
    require_admitted: bool = True,
) -> str:
    parsed = parse_worker_network_launch_authority(
        authority,
        accepted_control_plane_pin=accepted_control_plane_pin,
        require_admitted=require_admitted,
    )
    return _canonical_bytes(parsed).decode("utf-8")


def _invocation_values(invocation_binding: object) -> dict[str, str]:
    control_plane = getattr(invocation_binding, "control_plane", None)
    return {
        "invocation_binding_id": str(
            getattr(invocation_binding, "content_id", "") or ""
        ),
        "invocation_id": str(
            getattr(invocation_binding, "invocation_id", "") or ""
        ),
        "logical_attempt_id": str(
            getattr(invocation_binding, "logical_attempt_id", "") or ""
        ),
        "task_id": str(getattr(invocation_binding, "task_id", "") or ""),
        "worktree_id": str(
            getattr(invocation_binding, "worktree_id", "") or ""
        ),
        "route_id": str(getattr(invocation_binding, "route_id", "") or ""),
        "control_plane_capsule_id": str(
            getattr(control_plane, "capsule_id", "") or ""
        ),
    }


def _invocation_pin(invocation_binding: object) -> object:
    pin = getattr(invocation_binding, "control_plane", None)
    if pin is None:
        raise ValueError("worker-network invocation has no accepted control plane")
    return pin


def build_worker_network_attempt_authority(
    launch_authority: str | Mapping[str, Any],
    *,
    invocation_binding: object,
    workspace: Path,
    providers: Sequence[str],
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Verify deterministic artifacts and bind their exact CIDs for a runner."""

    pin = _invocation_pin(invocation_binding)
    launch = parse_worker_network_launch_authority(
        launch_authority,
        accepted_control_plane_pin=pin,
        require_admitted=True,
    )
    normalized_workspace = workspace.expanduser().resolve(strict=True)
    if normalized_workspace != Path(
        str(getattr(invocation_binding, "workspace_path", ""))
    ).expanduser().resolve(strict=True):
        raise ValueError("worker-network attempt workspace differs from invocation")
    provider_names = tuple(
        sorted({str(provider).strip().lower() for provider in providers})
    )
    if (
        not provider_names
        or any(provider not in PROVIDER_HOSTNAME_ALLOWLISTS for provider in provider_names)
        or list(provider_names)
        != launch["worker_network_authorization_policy"]["supported_providers"]
    ):
        raise ValueError("worker-network attempt provider population is invalid")
    entries: list[dict[str, str]] = []
    for provider in provider_names:
        authorization = load_worker_network_authorization(
            invocation_binding=invocation_binding,
            provider=provider,
            workspace=normalized_workspace,
            now_ms=now_ms,
            expected_worker_principal_did=launch["worker_principal_did"],
            expected_provider_principal_did=launch["provider_principal_did"],
        )
        entries.append(
            {
                "provider": provider,
                "artifact_cid": authorization.artifact_cid,
                "artifact_relative_path": worker_network_authorization_relative_path(
                    str(getattr(invocation_binding, "invocation_id", "")),
                    provider,
                ).as_posix(),
            }
        )
    body: dict[str, Any] = {
        "schema": EAAEF_WORKER_NETWORK_ATTEMPT_AUTHORITY_SCHEMA,
        "launch_authority": launch,
        **_invocation_values(invocation_binding),
        "workspace_cid": _content_id({"workspace": str(normalized_workspace)}),
        "providers": entries,
    }
    authority = {**body, "authority_cid": _content_id(body)}
    verify_worker_network_attempt_authority(
        authority,
        invocation_binding=invocation_binding,
        workspace=normalized_workspace,
        required_providers=provider_names,
        now_ms=now_ms,
    )
    return authority


def verify_worker_network_attempt_authority(
    value: str | Mapping[str, Any],
    *,
    invocation_binding: object,
    workspace: Path,
    required_providers: Sequence[str] = ("codex", "grok"),
    now_ms: int | None = None,
) -> dict[str, WorkerNetworkAuthorization]:
    """Reverify launch provenance and every source-addressed artifact."""

    authority = (
        _parse_json_object(value, noun="worker-network attempt authority")
        if isinstance(value, str)
        else dict(value)
        if isinstance(value, Mapping)
        else None
    )
    if authority is None or set(authority) != _ATTEMPT_FIELDS:
        raise ValueError("worker-network attempt authority shape is invalid")
    body = {key: item for key, item in authority.items() if key != "authority_cid"}
    if (
        authority.get("schema") != EAAEF_WORKER_NETWORK_ATTEMPT_AUTHORITY_SCHEMA
        or _CID.fullmatch(str(authority.get("authority_cid") or "")) is None
        or authority.get("authority_cid") != _content_id(body)
    ):
        raise ValueError("worker-network attempt authority identity is invalid")
    launch = parse_worker_network_launch_authority(
        authority.get("launch_authority"),
        accepted_control_plane_pin=_invocation_pin(invocation_binding),
        require_admitted=True,
    )
    expected_invocation = _invocation_values(invocation_binding)
    if any(authority.get(name) != expected for name, expected in expected_invocation.items()):
        raise ValueError("worker-network attempt invocation binding drifted")
    normalized_workspace = workspace.expanduser().resolve(strict=True)
    signed_workspace = Path(
        str(getattr(invocation_binding, "workspace_path", ""))
    ).expanduser().resolve(strict=True)
    if (
        normalized_workspace != signed_workspace
        or authority.get("workspace_cid")
        != _content_id({"workspace": str(normalized_workspace)})
    ):
        raise ValueError("worker-network attempt workspace binding drifted")
    requested = tuple(
        sorted({str(provider).strip().lower() for provider in required_providers})
    )
    providers = authority.get("providers")
    if not isinstance(providers, list) or len(providers) != len(requested):
        raise ValueError("worker-network attempt provider population drifted")
    observed_names: list[str] = []
    result: dict[str, WorkerNetworkAuthorization] = {}
    for entry in providers:
        if not isinstance(entry, Mapping) or set(entry) != _PROVIDER_FIELDS:
            raise ValueError("worker-network attempt provider binding is invalid")
        provider = str(entry.get("provider") or "")
        observed_names.append(provider)
        expected_relative = worker_network_authorization_relative_path(
            str(getattr(invocation_binding, "invocation_id", "")), provider
        ).as_posix()
        artifact_cid = str(entry.get("artifact_cid") or "")
        if (
            _CID.fullmatch(artifact_cid) is None
            or entry.get("artifact_relative_path") != expected_relative
        ):
            raise ValueError("worker-network authorization source binding drifted")
        result[provider] = load_worker_network_authorization(
            invocation_binding=invocation_binding,
            provider=provider,
            workspace=normalized_workspace,
            now_ms=now_ms,
            expected_artifact_cid=artifact_cid,
            expected_worker_principal_did=launch["worker_principal_did"],
            expected_provider_principal_did=launch["provider_principal_did"],
        )
    if tuple(observed_names) != requested or tuple(sorted(result)) != requested:
        raise ValueError("worker-network attempt providers are not canonical")
    return result


def canonical_worker_network_attempt_authority_json(
    authority: Mapping[str, Any],
) -> str:
    if set(authority) != _ATTEMPT_FIELDS:
        raise ValueError("worker-network attempt authority shape is invalid")
    return _canonical_bytes(dict(authority)).decode("utf-8")
