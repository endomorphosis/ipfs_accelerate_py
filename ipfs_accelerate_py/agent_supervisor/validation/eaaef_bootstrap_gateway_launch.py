"""Signed launch authority for the process-remote EAAEF bootstrap gateway.

The tracked scheduler profile contains only stable schema names, a
source-addressed path template, and reviewer trust roots.  A post-freeze
operator publishes one independently signed operational capability at the
deterministic path returned by
:func:`eaaef_bootstrap_operational_capability_relative_path`.  The capability
binds the already signed admission and configured-board capsule to the exact
operational schema, borrowed-transaction adapter, Quack owner, endpoints,
plan frontier, and command-authorization service.

Nothing in this module opens DuckDB, resolves an opaque secret handle, loads a
private key, or starts a process.  The command-authorization client speaks one
closed length-prefixed protocol over an already bound private Unix socket and
accepts only a fully verified ``AuthorizedStateCommand@1`` response.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import socket
import stat
import struct
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.agent_implementation_route import (
    AgentImplementationControlPlanePin,
)
from ipfs_accelerate_py.agent_supervisor.control.profile_authority import (
    LocalProfileTampered,
    verify_did_key_signature,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CommandKind,
    StateCommand,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_command_authorization import (
    AUTHORIZED_STATE_COMMAND_SCHEMA,
    AuthorizedStateCommand,
    QuackCommandAuthorizationError,
    QuackCommandAuthorizationPolicy,
    verify_authorized_state_command,
)
from ipfs_accelerate_py.agent_supervisor.validation.external_agent_configured_board_capsule import (
    VerifiedExternalAgentConfiguredBoardLiveSeal,
)

EAAEF_BOOTSTRAP_OPERATIONAL_CAPABILITY_INTERFACE: Final = (
    "EAAEFBootstrapDaemonOperationalCapability@2"
)
EAAEF_BOOTSTRAP_OPERATIONAL_CAPABILITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-bootstrap-daemon-operational-capability@2"
)
EAAEF_BOOTSTRAP_OPERATIONAL_CAPABILITY_REVIEW_ROLE: Final = (
    "independent_eaaef_bootstrap_gateway_reviewer"
)
EAAEF_BOOTSTRAP_GATEWAY_BINDING_INTERFACE: Final = (
    "EAAEFBootstrapGatewayBinding@1"
)
EAAEF_BOOTSTRAP_GATEWAY_BINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-bootstrap-gateway-binding@1"
)
EAAEF_COMMAND_AUTHORIZATION_SERVICE_INTERFACE: Final = (
    "EAAEFCommandAuthorizationService@1"
)
EAAEF_COMMAND_AUTHORIZATION_SERVICE_CAPABILITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-command-authorization-service-capability@1"
)
EAAEF_COMMAND_AUTHORIZATION_SERVICE_REVIEW_ROLE: Final = (
    "independent_eaaef_command_authorization_service_reviewer"
)
EAAEF_COMMAND_AUTHORIZATION_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-command-authorization-request@1"
)
EAAEF_BOOTSTRAP_GATEWAY_LAUNCH_AUTHORITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-bootstrap-gateway-launch-authority@1"
)
EAAEF_BOOTSTRAP_GATEWAY_LAUNCH_AUTHORITY_FLAG: Final = (
    "--eaaef-bootstrap-gateway-launch-authority-json"
)
EAAEF_BOOTSTRAP_OPERATIONAL_CAPABILITY_PATH_TEMPLATE: Final = (
    "eaaef-bootstrap-daemon-operational-capability--"
    "<source_head>--<plan_root_sha256>.json"
)
EAAEF_BOARD_NAMESPACE: Final = (
    "external-agent-autonomous-execution-fabric-v1"
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CONTENT_ID = re.compile(r"(?:sha256:[0-9a-f]{64}|b[a-z2-7]{20,})\Z")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,511}\Z")
_MAX_CAPABILITY_BYTES = 512 * 1024
_MAX_AUTHORIZATION_MESSAGE_BYTES = 128 * 1024
_MAX_UNIX_SOCKET_PATH_BYTES = 100
_MIN_AUTHORIZATION_REQUEST_TIMEOUT_MS = 100
_MAX_AUTHORIZATION_REQUEST_TIMEOUT_MS = 30_000
_MAX_SERVICE_CAPABILITY_LIFETIME_MS = 15 * 60 * 1000
_MAX_OPERATIONAL_CAPABILITY_LIFETIME_MS = 15 * 60 * 1000

_SERVICE_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "board_namespace",
        "transport_kind",
        "endpoint",
        "service_principal_did",
        "approver_principal_did",
        "authorized_client_principal_did",
        "authorization_policy_cid",
        "request_schema",
        "response_schema",
        "peer_credentials_required",
        "response_signature_verification_required",
        "private_key_available_to_child",
        "raw_token_available_to_child",
        "dynamic_endpoint_allowed",
        "maximum_request_bytes",
        "maximum_response_bytes",
        "request_timeout_ms",
        "expected_server_uid",
        "expected_server_pid",
        "expected_server_process_start_time_ticks",
        "issuance_nonce",
        "issued_at_ms",
        "expires_at_ms",
        "reviewer_did",
        "reviewer_role",
        "reviewer_signature",
        "capability_cid",
    }
)

_OPERATIONAL_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "board_namespace",
        "source_head",
        "source_tree",
        "configuration_root",
        "bootstrap_admission_receipt_cid",
        "configured_board_capsule_cid",
        "materialization_receipt_cid",
        "materialization_database_program_binding_cid",
        "materialization_operational_profile_cid",
        "operational_profile_verification",
        "operation_vocabulary_cid",
        "borrowed_transaction_handler_interface",
        "borrowed_transaction_handler_source_evidence_cid",
        "borrowed_transaction_adapter_qualification_cid",
        "gateway_interface",
        "gateway_binding_cid",
        "control_plane_schema_version",
        "state_schema_revision",
        "operational_profile_id",
        "authorization_policy",
        "authorization_policy_cid",
        "command_fabric_qualification_cid",
        "command_authorization_service",
        "active_plan_root_cid",
        "active_plan_revision",
        "frontier_cid",
        "operations",
        "excluded_operations",
        "store_id",
        "store_generation",
        "schema_revision",
        "board_scope",
        "shard_id",
        "owner_principal_did",
        "owner_session_id",
        "owner_generation",
        "lease_id",
        "lease_kind",
        "lease_mode",
        "command_principal_did",
        "fencing_token",
        "fence_epoch",
        "command_endpoint",
        "command_secret_handle",
        "state_endpoint",
        "state_secret_handle",
        "worker_principal_did",
        "issuance_nonce",
        "issued_at_ms",
        "expires_at_ms",
        "production_admitted",
        "owner_provisions_lease_after_capability_verification",
        "owner_renews_lease_after_reverification",
        "materializer_mints_external_lease",
        "direct_database_open",
        "portal_fallback",
        "local_sidecar_writes",
        "arbitrary_sql_enabled",
        "reviewer_did",
        "reviewer_role",
        "reviewer_signature",
        "capability_cid",
    }
)

_LAUNCH_FIELDS: Final = frozenset(
    {
        "schema",
        "board_namespace",
        "source_head",
        "source_tree",
        "configuration_root",
        "accepted_control_plane_capsule_id",
        "accepted_control_plane_pin_cid",
        "bootstrap_admission_receipt_cid",
        "configured_board_capsule_cid",
        "live_verification_cid",
        "active_plan_root_cid",
        "active_plan_revision",
        "frontier_cid",
        "operational_capability",
        "operational_capability_cid",
        "operational_capability_file_sha256",
        "authority_cid",
    }
)

_EXPECTED_OPERATIONAL_BINDING_FIELDS: Final = frozenset(
    {
        "source_head",
        "source_tree",
        "configuration_root",
        "bootstrap_admission_receipt_cid",
        "configured_board_capsule_cid",
        "materialization_receipt_cid",
        "materialization_database_program_binding_cid",
        "materialization_operational_profile_cid",
        "operation_vocabulary_cid",
        "borrowed_transaction_handler_interface",
        "borrowed_transaction_handler_source_evidence_cid",
        "borrowed_transaction_adapter_qualification_cid",
        "gateway_interface",
        "gateway_binding_cid",
        "control_plane_schema_version",
        "state_schema_revision",
        "operational_profile_id",
        "authorization_policy_cid",
        "command_fabric_qualification_cid",
        "active_plan_root_cid",
        "active_plan_revision",
        "frontier_cid",
        "store_id",
        "store_generation",
        "schema_revision",
        "board_scope",
        "shard_id",
        "owner_principal_did",
        "owner_session_id",
        "owner_generation",
        "lease_id",
        "lease_kind",
        "lease_mode",
        "command_principal_did",
        "fencing_token",
        "fence_epoch",
        "command_endpoint",
        "command_secret_handle",
        "state_endpoint",
        "state_secret_handle",
        "worker_principal_did",
    }
)


class EAAEFBootstrapGatewayLaunchError(RuntimeError):
    """A signed capability or its process-bound projection was invalid."""


class EAAEFCommandAuthorizationServiceError(EAAEFBootstrapGatewayLaunchError):
    """The external command-authorizer protocol failed closed."""


_VERIFIED_OPERATIONAL_CAPABILITY_TOKEN = object()


class VerifiedEAAEFBootstrapOperationalCapability(Mapping[str, Any]):
    """Immutable result of complete signed operational-capability review."""

    __slots__ = ("_value",)

    def __init__(self, token: object, value: Mapping[str, Any]) -> None:
        if token is not _VERIFIED_OPERATIONAL_CAPABILITY_TOKEN:
            raise TypeError(
                "verified EAAEF operational capabilities come from the verifier"
            )
        self._value = MappingProxyType(
            json.loads(
                _bounded_canonical_bytes(
                    dict(value), noun="verified EAAEF operational capability"
                ).decode("ascii")
            )
        )

    def __getitem__(self, key: str) -> Any:
        value = self._value[key]
        if isinstance(value, (dict, list)):
            return json.loads(_canonical_bytes(value).decode("ascii"))
        return value

    def __iter__(self):
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)


_VERIFIED_GATEWAY_LIVE_SEAL_TOKEN = object()


class VerifiedEAAEFBootstrapGatewayLiveSeal(Mapping[str, Any]):
    """Immutable result of joining a verified board seal to capability @2.

    The constructor is deliberately closed.  A JSON object that merely copies
    this report's fields is not a verified live seal and cannot reach the
    launch-authority builder.
    """

    __slots__ = (
        "_value",
        "_trusted_reviewer_dids",
        "_trusted_service_reviewer_dids",
        "_expected_operational_bindings",
        "_forbidden_reviewer_dids",
    )

    def __init__(
        self,
        token: object,
        value: Mapping[str, Any],
        *,
        trusted_reviewer_dids: Sequence[str],
        trusted_authorization_service_reviewer_dids: Sequence[str],
        expected_operational_bindings: Mapping[str, Any],
        forbidden_reviewer_dids: Sequence[str],
    ) -> None:
        if token is not _VERIFIED_GATEWAY_LIVE_SEAL_TOKEN:
            raise TypeError("verified EAAEF gateway live seals come from the verifier")
        detached = json.loads(_bounded_canonical_bytes(
            dict(value), noun="verified EAAEF gateway live seal"
        ).decode("ascii"))
        self._value = MappingProxyType(detached)
        self._trusted_reviewer_dids = tuple(trusted_reviewer_dids)
        self._trusted_service_reviewer_dids = tuple(
            trusted_authorization_service_reviewer_dids
        )
        self._expected_operational_bindings = MappingProxyType(
            json.loads(_canonical_bytes(dict(expected_operational_bindings)).decode("ascii"))
        )
        self._forbidden_reviewer_dids = tuple(forbidden_reviewer_dids)

    def __getitem__(self, key: str) -> Any:
        value = self._value[key]
        if isinstance(value, (dict, list)):
            return json.loads(_canonical_bytes(value).decode("ascii"))
        return value

    def __iter__(self):
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise EAAEFBootstrapGatewayLaunchError(
            "gateway capability is not canonical JSON"
        ) from exc


def _bounded_canonical_bytes(value: Any, *, noun: str) -> bytes:
    raw = _canonical_bytes(value)
    if not raw or len(raw) > _MAX_CAPABILITY_BYTES:
        raise EAAEFBootstrapGatewayLaunchError(f"{noun} is absent or oversized")
    return raw


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EAAEFBootstrapGatewayLaunchError(
                f"duplicate JSON key in gateway capability: {key}"
            )
        result[key] = value
    return result


def _parse_canonical_json_object(raw: bytes, *, noun: str) -> dict[str, Any]:
    if not raw or len(raw) > _MAX_CAPABILITY_BYTES:
        raise EAAEFBootstrapGatewayLaunchError(f"{noun} is absent or oversized")
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EAAEFBootstrapGatewayLaunchError(
            f"{noun} is invalid JSON"
        ) from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise EAAEFBootstrapGatewayLaunchError(
            f"{noun} is not canonical JSON"
        )
    return value


def _cid(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def eaaef_bootstrap_gateway_binding(
    capability: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the closed, stable gateway identity bound by capability @2.

    The projection deliberately excludes signatures, issue/expiry times,
    nonces, capability CIDs, and source-evidence CIDs.  Those values may be
    renewed independently; changing a repository protocol, owner, command
    principal, endpoint, store, profile, vocabulary, or handler changes this
    stable gateway identity.
    """

    if not isinstance(capability, Mapping):
        raise EAAEFBootstrapGatewayLaunchError(
            "bootstrap gateway binding source is not an object"
        )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_repository import (
        QUACK_STATE_REPOSITORY_INTERFACE,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_borrowed_transaction import (
        EAAEF_BORROWED_TRANSACTION_HANDLER_INTERFACE,
        EAAEF_BORROWED_TRANSACTION_HANDLER_SCHEMA,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_operational_schema import (
        EAAEF_OPERATIONAL_PROFILE_ID,
    )

    control_plane_schema = _identifier(
        capability.get("control_plane_schema_version"),
        "control_plane_schema_version",
    )
    state_schema = _identifier(
        capability.get("state_schema_revision"), "state_schema_revision"
    )
    profile_id = _identifier(
        capability.get("operational_profile_id"), "operational_profile_id"
    )
    if (
        control_plane_schema != QUACK_STATE_REPOSITORY_INTERFACE
        or state_schema != EAAEF_OPERATIONAL_PROFILE_ID
        or profile_id != EAAEF_OPERATIONAL_PROFILE_ID
        or capability.get("schema_revision") != state_schema
        or capability.get("borrowed_transaction_handler_interface")
        != EAAEF_BORROWED_TRANSACTION_HANDLER_INTERFACE
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "bootstrap gateway schema/profile identity is invalid"
        )
    command_principal = _did(
        capability.get("command_principal_did"), "command_principal_did"
    )
    worker_principal = _did(
        capability.get("worker_principal_did"), "worker_principal_did"
    )
    if command_principal != worker_principal:
        raise EAAEFBootstrapGatewayLaunchError(
            "bootstrap gateway command principal differs from worker authority"
        )
    projection = {
        "schema": EAAEF_BOOTSTRAP_GATEWAY_BINDING_SCHEMA,
        "interface": EAAEF_BOOTSTRAP_GATEWAY_BINDING_INTERFACE,
        "board_namespace": _identifier(
            capability.get("board_namespace"), "board_namespace"
        ),
        "shard_id": _identifier(capability.get("shard_id"), "shard_id"),
        "command_endpoint": _loopback_quack(
            capability.get("command_endpoint"), "command_endpoint"
        ),
        "state_endpoint": _loopback_quack(
            capability.get("state_endpoint"), "state_endpoint"
        ),
        "store_id": _identifier(capability.get("store_id"), "store_id"),
        "store_generation": _identifier(
            capability.get("store_generation"), "store_generation"
        ),
        "owner_principal_did": _did(
            capability.get("owner_principal_did"), "owner_principal_did"
        ),
        "owner_session_id": _identifier(
            capability.get("owner_session_id"), "owner_session_id"
        ),
        "owner_generation": _positive(
            capability.get("owner_generation"), "owner_generation"
        ),
        "command_principal_did": command_principal,
        "authorization_policy_cid": _sha(
            capability.get("authorization_policy_cid"),
            "authorization_policy_cid",
        ),
        "command_fabric_qualification_cid": _sha(
            capability.get("command_fabric_qualification_cid"),
            "command_fabric_qualification_cid",
        ),
        "borrowed_transaction_adapter_qualification_cid": _sha(
            capability.get("borrowed_transaction_adapter_qualification_cid"),
            "borrowed_transaction_adapter_qualification_cid",
        ),
        "operational_profile_verification_cid": _content_id(
            capability.get("materialization_operational_profile_cid"),
            "materialization_operational_profile_cid",
        ),
        "fence_epoch": _positive(capability.get("fence_epoch"), "fence_epoch"),
        "control_plane_schema_version": control_plane_schema,
        "state_schema_revision": state_schema,
        "operational_profile_id": profile_id,
        "operation_vocabulary_cid": _content_id(
            capability.get("operation_vocabulary_cid"),
            "operation_vocabulary_cid",
        ),
        "handler_interface": EAAEF_BORROWED_TRANSACTION_HANDLER_INTERFACE,
        "handler_schema": EAAEF_BORROWED_TRANSACTION_HANDLER_SCHEMA,
    }
    if projection["board_namespace"] != EAAEF_BOARD_NAMESPACE:
        raise EAAEFBootstrapGatewayLaunchError(
            "bootstrap gateway binding changed board namespace"
        )
    return MappingProxyType(projection)


def eaaef_bootstrap_gateway_binding_cid(
    capability: Mapping[str, Any],
) -> str:
    """Return the content identity of the stable gateway projection."""

    return _cid(dict(eaaef_bootstrap_gateway_binding(capability)))


def _sha(value: Any, noun: str) -> str:
    text = str(value or "")
    if _SHA256.fullmatch(text) is None:
        raise EAAEFBootstrapGatewayLaunchError(
            f"{noun} must be a full sha256 identity"
        )
    return text


def _content_id(value: Any, noun: str) -> str:
    text = str(value or "")
    if _CONTENT_ID.fullmatch(text) is None:
        raise EAAEFBootstrapGatewayLaunchError(
            f"{noun} must be a canonical content identity"
        )
    return text


def _git(value: Any, noun: str) -> str:
    text = str(value or "")
    if _GIT_OBJECT.fullmatch(text) is None:
        raise EAAEFBootstrapGatewayLaunchError(
            f"{noun} must be an exact Git object identity"
        )
    return text


def _identifier(value: Any, noun: str) -> str:
    text = str(value or "")
    if _SAFE_ID.fullmatch(text) is None:
        raise EAAEFBootstrapGatewayLaunchError(
            f"{noun} is not a bounded identifier"
        )
    return text


def _positive(value: Any, noun: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EAAEFBootstrapGatewayLaunchError(
            f"{noun} must be a positive integer"
        )
    if maximum is not None and value > maximum:
        raise EAAEFBootstrapGatewayLaunchError(f"{noun} exceeds its bound")
    return int(value)


def _nonnegative(value: Any, noun: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EAAEFBootstrapGatewayLaunchError(
            f"{noun} must be a non-negative integer"
        )
    if maximum is not None and value > maximum:
        raise EAAEFBootstrapGatewayLaunchError(f"{noun} exceeds its bound")
    return int(value)


def _did(value: Any, noun: str) -> str:
    text = str(value or "")
    if not text.startswith("did:key:z") or len(text.encode("utf-8")) > 512:
        raise EAAEFBootstrapGatewayLaunchError(
            f"{noun} must be an Ed25519 did:key"
        )
    return text


def _secret_handle(value: Any, noun: str) -> str:
    text = _identifier(value, noun)
    if not text.startswith(("secret-handle:", "handle:", "vault://")):
        raise EAAEFBootstrapGatewayLaunchError(
            f"{noun} must be an opaque secret handle"
        )
    return text


def _unix_endpoint(value: Any) -> str:
    endpoint = str(value or "")
    if not endpoint.startswith("unix:/") or "\x00" in endpoint:
        raise EAAEFBootstrapGatewayLaunchError(
            "authorization service endpoint must be an absolute unix: path"
        )
    path = Path(endpoint.removeprefix("unix:"))
    if (
        not path.is_absolute()
        or ".." in path.parts
        or len(os.fsencode(path)) > _MAX_UNIX_SOCKET_PATH_BYTES
        or path.parts[:2] != ("/", "run")
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "authorization service endpoint must be a bounded /run Unix socket"
        )
    return endpoint


def _linux_process_start_time_ticks(pid: int) -> int:
    """Return Linux's PID-reuse-resistant process birth field."""

    checked_pid = _positive(pid, "expected_server_pid")
    try:
        raw = Path(f"/proc/{checked_pid}/stat").read_text(encoding="utf-8")
        close = raw.rfind(")")
        fields = raw[close + 2 :].split()
        if close < 1 or len(fields) <= 19 or fields[0] == "Z":
            raise ValueError("malformed or zombie process stat")
        value = int(fields[19])
    except (OSError, UnicodeError, ValueError) as exc:
        raise EAAEFCommandAuthorizationServiceError(
            "authorization service process birth is unavailable"
        ) from exc
    return _positive(value, "observed_server_process_start_time_ticks")


def _safe_unix_socket_metadata(
    path: Path,
    *,
    expected_uid: int,
) -> os.stat_result:
    """Validate the complete /run path without following a symlink."""

    if path.parts[:2] != ("/", "run"):
        raise EAAEFCommandAuthorizationServiceError(
            "authorization service socket escaped /run"
        )
    current = Path("/run")
    for part in path.parts[2:-1]:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise EAAEFCommandAuthorizationServiceError(
                "authorization service socket parent is unavailable"
            ) from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid not in {0, expected_uid}
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise EAAEFCommandAuthorizationServiceError(
                "authorization service socket parent ownership is unsafe"
            )
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise EAAEFCommandAuthorizationServiceError(
            "authorization service socket is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISSOCK(metadata.st_mode)
        or metadata.st_uid != expected_uid
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or metadata.st_nlink != 1
    ):
        raise EAAEFCommandAuthorizationServiceError(
            "authorization service socket ownership is unsafe"
        )
    return metadata


def _loopback_quack(value: Any, noun: str) -> str:
    endpoint = str(value or "")
    if not endpoint.startswith(("quack:127.0.0.1:", "quack:[::1]:")):
        raise EAAEFBootstrapGatewayLaunchError(
            f"{noun} must be a loopback Quack endpoint"
        )
    try:
        port = int(endpoint.rsplit(":", 1)[1])
    except (TypeError, ValueError) as exc:
        raise EAAEFBootstrapGatewayLaunchError(
            f"{noun} has an invalid port"
        ) from exc
    if not 1024 <= port <= 65535:
        raise EAAEFBootstrapGatewayLaunchError(
            f"{noun} must use an unprivileged port"
        )
    return endpoint


def _verify_signature(
    value: Mapping[str, Any],
    *,
    reviewer_did: str,
    signature_field: str,
    cid_field: str,
) -> None:
    signed = dict(value)
    claimed_cid = str(signed.pop(cid_field, ""))
    signature = signed.pop(signature_field, None)
    if claimed_cid != _cid({**signed, signature_field: signature}):
        raise EAAEFBootstrapGatewayLaunchError(
            "signed capability self-address is invalid"
        )
    if not isinstance(signature, str) or not signature:
        raise EAAEFBootstrapGatewayLaunchError("signed capability is unsigned")
    try:
        verify_did_key_signature(
            identity_did=reviewer_did,
            payload=signed,
            signature=signature,
        )
    except (LocalProfileTampered, ValueError) as exc:
        raise EAAEFBootstrapGatewayLaunchError(
            "signed capability reviewer signature is invalid"
        ) from exc


def eaaef_bootstrap_operational_capability_relative_path(
    source_head: str,
    plan_root_cid: str,
    *,
    registry_prefix: str,
) -> Path:
    """Return the only repository-relative path accepted for the capability."""

    head = _git(source_head, "source_head")
    root = _sha(plan_root_cid, "plan_root_cid")
    prefix = Path(str(registry_prefix or ""))
    if prefix.is_absolute() or ".." in prefix.parts or not prefix.parts:
        raise EAAEFBootstrapGatewayLaunchError(
            "authority registry prefix is not repository-relative"
        )
    return prefix / (
        "eaaef-bootstrap-daemon-operational-capability--"
        f"{head}--{root.removeprefix('sha256:')}.json"
    )


def load_eaaef_bootstrap_operational_capability(
    repo_root: str | Path,
    *,
    source_head: str,
    plan_root_cid: str,
    registry_prefix: str,
    expected_file_sha256: str = "",
) -> tuple[Mapping[str, Any], str, str]:
    """Open one source-addressed capability with no links or path races."""

    root = Path(repo_root).resolve(strict=True)
    relative = eaaef_bootstrap_operational_capability_relative_path(
        source_head,
        plan_root_cid,
        registry_prefix=registry_prefix,
    )
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise EAAEFBootstrapGatewayLaunchError(
            "bootstrap operational capability requires nofollow openat support"
        )
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | nofollow | directory
    file_flags = os.O_RDONLY | os.O_CLOEXEC | nofollow
    try:
        directory_fd = os.open(root, directory_flags)
    except OSError as exc:
        raise EAAEFBootstrapGatewayLaunchError(
            "bootstrap operational capability root is unavailable"
        ) from exc
    try:
        root_metadata = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or root_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(root_metadata.st_mode) & 0o022
        ):
            raise EAAEFBootstrapGatewayLaunchError(
                "bootstrap operational capability root ownership is unsafe"
            )
        for part in relative.parts[:-1]:
            try:
                next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            except OSError as exc:
                raise EAAEFBootstrapGatewayLaunchError(
                    "bootstrap operational capability parent is unavailable"
                ) from exc
            try:
                metadata = os.fstat(next_fd)
                if (
                    not stat.S_ISDIR(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(metadata.st_mode) & 0o022
                ):
                    raise EAAEFBootstrapGatewayLaunchError(
                        "bootstrap operational capability parent ownership is unsafe"
                    )
            except BaseException:
                os.close(next_fd)
                raise
            os.close(directory_fd)
            directory_fd = next_fd
        try:
            descriptor = os.open(
                relative.name,
                file_flags,
                dir_fd=directory_fd,
            )
        except OSError as exc:
            raise EAAEFBootstrapGatewayLaunchError(
                "bootstrap operational capability is unavailable"
            ) from exc
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or before.st_nlink != 1
                or before.st_size <= 0
                or before.st_size > _MAX_CAPABILITY_BYTES
                or stat.S_IMODE(before.st_mode) & 0o077
            ):
                raise EAAEFBootstrapGatewayLaunchError(
                    "bootstrap operational capability is not an owner-only file"
                )
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(65_536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
            pathname = os.stat(
                relative.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        finally:
            os.close(descriptor)
    finally:
        os.close(directory_fd)
    identity = lambda item: (  # noqa: E731 - compact immutable stat projection
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
        or identity(before) != identity(after)
        or identity(before) != identity(pathname)
        or stat.S_ISLNK(pathname.st_mode)
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "bootstrap operational capability changed during stable read"
        )
    file_sha256 = "sha256:" + hashlib.sha256(raw).hexdigest()
    if expected_file_sha256 and file_sha256 != _sha(
        expected_file_sha256,
        "expected operational capability file sha256",
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "bootstrap operational capability file identity changed"
        )
    value = _parse_canonical_json_object(
        raw,
        noun="bootstrap operational capability",
    )
    if set(value) != _OPERATIONAL_FIELDS:
        raise EAAEFBootstrapGatewayLaunchError(
            "bootstrap operational capability shape is invalid"
        )
    return MappingProxyType(value), file_sha256, relative.as_posix()


def seal_eaaef_command_authorization_service_capability(
    statement: Mapping[str, Any],
    *,
    reviewer_signature: str,
) -> Mapping[str, Any]:
    """Join an external reviewer signature without loading a signing key."""

    if not isinstance(statement, Mapping) or set(statement) != (
        _SERVICE_FIELDS - {"reviewer_signature", "capability_cid"}
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "command authorization service statement shape is invalid"
        )
    _bounded_canonical_bytes(dict(statement), noun="command authorization service statement")
    signature = str(reviewer_signature or "")
    if not signature:
        raise EAAEFBootstrapGatewayLaunchError(
            "command authorization service reviewer signature is absent"
        )
    signed = {**dict(statement), "reviewer_signature": signature}
    result = {**signed, "capability_cid": _cid(signed)}
    _bounded_canonical_bytes(result, noun="command authorization service capability")
    return MappingProxyType(result)


def seal_eaaef_bootstrap_operational_capability(
    statement: Mapping[str, Any],
    *,
    reviewer_signature: str,
) -> Mapping[str, Any]:
    """Join an external capability-review signature without authority effects."""

    if not isinstance(statement, Mapping) or set(statement) != (
        _OPERATIONAL_FIELDS - {"reviewer_signature", "capability_cid"}
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "bootstrap operational capability statement shape is invalid"
        )
    _bounded_canonical_bytes(dict(statement), noun="bootstrap operational capability statement")
    signature = str(reviewer_signature or "")
    if not signature:
        raise EAAEFBootstrapGatewayLaunchError(
            "bootstrap operational capability reviewer signature is absent"
        )
    signed = {**dict(statement), "reviewer_signature": signature}
    result = {**signed, "capability_cid": _cid(signed)}
    _bounded_canonical_bytes(result, noun="bootstrap operational capability")
    return MappingProxyType(result)


def verify_eaaef_command_authorization_service_capability(
    value: object,
    *,
    trusted_reviewer_dids: Sequence[str],
    expected_authorization_policy_cid: str,
    expected_client_principal_did: str,
    expected_owner_principal_did: str,
    now_ms: int,
    forbidden_reviewer_dids: Sequence[str] = (),
) -> Mapping[str, Any]:
    """Verify the separately signed, private command-authorizer endpoint."""

    if not isinstance(value, Mapping) or set(value) != _SERVICE_FIELDS:
        raise EAAEFBootstrapGatewayLaunchError(
            "command authorization service capability shape is invalid"
        )
    capability = dict(value)
    _bounded_canonical_bytes(capability, noun="command authorization service capability")
    reviewer = _did(capability.get("reviewer_did"), "service reviewer_did")
    trusted = frozenset(str(item) for item in trusted_reviewer_dids)
    issued = _positive(capability.get("issued_at_ms"), "service issued_at_ms")
    expires = _positive(capability.get("expires_at_ms"), "service expires_at_ms")
    service_did = _did(
        capability.get("service_principal_did"), "service_principal_did"
    )
    approver_did = _did(
        capability.get("approver_principal_did"), "approver_principal_did"
    )
    client_did = _did(
        capability.get("authorized_client_principal_did"),
        "authorized_client_principal_did",
    )
    owner_did = _did(expected_owner_principal_did, "expected_owner_principal_did")
    forbidden_reviewers = frozenset(
        _did(item, "forbidden_reviewer_did") for item in forbidden_reviewer_dids
    )
    _nonnegative(
        capability.get("expected_server_uid"),
        "expected_server_uid",
        maximum=(2**32) - 2,
    )
    expected_server_pid = _positive(
        capability.get("expected_server_pid"), "expected_server_pid"
    )
    expected_start_ticks = _positive(
        capability.get("expected_server_process_start_time_ticks"),
        "expected_server_process_start_time_ticks",
    )
    request_timeout_ms = _positive(
        capability.get("request_timeout_ms"),
        "request_timeout_ms",
        maximum=_MAX_AUTHORIZATION_REQUEST_TIMEOUT_MS,
    )
    if (
        capability.get("schema")
        != EAAEF_COMMAND_AUTHORIZATION_SERVICE_CAPABILITY_SCHEMA
        or capability.get("interface")
        != EAAEF_COMMAND_AUTHORIZATION_SERVICE_INTERFACE
        or capability.get("board_namespace") != EAAEF_BOARD_NAMESPACE
        or capability.get("transport_kind")
        != "private_unix_length_prefixed_json"
        or capability.get("request_schema")
        != EAAEF_COMMAND_AUTHORIZATION_REQUEST_SCHEMA
        or capability.get("response_schema") != AUTHORIZED_STATE_COMMAND_SCHEMA
        or capability.get("authorization_policy_cid")
        != _sha(expected_authorization_policy_cid, "authorization_policy_cid")
        or client_did != _did(
            expected_client_principal_did, "expected_client_principal_did"
        )
        or reviewer not in trusted
        or capability.get("reviewer_role")
        != EAAEF_COMMAND_AUTHORIZATION_SERVICE_REVIEW_ROLE
        or len({reviewer, service_did, approver_did, client_did, owner_did}) != 5
        or not {
            reviewer,
            service_did,
            approver_did,
            client_did,
            owner_did,
        }.isdisjoint(forbidden_reviewers)
        or issued > now_ms
        or now_ms >= expires
        or issued >= expires
        or expires - issued > _MAX_SERVICE_CAPABILITY_LIFETIME_MS
        or request_timeout_ms < _MIN_AUTHORIZATION_REQUEST_TIMEOUT_MS
        or expected_server_pid <= 0
        or expected_start_ticks <= 0
        or capability.get("peer_credentials_required") is not True
        or capability.get("response_signature_verification_required") is not True
        or capability.get("private_key_available_to_child") is not False
        or capability.get("raw_token_available_to_child") is not False
        or capability.get("dynamic_endpoint_allowed") is not False
        or not _SAFE_ID.fullmatch(str(capability.get("issuance_nonce") or ""))
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "command authorization service capability binding is invalid"
        )
    _unix_endpoint(capability.get("endpoint"))
    maximum_request = _positive(
        capability.get("maximum_request_bytes"),
        "maximum_request_bytes",
        maximum=_MAX_AUTHORIZATION_MESSAGE_BYTES,
    )
    maximum_response = _positive(
        capability.get("maximum_response_bytes"),
        "maximum_response_bytes",
        maximum=_MAX_AUTHORIZATION_MESSAGE_BYTES,
    )
    if maximum_request < 4096 or maximum_response < 4096:
        raise EAAEFBootstrapGatewayLaunchError(
            "authorization service message bounds are too small"
        )
    _verify_signature(
        capability,
        reviewer_did=reviewer,
        signature_field="reviewer_signature",
        cid_field="capability_cid",
    )
    return MappingProxyType(capability)


def verify_eaaef_bootstrap_operational_capability(
    value: object,
    *,
    trusted_reviewer_dids: Sequence[str],
    trusted_authorization_service_reviewer_dids: Sequence[str],
    now_ms: int,
    expected: Mapping[str, Any],
    forbidden_reviewer_dids: Sequence[str] = (),
) -> VerifiedEAAEFBootstrapOperationalCapability:
    """Verify one exact production capability without creating a transport."""

    if not isinstance(value, Mapping) or set(value) != _OPERATIONAL_FIELDS:
        raise EAAEFBootstrapGatewayLaunchError(
            "bootstrap operational capability shape is invalid"
        )
    if not isinstance(expected, Mapping) or set(expected) != (
        _EXPECTED_OPERATIONAL_BINDING_FIELDS
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "expected operational artifact bindings are absent or open shaped"
        )
    capability = dict(value)
    _bounded_canonical_bytes(capability, noun="bootstrap operational capability")
    reviewer = _did(capability.get("reviewer_did"), "reviewer_did")
    if reviewer not in frozenset(str(item) for item in trusted_reviewer_dids):
        raise EAAEFBootstrapGatewayLaunchError(
            "bootstrap operational capability reviewer is untrusted"
        )
    issued = _positive(capability.get("issued_at_ms"), "issued_at_ms")
    expires = _positive(capability.get("expires_at_ms"), "expires_at_ms")
    _git(capability.get("source_head"), "source_head")
    _git(capability.get("source_tree"), "source_tree")
    owner = _did(capability.get("owner_principal_did"), "owner_principal_did")
    worker = _did(capability.get("worker_principal_did"), "worker_principal_did")
    command_principal = _did(
        capability.get("command_principal_did"), "command_principal_did"
    )
    forbidden_reviewers = frozenset(
        _did(item, "forbidden_reviewer_did") for item in forbidden_reviewer_dids
    )
    if (
        capability.get("schema") != EAAEF_BOOTSTRAP_OPERATIONAL_CAPABILITY_SCHEMA
        or capability.get("interface")
        != EAAEF_BOOTSTRAP_OPERATIONAL_CAPABILITY_INTERFACE
        or capability.get("board_namespace") != EAAEF_BOARD_NAMESPACE
        or capability.get("reviewer_role")
        != EAAEF_BOOTSTRAP_OPERATIONAL_CAPABILITY_REVIEW_ROLE
        or reviewer in {owner, worker}
        or reviewer in forbidden_reviewers
        or owner == worker
        or command_principal != worker
        or issued > now_ms
        or now_ms >= expires
        or issued >= expires
        or expires - issued > _MAX_OPERATIONAL_CAPABILITY_LIFETIME_MS
        or capability.get("production_admitted") is not True
        or capability.get("owner_provisions_lease_after_capability_verification")
        is not True
        or capability.get("owner_renews_lease_after_reverification") is not True
        or capability.get("materializer_mints_external_lease") is not False
        or capability.get("direct_database_open") is not False
        or capability.get("portal_fallback") is not False
        or capability.get("local_sidecar_writes") is not False
        or capability.get("arbitrary_sql_enabled") is not False
        or not _SAFE_ID.fullmatch(str(capability.get("issuance_nonce") or ""))
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "bootstrap operational capability authority is invalid"
        )
    for name in (
        "configuration_root",
        "bootstrap_admission_receipt_cid",
        "configured_board_capsule_cid",
        "materialization_receipt_cid",
        "materialization_database_program_binding_cid",
        "borrowed_transaction_adapter_qualification_cid",
        "authorization_policy_cid",
        "command_fabric_qualification_cid",
        "active_plan_root_cid",
        "frontier_cid",
    ):
        _sha(capability.get(name), name)
    for name in (
        "materialization_operational_profile_cid",
        "operation_vocabulary_cid",
        "borrowed_transaction_handler_source_evidence_cid",
    ):
        _content_id(capability.get(name), name)
    operations = capability.get("operations")
    excluded = capability.get("excluded_operations")
    if (
        not isinstance(operations, list)
        or not isinstance(excluded, list)
        or operations != sorted(set(str(item) for item in operations))
        or excluded != sorted(set(str(item) for item in excluded))
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "bootstrap operational capability operation sets are not canonical"
        )
    from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_bootstrap_daemon_gateway import (
        EAAEF_BOOTSTRAP_DAEMON_EXCLUDED_OPERATIONS,
        EAAEF_BOOTSTRAP_DAEMON_GATEWAY_INTERFACE,
        EAAEF_BOOTSTRAP_DAEMON_OPERATIONS,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_borrowed_transaction import (
        EAAEF_BORROWED_TRANSACTION_HANDLER_INTERFACE,
        eaaef_bootstrap_handler_source_evidence,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_operational_schema import (
        _REQUIRED_COLUMNS,
        EAAEF_OPERATIONAL_PROFILE_ID,
        EAAEF_OPERATIONAL_PROFILE_INTERFACE,
        EAAEF_OPERATIONAL_PROFILE_SCHEMA,
        eaaef_operation_vocabulary_cid,
        eaaef_operational_profile_contract,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_daemon_gateway import (
        quack_daemon_operation_command_vocabulary,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_identity import (
        canonical_content_cid,
    )

    if (
        frozenset(operations) != EAAEF_BOOTSTRAP_DAEMON_OPERATIONS
        or frozenset(excluded) != EAAEF_BOOTSTRAP_DAEMON_EXCLUDED_OPERATIONS
        or capability.get("operation_vocabulary_cid")
        != eaaef_operation_vocabulary_cid(EAAEF_BOOTSTRAP_DAEMON_OPERATIONS)
        or capability.get("borrowed_transaction_handler_interface")
        != EAAEF_BORROWED_TRANSACTION_HANDLER_INTERFACE
        or capability.get("gateway_interface")
        != EAAEF_BOOTSTRAP_DAEMON_GATEWAY_INTERFACE
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "bootstrap operational capability protocol identities drifted"
        )
    gateway_binding_cid = _sha(
        capability.get("gateway_binding_cid"), "gateway_binding_cid"
    )
    if gateway_binding_cid != eaaef_bootstrap_gateway_binding_cid(capability):
        raise EAAEFBootstrapGatewayLaunchError(
            "bootstrap operational capability gateway binding is invalid"
        )
    handler_evidence = dict(
        eaaef_bootstrap_handler_source_evidence(
            board_namespace=EAAEF_BOARD_NAMESPACE,
            shard_id=_identifier(capability.get("shard_id"), "shard_id"),
        )
    )
    if capability.get("borrowed_transaction_handler_source_evidence_cid") != (
        handler_evidence["handler_source_evidence_cid"]
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "bootstrap adapter source evidence differs from the sealed implementation"
        )
    profile = capability.get("operational_profile_verification")
    if not isinstance(profile, Mapping):
        raise EAAEFBootstrapGatewayLaunchError(
            "operational profile verification is absent"
        )
    profile_body = dict(profile)
    profile_verification_cid = str(profile_body.pop("verification_cid", ""))
    expected_profile_contract = dict(
        eaaef_operational_profile_contract(
            operation_vocabulary_cid=str(capability["operation_vocabulary_cid"])
        )
    )
    expected_profile_fields = set(expected_profile_contract) | {
        "valid",
        "schema_fingerprint",
        "required_index_set_cid",
        "required_columns",
        "verification_cid",
    }
    expected_required_columns = {
        table: list(columns) for table, columns in _REQUIRED_COLUMNS.items()
    }
    if (
        set(profile) != expected_profile_fields
        or profile.get("valid") is not True
        or profile.get("interface") != EAAEF_OPERATIONAL_PROFILE_INTERFACE
        or profile.get("schema") != EAAEF_OPERATIONAL_PROFILE_SCHEMA
        or profile.get("profile_id") != EAAEF_OPERATIONAL_PROFILE_ID
        or profile.get("operation_vocabulary_cid")
        != capability.get("operation_vocabulary_cid")
        or not _CONTENT_ID.fullmatch(str(profile.get("profile_contract_cid") or ""))
        or not _CONTENT_ID.fullmatch(str(profile.get("verification_cid") or ""))
        or not _CONTENT_ID.fullmatch(
            str(profile.get("required_index_set_cid") or "")
        )
        or not _SHA256.fullmatch(str(profile.get("migration_checksum") or ""))
        or not _CONTENT_ID.fullmatch(str(profile.get("catalog_fingerprint") or ""))
        or not _CONTENT_ID.fullmatch(str(profile.get("schema_fingerprint") or ""))
        or profile.get("runtime_ddl_allowed") is not False
        or profile.get("direct_database_open_allowed") is not False
        or profile.get("sidecar_writes_allowed") is not False
        or profile.get("required_columns") != expected_required_columns
        or capability.get("schema_revision") != EAAEF_OPERATIONAL_PROFILE_ID
        or capability.get("state_schema_revision")
        != EAAEF_OPERATIONAL_PROFILE_ID
        or capability.get("operational_profile_id")
        != EAAEF_OPERATIONAL_PROFILE_ID
        or profile.get("profile_id")
        not in {
            capability.get("schema_revision"),
            capability.get("state_schema_revision"),
            capability.get("operational_profile_id"),
        }
        or capability.get("materialization_operational_profile_cid")
        != profile_verification_cid
        or any(
            profile.get(name) != expected_value
            for name, expected_value in expected_profile_contract.items()
        )
        or profile_verification_cid != canonical_content_cid(profile_body)
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "operational profile verification is invalid or cross-bound"
        )
    policy_value = capability.get("authorization_policy")
    if not isinstance(policy_value, Mapping):
        raise EAAEFBootstrapGatewayLaunchError("authorization policy is absent")
    try:
        policy = QuackCommandAuthorizationPolicy(
            board_namespace=policy_value.get("board_namespace"),
            shard_id=policy_value.get("shard_id"),
            store_id=policy_value.get("store_id"),
            authority_ref_cid=policy_value.get("authority_ref_cid"),
            owner_principal_did=policy_value.get("owner_principal_did"),
            owner_generation=policy_value.get("owner_generation"),
            fence_epoch=policy_value.get("fence_epoch"),
            trusted_approver_dids=frozenset(
                policy_value.get("trusted_approver_dids") or ()
            ),
            authorized_principal_dids=frozenset(
                policy_value.get("authorized_principal_dids") or ()
            ),
            allowed_command_kinds=frozenset(
                policy_value.get("allowed_command_kinds") or ()
            ),
            maximum_authorization_lifetime_ms=policy_value.get(
                "maximum_authorization_lifetime_ms"
            ),
        )
    except (QuackCommandAuthorizationError, TypeError, ValueError) as exc:
        raise EAAEFBootstrapGatewayLaunchError(
            "authorization policy is invalid"
        ) from exc
    if (
        dict(policy.to_dict()) != dict(policy_value)
        or policy.policy_cid != capability.get("authorization_policy_cid")
        or policy.board_namespace != EAAEF_BOARD_NAMESPACE
        or policy.shard_id != capability.get("shard_id")
        or policy.store_id != capability.get("store_id")
        or policy.owner_principal_did != owner
        or policy.owner_generation != capability.get("owner_generation")
        or policy.fence_epoch != capability.get("fence_epoch")
        or policy.authorized_principal_dids != frozenset({command_principal})
        or policy.allowed_command_kinds
        != frozenset(
            CommandKind(quack_daemon_operation_command_vocabulary()[operation])
            for operation in EAAEF_BOOTSTRAP_DAEMON_OPERATIONS
        )
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "authorization policy differs from bootstrap owner authority"
        )
    command_endpoint = _loopback_quack(
        capability.get("command_endpoint"), "command_endpoint"
    )
    state_endpoint = _loopback_quack(
        capability.get("state_endpoint"), "state_endpoint"
    )
    if command_endpoint == state_endpoint:
        raise EAAEFBootstrapGatewayLaunchError(
            "command and projection endpoints must be distinct"
        )
    _secret_handle(capability.get("command_secret_handle"), "command_secret_handle")
    _secret_handle(capability.get("state_secret_handle"), "state_secret_handle")
    _positive(capability.get("owner_generation"), "owner_generation")
    _positive(capability.get("fencing_token"), "fencing_token")
    _positive(capability.get("fence_epoch"), "fence_epoch")
    _positive(capability.get("active_plan_revision"), "active_plan_revision")
    _identifier(capability.get("store_id"), "store_id")
    _identifier(capability.get("store_generation"), "store_generation")
    _identifier(capability.get("schema_revision"), "schema_revision")
    shard = _identifier(capability.get("shard_id"), "shard_id")
    _identifier(capability.get("owner_session_id"), "owner_session_id")
    _identifier(capability.get("lease_id"), "lease_id")
    if (
        capability.get("board_scope")
        != f"board:{EAAEF_BOARD_NAMESPACE}:{shard}"
        or capability.get("lease_kind") != "board_shard_scheduler"
        or capability.get("lease_mode") != "shared_scheduler"
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "bootstrap scheduler lease seed is not the closed board-shard lease"
        )
    if any(
        marker in str(capability.get("store_id") or "")
        for marker in ("/", "\\", ".duckdb", ".ddb")
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "store_id must remain an opaque owner identity"
        )
    service = verify_eaaef_command_authorization_service_capability(
        capability.get("command_authorization_service"),
        trusted_reviewer_dids=trusted_authorization_service_reviewer_dids,
        expected_authorization_policy_cid=policy.policy_cid,
        expected_client_principal_did=worker,
        expected_owner_principal_did=owner,
        now_ms=now_ms,
        forbidden_reviewer_dids=tuple(forbidden_reviewers | {reviewer}),
    )
    if (
        policy.trusted_approver_dids
        != frozenset({service["approver_principal_did"]})
        or reviewer
        in {
            service["reviewer_did"],
            service["service_principal_did"],
            service["approver_principal_did"],
        }
        or not (
            int(service["issued_at_ms"])
            <= issued
            < expires
            <= int(service["expires_at_ms"])
        )
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "authorization service role or lifetime is not independently bound"
        )
    _verify_signature(
        capability,
        reviewer_did=reviewer,
        signature_field="reviewer_signature",
        cid_field="capability_cid",
    )
    mismatched = sorted(
        name
        for name, expected_value in expected.items()
        if capability.get(name) != expected_value
    )
    if mismatched:
        raise EAAEFBootstrapGatewayLaunchError(
            "bootstrap operational capability crossed live identities: "
            + ",".join(mismatched)
        )
    result = dict(capability)
    result["command_authorization_service"] = dict(service)
    return VerifiedEAAEFBootstrapOperationalCapability(
        _VERIFIED_OPERATIONAL_CAPABILITY_TOKEN,
        result,
    )


def verify_eaaef_bootstrap_operation_submission(
    envelope: AuthorizedStateCommand,
    intent: Mapping[str, Any],
    *,
    verified_capability: VerifiedEAAEFBootstrapOperationalCapability,
    authorization_policy: QuackCommandAuthorizationPolicy,
    now_ms: int,
) -> Mapping[str, Any]:
    """Verify the exact EAAEF capability/envelope/intent join.

    The generic 39-operation verifier remains unchanged.  This verifier uses
    the same frozen intent envelope while admitting only the EAAEF 31-operation
    subset, the independently reviewed stable gateway binding, and the extra
    authorization-request correlation field required by the process-remote
    signer protocol.
    """

    if type(verified_capability) is not VerifiedEAAEFBootstrapOperationalCapability:
        raise EAAEFBootstrapGatewayLaunchError(
            "EAAEF operation submission requires a typed verified capability"
        )
    if (
        type(envelope) is not AuthorizedStateCommand
        or type(envelope.command) is not StateCommand
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "EAAEF operation submission requires exact AuthorizedStateCommand@1 "
            "and StateCommand@1 base types"
        )
    if not isinstance(authorization_policy, QuackCommandAuthorizationPolicy):
        raise EAAEFBootstrapGatewayLaunchError(
            "EAAEF operation submission authorization policy is untyped"
        )
    if not isinstance(intent, Mapping):
        raise EAAEFBootstrapGatewayLaunchError(
            "EAAEF operation submission intent is not an object"
        )
    capability = dict(verified_capability)
    from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_bootstrap_daemon_gateway import (
        EAAEF_BOOTSTRAP_DAEMON_OPERATIONS,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_daemon_gateway import (
        QuackDaemonGatewayError,
        quack_daemon_operation_command_vocabulary,
        quack_daemon_operation_intent,
        quack_daemon_operation_intent_from_envelope,
        quack_daemon_state_command_parameters,
    )

    plain_intent = dict(intent)
    operation = str(plain_intent.get("operation") or "")
    arguments = plain_intent.get("arguments")
    if operation not in EAAEF_BOOTSTRAP_DAEMON_OPERATIONS or not isinstance(
        arguments, Mapping
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "EAAEF operation submission is outside the exact 31-operation vocabulary"
        )
    try:
        rebuilt_intent = dict(
            quack_daemon_operation_intent(
                gateway_binding_cid=str(capability["gateway_binding_cid"]),
                operational_capability_cid=str(capability["capability_cid"]),
                operation=operation,
                arguments=dict(arguments),
            )
        )
        transported_intent = dict(
            quack_daemon_operation_intent_from_envelope(envelope)
        )
    except QuackDaemonGatewayError as exc:
        raise EAAEFBootstrapGatewayLaunchError(
            "EAAEF operation submission intent is malformed"
        ) from exc
    if (
        plain_intent != rebuilt_intent
        or transported_intent != rebuilt_intent
        or plain_intent.get("gateway_binding_cid")
        != capability.get("gateway_binding_cid")
        or plain_intent.get("operational_capability_cid")
        != capability.get("capability_cid")
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "EAAEF operation submission intent identity is invalid"
        )
    try:
        verify_authorized_state_command(
            envelope,
            policy=authorization_policy,
            now_ms=int(now_ms),
        )
    except QuackCommandAuthorizationError as exc:
        raise EAAEFBootstrapGatewayLaunchError(
            "EAAEF operation submission authorization is invalid"
        ) from exc
    command = envelope.command
    parameters = dict(command.parameters)
    authorization_request_cid = _sha(
        parameters.get("authorization_request_cid"),
        "authorization_request_cid",
    )
    try:
        expected_parameters = dict(
            quack_daemon_state_command_parameters(
                rebuilt_intent,
                request_id=envelope.request_id,
                principal_did=envelope.principal_did,
                authority_ref_cid=envelope.authority_ref_cid,
                lease_id=envelope.lease_id,
                scope_id=envelope.scope_id,
                deadline_ms=envelope.deadline_ms,
                fencing_token=parameters.get("fencing_token"),
                idempotency_key=command.idempotency_key,
            )
        )
    except QuackDaemonGatewayError as exc:
        raise EAAEFBootstrapGatewayLaunchError(
            "EAAEF operation submission parameters are malformed"
        ) from exc
    expected_parameters["authorization_request_cid"] = authorization_request_cid
    vocabulary = quack_daemon_operation_command_vocabulary()
    service = capability.get("command_authorization_service")
    if not isinstance(service, Mapping):
        raise EAAEFBootstrapGatewayLaunchError(
            "EAAEF operation submission service capability is absent"
        )
    expected_command_id = f"{envelope.request_id}:{operation.replace('.', '-')}"
    checks = (
        command.command_kind is CommandKind(vocabulary[operation]),
        command.command_id == expected_command_id,
        command.store_id == capability.get("store_id"),
        command.session_id == envelope.lease_id,
        command.expected_generation == capability.get("owner_generation"),
        command.fence_epoch == capability.get("fence_epoch"),
        parameters == expected_parameters,
        envelope.board_namespace == capability.get("board_namespace"),
        envelope.shard_id == capability.get("shard_id"),
        envelope.owner_principal_did == capability.get("owner_principal_did"),
        envelope.principal_did == capability.get("command_principal_did"),
        envelope.approver_did == service.get("approver_principal_did"),
        envelope.authority_ref_cid == authorization_policy.authority_ref_cid,
        envelope.issued_at_ms
        >= max(
            int(capability["issued_at_ms"]),
            int(service["issued_at_ms"]),
        ),
        envelope.expires_at_ms
        <= min(
            int(capability["expires_at_ms"]),
            int(service["expires_at_ms"]),
        ),
        envelope.deadline_ms
        <= min(
            int(capability["expires_at_ms"]),
            int(service["expires_at_ms"]),
        ),
        authorization_policy.policy_cid
        == capability.get("authorization_policy_cid"),
        authorization_policy.board_namespace == capability.get("board_namespace"),
        authorization_policy.shard_id == capability.get("shard_id"),
        authorization_policy.store_id == capability.get("store_id"),
        authorization_policy.owner_principal_did
        == capability.get("owner_principal_did"),
        authorization_policy.owner_generation == capability.get("owner_generation"),
        authorization_policy.fence_epoch == capability.get("fence_epoch"),
        authorization_policy.authorized_principal_dids
        == frozenset({str(capability.get("command_principal_did") or "")}),
        authorization_policy.trusted_approver_dids
        == frozenset({str(service.get("approver_principal_did") or "")}),
    )
    if not all(checks):
        raise EAAEFBootstrapGatewayLaunchError(
            "EAAEF operation capability/envelope/CAS identity join failed"
        )
    return MappingProxyType(
        {
            "operation": operation,
            "arguments": dict(arguments),
            "intent_cid": rebuilt_intent["intent_cid"],
            "authorization_request_cid": authorization_request_cid,
            "fencing_token": int(parameters["fencing_token"]),
            "expected_version": command.expected_revision,
            "idempotency_key": command.idempotency_key,
            "one_use_nonce": envelope.one_use_nonce,
            "lease_id": envelope.lease_id,
            "scope_id": envelope.scope_id,
            "effect": envelope.effect,
        }
    )


def verify_eaaef_bootstrap_gateway_live_seal(
    configured_board_live_seal: VerifiedExternalAgentConfiguredBoardLiveSeal,
    *,
    operational_capability: Mapping[str, Any],
    operational_capability_file_sha256: str,
    operational_capability_relative_path: str,
    authority_registry_prefix: str,
    trusted_reviewer_dids: Sequence[str],
    trusted_authorization_service_reviewer_dids: Sequence[str],
    expected_operational_bindings: Mapping[str, Any],
    now_ms: int,
    forbidden_reviewer_dids: Sequence[str] = (),
) -> VerifiedEAAEFBootstrapGatewayLiveSeal:
    """Join a real configured-board live-seal token to capability @2.

    This is the only production constructor for the typed gateway live seal.
    It deliberately rejects same-shaped mappings so a caller cannot promote a
    self-addressed report without reopening the admission/capsule evidence.
    """

    if type(configured_board_live_seal) is not (
        VerifiedExternalAgentConfiguredBoardLiveSeal
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "gateway launch requires the exact configured-board live-seal token"
        )
    base = dict(configured_board_live_seal)
    base_body = {key: item for key, item in base.items() if key != "verification_cid"}
    if (
        base.get("valid") is not True
        or base.get("authority_mutated") is not False
        or base.get("process_started") is not False
        or base.get("verification_cid") != _cid(base_body)
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "configured-board live-seal token is internally inconsistent"
        )
    verified = verify_eaaef_bootstrap_operational_capability(
        operational_capability,
        trusted_reviewer_dids=trusted_reviewer_dids,
        trusted_authorization_service_reviewer_dids=(
            trusted_authorization_service_reviewer_dids
        ),
        now_ms=now_ms,
        expected=expected_operational_bindings,
        forbidden_reviewer_dids=forbidden_reviewer_dids,
    )
    active_plan = base.get("active_plan")
    if not isinstance(active_plan, Mapping):
        raise EAAEFBootstrapGatewayLaunchError(
            "configured-board live seal lacks its active plan"
        )
    joins = {
        "source_head": base.get("source_head"),
        "source_tree": base.get("source_tree"),
        "configuration_root": base.get("configuration_root"),
        "bootstrap_admission_receipt_cid": base.get(
            "bootstrap_admission_receipt_cid"
        ),
        "configured_board_capsule_cid": base.get(
            "configured_board_capsule_cid"
        ),
        "active_plan_root_cid": active_plan.get("plan_root_cid"),
        "active_plan_revision": active_plan.get("revision"),
        "frontier_cid": base.get("frontier_cid"),
    }
    mismatched = sorted(
        field for field, value in joins.items() if verified.get(field) != value
    )
    if mismatched:
        raise EAAEFBootstrapGatewayLaunchError(
            "bootstrap capability differs from the configured-board live seal:"
            + ",".join(mismatched)
        )
    relative = eaaef_bootstrap_operational_capability_relative_path(
        str(verified["source_head"]),
        str(verified["active_plan_root_cid"]),
        registry_prefix=authority_registry_prefix,
    ).as_posix()
    if operational_capability_relative_path != relative:
        raise EAAEFBootstrapGatewayLaunchError(
            "bootstrap capability path is not source/plan addressed"
        )
    file_sha256 = _sha(
        operational_capability_file_sha256,
        "operational_capability_file_sha256",
    )
    observed_file_sha256 = "sha256:" + hashlib.sha256(
        _canonical_bytes(dict(verified))
    ).hexdigest()
    if file_sha256 != observed_file_sha256:
        raise EAAEFBootstrapGatewayLaunchError(
            "bootstrap capability file identity is detached from its canonical bytes"
        )
    report = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "eaaef-bootstrap-gateway-live-seal-verification@1"
        ),
        "valid": True,
        "configured_board_live_seal_verification_cid": str(
            base["verification_cid"]
        ),
        "source_head": verified["source_head"],
        "source_tree": verified["source_tree"],
        "configuration_root": verified["configuration_root"],
        "accepted_control_plane_pin_cid": str(
            base.get("accepted_control_plane_pin_cid") or ""
        ),
        "bootstrap_admission_receipt_cid": verified[
            "bootstrap_admission_receipt_cid"
        ],
        "configured_board_capsule_cid": verified[
            "configured_board_capsule_cid"
        ],
        "active_plan": {
            "plan_root_cid": verified["active_plan_root_cid"],
            "revision": verified["active_plan_revision"],
        },
        "frontier_cid": verified["frontier_cid"],
        "bootstrap_operational_capability": dict(verified),
        "bootstrap_operational_capability_cid": verified["capability_cid"],
        "bootstrap_operational_capability_file_sha256": file_sha256,
        "bootstrap_operational_capability_relative_path": relative,
        "authority_mutated": False,
        "process_started": False,
    }
    report["verification_cid"] = _cid(report)
    return VerifiedEAAEFBootstrapGatewayLiveSeal(
        _VERIFIED_GATEWAY_LIVE_SEAL_TOKEN,
        report,
        trusted_reviewer_dids=trusted_reviewer_dids,
        trusted_authorization_service_reviewer_dids=(
            trusted_authorization_service_reviewer_dids
        ),
        expected_operational_bindings=expected_operational_bindings,
        forbidden_reviewer_dids=forbidden_reviewer_dids,
    )


def parse_eaaef_bootstrap_gateway_launch_authority(
    value: str | Mapping[str, Any],
    *,
    accepted_control_plane_pin: object,
    verified_live_seal: VerifiedEAAEFBootstrapGatewayLiveSeal,
    now_ms: int,
) -> Mapping[str, Any]:
    """Fully reverify the canonical public packet at every child boundary."""

    if type(verified_live_seal) is not VerifiedEAAEFBootstrapGatewayLiveSeal:
        raise EAAEFBootstrapGatewayLaunchError(
            "gateway child requires a freshly verified gateway live-seal token"
        )
    live = dict(verified_live_seal)
    live_body = {key: item for key, item in live.items() if key != "verification_cid"}
    if live.get("verification_cid") != _cid(live_body):
        raise EAAEFBootstrapGatewayLaunchError(
            "verified gateway live-seal token is internally inconsistent"
        )

    if isinstance(value, str):
        raw = value.encode("utf-8")
        authority = _parse_canonical_json_object(
            raw,
            noun="gateway launch authority",
        )
    elif isinstance(value, Mapping):
        authority = dict(value)
        _bounded_canonical_bytes(authority, noun="gateway launch authority")
    else:
        raise EAAEFBootstrapGatewayLaunchError(
            "gateway launch authority is not an object"
        )
    if not isinstance(authority, dict) or set(authority) != _LAUNCH_FIELDS:
        raise EAAEFBootstrapGatewayLaunchError(
            "gateway launch authority shape is invalid"
        )
    body = {key: item for key, item in authority.items() if key != "authority_cid"}
    if (
        authority.get("schema") != EAAEF_BOOTSTRAP_GATEWAY_LAUNCH_AUTHORITY_SCHEMA
        or authority.get("board_namespace") != EAAEF_BOARD_NAMESPACE
        or authority.get("authority_cid") != _cid(body)
        or authority.get("operational_capability_cid")
        != (authority.get("operational_capability") or {}).get("capability_cid")
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "gateway launch authority binding is invalid"
        )
    _git(authority.get("source_head"), "source_head")
    _git(authority.get("source_tree"), "source_tree")
    for name in (
        "configuration_root",
        "accepted_control_plane_capsule_id",
        "accepted_control_plane_pin_cid",
        "bootstrap_admission_receipt_cid",
        "configured_board_capsule_cid",
        "live_verification_cid",
        "active_plan_root_cid",
        "frontier_cid",
        "operational_capability_cid",
        "operational_capability_file_sha256",
    ):
        _sha(authority.get(name), name)
    _positive(authority.get("active_plan_revision"), "active_plan_revision")
    if type(accepted_control_plane_pin) is not AgentImplementationControlPlanePin:
        raise EAAEFBootstrapGatewayLaunchError(
            "accepted control-plane pin must be the exact closed type"
        )
    pin = accepted_control_plane_pin.as_dict()
    capability_value = authority.get("operational_capability")
    verified = verify_eaaef_bootstrap_operational_capability(
        capability_value,
        trusted_reviewer_dids=verified_live_seal._trusted_reviewer_dids,
        trusted_authorization_service_reviewer_dids=(
            verified_live_seal._trusted_service_reviewer_dids
        ),
        now_ms=now_ms,
        expected=verified_live_seal._expected_operational_bindings,
        forbidden_reviewer_dids=verified_live_seal._forbidden_reviewer_dids,
    )
    outer_to_nested = {
        "source_head": "source_head",
        "source_tree": "source_tree",
        "configuration_root": "configuration_root",
        "bootstrap_admission_receipt_cid": "bootstrap_admission_receipt_cid",
        "configured_board_capsule_cid": "configured_board_capsule_cid",
        "active_plan_root_cid": "active_plan_root_cid",
        "active_plan_revision": "active_plan_revision",
        "frontier_cid": "frontier_cid",
        "operational_capability_cid": "capability_cid",
    }
    crossed = sorted(
        outer
        for outer, nested in outer_to_nested.items()
        if authority.get(outer) != verified.get(nested)
    )
    live_capability = live.get("bootstrap_operational_capability")
    if (
        crossed
        or not isinstance(live_capability, Mapping)
        or verified["capability_cid"]
        != live["bootstrap_operational_capability_cid"]
        or _canonical_bytes(dict(verified))
        != _canonical_bytes(dict(live_capability))
        or authority["accepted_control_plane_capsule_id"] != pin["capsule_id"]
        or authority["accepted_control_plane_pin_cid"] != _cid(pin)
        or authority["source_head"] != pin["source_head"]
        or authority["source_tree"] != pin["source_tree"]
        or authority["live_verification_cid"]
        != _sha(live["verification_cid"], "verified_live_seal.verification_cid")
        or authority["operational_capability_file_sha256"]
        != _sha(
            live["bootstrap_operational_capability_file_sha256"],
            "verified_live_seal.bootstrap_operational_capability_file_sha256",
        )
        or authority["accepted_control_plane_pin_cid"]
        != live["accepted_control_plane_pin_cid"]
    ):
        raise EAAEFBootstrapGatewayLaunchError(
            "gateway launch authority differs from signed live identities"
            + (":" + ",".join(crossed) if crossed else "")
        )
    result = dict(authority)
    result["operational_capability"] = dict(verified)
    return MappingProxyType(result)


def build_eaaef_bootstrap_gateway_launch_authority(
    live_verification: VerifiedEAAEFBootstrapGatewayLiveSeal,
    *,
    accepted_control_plane_pin: object,
    now_ms: int,
) -> Mapping[str, Any]:
    """Project one path-free public DTO from a fully verified live seal."""

    if type(live_verification) is not VerifiedEAAEFBootstrapGatewayLiveSeal:
        raise EAAEFBootstrapGatewayLaunchError(
            "gateway launch builder requires the exact verified live-seal token"
        )

    capability_value = live_verification.get("bootstrap_operational_capability")
    if not isinstance(capability_value, Mapping):
        raise EAAEFBootstrapGatewayLaunchError(
            "live verification lacks the bootstrap operational capability"
        )
    if type(accepted_control_plane_pin) is not AgentImplementationControlPlanePin:
        raise EAAEFBootstrapGatewayLaunchError(
            "accepted control-plane pin must be the exact closed type"
        )
    pin = accepted_control_plane_pin.as_dict()
    capability = verify_eaaef_bootstrap_operational_capability(
        capability_value,
        trusted_reviewer_dids=live_verification._trusted_reviewer_dids,
        trusted_authorization_service_reviewer_dids=(
            live_verification._trusted_service_reviewer_dids
        ),
        now_ms=now_ms,
        expected=live_verification._expected_operational_bindings,
        forbidden_reviewer_dids=live_verification._forbidden_reviewer_dids,
    )
    active_plan = live_verification.get("active_plan")
    if not isinstance(active_plan, Mapping):
        raise EAAEFBootstrapGatewayLaunchError(
            "live verification lacks an active plan"
        )
    body = {
        "schema": EAAEF_BOOTSTRAP_GATEWAY_LAUNCH_AUTHORITY_SCHEMA,
        "board_namespace": EAAEF_BOARD_NAMESPACE,
        "source_head": str(live_verification.get("source_head") or ""),
        "source_tree": str(live_verification.get("source_tree") or ""),
        "configuration_root": str(
            live_verification.get("configuration_root") or ""
        ),
        "accepted_control_plane_capsule_id": str(pin.get("capsule_id") or ""),
        "accepted_control_plane_pin_cid": _cid(dict(pin)),
        "bootstrap_admission_receipt_cid": str(
            live_verification.get("bootstrap_admission_receipt_cid") or ""
        ),
        "configured_board_capsule_cid": str(
            live_verification.get("configured_board_capsule_cid") or ""
        ),
        "live_verification_cid": str(
            live_verification.get("verification_cid") or ""
        ),
        "active_plan_root_cid": str(active_plan.get("plan_root_cid") or ""),
        "active_plan_revision": active_plan.get("revision"),
        "frontier_cid": str(live_verification.get("frontier_cid") or ""),
        "operational_capability": dict(capability),
        "operational_capability_cid": str(capability.get("capability_cid") or ""),
        "operational_capability_file_sha256": str(
            live_verification.get("bootstrap_operational_capability_file_sha256")
            or ""
        ),
    }
    return parse_eaaef_bootstrap_gateway_launch_authority(
        {**body, "authority_cid": _cid(body)},
        accepted_control_plane_pin=accepted_control_plane_pin,
        verified_live_seal=live_verification,
        now_ms=now_ms,
    )


def canonical_eaaef_bootstrap_gateway_launch_authority_json(
    value: str | Mapping[str, Any],
    *,
    accepted_control_plane_pin: object,
    verified_live_seal: VerifiedEAAEFBootstrapGatewayLiveSeal,
    now_ms: int,
) -> str:
    return _canonical_bytes(
        dict(
            parse_eaaef_bootstrap_gateway_launch_authority(
                value,
                accepted_control_plane_pin=accepted_control_plane_pin,
                verified_live_seal=verified_live_seal,
                now_ms=now_ms,
            )
        )
    ).decode("ascii")


def _authorization_policy_from_mapping(
    value: Mapping[str, Any],
) -> QuackCommandAuthorizationPolicy:
    try:
        policy = QuackCommandAuthorizationPolicy(
            board_namespace=value.get("board_namespace"),
            shard_id=value.get("shard_id"),
            store_id=value.get("store_id"),
            authority_ref_cid=value.get("authority_ref_cid"),
            owner_principal_did=value.get("owner_principal_did"),
            owner_generation=value.get("owner_generation"),
            fence_epoch=value.get("fence_epoch"),
            trusted_approver_dids=frozenset(
                value.get("trusted_approver_dids") or ()
            ),
            authorized_principal_dids=frozenset(
                value.get("authorized_principal_dids") or ()
            ),
            allowed_command_kinds=frozenset(
                value.get("allowed_command_kinds") or ()
            ),
            maximum_authorization_lifetime_ms=value.get(
                "maximum_authorization_lifetime_ms"
            ),
        )
    except (QuackCommandAuthorizationError, TypeError, ValueError) as exc:
        raise EAAEFCommandAuthorizationServiceError(
            "authorization service policy is invalid"
        ) from exc
    if dict(policy.to_dict()) != dict(value):
        raise EAAEFCommandAuthorizationServiceError(
            "authorization service policy is not canonical"
        )
    return policy


_VERIFIED_CLIENT_FACTORY_TOKEN = object()


class EAAEFCommandAuthorizationServiceClient:
    """Closed client for a separately deployed independent command signer.

    Instances can only be constructed from the signed operational capability.
    The complete operational and nested service signatures, trust roots,
    lifetimes, and expected identities are rechecked before every request.
    """

    __slots__ = (
        "_operational_capability",
        "_trusted_reviewer_dids",
        "_trusted_service_reviewer_dids",
        "_expected",
        "_forbidden_reviewer_dids",
        "_clock_ms",
        "_monotonic_ms",
        "_operational_capability_cid",
        "_service_capability_cid",
    )

    def __init__(
        self,
        factory_token: object,
        *,
        operational_capability: Mapping[str, Any],
        trusted_reviewer_dids: Sequence[str],
        trusted_authorization_service_reviewer_dids: Sequence[str],
        expected: Mapping[str, Any],
        forbidden_reviewer_dids: Sequence[str],
        clock_ms: Any,
        monotonic_ms: Any,
    ) -> None:
        if factory_token is not _VERIFIED_CLIENT_FACTORY_TOKEN:
            raise EAAEFCommandAuthorizationServiceError(
                "authorization service client requires the signed factory"
            )
        if not callable(clock_ms) or not callable(monotonic_ms):
            raise EAAEFCommandAuthorizationServiceError(
                "authorization service client requires a clock"
            )
        # Round-trip through canonical JSON so later caller mutation cannot
        # change the capability that the client re-verifies.
        self._operational_capability = json.loads(
            _canonical_bytes(dict(operational_capability)).decode("ascii")
        )
        self._trusted_reviewer_dids = tuple(trusted_reviewer_dids)
        self._trusted_service_reviewer_dids = tuple(
            trusted_authorization_service_reviewer_dids
        )
        if not isinstance(expected, Mapping) or set(expected) != (
            _EXPECTED_OPERATIONAL_BINDING_FIELDS
        ):
            raise EAAEFCommandAuthorizationServiceError(
                "authorization service client expected bindings are absent"
            )
        self._expected = dict(expected)
        self._forbidden_reviewer_dids = tuple(forbidden_reviewer_dids)
        self._clock_ms = clock_ms
        self._monotonic_ms = monotonic_ms
        verified, _policy = self._reverify()
        self._operational_capability_cid = str(verified["capability_cid"])
        self._service_capability_cid = str(
            verified["command_authorization_service"]["capability_cid"]
        )

    @classmethod
    def from_signed_operational_capability(
        cls,
        *,
        operational_capability: Mapping[str, Any],
        trusted_reviewer_dids: Sequence[str],
        trusted_authorization_service_reviewer_dids: Sequence[str],
        clock_ms: Any,
        expected: Mapping[str, Any],
        forbidden_reviewer_dids: Sequence[str] = (),
        monotonic_ms: Any = lambda: time.monotonic_ns() // 1_000_000,
    ) -> EAAEFCommandAuthorizationServiceClient:
        """Construct only after full signed operational-capability review."""

        return cls(
            _VERIFIED_CLIENT_FACTORY_TOKEN,
            operational_capability=operational_capability,
            trusted_reviewer_dids=trusted_reviewer_dids,
            trusted_authorization_service_reviewer_dids=(
                trusted_authorization_service_reviewer_dids
            ),
            expected=expected,
            forbidden_reviewer_dids=forbidden_reviewer_dids,
            clock_ms=clock_ms,
            monotonic_ms=monotonic_ms,
        )

    def _reverify(
        self,
    ) -> tuple[Mapping[str, Any], QuackCommandAuthorizationPolicy]:
        try:
            now_ms = int(self._clock_ms())
        except (TypeError, ValueError) as exc:
            raise EAAEFCommandAuthorizationServiceError(
                "authorization service clock is invalid"
            ) from exc
        try:
            verified = verify_eaaef_bootstrap_operational_capability(
                self._operational_capability,
                trusted_reviewer_dids=self._trusted_reviewer_dids,
                trusted_authorization_service_reviewer_dids=(
                    self._trusted_service_reviewer_dids
                ),
                now_ms=now_ms,
                expected=self._expected,
                forbidden_reviewer_dids=self._forbidden_reviewer_dids,
            )
        except EAAEFBootstrapGatewayLaunchError as exc:
            raise EAAEFCommandAuthorizationServiceError(
                "authorization service capability re-verification failed"
            ) from exc
        policy_value = verified.get("authorization_policy")
        if not isinstance(policy_value, Mapping):
            raise EAAEFCommandAuthorizationServiceError(
                "authorization service policy is absent"
            )
        policy = _authorization_policy_from_mapping(policy_value)
        if hasattr(self, "_operational_capability_cid") and (
            verified.get("capability_cid") != self._operational_capability_cid
            or verified["command_authorization_service"].get("capability_cid")
            != self._service_capability_cid
        ):
            raise EAAEFCommandAuthorizationServiceError(
                "authorization service capability pin changed"
            )
        return verified, policy

    @property
    def capability_cid(self) -> str:
        return self._service_capability_cid

    def _remaining_io_timeout_seconds(self, deadline_ms: int) -> float:
        remaining_ms = int(deadline_ms) - int(self._monotonic_ms())
        if remaining_ms <= 0:
            raise EAAEFCommandAuthorizationServiceError(
                "authorization service absolute request deadline expired"
            )
        return remaining_ms / 1000.0

    def _recv_exact(
        self,
        connection: socket.socket,
        count: int,
        *,
        deadline_ms: int,
    ) -> bytes:
        chunks: list[bytes] = []
        remaining = count
        while remaining:
            connection.settimeout(
                self._remaining_io_timeout_seconds(deadline_ms)
            )
            chunk = connection.recv(remaining)
            if not chunk:
                raise EAAEFCommandAuthorizationServiceError(
                    "authorization service closed a partial response"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def authorize(self, intent: Mapping[str, Any]) -> AuthorizedStateCommand:
        """Request and verify one fresh command envelope; never sign locally."""

        if not isinstance(intent, Mapping):
            raise EAAEFCommandAuthorizationServiceError(
                "command authorization intent is not an object"
            )
        try:
            intent_snapshot = json.loads(
                _bounded_canonical_bytes(
                    dict(intent),
                    noun="command authorization intent",
                ).decode("ascii")
            )
        except EAAEFBootstrapGatewayLaunchError as exc:
            raise EAAEFCommandAuthorizationServiceError(
                "command authorization intent is not canonical and bounded"
            ) from exc
        verified, policy = self._reverify()
        service = verified["command_authorization_service"]
        request_id = "authorization-request:" + secrets.token_urlsafe(24)
        request_nonce = "authorization-nonce:" + secrets.token_urlsafe(24)
        request_body = {
            "schema": EAAEF_COMMAND_AUTHORIZATION_REQUEST_SCHEMA,
            "request_id": request_id,
            "request_nonce": request_nonce,
            "service_capability_cid": self.capability_cid,
            "operational_capability_cid": verified["capability_cid"],
            "service_capability_issuance_nonce": service["issuance_nonce"],
            "operational_capability_issuance_nonce": verified["issuance_nonce"],
            "authorization_policy_cid": policy.policy_cid,
            "client_principal_did": service[
                "authorized_client_principal_did"
            ],
            "intent": intent_snapshot,
        }
        request = {
            **request_body,
            "request_cid": _cid(request_body),
        }
        payload = _canonical_bytes(request)
        maximum_request = int(service["maximum_request_bytes"])
        if len(payload) > maximum_request:
            raise EAAEFCommandAuthorizationServiceError(
                "authorization request exceeds the signed service bound"
            )
        endpoint = _unix_endpoint(service["endpoint"])
        path = Path(endpoint.removeprefix("unix:"))
        expected_uid = int(service["expected_server_uid"])
        before = _safe_unix_socket_metadata(path, expected_uid=expected_uid)
        response_limit = int(service["maximum_response_bytes"])
        wall_now_ms = int(self._clock_ms())
        wall_remaining_ms = min(
            int(service["expires_at_ms"]),
            int(verified["expires_at_ms"]),
        ) - wall_now_ms
        if wall_remaining_ms <= 0:
            raise EAAEFCommandAuthorizationServiceError(
                "authorization service capability expired before transport"
            )
        deadline_ms = int(self._monotonic_ms()) + min(
            int(service["request_timeout_ms"]),
            wall_remaining_ms,
        )
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(
                    self._remaining_io_timeout_seconds(deadline_ms)
                )
                connection.connect(str(path))
                after = _safe_unix_socket_metadata(path, expected_uid=expected_uid)
                if (
                    before.st_dev,
                    before.st_ino,
                    before.st_mode,
                    before.st_uid,
                    before.st_nlink,
                ) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_mode,
                    after.st_uid,
                    after.st_nlink,
                ):
                    raise EAAEFCommandAuthorizationServiceError(
                        "authorization service socket changed during connect"
                    )
                peer_credential = getattr(socket, "SO_PEERCRED", None)
                if peer_credential is None:
                    raise EAAEFCommandAuthorizationServiceError(
                        "authorization service peer credentials are unavailable"
                    )
                peer_raw = connection.getsockopt(
                    socket.SOL_SOCKET,
                    peer_credential,
                    struct.calcsize("3i"),
                )
                if len(peer_raw) != struct.calcsize("3i"):
                    raise EAAEFCommandAuthorizationServiceError(
                        "authorization service peer credentials are malformed"
                    )
                peer_pid, peer_uid, _peer_gid = struct.unpack("3i", peer_raw)
                if (
                    peer_pid != int(service["expected_server_pid"])
                    or peer_uid != expected_uid
                    or _linux_process_start_time_ticks(peer_pid)
                    != int(service["expected_server_process_start_time_ticks"])
                ):
                    raise EAAEFCommandAuthorizationServiceError(
                        "authorization service peer process identity differs"
                    )
                connection.settimeout(
                    self._remaining_io_timeout_seconds(deadline_ms)
                )
                connection.sendall(struct.pack("!I", len(payload)) + payload)
                length = struct.unpack(
                    "!I",
                    self._recv_exact(connection, 4, deadline_ms=deadline_ms),
                )[0]
                if not 1 <= length <= response_limit:
                    raise EAAEFCommandAuthorizationServiceError(
                        "authorization response exceeds the signed service bound"
                    )
                response_raw = self._recv_exact(
                    connection,
                    length,
                    deadline_ms=deadline_ms,
                )
        except EAAEFCommandAuthorizationServiceError:
            raise
        except OSError as exc:
            raise EAAEFCommandAuthorizationServiceError(
                "authorization service transport failed"
            ) from exc
        try:
            response = _parse_canonical_json_object(
                response_raw,
                noun="authorization service response",
            )
            envelope = AuthorizedStateCommand.from_dict(response)
            verified_after, policy_after = self._reverify()
            service_after = verified_after["command_authorization_service"]
            if policy_after.policy_cid != policy.policy_cid:
                raise EAAEFCommandAuthorizationServiceError(
                    "authorization policy changed during request"
                )
            verify_authorized_state_command(
                envelope,
                policy=policy_after,
                now_ms=int(self._clock_ms()),
            )
        except EAAEFCommandAuthorizationServiceError:
            raise
        except (
            EAAEFBootstrapGatewayLaunchError,
            UnicodeError,
            QuackCommandAuthorizationError,
            TypeError,
            ValueError,
        ) as exc:
            raise EAAEFCommandAuthorizationServiceError(
                "authorization service returned an invalid signed command"
            ) from exc
        try:
            verify_eaaef_bootstrap_operation_submission(
                envelope,
                request_body["intent"],
                verified_capability=verified_after,
                authorization_policy=policy_after,
                now_ms=int(self._clock_ms()),
            )
        except EAAEFBootstrapGatewayLaunchError as exc:
            if "intent" in str(exc):
                message = "authorization service returned a malformed operation intent"
            else:
                message = (
                    "authorization service response is not request/capability bound"
                )
            raise EAAEFCommandAuthorizationServiceError(message) from exc
        from ipfs_accelerate_py.agent_supervisor.task_sources.quack_daemon_gateway import (
            QuackDaemonGatewayError,
            quack_daemon_operation_intent_from_envelope,
        )

        try:
            transported_intent = quack_daemon_operation_intent_from_envelope(
                envelope
            )
        except QuackDaemonGatewayError as exc:
            raise EAAEFCommandAuthorizationServiceError(
                "authorization service returned a malformed operation intent"
            ) from exc
        parameters = dict(envelope.command.parameters)
        if (
            dict(transported_intent) != request_body["intent"]
            or envelope.request_id != request_id
            or envelope.one_use_nonce != request_nonce
            or parameters.get("authorization_request_cid")
            != request["request_cid"]
            or envelope.issued_at_ms
            < max(
                int(verified_after["issued_at_ms"]),
                int(service_after["issued_at_ms"]),
            )
            or envelope.expires_at_ms
            > min(
                int(verified_after["expires_at_ms"]),
                int(service_after["expires_at_ms"]),
            )
            or envelope.deadline_ms
            > min(
                int(verified_after["expires_at_ms"]),
                int(service_after["expires_at_ms"]),
            )
        ):
            raise EAAEFCommandAuthorizationServiceError(
                "authorization service response is not request/capability bound"
            )
        return envelope


__all__ = (
    "EAAEF_BOOTSTRAP_GATEWAY_BINDING_INTERFACE",
    "EAAEF_BOOTSTRAP_GATEWAY_BINDING_SCHEMA",
    "EAAEF_BOOTSTRAP_GATEWAY_LAUNCH_AUTHORITY_FLAG",
    "EAAEF_BOOTSTRAP_GATEWAY_LAUNCH_AUTHORITY_SCHEMA",
    "EAAEF_BOOTSTRAP_OPERATIONAL_CAPABILITY_INTERFACE",
    "EAAEF_BOOTSTRAP_OPERATIONAL_CAPABILITY_PATH_TEMPLATE",
    "EAAEF_BOOTSTRAP_OPERATIONAL_CAPABILITY_REVIEW_ROLE",
    "EAAEF_BOOTSTRAP_OPERATIONAL_CAPABILITY_SCHEMA",
    "EAAEF_COMMAND_AUTHORIZATION_REQUEST_SCHEMA",
    "EAAEF_COMMAND_AUTHORIZATION_SERVICE_CAPABILITY_SCHEMA",
    "EAAEF_COMMAND_AUTHORIZATION_SERVICE_INTERFACE",
    "EAAEF_COMMAND_AUTHORIZATION_SERVICE_REVIEW_ROLE",
    "EAAEFBootstrapGatewayLaunchError",
    "EAAEFCommandAuthorizationServiceClient",
    "EAAEFCommandAuthorizationServiceError",
    "VerifiedEAAEFBootstrapOperationalCapability",
    "VerifiedEAAEFBootstrapGatewayLiveSeal",
    "build_eaaef_bootstrap_gateway_launch_authority",
    "canonical_eaaef_bootstrap_gateway_launch_authority_json",
    "eaaef_bootstrap_operational_capability_relative_path",
    "eaaef_bootstrap_gateway_binding",
    "eaaef_bootstrap_gateway_binding_cid",
    "load_eaaef_bootstrap_operational_capability",
    "parse_eaaef_bootstrap_gateway_launch_authority",
    "seal_eaaef_bootstrap_operational_capability",
    "seal_eaaef_command_authorization_service_capability",
    "verify_eaaef_bootstrap_operational_capability",
    "verify_eaaef_bootstrap_gateway_live_seal",
    "verify_eaaef_bootstrap_operation_submission",
    "verify_eaaef_command_authorization_service_capability",
)
