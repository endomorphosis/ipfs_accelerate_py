"""Closed typed command transport for the exclusive Quack state owner.

This module is deliberately smaller than a general RPC or SQL service.  A
client can name an operation from the server's immutable catalog and bind
JSON scalar parameters.  SQL text, paths, credentials, callbacks, and catalog
extensions are never accepted over the wire.

The transport is a private Unix-domain socket owned by the same process that
owns ``control.duckdb``.  A command transaction holds the owner's execution
lock from BEGIN through COMMIT/ROLLBACK, so a domain mutation, its event, its
outbox row, its idempotency receipt, and the generation advance share one
DuckDB transaction.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
import socket
import stat
import struct
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ..merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
    owner_liveness,
)
from .control_plane_contracts import StateCommand, canonical_json_bytes, content_identity
from .database_task_source import TaskSourceIntegrityError
from .task_execution_route_policy import TaskExecutionRouteBinding

TYPED_STATE_OWNER_INTERFACE: Final = "TypedStateOwnerCommandGateway@1"
_UTC: Final = timezone.utc  # noqa: UP017 - Python 3.8 compatibility.
TYPED_STATE_OWNER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/typed-state-owner-command@1"
)
TYPED_STATE_OWNER_SOCKET_ENV: Final = "IPFS_ACCELERATE_AGENT_STATE_OWNER_SOCKET"
TYPED_STATE_OWNER_TOKEN_ENV: Final = "IPFS_ACCELERATE_AGENT_QUACK_TOKEN"
TYPED_STATE_OWNER_SOCKET_FILENAME: Final = "typed-state-owner.sock"
TYPED_STATE_OWNER_TOKEN_FILENAME: Final = "typed-state-owner.token"
TYPED_RETRY_COOLDOWN_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/typed-retry-cooldown@1"
)
TYPED_DATABASE_CLAIM_PROCESS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/typed-database-claim-process@1"
)
TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/typed-database-claim-reservation@1"
)
TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/typed-database-attempt-admission@1"
)
TYPED_DATABASE_CLAIM_RECOVERY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/typed-database-claim-recovery@1"
)
TYPED_DATABASE_CLAIM_RECOVERY_OPERATION: Final = (
    "database_claim_lost_sidecar_recovery"
)
TYPED_DATABASE_CLAIM_RECOVERY_COMMAND: Final = (
    "task.claim.reservation.recover"
)
TYPED_DATABASE_CLAIM_RECOVERY_REASON: Final = (
    "database_claim_lost_sidecar_dead_process"
)
TYPED_DATABASE_STRICT_RESUME_REJECTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "typed-database-strict-resume-rejection@1"
)
TYPED_DATABASE_STRICT_RESUME_REQUEUE_OPERATION: Final = (
    "database_strict_resume_requeue"
)
TYPED_DATABASE_STRICT_RESUME_QUARANTINE_OPERATION: Final = (
    "database_strict_resume_quarantine"
)
TYPED_RETRYING_RECEIPT_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        "database_portal_retry",
        "database_portal_validation_retry",
        "database_portal_validation_retry_recovery",
        "database_portal_protected_path_retry_recovery",
        "database_portal_external_protected_checkout_retry_recovery",
        "database_portal_inflight_process_retry_recovery",
        "database_portal_validation_retry_seed_conflict_retry_recovery",
        "database_portal_leftover_wait_deferral_budget_retry_recovery",
        "database_portal_pooled_worktree_create_retry_recovery",
        "database_post_merge_declared_outputs_repair_recovery",
        "database_post_merge_declared_outputs_requalification_recovery",
        "database_portal_inflight_deferral_unstall",
        TYPED_DATABASE_CLAIM_RECOVERY_OPERATION,
    }
)
# Linux permits 107 pathname bytes in ``sockaddr_un.sun_path`` while other
# supported Unix platforms can be slightly smaller.  Keep a little headroom
# for the trailing NUL and fail over before ``bind(2)`` becomes platform
# dependent.
_SAFE_UNIX_SOCKET_PATH_BYTES: Final = 100
_COMPACT_SOCKET_ROOT_PREFIX: Final = "ipfs-accelerate-typed-owner"
MAX_FRAME_BYTES: Final = 16 * 1024 * 1024
MAX_PARAMETER_COUNT: Final = 512
MAX_ROW_COUNT: Final = 4096
MIN_GRANT_TTL_SECONDS: Final = 1.0
MAX_GRANT_TTL_SECONDS: Final = 86_400.0
DEFAULT_GRANT_TTL_SECONDS: Final = 3_600.0
MAX_REMOTE_EVENT_WAIT_SECONDS: Final = 60.0
TYPED_TASK_STATUS_VOCABULARY: Final[frozenset[str]] = frozenset(
    {
        "ready",
        "todo",
        "queued",
        "pending",
        "proposed",
        "admitted",
        "retrying",
        "claimed",
        "in_progress",
        "running",
        "blocked",
        "completed",
        "complete",
        "done",
        "skipped",
        "cancelled",
        "canceled",
        "failed",
        "quarantined",
        "rejected",
    }
)
STATUS_BOOTSTRAP_CLIENT_ID: Final = "casf-bootstrap-operator:typed-status"
STATUS_BOOTSTRAP_GRANT_TTL_SECONDS: Final = 60.0
STATUS_BOOTSTRAP_ALLOWED_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        "whoami_metadata",
        "load_store_generation",
        "list_tasks_page",
        "list_ready_task_aliases",
        "count_tasks",
        "max_event_watermark",
        "casf_select_supervisor_bootstrap_health",
    }
)
_STATUS_BOOTSTRAP_ENTITY_SCOPE_NAMES: Final[tuple[str, ...]] = (
    "supervisor_id",
    "subscription_id",
    "consumer_id",
)
_STATUS_OWNER_PLAN_READS: Final[frozenset[str]] = frozenset(
    {
        "list_tasks_page",
        "list_ready_task_aliases",
        "count_tasks",
        "max_event_watermark",
    }
)
_EAAEF_COMMAND_SERVICE_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        "eaaef.command.submit",
        "eaaef.command.lookup",
    }
)
_EAAEF_PLAN_R2_SERVICE_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        "plan_r2.prepare",
        "plan_r2.apply",
        "plan_r2.observe",
    }
)
_EVENT_WAIT_SERVICE_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        "event.wait",
        "event.wait.cancel",
        "event.wait.clear_cancellation",
    }
)
_ISSUABLE_SERVICE_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        *_EVENT_WAIT_SERVICE_OPERATIONS,
        *_EAAEF_COMMAND_SERVICE_OPERATIONS,
        *_EAAEF_PLAN_R2_SERVICE_OPERATIONS,
    }
)
_RLOCK_TYPE: Final = type(threading.RLock())
# Exact observed operation surfaces for the bounded first-tranche child.
# These are capability allowlists, not convenience catalogs: adding a named
# query or mutation requires an explicit owner-side review and focused test.
SUPERVISOR_RUNTIME_CHILD_ALLOWED_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        "whoami_metadata",
        "load_store_generation",
        "txn_load_generation",
        "txn_lookup_idempotency",
        "txn_advance_store_revision",
        "txn_record_idempotency",
        "casf_select_supervisor",
        "casf_select_current_supervisor_runtime",
        "casf_select_latest_supervisor_runtime_revision",
        "casf_count_supervisor_active_attempts",
        "casf_count_supervisor_active_effects",
        "casf_count_supervisor_active_slots",
        "casf_insert_process_birth_attestation",
        "casf_insert_supervisor_runtime_lease",
        "casf_supersede_supervisor_runtime_lease",
        "casf_update_supervisor_process_birth",
        "casf_update_supervisor_lifecycle",
        "casf_seed_global_head",
        "casf_advance_global_head",
        "casf_seed_stream_head",
        "casf_advance_stream_head",
        "casf_insert_domain_event",
        "casf_insert_changed_fact",
        "casf_insert_outbox",
    }
)
SUPERVISOR_EVENT_CHILD_ALLOWED_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        "whoami_metadata",
        "load_store_generation",
        "txn_load_generation",
        "txn_lookup_idempotency",
        "txn_advance_store_revision",
        "txn_record_idempotency",
        "casf_select_subscription",
        "casf_select_subscription_selectors",
        "casf_select_consumer_cursor",
        "casf_list_routed_wait_events",
        "casf_list_deliverable_queue",
        "casf_select_delivery_queue",
        "casf_select_outbox_for_delivery",
        "casf_select_queue_for_attempt",
        "casf_insert_delivery_attempt",
        "casf_mark_queue_delivered",
        "casf_select_event_for_ack",
        "casf_select_delivery_for_ack",
        "casf_mark_delivery_acknowledged",
        "casf_mark_queue_acknowledged",
        "casf_reset_subscription_failures",
        "casf_insert_event_acknowledgement",
        "casf_advance_consumer_cursor",
        *_EVENT_WAIT_SERVICE_OPERATIONS,
    }
)


class TypedStateOwnerError(RuntimeError):
    """Base typed owner-command failure."""


class TypedStateOwnerProtocolError(TypedStateOwnerError):
    """A request or response violated the closed wire contract."""


class TypedStateOwnerAuthorizationError(TypedStateOwnerError):
    """Authentication or server-side command admission failed."""


class TypedStateOwnerRemoteError(TypedStateOwnerError):
    """The owner rejected or failed one named operation."""

    def __init__(self, error_code: str, error_type: str = "") -> None:
        code = str(error_code or "owner_operation_failed")
        kind = str(error_type or "remote_error")
        # The remote exception message is intentionally never transported: a
        # DuckDB driver may echo SQL text or credential-bearing ATTACH text.
        super().__init__(f"typed state-owner {code} ({kind})")
        self.error_code = code
        self.error_type = kind


@dataclass(frozen=True)
class OwnerOperation:
    """One immutable server-owned SQL operation."""

    name: str
    sql: str
    parameter_count: int
    mutation: bool
    bootstrap_only: bool = False
    parameter_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        sql = str(self.sql or "").strip()
        if not name or not name.replace("_", "a").isalnum():
            raise TypedStateOwnerProtocolError("owner operation name is invalid")
        if not sql or ";" in sql or "\x00" in sql:
            raise TypedStateOwnerProtocolError("owner operation SQL is not closed")
        count = int(self.parameter_count)
        if count < 0 or count > MAX_PARAMETER_COUNT or sql.count("?") != count:
            raise TypedStateOwnerProtocolError(
                f"owner operation {name} has an invalid parameter contract"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "sql", sql)
        object.__setattr__(self, "parameter_count", count)
        object.__setattr__(self, "mutation", bool(self.mutation))
        object.__setattr__(self, "bootstrap_only", bool(self.bootstrap_only))
        names = tuple(str(item or "").strip() for item in self.parameter_names)
        if names and (
            len(names) != count
            or len(set(names)) != len(names)
            or any(not item.replace("_", "a").isalnum() for item in names)
        ):
            raise TypedStateOwnerProtocolError(
                f"owner operation {name} has invalid parameter names"
            )
        object.__setattr__(self, "parameter_names", names)

    def public_dict(self) -> dict[str, Any]:
        """Return a SQL-free catalog projection."""

        return {
            "name": self.name,
            "parameter_count": self.parameter_count,
            "parameter_names": list(self.parameter_names),
            "mutation": self.mutation,
            "bootstrap_only": self.bootstrap_only,
            "sql_digest": "sha256:"
            + hashlib.sha256(_normalize_sql(self.sql).encode("utf-8")).hexdigest(),
        }


@dataclass(frozen=True)
class OwnerClientGrant:
    """Server-issued capability for one exact client/process/scope."""

    grant_id: str
    client_id: str
    process_birth_id: str
    allowed_operations: frozenset[str]
    allowed_command_operations: frozenset[str]
    tenant_id: str = ""
    federation_id: str = ""
    entity_scopes: tuple[tuple[str, str], ...] = ()
    authority_profile: str = ""
    peer_pid: int = 0
    peer_uid: int = -1
    peer_start_time_ticks: int = 0
    issued_at: int = 0
    expires_at: int = 0

    def __post_init__(self) -> None:
        for field_name in ("grant_id", "client_id"):
            value = str(getattr(self, field_name) or "").strip()
            if not value or len(value) > 256:
                raise TypedStateOwnerAuthorizationError(
                    f"owner grant {field_name} is invalid"
                )
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "process_birth_id", str(self.process_birth_id or "").strip())
        object.__setattr__(
            self,
            "allowed_operations",
            frozenset(str(item) for item in self.allowed_operations),
        )
        object.__setattr__(
            self,
            "allowed_command_operations",
            frozenset(str(item) for item in self.allowed_command_operations),
        )
        object.__setattr__(self, "tenant_id", str(self.tenant_id or "").strip())
        object.__setattr__(self, "federation_id", str(self.federation_id or "").strip())
        scopes = tuple(
            (str(name or "").strip(), str(value or "").strip())
            for name, value in self.entity_scopes
        )
        permitted_scope_names = {
            "supervisor_id",
            "subagent_id",
            "repository_id",
            "tree_id",
            "task_id",
            "task_cid",
            "subscription_id",
            "consumer_id",
            "event_id",
        }
        if (
            len(scopes) > len(permitted_scope_names)
            or len({name for name, _value in scopes}) != len(scopes)
            or any(name not in permitted_scope_names or not value for name, value in scopes)
        ):
            raise TypedStateOwnerAuthorizationError(
                "owner grant entity scope is invalid"
            )
        object.__setattr__(self, "entity_scopes", tuple(sorted(scopes)))
        authority_profile = str(self.authority_profile or "").strip()
        if authority_profile not in {"", "dedicated_store_status_portfolio"}:
            raise TypedStateOwnerAuthorizationError(
                "owner grant authority profile is invalid"
            )
        object.__setattr__(self, "authority_profile", authority_profile)
        peer_pid = int(self.peer_pid)
        peer_uid = int(self.peer_uid)
        peer_start = int(self.peer_start_time_ticks)
        if isinstance(self.issued_at, bool) or isinstance(self.expires_at, bool):
            raise TypedStateOwnerAuthorizationError(
                "owner grant expiry is outside the closed lifetime bound"
            )
        issued_at = int(self.issued_at)
        expires_at = int(self.expires_at)
        if peer_pid < 1 or peer_uid < 0 or peer_start < 0:
            raise TypedStateOwnerAuthorizationError(
                "owner grant requires a kernel-verifiable peer process"
            )
        if (
            issued_at < 1
            or expires_at <= issued_at
            or expires_at - issued_at > int(MAX_GRANT_TTL_SECONDS * 1_000)
        ):
            raise TypedStateOwnerAuthorizationError(
                "owner grant expiry is outside the closed lifetime bound"
            )
        object.__setattr__(self, "peer_pid", peer_pid)
        object.__setattr__(self, "peer_uid", peer_uid)
        object.__setattr__(self, "peer_start_time_ticks", peer_start)
        object.__setattr__(self, "issued_at", issued_at)
        object.__setattr__(self, "expires_at", expires_at)

    def public_dict(self) -> dict[str, Any]:
        return {
            "grant_id": self.grant_id,
            "client_id": self.client_id,
            "process_birth_id": self.process_birth_id,
            "allowed_operations": sorted(self.allowed_operations),
            "allowed_command_operations": sorted(self.allowed_command_operations),
            "tenant_id": self.tenant_id,
            "federation_id": self.federation_id,
            "entity_scopes": dict(self.entity_scopes),
            "authority_profile": self.authority_profile,
            "peer_pid": self.peer_pid,
            "peer_uid": self.peer_uid,
            "peer_start_time_ticks": self.peer_start_time_ticks,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }


def _process_start_time_ticks(pid: int) -> int:
    """Read one PID-reuse-resistant start time from procfs.

    The typed transport is Linux/Unix-domain-only today.  Missing procfs is a
    typed capability blocker instead of weakening a grant to caller-asserted
    process identity.
    """

    selected = int(pid)
    if selected < 1:
        raise TypedStateOwnerAuthorizationError("grant peer PID is invalid")
    try:
        # Field 22 follows the parenthesized comm value.  Split at the final
        # ')' because a process name may itself contain spaces or parentheses.
        stat_text = Path(f"/proc/{selected}/stat").read_text(encoding="utf-8")
        suffix = stat_text.rsplit(")", 1)[1].strip().split()
        start_time = int(suffix[19])
    except (FileNotFoundError, IndexError, OSError, ValueError) as exc:
        raise TypedStateOwnerAuthorizationError(
            "kernel peer process identity is unavailable"
        ) from exc
    # Linux PID 1 may legitimately report starttime 0 when it is the boot
    # process; the tuple remains kernel-bound and PID-reuse-resistant.
    if start_time < 0:
        raise TypedStateOwnerAuthorizationError(
            "kernel peer process start identity is invalid"
        )
    return start_time


def _process_runtime_facts(pid: int) -> tuple[int, int, str]:
    """Read the kernel facts accepted in a supervisor runtime attestation."""

    selected = int(pid)
    try:
        stat_text = Path(f"/proc/{selected}/stat").read_text(encoding="utf-8")
        suffix = stat_text.rsplit(")", 1)[1].strip().split()
        parent_pid = int(suffix[1])
        start_time = int(suffix[19])
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
    except (FileNotFoundError, IndexError, OSError, ValueError) as exc:
        raise TypedStateOwnerAuthorizationError(
            "kernel supervisor process facts are unavailable"
        ) from exc
    if parent_pid < 0 or start_time < 0 or not boot_id or len(boot_id) > 128:
        raise TypedStateOwnerAuthorizationError(
            "kernel supervisor process facts are invalid"
        )
    return start_time, parent_pid, boot_id


def _process_birth_content_id(
    pid: int,
    start_time_ticks: int,
    boot_id: str,
    parent_pid: int,
) -> str:
    """Return the content identity used by launcher-attested process grants."""

    material = (
        f"{int(pid)}:{int(start_time_ticks)}:"
        f"{str(boot_id or '')}:{int(parent_pid)}"
    )
    return f"birth:{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"


def _claim_process_attestation(grant: OwnerClientGrant) -> dict[str, Any]:
    """Resolve the exact kernel/grant identity admitted for a database claim."""

    start_time, parent_pid, boot_id = _process_runtime_facts(grant.peer_pid)
    expected_birth_id = _process_birth_content_id(
        grant.peer_pid,
        start_time,
        boot_id,
        parent_pid,
    )
    if (
        start_time != grant.peer_start_time_ticks
        or grant.process_birth_id != expected_birth_id
    ):
        raise TypedStateOwnerAuthorizationError(
            "database claim process grant differs from kernel birth identity"
        )
    return {
        "schema": TYPED_DATABASE_CLAIM_PROCESS_SCHEMA,
        "grant_id": grant.grant_id,
        "client_id": grant.client_id,
        "process_birth_id": grant.process_birth_id,
        "pid": grant.peer_pid,
        "uid": grant.peer_uid,
        "start_time_ticks": start_time,
        "boot_id": boot_id,
        "parent_pid": parent_pid,
    }


def _kernel_peer_identity(channel: socket.socket) -> tuple[int, int, int]:
    """Return PID, UID, and PID start time proven by the Unix socket kernel."""

    peer_option = getattr(socket, "SO_PEERCRED", None)
    if peer_option is None:
        raise TypedStateOwnerAuthorizationError(
            "kernel peer credentials are unavailable for typed owner transport"
        )
    try:
        size = struct.calcsize("3i")
        raw = channel.getsockopt(socket.SOL_SOCKET, peer_option, size)
        peer_pid, peer_uid, _peer_gid = struct.unpack("3i", raw)
    except (OSError, struct.error) as exc:
        raise TypedStateOwnerAuthorizationError(
            "kernel peer credentials could not be verified"
        ) from exc
    return peer_pid, peer_uid, _process_start_time_ticks(peer_pid)


def _normalize_sql(sql: str) -> str:
    return " ".join(str(sql or "").strip().split())


def _strict_scalar_equal(observed: Any, expected: Any) -> bool:
    """Compare receipt scalars without Python's bool/int equivalence."""

    return type(observed) is type(expected) and observed == expected


def _validated_database_claim_identity(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact claim/fence tuple carried by a shared receipt."""

    text_fields = (
        "claim_id",
        "attempt_id",
        "lease_id",
        "owner_session_id",
    )
    integer_fields = (
        "attempt_number",
        "fencing_token",
        "fence_epoch",
    )
    identity: dict[str, Any] = {}
    for name in text_fields:
        value = receipt.get(name)
        if (
            type(value) is not str
            or not value.strip()
            or len(value.encode("utf-8")) > 1_024
            or any(marker in value for marker in ("\x00", "\n", "\r"))
        ):
            raise TypedStateOwnerAuthorizationError(
                f"database claim receipt {name} is invalid"
            )
        identity[name] = value
    for name in integer_fields:
        value = receipt.get(name)
        if type(value) is not int or value < 1:
            raise TypedStateOwnerAuthorizationError(
                f"database claim receipt {name} is invalid"
            )
        identity[name] = value
    return identity


def typed_database_strict_resume_rejection_receipt_id(
    receipt: Mapping[str, Any],
) -> str:
    """Return the content identity of one strict-resume rejection receipt."""

    body = dict(receipt)
    body.pop("receipt_id", None)
    return "sha256:" + hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _validated_database_strict_resume_rejection_receipt(
    receipt: Any,
) -> dict[str, Any]:
    """Validate the closed scheduling receipt used as a durable attempt floor."""

    required = {
        "schema",
        "operation",
        "task_cid",
        "claim_id",
        "attempt_id",
        "attempt_number",
        "lease_id",
        "owner_session_id",
        "fencing_token",
        "fence_epoch",
        "rejected_task_alias",
        "rejected_task_revision",
        "provider_phase_committed",
        "provider_invocation_receipt_present",
        "max_task_attempts",
        "attempt_budget_exhausted",
        "task_shard_count",
        "task_shard_index",
        "reasons",
        "shared_claim_binding",
        "execution_route_binding",
        "execution_route_policy_id",
        "execution_route_origin_revision",
        "receipt_id",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required:
        raise TaskSourceIntegrityError(
            "typed strict-resume rejection receipt has unknown or missing fields"
        )
    values = dict(receipt)
    operation = values.get("operation")
    if (
        values.get("schema")
        != TYPED_DATABASE_STRICT_RESUME_REJECTION_SCHEMA
        or operation
        not in {
            TYPED_DATABASE_STRICT_RESUME_REQUEUE_OPERATION,
            TYPED_DATABASE_STRICT_RESUME_QUARANTINE_OPERATION,
        }
    ):
        raise TaskSourceIntegrityError(
            "typed strict-resume rejection operation is invalid"
        )
    identity_names = (
        "claim_id",
        "attempt_id",
        "lease_id",
        "owner_session_id",
        "attempt_number",
        "fencing_token",
        "fence_epoch",
    )
    for name in ("task_cid", *identity_names[:4]):
        value = values.get(name)
        if (
            type(value) is not str
            or not value.strip()
            or len(value.encode("utf-8")) > 1_024
            or any(marker in value for marker in ("\x00", "\n", "\r"))
        ):
            raise TaskSourceIntegrityError(
                f"typed strict-resume rejection {name} is invalid"
            )
    for name in (
        "attempt_number",
        "fencing_token",
        "fence_epoch",
        "rejected_task_revision",
        "task_shard_count",
    ):
        value = values.get(name)
        if type(value) is not int or value < 1:
            raise TaskSourceIntegrityError(
                f"typed strict-resume rejection {name} is invalid"
            )
    max_task_attempts = values.get("max_task_attempts")
    shard_index = values.get("task_shard_index")
    route_origin = values.get("execution_route_origin_revision")
    if (
        type(max_task_attempts) is not int
        or not 0 <= max_task_attempts <= 10_000
        or type(shard_index) is not int
        or not 0 <= shard_index < values["task_shard_count"]
        or type(route_origin) is not int
        or route_origin < 0
        or type(values.get("rejected_task_alias")) is not str
        or type(values.get("execution_route_policy_id")) is not str
    ):
        raise TaskSourceIntegrityError(
            "typed strict-resume rejection bounds are invalid"
        )
    for name in (
        "provider_phase_committed",
        "provider_invocation_receipt_present",
        "attempt_budget_exhausted",
    ):
        if type(values.get(name)) is not bool:
            raise TaskSourceIntegrityError(
                f"typed strict-resume rejection {name} is invalid"
            )
    exhausted = bool(
        max_task_attempts > 0
        and values["attempt_number"] >= max_task_attempts
    )
    if values["attempt_budget_exhausted"] is not exhausted:
        raise TaskSourceIntegrityError(
            "typed strict-resume rejection budget evidence is inconsistent"
        )
    provider_started = bool(
        values["provider_phase_committed"]
        or values["provider_invocation_receipt_present"]
    )
    if operation == TYPED_DATABASE_STRICT_RESUME_REQUEUE_OPERATION:
        if exhausted or provider_started:
            raise TaskSourceIntegrityError(
                "typed strict-resume requeue is not pre-provider and under budget"
            )
    elif not exhausted and not provider_started:
        raise TaskSourceIntegrityError(
            "typed strict-resume quarantine has no terminal authority"
        )
    reasons = values.get("reasons")
    if (
        not isinstance(reasons, list)
        or not reasons
        or len(reasons) > 64
        or any(
            type(reason) is not str
            or not reason.strip()
            or len(reason.encode("utf-8")) > 256
            for reason in reasons
        )
        or reasons != sorted(set(reasons))
    ):
        raise TaskSourceIntegrityError(
            "typed strict-resume rejection reasons are invalid"
        )
    shared = values.get("shared_claim_binding")
    shared_fields = {*identity_names, "operation", "claim_phase_schema"}
    if not isinstance(shared, Mapping) or set(shared) != shared_fields:
        raise TaskSourceIntegrityError(
            "typed strict-resume rejection shared binding is invalid"
        )
    if any(
        not _strict_scalar_equal(shared.get(name), values[name])
        for name in identity_names
    ):
        raise TaskSourceIntegrityError(
            "typed strict-resume rejection shared identity differs"
        )
    shared_operation = shared.get("operation")
    shared_schema = shared.get("claim_phase_schema")
    if not (
        (
            shared_operation == "database_claim"
            and shared_schema == TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA
        )
        or (
            shared_operation == "database_attempt_admitted"
            and shared_schema == TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA
        )
    ):
        raise TaskSourceIntegrityError(
            "typed strict-resume rejection shared phase is invalid"
        )
    route_raw = values.get("execution_route_binding")
    if not isinstance(route_raw, Mapping) or not route_raw:
        raise TaskSourceIntegrityError(
            "typed strict-resume rejection route is absent"
        )
    try:
        route = TaskExecutionRouteBinding.from_dict(route_raw).to_dict()
    except (TypeError, ValueError, TaskSourceIntegrityError) as exc:
        raise TaskSourceIntegrityError(
            "typed strict-resume rejection route is invalid"
        ) from exc
    if (
        dict(route_raw) != route
        or route["task_cid"] != values["task_cid"]
        or values["execution_route_policy_id"] != route["policy_id"]
        or values["execution_route_origin_revision"]
        != route["task_revision"]
    ):
        raise TaskSourceIntegrityError(
            "typed strict-resume rejection route lineage differs"
        )
    if values.get("receipt_id") != (
        typed_database_strict_resume_rejection_receipt_id(values)
    ):
        raise TaskSourceIntegrityError(
            "typed strict-resume rejection receipt identity differs"
        )
    return values


def _require_database_claim_process_attestation(
    receipt: Mapping[str, Any],
    *,
    grant: OwnerClientGrant,
) -> dict[str, Any]:
    """Require the receipt to reproduce the active owner-derived birth tuple."""

    observed = receipt.get("claim_process_attestation")
    expected = _claim_process_attestation(grant)
    if (
        not isinstance(observed, Mapping)
        or set(observed) != set(expected)
        or any(
            not _strict_scalar_equal(observed.get(name), value)
            for name, value in expected.items()
        )
    ):
        raise TypedStateOwnerAuthorizationError(
            "database claim receipt process attestation differs from its active grant"
        )
    return expected


def _validated_database_claim_process_attestation(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a previously owner-attested claim process without reviving it."""

    observed = receipt.get("claim_process_attestation")
    required = {
        "schema",
        "grant_id",
        "client_id",
        "process_birth_id",
        "pid",
        "uid",
        "start_time_ticks",
        "boot_id",
        "parent_pid",
    }
    if not isinstance(observed, Mapping) or set(observed) != required:
        raise TypedStateOwnerAuthorizationError(
            "database claim receipt process attestation is malformed"
        )
    attestation = dict(observed)
    text_fields = (
        "grant_id",
        "client_id",
        "process_birth_id",
        "boot_id",
    )
    if (
        attestation.get("schema") != TYPED_DATABASE_CLAIM_PROCESS_SCHEMA
        or any(
            type(attestation.get(name)) is not str
            or not attestation[name].strip()
            or len(attestation[name].encode("utf-8")) > 1_024
            or any(
                marker in attestation[name]
                for marker in ("\x00", "\n", "\r")
            )
            for name in text_fields
        )
        or type(attestation.get("pid")) is not int
        or attestation["pid"] < 1
        or type(attestation.get("uid")) is not int
        or attestation["uid"] < 0
        or type(attestation.get("start_time_ticks")) is not int
        or attestation["start_time_ticks"] < 1
        or type(attestation.get("parent_pid")) is not int
        or attestation["parent_pid"] < 0
        or attestation["process_birth_id"]
        != _process_birth_content_id(
            attestation["pid"],
            attestation["start_time_ticks"],
            attestation["boot_id"],
            attestation["parent_pid"],
        )
    ):
        raise TypedStateOwnerAuthorizationError(
            "database claim receipt process attestation is malformed"
        )
    return attestation


def _validated_retry_cooldown_parameters(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the complete semantic payload for one typed cooldown."""

    parameters = dict(value)
    required = {
        "schema",
        "operation",
        "task_cid",
        "expected_task_revision",
        "expected_task_status",
        "attempt_id",
        "claim_id",
        "lease_id",
        "owner_session_id",
        "attempt_number",
        "fencing_token",
        "fence_epoch",
        "delay_ms",
        "started_at_ms",
        "retry_not_before_ms",
        "selection_penalty",
        "consecutive_failures",
        "reason",
        "resolution_cid",
        "expected_queue_revision",
        "expected_queue_attempt",
        "extension_schema",
        "extension_json",
    }
    if set(parameters) != required:
        raise TypedStateOwnerAuthorizationError(
            "retry cooldown command differs from its closed schema"
        )
    if (
        parameters.get("schema") != TYPED_RETRY_COOLDOWN_SCHEMA
        or parameters.get("operation") != "task.retry.cooldown.record"
        or parameters.get("extension_schema") != TYPED_RETRY_COOLDOWN_SCHEMA
        or parameters.get("expected_task_status")
        not in {"blocked", "in_progress", "retrying"}
    ):
        raise TypedStateOwnerAuthorizationError(
            "retry cooldown command schema or control status is invalid"
        )
    text_bounds = {
        "task_cid": 1_024,
        "attempt_id": 1_024,
        "claim_id": 1_024,
        "lease_id": 1_024,
        "owner_session_id": 1_024,
        "reason": 2_048,
        "resolution_cid": 1_024,
    }
    for name, maximum in text_bounds.items():
        member = parameters.get(name)
        if (
            not isinstance(member, str)
            or not member.strip()
            or member != member.strip()
            or len(member.encode("utf-8")) > maximum
            or any(marker in member for marker in ("\x00", "\n", "\r"))
        ):
            raise TypedStateOwnerAuthorizationError(
                f"retry cooldown {name} is invalid"
            )
    positive = (
        "attempt_number",
        "fencing_token",
        "fence_epoch",
    )
    if any(
        isinstance(parameters.get(name), bool)
        or not isinstance(parameters.get(name), int)
        or int(parameters[name]) < 1
        for name in positive
    ):
        raise TypedStateOwnerAuthorizationError(
            "retry cooldown positive revision/fence identity is invalid"
        )
    bounded_nonnegative = {
        # A materialized retrying task at revision 1 canonically binds the
        # transition's predecessor revision 0.  The owner separately proves
        # the current durable task revision is exactly predecessor + 1.
        "expected_task_revision": 9_223_372_036_854_775_807,
        "delay_ms": 86_400_000,
        "started_at_ms": 9_223_372_036_854_775_807,
        "retry_not_before_ms": 9_223_372_036_854_775_807,
        "selection_penalty": 1_000_000,
        "consecutive_failures": 10_000,
        "expected_queue_attempt": 10_000,
    }
    if any(
        isinstance(parameters.get(name), bool)
        or not isinstance(parameters.get(name), int)
        or not 0 <= int(parameters[name]) <= maximum
        for name, maximum in bounded_nonnegative.items()
    ):
        raise TypedStateOwnerAuthorizationError(
            "retry cooldown bounded counter or deadline is invalid"
        )
    queue_revision = parameters.get("expected_queue_revision")
    if (
        isinstance(queue_revision, bool)
        or not isinstance(queue_revision, int)
        or queue_revision < -1
        or (queue_revision == -1) != (parameters["expected_queue_attempt"] == 0)
        or (
            queue_revision >= 0
            and (
                queue_revision < 1
                or parameters["expected_queue_attempt"] < 1
                or parameters["expected_queue_attempt"]
                >= parameters["attempt_number"]
            )
        )
        or parameters["consecutive_failures"] != parameters["attempt_number"]
        or parameters["retry_not_before_ms"] - parameters["started_at_ms"]
        != parameters["delay_ms"]
    ):
        raise TypedStateOwnerAuthorizationError(
            "retry cooldown absence/revision binding is invalid"
        )
    expected_extension = {
        "schema": TYPED_RETRY_COOLDOWN_SCHEMA,
        "task_cid": parameters["task_cid"],
        "expected_task_revision": parameters["expected_task_revision"],
        "attempt_id": parameters["attempt_id"],
        "claim_id": parameters["claim_id"],
        "lease_id": parameters["lease_id"],
        "owner_session_id": parameters["owner_session_id"],
        "attempt_number": parameters["attempt_number"],
        "fencing_token": parameters["fencing_token"],
        "fence_epoch": parameters["fence_epoch"],
        "delay_ms": parameters["delay_ms"],
        "started_at_ms": parameters["started_at_ms"],
        "retry_not_before_ms": parameters["retry_not_before_ms"],
        "selection_penalty": parameters["selection_penalty"],
        "consecutive_failures": parameters["consecutive_failures"],
        "reason": parameters["reason"],
        "expected_queue_revision": parameters["expected_queue_revision"],
        "expected_queue_attempt": parameters["expected_queue_attempt"],
    }
    try:
        extension = json.loads(str(parameters["extension_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TypedStateOwnerAuthorizationError(
            "retry cooldown extension is malformed"
        ) from exc
    expected_resolution = content_identity(
        {
            "typed_retry_cooldown": expected_extension,
            "started_at_ms": parameters["started_at_ms"],
        }
    )
    if extension != expected_extension or parameters["resolution_cid"] != (
        expected_resolution
    ):
        raise TypedStateOwnerAuthorizationError(
            "retry cooldown extension or resolution identity differs"
        )
    return parameters


def _retry_cooldown_command_digest(parameters: Mapping[str, Any]) -> str:
    """Return the identity a cooldown command must bind for replay safety."""

    validated = _validated_retry_cooldown_parameters(parameters)
    material = {
        name: member
        for name, member in validated.items()
        if name not in {"extension_schema", "extension_json"}
    }
    return hashlib.sha256(canonical_json_bytes(material)).hexdigest()


def _validated_dead_claim_recovery_parameters(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the closed atomic lost-sidecar recovery command."""

    parameters = dict(value)
    if set(parameters) != {
        "schema",
        "operation",
        "task_cid",
        "expected_task_revision",
        "expected_task_status",
        "attempt_id",
        "claim_id",
        "lease_id",
        "owner_session_id",
        "attempt_number",
        "fencing_token",
        "fence_epoch",
        "delay_ms",
        "started_at_ms",
        "retry_not_before_ms",
        "selection_penalty",
        "consecutive_failures",
        "reason",
        "resolution_cid",
        "expected_queue_revision",
        "expected_queue_attempt",
        "extension_schema",
        "extension_json",
        "status",
        "body_json",
    }:
        raise TypedStateOwnerAuthorizationError(
            "dead claim recovery command differs from its closed schema"
        )
    if (
        parameters.get("operation") != TYPED_DATABASE_CLAIM_RECOVERY_COMMAND
        or parameters.get("status") != "retrying"
        or parameters.get("expected_task_status") != "in_progress"
        or parameters.get("delay_ms") != 0
        or parameters.get("selection_penalty") != 0
        or parameters.get("reason") != TYPED_DATABASE_CLAIM_RECOVERY_REASON
        or parameters.get("expected_queue_revision") != -1
        or parameters.get("expected_queue_attempt") != 0
    ):
        raise TypedStateOwnerAuthorizationError(
            "dead claim recovery command is outside its closed transition"
        )
    body_json = parameters.get("body_json")
    if (
        type(body_json) is not str
        or not body_json
        or len(body_json.encode("utf-8")) > 1024 * 1024
    ):
        raise TypedStateOwnerAuthorizationError(
            "dead claim recovery task body is invalid"
        )
    try:
        body = json.loads(body_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TypedStateOwnerAuthorizationError(
            "dead claim recovery task body is malformed"
        ) from exc
    if not isinstance(body, Mapping):
        raise TypedStateOwnerAuthorizationError(
            "dead claim recovery task body is malformed"
        )
    cooldown_parameters = {
        name: member
        for name, member in parameters.items()
        if name not in {"status", "body_json"}
    }
    cooldown_parameters["operation"] = "task.retry.cooldown.record"
    _validated_retry_cooldown_parameters(cooldown_parameters)
    return {
        **parameters,
        "body": dict(body),
        "cooldown_parameters": cooldown_parameters,
    }


def _validated_retry_mutation_parameters(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the common cooldown mutation payload for either typed command."""

    if value.get("operation") == TYPED_DATABASE_CLAIM_RECOVERY_COMMAND:
        return dict(
            _validated_dead_claim_recovery_parameters(value)[
                "cooldown_parameters"
            ]
        )
    return _validated_retry_cooldown_parameters(value)


def _dead_claim_recovery_command_digest(
    parameters: Mapping[str, Any],
) -> str:
    """Bind deterministic replay to the complete atomic recovery payload."""

    validated = _validated_dead_claim_recovery_parameters(parameters)
    material = {
        name: member
        for name, member in validated.items()
        if name not in {"body", "cooldown_parameters"}
    }
    return hashlib.sha256(canonical_json_bytes(material)).hexdigest()


_RETRY_COOLDOWN_ROW_FIELDS: Final[tuple[str, ...]] = (
    "task_cid",
    "claim_cid",
    "resolution_cid",
    "claimant_did",
    "logical_epoch",
    "fencing_token",
    "expires_at_ms",
    "attempt",
    "state",
    "started_at_ms",
    "release_reason",
    "retry_not_before_ms",
    "owner_session_id",
    "fence_epoch",
    "revision",
    "extension_schema",
    "extension_json",
)


def _validated_stored_retry_cooldown(
    row: Any,
    *,
    task_cid: str,
) -> dict[str, Any]:
    """Validate every persisted lease field before admitting replacement."""

    if isinstance(row, Mapping):
        try:
            values = {name: row[name] for name in _RETRY_COOLDOWN_ROW_FIELDS}
        except (KeyError, TypeError) as exc:
            raise TypedStateOwnerAuthorizationError(
                "retry cooldown prior row is incomplete"
            ) from exc
    elif isinstance(row, Sequence) and not isinstance(
        row, (str, bytes, bytearray)
    ):
        if len(row) != len(_RETRY_COOLDOWN_ROW_FIELDS):
            raise TypedStateOwnerAuthorizationError(
                "retry cooldown prior row is incomplete"
            )
        # ``strict=`` is unavailable on the Python 3.8 compatibility floor;
        # the exact length check above provides the same closed-row guarantee.
        values = dict(zip(_RETRY_COOLDOWN_ROW_FIELDS, row))  # noqa: B905
    else:
        raise TypedStateOwnerAuthorizationError(
            "retry cooldown prior row is malformed"
        )
    integer_fields = (
        "logical_epoch",
        "fencing_token",
        "expires_at_ms",
        "attempt",
        "started_at_ms",
        "retry_not_before_ms",
        "fence_epoch",
        "revision",
    )
    if any(
        isinstance(values[name], bool)
        or not isinstance(values[name], int)
        or values[name] < 0
        for name in integer_fields
    ):
        raise TypedStateOwnerAuthorizationError(
            "retry cooldown prior row has an invalid integer field"
        )
    try:
        extension = json.loads(str(values["extension_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TypedStateOwnerAuthorizationError(
            "retry cooldown prior row extension is malformed"
        ) from exc
    if not isinstance(extension, Mapping):
        raise TypedStateOwnerAuthorizationError(
            "retry cooldown prior row extension is malformed"
        )
    extension_values = dict(extension)
    _validated_retry_cooldown_parameters(
        {
            **extension_values,
            "operation": "task.retry.cooldown.record",
            "expected_task_status": "in_progress",
            "resolution_cid": values["resolution_cid"],
            "extension_schema": values["extension_schema"],
            "extension_json": values["extension_json"],
        }
    )
    revision = int(values["revision"])
    attempt = int(values["attempt"])
    expected_prior_revision = -1 if revision == 1 else revision - 1
    expected_prior_attempt = int(extension_values["expected_queue_attempt"])
    if (
        values["task_cid"] != task_cid
        or values["extension_schema"] != TYPED_RETRY_COOLDOWN_SCHEMA
        or values["state"] != "released"
        or values["expires_at_ms"] != 0
        or attempt < 1
        or revision < 1
        or values["logical_epoch"] != values["fence_epoch"]
        or values["claimant_did"] != values["owner_session_id"]
        or extension_values["task_cid"] != task_cid
        or extension_values["claim_id"] != values["claim_cid"]
        or extension_values["owner_session_id"]
        != values["owner_session_id"]
        or extension_values["attempt_number"] != attempt
        or extension_values["fencing_token"] != values["fencing_token"]
        or extension_values["fence_epoch"] != values["fence_epoch"]
        or extension_values["started_at_ms"] != values["started_at_ms"]
        or extension_values["retry_not_before_ms"]
        != values["retry_not_before_ms"]
        or extension_values["reason"] != values["release_reason"]
        or extension_values["expected_queue_revision"]
        != expected_prior_revision
        or (revision == 1 and expected_prior_attempt != 0)
        or (
            revision > 1
            and not 1 <= expected_prior_attempt < attempt
        )
    ):
        raise TypedStateOwnerAuthorizationError(
            "retry cooldown prior row differs from its typed receipt"
        )
    return {**values, "extension": extension_values}


_TRANSACTION_SQL: Final[Mapping[str, OwnerOperation]] = MappingProxyType(
    {
        "txn_load_generation": OwnerOperation(
            "txn_load_generation",
            """
            SELECT generation, schema_revision, fence_epoch, revision,
                   database_uuid, birth_id
            FROM store_generations
            ORDER BY generation DESC
            LIMIT 1
            """,
            0,
            False,
        ),
        "txn_lookup_idempotency": OwnerOperation(
            "txn_lookup_idempotency",
            """
            SELECT idempotency_key, command_kind, command_id, store_id,
                   session_id, result_digest, created_at, expires_at, body_json
            FROM idempotency_records
            WHERE idempotency_key = ?
            LIMIT 1
            """,
            1,
            False,
            parameter_names=("idempotency_key",),
        ),
        "txn_record_idempotency": OwnerOperation(
            "txn_record_idempotency",
            """
            INSERT INTO idempotency_records (
                idempotency_key, command_kind, command_id, store_id,
                session_id, result_digest, created_at, expires_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            9,
            True,
            parameter_names=(
                "idempotency_key",
                "command_kind",
                "command_id",
                "store_id",
                "session_id",
                "result_digest",
                "created_at",
                "expires_at",
                "body_json",
            ),
        ),
        "txn_advance_store_revision": OwnerOperation(
            "txn_advance_store_revision",
            """
            UPDATE store_generations
            SET revision = ?
            WHERE generation = ? AND revision = ? AND fence_epoch = ?
            """,
            4,
            True,
            parameter_names=(
                "new_revision",
                "generation",
                "expected_revision",
                "fence_epoch",
            ),
        ),
        "txn_cas_task_status": OwnerOperation(
            "txn_cas_task_status",
            """
            UPDATE "tasks" SET "status" = ?, "updated_at" = ?, "revision" = ?
            WHERE "task_cid" = ? AND "revision" = ? RETURNING "revision"
            """,
            5,
            True,
            parameter_names=(
                "status",
                "updated_at",
                "new_revision",
                "task_cid",
                "expected_task_revision",
            ),
        ),
    }
)

_TRANSACTION_SQL_LOOKUP: Final[Mapping[str, str]] = MappingProxyType(
    {_normalize_sql(value.sql): name for name, value in _TRANSACTION_SQL.items()}
)

_TRANSACTION_MUTATIONS: Final[frozenset[str]] = frozenset(
    {"txn_advance_store_revision", "txn_record_idempotency"}
)
_EVENT_MUTATIONS: Final[frozenset[str]] = frozenset(
    {
        "casf_seed_global_head",
        "casf_advance_global_head",
        "casf_seed_stream_head",
        "casf_advance_stream_head",
        "casf_insert_domain_event",
        "casf_insert_event_parent",
        "casf_insert_changed_fact",
        "casf_insert_outbox",
    }
)

# Closed server policy: a caller's StateCommand label cannot enlarge this map.
# The common transaction/idempotency operations are added separately below.
_COMMAND_MUTATION_CATALOG: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "federation.create": frozenset(
            {
                "casf_insert_federation",
                "casf_insert_authorization_decision",
                "casf_insert_policy",
                "casf_insert_federation_budget",
                "casf_seed_subagent_slot",
                "casf_transition_admission_budget_reservation",
            }
        ),
        "budget.reserve": frozenset(
            {
                "casf_insert_admission_budget_reservation",
                "casf_insert_admission_budget_dimension",
            }
        ),
        "budget.release": frozenset(
            {"casf_transition_admission_budget_reservation"}
        ),
        "supervisor.register": frozenset(
            {
                "casf_insert_policy_decision",
                "casf_insert_supervisor_definition",
                "casf_insert_supervisor_assignment",
                "casf_insert_supervisor_capability",
                "casf_insert_supervisor",
            }
        ),
        "supervisor.runtime.attest": frozenset(
            {
                "casf_insert_process_birth_attestation",
                "casf_insert_supervisor_runtime_lease",
                "casf_supersede_supervisor_runtime_lease",
                "casf_update_supervisor_process_birth",
            }
        ),
        "supervisor.transition": frozenset({"casf_update_supervisor_lifecycle"}),
        "subagent.register": frozenset(
            {
                "casf_insert_policy_decision",
                "casf_insert_subagent_definition",
                "casf_insert_subagent_assignment",
                "casf_insert_subagent_capability",
                "casf_insert_subagent",
            }
        ),
        "subagent.slot.reserve": frozenset(
            {
                "casf_reserve_subagent_slot",
                "casf_activate_subagent",
                "casf_insert_subagent_slot_ledger",
            }
        ),
        "subagent.slot.release": frozenset(
            {
                "casf_release_subagent_slot",
                "casf_deactivate_subagent",
                "casf_insert_subagent_slot_ledger",
            }
        ),
        "subagent.outcome": frozenset({"casf_insert_subagent_outcome"}),
        "subscription.register": frozenset(
            {
                "casf_insert_subscription",
                "casf_insert_subscription_selector",
                "casf_insert_consumer_cursor",
            }
        ),
        "event.route.persist": frozenset(
            {
                "casf_insert_coalescing_coverage",
                "casf_insert_coalescing_input",
                "casf_insert_delivery_queue",
            }
        ),
        "event.outbox.disposition": frozenset(
            {
                "casf_insert_outbox_routing_disposition",
                "casf_insert_outbox_routing_disposition_event",
                "casf_mark_outbox_routed",
            }
        ),
        "event.delivery.record": frozenset(
            {
                "casf_insert_delivery_attempt",
                "casf_mark_queue_delivered",
            }
        ),
        "event.delivery.fail": frozenset(
            {
                "casf_mark_delivery_failed",
                "casf_update_queue_after_failure",
                "casf_increment_subscription_failures",
                "casf_insert_dead_letter",
                "casf_quarantine_subscription",
            }
        ),
        "event.acknowledge": frozenset(
            {
                "casf_insert_event_acknowledgement",
                "casf_advance_consumer_cursor",
                "casf_mark_delivery_acknowledged",
                "casf_mark_queue_acknowledged",
                "casf_reset_subscription_failures",
            }
        ),
        "task.status.cas": frozenset({"txn_cas_task_status"}),
        "task.status.cas.receipt": frozenset(
            {"executor_cas_task_status_receipt"}
        ),
        "task.retry.cooldown.record": frozenset(
            {
                "executor_insert_retry_cooldown",
                "executor_update_retry_cooldown",
            }
        ),
        TYPED_DATABASE_CLAIM_RECOVERY_COMMAND: frozenset(
            {
                "executor_cas_task_status_receipt",
                "executor_insert_retry_cooldown",
            }
        ),
        "task.validation.record.passed": frozenset(
            {
                "executor_insert_validation_run",
                "executor_insert_validation_result",
                "executor_insert_validation_evidence",
            }
        ),
        "task.validation.record.nonpassing": frozenset(
            {
                "executor_insert_validation_run",
                "executor_insert_validation_result",
            }
        ),
    }
)

_FEDERATION_COMMANDS: Final[frozenset[str]] = frozenset(
    set(_COMMAND_MUTATION_CATALOG)
    - {
        "task.status.cas",
        "task.status.cas.receipt",
        "task.retry.cooldown.record",
        TYPED_DATABASE_CLAIM_RECOVERY_COMMAND,
        "task.validation.record.passed",
        "task.validation.record.nonpassing",
    }
)
_EVENT_EMITTING_COMMANDS: Final[frozenset[str]] = frozenset(
    {
        "federation.create",
        "budget.reserve",
        "budget.release",
        "supervisor.register",
        "supervisor.runtime.attest",
        "supervisor.transition",
        "subagent.register",
        "subagent.slot.reserve",
        "subagent.slot.release",
        "subagent.outcome",
        "subscription.register",
    }
)
_COMMAND_REQUIRED_DOMAIN_MUTATIONS: Final[Mapping[str, frozenset[str]]] = (
    MappingProxyType(
        {
            "federation.create": _COMMAND_MUTATION_CATALOG["federation.create"],
            "budget.reserve": _COMMAND_MUTATION_CATALOG["budget.reserve"],
            "budget.release": _COMMAND_MUTATION_CATALOG["budget.release"],
            "supervisor.register": _COMMAND_MUTATION_CATALOG[
                "supervisor.register"
            ],
            "supervisor.transition": _COMMAND_MUTATION_CATALOG[
                "supervisor.transition"
            ],
            "supervisor.runtime.attest": frozenset(
                {
                    "casf_insert_process_birth_attestation",
                    "casf_insert_supervisor_runtime_lease",
                    "casf_update_supervisor_process_birth",
                }
            ),
            "subagent.register": _COMMAND_MUTATION_CATALOG["subagent.register"],
            "subagent.slot.reserve": _COMMAND_MUTATION_CATALOG[
                "subagent.slot.reserve"
            ],
            "subagent.slot.release": _COMMAND_MUTATION_CATALOG[
                "subagent.slot.release"
            ],
            "subagent.outcome": _COMMAND_MUTATION_CATALOG["subagent.outcome"],
            "subscription.register": _COMMAND_MUTATION_CATALOG[
                "subscription.register"
            ],
            "event.route.persist": _COMMAND_MUTATION_CATALOG[
                "event.route.persist"
            ],
            "event.outbox.disposition": _COMMAND_MUTATION_CATALOG[
                "event.outbox.disposition"
            ],
            "event.delivery.record": frozenset(
                {"casf_insert_delivery_attempt", "casf_mark_queue_delivered"}
            ),
            "event.delivery.fail": frozenset(
                {
                    "casf_mark_delivery_failed",
                    "casf_increment_subscription_failures",
                    "casf_update_queue_after_failure",
                }
            ),
            "event.acknowledge": _COMMAND_MUTATION_CATALOG["event.acknowledge"],
            "task.status.cas": frozenset({"txn_cas_task_status"}),
            "task.status.cas.receipt": _COMMAND_MUTATION_CATALOG[
                "task.status.cas.receipt"
            ],
            TYPED_DATABASE_CLAIM_RECOVERY_COMMAND: (
                _COMMAND_MUTATION_CATALOG[
                    TYPED_DATABASE_CLAIM_RECOVERY_COMMAND
                ]
            ),
            "task.validation.record.passed": _COMMAND_MUTATION_CATALOG[
                "task.validation.record.passed"
            ],
            "task.validation.record.nonpassing": _COMMAND_MUTATION_CATALOG[
                "task.validation.record.nonpassing"
            ],
        }
    )
)
_REPEATABLE_MUTATIONS: Final[frozenset[str]] = frozenset(
    {
        "casf_seed_subagent_slot",
        "casf_insert_admission_budget_dimension",
        "casf_insert_supervisor_capability",
        "casf_insert_subagent_capability",
        "casf_insert_subscription_selector",
        "casf_insert_coalescing_coverage",
        "casf_insert_coalescing_input",
        "casf_insert_delivery_queue",
        "casf_insert_outbox_routing_disposition_event",
        "casf_mark_outbox_routed",
        "casf_insert_event_parent",
        "casf_insert_changed_fact",
    }
)
_EVENT_CORE_SEQUENCE: Final[tuple[str, ...]] = (
    "casf_seed_global_head",
    "casf_advance_global_head",
    "casf_seed_stream_head",
    "casf_advance_stream_head",
    "casf_insert_domain_event",
    "casf_insert_outbox",
)
_POST_EVENT_DOMAIN_MUTATIONS: Final[Mapping[str, frozenset[str]]] = (
    MappingProxyType(
        {
            "subagent.slot.reserve": frozenset(
                {"casf_insert_subagent_slot_ledger"}
            ),
            "subagent.slot.release": frozenset(
                {"casf_insert_subagent_slot_ledger"}
            ),
        }
    )
)
MAX_TRANSACTION_MUTATIONS: Final = 4_096
_MUTATION_REPEAT_LIMITS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "casf_insert_outbox_routing_disposition_event": 1_024,
        "casf_mark_outbox_routed": 1_024,
    }
)
_SCOPE_FIELDS: Final[tuple[str, ...]] = (
    "tenant_id",
    "federation_id",
    "supervisor_id",
    "subagent_id",
    "repository_id",
    "tree_id",
    "task_id",
    "task_cid",
    "subscription_id",
    "consumer_id",
    "event_id",
)
_SCOPE_INDEPENDENT_READS: Final[frozenset[str]] = frozenset(
    {
        # Tenant-scoped capacity aggregation intentionally spans the
        # federation being admitted.  Its present tenant_id parameter is
        # still compared below; only absent narrower entity scopes are
        # tolerated for this closed server-owned operation.
        "casf_select_active_admission_budget_usage",
        "whoami_metadata",
        "load_store_generation",
    }
)


def internal_operation_for_sql(sql: str) -> str:
    """Map only exact trusted transaction SQL to a server operation name."""

    name = _TRANSACTION_SQL_LOOKUP.get(_normalize_sql(sql), "")
    if not name:
        raise TypedStateOwnerProtocolError(
            "raw SQL is forbidden by the typed state-owner boundary"
        )
    return name


def build_control_plane_operation_catalog() -> Mapping[str, OwnerOperation]:
    """Build the immutable catalog from trusted in-tree template producers.

    Imports are intentionally local: the client and server both import this
    protocol module, while the template producers import the client contracts.
    No caller can contribute a template to this server catalog.
    """

    from ..federation.registry import _casf_templates
    from .control_plane_repository import _REPOSITORY_TEMPLATES
    from .quack_state_client import DEFAULT_STATEMENT_TEMPLATES, StatementKind

    catalog: dict[str, OwnerOperation] = dict(_TRANSACTION_SQL)
    templates = [
        *DEFAULT_STATEMENT_TEMPLATES.values(),
        *_REPOSITORY_TEMPLATES.values(),
        *_casf_templates(),
    ]
    for template in templates:
        operation = OwnerOperation(
            name=template.name,
            sql=template.sql,
            parameter_count=len(template.parameter_names),
            mutation=template.kind is StatementKind.MUTATION,
            bootstrap_only=template.name in {
                "seed_store_generation",
                "seed_client_session",
            },
            parameter_names=tuple(template.parameter_names),
        )
        existing = catalog.get(operation.name)
        if existing is not None and existing != operation:
            raise TypedStateOwnerProtocolError(
                f"conflicting server operation catalog entry: {operation.name}"
            )
        catalog[operation.name] = operation
    return MappingProxyType(dict(sorted(catalog.items())))


def catalog_fingerprint(catalog: Mapping[str, OwnerOperation]) -> str:
    public = [catalog[name].public_dict() for name in sorted(catalog)]
    return "sha256:" + hashlib.sha256(canonical_json_bytes(public)).hexdigest()


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and ("\x00" in value or len(value.encode()) > 1_048_576):
            raise TypedStateOwnerProtocolError("bound string exceeds the closed contract")
        return value
    if isinstance(value, float):
        if value != value or abs(value) == float("inf"):
            raise TypedStateOwnerProtocolError("bound float must be finite")
        return value
    # DuckDB DATE/TIMESTAMP/UUID values are observations, never identities in
    # this transport; preserve them as bounded canonical strings.
    text = str(value)
    if "\x00" in text or len(text.encode()) > 1_048_576:
        raise TypedStateOwnerProtocolError("observed scalar exceeds the closed contract")
    return text


def _closed_parameters(value: Any) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypedStateOwnerProtocolError("parameters must be a bounded array")
    if len(value) > MAX_PARAMETER_COUNT:
        raise TypedStateOwnerProtocolError("parameter population exceeds bound")
    return [_json_scalar(item) for item in value]


def _send_frame(channel: socket.socket, payload: Mapping[str, Any]) -> None:
    body = canonical_json_bytes(dict(payload))
    if len(body) > MAX_FRAME_BYTES:
        raise TypedStateOwnerProtocolError("typed state-owner frame exceeds bound")
    channel.sendall(len(body).to_bytes(4, "big") + body)


def _receive_exact(channel: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        part = channel.recv(remaining)
        if not part:
            raise TypedStateOwnerProtocolError("typed state-owner channel closed")
        chunks.append(part)
        remaining -= len(part)
    return b"".join(chunks)


def _receive_frame(channel: socket.socket) -> dict[str, Any]:
    size = int.from_bytes(_receive_exact(channel, 4), "big")
    if size < 2 or size > MAX_FRAME_BYTES:
        raise TypedStateOwnerProtocolError("typed state-owner frame size is invalid")
    try:
        value = json.loads(_receive_exact(channel, size).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TypedStateOwnerProtocolError("typed state-owner frame is not JSON") from exc
    if not isinstance(value, dict):
        raise TypedStateOwnerProtocolError("typed state-owner frame must be an object")
    return value


def _result_columns(result: Any) -> tuple[str, ...]:
    direct = getattr(result, "_columns", None)
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
        return tuple(str(item) for item in direct)
    description = getattr(result, "description", None) or ()
    return tuple(str(item[0] if isinstance(item, Sequence) else item) for item in description)


def _result_rows(result: Any) -> list[list[Any]]:
    fetchall = getattr(result, "fetchall", None)
    rows = list(fetchall() or []) if callable(fetchall) else []
    if len(rows) > MAX_ROW_COUNT:
        raise TypedStateOwnerProtocolError("operation result population exceeds bound")
    output: list[list[Any]] = []
    for row in rows:
        if isinstance(row, Mapping):
            output.append([_json_scalar(value) for value in row.values()])
        elif isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
            output.append([_json_scalar(value) for value in row])
        else:
            output.append([_json_scalar(row)])
    return output


class TypedOwnerResult:
    """Small DB-API-shaped result returned to existing transaction code."""

    def __init__(
        self,
        columns: Sequence[str],
        rows: Sequence[Sequence[Any]],
        rowcount: int,
    ) -> None:
        self._columns = tuple(str(item) for item in columns)
        self.description = tuple((name,) for name in self._columns)
        self._rows = [tuple(item) for item in rows]
        self._offset = 0
        self.rowcount = int(rowcount)

    def fetchall(self) -> list[tuple[Any, ...]]:
        rows = self._rows[self._offset :]
        self._offset = len(self._rows)
        return list(rows)

    def fetchone(self) -> tuple[Any, ...] | None:
        if self._offset >= len(self._rows):
            return None
        row = self._rows[self._offset]
        self._offset += 1
        return row


class TypedStateOwnerGateway:
    """Server-owned, authenticated, closed operation executor."""

    def __init__(
        self,
        *,
        connection: Any,
        socket_path: Path,
        store_id: str,
        identity: Mapping[str, Any],
        catalog: Mapping[str, OwnerOperation] | None = None,
        owner_liveness_probe: Any | None = None,
        transaction_lock: Any | None = None,
    ) -> None:
        self._connection = connection
        self.socket_path = Path(socket_path)
        self.store_id = str(store_id or "").strip()
        self.identity = MappingProxyType(dict(identity))
        self.catalog = catalog or build_control_plane_operation_catalog()
        self.catalog_id = catalog_fingerprint(self.catalog)
        self._owner_liveness_probe = owner_liveness_probe or owner_liveness
        if not callable(self._owner_liveness_probe):
            raise TypedStateOwnerProtocolError(
                "owner liveness probe must be callable"
            )
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._transaction_lock = (
            threading.RLock() if transaction_lock is None else transaction_lock
        )
        if any(
            not callable(getattr(self._transaction_lock, name, None))
            for name in ("acquire", "release", "__enter__", "__exit__")
        ):
            raise TypeError(
                "transaction_lock must provide acquire, release, and context entry"
            )
        self._clients: set[threading.Thread] = set()
        self._channels: set[socket.socket] = set()
        self._clients_lock = threading.Lock()
        self._request_count = 0
        self._committed_transactions = 0
        self._last_error_type = ""
        self._grants: dict[str, OwnerClientGrant] = {}
        self._revoked_grants: set[str] = set()
        self._grants_lock = threading.Lock()
        self._status_bootstrap_token_digest: bytes | None = None
        self._status_bootstrap_uid = -1
        self._status_bootstrap_scope: dict[str, str] = {}
        self._event_wait_handler: Any | None = None
        self._event_wait_cancel_handler: Any | None = None
        self._event_wait_clear_handler: Any | None = None
        self._commit_observer: Any | None = None
        self._last_observer_error_type = ""
        self._eaaef_service_bind_lock = threading.Lock()
        self._eaaef_typed_owner_command_service: Any | None = None
        self._eaaef_plan_r2_owner_service: Any | None = None

    def _require_live_server_binding(self) -> None:
        thread = self._thread
        if (
            self._listener is None
            or self._stop.is_set()
            or thread is None
            or not thread.is_alive()
        ):
            raise TypedStateOwnerAuthorizationError(
                "EAAEF binding requires a live server-owned gateway"
            )

    def _bind_eaaef_typed_owner_command_service_from_server(
        self,
        *,
        admission: Any,
    ) -> Any:
        """Monotonically bind EAAEF commands to this owner's resources.

        The caller cannot provide a connection or transaction lock.  Both are
        selected here by the one object that owns them, so the adapter cannot
        accidentally establish a second DuckDB writer boundary.
        """

        from .eaaef_typed_owner_service import (
            EAAEFTypedOwnerServiceError,
            _bind_eaaef_typed_owner_command_service_from_gateway,
        )

        with self._transaction_lock:
            with self._eaaef_service_bind_lock:
                self._require_live_server_binding()
                if self._eaaef_typed_owner_command_service is not None:
                    raise EAAEFTypedOwnerServiceError(
                        "typed owner EAAEF command service is already bound"
                    )
                service = _bind_eaaef_typed_owner_command_service_from_gateway(
                    owner_gateway=self,
                    admission=admission,
                )
                self._eaaef_typed_owner_command_service = service
                return service

    def _bind_eaaef_plan_r2_owner_service_from_server(
        self,
        *,
        admission: Any,
        plan_r2_operational_capability: Mapping[str, Any],
        authorization: Mapping[str, Any],
        trusted_capability_reviewer_dids: Sequence[str],
        trusted_operator_dids: Sequence[str],
        trusted_security_reviewer_dids: Sequence[str],
    ) -> Any:
        """Monotonically bind Plan-R2 to the same connection and lock."""

        from .eaaef_plan_r2_owner_service import (
            EAAEFPlanR2OwnerServiceError,
            _bind_eaaef_plan_r2_owner_service_from_gateway,
        )

        with self._transaction_lock:
            with self._eaaef_service_bind_lock:
                self._require_live_server_binding()
                bootstrap_service = self._eaaef_typed_owner_command_service
                if bootstrap_service is None:
                    raise EAAEFPlanR2OwnerServiceError(
                        "Plan-R2 requires the already-bound R1 owner service"
                    )
                bootstrap_service._require_open()  # noqa: SLF001
                if self._eaaef_plan_r2_owner_service is not None:
                    raise EAAEFPlanR2OwnerServiceError(
                        "typed owner Plan-R2 service is already bound"
                    )
                service = _bind_eaaef_plan_r2_owner_service_from_gateway(
                    owner_gateway=self,
                    bootstrap_service=bootstrap_service,
                    admission=admission,
                    plan_r2_operational_capability=(
                        plan_r2_operational_capability
                    ),
                    authorization=authorization,
                    trusted_capability_reviewer_dids=(
                        trusted_capability_reviewer_dids
                    ),
                    trusted_operator_dids=trusted_operator_dids,
                    trusted_security_reviewer_dids=(
                        trusted_security_reviewer_dids
                    ),
                )
                self._eaaef_plan_r2_owner_service = service
                return service

    def configure_status_bootstrap(self) -> str:
        """Create the owner-local credential for peer-bound status sessions.

        The persisted credential is deliberately not an ``OwnerClientGrant``:
        a later status invocation has a different PID from the short-lived
        launcher. The owner authenticates this value and the kernel peer UID,
        then mints a short-lived read-only grant bound to that exact PID, UID,
        and procfs start tuple for the lifetime of one connection.
        """

        missing = STATUS_BOOTSTRAP_ALLOWED_OPERATIONS - set(self.catalog)
        if missing:
            raise TypedStateOwnerAuthorizationError(
                "status bootstrap operation catalog is incomplete"
            )
        token = secrets.token_hex(32)
        digest = hashlib.sha256(token.encode("ascii")).digest()
        with self._grants_lock:
            if self._status_bootstrap_token_digest is not None:
                raise TypedStateOwnerAuthorizationError(
                    "status bootstrap credential is already configured"
                )
            self._status_bootstrap_token_digest = digest
            self._status_bootstrap_uid = os.getuid()
        return token

    def _resolve_status_bootstrap_scope(self) -> dict[str, str]:
        rows = self._connection.execute(
            """
            SELECT federations.tenant_id, federations.federation_id,
                   supervisors.supervisor_id, subscriptions.subscription_id,
                   subscriptions.consumer_id
            FROM federations
            INNER JOIN supervisor_instances AS supervisors
              ON supervisors.tenant_id = federations.tenant_id
             AND supervisors.federation_id = federations.federation_id
            INNER JOIN event_subscriptions AS subscriptions
              ON subscriptions.tenant_id = federations.tenant_id
             AND subscriptions.federation_id = federations.federation_id
             AND subscriptions.supervisor_id = supervisors.supervisor_id
            WHERE subscriptions.status = 'active'
              AND supervisors.lifecycle_state NOT IN (
                  'COMPLETED', 'FAILED', 'STOPPED', 'QUARANTINED'
              )
            ORDER BY federations.federation_id, supervisors.supervisor_id,
                     subscriptions.subscription_id
            LIMIT 2
            """
        ).fetchall()
        federation_count = self._connection.execute(
            "SELECT COUNT(*) FROM federations"
        ).fetchone()
        if (
            federation_count is None
            or int(federation_count[0]) != 1
            or len(rows) != 1
        ):
            raise TypedStateOwnerAuthorizationError(
                "status bootstrap requires one dedicated-store federation slice"
            )
        row = rows[0]
        return {
            "tenant_id": str(row[0] or "").strip(),
            "federation_id": str(row[1] or "").strip(),
            "supervisor_id": str(row[2] or "").strip(),
            "subscription_id": str(row[3] or "").strip(),
            "consumer_id": str(row[4] or "").strip(),
        }

    def bind_status_bootstrap_scope(self) -> None:
        """Monotonically bind status reads to one admitted federation slice."""

        # All gateway clients share one DuckDB connection.  Resolve the
        # bootstrap slice under the same transaction lock used by commands so
        # this read cannot observe or join an unrelated client's uncommitted
        # transaction.
        with self._transaction_lock:
            scope = self._resolve_status_bootstrap_scope()
        if any(not value or len(value) > 256 for value in scope.values()):
            raise TypedStateOwnerAuthorizationError(
                "status bootstrap scope is invalid"
            )
        with self._grants_lock:
            if self._status_bootstrap_token_digest is None:
                raise TypedStateOwnerAuthorizationError(
                    "status bootstrap credential is not configured"
                )
            if self._status_bootstrap_scope and self._status_bootstrap_scope != scope:
                raise TypedStateOwnerAuthorizationError(
                    "status bootstrap scope is already bound"
                )
            self._status_bootstrap_scope = scope

    def _issue_status_session_grant(
        self,
        *,
        peer_identity: tuple[int, int, int],
        process_birth_id: str,
    ) -> OwnerClientGrant:
        """Mint one non-exported status grant for the observed socket peer."""

        peer_pid, peer_uid, peer_start = peer_identity
        issued_at = int(time.time() * 1_000)
        with self._grants_lock:
            scope = dict(self._status_bootstrap_scope)
        if not scope:
            raise TypedStateOwnerAuthorizationError(
                "status bootstrap scope is unavailable"
            )
        # Do not hold _grants_lock while acquiring the transaction lock: the
        # connection-serving path records sessions under the transaction lock
        # and later retires grants under _grants_lock.
        with self._transaction_lock:
            live_scope = self._resolve_status_bootstrap_scope()
        if live_scope != scope:
            raise TypedStateOwnerAuthorizationError(
                "status bootstrap dedicated-store authority changed"
            )
        grant = OwnerClientGrant(
            grant_id=f"owner-grant:status:{uuid.uuid4()}",
            client_id=STATUS_BOOTSTRAP_CLIENT_ID,
            process_birth_id=process_birth_id,
            allowed_operations=STATUS_BOOTSTRAP_ALLOWED_OPERATIONS,
            allowed_command_operations=frozenset(),
            tenant_id=scope["tenant_id"],
            federation_id=scope["federation_id"],
            entity_scopes=tuple(
                (name, scope[name])
                for name in _STATUS_BOOTSTRAP_ENTITY_SCOPE_NAMES
            ),
            authority_profile="dedicated_store_status_portfolio",
            peer_pid=peer_pid,
            peer_uid=peer_uid,
            peer_start_time_ticks=peer_start,
            issued_at=issued_at,
            expires_at=issued_at
            + int(STATUS_BOOTSTRAP_GRANT_TTL_SECONDS * 1_000),
        )
        ephemeral_token = secrets.token_hex(32)
        with self._grants_lock:
            self._grants[ephemeral_token] = grant
        return grant

    def _retire_status_session_grant(self, grant_id: str) -> None:
        """Remove a connection-local status grant without growing revocation state."""

        with self._grants_lock:
            self._grants = {
                token: candidate
                for token, candidate in self._grants.items()
                if candidate.grant_id != grant_id
            }
            self._revoked_grants.discard(grant_id)

    def bind_commit_observer(self, observer: Any) -> None:
        """Install one owner-only durable-commit notification hook."""

        if not callable(observer):
            raise TypedStateOwnerProtocolError(
                "commit observer must be a server-owned callable"
            )
        with self._grants_lock:
            if self._commit_observer is not None:
                if self._commit_observer is observer:
                    return
                raise TypedStateOwnerProtocolError(
                    "typed state-owner commit observer is already bound"
                )
            self._commit_observer = observer

    def bind_event_wait_handlers(
        self,
        *,
        wait: Any,
        cancel: Any,
        clear_cancellation: Any,
    ) -> None:
        """Bind the state-owner condition without exposing a client callback.

        The handlers are installed by :class:`QuackStateServer` after its
        durable routed-event source is sealed.  Binding is monotonic for the
        gateway lifetime so a later component cannot replace event authority.
        """

        if not all(callable(item) for item in (wait, cancel, clear_cancellation)):
            raise TypedStateOwnerProtocolError(
                "event wait handlers must be server-owned callables"
            )
        with self._grants_lock:
            existing = (
                self._event_wait_handler,
                self._event_wait_cancel_handler,
                self._event_wait_clear_handler,
            )
            if any(item is not None for item in existing):
                if existing == (wait, cancel, clear_cancellation):
                    return
                raise TypedStateOwnerProtocolError(
                    "typed event wait handlers are already bound"
                )
            self._event_wait_handler = wait
            self._event_wait_cancel_handler = cancel
            self._event_wait_clear_handler = clear_cancellation

    def issue_grant(
        self,
        *,
        client_id: str,
        process_birth_id: str = "",
        allowed_operations: Sequence[str] = (),
        allowed_command_operations: Sequence[str] = (),
        tenant_id: str = "",
        federation_id: str = "",
        entity_scopes: Mapping[str, str] | None = None,
        peer_pid: int | None = None,
        ttl_seconds: float = DEFAULT_GRANT_TTL_SECONDS,
    ) -> tuple[str, OwnerClientGrant]:
        """Mint one non-promotable client capability in owner memory."""

        operations = frozenset(str(item) for item in allowed_operations)
        commands = frozenset(str(item) for item in allowed_command_operations)
        if not operations.issubset(
            set(self.catalog) | set(_ISSUABLE_SERVICE_OPERATIONS)
        ):
            raise TypedStateOwnerAuthorizationError(
                "grant contains an operation absent from the server catalog"
            )
        if not commands.issubset(_COMMAND_MUTATION_CATALOG):
            raise TypedStateOwnerAuthorizationError(
                "grant contains a command absent from the server policy"
            )
        try:
            ttl = float(ttl_seconds)
        except (TypeError, ValueError) as exc:
            raise TypedStateOwnerAuthorizationError(
                "grant lifetime is invalid"
            ) from exc
        if (
            not math.isfinite(ttl)
            or ttl < MIN_GRANT_TTL_SECONDS
            or ttl > MAX_GRANT_TTL_SECONDS
        ):
            raise TypedStateOwnerAuthorizationError(
                "grant lifetime is outside the closed bound"
            )
        try:
            selected_pid = os.getpid() if peer_pid is None else int(peer_pid)
            selected_uid = os.stat(f"/proc/{selected_pid}").st_uid
            selected_start = _process_start_time_ticks(selected_pid)
        except (OSError, TypeError, ValueError) as exc:
            raise TypedStateOwnerAuthorizationError(
                "grant peer process is unavailable"
            ) from exc
        issued_at = int(time.time() * 1_000)
        grant = OwnerClientGrant(
            grant_id=f"owner-grant:{uuid.uuid4()}",
            client_id=client_id,
            process_birth_id=process_birth_id,
            allowed_operations=operations,
            allowed_command_operations=commands,
            tenant_id=tenant_id,
            federation_id=federation_id,
            entity_scopes=tuple((entity_scopes or {}).items()),
            peer_pid=selected_pid,
            peer_uid=selected_uid,
            peer_start_time_ticks=selected_start,
            issued_at=issued_at,
            expires_at=issued_at + int(ttl * 1_000),
        )
        token = uuid.uuid4().hex + uuid.uuid4().hex
        with self._grants_lock:
            self._grants[token] = grant
        return token, grant

    def revoke_grant(self, grant_id: str) -> None:
        selected = str(grant_id or "")
        with self._grants_lock:
            self._revoked_grants.add(selected)
            self._grants = {
                token: grant
                for token, grant in self._grants.items()
                if grant.grant_id != selected
            }

    def renew_grant(
        self,
        grant_id: str,
        *,
        ttl_seconds: float = DEFAULT_GRANT_TTL_SECONDS,
    ) -> OwnerClientGrant:
        """Extend one still-live exact grant without changing its authority.

        Long-lived clients retain a connection-local grant record. Renewal
        therefore keeps the stable grant ID and token while replacing only
        its bounded issuance window in the server table. Every subsequent
        request resolves that current table record before authorization.
        """

        selected_id = str(grant_id or "").strip()
        try:
            ttl = float(ttl_seconds)
        except (TypeError, ValueError) as exc:
            raise TypedStateOwnerAuthorizationError(
                "grant renewal lifetime is invalid"
            ) from exc
        if (
            not selected_id
            or not math.isfinite(ttl)
            or ttl < MIN_GRANT_TTL_SECONDS
            or ttl > MAX_GRANT_TTL_SECONDS
        ):
            raise TypedStateOwnerAuthorizationError(
                "grant renewal lifetime is outside the closed bound"
            )
        now_ms = int(time.time() * 1_000)
        with self._grants_lock:
            matches = tuple(
                (token, candidate)
                for token, candidate in self._grants.items()
                if candidate.grant_id == selected_id
            )
            if (
                selected_id in self._revoked_grants
                or len(matches) != 1
                or now_ms >= matches[0][1].expires_at
            ):
                raise TypedStateOwnerAuthorizationError(
                    "owner grant cannot be renewed"
                )
            token, current = matches[0]
            try:
                peer_uid = os.stat(f"/proc/{current.peer_pid}").st_uid
                peer_start = _process_start_time_ticks(current.peer_pid)
            except OSError as exc:
                raise TypedStateOwnerAuthorizationError(
                    "grant renewal peer process is unavailable"
                ) from exc
            if (
                peer_uid != current.peer_uid
                or peer_start != current.peer_start_time_ticks
            ):
                raise TypedStateOwnerAuthorizationError(
                    "grant renewal peer process identity differs"
                )
            renewed = replace(
                current,
                issued_at=now_ms,
                expires_at=now_ms + int(ttl * 1_000),
            )
            self._grants[token] = renewed
            return renewed

    def _require_active_grant(
        self,
        grant: OwnerClientGrant,
        *,
        peer_identity: tuple[int, int, int],
    ) -> OwnerClientGrant:
        """Revalidate revocation, expiry, and kernel peer identity per request."""

        with self._grants_lock:
            revoked = grant.grant_id in self._revoked_grants
            current = tuple(
                candidate
                for candidate in self._grants.values()
                if candidate.grant_id == grant.grant_id
            )
            if revoked or len(current) != 1:
                raise TypedStateOwnerAuthorizationError(
                    "owner grant is revoked"
                )
            active = current[0]
            if int(time.time() * 1_000) >= active.expires_at:
                self._revoked_grants.add(active.grant_id)
                self._grants = {
                    token: candidate
                    for token, candidate in self._grants.items()
                    if candidate.grant_id != active.grant_id
                }
                raise TypedStateOwnerAuthorizationError(
                    "owner grant is expired"
                )
            peer_pid, peer_uid, peer_start = peer_identity
            if (
                peer_pid != active.peer_pid
                or peer_uid != active.peer_uid
                or peer_start != active.peer_start_time_ticks
            ):
                raise TypedStateOwnerAuthorizationError(
                    "kernel peer identity differs from the owner grant"
                )
            return active

    def start(self) -> None:
        self.socket_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        parent_metadata = os.lstat(self.socket_path.parent)
        if not stat.S_ISDIR(parent_metadata.st_mode):
            raise TypedStateOwnerProtocolError(
                "gateway socket parent is not a real directory"
            )
        if parent_metadata.st_uid != os.geteuid():
            raise TypedStateOwnerProtocolError(
                "gateway socket parent is not owned by the current process user"
            )
        os.chmod(self.socket_path.parent, 0o700)
        try:
            metadata = os.lstat(self.socket_path)
        except FileNotFoundError:
            metadata = None
        if metadata is not None:
            if not stat.S_ISSOCK(metadata.st_mode):
                raise TypedStateOwnerProtocolError("gateway socket path is not a socket")
            self.socket_path.unlink()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            listener.listen(64)
            listener.settimeout(0.25)
        except BaseException:
            listener.close()
            raise
        self._listener = listener
        self._thread = threading.Thread(
            target=self._serve,
            name="typed-state-owner-gateway",
            daemon=True,
        )
        self._thread.start()

    def _quiesce(self) -> None:
        """Stop admission and close channels without taking the owner lock.

        A client transaction retains the owner-wide lock between ``begin``
        and ``commit``/``rollback``.  Closing its channel is therefore the
        only safe way for the outer server to make an abandoned transaction
        unwind before that server waits for the same lock.
        """

        self._stop.set()
        listener = self._listener
        self._listener = None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        accept_thread = self._thread
        if (
            accept_thread is not None
            and accept_thread is not threading.current_thread()
        ):
            accept_thread.join(timeout=3.0)
        with self._clients_lock:
            channels = tuple(self._channels)
        for channel in channels:
            try:
                channel.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                channel.close()
            except OSError:
                pass
        if accept_thread is not None and accept_thread.is_alive():
            raise TypedStateOwnerProtocolError(
                "typed owner accept loop did not quiesce"
            )

    def stop(self) -> None:
        self._quiesce()
        with self._clients_lock:
            clients = tuple(self._clients)
        deadline = time.monotonic() + 5.0
        for thread in clients:
            if thread is threading.current_thread():
                continue
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        with self._clients_lock:
            live_clients = tuple(
                thread for thread in self._clients if thread.is_alive()
            )
        if live_clients:
            raise TypedStateOwnerProtocolError(
                "typed owner clients did not quiesce before teardown"
            )
        with self._transaction_lock:
            with self._eaaef_service_bind_lock:
                plan_service = self._eaaef_plan_r2_owner_service
                command_service = self._eaaef_typed_owner_command_service
                if plan_service is not None:
                    plan_service.close()
                if command_service is not None:
                    command_service.close()
                self._eaaef_plan_r2_owner_service = None
                self._eaaef_typed_owner_command_service = None
        self._thread = None
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass

    def capability(self) -> dict[str, Any]:
        return {
            "interface": TYPED_STATE_OWNER_INTERFACE,
            "available": self._listener is not None and not self._stop.is_set(),
            "server_owned": True,
            "exclusive_connection": True,
            "raw_sql_permitted": False,
            "catalog_id": self.catalog_id,
            "operation_count": len(self.catalog),
            "request_count": self._request_count,
            "committed_transactions": self._committed_transactions,
            "active_grants": len(self._grants),
            "revoked_grants": len(self._revoked_grants),
            "grant_expiry_required": True,
            "kernel_peer_credentials_required": True,
            "typed_event_wait_bound": self._event_wait_handler is not None,
            # Owner status is canonical DAG-JSON and therefore float-free.
            "typed_event_wait_maximum_seconds": int(
                MAX_REMOTE_EVENT_WAIT_SECONDS
            ),
            "commit_observer_bound": self._commit_observer is not None,
            "last_observer_error_type": self._last_observer_error_type,
            "last_error_type": self._last_error_type,
        }

    @staticmethod
    def _authorize_event_wait_identity(
        grant: OwnerClientGrant,
        *,
        consumer_id: str,
        subscription_id: str = "",
    ) -> None:
        """Require an exact bounded event-consumer capability.

        Tenant and federation are deliberately absent from the wait payload;
        they are resolved by the server from the durable subscription.  The
        grant must nevertheless carry both scopes, and the server handler
        compares those resolved values before entering the condition wait.
        """

        if not grant.tenant_id or not grant.federation_id:
            raise TypedStateOwnerAuthorizationError(
                "event wait grant requires tenant and federation scope"
            )
        scopes = dict(grant.entity_scopes)
        if scopes.get("consumer_id", consumer_id) != consumer_id:
            raise TypedStateOwnerAuthorizationError(
                "event wait consumer differs from the client grant"
            )
        if subscription_id and (
            scopes.get("subscription_id", subscription_id) != subscription_id
        ):
            raise TypedStateOwnerAuthorizationError(
                "event wait subscription differs from the client grant"
            )

    @staticmethod
    def _authorize_event_wait_deadline(
        grant: OwnerClientGrant,
        *,
        deadline: str,
    ) -> None:
        try:
            parsed = datetime.fromisoformat(str(deadline).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("naive deadline")
            deadline_ms = int(parsed.timestamp() * 1_000)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypedStateOwnerProtocolError(
                "event wait deadline is invalid"
            ) from exc
        now_ms = int(time.time() * 1_000)
        maximum_ms = int(MAX_REMOTE_EVENT_WAIT_SECONDS * 1_000)
        if deadline_ms < now_ms or deadline_ms - now_ms > maximum_ms:
            raise TypedStateOwnerAuthorizationError(
                "event wait deadline is outside the service bound"
            )
        if deadline_ms >= grant.expires_at:
            raise TypedStateOwnerAuthorizationError(
                "event wait deadline exceeds the client grant"
            )

    def _serve(self) -> None:
        while not self._stop.is_set():
            listener = self._listener
            if listener is None:
                return
            try:
                channel, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            thread = threading.Thread(
                target=self._serve_client,
                args=(channel,),
                name="typed-state-owner-client",
                daemon=True,
            )
            with self._clients_lock:
                self._clients.add(thread)
                self._channels.add(channel)
            thread.start()

    @staticmethod
    def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], kind: str) -> None:
        unknown = set(payload) - allowed
        if unknown:
            raise TypedStateOwnerProtocolError(
                f"{kind} contains unknown normative fields"
            )

    def _serve_client(self, channel: socket.socket) -> None:
        transaction_active = False
        command: StateCommand | None = None
        mutation_manifest: list[tuple[str, dict[str, Any]]] = []
        semantic_authority: dict[str, Any] = {}
        semantic_authority_captured = False
        client_id = ""
        grant: OwnerClientGrant | None = None
        status_session_grant_id = ""
        session_id = ""
        try:
            channel.settimeout(30.0)
            peer_identity = _kernel_peer_identity(channel)
            opened = _receive_frame(channel)
            self._reject_unknown(
                opened,
                {
                    "schema",
                    "action",
                    "request_id",
                    "token",
                    "client_id",
                    "process_birth_id",
                    "store_id",
                },
                "open request",
            )
            supplied_token = str(opened.get("token") or "")
            action = str(opened.get("action") or "")
            client_id = str(opened.get("client_id") or "").strip()
            process_birth_id = str(opened.get("process_birth_id") or "").strip()
            if (
                opened.get("schema") != TYPED_STATE_OWNER_SCHEMA
                or opened.get("store_id") != self.store_id
                or not 16 <= len(supplied_token) <= 256
                or not process_birth_id
                or len(process_birth_id) > 256
            ):
                raise TypedStateOwnerAuthorizationError("gateway authentication failed")
            if action == "open_status":
                supplied_digest = hashlib.sha256(
                    supplied_token.encode("utf-8")
                ).digest()
                with self._grants_lock:
                    expected_digest = self._status_bootstrap_token_digest
                    expected_uid = self._status_bootstrap_uid
                if (
                    expected_digest is None
                    or not hmac.compare_digest(supplied_digest, expected_digest)
                    or peer_identity[1] != expected_uid
                    or client_id != STATUS_BOOTSTRAP_CLIENT_ID
                ):
                    raise TypedStateOwnerAuthorizationError(
                        "gateway authentication failed"
                    )
                grant = self._issue_status_session_grant(
                    peer_identity=peer_identity,
                    process_birth_id=process_birth_id,
                )
                status_session_grant_id = grant.grant_id
            elif action == "open":
                with self._grants_lock:
                    for candidate_token, candidate_grant in self._grants.items():
                        if hmac.compare_digest(supplied_token, candidate_token):
                            grant = candidate_grant
                            break
                if grant is None:
                    raise TypedStateOwnerAuthorizationError(
                        "gateway authentication failed"
                    )
            else:
                raise TypedStateOwnerAuthorizationError("gateway authentication failed")
            grant = self._require_active_grant(
                grant,
                peer_identity=peer_identity,
            )
            if (
                not client_id
                or len(client_id) > 256
                or client_id != grant.client_id
                or (
                    grant.process_birth_id
                    and process_birth_id != grant.process_birth_id
                )
            ):
                raise TypedStateOwnerAuthorizationError("gateway client identity is invalid")
            session_id = f"session:owner:{uuid.uuid4()}"
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with self._transaction_lock:
                self._connection.execute(
                    """
                    INSERT INTO client_sessions (
                        session_id, server_id, owner_id, process_birth_id,
                        attached_at, last_seen_at, fence_epoch, generation,
                        status, revision
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'attached', 0)
                    """,
                    [
                        session_id,
                        str(self.identity.get("server_id") or ""),
                        client_id,
                        process_birth_id,
                        now,
                        now,
                        int(self.identity.get("fence_epoch") or 0),
                        int(self.identity.get("generation") or 0),
                    ],
                )
            _send_frame(
                channel,
                {
                    "schema": TYPED_STATE_OWNER_SCHEMA,
                    "request_id": str(opened.get("request_id") or ""),
                    "ok": True,
                    "identity": dict(self.identity),
                    "catalog_id": self.catalog_id,
                    "session_id": session_id,
                    "grant": grant.public_dict(),
                },
            )
            # Admitted clients wait without periodic traffic.  Shutdown closes
            # every tracked channel, and expiry/revocation is rechecked before
            # the next request, so an idle socket needs no polling timeout.
            channel.settimeout(None)
            while not self._stop.is_set():
                request = _receive_frame(channel)
                action = str(request.get("action") or "")
                request_id = str(request.get("request_id") or "")
                try:
                    grant = self._require_active_grant(
                        grant,
                        peer_identity=peer_identity,
                    )
                    if action == "begin":
                        self._reject_unknown(
                            request,
                            {"schema", "action", "request_id", "command"},
                            "begin request",
                        )
                        if transaction_active:
                            raise TypedStateOwnerProtocolError("transaction already active")
                        command = StateCommand.from_dict(request.get("command") or {})
                        if not self._transaction_lock.acquire(timeout=30.0):
                            raise TypedStateOwnerProtocolError(
                                "owner transaction admission timed out"
                            )
                        try:
                            grant = self._require_active_grant(
                                grant,
                                peer_identity=peer_identity,
                            )
                            self._authorize_command(command, client_id, grant=grant)
                            self._connection.execute("BEGIN TRANSACTION")
                        except BaseException:
                            command = None
                            self._transaction_lock.release()
                            raise
                        transaction_active = True
                        mutation_manifest = []
                        semantic_authority = {}
                        semantic_authority_captured = False
                        response = {"ok": True}
                    elif action == "execute":
                        self._reject_unknown(
                            request,
                            {"schema", "action", "request_id", "operation", "parameters"},
                            "execute request",
                        )
                        operation_name = str(request.get("operation") or "")
                        operation = self.catalog.get(operation_name)
                        if operation is None:
                            raise TypedStateOwnerProtocolError("unknown server operation")
                        parameters = _closed_parameters(request.get("parameters"))
                        if len(parameters) != operation.parameter_count:
                            raise TypedStateOwnerProtocolError("operation parameter count differs")
                        if operation.name not in grant.allowed_operations:
                            raise TypedStateOwnerAuthorizationError(
                                "operation is outside the client grant"
                            )
                        self._authorize_operation_scope(
                            operation,
                            parameters,
                            grant=grant,
                            command=command if transaction_active else None,
                        )
                        if operation.mutation and not transaction_active:
                            raise TypedStateOwnerAuthorizationError(
                                "mutation requires an admitted command transaction"
                            )
                        if operation.mutation and transaction_active:
                            if command is None:
                                raise TypedStateOwnerAuthorizationError(
                                    "mutation transaction has no admitted command"
                                )
                            self._authorize_mutation(
                                command, operation, parameters, grant=grant
                            )
                            if not semantic_authority_captured:
                                semantic_authority = self._capture_semantic_authority(
                                    command,
                                    grant=grant,
                                )
                                semantic_authority_captured = True
                            self._record_manifest_mutation(
                                command,
                                operation,
                                parameters,
                                mutation_manifest,
                            )
                        if transaction_active:
                            result = self._execute(operation, parameters)
                        else:
                            with self._transaction_lock:
                                grant = self._require_active_grant(
                                    grant,
                                    peer_identity=peer_identity,
                                )
                                result = self._execute(operation, parameters)
                        response = result
                    elif action == "wait_events":
                        self._reject_unknown(
                            request,
                            {"schema", "action", "request_id", "wait_request"},
                            "event wait request",
                        )
                        if transaction_active:
                            raise TypedStateOwnerAuthorizationError(
                                "event wait is unavailable inside a transaction"
                            )
                        if "event.wait" not in grant.allowed_operations:
                            raise TypedStateOwnerAuthorizationError(
                                "event wait is outside the client grant"
                            )
                        from ..federation.events import EventWaitRequest

                        wait_request = EventWaitRequest.from_dict(
                            request.get("wait_request") or {}
                        )
                        self._authorize_event_wait_identity(
                            grant,
                            consumer_id=wait_request.consumer_id,
                            subscription_id=wait_request.subscription_id,
                        )
                        self._authorize_event_wait_deadline(
                            grant,
                            deadline=wait_request.deadline,
                        )
                        handler = self._event_wait_handler
                        if not callable(handler):
                            raise TypedStateOwnerProtocolError(
                                "server-owned event wait is unavailable"
                            )
                        batch = handler(wait_request, grant)
                        response = {"ok": True, "batch": batch.to_dict()}
                    elif action in {"cancel_event_wait", "clear_event_wait_cancellation"}:
                        self._reject_unknown(
                            request,
                            {"schema", "action", "request_id", "consumer_id"},
                            f"{action} request",
                        )
                        if transaction_active:
                            raise TypedStateOwnerAuthorizationError(
                                "event wait control is unavailable inside a transaction"
                            )
                        operation_name = (
                            "event.wait.cancel"
                            if action == "cancel_event_wait"
                            else "event.wait.clear_cancellation"
                        )
                        if operation_name not in grant.allowed_operations:
                            raise TypedStateOwnerAuthorizationError(
                                "event wait control is outside the client grant"
                            )
                        consumer_id = str(request.get("consumer_id") or "").strip()
                        if not consumer_id or len(consumer_id) > 256:
                            raise TypedStateOwnerProtocolError(
                                "event wait consumer identity is invalid"
                            )
                        self._authorize_event_wait_identity(
                            grant,
                            consumer_id=consumer_id,
                        )
                        handler = (
                            self._event_wait_cancel_handler
                            if action == "cancel_event_wait"
                            else self._event_wait_clear_handler
                        )
                        if not callable(handler):
                            raise TypedStateOwnerProtocolError(
                                "server-owned event wait control is unavailable"
                            )
                        handler(consumer_id, grant)
                        response = {"ok": True}
                    elif action in _EAAEF_COMMAND_SERVICE_OPERATIONS:
                        eaaef_request_fields = {
                            "schema",
                            "action",
                            "request_id",
                            "envelope",
                            "merge_admission_cid",
                            "operational_capability_cid",
                        }
                        if set(request) != eaaef_request_fields:
                            raise TypedStateOwnerProtocolError(
                                f"{action} request must contain the exact "
                                "normative fields"
                            )
                        if request.get("schema") != TYPED_STATE_OWNER_SCHEMA:
                            raise TypedStateOwnerProtocolError(
                                "EAAEF request schema differs"
                            )
                        if transaction_active:
                            raise TypedStateOwnerAuthorizationError(
                                "EAAEF service is unavailable inside a transaction"
                            )
                        if action not in grant.allowed_operations:
                            raise TypedStateOwnerAuthorizationError(
                                "EAAEF service is outside the client grant"
                            )
                        with self._eaaef_service_bind_lock:
                            service = self._eaaef_typed_owner_command_service
                        if service is None:
                            raise TypedStateOwnerProtocolError(
                                "server-owned EAAEF service is unavailable"
                            )
                        merge_admission_cid = request.get(
                            "merge_admission_cid"
                        )
                        operational_capability_cid = request.get(
                            "operational_capability_cid"
                        )
                        if (
                            type(merge_admission_cid) is not str
                            or merge_admission_cid != service.admission_cid
                            or type(operational_capability_cid) is not str
                            or operational_capability_cid
                            != service.operational_capability_cid
                        ):
                            raise TypedStateOwnerAuthorizationError(
                                "EAAEF request authority differs from the bound service"
                            )
                        from .quack_command_authorization import (
                            AuthorizedStateCommand,
                        )

                        envelope = AuthorizedStateCommand.from_dict(
                            request.get("envelope") or {}
                        )
                        with self._transaction_lock:
                            grant = self._require_active_grant(
                                grant,
                                peer_identity=peer_identity,
                            )
                            if self._stop.is_set():
                                raise TypedStateOwnerAuthorizationError(
                                    "EAAEF service is stopping"
                                )
                            if action == "eaaef.command.submit":
                                receipt = service.submit_authorized_operation(
                                    envelope
                                )
                            else:
                                receipt = (
                                    service.lookup_authorized_operation_receipt(
                                        envelope
                                    )
                                )
                        response = {
                            "ok": True,
                            "receipt": (
                                None if receipt is None else dict(receipt)
                            ),
                        }
                    elif action in _EAAEF_PLAN_R2_SERVICE_OPERATIONS:
                        plan_request_fields = {
                            "schema",
                            "action",
                            "request_id",
                            "remote_capability_cid",
                            "plan_r2_operational_capability_cid",
                            "plan_r2_authorization_cid",
                            "envelope",
                            "operation_payload",
                        }
                        if set(request) != plan_request_fields:
                            raise TypedStateOwnerProtocolError(
                                f"{action} request must contain the exact "
                                "normative fields"
                            )
                        if request.get("schema") != TYPED_STATE_OWNER_SCHEMA:
                            raise TypedStateOwnerProtocolError(
                                "Plan-R2 request schema differs"
                            )
                        if transaction_active:
                            raise TypedStateOwnerAuthorizationError(
                                "Plan-R2 service is unavailable inside a transaction"
                            )
                        if action not in grant.allowed_operations:
                            raise TypedStateOwnerAuthorizationError(
                                "Plan-R2 service is outside the client grant"
                            )
                        with self._eaaef_service_bind_lock:
                            plan_service = self._eaaef_plan_r2_owner_service
                        if plan_service is None:
                            raise TypedStateOwnerProtocolError(
                                "server-owned Plan-R2 service is unavailable"
                            )
                        request_size = len(
                            json.dumps(
                                request,
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=False,
                            ).encode("utf-8")
                        )
                        if request_size > plan_service.maximum_request_bytes:
                            raise TypedStateOwnerProtocolError(
                                "Plan-R2 request exceeds its signed byte bound"
                            )
                        authority_fields = {
                            "remote_capability_cid": (
                                plan_service.remote_capability_cid
                            ),
                            "plan_r2_operational_capability_cid": (
                                plan_service.operational_capability_cid
                            ),
                            "plan_r2_authorization_cid": (
                                plan_service.authorization_cid
                            ),
                        }
                        if any(
                            type(request.get(name)) is not str
                            or request.get(name) != expected
                            for name, expected in authority_fields.items()
                        ):
                            raise TypedStateOwnerAuthorizationError(
                                "Plan-R2 request authority differs from the bound service"
                            )
                        operation_payload = request.get("operation_payload")
                        if (
                            type(operation_payload) is not dict
                            or operation_payload.get("operation") != action
                        ):
                            raise TypedStateOwnerProtocolError(
                                "Plan-R2 action differs from its operation payload"
                            )
                        from .quack_command_authorization import (
                            AuthorizedStateCommand,
                        )

                        plan_envelope = AuthorizedStateCommand.from_dict(
                            request.get("envelope") or {}
                        )
                        with self._transaction_lock:
                            grant = self._require_active_grant(
                                grant,
                                peer_identity=peer_identity,
                            )
                            if self._stop.is_set():
                                raise TypedStateOwnerAuthorizationError(
                                    "Plan-R2 service is stopping"
                                )
                            plan_result = (
                                plan_service.submit_authorized_plan_r2_operation(
                                    plan_envelope,
                                    operation_payload,
                                )
                            )
                        response = {
                            "ok": True,
                            "result": dict(plan_result),
                        }
                        response_size = len(
                            json.dumps(
                                {
                                    "schema": TYPED_STATE_OWNER_SCHEMA,
                                    "request_id": request_id,
                                    **response,
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=False,
                            ).encode("utf-8")
                        )
                        if response_size > plan_service.maximum_response_bytes:
                            raise TypedStateOwnerProtocolError(
                                "Plan-R2 response exceeds its signed byte bound"
                            )
                    elif action in {"commit", "rollback"}:
                        self._reject_unknown(
                            request,
                            {"schema", "action", "request_id"},
                            f"{action} request",
                        )
                        if not transaction_active:
                            raise TypedStateOwnerProtocolError("no active transaction")
                        if action == "commit":
                            if command is None:
                                raise TypedStateOwnerAuthorizationError(
                                    "transaction manifest has no admitted command"
                                )
                            self._validate_transaction_manifest(
                                command,
                                mutation_manifest,
                                semantic_authority=semantic_authority,
                            )
                            self._connection.commit()
                            self._committed_transactions += 1
                            observer = self._commit_observer
                            committed_command = command
                            committed_manifest = tuple(mutation_manifest)
                        else:
                            self._connection.rollback()
                            observer = None
                            committed_command = None
                            committed_manifest = ()
                        transaction_active = False
                        command = None
                        mutation_manifest = []
                        semantic_authority = {}
                        semantic_authority_captured = False
                        self._transaction_lock.release()
                        if callable(observer):
                            try:
                                # The sole DuckDB transaction lock protects only
                                # authoritative state.  Downstream notifications
                                # run after it is released so a blocked observer
                                # cannot stall unrelated owner operations.
                                observer(committed_command, committed_manifest)
                            except BaseException as observer_error:
                                # The authoritative transaction is already
                                # durable.  Notification is an optimization;
                                # backlog replay remains authoritative and a
                                # post-commit callback must not manufacture an
                                # ambiguous command failure for the client.
                                self._last_observer_error_type = type(
                                    observer_error
                                ).__name__
                        response = {"ok": True}
                    elif action == "close":
                        self._reject_unknown(
                            request,
                            {"schema", "action", "request_id"},
                            "close request",
                        )
                        _send_frame(
                            channel,
                            {
                                "schema": TYPED_STATE_OWNER_SCHEMA,
                                "request_id": request_id,
                                "ok": True,
                            },
                        )
                        return
                    else:
                        raise TypedStateOwnerProtocolError("unknown typed state-owner action")
                    self._request_count += 1
                    _send_frame(
                        channel,
                        {
                            "schema": TYPED_STATE_OWNER_SCHEMA,
                            "request_id": request_id,
                            **response,
                        },
                    )
                except BaseException as exc:
                    if transaction_active:
                        try:
                            self._connection.rollback()
                        except Exception:
                            pass
                        transaction_active = False
                        command = None
                        mutation_manifest = []
                        semantic_authority = {}
                        semantic_authority_captured = False
                        self._transaction_lock.release()
                    _send_frame(
                        channel,
                        {
                            "schema": TYPED_STATE_OWNER_SCHEMA,
                            "request_id": request_id,
                            "ok": False,
                            "error_code": self._error_code(exc),
                            "error_type": type(exc).__name__,
                        },
                    )
        except (OSError, TypedStateOwnerError, ValueError) as exc:
            # Authentication and protocol failures are deliberately silent.
            self._last_error_type = type(exc).__name__
            return
        finally:
            if transaction_active:
                try:
                    self._connection.rollback()
                except Exception:
                    pass
                self._transaction_lock.release()
            try:
                channel.close()
            except OSError:
                pass
            if status_session_grant_id:
                self._retire_status_session_grant(status_session_grant_id)
            with self._clients_lock:
                self._clients.discard(threading.current_thread())
                self._channels.discard(channel)

    def _authorize_command(
        self,
        command: StateCommand,
        client_id: str,
        *,
        grant: OwnerClientGrant,
    ) -> None:
        # The caller's authority_class is observational.  Authority comes from
        # this server-minted credential, the attached session row, and exact
        # live generation/fence/revision checks.
        if command.store_id != self.store_id:
            raise TypedStateOwnerAuthorizationError("command store differs")
        operation = str(command.parameters.get("operation") or "")
        if operation not in _COMMAND_MUTATION_CATALOG:
            raise TypedStateOwnerAuthorizationError(
                "command operation is absent from the server policy catalog"
            )
        if operation not in grant.allowed_command_operations:
            raise TypedStateOwnerAuthorizationError(
                "command operation is outside the client grant"
            )
        expected_kind = {
            "budget.release": "release",
            "supervisor.runtime.attest": "claim",
            "subagent.slot.reserve": "claim",
            "subagent.slot.release": "release",
            "task.status.cas": "claim",
            "task.status.cas.receipt": "claim",
            TYPED_DATABASE_CLAIM_RECOVERY_COMMAND: "claim",
        }.get(operation, "append")
        if command.command_kind.value != expected_kind:
            raise TypedStateOwnerAuthorizationError(
                "command kind differs from the server operation policy"
            )
        if operation == "task.retry.cooldown.record":
            digest = _retry_cooldown_command_digest(command.parameters)
            if (
                command.command_id != f"cmd:retry-cooldown:{digest}"
                or command.idempotency_key
                != f"executor-retry-cooldown:{digest}"
            ):
                raise TypedStateOwnerAuthorizationError(
                    "retry cooldown replay identity differs from its parameters"
                )
        if operation == TYPED_DATABASE_CLAIM_RECOVERY_COMMAND:
            digest = _dead_claim_recovery_command_digest(command.parameters)
            if (
                command.command_id != f"cmd:dead-claim-recovery:{digest}"
                or command.idempotency_key
                != f"executor-dead-claim-recovery:{digest}"
            ):
                raise TypedStateOwnerAuthorizationError(
                    "dead claim recovery replay identity differs from its parameters"
                )
        if operation in {"task.status.cas", "task.status.cas.receipt"}:
            requested_status = command.parameters.get("status")
            if (
                not isinstance(requested_status, str)
                or requested_status not in TYPED_TASK_STATUS_VOCABULARY
            ):
                raise TypedStateOwnerAuthorizationError(
                    "task status is outside the closed command vocabulary"
                )
            if (
                operation == "task.status.cas"
                and requested_status == "in_progress"
                and grant.client_id.startswith(
                    "database-implementation-daemon:"
                )
            ):
                raise TypedStateOwnerAuthorizationError(
                    "typed executor in-progress transition requires a claim receipt"
                )
        if operation in _FEDERATION_COMMANDS:
            for field in ("tenant_id", "federation_id"):
                value = str(command.parameters.get(field) or "").strip()
                if not value:
                    raise TypedStateOwnerAuthorizationError(
                        "federation command lacks exact tenant/federation scope"
                    )
            if grant.tenant_id and command.parameters["tenant_id"] != grant.tenant_id:
                raise TypedStateOwnerAuthorizationError(
                    "command tenant differs from the client grant"
                )
            if (
                grant.federation_id
                and command.parameters["federation_id"] != grant.federation_id
            ):
                raise TypedStateOwnerAuthorizationError(
                    "command federation differs from the client grant"
                )
        for field, expected in grant.entity_scopes:
            if str(command.parameters.get(field) or "") != expected:
                raise TypedStateOwnerAuthorizationError(
                    "command entity scope differs from the client grant"
                )
        row = self._connection.execute(
            """
            SELECT owner_id, generation, fence_epoch, status
            FROM client_sessions WHERE session_id = ? LIMIT 1
            """,
            [command.session_id],
        ).fetchone()
        if row is None or str(row[0]) != client_id or str(row[3]) != "attached":
            raise TypedStateOwnerAuthorizationError("command session is not admitted")
        if int(row[1]) != command.expected_generation or int(row[2]) != command.fence_epoch:
            raise TypedStateOwnerAuthorizationError("command session generation is stale")
        head = self._connection.execute(
            """
            SELECT generation, revision, fence_epoch FROM store_generations
            ORDER BY generation DESC LIMIT 1
            """
        ).fetchone()
        if head is None or (
            int(head[0]) != command.expected_generation
            or int(head[2]) != command.fence_epoch
        ):
            raise TypedStateOwnerAuthorizationError("command head or fence is stale")
        # Revision contention is not an authorization failure.  The command
        # now owns the exclusive state-owner transaction lock, and
        # StateTransaction performs the exact revision CAS before any domain
        # mutation.  Let it return OptimisticConflictError so the bounded
        # client retry can reload the current head; treating ordinary writer
        # contention as authorization_denied makes concurrent supervisors
        # fail permanently instead of retrying safely.

    @staticmethod
    def _authorize_operation_scope(
        operation: OwnerOperation,
        parameters: Sequence[Any],
        *,
        grant: OwnerClientGrant,
        command: StateCommand | None,
    ) -> None:
        """Prevent a scoped credential from querying a different authority slice."""

        if operation.mutation:
            return
        command_bindings = {
            field: str(command.parameters[field])
            for field in _SCOPE_FIELDS
            if command is not None
            and command.parameters.get(field) not in (None, "")
        }
        grant_bindings = {
            **({"tenant_id": grant.tenant_id} if grant.tenant_id else {}),
            **({"federation_id": grant.federation_id} if grant.federation_id else {}),
            **dict(grant.entity_scopes),
        }
        if operation.name in _STATUS_OWNER_PLAN_READS:
            if (
                grant.client_id == STATUS_BOOTSTRAP_CLIENT_ID
                and grant.authority_profile
                == "dedicated_store_status_portfolio"
                and grant.tenant_id
                and grant.federation_id
                and all(
                    dict(grant.entity_scopes).get(name)
                    for name in _STATUS_BOOTSTRAP_ENTITY_SCOPE_NAMES
                )
            ):
                # The legacy CASF-000..043 task population is a dedicated-store
                # portfolio, not tenant-shaped rows. This explicit profile is
                # owner-issued only after binding the live federation slice;
                # it must never be inferred from an unscoped status token.
                return
            raise TypedStateOwnerAuthorizationError(
                "portfolio task read requires the dedicated status profile"
            )
        if not command_bindings and not grant_bindings:
            return
        # The generation head is store-global and only reports the already
        # admitted command's exact CAS authority.  Other transaction reads do
        # not receive an ambient exemption: in particular, idempotency bodies
        # may contain tenant-scoped result data.
        if command is not None and operation.name == "txn_load_generation":
            return
        if command is not None and operation.name in {
            "lookup_idempotency",
            "txn_lookup_idempotency",
        }:
            if (
                len(parameters) != 1
                or str(parameters[0]) != command.idempotency_key
            ):
                raise TypedStateOwnerAuthorizationError(
                    "idempotency query differs from the admitted command"
                )
            return
        bound = {
            name: parameters[index]
            for index, name in enumerate(operation.parameter_names)
        }

        def present_scope(field: str) -> list[str]:
            candidates = (field, f"scope_{field}", f"unique_{field}")
            return [candidate for candidate in candidates if candidate in bound]

        # A server-minted client grant is an outer security boundary.  Unlike
        # a command's child identity, it may never be silently widened to a
        # parent query unless that exact named read is explicitly independent.
        for field, expected in grant_bindings.items():
            present = present_scope(field)
            if present:
                if any(str(bound[candidate]) != expected for candidate in present):
                    raise TypedStateOwnerAuthorizationError(
                        "query scope differs from the client grant"
                    )
                continue
            if operation.name not in _SCOPE_INDEPENDENT_READS:
                raise TypedStateOwnerAuthorizationError(
                    "scoped grant cannot use an unscoped query"
                )

        tenant_present = bool(present_scope("tenant_id"))
        federation_present = bool(present_scope("federation_id"))
        for field, expected in command_bindings.items():
            present = present_scope(field)
            if present:
                if any(str(bound[candidate]) != expected for candidate in present):
                    raise TypedStateOwnerAuthorizationError(
                        "query scope differs from the admitted command"
                    )
                continue
            if operation.name in _SCOPE_INDEPENDENT_READS:
                continue
            # Creation/registration reads parent authority before the child
            # row exists.  A client with no narrower entity grant may perform
            # that read only when the operation still binds the command's
            # exact tenant and federation.  Entity-scoped grants remain
            # fail-closed in the loop above.
            if (
                field not in {"tenant_id", "federation_id"}
                and tenant_present
                and federation_present
            ):
                continue
            raise TypedStateOwnerAuthorizationError(
                "command-scoped transaction cannot use an unscoped query"
            )

    def _capture_semantic_authority(
        self,
        command: StateCommand,
        *,
        grant: OwnerClientGrant,
    ) -> dict[str, Any]:
        """Resolve pre-mutation authority for child-executable commands.

        Client-side registry checks remain defense in depth. This snapshot is
        captured by the exclusive owner immediately before the first mutation,
        while its transaction lock is held, so a client cannot replace
        lifecycle, delivery, or cursor semantics with a structurally valid
        low-level manifest.
        """

        operation = str(command.parameters.get("operation") or "")
        if operation == TYPED_DATABASE_CLAIM_RECOVERY_COMMAND:
            recovery = _validated_dead_claim_recovery_parameters(
                command.parameters
            )
            values = dict(recovery["cooldown_parameters"])
            task_rows = self._connection.execute(
                """
                SELECT status, revision, body_json FROM tasks
                WHERE task_cid = ? LIMIT 2
                """,
                [values["task_cid"]],
            ).fetchall()
            if len(task_rows) != 1:
                raise TypedStateOwnerAuthorizationError(
                    "dead claim recovery task authority is absent or ambiguous"
                )
            task_row = task_rows[0]
            if (
                str(task_row[0] or "").strip().lower() != "in_progress"
                or type(task_row[1]) is not int
                or int(task_row[1]) != values["expected_task_revision"]
            ):
                raise TypedStateOwnerAuthorizationError(
                    "dead claim recovery task revision is stale"
                )
            try:
                prior_body = json.loads(str(task_row[2] or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise TypedStateOwnerAuthorizationError(
                    "dead claim recovery prior task body is malformed"
                ) from exc
            prior_receipt = (
                prior_body.get("completion_receipt")
                if isinstance(prior_body, Mapping)
                else None
            )
            if not isinstance(prior_receipt, Mapping):
                raise TypedStateOwnerAuthorizationError(
                    "dead claim recovery has no typed reservation"
                )
            prior_identity = _validated_database_claim_identity(prior_receipt)
            exact_identity = {
                "claim_id": values["claim_id"],
                "attempt_id": values["attempt_id"],
                "attempt_number": values["attempt_number"],
                "lease_id": values["lease_id"],
                "owner_session_id": values["owner_session_id"],
                "fencing_token": values["fencing_token"],
                "fence_epoch": values["fence_epoch"],
            }
            historic_attestation = (
                _validated_database_claim_process_attestation(prior_receipt)
            )
            try:
                execution_route = TaskExecutionRouteBinding.from_dict(
                    prior_receipt.get("execution_route_binding")
                ).to_dict()
            except (TypeError, ValueError, TaskSourceIntegrityError) as exc:
                raise TypedStateOwnerAuthorizationError(
                    "dead claim recovery reservation has no exact execution route"
                ) from exc
            if (
                prior_receipt.get("operation") != "database_claim"
                or prior_receipt.get("claim_phase_schema")
                != TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA
                or any(
                    not _strict_scalar_equal(
                        prior_identity.get(name), expected
                    )
                    for name, expected in exact_identity.items()
                )
                or not _strict_scalar_equal(
                    prior_receipt.get("claimed_from_revision"),
                    values["expected_task_revision"] - 1,
                )
                or historic_attestation.get("client_id") != grant.client_id
                or execution_route["task_cid"] != values["task_cid"]
                or prior_receipt.get("execution_route_policy_id")
                != execution_route["policy_id"]
                or not _strict_scalar_equal(
                    prior_receipt.get("execution_route_origin_revision"),
                    execution_route["task_revision"],
                )
            ):
                raise TypedStateOwnerAuthorizationError(
                    "dead claim recovery reservation differs from its exact authority"
                )
            current_attestation = _claim_process_attestation(grant)
            if (
                historic_attestation["process_birth_id"]
                == current_attestation["process_birth_id"]
            ):
                raise TypedStateOwnerAuthorizationError(
                    "dead claim recovery cannot replace its current process birth"
                )
            historic_birth = ProcessBirthIdentity(
                pid=int(historic_attestation["pid"]),
                start_time_ticks=int(
                    historic_attestation["start_time_ticks"]
                ),
                boot_id=str(historic_attestation["boot_id"]),
                parent_pid=int(historic_attestation["parent_pid"]),
            )
            try:
                liveness = OwnerLiveness(
                    self._owner_liveness_probe(historic_birth)
                )
            except BaseException:
                liveness = OwnerLiveness.UNKNOWN
            if liveness is not OwnerLiveness.DEAD:
                raise TypedStateOwnerAuthorizationError(
                    "dead claim recovery requires a provably dead historic process"
                )
            queue_rows = self._connection.execute(
                "SELECT task_cid FROM leases WHERE task_cid = ? LIMIT 2",
                [values["task_cid"]],
            ).fetchall()
            if queue_rows:
                raise TypedStateOwnerAuthorizationError(
                    "dead claim recovery requires exact cooldown absence"
                )
            reservation_cid = content_identity(
                {"typed_database_claim_reservation": dict(prior_receipt)}
            )
            recovery_receipt = {
                "schema": TYPED_DATABASE_CLAIM_RECOVERY_SCHEMA,
                "operation": TYPED_DATABASE_CLAIM_RECOVERY_OPERATION,
                **exact_identity,
                "recovered_claim_phase_schema": (
                    TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA
                ),
                "recovered_claimed_from_revision": int(
                    prior_receipt["claimed_from_revision"]
                ),
                "recovered_reservation_cid": reservation_cid,
                "recovered_claim_process_attestation": dict(
                    historic_attestation
                ),
                "recovery_process_attestation": dict(current_attestation),
                "recovered_from_revision": values["expected_task_revision"],
                "queue_reason": values["reason"],
                "backoff_ms": 0,
                "retry_not_before_ms": values["retry_not_before_ms"],
                "control_expected_revision": values[
                    "expected_task_revision"
                ],
                "execution_route_binding": execution_route,
                "execution_route_policy_id": execution_route["policy_id"],
                "execution_route_origin_revision": int(
                    execution_route["task_revision"]
                ),
            }
            expected_body = dict(prior_body)
            expected_body["completion_receipt"] = recovery_receipt
            expected_body_json = canonical_json_bytes(expected_body).decode(
                "utf-8"
            )
            if (
                recovery["body"] != expected_body
                or command.parameters.get("body_json")
                != expected_body_json
            ):
                raise TypedStateOwnerAuthorizationError(
                    "dead claim recovery receipt differs from owner-derived authority"
                )
            return {
                "operation": operation,
                "task_cid": values["task_cid"],
                "expected_revision": values["expected_task_revision"],
                "body_json": expected_body_json,
                "cooldown_parameters": values,
                "recovery_receipt": recovery_receipt,
                "historic_liveness": liveness.value,
            }
        if operation == "task.status.cas.receipt":
            try:
                next_body = json.loads(str(command.parameters.get("body_json") or ""))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise TypedStateOwnerAuthorizationError(
                    "task status receipt body is malformed"
                ) from exc
            next_receipt = (
                next_body.get("completion_receipt")
                if isinstance(next_body, Mapping)
                else None
            )
            phase_schema = (
                str(next_receipt.get("claim_phase_schema") or "")
                if isinstance(next_receipt, Mapping)
                else ""
            )
            typed_executor_claim = bool(
                grant.client_id.startswith("database-implementation-daemon:")
                and str(command.parameters.get("status") or "").strip()
                == "in_progress"
            )
            strict_rejection_operation = (
                next_receipt.get("operation")
                if isinstance(next_receipt, Mapping)
                else None
            )
            typed_strict_rejection = bool(
                grant.client_id.startswith("database-implementation-daemon:")
                and (
                    (
                        isinstance(next_receipt, Mapping)
                        and next_receipt.get("schema")
                        == TYPED_DATABASE_STRICT_RESUME_REJECTION_SCHEMA
                    )
                    or strict_rejection_operation
                    in {
                        TYPED_DATABASE_STRICT_RESUME_REQUEUE_OPERATION,
                        TYPED_DATABASE_STRICT_RESUME_QUARANTINE_OPERATION,
                    }
                )
            )
            if typed_strict_rejection:
                try:
                    rejection = (
                        _validated_database_strict_resume_rejection_receipt(
                            next_receipt
                        )
                    )
                except TaskSourceIntegrityError as exc:
                    raise TypedStateOwnerAuthorizationError(str(exc)) from exc
                task_cid = str(command.parameters.get("task_cid") or "").strip()
                expected_revision = command.parameters.get(
                    "expected_task_revision"
                )
                requested_status = str(
                    command.parameters.get("status") or ""
                ).strip()
                expected_status = (
                    "ready"
                    if strict_rejection_operation
                    == TYPED_DATABASE_STRICT_RESUME_REQUEUE_OPERATION
                    else "quarantined"
                )
                if (
                    not task_cid
                    or type(expected_revision) is not int
                    or expected_revision < 1
                    or requested_status != expected_status
                    or rejection["task_cid"] != task_cid
                    or rejection["rejected_task_revision"]
                    != expected_revision
                ):
                    raise TypedStateOwnerAuthorizationError(
                        "typed strict-resume rejection command is invalid"
                    )
                task_rows = self._connection.execute(
                    """
                    SELECT status, revision, task_alias, body_json FROM tasks
                    WHERE task_cid = ? LIMIT 2
                    """,
                    [task_cid],
                ).fetchall()
                if len(task_rows) != 1:
                    raise TypedStateOwnerAuthorizationError(
                        "typed strict-resume rejection task is absent or ambiguous"
                    )
                task_row = task_rows[0]
                try:
                    prior_body = json.loads(str(task_row[3] or "{}"))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise TypedStateOwnerAuthorizationError(
                        "typed strict-resume rejection prior body is malformed"
                    ) from exc
                prior_receipt = (
                    prior_body.get("completion_receipt")
                    if isinstance(prior_body, Mapping)
                    else None
                )
                if not isinstance(prior_receipt, Mapping):
                    raise TypedStateOwnerAuthorizationError(
                        "typed strict-resume rejection has no prior claim"
                    )
                prior_identity = _validated_database_claim_identity(
                    prior_receipt
                )
                _require_database_claim_process_attestation(
                    prior_receipt,
                    grant=grant,
                )
                prior_operation = prior_receipt.get("operation")
                prior_schema = prior_receipt.get("claim_phase_schema")
                if prior_operation == "database_claim":
                    prior_revision_bound = bool(
                        prior_schema
                        == TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA
                        and _strict_scalar_equal(
                            prior_receipt.get("claimed_from_revision"),
                            expected_revision - 1,
                        )
                    )
                else:
                    prior_revision_bound = bool(
                        prior_operation == "database_attempt_admitted"
                        and prior_schema
                        == TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA
                        and _strict_scalar_equal(
                            prior_receipt.get("admitted_from_revision"),
                            expected_revision - 1,
                        )
                        and _strict_scalar_equal(
                            prior_receipt.get("claimed_from_revision"),
                            expected_revision - 2,
                        )
                        and prior_receipt.get("attempt_execution_phase")
                        == "claimed"
                        and _strict_scalar_equal(
                            prior_receipt.get("attempt_execution_revision"),
                            1,
                        )
                    )
                expected_shared_binding = {
                    **prior_identity,
                    "operation": prior_operation,
                    "claim_phase_schema": prior_schema,
                }
                try:
                    prior_route = TaskExecutionRouteBinding.from_dict(
                        prior_receipt.get("execution_route_binding")
                    ).to_dict()
                except (TypeError, ValueError, TaskSourceIntegrityError) as exc:
                    raise TypedStateOwnerAuthorizationError(
                        "typed strict-resume rejection prior route is invalid"
                    ) from exc
                if (
                    str(task_row[0] or "").strip().lower() != "in_progress"
                    or int(task_row[1]) != expected_revision
                    or str(task_row[2] or "")
                    != rejection["rejected_task_alias"]
                    or not prior_revision_bound
                    or any(
                        not _strict_scalar_equal(
                            rejection.get(name), value
                        )
                        for name, value in prior_identity.items()
                    )
                    or rejection["shared_claim_binding"]
                    != expected_shared_binding
                    or rejection["execution_route_binding"] != prior_route
                    or rejection["execution_route_policy_id"]
                    != prior_route["policy_id"]
                    or rejection["execution_route_origin_revision"]
                    != prior_route["task_revision"]
                ):
                    raise TypedStateOwnerAuthorizationError(
                        "typed strict-resume rejection differs from prior authority"
                    )
                return {
                    "operation": "task.database.strict_resume_rejection",
                    "task_cid": task_cid,
                    "status": expected_status,
                    "expected_revision": expected_revision,
                    "body_json": str(command.parameters["body_json"]),
                    "receipt": rejection,
                }
            if typed_executor_claim and phase_schema not in {
                TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA,
                TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA,
            }:
                raise TypedStateOwnerAuthorizationError(
                    "typed executor in-progress receipt omits its closed claim phase"
                )
            if typed_executor_claim:
                if not isinstance(next_receipt, Mapping):
                    raise TypedStateOwnerAuthorizationError(
                        "typed database claim receipt is unavailable"
                    )
                task_cid = str(command.parameters.get("task_cid") or "").strip()
                expected_revision = command.parameters.get(
                    "expected_task_revision"
                )
                next_status = str(command.parameters.get("status") or "").strip()
                if (
                    not task_cid
                    or type(expected_revision) is not int
                    or expected_revision < 0
                    or next_status != "in_progress"
                ):
                    raise TypedStateOwnerAuthorizationError(
                        "typed database claim command is invalid"
                    )
                task_rows = self._connection.execute(
                    """
                    SELECT status, revision, body_json FROM tasks
                    WHERE task_cid = ? LIMIT 2
                    """,
                    [task_cid],
                ).fetchall()
                if len(task_rows) != 1:
                    raise TypedStateOwnerAuthorizationError(
                        "typed database claim task authority is absent or ambiguous"
                    )
                task_row = task_rows[0]
                if int(task_row[1]) != expected_revision:
                    raise TypedStateOwnerAuthorizationError(
                        "typed database claim task revision is stale"
                    )
                try:
                    prior_body = json.loads(str(task_row[2] or "{}"))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise TypedStateOwnerAuthorizationError(
                        "typed database claim prior task body is malformed"
                    ) from exc
                prior_receipt = (
                    prior_body.get("completion_receipt")
                    if isinstance(prior_body, Mapping)
                    else None
                )
                next_identity = _validated_database_claim_identity(next_receipt)
                next_attestation = _require_database_claim_process_attestation(
                    next_receipt,
                    grant=grant,
                )
                if phase_schema == TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA:
                    if (
                        next_receipt.get("operation") != "database_claim"
                        or not _strict_scalar_equal(
                            next_receipt.get("claimed_from_revision"),
                            expected_revision,
                        )
                        or str(task_row[0] or "").strip().lower()
                        not in {
                            "proposed",
                            "admitted",
                            "pending",
                            "ready",
                            "todo",
                            "queued",
                            "retrying",
                            "in_progress",
                        }
                    ):
                        raise TypedStateOwnerAuthorizationError(
                            "typed database claim reservation authority is stale"
                        )
                    if str(task_row[0] or "").strip().lower() == "in_progress":
                        if not isinstance(prior_receipt, Mapping):
                            raise TypedStateOwnerAuthorizationError(
                                "typed fenced reservation has no prior admission"
                            )
                        prior_identity = _validated_database_claim_identity(
                            prior_receipt
                        )
                        prior_attestation = prior_receipt.get(
                            "claim_process_attestation"
                        )
                        if (
                            prior_receipt.get("operation")
                            != "database_attempt_admitted"
                            or prior_receipt.get("claim_phase_schema")
                            != TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA
                            or not isinstance(prior_attestation, Mapping)
                            or set(prior_attestation) != set(next_attestation)
                            or any(
                                not _strict_scalar_equal(
                                    prior_attestation.get(name), value
                                )
                                for name, value in next_attestation.items()
                            )
                            or prior_identity["owner_session_id"]
                            != next_identity["owner_session_id"]
                            or prior_identity["attempt_number"]
                            >= next_identity["attempt_number"]
                            or prior_identity["fencing_token"]
                            >= next_identity["fencing_token"]
                            or prior_identity["fence_epoch"]
                            > next_identity["fence_epoch"]
                        ):
                            raise TypedStateOwnerAuthorizationError(
                                "typed fenced reservation is not a newer live claim"
                            )
                else:
                    if not isinstance(prior_receipt, Mapping):
                        raise TypedStateOwnerAuthorizationError(
                            "typed database attempt admission has no reservation"
                        )
                    prior_identity = _validated_database_claim_identity(
                        prior_receipt
                    )
                    prior_attestation = prior_receipt.get(
                        "claim_process_attestation"
                    )
                    if (
                        str(task_row[0] or "").strip().lower()
                        != "in_progress"
                        or next_receipt.get("operation")
                        != "database_attempt_admitted"
                        or prior_receipt.get("operation") != "database_claim"
                        or prior_receipt.get("claim_phase_schema")
                        != TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA
                        or any(
                            not _strict_scalar_equal(
                                next_identity.get(name), value
                            )
                            for name, value in prior_identity.items()
                        )
                        or not isinstance(prior_attestation, Mapping)
                        or set(prior_attestation) != set(next_attestation)
                        or any(
                            not _strict_scalar_equal(
                                prior_attestation.get(name), value
                            )
                            for name, value in next_attestation.items()
                        )
                        or not _strict_scalar_equal(
                            next_receipt.get("claimed_from_revision"),
                            prior_receipt.get("claimed_from_revision"),
                        )
                        or not _strict_scalar_equal(
                            next_receipt.get("admitted_from_revision"),
                            expected_revision,
                        )
                        or not _strict_scalar_equal(
                            next_receipt.get("attempt_execution_revision"),
                            1,
                        )
                        or next_receipt.get("attempt_execution_phase")
                        != "claimed"
                    ):
                        raise TypedStateOwnerAuthorizationError(
                            "typed database attempt admission differs from its reservation"
                        )
                return {
                    "operation": "task.database.claim.phase",
                    "task_cid": task_cid,
                    "status": next_status,
                    "expected_revision": expected_revision,
                    "body_json": str(command.parameters["body_json"]),
                    "claim_phase_schema": phase_schema,
                    "claim_identity": next_identity,
                    "claim_process_attestation": next_attestation,
                }
        if operation == "task.retry.cooldown.record":
            parameters = _validated_retry_cooldown_parameters(command.parameters)
            task_rows = self._connection.execute(
                """
                SELECT status, revision, body_json FROM tasks
                WHERE task_cid = ? LIMIT 2
                """,
                [parameters["task_cid"]],
            ).fetchall()
            if len(task_rows) != 1:
                raise TypedStateOwnerAuthorizationError(
                    "retry cooldown task authority is absent or ambiguous"
                )
            task_row = task_rows[0]
            try:
                task_body = json.loads(str(task_row[2] or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise TypedStateOwnerAuthorizationError(
                    "retry cooldown task receipt is malformed"
                ) from exc
            receipt = (
                task_body.get("completion_receipt")
                if isinstance(task_body, Mapping)
                else None
            )
            receipt_values = dict(receipt) if isinstance(receipt, Mapping) else {}
            exact_receipt = {
                "claim_id": parameters["claim_id"],
                "attempt_id": parameters["attempt_id"],
                "attempt_number": parameters["attempt_number"],
                "lease_id": parameters["lease_id"],
                "owner_session_id": parameters["owner_session_id"],
                "fencing_token": parameters["fencing_token"],
                "fence_epoch": parameters["fence_epoch"],
            }
            receipt_operation = str(receipt_values.get("operation") or "")
            if parameters["expected_task_status"] == "in_progress":
                reservation_matches = False
                if receipt_operation == "database_claim":
                    _validated_database_claim_process_attestation(
                        receipt_values
                    )
                    reservation_matches = bool(
                        receipt_values.get("claim_phase_schema")
                        == TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA
                        and _strict_scalar_equal(
                            receipt_values.get("claimed_from_revision"),
                            parameters["expected_task_revision"] - 1,
                        )
                    )
                admission_matches = False
                if receipt_operation == "database_attempt_admitted":
                    _validated_database_claim_process_attestation(
                        receipt_values
                    )
                    admission_matches = bool(
                        receipt_values.get("claim_phase_schema")
                        == TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA
                        and _strict_scalar_equal(
                            receipt_values.get("admitted_from_revision"),
                            parameters["expected_task_revision"] - 1,
                        )
                    )
                receipt_state_matches = bool(
                    reservation_matches
                    or admission_matches
                )
            elif parameters["expected_task_status"] == "retrying":
                receipt_state_matches = bool(
                    receipt_operation in TYPED_RETRYING_RECEIPT_OPERATIONS
                    and _strict_scalar_equal(
                        receipt_values.get("control_expected_revision"),
                        parameters["expected_task_revision"],
                    )
                    and _strict_scalar_equal(
                        receipt_values.get("queue_reason"),
                        parameters["reason"],
                    )
                    and _strict_scalar_equal(
                        receipt_values.get("backoff_ms"),
                        parameters["delay_ms"],
                    )
                    and _strict_scalar_equal(
                        receipt_values.get("retry_not_before_ms"),
                        parameters["retry_not_before_ms"],
                    )
                )
            else:
                receipt_state_matches = bool(
                    receipt_operation
                    in {
                        "database_portal_terminal_failure",
                        "database_portal_typed_deferral_budget_exhausted",
                    }
                    and receipt_values.get("retryable") is False
                    and type(receipt_values.get("control_expected_status"))
                    is str
                    and receipt_values["control_expected_status"]
                    in {"in_progress", "retrying"}
                    and _strict_scalar_equal(
                        receipt_values.get("control_expected_revision"),
                        parameters["expected_task_revision"] - 1,
                    )
                )
            if (
                str(task_row[0] or "").strip().lower()
                != parameters["expected_task_status"]
                or int(task_row[1])
                != (
                    parameters["expected_task_revision"] + 1
                    if parameters["expected_task_status"] == "retrying"
                    else parameters["expected_task_revision"]
                )
                or not isinstance(receipt, Mapping)
                or not receipt_state_matches
                or any(
                    not _strict_scalar_equal(receipt_values.get(name), expected)
                    for name, expected in exact_receipt.items()
                )
            ):
                raise TypedStateOwnerAuthorizationError(
                    "retry cooldown claim or task revision authority is stale"
                )
            queue_rows = self._connection.execute(
                """
                SELECT task_cid, claim_cid, resolution_cid, claimant_did,
                       logical_epoch, fencing_token, expires_at_ms, attempt,
                       state, started_at_ms, release_reason,
                       retry_not_before_ms, owner_session_id, fence_epoch,
                       revision, extension_schema, extension_json
                FROM leases WHERE task_cid = ? LIMIT 2
                """,
                [parameters["task_cid"]],
            ).fetchall()
            if len(queue_rows) > 1:
                raise TypedStateOwnerAuthorizationError(
                    "retry cooldown queue authority is ambiguous"
            )
            prior_queue: dict[str, Any] = {}
            if queue_rows:
                validated_prior = _validated_stored_retry_cooldown(
                    queue_rows[0],
                    task_cid=parameters["task_cid"],
                )
                prior_queue = {
                    **validated_prior,
                    "claim_id": validated_prior["claim_cid"],
                    "attempt_number": validated_prior["attempt"],
                }
            expected_queue_revision = parameters["expected_queue_revision"]
            expected_queue_attempt = parameters["expected_queue_attempt"]
            if (
                (expected_queue_revision == -1 and prior_queue)
                or (expected_queue_revision >= 0 and not prior_queue)
                or (
                    prior_queue
                    and (
                        prior_queue["task_cid"] != parameters["task_cid"]
                        or prior_queue["revision"] != expected_queue_revision
                        or prior_queue["attempt_number"]
                        != expected_queue_attempt
                        or prior_queue["attempt_number"]
                        >= parameters["attempt_number"]
                        or prior_queue["extension_schema"]
                        != TYPED_RETRY_COOLDOWN_SCHEMA
                        or not isinstance(prior_queue["extension"], Mapping)
                        or prior_queue["extension"].get("schema")
                        != TYPED_RETRY_COOLDOWN_SCHEMA
                        or prior_queue["extension"].get("task_cid")
                        != parameters["task_cid"]
                        or prior_queue["extension"].get("claim_id")
                        != prior_queue["claim_id"]
                        or prior_queue["extension"].get("attempt_number")
                        != prior_queue["attempt_number"]
                    )
                )
            ):
                raise TypedStateOwnerAuthorizationError(
                    "retry cooldown expected queue absence/revision is stale"
                )
            return {
                "operation": operation,
                "task": {
                    "task_cid": parameters["task_cid"],
                    "status": str(task_row[0]),
                    "revision": int(task_row[1]),
                    "receipt": receipt_values,
                },
                "prior_queue": prior_queue,
            }
        if operation not in {
            "supervisor.runtime.attest",
            "supervisor.transition",
            "event.delivery.record",
            "event.acknowledge",
        }:
            return {}
        scope = {
            name: str(command.parameters.get(name) or "").strip()
            for name in (
                "tenant_id",
                "federation_id",
                "supervisor_id",
                "subscription_id",
                "consumer_id",
                "event_id",
            )
        }
        if not scope["tenant_id"] or not scope["federation_id"]:
            raise TypedStateOwnerAuthorizationError(
                "semantic command lacks authoritative scope"
            )
        if operation.startswith("supervisor."):
            rows = self._connection.execute(
                """
                SELECT lifecycle_state, revision, fencing_epoch, lease_id,
                       process_birth_id
                FROM supervisor_instances
                WHERE supervisor_id = ? AND tenant_id = ? AND federation_id = ?
                LIMIT 2
                """,
                [
                    scope["supervisor_id"],
                    scope["tenant_id"],
                    scope["federation_id"],
                ],
            ).fetchall()
            if len(rows) != 1:
                raise TypedStateOwnerAuthorizationError(
                    "supervisor semantic authority is absent or ambiguous"
                )
            row = rows[0]
            current = {
                "lifecycle_state": str(row[0]),
                "revision": int(row[1]),
                "fencing_epoch": int(row[2]),
                "lease_id": str(row[3]),
                "process_birth_id": str(row[4]),
            }
            if current["fencing_epoch"] != command.fence_epoch:
                raise TypedStateOwnerAuthorizationError(
                    "supervisor semantic fence differs from the store fence"
                )
            if operation == "supervisor.runtime.attest":
                expected_revision = command.parameters.get("expected_revision")
                expected_fence = command.parameters.get("expected_fencing_epoch")
                if (
                    isinstance(expected_revision, bool)
                    or not isinstance(expected_revision, int)
                    or expected_revision != current["revision"]
                    or isinstance(expected_fence, bool)
                    or not isinstance(expected_fence, int)
                    or expected_fence != current["fencing_epoch"]
                    or current["lifecycle_state"]
                    not in {"ADMITTED", "STARTING", "IDLE", "ACTIVE", "PAUSED", "RECOVERING"}
                    or not grant.process_birth_id
                ):
                    raise TypedStateOwnerAuthorizationError(
                        "supervisor runtime admission authority is stale or ineligible"
                    )
                expected_birth_id = "process-birth-attestation:" + content_identity(
                    {
                        "tenant_id": scope["tenant_id"],
                        "federation_id": scope["federation_id"],
                        "supervisor_id": scope["supervisor_id"],
                        "subagent_id": "",
                        "canonical_birth_id": grant.process_birth_id,
                    }
                )
                latest = self._connection.execute(
                    """
                    SELECT runtime_lease_id, process_birth_id, revision, status
                    FROM supervisor_runtime_leases
                    WHERE tenant_id = ? AND federation_id = ?
                      AND supervisor_id = ? AND lease_id = ?
                      AND fencing_epoch = ?
                    ORDER BY revision DESC LIMIT 1
                    """,
                    [
                        scope["tenant_id"],
                        scope["federation_id"],
                        scope["supervisor_id"],
                        current["lease_id"],
                        current["fencing_epoch"],
                    ],
                ).fetchall()
                active = self._connection.execute(
                    """
                    SELECT runtime_lease_id, process_birth_id
                    FROM supervisor_runtime_leases
                    WHERE tenant_id = ? AND federation_id = ?
                      AND supervisor_id = ? AND lease_id = ?
                      AND fencing_epoch = ? AND status = 'active'
                    ORDER BY revision DESC LIMIT 2
                    """,
                    [
                        scope["tenant_id"],
                        scope["federation_id"],
                        scope["supervisor_id"],
                        current["lease_id"],
                        current["fencing_epoch"],
                    ],
                ).fetchall()
                if (
                    (latest and str(latest[0][1]) != expected_birth_id)
                    or len(active) > 1
                    or (active and str(active[0][1]) != expected_birth_id)
                ):
                    raise TypedStateOwnerAuthorizationError(
                        "runtime takeover requires a new fencing epoch"
                    )
                return {
                    "operation": operation,
                    "scope": scope,
                    "supervisor": current,
                    "expected_process_birth_id": expected_birth_id,
                    "latest_runtime_lease_id": str(latest[0][0]) if latest else "",
                    "latest_runtime_revision": int(latest[0][2]) if latest else 0,
                    "runtime_revision": int(latest[0][2]) + 1 if latest else 1,
                }

            from ..federation.contracts import FederationLifecycleState
            from ..federation.lifecycle import assert_transition

            try:
                requested = FederationLifecycleState(
                    str(command.parameters.get("requested_state") or "")
                )
                active_attempts_row = self._connection.execute(
                    """
                    SELECT COUNT(DISTINCT attempts.attempt_id)
                    FROM task_attempts AS attempts
                    INNER JOIN subagent_instances AS agents
                      ON agents.task_id = attempts.task_cid
                    WHERE agents.tenant_id = ? AND agents.federation_id = ?
                      AND agents.supervisor_id = ?
                      AND attempts.status NOT IN (
                        'accepted','cancelled','completed','failed','rejected','stopped'
                      )
                    """,
                    [scope["tenant_id"], scope["federation_id"], scope["supervisor_id"]],
                ).fetchone()
                active_slots_row = self._connection.execute(
                    """
                    SELECT COUNT(*) FROM subagent_execution_slots
                    WHERE tenant_id = ? AND federation_id = ? AND supervisor_id = ?
                      AND state = 'active' AND subagent_id IS NOT NULL
                    """,
                    [scope["tenant_id"], scope["federation_id"], scope["supervisor_id"]],
                ).fetchone()
                active_effects_row = self._connection.execute(
                    """
                    SELECT COUNT(*) FROM federation_effect_reservations
                    WHERE tenant_id = ? AND federation_id = ? AND supervisor_id = ?
                      AND state NOT IN (
                        'cancelled','compensated','completed','failed','released','revoked'
                      )
                    """,
                    [scope["tenant_id"], scope["federation_id"], scope["supervisor_id"]],
                ).fetchone()
                target = assert_transition(
                    current["lifecycle_state"],
                    requested,
                    active_attempts=int(active_attempts_row[0])
                    + int(active_slots_row[0]),
                    active_effects=int(active_effects_row[0]),
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                raise TypedStateOwnerAuthorizationError(
                    "supervisor lifecycle transition is not authoritative"
                ) from exc
            if target.value in {"STARTING", "ACTIVE", "COMPLETED"}:
                now = datetime.now(_UTC).isoformat().replace("+00:00", "Z")
                runtime = self._connection.execute(
                    """
                    SELECT leases.process_birth_id, leases.process_id,
                           leases.process_start_time_ticks, leases.process_boot_id,
                           leases.process_parent_id, births.process_id,
                           births.start_marker, births.host_identity_ref,
                           leases.evidence_ref
                    FROM supervisor_runtime_leases AS leases
                    INNER JOIN process_births AS births
                      ON births.process_birth_id = leases.process_birth_id
                     AND births.tenant_id = leases.tenant_id
                     AND births.federation_id = leases.federation_id
                     AND births.supervisor_id = leases.supervisor_id
                    WHERE leases.tenant_id = ? AND leases.federation_id = ?
                      AND leases.supervisor_id = ? AND leases.lease_id = ?
                      AND leases.fencing_epoch = ? AND leases.status = 'active'
                      AND leases.revoked_at IS NULL AND leases.expires_at > ?
                      AND births.status = 'active' AND births.stopped_at IS NULL
                    ORDER BY leases.revision DESC LIMIT 2
                    """,
                    [
                        scope["tenant_id"], scope["federation_id"],
                        scope["supervisor_id"], current["lease_id"],
                        current["fencing_epoch"], now,
                    ],
                ).fetchall()
                if len(runtime) != 1 or (
                    str(runtime[0][0]) != current["process_birth_id"]
                    or int(runtime[0][1]) != int(runtime[0][5])
                    or str(runtime[0][2]) != str(runtime[0][6])
                    or not str(runtime[0][3])
                    or not str(runtime[0][7])
                    or not str(runtime[0][8])
                ):
                    raise TypedStateOwnerAuthorizationError(
                        "executable lifecycle lacks current process-bound runtime authority"
                    )
            return {
                "operation": operation,
                "scope": scope,
                "supervisor": current,
                "target_state": target.value,
            }

        subscription_rows = self._connection.execute(
            """
            SELECT revision, consumer_id, status, expires_at
            FROM event_subscriptions
            WHERE subscription_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 2
            """,
            [scope["subscription_id"], scope["tenant_id"], scope["federation_id"]],
        ).fetchall()
        cursor_rows = self._connection.execute(
            """
            SELECT subscription_revision, global_sequence, store_generation,
                   fencing_epoch, revision
            FROM consumer_cursors
            WHERE consumer_id = ? AND subscription_id = ?
              AND tenant_id = ? AND federation_id = ?
            LIMIT 2
            """,
            [
                scope["consumer_id"], scope["subscription_id"],
                scope["tenant_id"], scope["federation_id"],
            ],
        ).fetchall()
        if (
            len(subscription_rows) != 1
            or len(cursor_rows) != 1
            or str(subscription_rows[0][1]) != scope["consumer_id"]
            or str(subscription_rows[0][2]) != "active"
            or int(cursor_rows[0][0]) != int(subscription_rows[0][0])
            or int(cursor_rows[0][3]) != command.fence_epoch
        ):
            raise TypedStateOwnerAuthorizationError(
                "event consumer authority is stale, inactive, or ambiguous"
            )
        expected_subscription_revision = command.parameters.get(
            "subscription_revision"
        )
        expected_fence = command.parameters.get("expected_fencing_epoch")
        if (
            (
                operation == "event.delivery.record"
                and (
                    isinstance(expected_subscription_revision, bool)
                    or not isinstance(expected_subscription_revision, int)
                    or expected_subscription_revision
                    != int(subscription_rows[0][0])
                )
            )
            or isinstance(expected_fence, bool)
            or not isinstance(expected_fence, int)
            or expected_fence != command.fence_epoch
        ):
            raise TypedStateOwnerAuthorizationError(
                "event command subscription revision or fence is stale"
            )
        common = {
            "operation": operation,
            "scope": scope,
            "subscription_revision": int(subscription_rows[0][0]),
            "cursor_sequence": int(cursor_rows[0][1]),
            "store_generation": int(cursor_rows[0][2]),
            "cursor_fencing_epoch": int(cursor_rows[0][3]),
            "cursor_revision": int(cursor_rows[0][4]),
        }
        if operation == "event.delivery.record":
            queue = self._connection.execute(
                """
                SELECT queue.delivery_id, queue.outbox_id, queue.attempt_number,
                       queue.revision, queue.status, events.global_sequence,
                       events.event_cid, outbox.event_cid
                FROM event_delivery_queue AS queue
                INNER JOIN domain_events AS events
                  ON events.event_id = queue.representative_event_id
                 AND events.tenant_id = queue.tenant_id
                 AND events.federation_id = queue.federation_id
                INNER JOIN transactional_outbox AS outbox
                  ON outbox.outbox_id = queue.outbox_id
                 AND outbox.event_id = events.event_id
                 AND outbox.tenant_id = events.tenant_id
                 AND outbox.federation_id = events.federation_id
                WHERE queue.tenant_id = ? AND queue.federation_id = ?
                  AND queue.subscription_id = ? AND queue.subscription_revision = ?
                  AND queue.consumer_id = ? AND queue.representative_event_id = ?
                  AND queue.fencing_epoch = ? AND queue.status IN ('pending','retry')
                LIMIT 2
                """,
                [
                    scope["tenant_id"], scope["federation_id"],
                    scope["subscription_id"], common["subscription_revision"],
                    scope["consumer_id"], scope["event_id"], command.fence_epoch,
                ],
            ).fetchall()
            attempt_id = str(command.parameters.get("attempt_id") or "")
            existing_attempt = self._connection.execute(
                "SELECT COUNT(*) FROM delivery_attempts WHERE attempt_id = ?",
                [attempt_id],
            ).fetchone()
            if (
                len(queue) != 1
                or not attempt_id
                or int(existing_attempt[0]) != 0
                or int(queue[0][5]) <= common["cursor_sequence"]
                or str(queue[0][6]) != str(queue[0][7])
            ):
                raise TypedStateOwnerAuthorizationError(
                    "delivery attempt lacks one authoritative route-first queue"
                )
            common.update(
                {
                    "attempt_id": attempt_id,
                    "delivery_id": str(queue[0][0]),
                    "outbox_id": str(queue[0][1]),
                    "attempt_number": int(queue[0][2]) + 1,
                    "prior_attempt_number": int(queue[0][2]),
                    "queue_revision": int(queue[0][3]),
                    "event_sequence": int(queue[0][5]),
                }
            )
            return common

        expected_cursor_revision = command.parameters.get("expected_cursor_revision")
        if (
            isinstance(expected_cursor_revision, bool)
            or not isinstance(expected_cursor_revision, int)
            or expected_cursor_revision != common["cursor_revision"]
        ):
            raise TypedStateOwnerAuthorizationError(
                "acknowledgement cursor revision is stale"
            )
        attempt_id = str(command.parameters.get("delivery_attempt_id") or "")
        delivery = self._connection.execute(
            """
            SELECT attempts.delivery_id, attempts.attempt_number, attempts.status,
                   queue.revision, queue.status, events.global_sequence
            FROM delivery_attempts AS attempts
            INNER JOIN event_delivery_queue AS queue
              ON queue.delivery_id = attempts.delivery_id
             AND queue.tenant_id = attempts.tenant_id
             AND queue.federation_id = attempts.federation_id
             AND queue.subscription_id = attempts.subscription_id
             AND queue.subscription_revision = attempts.subscription_revision
             AND queue.consumer_id = attempts.consumer_id
            INNER JOIN domain_events AS events
              ON events.event_id = attempts.event_id
             AND events.tenant_id = attempts.tenant_id
             AND events.federation_id = attempts.federation_id
            WHERE attempts.attempt_id = ? AND attempts.tenant_id = ?
              AND attempts.federation_id = ? AND attempts.event_id = ?
              AND attempts.subscription_id = ?
              AND attempts.subscription_revision = ?
              AND attempts.consumer_id = ? AND attempts.fencing_epoch = ?
              AND NOT EXISTS (
                SELECT 1 FROM delivery_attempts AS newer
                WHERE newer.event_id = attempts.event_id
                  AND newer.subscription_id = attempts.subscription_id
                  AND newer.subscription_revision = attempts.subscription_revision
                  AND newer.consumer_id = attempts.consumer_id
                  AND newer.attempt_number > attempts.attempt_number
              )
            LIMIT 2
            """,
            [
                attempt_id, scope["tenant_id"], scope["federation_id"],
                scope["event_id"], scope["subscription_id"],
                common["subscription_revision"], scope["consumer_id"],
                command.fence_epoch,
            ],
        ).fetchall()
        next_eligible = self._connection.execute(
            """
            SELECT queue.representative_event_id, events.global_sequence
            FROM event_delivery_queue AS queue
            INNER JOIN domain_events AS events
              ON events.event_id = queue.representative_event_id
             AND events.tenant_id = queue.tenant_id
             AND events.federation_id = queue.federation_id
            WHERE queue.tenant_id = ? AND queue.federation_id = ?
              AND queue.subscription_id = ? AND queue.subscription_revision = ?
              AND queue.consumer_id = ?
              AND queue.status IN ('pending','retry','delivered')
              AND events.global_sequence > ?
            ORDER BY events.global_sequence, queue.delivery_id LIMIT 1
            """,
            [
                scope["tenant_id"], scope["federation_id"],
                scope["subscription_id"], common["subscription_revision"],
                scope["consumer_id"], common["cursor_sequence"],
            ],
        ).fetchall()
        acknowledgement_id = str(command.parameters.get("acknowledgement_id") or "")
        existing_ack = self._connection.execute(
            "SELECT COUNT(*) FROM event_acknowledgements WHERE acknowledgement_id = ?",
            [acknowledgement_id],
        ).fetchone()
        if (
            len(delivery) != 1
            or not next_eligible
            or str(next_eligible[0][0]) != scope["event_id"]
            or int(next_eligible[0][1]) != int(delivery[0][5])
            or str(delivery[0][2]) != "delivered"
            or str(delivery[0][4]) != "delivered"
            or int(delivery[0][5]) <= common["cursor_sequence"]
            or not acknowledgement_id
            or int(existing_ack[0]) != 0
        ):
            raise TypedStateOwnerAuthorizationError(
                "acknowledgement would skip or forge authoritative delivery work"
            )
        common.update(
            {
                "attempt_id": attempt_id,
                "delivery_id": str(delivery[0][0]),
                "attempt_number": int(delivery[0][1]),
                "queue_revision": int(delivery[0][3]),
                "event_sequence": int(delivery[0][5]),
                "acknowledgement_id": acknowledgement_id,
                "disposition": str(command.parameters.get("disposition") or ""),
            }
        )
        return common

    @staticmethod
    def _authorize_mutation(
        command: StateCommand,
        operation: OwnerOperation,
        parameters: Sequence[Any],
        *,
        grant: OwnerClientGrant,
    ) -> None:
        command_operation = str(command.parameters.get("operation") or "")
        admitted = (
            _COMMAND_MUTATION_CATALOG.get(command_operation, frozenset())
            | _TRANSACTION_MUTATIONS
            | (
                _EVENT_MUTATIONS
                if command_operation in _EVENT_EMITTING_COMMANDS
                else frozenset()
            )
        )
        if operation.name not in admitted:
            raise TypedStateOwnerAuthorizationError(
                "mutation is not admitted for this command operation"
            )
        if operation.name not in grant.allowed_operations:
            raise TypedStateOwnerAuthorizationError(
                "mutation operation is outside the client grant"
            )
        bound = {
            name: parameters[index]
            for index, name in enumerate(operation.parameter_names)
        }
        if operation.name in {
            "casf_insert_process_birth_attestation",
            "casf_insert_supervisor_runtime_lease",
        }:
            start_time, parent_pid, boot_id = _process_runtime_facts(
                grant.peer_pid
            )
            expected_process = {
                "process_id": grant.peer_pid,
                "process_start_time_ticks": start_time,
                "process_parent_id": parent_pid,
                "process_boot_id": boot_id,
            }
            if operation.name == "casf_insert_process_birth_attestation":
                expected_process = {
                    "process_id": grant.peer_pid,
                    "start_marker": str(start_time),
                }
            if any(
                str(bound.get(field)) != str(value)
                for field, value in expected_process.items()
            ):
                raise TypedStateOwnerAuthorizationError(
                    "supervisor runtime attestation differs from kernel peer facts"
                )
        for field in _SCOPE_FIELDS:
            expected = command.parameters.get(field)
            if expected in (None, ""):
                continue
            candidates = (field, f"scope_{field}", f"unique_{field}")
            for candidate in candidates:
                if candidate in bound and str(bound[candidate]) != str(expected):
                    raise TypedStateOwnerAuthorizationError(
                        "mutation scope differs from the admitted command"
                    )
        if command_operation in _FEDERATION_COMMANDS:
            for field in ("tenant_id", "federation_id"):
                candidates = (field, f"scope_{field}")
                present = [candidate for candidate in candidates if candidate in bound]
                if present and any(
                    str(bound[candidate]) != str(command.parameters[field])
                    for candidate in present
                ):
                    raise TypedStateOwnerAuthorizationError(
                        "mutation tenant/federation scope differs"
                    )

    @staticmethod
    def _manifest_bindings(
        operation: OwnerOperation,
        parameters: Sequence[Any],
    ) -> dict[str, Any]:
        return {
            name: parameters[index]
            for index, name in enumerate(operation.parameter_names)
        }

    @classmethod
    def _record_manifest_mutation(
        cls,
        command: StateCommand,
        operation: OwnerOperation,
        parameters: Sequence[Any],
        manifest: list[tuple[str, dict[str, Any]]],
    ) -> None:
        """Admit one mutation into the closed owner-side transaction machine."""

        if len(manifest) >= MAX_TRANSACTION_MUTATIONS:
            raise TypedStateOwnerAuthorizationError(
                "transaction mutation manifest exceeds its closed bound"
            )
        name = operation.name
        names = [item[0] for item in manifest]
        if name not in _REPEATABLE_MUTATIONS and name in names:
            raise TypedStateOwnerAuthorizationError(
                "transaction repeats a singleton mutation role"
            )
        repeat_limit = _MUTATION_REPEAT_LIMITS.get(name, MAX_PARAMETER_COUNT)
        if name in _REPEATABLE_MUTATIONS and names.count(name) >= repeat_limit:
            raise TypedStateOwnerAuthorizationError(
                "transaction repeats a mutation role beyond its closed bound"
            )
        if "txn_record_idempotency" in names:
            raise TypedStateOwnerAuthorizationError(
                "transaction mutation follows its idempotency seal"
            )
        if "txn_advance_store_revision" in names and name != "txn_record_idempotency":
            raise TypedStateOwnerAuthorizationError(
                "transaction mutation follows its store revision seal"
            )
        if name == "txn_record_idempotency" and (
            not names or names[-1] != "txn_advance_store_revision"
        ):
            raise TypedStateOwnerAuthorizationError(
                "idempotency seal must immediately follow store revision advance"
            )

        command_operation = str(command.parameters.get("operation") or "")
        event_core_seen = [item for item in names if item in _EVENT_CORE_SEQUENCE]
        if name in _EVENT_CORE_SEQUENCE:
            expected_index = len(event_core_seen)
            if (
                expected_index >= len(_EVENT_CORE_SEQUENCE)
                or name != _EVENT_CORE_SEQUENCE[expected_index]
            ):
                raise TypedStateOwnerAuthorizationError(
                    "event/outbox mutation roles are missing, duplicated, or out of order"
                )
        elif name in {"casf_insert_event_parent", "casf_insert_changed_fact"}:
            if (
                "casf_insert_domain_event" not in names
                or "casf_insert_outbox" in names
            ):
                raise TypedStateOwnerAuthorizationError(
                    "event lineage must follow its event and precede its outbox"
                )
        elif name in _COMMAND_MUTATION_CATALOG.get(
            command_operation, frozenset()
        ):
            if event_core_seen and "casf_insert_outbox" not in names:
                raise TypedStateOwnerAuthorizationError(
                    "domain mutation cannot split the event/outbox manifest"
                )
            if "casf_insert_outbox" in names and name not in (
                _POST_EVENT_DOMAIN_MUTATIONS.get(command_operation, frozenset())
            ):
                raise TypedStateOwnerAuthorizationError(
                    "domain mutation follows the sealed event/outbox pair"
                )

        bound = cls._manifest_bindings(operation, parameters)
        if name == "txn_advance_store_revision":
            expected = {
                "new_revision": command.expected_revision + 1,
                "generation": command.expected_generation,
                "expected_revision": command.expected_revision,
                "fence_epoch": command.fence_epoch,
            }
            if any(bound.get(field) != value for field, value in expected.items()):
                raise TypedStateOwnerAuthorizationError(
                    "store revision mutation differs from the admitted command"
                )
        elif name == "txn_record_idempotency":
            expected = {
                "idempotency_key": command.idempotency_key,
                "command_kind": command.command_kind.value,
                "command_id": command.command_id,
                "store_id": command.store_id,
                "session_id": command.session_id,
            }
            if any(str(bound.get(field) or "") != value for field, value in expected.items()):
                raise TypedStateOwnerAuthorizationError(
                    "idempotency mutation differs from the admitted command"
                )
        elif name == "txn_cas_task_status":
            expected_task_revision = command.parameters.get(
                "expected_task_revision"
            )
            expected = {
                "task_cid": command.parameters.get("task_cid"),
                "expected_task_revision": expected_task_revision,
                "new_revision": (
                    expected_task_revision + 1
                    if isinstance(expected_task_revision, int)
                    and not isinstance(expected_task_revision, bool)
                    else None
                ),
                "status": command.parameters.get("status"),
            }
            if any(bound.get(field) != value for field, value in expected.items()):
                raise TypedStateOwnerAuthorizationError(
                    "task status mutation differs from the admitted command"
                )
        elif name == "executor_cas_task_status_receipt":
            expected_task_revision = command.parameters.get(
                "expected_task_revision"
            )
            expected = {
                "task_cid": command.parameters.get("task_cid"),
                "expected_task_revision": expected_task_revision,
                "new_revision": (
                    expected_task_revision + 1
                    if isinstance(expected_task_revision, int)
                    and not isinstance(expected_task_revision, bool)
                    else None
                ),
                "status": command.parameters.get("status"),
                "body_json": command.parameters.get("body_json"),
            }
            if any(bound.get(field) != value for field, value in expected.items()):
                raise TypedStateOwnerAuthorizationError(
                    "task status receipt mutation differs from the admitted command"
                )
        elif name == "executor_insert_validation_run":
            expected = {
                "run_id": command.parameters.get("run_id"),
                "task_cid": command.parameters.get("task_cid"),
                "attempt_id": command.parameters.get("attempt_id"),
                "started_at": command.parameters.get("started_at"),
                "finished_at": command.parameters.get("finished_at"),
                "status": command.parameters.get("outcome"),
                "command_digest": command.parameters.get("command_digest"),
                "body_json": command.parameters.get("run_body_json"),
            }
            if any(bound.get(field) != value for field, value in expected.items()):
                raise TypedStateOwnerAuthorizationError(
                    "validation run mutation differs from the admitted command"
                )
        elif name == "executor_insert_validation_result":
            expected = {
                "result_id": command.parameters.get("result_id"),
                "run_id": command.parameters.get("run_id"),
                "task_cid": command.parameters.get("task_cid"),
                "ordinal": 0,
                "outcome": command.parameters.get("outcome"),
                "evidence_digest": command.parameters.get("evidence_digest"),
                "body_json": command.parameters.get("result_body_json"),
            }
            if any(bound.get(field) != value for field, value in expected.items()):
                raise TypedStateOwnerAuthorizationError(
                    "validation result mutation differs from the admitted command"
                )
        elif name == "executor_insert_validation_evidence":
            expected = {
                "evidence_id": command.parameters.get("evidence_id"),
                "parent_evidence_id": "",
                "task_cid": command.parameters.get("task_cid"),
                "evidence_kind": "validation",
                "digest": command.parameters.get("evidence_digest"),
                "created_at": command.parameters.get("finished_at"),
                "body_json": command.parameters.get("evidence_body_json"),
            }
            if any(bound.get(field) != value for field, value in expected.items()):
                raise TypedStateOwnerAuthorizationError(
                    "validation evidence mutation differs from the admitted command"
                )
        elif name in {
            "executor_insert_retry_cooldown",
            "executor_update_retry_cooldown",
        }:
            values = _validated_retry_mutation_parameters(command.parameters)
            expected_queue_revision = values["expected_queue_revision"]
            common = {
                "task_cid": values["task_cid"],
                "claim_id": values["claim_id"],
                "resolution_cid": values["resolution_cid"],
                "claimant_did": values["owner_session_id"],
                "logical_epoch": values["fence_epoch"],
                "fencing_token": values["fencing_token"],
                "expires_at_ms": 0,
                "attempt_number": values["attempt_number"],
                "state": "released",
                "started_at_ms": values["started_at_ms"],
                "reason": values["reason"],
                "retry_not_before_ms": values["retry_not_before_ms"],
                "owner_session_id": values["owner_session_id"],
                "fence_epoch": values["fence_epoch"],
                "new_queue_revision": (
                    1
                    if expected_queue_revision == -1
                    else expected_queue_revision + 1
                ),
                "extension_schema": TYPED_RETRY_COOLDOWN_SCHEMA,
                "extension_json": values["extension_json"],
            }
            expected = (
                {
                    **common,
                    "expected_queue_revision_for_insert": -1,
                }
                if name == "executor_insert_retry_cooldown"
                else {
                    **common,
                    "expected_queue_revision": expected_queue_revision,
                    "expected_queue_attempt": values["expected_queue_attempt"],
                    "new_attempt_guard": values["attempt_number"],
                    "expected_existing_extension_schema": (
                        TYPED_RETRY_COOLDOWN_SCHEMA
                    ),
                }
            )
            if (
                (name == "executor_insert_retry_cooldown")
                != (expected_queue_revision == -1)
            ):
                raise TypedStateOwnerAuthorizationError(
                    "retry cooldown mutation kind differs from queue presence"
                )
            if any(bound.get(field) != value for field, value in expected.items()):
                raise TypedStateOwnerAuthorizationError(
                    "retry cooldown mutation differs from the admitted command"
                )
        manifest.append((name, bound))

    def _validate_transaction_manifest(
        self,
        command: StateCommand,
        manifest: Sequence[tuple[str, Mapping[str, Any]]],
        *,
        semantic_authority: Mapping[str, Any] | None = None,
    ) -> None:
        """Reject partial effects; only a complete manifest or replay may commit."""

        if not manifest:
            row = self._connection.execute(
                """
                SELECT command_kind, command_id, store_id
                FROM idempotency_records
                WHERE idempotency_key = ? LIMIT 1
                """,
                [command.idempotency_key],
            ).fetchone()
            if row is None or (
                str(row[0]) != command.command_kind.value
                or str(row[1]) != command.command_id
                or str(row[2]) != command.store_id
            ):
                raise TypedStateOwnerAuthorizationError(
                    "empty transaction is not an authoritative idempotent replay"
                )
            return

        names = [item[0] for item in manifest]
        command_operation = str(command.parameters.get("operation") or "")
        required = _COMMAND_REQUIRED_DOMAIN_MUTATIONS.get(
            command_operation, frozenset()
        )
        if not required.issubset(names):
            raise TypedStateOwnerAuthorizationError(
                "transaction omits a required domain mutation role"
            )
        if names[-2:] != [
            "txn_advance_store_revision",
            "txn_record_idempotency",
        ]:
            raise TypedStateOwnerAuthorizationError(
                "transaction lacks its ordered revision and idempotency seals"
            )
        for seal in ("txn_advance_store_revision", "txn_record_idempotency"):
            if names.count(seal) != 1:
                raise TypedStateOwnerAuthorizationError(
                    "transaction seal mutation count differs"
                )

        if command_operation in _EVENT_EMITTING_COMMANDS:
            for event_role in _EVENT_CORE_SEQUENCE:
                if names.count(event_role) != 1:
                    raise TypedStateOwnerAuthorizationError(
                        "transaction event/outbox mutation count differs"
                    )
            event_positions = [names.index(item) for item in _EVENT_CORE_SEQUENCE]
            if event_positions != sorted(event_positions):
                raise TypedStateOwnerAuthorizationError(
                    "transaction event/outbox mutation order differs"
                )
            self._validate_event_outbox_identity(manifest)
        elif any(name in _EVENT_MUTATIONS for name in names):
            raise TypedStateOwnerAuthorizationError(
                "non-event command contains an event mutation"
            )
        if command_operation == "supervisor.runtime.attest":
            self._validate_supervisor_runtime_identity(command, manifest)
        if command_operation == "event.outbox.disposition":
            self._validate_outbox_disposition_identity(command, manifest)
        if semantic_authority:
            self._validate_semantic_manifest(
                command,
                manifest,
                semantic_authority,
            )

    def _validate_semantic_manifest(
        self,
        command: StateCommand,
        manifest: Sequence[tuple[str, Mapping[str, Any]]],
        authority: Mapping[str, Any],
    ) -> None:
        """Close semantic identities and post-state before durable commit."""

        operation = str(authority.get("operation") or "")
        by_name: dict[str, list[Mapping[str, Any]]] = {}
        for name, bound in manifest:
            by_name.setdefault(name, []).append(bound)

        def one(name: str) -> Mapping[str, Any]:
            values = by_name.get(name, [])
            if len(values) != 1:
                raise TypedStateOwnerAuthorizationError(
                    f"semantic command requires exactly one {name} mutation"
                )
            return values[0]

        def exact(bound: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
            if any(
                str(bound.get(field) if bound.get(field) is not None else "")
                != str(value if value is not None else "")
                for field, value in expected.items()
            ):
                raise TypedStateOwnerAuthorizationError(
                    "semantic mutation differs from owner-resolved authority"
                )

        if operation == TYPED_DATABASE_CLAIM_RECOVERY_COMMAND:
            values = dict(authority["cooldown_parameters"])
            expected_revision = int(authority["expected_revision"])
            cooldown_mutation = one("executor_insert_retry_cooldown")
            if by_name.get("executor_update_retry_cooldown"):
                raise TypedStateOwnerAuthorizationError(
                    "dead claim recovery cannot replace a cooldown row"
                )
            exact(
                one("executor_cas_task_status_receipt"),
                {
                    "task_cid": authority["task_cid"],
                    "expected_task_revision": expected_revision,
                    "new_revision": expected_revision + 1,
                    "status": "retrying",
                    "body_json": authority["body_json"],
                },
            )
            exact(
                cooldown_mutation,
                {
                    "task_cid": values["task_cid"],
                    "claim_id": values["claim_id"],
                    "resolution_cid": values["resolution_cid"],
                    "claimant_did": values["owner_session_id"],
                    "logical_epoch": values["fence_epoch"],
                    "fencing_token": values["fencing_token"],
                    "expires_at_ms": 0,
                    "attempt_number": values["attempt_number"],
                    "state": "released",
                    "started_at_ms": values["started_at_ms"],
                    "reason": values["reason"],
                    "retry_not_before_ms": values[
                        "retry_not_before_ms"
                    ],
                    "owner_session_id": values["owner_session_id"],
                    "fence_epoch": values["fence_epoch"],
                    "new_queue_revision": 1,
                    "extension_schema": TYPED_RETRY_COOLDOWN_SCHEMA,
                    "extension_json": values["extension_json"],
                    "expected_queue_revision_for_insert": -1,
                },
            )
            task_rows = self._connection.execute(
                """
                SELECT status, revision, body_json FROM tasks
                WHERE task_cid = ? LIMIT 2
                """,
                [authority["task_cid"]],
            ).fetchall()
            observed_task = (
                tuple(task_rows[0][index] for index in range(3))
                if len(task_rows) == 1
                else ()
            )
            if observed_task != (
                "retrying",
                expected_revision + 1,
                authority["body_json"],
            ):
                raise TypedStateOwnerAuthorizationError(
                    "dead claim recovery task post-state differs"
                )
            queue_rows = self._connection.execute(
                """
                SELECT task_cid, claim_cid, resolution_cid, claimant_did,
                       logical_epoch, fencing_token, expires_at_ms, attempt,
                       state, started_at_ms, release_reason,
                       retry_not_before_ms, owner_session_id, fence_epoch,
                       revision, extension_schema, extension_json
                FROM leases WHERE task_cid = ? LIMIT 2
                """,
                [authority["task_cid"]],
            ).fetchall()
            expected_queue = (
                values["task_cid"],
                values["claim_id"],
                values["resolution_cid"],
                values["owner_session_id"],
                values["fence_epoch"],
                values["fencing_token"],
                0,
                values["attempt_number"],
                "released",
                values["started_at_ms"],
                values["reason"],
                values["retry_not_before_ms"],
                values["owner_session_id"],
                values["fence_epoch"],
                1,
                TYPED_RETRY_COOLDOWN_SCHEMA,
                values["extension_json"],
            )
            observed_queue = (
                tuple(
                    queue_rows[0][index]
                    for index in range(len(expected_queue))
                )
                if len(queue_rows) == 1
                else ()
            )
            if observed_queue != expected_queue:
                raise TypedStateOwnerAuthorizationError(
                    "dead claim recovery cooldown post-state differs"
                )
            return

        if operation == "task.database.claim.phase":
            mutation = one("executor_cas_task_status_receipt")
            expected_revision = int(authority["expected_revision"])
            exact(
                mutation,
                {
                    "task_cid": authority["task_cid"],
                    "expected_task_revision": expected_revision,
                    "new_revision": expected_revision + 1,
                    "status": "in_progress",
                    "body_json": authority["body_json"],
                },
            )
            rows = self._connection.execute(
                """
                SELECT status, revision, body_json FROM tasks
                WHERE task_cid = ? LIMIT 2
                """,
                [authority["task_cid"]],
            ).fetchall()
            observed_row = (
                tuple(rows[0][index] for index in range(3))
                if len(rows) == 1
                else ()
            )
            if observed_row != (
                "in_progress",
                expected_revision + 1,
                authority["body_json"],
            ):
                raise TypedStateOwnerAuthorizationError(
                    "typed database claim phase post-state differs"
                )
            return

        if operation == "task.database.strict_resume_rejection":
            mutation = one("executor_cas_task_status_receipt")
            expected_revision = int(authority["expected_revision"])
            exact(
                mutation,
                {
                    "task_cid": authority["task_cid"],
                    "expected_task_revision": expected_revision,
                    "new_revision": expected_revision + 1,
                    "status": authority["status"],
                    "body_json": authority["body_json"],
                },
            )
            rows = self._connection.execute(
                """
                SELECT status, revision, body_json FROM tasks
                WHERE task_cid = ? LIMIT 2
                """,
                [authority["task_cid"]],
            ).fetchall()
            observed_row = (
                tuple(rows[0][index] for index in range(3))
                if len(rows) == 1
                else ()
            )
            if observed_row != (
                authority["status"],
                expected_revision + 1,
                authority["body_json"],
            ):
                raise TypedStateOwnerAuthorizationError(
                    "typed strict-resume rejection post-state differs"
                )
            return

        if operation == "task.retry.cooldown.record":
            values = _validated_retry_cooldown_parameters(command.parameters)
            prior_queue = dict(authority.get("prior_queue") or {})
            expected_mutation = (
                "executor_update_retry_cooldown"
                if prior_queue
                else "executor_insert_retry_cooldown"
            )
            alternate = (
                "executor_insert_retry_cooldown"
                if prior_queue
                else "executor_update_retry_cooldown"
            )
            mutation = one(expected_mutation)
            if by_name.get(alternate):
                raise TypedStateOwnerAuthorizationError(
                    "retry cooldown transaction contains both mutation roles"
                )
            task = dict(authority.get("task") or {})
            authoritative_task_revision = (
                values["expected_task_revision"] + 1
                if values["expected_task_status"] == "retrying"
                else values["expected_task_revision"]
            )
            if (
                task.get("task_cid") != values["task_cid"]
                or task.get("status") != values["expected_task_status"]
                or task.get("revision") != authoritative_task_revision
                or values["expected_queue_revision"]
                != (prior_queue.get("revision") if prior_queue else -1)
                or values["expected_queue_attempt"]
                != (prior_queue.get("attempt_number") if prior_queue else 0)
            ):
                raise TypedStateOwnerAuthorizationError(
                    "retry cooldown semantic authority changed before commit"
                )
            # The per-operation manifest checks every bound parameter.  The
            # semantic pass additionally proves that exactly that mutation is
            # now the one authoritative queue row before commit.
            if mutation.get("task_cid") != values["task_cid"]:
                raise TypedStateOwnerAuthorizationError(
                    "retry cooldown mutation task differs from authority"
                )
            rows = self._connection.execute(
                """
                SELECT task_cid, claim_cid, resolution_cid, claimant_did,
                       logical_epoch, fencing_token, expires_at_ms, attempt,
                       state, started_at_ms, release_reason,
                       retry_not_before_ms, owner_session_id, fence_epoch,
                       revision, extension_schema, extension_json
                FROM leases WHERE task_cid = ? LIMIT 2
                """,
                [values["task_cid"]],
            ).fetchall()
            expected_revision = (
                1 if not prior_queue else int(prior_queue["revision"]) + 1
            )
            if len(rows) != 1:
                raise TypedStateOwnerAuthorizationError(
                    "retry cooldown post-state is absent or ambiguous"
                )
            row = rows[0]
            expected_row = (
                values["task_cid"],
                values["claim_id"],
                values["resolution_cid"],
                values["owner_session_id"],
                values["fence_epoch"],
                values["fencing_token"],
                0,
                values["attempt_number"],
                "released",
                values["started_at_ms"],
                values["reason"],
                values["retry_not_before_ms"],
                values["owner_session_id"],
                values["fence_epoch"],
                expected_revision,
                TYPED_RETRY_COOLDOWN_SCHEMA,
                values["extension_json"],
            )
            observed_row = tuple(row[index] for index in range(len(expected_row)))
            if observed_row != expected_row:
                raise TypedStateOwnerAuthorizationError(
                    "retry cooldown post-state differs from its admitted receipt"
                )
            return

        scope = dict(authority.get("scope") or {})
        if operation == "supervisor.transition":
            current = dict(authority["supervisor"])
            target = str(authority["target_state"])
            exact(
                one("casf_update_supervisor_lifecycle"),
                {
                    "lifecycle_state": target,
                    "status": target,
                    "new_fencing_epoch": current["fencing_epoch"],
                    "supervisor_id": scope["supervisor_id"],
                    "tenant_id": scope["tenant_id"],
                    "federation_id": scope["federation_id"],
                    "expected_revision": current["revision"],
                    "expected_fencing_epoch": current["fencing_epoch"],
                },
            )
            event = one("casf_insert_domain_event")
            exact(
                event,
                {
                    "event_type": "SUPERVISOR_HEALTH_CHANGED",
                    "stream_id": scope["federation_id"],
                    "tenant_id": scope["tenant_id"],
                    "federation_id": scope["federation_id"],
                    "supervisor_id": scope["supervisor_id"],
                    "session_id": command.session_id,
                    "payload_ref": f"state:{target}",
                    "effect_class": "authoritative_state",
                },
            )
            if not str(event.get("deduplication_key") or "").startswith(
                "supervisor-transition:"
            ):
                raise TypedStateOwnerAuthorizationError(
                    "supervisor transition event deduplication class differs"
                )
            rows = self._connection.execute(
                """
                SELECT lifecycle_state, status, revision, fencing_epoch
                FROM supervisor_instances
                WHERE supervisor_id = ? AND tenant_id = ? AND federation_id = ?
                LIMIT 2
                """,
                [scope["supervisor_id"], scope["tenant_id"], scope["federation_id"]],
            ).fetchall()
            if len(rows) != 1 or (
                str(rows[0][0]) != target
                or str(rows[0][1]) != target
                or int(rows[0][2]) != int(current["revision"]) + 1
                or int(rows[0][3]) != int(current["fencing_epoch"])
            ):
                raise TypedStateOwnerAuthorizationError(
                    "supervisor lifecycle CAS did not produce exact post-state"
                )
            return

        if operation == "supervisor.runtime.attest":
            current = dict(authority["supervisor"])
            process_birth_id = str(authority["expected_process_birth_id"])
            runtime_revision = int(authority["runtime_revision"])
            birth = one("casf_insert_process_birth_attestation")
            lease = one("casf_insert_supervisor_runtime_lease")
            update = one("casf_update_supervisor_process_birth")
            exact(
                birth,
                {
                    "process_birth_id": process_birth_id,
                    "tenant_id": scope["tenant_id"],
                    "federation_id": scope["federation_id"],
                    "supervisor_id": scope["supervisor_id"],
                    "subagent_id": "",
                },
            )
            exact(
                lease,
                {
                    "tenant_id": scope["tenant_id"],
                    "federation_id": scope["federation_id"],
                    "supervisor_id": scope["supervisor_id"],
                    "lease_id": current["lease_id"],
                    "process_birth_id": process_birth_id,
                    "process_id": birth["process_id"],
                    "process_start_time_ticks": birth["start_marker"],
                    "fencing_epoch": current["fencing_epoch"],
                    "revision": runtime_revision,
                },
            )
            exact(
                update,
                {
                    "process_birth_id": process_birth_id,
                    "supervisor_id": scope["supervisor_id"],
                    "tenant_id": scope["tenant_id"],
                    "federation_id": scope["federation_id"],
                    "expected_revision": current["revision"],
                    "fencing_epoch": current["fencing_epoch"],
                    "current_process_birth_id": process_birth_id,
                },
            )
            superseded = by_name.get("casf_supersede_supervisor_runtime_lease", [])
            latest_id = str(authority.get("latest_runtime_lease_id") or "")
            if latest_id:
                if len(superseded) != 1:
                    raise TypedStateOwnerAuthorizationError(
                        "runtime renewal did not supersede its exact prior lease"
                    )
                exact(
                    superseded[0],
                    {
                        "runtime_lease_id": latest_id,
                        "tenant_id": scope["tenant_id"],
                        "federation_id": scope["federation_id"],
                        "supervisor_id": scope["supervisor_id"],
                        "expected_revision": authority["latest_runtime_revision"],
                    },
                )
            elif superseded:
                raise TypedStateOwnerAuthorizationError(
                    "initial runtime attestation cannot supersede a lease"
                )
            event = one("casf_insert_domain_event")
            exact(
                event,
                {
                    "event_type": "SUPERVISOR_HEALTH_CHANGED",
                    "stream_id": scope["federation_id"],
                    "tenant_id": scope["tenant_id"],
                    "federation_id": scope["federation_id"],
                    "supervisor_id": scope["supervisor_id"],
                    "session_id": command.session_id,
                    "payload_ref": lease["evidence_ref"],
                    "effect_class": "authoritative_state",
                },
            )
            if not str(event.get("deduplication_key") or "").startswith(
                "supervisor-runtime-attest:"
            ):
                raise TypedStateOwnerAuthorizationError(
                    "runtime attestation event deduplication class differs"
                )
            rows = self._connection.execute(
                """
                SELECT supervisors.process_birth_id, leases.revision,
                       leases.status, births.status
                FROM supervisor_instances AS supervisors
                INNER JOIN supervisor_runtime_leases AS leases
                  ON leases.process_birth_id = supervisors.process_birth_id
                 AND leases.tenant_id = supervisors.tenant_id
                 AND leases.federation_id = supervisors.federation_id
                 AND leases.supervisor_id = supervisors.supervisor_id
                INNER JOIN process_births AS births
                  ON births.process_birth_id = leases.process_birth_id
                 AND births.tenant_id = leases.tenant_id
                 AND births.federation_id = leases.federation_id
                 AND births.supervisor_id = leases.supervisor_id
                WHERE supervisors.supervisor_id = ?
                  AND supervisors.tenant_id = ? AND supervisors.federation_id = ?
                  AND leases.runtime_lease_id = ?
                LIMIT 2
                """,
                [
                    scope["supervisor_id"], scope["tenant_id"],
                    scope["federation_id"], lease["runtime_lease_id"],
                ],
            ).fetchall()
            if len(rows) != 1 or (
                str(rows[0][0]) != process_birth_id
                or int(rows[0][1]) != runtime_revision
                or str(rows[0][2]) != "active"
                or str(rows[0][3]) != "active"
            ):
                raise TypedStateOwnerAuthorizationError(
                    "runtime attestation did not produce exact process-bound post-state"
                )
            return

        if operation == "event.delivery.record":
            attempt = one("casf_insert_delivery_attempt")
            queue = one("casf_mark_queue_delivered")
            exact(
                attempt,
                {
                    "attempt_id": authority["attempt_id"],
                    "tenant_id": scope["tenant_id"],
                    "federation_id": scope["federation_id"],
                    "event_id": scope["event_id"],
                    "outbox_id": authority["outbox_id"],
                    "delivery_id": authority["delivery_id"],
                    "subscription_id": scope["subscription_id"],
                    "subscription_revision": authority["subscription_revision"],
                    "consumer_id": scope["consumer_id"],
                    "attempt_number": authority["attempt_number"],
                    "fencing_epoch": command.fence_epoch,
                    "status": "delivered",
                    "error_code": "",
                },
            )
            exact(
                queue,
                {
                    "attempt_number": authority["attempt_number"],
                    "delivery_id": authority["delivery_id"],
                    "tenant_id": scope["tenant_id"],
                    "federation_id": scope["federation_id"],
                    "subscription_id": scope["subscription_id"],
                    "subscription_revision": authority["subscription_revision"],
                    "consumer_id": scope["consumer_id"],
                    "prior_attempt_number": authority["prior_attempt_number"],
                    "expected_revision": authority["queue_revision"],
                    "fencing_epoch": command.fence_epoch,
                },
            )
            rows = self._connection.execute(
                """
                SELECT attempts.status, attempts.attempt_number,
                       queue.status, queue.attempt_number, queue.revision
                FROM delivery_attempts AS attempts
                INNER JOIN event_delivery_queue AS queue
                  ON queue.delivery_id = attempts.delivery_id
                 AND queue.tenant_id = attempts.tenant_id
                 AND queue.federation_id = attempts.federation_id
                 AND queue.subscription_id = attempts.subscription_id
                 AND queue.consumer_id = attempts.consumer_id
                WHERE attempts.attempt_id = ? LIMIT 2
                """,
                [authority["attempt_id"]],
            ).fetchall()
            if len(rows) != 1 or (
                str(rows[0][0]) != "delivered"
                or int(rows[0][1]) != int(authority["attempt_number"])
                or str(rows[0][2]) != "delivered"
                or int(rows[0][3]) != int(authority["attempt_number"])
                or int(rows[0][4]) != int(authority["queue_revision"]) + 1
            ):
                raise TypedStateOwnerAuthorizationError(
                    "delivery record did not produce exact queue post-state"
                )
            return

        if operation != "event.acknowledge":
            raise TypedStateOwnerAuthorizationError(
                "semantic authority operation is unsupported"
            )
        if str(authority.get("disposition") or "") != "processed":
            raise TypedStateOwnerAuthorizationError(
                "runtime acknowledgement disposition is not admitted"
            )
        exact(
            one("casf_mark_delivery_acknowledged"),
            {
                "attempt_id": authority["attempt_id"],
                "tenant_id": scope["tenant_id"],
                "federation_id": scope["federation_id"],
                "event_id": scope["event_id"],
                "subscription_id": scope["subscription_id"],
                "consumer_id": scope["consumer_id"],
                "subscription_revision": authority["subscription_revision"],
                "fencing_epoch": command.fence_epoch,
            },
        )
        exact(
            one("casf_mark_queue_acknowledged"),
            {
                "delivery_id": authority["delivery_id"],
                "tenant_id": scope["tenant_id"],
                "federation_id": scope["federation_id"],
                "subscription_id": scope["subscription_id"],
                "subscription_revision": authority["subscription_revision"],
                "consumer_id": scope["consumer_id"],
                "fencing_epoch": command.fence_epoch,
                "expected_revision": authority["queue_revision"],
            },
        )
        exact(
            one("casf_reset_subscription_failures"),
            {
                "tenant_id": scope["tenant_id"],
                "federation_id": scope["federation_id"],
                "subscription_id": scope["subscription_id"],
                "subscription_revision": authority["subscription_revision"],
                "consumer_id": scope["consumer_id"],
            },
        )
        acknowledgement = one("casf_insert_event_acknowledgement")
        exact(
            acknowledgement,
            {
                "acknowledgement_id": authority["acknowledgement_id"],
                "tenant_id": scope["tenant_id"],
                "federation_id": scope["federation_id"],
                "event_id": scope["event_id"],
                "subscription_id": scope["subscription_id"],
                "consumer_id": scope["consumer_id"],
                "subscription_revision": authority["subscription_revision"],
                "global_sequence": authority["event_sequence"],
                "delivery_attempt_id": authority["attempt_id"],
                "cursor_revision": int(authority["cursor_revision"]) + 1,
                "fencing_epoch": command.fence_epoch,
                "disposition": "processed",
            },
        )
        cursor = one("casf_advance_consumer_cursor")
        exact(
            cursor,
            {
                "global_sequence": authority["event_sequence"],
                "store_generation": command.expected_generation,
                "last_event_id": scope["event_id"],
                "consumer_id": scope["consumer_id"],
                "subscription_id": scope["subscription_id"],
                "subscription_revision": authority["subscription_revision"],
                "expected_revision": authority["cursor_revision"],
                "expected_fencing_epoch": command.fence_epoch,
                "upper_global_sequence": authority["event_sequence"],
            },
        )
        rows = self._connection.execute(
            """
            SELECT attempts.status, queue.status, queue.revision,
                   cursors.global_sequence, cursors.last_event_id,
                   cursors.revision, acknowledgements.global_sequence
            FROM delivery_attempts AS attempts
            INNER JOIN event_delivery_queue AS queue
              ON queue.delivery_id = attempts.delivery_id
             AND queue.tenant_id = attempts.tenant_id
             AND queue.federation_id = attempts.federation_id
             AND queue.subscription_id = attempts.subscription_id
             AND queue.consumer_id = attempts.consumer_id
            INNER JOIN consumer_cursors AS cursors
              ON cursors.consumer_id = attempts.consumer_id
             AND cursors.subscription_id = attempts.subscription_id
             AND cursors.tenant_id = attempts.tenant_id
             AND cursors.federation_id = attempts.federation_id
            INNER JOIN event_acknowledgements AS acknowledgements
              ON acknowledgements.delivery_attempt_id = attempts.attempt_id
             AND acknowledgements.event_id = attempts.event_id
             AND acknowledgements.consumer_id = attempts.consumer_id
            WHERE attempts.attempt_id = ?
              AND acknowledgements.acknowledgement_id = ? LIMIT 2
            """,
            [authority["attempt_id"], authority["acknowledgement_id"]],
        ).fetchall()
        if len(rows) != 1 or (
            str(rows[0][0]) != "acknowledged"
            or str(rows[0][1]) != "acknowledged"
            or int(rows[0][2]) != int(authority["queue_revision"]) + 1
            or int(rows[0][3]) != int(authority["event_sequence"])
            or str(rows[0][4]) != scope["event_id"]
            or int(rows[0][5]) != int(authority["cursor_revision"]) + 1
            or int(rows[0][6]) != int(authority["event_sequence"])
        ):
            raise TypedStateOwnerAuthorizationError(
                "acknowledgement did not produce exact non-skipping cursor post-state"
            )

    @staticmethod
    def _validate_supervisor_runtime_identity(
        command: StateCommand,
        manifest: Sequence[tuple[str, Mapping[str, Any]]],
    ) -> None:
        by_name: dict[str, list[Mapping[str, Any]]] = {}
        for name, bound in manifest:
            by_name.setdefault(name, []).append(bound)
        birth = by_name["casf_insert_process_birth_attestation"][0]
        lease = by_name["casf_insert_supervisor_runtime_lease"][0]
        update = by_name["casf_update_supervisor_process_birth"][0]
        process_birth_id = str(birth.get("process_birth_id") or "")
        expected_supervisor = str(command.parameters.get("supervisor_id") or "")
        if (
            not process_birth_id
            or str(lease.get("process_birth_id") or "") != process_birth_id
            or str(update.get("process_birth_id") or "") != process_birth_id
            or str(update.get("current_process_birth_id") or "")
            != process_birth_id
            or any(
                str(item.get("supervisor_id") or "") != expected_supervisor
                for item in (birth, lease, update)
            )
            or str(lease.get("process_id")) != str(birth.get("process_id"))
            or str(lease.get("process_start_time_ticks"))
            != str(birth.get("start_marker"))
            or int(lease.get("fencing_epoch") or 0) != command.fence_epoch
        ):
            raise TypedStateOwnerAuthorizationError(
                "supervisor runtime mutation identities do not close"
            )
        superseded = by_name.get("casf_supersede_supervisor_runtime_lease", [])
        if len(superseded) > 1:
            raise TypedStateOwnerAuthorizationError(
                "supervisor runtime renewal supersedes multiple leases"
            )
        if superseded and (
            str(superseded[0].get("supervisor_id") or "")
            != expected_supervisor
        ):
            raise TypedStateOwnerAuthorizationError(
                "supervisor runtime renewal crosses supervisor identity"
            )

    @staticmethod
    def _validate_outbox_disposition_identity(
        command: StateCommand,
        manifest: Sequence[tuple[str, Mapping[str, Any]]],
    ) -> None:
        by_name: dict[str, list[Mapping[str, Any]]] = {}
        for name, bound in manifest:
            by_name.setdefault(name, []).append(bound)
        disposition = by_name["casf_insert_outbox_routing_disposition"][0]
        members = by_name["casf_insert_outbox_routing_disposition_event"]
        marked = by_name["casf_mark_outbox_routed"]
        event_count = command.parameters.get("event_count")
        if (
            isinstance(event_count, bool)
            or not isinstance(event_count, int)
            or not 1 <= event_count <= 1_024
            or len(members) != event_count
            or len(marked) != event_count
        ):
            raise TypedStateOwnerAuthorizationError(
                "outbox disposition event population differs"
            )
        disposition_id = str(disposition.get("disposition_id") or "")
        route_batch_id = str(command.parameters.get("route_batch_id") or "")
        if (
            not disposition_id
            or str(command.parameters.get("disposition_id") or "")
            != disposition_id
            or str(disposition.get("route_batch_id") or "") != route_batch_id
            or int(disposition.get("event_count") or 0) != event_count
        ):
            raise TypedStateOwnerAuthorizationError(
                "outbox disposition receipt differs from its command"
            )
        member_ids = [str(item.get("event_id") or "") for item in members]
        marked_ids = [str(item.get("event_id") or "") for item in marked]
        ordinals = [int(item.get("ordinal") or 0) for item in members]
        if (
            len(set(member_ids)) != event_count
            or member_ids != marked_ids
            or ordinals != list(range(1, event_count + 1))
            or any(
                str(item.get("disposition_id") or "") != disposition_id
                for item in members
            )
            or any(
                int(member.get("global_sequence") or 0)
                != int(mark.get("global_sequence") or 0)
                # Python 3.8 compatibility forbids ``zip(strict=True)``.  The
                # equal-length predicate above supplies the same safety check.
                for member, mark in zip(members, marked)  # noqa: B905
            )
        ):
            raise TypedStateOwnerAuthorizationError(
                "outbox disposition members and CAS mutations differ"
            )

    @staticmethod
    def _validate_event_outbox_identity(
        manifest: Sequence[tuple[str, Mapping[str, Any]]],
    ) -> None:
        by_name: dict[str, list[Mapping[str, Any]]] = {}
        for name, bound in manifest:
            by_name.setdefault(name, []).append(bound)
        event = by_name["casf_insert_domain_event"][0]
        outbox = by_name["casf_insert_outbox"][0]
        for field in (
            "event_id",
            "event_cid",
            "stream_id",
            "stream_sequence",
            "global_sequence",
            "tenant_id",
            "federation_id",
        ):
            if event.get(field) != outbox.get(field):
                raise TypedStateOwnerAuthorizationError(
                    "event and outbox identities differ"
                )
        for name in ("casf_insert_event_parent", "casf_insert_changed_fact"):
            if any(item.get("event_id") != event.get("event_id") for item in by_name.get(name, [])):
                raise TypedStateOwnerAuthorizationError(
                    "event lineage identity differs from its event"
                )
        for name in ("casf_seed_stream_head", "casf_advance_stream_head"):
            stream = by_name[name][0]
            for field in ("stream_id", "tenant_id", "federation_id"):
                if stream.get(field) != event.get(field):
                    raise TypedStateOwnerAuthorizationError(
                        "event stream head scope differs from its event"
                    )

    def _execute(self, operation: OwnerOperation, parameters: list[Any]) -> dict[str, Any]:
        result = self._connection.execute(
            operation.sql,
            parameters if parameters else None,
        )
        return {
            "ok": True,
            "columns": list(_result_columns(result)),
            "rows": _result_rows(result),
            "rowcount": int(getattr(result, "rowcount", -1) or -1),
        }

    @staticmethod
    def _error_code(exc: BaseException) -> str:
        text = str(exc).casefold()
        if "unique" in text or "constraint" in text or "duplicate" in text:
            return "constraint_conflict"
        if isinstance(exc, TypedStateOwnerAuthorizationError):
            return "authorization_denied"
        if isinstance(exc, TypedStateOwnerProtocolError):
            return "protocol_denied"
        return "operation_failed"


class TypedStateOwnerConnection:
    """Client-side DB-API compatibility facade over named owner operations."""

    def __init__(
        self,
        *,
        socket_path: Path,
        token: str,
        client_id: str,
        process_birth_id: str,
        store_id: str,
        timeout_seconds: float = 30.0,
        status_bootstrap: bool = False,
    ) -> None:
        if type(token) is not str or len(token) < 16:
            raise TypedStateOwnerAuthorizationError("typed owner token is unavailable")
        socket_identity = os.path.abspath(os.fspath(socket_path))
        self.bootstrap_socket_path = socket_identity
        self.bootstrap_token_digest = hashlib.sha256(
            token.encode("utf-8")
        ).hexdigest()
        self.status_bootstrap = bool(status_bootstrap)
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.settimeout(float(timeout_seconds))
        self._socket.connect(socket_identity)
        self._request_lock = threading.RLock()
        self._closed = False
        self._active = False
        self._prepared_command: StateCommand | None = None
        self._request_index = 0
        opened = self._request(
            "open_status" if status_bootstrap else "open",
            token=token,
            client_id=client_id,
            process_birth_id=process_birth_id,
            store_id=store_id,
        )
        self.identity = MappingProxyType(dict(opened.get("identity") or {}))
        self.catalog_id = str(opened.get("catalog_id") or "")
        self.session_id = str(opened.get("session_id") or "")
        self.grant = MappingProxyType(dict(opened.get("grant") or {}))
        if not self.session_id:
            raise TypedStateOwnerProtocolError(
                "typed owner handshake returned no admitted session"
            )

    @property
    def supports_event_wait(self) -> bool:
        """Whether this exact server-issued grant admits typed long waits."""

        operations = self.grant.get("allowed_operations") or ()
        return "event.wait" in operations

    def wait_for_events(self, request: Any) -> Any:
        """Execute one bounded long wait inside the exclusive state owner."""

        from ..federation.events import EventBatch, EventWaitRequest

        if not isinstance(request, EventWaitRequest):
            raise TypedStateOwnerProtocolError(
                "typed owner event wait requires EventWaitRequest"
            )
        response = self._request("wait_events", wait_request=request.to_dict())
        return EventBatch.from_dict(response.get("batch") or {})

    def cancel_event_wait(self, consumer_id: str) -> None:
        self._request(
            "cancel_event_wait",
            consumer_id=str(consumer_id or "").strip(),
        )

    def clear_event_wait_cancellation(self, consumer_id: str) -> None:
        self._request(
            "clear_event_wait_cancellation",
            consumer_id=str(consumer_id or "").strip(),
        )

    def submit_eaaef_authorized_operation(
        self,
        envelope: Any,
        *,
        merge_admission_cid: str,
        operational_capability_cid: str,
    ) -> Mapping[str, Any]:
        """Submit one exact signed EAAEF envelope over the owner socket."""

        from .quack_command_authorization import AuthorizedStateCommand

        if type(envelope) is not AuthorizedStateCommand:
            raise TypedStateOwnerProtocolError(
                "EAAEF submission requires exact AuthorizedStateCommand@1"
            )
        response = self._request(
            "eaaef.command.submit",
            envelope=envelope.to_dict(),
            merge_admission_cid=str(merge_admission_cid),
            operational_capability_cid=str(operational_capability_cid),
        )
        receipt = response.get("receipt")
        if not isinstance(receipt, Mapping):
            raise TypedStateOwnerProtocolError(
                "EAAEF submission returned no typed receipt"
            )
        return MappingProxyType(dict(receipt))

    def lookup_eaaef_authorized_operation_receipt(
        self,
        envelope: Any,
        *,
        merge_admission_cid: str,
        operational_capability_cid: str,
    ) -> Mapping[str, Any] | None:
        """Look up one durable EAAEF receipt over the owner socket."""

        from .quack_command_authorization import AuthorizedStateCommand

        if type(envelope) is not AuthorizedStateCommand:
            raise TypedStateOwnerProtocolError(
                "EAAEF lookup requires exact AuthorizedStateCommand@1"
            )
        response = self._request(
            "eaaef.command.lookup",
            envelope=envelope.to_dict(),
            merge_admission_cid=str(merge_admission_cid),
            operational_capability_cid=str(operational_capability_cid),
        )
        receipt = response.get("receipt")
        if receipt is None:
            return None
        if not isinstance(receipt, Mapping):
            raise TypedStateOwnerProtocolError(
                "EAAEF lookup returned a malformed receipt"
            )
        return MappingProxyType(dict(receipt))

    def submit_eaaef_plan_r2_operation(
        self,
        envelope: Any,
        operation_payload: Mapping[str, Any],
        *,
        remote_capability_cid: str,
        plan_r2_operational_capability_cid: str,
        plan_r2_authorization_cid: str,
    ) -> Mapping[str, Any]:
        """Submit one of the exact three Plan-R2 operations to the owner."""

        from .quack_command_authorization import AuthorizedStateCommand

        if type(envelope) is not AuthorizedStateCommand:
            raise TypedStateOwnerProtocolError(
                "Plan-R2 submission requires exact AuthorizedStateCommand@1"
            )
        if type(operation_payload) is not dict:
            raise TypedStateOwnerProtocolError(
                "Plan-R2 operation payload must be one exact object"
            )
        payload = dict(operation_payload)
        operation = payload.get("operation")
        if (
            type(operation) is not str
            or operation not in _EAAEF_PLAN_R2_SERVICE_OPERATIONS
        ):
            raise TypedStateOwnerProtocolError(
                "Plan-R2 submission is outside the exact operation vocabulary"
            )
        response = self._request(
            operation,
            remote_capability_cid=str(remote_capability_cid),
            plan_r2_operational_capability_cid=str(
                plan_r2_operational_capability_cid
            ),
            plan_r2_authorization_cid=str(plan_r2_authorization_cid),
            envelope=envelope.to_dict(),
            operation_payload=payload,
        )
        if set(response) != {"schema", "request_id", "ok", "result"}:
            raise TypedStateOwnerProtocolError(
                "Plan-R2 response does not use the exact wire fields"
            )
        result = response.get("result")
        if not isinstance(result, Mapping):
            raise TypedStateOwnerProtocolError(
                "Plan-R2 submission returned no typed result"
            )
        return MappingProxyType(dict(result))

    def prepare_command(self, command: StateCommand) -> None:
        if not isinstance(command, StateCommand):
            raise TypedStateOwnerProtocolError("prepared command must be typed")
        if self._active:
            raise TypedStateOwnerProtocolError("cannot replace an active command")
        self._prepared_command = command

    def execute_operation(
        self,
        operation: str,
        parameters: Sequence[Any] | None = None,
    ) -> TypedOwnerResult:
        response = self._request(
            "execute",
            operation=str(operation or ""),
            parameters=_closed_parameters(parameters),
        )
        return TypedOwnerResult(
            response.get("columns") or (),
            response.get("rows") or (),
            int(response.get("rowcount") or -1),
        )

    def execute(self, sql: str, parameters: Sequence[Any] | None = None) -> TypedOwnerResult:
        normalized = _normalize_sql(sql).upper()
        if normalized == "BEGIN TRANSACTION":
            if self._prepared_command is None:
                raise TypedStateOwnerAuthorizationError(
                    "Quack mutations require a typed admitted StateCommand"
                )
            self._request("begin", command=self._prepared_command.to_dict())
            self._active = True
            return TypedOwnerResult((), (), -1)
        if normalized == "COMMIT":
            self.commit()
            return TypedOwnerResult((), (), -1)
        if normalized == "ROLLBACK":
            self.rollback()
            return TypedOwnerResult((), (), -1)
        return self.execute_operation(internal_operation_for_sql(sql), parameters)

    def commit(self) -> None:
        if not self._active:
            raise TypedStateOwnerProtocolError(
                "typed owner commit requires an active transaction"
            )
        self._request("commit")
        self._active = False
        self._prepared_command = None

    def rollback(self) -> None:
        if not self._active:
            self._prepared_command = None
            return
        try:
            self._request("rollback")
        finally:
            self._active = False
            self._prepared_command = None

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.rollback()
            self._request("close")
        except Exception:
            pass
        finally:
            self._closed = True
            self._socket.close()

    def _request(self, action: str, **fields: Any) -> dict[str, Any]:
        with self._request_lock:
            if self._closed:
                raise TypedStateOwnerProtocolError("typed owner connection is closed")
            self._request_index += 1
            request_id = f"request:{os.getpid()}:{self._request_index}:{uuid.uuid4().hex}"
            _send_frame(
                self._socket,
                {
                    "schema": TYPED_STATE_OWNER_SCHEMA,
                    "action": action,
                    "request_id": request_id,
                    **fields,
                },
            )
            response = _receive_frame(self._socket)
            allowed = {
                "schema",
                "request_id",
                "ok",
                "identity",
                "catalog_id",
                "session_id",
                "grant",
                "batch",
                "columns",
                "rows",
                "rowcount",
                "receipt",
                "result",
                "error_code",
                "error_type",
            }
            if (
                set(response) - allowed
                or response.get("schema") != TYPED_STATE_OWNER_SCHEMA
                or response.get("request_id") != request_id
            ):
                raise TypedStateOwnerProtocolError("typed owner response identity differs")
            if response.get("ok") is not True:
                # The exclusive owner rolls back and releases its transaction
                # lock on every request failure.  Mirror that terminal state
                # locally so a bounded retry can prepare a fresh command
                # instead of retaining a phantom active transaction.
                if self._active:
                    self._active = False
                    self._prepared_command = None
                raise TypedStateOwnerRemoteError(
                    str(response.get("error_code") or "operation_failed"),
                    str(response.get("error_type") or ""),
                )
            return response


def compact_default_owner_socket_path(
    candidate: Path | str,
    *,
    identity: Path | str,
) -> Path:
    """Keep an owner-selected default within portable ``AF_UNIX`` limits."""

    path = Path(candidate).expanduser().resolve(strict=False)
    if len(os.fsencode(path)) <= _SAFE_UNIX_SOCKET_PATH_BYTES:
        return path
    identity_path = Path(identity).expanduser().resolve(strict=False)
    store_digest = hashlib.sha256(os.fsencode(identity_path)).hexdigest()[:32]
    return (
        Path("/tmp")
        / f"{_COMPACT_SOCKET_ROOT_PREFIX}-{os.geteuid()}"
        / f"{store_digest}.sock"
    )


def typed_owner_socket_path(store_id: str, explicit: str = "") -> Path:
    """Resolve the launcher-controlled socket; never accept a database path payload."""

    selected = str(explicit or os.environ.get(TYPED_STATE_OWNER_SOCKET_ENV, "") or "").strip()
    if selected:
        path = Path(selected).expanduser().resolve(strict=False)
    else:
        store = Path(str(store_id or "")).expanduser().resolve(strict=False)
        path = store.parent / "quack-owner" / TYPED_STATE_OWNER_SOCKET_FILENAME
        path = compact_default_owner_socket_path(path, identity=store)
    return path


def open_typed_state_owner_connection(
    *,
    store_id: str,
    client_id: str,
    process_birth_id: str,
    timeout_seconds: float = 30.0,
) -> TypedStateOwnerConnection:
    from ..runtime.process_security import state_authority_credential

    token = state_authority_credential(TYPED_STATE_OWNER_TOKEN_ENV)
    return TypedStateOwnerConnection(
        socket_path=typed_owner_socket_path(store_id),
        token=token,
        client_id=client_id,
        process_birth_id=process_birth_id,
        store_id=store_id,
        timeout_seconds=timeout_seconds,
    )


__all__ = [
    "OwnerOperation",
    "OwnerClientGrant",
    "STATUS_BOOTSTRAP_ALLOWED_OPERATIONS",
    "STATUS_BOOTSTRAP_CLIENT_ID",
    "STATUS_BOOTSTRAP_GRANT_TTL_SECONDS",
    "SUPERVISOR_EVENT_CHILD_ALLOWED_OPERATIONS",
    "SUPERVISOR_RUNTIME_CHILD_ALLOWED_OPERATIONS",
    "TYPED_STATE_OWNER_INTERFACE",
    "TYPED_STATE_OWNER_SCHEMA",
    "TYPED_STATE_OWNER_SOCKET_ENV",
    "TYPED_STATE_OWNER_SOCKET_FILENAME",
    "TYPED_STATE_OWNER_TOKEN_ENV",
    "TYPED_STATE_OWNER_TOKEN_FILENAME",
    "TYPED_RETRY_COOLDOWN_SCHEMA",
    "compact_default_owner_socket_path",
    "TYPED_TASK_STATUS_VOCABULARY",
    "TypedOwnerResult",
    "TypedStateOwnerAuthorizationError",
    "TypedStateOwnerConnection",
    "TypedStateOwnerError",
    "TypedStateOwnerGateway",
    "TypedStateOwnerProtocolError",
    "TypedStateOwnerRemoteError",
    "build_control_plane_operation_catalog",
    "catalog_fingerprint",
    "internal_operation_for_sql",
    "open_typed_state_owner_connection",
    "typed_owner_socket_path",
]
