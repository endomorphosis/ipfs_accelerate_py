"""Fail-closed binding for implementation daemons using Quack command ingress.

The legacy Quack daemon path attached the task table remotely but kept claims,
attempts, provider/effect receipts, and completion barriers in per-process
DuckDB sidecars.  Such a topology cannot coordinate parallel supervisors.

This module is deliberately a binding contract, not another store or
scheduler.  A qualified adapter supplies three closed components (task,
coordination, and execution) backed by one ``AuthorizedStateCommand@1`` /
``QuackCommandFabric@1`` owner.  The implementation daemon refuses Quack mode
unless this exact capability is present.  Components expose typed methods;
they never expose a database path, connection, or arbitrary SQL surface.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..control.profile_authority import LocalProfileTampered, verify_did_key_signature
from ..runtime.external_agent_control_plane_promotion import (
    exact_plan_r2_operation_vocabulary,
)
from .control_plane_contracts import (
    CommandKind,
    CommandOutcome,
    canonical_json_bytes,
    content_identity,
)
from .quack_command_authorization import (
    AUTHORIZED_STATE_COMMAND_INTERFACE,
    AUTHORIZED_STATE_COMMAND_SCHEMA,
    AuthorizedStateCommand,
    QuackCommandAuthorizationError,
    QuackCommandAuthorizationPolicy,
    verify_authorized_state_command,
)
from .quack_command_fabric import (
    QUACK_COMMAND_FABRIC_INTERFACE,
    QUACK_DAEMON_CANONICAL_HANDLER_INTERFACE,
    QuackCommandClient,
    QuackDaemonOwnerGateway,
    QuackPlanR2OwnerGateway,
    QuackReadClient,
)

QUACK_DAEMON_COMMAND_GATEWAY_INTERFACE: Final = "QuackDaemonCommandGateway@1"
QUACK_DAEMON_COMMAND_GATEWAY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-daemon-command-gateway@1"
)
QUACK_DAEMON_GATEWAY_COMPONENT_INTERFACE: Final = "QuackDaemonGatewayComponent@1"
QUACK_DAEMON_OWNER_DISPATCHER_INTERFACE: Final = (
    "AuthorizedStateCommandDaemonOwnerDispatcher@1"
)
QUACK_DAEMON_OPERATION_INTENT_INTERFACE: Final = "QuackDaemonOperationIntent@1"
QUACK_DAEMON_OPERATION_INTENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-daemon-operation-intent@1"
)
QUACK_DAEMON_OPERATIONAL_CAPABILITY_INTERFACE: Final = (
    "QuackDaemonOperationalCapability@1"
)
QUACK_DAEMON_OPERATIONAL_CAPABILITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-daemon-operational-capability@1"
)
QUACK_DAEMON_CANONICAL_HANDLER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-daemon-canonical-owner-handler@1"
)
QUACK_DAEMON_OPERATION_DISPOSITION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-daemon-operation-disposition@1"
)
QUACK_DAEMON_HANDLER_QUALIFICATION_STATUS: Final = (
    "implemented_unqualified_fail_closed"
)

# Leave deterministic headroom below AuthorizedStateCommand@1's 73,728-byte
# ingress limit for the envelope, signatures, authority bindings, and CIDs.
# Large values must cross as content references, not inline daemon arguments.
MAX_DAEMON_OPERATION_ARGUMENT_BYTES: Final = 48 * 1024
MAX_DAEMON_OPERATION_CAPABILITY_LIFETIME_MS: Final = 15 * 60 * 1000

# This is the complete v1 mutation/read vocabulary needed by the execution
# daemon plus the already admitted merge and Plan-R2 paths.  A capability may
# not silently omit a method or add an unreviewed escape hatch.
REQUIRED_QUACK_DAEMON_OPERATIONS: Final = frozenset(
    {
        "task.materialize",
        "task.list",
        "task.ready",
        "task.get",
        "task.cas_status",
        "task.record_validation",
        "coordination.register_task",
        "coordination.claim_ready",
        "coordination.get_claim",
        "coordination.protect_claim",
        "coordination.renew_lease",
        "coordination.prepare_completion",
        "coordination.get_prepared_completion",
        "coordination.complete_claim",
        "coordination.settle_claim",
        "coordination.list_unsettled_completions",
        "coordination.reconcile_promoted_completion",
        "coordination.recover_prepared_completion",
        "coordination.abort_prepared_completion",
        "coordination.expire_claim",
        "execution.bind_daemon",
        "execution.record_event",
        "execution.ensure_attempt",
        "execution.get_attempt",
        "execution.list_running_attempts",
        "execution.commit_phase",
        "execution.commit_reconciled_attempt",
        "execution.phase_history",
        "provider.reserve",
        "provider.commit",
        "effect.reserve",
        "effect.commit",
        "validation.record",
        "merge.enqueue",
        "merge.observe",
        "merge.accept",
        "plan_r2.prepare",
        "plan_r2.apply",
        "plan_r2.observe",
    }
)

# Every daemon operation is carried by the existing StateCommand@1 vocabulary.
# This map is intentionally total and closed: adding a daemon operation requires
# a protocol revision and a review of both its StateCommand effect and its owner
# transaction implementation.
_OPERATION_COMMAND_KINDS: Final[Mapping[str, CommandKind]] = MappingProxyType(
    {
        "task.materialize": CommandKind.APPEND,
        "task.list": CommandKind.OBSERVE,
        "task.ready": CommandKind.OBSERVE,
        "task.get": CommandKind.OBSERVE,
        "task.cas_status": CommandKind.CLAIM,
        "task.record_validation": CommandKind.APPEND,
        "coordination.register_task": CommandKind.APPEND,
        "coordination.claim_ready": CommandKind.CLAIM,
        "coordination.get_claim": CommandKind.OBSERVE,
        "coordination.protect_claim": CommandKind.OBSERVE,
        "coordination.renew_lease": CommandKind.RENEW,
        "coordination.prepare_completion": CommandKind.PROJECT,
        "coordination.get_prepared_completion": CommandKind.OBSERVE,
        "coordination.complete_claim": CommandKind.RELEASE,
        "coordination.settle_claim": CommandKind.RELEASE,
        "coordination.list_unsettled_completions": CommandKind.OBSERVE,
        "coordination.reconcile_promoted_completion": CommandKind.RECOVER,
        "coordination.recover_prepared_completion": CommandKind.RECOVER,
        "coordination.abort_prepared_completion": CommandKind.RECOVER,
        "coordination.expire_claim": CommandKind.RELEASE,
        "execution.bind_daemon": CommandKind.APPEND,
        "execution.record_event": CommandKind.APPEND,
        "execution.ensure_attempt": CommandKind.APPEND,
        "execution.get_attempt": CommandKind.OBSERVE,
        "execution.list_running_attempts": CommandKind.OBSERVE,
        "execution.commit_phase": CommandKind.PROJECT,
        "execution.commit_reconciled_attempt": CommandKind.RECOVER,
        "execution.phase_history": CommandKind.OBSERVE,
        "provider.reserve": CommandKind.CLAIM,
        "provider.commit": CommandKind.RELEASE,
        "effect.reserve": CommandKind.CLAIM,
        "effect.commit": CommandKind.RELEASE,
        "validation.record": CommandKind.APPEND,
        "merge.enqueue": CommandKind.APPEND,
        "merge.observe": CommandKind.OBSERVE,
        "merge.accept": CommandKind.PROJECT,
        "plan_r2.prepare": CommandKind.OBSERVE,
        "plan_r2.apply": CommandKind.MIGRATE,
        "plan_r2.observe": CommandKind.OBSERVE,
    }
)
if frozenset(_OPERATION_COMMAND_KINDS) != REQUIRED_QUACK_DAEMON_OPERATIONS:
    raise RuntimeError("daemon operation/StateCommand registry is incomplete")

# Plan-R2 already has a promoted closed vocabulary.  The daemon superset must
# reuse it exactly; it is not permitted to reinterpret those operations under
# generic daemon command kinds.
_PROMOTED_PLAN_R2_COMMAND_KINDS: Final[Mapping[str, CommandKind]] = MappingProxyType(
    {
        str(item["operation"]): CommandKind(str(item["command_kind"]))
        for item in exact_plan_r2_operation_vocabulary()
    }
)
if {
    operation: _OPERATION_COMMAND_KINDS[operation]
    for operation in _PROMOTED_PLAN_R2_COMMAND_KINDS
} != dict(_PROMOTED_PLAN_R2_COMMAND_KINDS):
    raise RuntimeError("daemon Plan-R2 registry differs from Promotion@2 vocabulary")


def quack_daemon_operation_command_vocabulary() -> Mapping[str, str]:
    """Return the detached exact daemon operation/StateCommand-kind registry."""

    return MappingProxyType(
        {
            operation: command_kind.value
            for operation, command_kind in _OPERATION_COMMAND_KINDS.items()
        }
    )

_OPERATIONAL_GUARANTEES: Final = frozenset(
    {
        "one_mutable_owner",
        "operational_database_private",
        "authorized_state_command_required",
        "owner_verifies_command_signature",
        "live_lease_verified_in_transaction",
        "fencing_token_verified_in_transaction",
        "replay_claims_consumed_in_transaction",
        "cas_and_effect_applied_in_transaction",
        "durable_idempotent_receipt",
        "no_portal_fallback",
        "no_local_sidecar",
        "no_direct_database_open",
        "no_arbitrary_sql",
    }
)
_OPERATIONAL_CAPABILITY_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "board_namespace",
        "shard_id",
        "store_id",
        "control_plane_schema_version",
        "state_schema_revision",
        "command_endpoint",
        "state_endpoint",
        "owner_principal_did",
        "owner_generation",
        "fence_epoch",
        "authorization_policy_cid",
        "command_fabric_qualification_cid",
        "authorized_state_command_schema",
        "authorized_state_command_interface",
        "dispatcher_interface",
        "operations",
        "guarantees",
        "allowed",
        "blockers",
        "issued_at_ms",
        "expires_at_ms",
        "reviewer_identity_did",
        "reviewer_signature",
        "capability_cid",
    }
)

_TASK_COMPONENT_METHODS: Final = frozenset(
    {
        "materialize",
        "list_tasks",
        "ready_tasks",
        "get",
        "compare_and_set_status",
        "record_validation_result",
    }
)
_COORDINATION_COMPONENT_METHODS: Final = frozenset(
    {
        "register_task",
        "claim_ready_task",
        "get_task_claim",
        "protect_task_claim",
        "renew",
        "prepare_task_completion",
        "get_prepared_task_completion",
        "complete_task_claim",
        "settle_task_claim",
        "list_unsettled_task_completions",
        "reconcile_promoted_task_completion",
        "recover_prepared_task_completion",
        "abort_prepared_task_completion",
        "expire_task_claim",
    }
)
_EXECUTION_COMPONENT_METHODS: Final = frozenset(
    {
        "bind_daemon",
        "record_event",
        "ensure_attempt",
        "get_attempt",
        "list_running_attempts",
        "commit_phase",
        "commit_reconciled_attempt",
        "phase_history",
        "get_idempotent_result",
        "record_idempotent_result",
        "reserve_provider",
        "commit_provider",
        "reserve_effect",
        "commit_effect",
        "record_validation",
    }
)
_MERGE_COMPONENT_METHODS: Final = frozenset({"enqueue", "observe", "accept"})
_PLAN_COMPONENT_METHODS: Final = frozenset({"prepare", "apply", "observe"})

_OPERATION_INTENT_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "gateway_binding_cid",
        "operational_capability_cid",
        "operation",
        "arguments",
        "arguments_cid",
        "intent_cid",
    }
)


class QuackDaemonGatewayError(RuntimeError):
    """A daemon command gateway is missing or not exactly admitted."""


class QuackDaemonOwnerOperationNoGo(QuackDaemonGatewayError):
    """One recognized daemon operation has no safe owner-transaction adapter.

    This is deliberately distinct from an unknown operation.  The complete v1
    registry is reviewed below, but an entry is executable only when its
    canonical repository semantics fit the already-active ``StateTransaction``
    without a nested commit, a second database, or an external effect.
    """

    def __init__(self, operation: str, reason_code: str) -> None:
        self.operation = str(operation)
        self.reason_code = str(reason_code)
        super().__init__(
            "daemon_operation_no_go:"
            f"operation={self.operation};reason_code={self.reason_code}"
        )


def _text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text or "\x00" in text or len(text.encode("utf-8")) > 512:
        raise QuackDaemonGatewayError(f"{name} must be a bounded non-empty string")
    return text


def _positive(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise QuackDaemonGatewayError(f"{name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise QuackDaemonGatewayError(f"{name} must be a positive integer") from exc
    if number < 1:
        raise QuackDaemonGatewayError(f"{name} must be a positive integer")
    return number


def _loopback_quack_uri(value: Any, name: str) -> str:
    uri = _text(value, name)
    if not uri.startswith("quack:"):
        raise QuackDaemonGatewayError(f"{name} must be a quack loopback endpoint")
    host_port = uri.removeprefix("quack:").removeprefix("//")
    host, separator, port_text = host_port.rpartition(":")
    try:
        address = ipaddress.ip_address(host.strip("[]"))
        port = int(port_text)
    except (ValueError, TypeError) as exc:
        raise QuackDaemonGatewayError(f"{name} is not a valid endpoint") from exc
    if not separator or not address.is_loopback or not 1024 <= port <= 65535:
        raise QuackDaemonGatewayError(f"{name} must use a loopback address and unprivileged port")
    return uri


def _sha256_cid(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def _strict_positive(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise QuackDaemonGatewayError(f"{name} must be a positive integer")
    return value


def _sha256(value: Any, name: str) -> str:
    text = _text(value, name)
    if not text.startswith("sha256:") or len(text) != 71:
        raise QuackDaemonGatewayError(f"{name} must be a sha256 content identity")
    try:
        int(text.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise QuackDaemonGatewayError(
            f"{name} must be a sha256 content identity"
        ) from exc
    return text


_OWNER_TRANSACTION_ADMITTED_OPERATIONS: Final = frozenset(
    {"task.list", "task.ready", "task.get"}
)
_OWNER_TRANSACTION_NO_GO_REASONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "task.materialize": "plan_r2_population_transition_required",
        "task.cas_status": "canonical_intent_completion_gate_transaction_adapter_unavailable",
        "task.record_validation": "canonical_intent_evidence_transaction_adapter_unavailable",
        **{
            operation: "canonical_coordination_schema_transaction_adapter_unavailable"
            for operation in REQUIRED_QUACK_DAEMON_OPERATIONS
            if operation.startswith("coordination.")
        },
        **{
            operation: "canonical_execution_schema_transaction_adapter_unavailable"
            for operation in REQUIRED_QUACK_DAEMON_OPERATIONS
            if operation.startswith("execution.")
        },
        "provider.reserve": "provider_reservation_before_container_launch_unqualified",
        "provider.commit": "provider_receipt_independent_verification_unqualified",
        "effect.reserve": "effect_reservation_before_external_effect_unqualified",
        "effect.commit": "effect_receipt_independent_verification_unqualified",
        "validation.record": "canonical_validation_schema_transaction_adapter_unavailable",
        "merge.enqueue": "host_merge_admission_is_separate_authority",
        "merge.observe": "host_merge_admission_is_separate_authority",
        "merge.accept": "host_merge_admission_is_separate_authority",
        "plan_r2.prepare": "dedicated_plan_r2_owner_gateway_required",
        "plan_r2.apply": "dedicated_plan_r2_owner_gateway_required",
        "plan_r2.observe": "dedicated_plan_r2_owner_gateway_required",
    }
)
if (
    _OWNER_TRANSACTION_ADMITTED_OPERATIONS
    | frozenset(_OWNER_TRANSACTION_NO_GO_REASONS)
) != REQUIRED_QUACK_DAEMON_OPERATIONS:
    raise RuntimeError("canonical daemon owner disposition registry is incomplete")
if _OWNER_TRANSACTION_ADMITTED_OPERATIONS & frozenset(
    _OWNER_TRANSACTION_NO_GO_REASONS
):
    raise RuntimeError("canonical daemon owner disposition registry overlaps")


def quack_daemon_owner_operation_dispositions() -> Mapping[str, Mapping[str, Any]]:
    """Return the exact reviewed owner-transaction disposition for all 39 ops.

    ``admitted_owner_transaction`` is intentionally narrow.  A typed no-go is
    preferable to adapting a canonical repository that would open another
    connection, commit a nested transaction, or use an incompatible table
    layout.  This matrix is source capability evidence, not a production
    signature or a qualification receipt.
    """

    return MappingProxyType(
        {
            operation: MappingProxyType(
                {
                    "schema": QUACK_DAEMON_OPERATION_DISPOSITION_SCHEMA,
                    "operation": operation,
                    "command_kind": _OPERATION_COMMAND_KINDS[operation].value,
                    "disposition": (
                        "admitted_owner_transaction"
                        if operation in _OWNER_TRANSACTION_ADMITTED_OPERATIONS
                        else "typed_no_go"
                    ),
                    "reason_code": _OWNER_TRANSACTION_NO_GO_REASONS.get(
                        operation, ""
                    ),
                }
            )
            for operation in sorted(REQUIRED_QUACK_DAEMON_OPERATIONS)
        }
    )


class QuackDaemonCanonicalOwnerOperationHandler:
    """Built-in, closed owner handler for the complete daemon vocabulary.

    The handler recognizes every v1 operation and admits only operations that
    are losslessly expressible on the existing datasets-authoritative control
    schema in the caller's already-active ``StateTransaction``.  It never
    opens a path, creates a connection, commits, rolls back, performs an
    external effect, or accepts an injected callback.

    The three admitted task queries reuse the canonical intent projection
    tables and readiness rules.  The other entries return a stable typed
    no-go until their existing canonical repositories gain a borrowed-
    transaction adapter and schema compatibility.  In particular, Plan-R2
    remains owned by ``AuthorizedStateCommandPlanR2OwnerGateway@1`` and merge
    acceptance remains a separate host authority.
    """

    INTERFACE: ClassVar[str] = QUACK_DAEMON_CANONICAL_HANDLER_INTERFACE
    SCHEMA: ClassVar[str] = QUACK_DAEMON_CANONICAL_HANDLER_SCHEMA
    QUALIFICATION_STATUS: ClassVar[str] = QUACK_DAEMON_HANDLER_QUALIFICATION_STATUS

    _READY_STATUSES: ClassVar[frozenset[str]] = frozenset(
        {"proposed", "admitted", "pending", "ready", "todo", "queued", "retrying"}
    )
    _COMPLETED_STATUSES: ClassVar[frozenset[str]] = frozenset(
        {"completed", "skipped", "complete", "done"}
    )
    _MAX_QUERY_LIMIT: ClassVar[int] = 1_000

    @classmethod
    def operation_dispositions(cls) -> Mapping[str, Mapping[str, Any]]:
        del cls
        return quack_daemon_owner_operation_dispositions()

    @classmethod
    def operation_admitted(cls, operation: str) -> bool:
        del cls
        return str(operation) in _OWNER_TRANSACTION_ADMITTED_OPERATIONS

    @classmethod
    def require_operation(cls, operation: str) -> None:
        del cls
        name = str(operation or "")
        if name not in REQUIRED_QUACK_DAEMON_OPERATIONS:
            raise QuackDaemonGatewayError(
                "daemon operation is outside the closed v1 vocabulary"
            )
        reason = _OWNER_TRANSACTION_NO_GO_REASONS.get(name)
        if reason:
            raise QuackDaemonOwnerOperationNoGo(name, reason)

    @staticmethod
    def _active_connection(transaction: Any) -> Any:
        if (
            getattr(transaction, "INTERFACE", "") != "StateTransaction@1"
            or not getattr(transaction, "active", False)
        ):
            raise QuackDaemonGatewayError(
                "canonical daemon owner operation requires an active StateTransaction@1"
            )
        connection = getattr(transaction, "_connection", None)
        if connection is None or not callable(getattr(connection, "execute", None)):
            raise QuackDaemonGatewayError(
                "canonical daemon owner transaction lost its private connection"
            )
        return connection

    @classmethod
    def _limit(cls, arguments: Mapping[str, Any]) -> int:
        if set(arguments) != {"limit"}:
            raise QuackDaemonGatewayError("task query arguments are not the exact v1 shape")
        value = arguments.get("limit")
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= cls._MAX_QUERY_LIMIT
        ):
            raise QuackDaemonGatewayError(
                f"task query limit must be in [1, {cls._MAX_QUERY_LIMIT}]"
            )
        return value

    @staticmethod
    def _decode_object(raw: Any, *, noun: str) -> dict[str, Any]:
        try:
            value = json.loads(str(raw or "{}"))
        except (TypeError, ValueError) as exc:
            raise QuackDaemonGatewayError(f"{noun} is corrupt") from exc
        if not isinstance(value, Mapping):
            raise QuackDaemonGatewayError(f"{noun} is not an object")
        return dict(value)

    @classmethod
    def _task_record(cls, connection: Any, task_key: str) -> dict[str, Any] | None:
        key = _text(task_key, "task_cid")
        rows = connection.execute(
            """
            SELECT task_cid, task_alias, goal_cid, plan_cid, objective_id,
                   ordinal, status, revision, priority, body_json
            FROM tasks
            WHERE task_cid = ? OR task_alias = ?
            ORDER BY task_cid
            LIMIT 2
            """,
            [key, key],
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise QuackDaemonGatewayError("task CID/alias lookup is ambiguous")
        row = rows[0]
        task_cid = str(row[0])
        dependencies = [
            str(item[0])
            for item in connection.execute(
                "SELECT dependency_task_cid FROM task_dependencies "
                "WHERE task_cid = ? ORDER BY dependency_task_cid",
                [task_cid],
            ).fetchall()
        ]
        outputs = [
            {
                "ordinal": int(item[0]),
                "path": str(item[1]),
                "effect": cls._decode_object(item[2], noun="task output effect"),
            }
            for item in connection.execute(
                "SELECT ordinal, path, effect_json FROM task_outputs "
                "WHERE task_cid = ? ORDER BY ordinal",
                [task_cid],
            ).fetchall()
        ]
        acceptance = [
            {
                "ordinal": int(item[0]),
                "criterion": str(item[1]),
                "evidence_policy": cls._decode_object(
                    item[2], noun="task acceptance policy"
                ),
            }
            for item in connection.execute(
                "SELECT ordinal, criterion, evidence_policy_json "
                "FROM task_acceptance WHERE task_cid = ? ORDER BY ordinal",
                [task_cid],
            ).fetchall()
        ]
        validations = []
        for item in connection.execute(
            "SELECT ordinal, argv_json, policy_json FROM task_validations "
            "WHERE task_cid = ? ORDER BY ordinal",
            [task_cid],
        ).fetchall():
            try:
                argv = json.loads(str(item[1] or "[]"))
            except (TypeError, ValueError) as exc:
                raise QuackDaemonGatewayError("task validation argv is corrupt") from exc
            if not isinstance(argv, list) or not all(isinstance(part, str) for part in argv):
                raise QuackDaemonGatewayError("task validation argv is not a string list")
            validations.append(
                {
                    "ordinal": int(item[0]),
                    "argv": argv,
                    "policy": cls._decode_object(
                        item[2], noun="task validation policy"
                    ),
                }
            )
        return {
            "task_cid": task_cid,
            "task_alias": str(row[1]),
            "goal_cid": str(row[2]),
            "plan_cid": str(row[3] or ""),
            "objective_id": str(row[4] or ""),
            "ordinal": int(row[5]),
            "status": str(row[6]),
            "revision": int(row[7]),
            "priority": str(row[8] or ""),
            "body": cls._decode_object(row[9], noun="task body"),
            "dependencies": dependencies,
            "outputs": outputs,
            "acceptance": acceptance,
            "validations": validations,
        }

    @classmethod
    def _list_tasks(
        cls, transaction: Any, arguments: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        limit = cls._limit(arguments)
        connection = cls._active_connection(transaction)
        task_cids = [
            str(row[0])
            for row in connection.execute(
                "SELECT task_cid FROM tasks ORDER BY ordinal, task_cid LIMIT ?",
                [limit],
            ).fetchall()
        ]
        records = [cls._task_record(connection, task_cid) for task_cid in task_cids]
        return {
            "tasks": [record for record in records if record is not None],
            "revision": int(transaction.load_generation().revision),
            "next_cursor": "",
        }

    @classmethod
    def _ready_tasks(
        cls, transaction: Any, arguments: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        limit = cls._limit(arguments)
        connection = cls._active_connection(transaction)
        task_rows = connection.execute(
            "SELECT task_cid, status FROM tasks ORDER BY ordinal, task_cid"
        ).fetchall()
        dependencies: dict[str, set[str]] = {}
        for row in connection.execute(
            "SELECT task_cid, dependency_task_cid FROM task_dependencies"
        ).fetchall():
            dependencies.setdefault(str(row[0]), set()).add(str(row[1]))
        completed = {
            str(row[0])
            for row in task_rows
            if str(row[1]) in cls._COMPLETED_STATUSES
        }
        active_blocks = {
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT task_cid FROM task_blocks WHERE state = 'active'"
            ).fetchall()
        }
        now_ms = int(time.time_ns() // 1_000_000)
        cooldown = {
            str(row[0]): int(row[1] or 0)
            for row in connection.execute(
                "SELECT task_cid, retry_not_before_ms FROM leases"
            ).fetchall()
        }
        selected: list[dict[str, Any]] = []
        for task_cid, status in task_rows:
            key = str(task_cid)
            if (
                str(status) not in cls._READY_STATUSES
                or key in active_blocks
                or cooldown.get(key, 0) > now_ms
                or not dependencies.get(key, set()).issubset(completed)
            ):
                continue
            record = cls._task_record(connection, key)
            if record is not None:
                selected.append(record)
            if len(selected) >= limit:
                break
        return {
            "tasks": selected,
            "revision": int(transaction.load_generation().revision),
            "next_cursor": "",
        }

    def apply_authorized_daemon_operation(
        self,
        *,
        operation: str,
        arguments: Mapping[str, Any],
        transaction: Any,
        command: Any,
        lease: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Apply one admitted operation without owning transaction lifecycle."""

        name = str(operation or "")
        self.require_operation(name)
        if not isinstance(arguments, Mapping) or not isinstance(lease, Mapping):
            raise QuackDaemonGatewayError(
                "canonical daemon owner operation lost typed arguments or lease"
            )
        if getattr(command, "command_kind", None) is not _OPERATION_COMMAND_KINDS[name]:
            raise QuackDaemonGatewayError(
                "canonical daemon owner operation command kind changed after admission"
            )
        if name == "task.list":
            value: Any = self._list_tasks(transaction, arguments)
        elif name == "task.ready":
            value = self._ready_tasks(transaction, arguments)
        elif name == "task.get":
            if set(arguments) != {"task_cid"}:
                raise QuackDaemonGatewayError(
                    "task.get arguments are not the exact v1 shape"
                )
            value = self._task_record(
                self._active_connection(transaction),
                _text(arguments.get("task_cid"), "task_cid"),
            )
        else:  # pragma: no cover - require_operation is the closed guard.
            raise QuackDaemonGatewayError("canonical daemon operation registry diverged")
        canonical_json_bytes(value)
        return MappingProxyType({"value": value})

    def evidence(self) -> Mapping[str, Any]:
        dispositions = self.operation_dispositions()
        admitted = sorted(
            operation
            for operation, record in dispositions.items()
            if record["disposition"] == "admitted_owner_transaction"
        )
        no_go = sorted(set(dispositions) - set(admitted))
        return MappingProxyType(
            {
                "schema": self.SCHEMA,
                "interface": self.INTERFACE,
                "qualification_status": self.QUALIFICATION_STATUS,
                "operation_count": len(dispositions),
                "admitted_operations": admitted,
                "typed_no_go_operations": no_go,
                "all_operations_recognized": set(dispositions)
                == REQUIRED_QUACK_DAEMON_OPERATIONS,
                "production_admitted": False,
                "opens_database": False,
                "owns_transaction_lifecycle": False,
                "performs_external_effects": False,
            }
        )


def _validate_operational_capability_body(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the closed signed body without treating it as admitted."""

    if set(value) != _OPERATIONAL_CAPABILITY_FIELDS - {
        "reviewer_signature",
        "capability_cid",
    }:
        raise QuackDaemonGatewayError(
            "daemon operational capability signing body is not the exact v1 schema"
        )
    body = dict(value)
    if (
        body.get("schema") != QUACK_DAEMON_OPERATIONAL_CAPABILITY_SCHEMA
        or body.get("interface") != QUACK_DAEMON_OPERATIONAL_CAPABILITY_INTERFACE
    ):
        raise QuackDaemonGatewayError("daemon operational capability schema is unsupported")
    for name in (
        "board_namespace",
        "shard_id",
        "store_id",
        "control_plane_schema_version",
        "state_schema_revision",
    ):
        body[name] = _text(body.get(name), name)
    body["command_endpoint"] = _loopback_quack_uri(
        body.get("command_endpoint"), "command_endpoint"
    )
    body["state_endpoint"] = _loopback_quack_uri(
        body.get("state_endpoint"), "state_endpoint"
    )
    if body["command_endpoint"] == body["state_endpoint"]:
        raise QuackDaemonGatewayError("command and state endpoints must be distinct")
    for name in ("owner_principal_did", "reviewer_identity_did"):
        identity = _text(body.get(name), name)
        if not identity.startswith("did:key:z"):
            raise QuackDaemonGatewayError(f"{name} must be an Ed25519 did:key")
        body[name] = identity
    if body["owner_principal_did"] == body["reviewer_identity_did"]:
        raise QuackDaemonGatewayError(
            "the mutable owner cannot review its own operational capability"
        )
    for name in ("owner_generation", "fence_epoch", "issued_at_ms", "expires_at_ms"):
        body[name] = _strict_positive(body.get(name), name)
    if (
        body["expires_at_ms"] <= body["issued_at_ms"]
        or body["expires_at_ms"] - body["issued_at_ms"]
        > MAX_DAEMON_OPERATION_CAPABILITY_LIFETIME_MS
    ):
        raise QuackDaemonGatewayError(
            "daemon operational capability lifetime is invalid or too broad"
        )
    for name in ("authorization_policy_cid", "command_fabric_qualification_cid"):
        body[name] = _sha256(body.get(name), name)
    if (
        body.get("authorized_state_command_schema")
        != AUTHORIZED_STATE_COMMAND_SCHEMA
        or body.get("authorized_state_command_interface")
        != AUTHORIZED_STATE_COMMAND_INTERFACE
        or body.get("dispatcher_interface")
        != QUACK_DAEMON_OWNER_DISPATCHER_INTERFACE
    ):
        raise QuackDaemonGatewayError(
            "daemon operational capability binds an unsupported command/dispatcher contract"
        )
    operations = body.get("operations")
    if (
        not isinstance(operations, list)
        or operations != sorted(REQUIRED_QUACK_DAEMON_OPERATIONS)
    ):
        raise QuackDaemonGatewayError(
            "daemon operational capability operation vocabulary is not exact"
        )
    guarantees = body.get("guarantees")
    if (
        not isinstance(guarantees, Mapping)
        or set(guarantees) != _OPERATIONAL_GUARANTEES
        or any(guarantees[name] is not True for name in _OPERATIONAL_GUARANTEES)
    ):
        raise QuackDaemonGatewayError(
            "daemon operational capability guarantees are incomplete"
        )
    body["guarantees"] = {
        name: True for name in sorted(_OPERATIONAL_GUARANTEES)
    }
    if body.get("allowed") is not True or body.get("blockers") != []:
        raise QuackDaemonGatewayError(
            "daemon operational capability is not an unblocked external admission"
        )
    return body


def quack_daemon_operational_capability_signing_payload(
    capability_body: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return canonical public bytes for an independent external signer.

    This helper never loads a key or creates a signature.  In particular, the
    owner process cannot turn structural self-assertions into production
    authority.
    """

    if not isinstance(capability_body, Mapping):
        raise QuackDaemonGatewayError("daemon operational capability body must be an object")
    return MappingProxyType(_validate_operational_capability_body(capability_body))


def seal_quack_daemon_operational_capability(
    prepared_payload: Mapping[str, Any],
    *,
    reviewer_signature: str,
) -> Mapping[str, Any]:
    """Join externally produced signature bytes to a canonical capability."""

    body = dict(quack_daemon_operational_capability_signing_payload(prepared_payload))
    signature = _text(reviewer_signature, "reviewer_signature")
    signed = {**body, "reviewer_signature": signature}
    return MappingProxyType({**signed, "capability_cid": _sha256_cid(signed)})


def verify_quack_daemon_operational_capability(
    capability: Mapping[str, Any],
    *,
    trusted_reviewer_dids: Sequence[str],
    now_ms: int,
) -> Mapping[str, Any]:
    """Verify the exact signed v1 production capability and its lifetime."""

    if not isinstance(capability, Mapping) or set(capability) != _OPERATIONAL_CAPABILITY_FIELDS:
        raise QuackDaemonGatewayError(
            "daemon operational capability must use the exact closed v1 schema"
        )
    unsigned = dict(capability)
    claimed_cid = _sha256(unsigned.pop("capability_cid", None), "capability_cid")
    signature = _text(unsigned.pop("reviewer_signature", None), "reviewer_signature")
    body = _validate_operational_capability_body(unsigned)
    signed = {**body, "reviewer_signature": signature}
    if claimed_cid != _sha256_cid(signed):
        raise QuackDaemonGatewayError("daemon operational capability CID mismatch")
    now = _strict_positive(now_ms, "now_ms")
    if body["issued_at_ms"] > now or now >= body["expires_at_ms"]:
        raise QuackDaemonGatewayError("daemon operational capability is not currently valid")
    reviewers = frozenset(_text(value, "trusted_reviewer_did") for value in trusted_reviewer_dids)
    reviewer = body["reviewer_identity_did"]
    if not reviewers or reviewer not in reviewers:
        raise QuackDaemonGatewayError("daemon operational capability reviewer is not trusted")
    try:
        verify_did_key_signature(
            identity_did=reviewer,
            payload=body,
            signature=signature,
        )
    except (LocalProfileTampered, ValueError) as exc:
        raise QuackDaemonGatewayError(
            "daemon operational capability signature is invalid"
        ) from exc
    return MappingProxyType({**signed, "capability_cid": claimed_cid})


@dataclass(frozen=True)
class QuackDaemonGatewayCapability:
    """Content-bound qualification for one complete command-gateway shard."""

    board_namespace: str
    shard_id: str
    store_id: str
    control_plane_schema_version: str
    state_schema_revision: str
    command_endpoint: str
    state_endpoint: str
    owner_principal_did: str
    owner_generation: int
    fence_epoch: int
    authorization_policy_cid: str
    command_fabric_qualification_cid: str
    operations: frozenset[str] = REQUIRED_QUACK_DAEMON_OPERATIONS
    operational_capability_cid: str = ""
    qualification_status: str = "structural_only_unavailable"
    production_admitted: bool = False
    direct_database_open: bool = False
    local_sidecar_writes: bool = False
    operational_database_served: bool = False
    arbitrary_sql_enabled: bool = False
    transport_token_is_authority: bool = False

    SCHEMA: ClassVar[str] = QUACK_DAEMON_COMMAND_GATEWAY_SCHEMA
    INTERFACE: ClassVar[str] = QUACK_DAEMON_COMMAND_GATEWAY_INTERFACE

    def __post_init__(self) -> None:
        for name in (
            "board_namespace",
            "shard_id",
            "store_id",
            "control_plane_schema_version",
            "state_schema_revision",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "command_endpoint",
            _loopback_quack_uri(self.command_endpoint, "command_endpoint"),
        )
        object.__setattr__(
            self,
            "state_endpoint",
            _loopback_quack_uri(self.state_endpoint, "state_endpoint"),
        )
        if self.command_endpoint == self.state_endpoint:
            raise QuackDaemonGatewayError("command and state endpoints must be distinct")
        owner = _text(self.owner_principal_did, "owner_principal_did")
        if not owner.startswith("did:key:z"):
            raise QuackDaemonGatewayError("owner_principal_did must be an Ed25519 did:key")
        object.__setattr__(self, "owner_principal_did", owner)
        object.__setattr__(
            self, "owner_generation", _positive(self.owner_generation, "owner_generation")
        )
        object.__setattr__(self, "fence_epoch", _positive(self.fence_epoch, "fence_epoch"))
        for name in (
            "authorization_policy_cid",
            "command_fabric_qualification_cid",
        ):
            value = _text(getattr(self, name), name)
            if not value.startswith("sha256:") or len(value) != 71:
                raise QuackDaemonGatewayError(f"{name} must be a sha256 content identity")
            object.__setattr__(self, name, value)
        operations = frozenset(_text(item, "operation") for item in self.operations)
        if operations != REQUIRED_QUACK_DAEMON_OPERATIONS:
            missing = sorted(REQUIRED_QUACK_DAEMON_OPERATIONS - operations)
            extra = sorted(operations - REQUIRED_QUACK_DAEMON_OPERATIONS)
            detail = []
            if missing:
                detail.append("missing=" + ",".join(missing))
            if extra:
                detail.append("unreviewed=" + ",".join(extra))
            raise QuackDaemonGatewayError(
                "gateway operation vocabulary is not exact"
                + (": " + "; ".join(detail) if detail else "")
            )
        object.__setattr__(self, "operations", operations)
        operational_cid = str(self.operational_capability_cid or "").strip()
        if self.qualification_status == "structural_only_unavailable":
            if self.production_admitted is not False or operational_cid:
                raise QuackDaemonGatewayError(
                    "structural gateway binding cannot claim production admission"
                )
        elif self.qualification_status == "signed_external_capability_verified_unqualified":
            if self.production_admitted is not False:
                raise QuackDaemonGatewayError(
                    "external capability verification alone cannot admit production execution"
                )
            operational_cid = _sha256(
                operational_cid, "operational_capability_cid"
            )
        else:
            raise QuackDaemonGatewayError(
                "gateway qualification status is not a closed v1 value"
            )
        object.__setattr__(self, "operational_capability_cid", operational_cid)
        unsafe = {
            "direct_database_open": self.direct_database_open,
            "local_sidecar_writes": self.local_sidecar_writes,
            "operational_database_served": self.operational_database_served,
            "arbitrary_sql_enabled": self.arbitrary_sql_enabled,
            "transport_token_is_authority": self.transport_token_is_authority,
        }
        enabled = sorted(name for name, value in unsafe.items() if value is not False)
        if enabled:
            raise QuackDaemonGatewayError(
                "gateway capability enables forbidden authority surfaces: " + ", ".join(enabled)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "authorization_interface": AuthorizedStateCommand.INTERFACE,
            "command_fabric_interface": QUACK_COMMAND_FABRIC_INTERFACE,
            "component_interface": QUACK_DAEMON_GATEWAY_COMPONENT_INTERFACE,
            "board_namespace": self.board_namespace,
            "shard_id": self.shard_id,
            "store_id": self.store_id,
            "control_plane_schema_version": self.control_plane_schema_version,
            "state_schema_revision": self.state_schema_revision,
            "command_endpoint": self.command_endpoint,
            "state_endpoint": self.state_endpoint,
            "owner_principal_did": self.owner_principal_did,
            "owner_generation": self.owner_generation,
            "fence_epoch": self.fence_epoch,
            "authorization_policy_cid": self.authorization_policy_cid,
            "command_fabric_qualification_cid": self.command_fabric_qualification_cid,
            "operations": sorted(self.operations),
            "operational_capability_cid": self.operational_capability_cid,
            "qualification_status": self.qualification_status,
            "production_admitted": self.production_admitted,
            "direct_database_open": self.direct_database_open,
            "local_sidecar_writes": self.local_sidecar_writes,
            "operational_database_served": self.operational_database_served,
            "arbitrary_sql_enabled": self.arbitrary_sql_enabled,
            "transport_token_is_authority": self.transport_token_is_authority,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    @classmethod
    def from_verified_operational_capability(
        cls,
        capability: Mapping[str, Any],
    ) -> QuackDaemonGatewayCapability:
        """Construct the local binding from a previously verified record."""

        return cls(
            board_namespace=str(capability["board_namespace"]),
            shard_id=str(capability["shard_id"]),
            store_id=str(capability["store_id"]),
            control_plane_schema_version=str(
                capability["control_plane_schema_version"]
            ),
            state_schema_revision=str(capability["state_schema_revision"]),
            command_endpoint=str(capability["command_endpoint"]),
            state_endpoint=str(capability["state_endpoint"]),
            owner_principal_did=str(capability["owner_principal_did"]),
            owner_generation=int(capability["owner_generation"]),
            fence_epoch=int(capability["fence_epoch"]),
            authorization_policy_cid=str(capability["authorization_policy_cid"]),
            command_fabric_qualification_cid=str(
                capability["command_fabric_qualification_cid"]
            ),
            operations=frozenset(capability["operations"]),
            operational_capability_cid=str(capability["capability_cid"]),
            qualification_status="signed_external_capability_verified_unqualified",
            production_admitted=False,
        )


def _plain_value(value: Any, *, depth: int = 0) -> Any:
    """Convert one bounded gateway value into canonical JSON data."""

    if depth > 16:
        raise QuackDaemonGatewayError("daemon operation arguments exceed depth bound")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise QuackDaemonGatewayError("daemon operation arguments cannot contain floats")
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain_value(to_dict(), depth=depth + 1)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key in sorted(value, key=str):
            key = str(raw_key)
            normalized = key.lower().replace("-", "_")
            if any(
                token in normalized
                for token in (
                    "password",
                    "private_key",
                    "api_key",
                    "access_token",
                    "refresh_token",
                    "client_secret",
                )
            ):
                raise QuackDaemonGatewayError(
                    "daemon operation arguments cannot carry inline secret fields"
                )
            result[key] = _plain_value(value[raw_key], depth=depth + 1)
        return result
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_plain_value(item, depth=depth + 1) for item in value),
            key=lambda item: canonical_json_bytes(item),
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [_plain_value(item, depth=depth + 1) for item in value]
    raise QuackDaemonGatewayError(
        f"daemon operation arguments contain unsupported {type(value).__name__}"
    )


def quack_daemon_operation_intent(
    *,
    gateway_binding_cid: str,
    operational_capability_cid: str,
    operation: str,
    arguments: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build one bounded, content-addressed request for external approval."""

    binding = _text(gateway_binding_cid, "gateway_binding_cid")
    capability_cid = _sha256(
        operational_capability_cid, "operational_capability_cid"
    )
    operation_name = _text(operation, "operation")
    if operation_name not in REQUIRED_QUACK_DAEMON_OPERATIONS:
        raise QuackDaemonGatewayError("daemon operation is outside the closed v1 vocabulary")
    normalized_arguments = _plain_value(arguments)
    if not isinstance(normalized_arguments, dict):
        raise QuackDaemonGatewayError("daemon operation arguments must be an object")
    encoded = canonical_json_bytes(normalized_arguments)
    if len(encoded) > MAX_DAEMON_OPERATION_ARGUMENT_BYTES:
        raise QuackDaemonGatewayError("daemon operation arguments exceed their byte bound")
    arguments_cid = "sha256:" + hashlib.sha256(encoded).hexdigest()
    unsigned = {
        "schema": QUACK_DAEMON_OPERATION_INTENT_SCHEMA,
        "interface": QUACK_DAEMON_OPERATION_INTENT_INTERFACE,
        "gateway_binding_cid": binding,
        "operational_capability_cid": capability_cid,
        "operation": operation_name,
        "arguments": normalized_arguments,
        "arguments_cid": arguments_cid,
    }
    return MappingProxyType({**unsigned, "intent_cid": _sha256_cid(unsigned)})


def quack_daemon_state_command_parameters(
    intent: Mapping[str, Any],
    *,
    request_id: str,
    principal_did: str,
    authority_ref_cid: str,
    lease_id: str,
    scope_id: str,
    deadline_ms: int,
    fencing_token: int,
    idempotency_key: str,
) -> Mapping[str, Any]:
    """Return the exact StateCommand.parameters join for one daemon request."""

    if not isinstance(intent, Mapping) or set(intent) != _OPERATION_INTENT_FIELDS:
        raise QuackDaemonGatewayError("daemon operation intent is not the exact v1 schema")
    # StateCommand's canonical encoder accepts plain JSON values, so this
    # transport projection is deliberately a detached dict rather than a
    # MappingProxyType wrapper.
    return {
        "daemon_operation": _text(intent.get("operation"), "operation"),
        "operation_intent_cid": _sha256(intent.get("intent_cid"), "intent_cid"),
        "arguments_cid": _sha256(intent.get("arguments_cid"), "arguments_cid"),
        "gateway_binding_cid": _text(
            intent.get("gateway_binding_cid"), "gateway_binding_cid"
        ),
        "operational_capability_cid": _sha256(
            intent.get("operational_capability_cid"),
            "operational_capability_cid",
        ),
        "daemon_operation_arguments_json": canonical_json_bytes(
            dict(intent.get("arguments") or {})
        ).decode("utf-8"),
        "request_id": _text(request_id, "request_id"),
        "principal": _text(principal_did, "principal_did"),
        "authority_ref": _text(authority_ref_cid, "authority_ref_cid"),
        "lease_id": _text(lease_id, "lease_id"),
        "task_cid": _text(scope_id, "scope_id"),
        "deadline_ms": _strict_positive(deadline_ms, "deadline_ms"),
        "fencing_token": _strict_positive(fencing_token, "fencing_token"),
        "idempotency_key": _text(idempotency_key, "idempotency_key"),
    }


def quack_daemon_operation_intent_from_envelope(
    envelope: AuthorizedStateCommand,
) -> Mapping[str, Any]:
    """Recover the exact signed intent carried through command_inbox.

    Quack's v1 ingress transports a single AuthorizedStateCommand JSON value.
    The bounded intent therefore travels inside the already signed
    StateCommand.parameters instead of through a second, independently mutable
    relation.  Large artifacts remain content references, not inline archives.
    """

    if not isinstance(envelope, AuthorizedStateCommand):
        raise QuackDaemonGatewayError("daemon transport requires AuthorizedStateCommand@1")
    parameters = dict(envelope.command.parameters)
    required_transport_fields = {
        "daemon_operation",
        "operation_intent_cid",
        "arguments_cid",
        "gateway_binding_cid",
        "operational_capability_cid",
        "daemon_operation_arguments_json",
    }
    if not required_transport_fields.issubset(parameters):
        raise QuackDaemonGatewayError(
            "authorized daemon command does not carry the complete operation intent"
        )
    arguments_json = parameters["daemon_operation_arguments_json"]
    if not isinstance(arguments_json, str):
        raise QuackDaemonGatewayError("transported daemon arguments are not canonical JSON")
    try:
        arguments = json.loads(arguments_json)
    except (TypeError, ValueError) as exc:
        raise QuackDaemonGatewayError(
            "transported daemon arguments are not valid JSON"
        ) from exc
    if not isinstance(arguments, Mapping) or (
        canonical_json_bytes(arguments).decode("utf-8") != arguments_json
    ):
        raise QuackDaemonGatewayError(
            "transported daemon arguments are not a canonical object"
        )
    intent = quack_daemon_operation_intent(
        gateway_binding_cid=str(parameters["gateway_binding_cid"]),
        operational_capability_cid=str(parameters["operational_capability_cid"]),
        operation=str(parameters["daemon_operation"]),
        arguments=arguments,
    )
    if (
        intent["intent_cid"] != parameters["operation_intent_cid"]
        or intent["arguments_cid"] != parameters["arguments_cid"]
    ):
        raise QuackDaemonGatewayError(
            "transported daemon operation intent does not match its signed CIDs"
        )
    return intent


def verify_quack_daemon_operation_submission(
    envelope: AuthorizedStateCommand,
    intent: Mapping[str, Any],
    *,
    capability: QuackDaemonGatewayCapability,
    authorization_policy: QuackCommandAuthorizationPolicy,
    now_ms: int,
) -> Mapping[str, Any]:
    """Verify the complete capability/envelope/operation identity join."""

    if not isinstance(capability, QuackDaemonGatewayCapability):
        raise QuackDaemonGatewayError("typed gateway capability is required")
    if not capability.operational_capability_cid:
        raise QuackDaemonGatewayError(
            "daemon operation requires a verified external capability identity"
        )
    if not isinstance(intent, Mapping) or set(intent) != _OPERATION_INTENT_FIELDS:
        raise QuackDaemonGatewayError("daemon operation intent is not the exact v1 schema")
    plain = dict(intent)
    claimed_intent = _sha256(plain.pop("intent_cid", None), "intent_cid")
    if (
        plain.get("schema") != QUACK_DAEMON_OPERATION_INTENT_SCHEMA
        or plain.get("interface") != QUACK_DAEMON_OPERATION_INTENT_INTERFACE
        or claimed_intent != _sha256_cid(plain)
        or plain.get("gateway_binding_cid") != capability.content_id
        or plain.get("operational_capability_cid")
        != capability.operational_capability_cid
    ):
        raise QuackDaemonGatewayError("daemon operation intent identity is invalid")
    arguments = plain.get("arguments")
    if not isinstance(arguments, Mapping):
        raise QuackDaemonGatewayError("daemon operation arguments must be an object")
    encoded_arguments = canonical_json_bytes(dict(arguments))
    if (
        len(encoded_arguments) > MAX_DAEMON_OPERATION_ARGUMENT_BYTES
        or plain.get("arguments_cid")
        != "sha256:" + hashlib.sha256(encoded_arguments).hexdigest()
    ):
        raise QuackDaemonGatewayError("daemon operation argument identity is invalid")
    operation = str(plain.get("operation") or "")
    expected_kind = _OPERATION_COMMAND_KINDS.get(operation)
    if expected_kind is None:
        raise QuackDaemonGatewayError("daemon operation is outside the closed v1 vocabulary")
    try:
        verify_authorized_state_command(
            envelope,
            policy=authorization_policy,
            now_ms=now_ms,
        )
    except QuackCommandAuthorizationError as exc:
        raise QuackDaemonGatewayError("daemon operation authorization is invalid") from exc
    command = envelope.command
    parameters = dict(command.parameters)
    fencing_token = parameters.get("fencing_token")
    expected_parameters = dict(
        quack_daemon_state_command_parameters(
            intent,
            request_id=envelope.request_id,
            principal_did=envelope.principal_did,
            authority_ref_cid=envelope.authority_ref_cid,
            lease_id=envelope.lease_id,
            scope_id=envelope.scope_id,
            deadline_ms=envelope.deadline_ms,
            fencing_token=fencing_token,
            idempotency_key=command.idempotency_key,
        )
    )
    expected_command_id = f"{envelope.request_id}:{operation.replace('.', '-')}"
    checks = (
        command.command_kind is expected_kind,
        command.command_id == expected_command_id,
        command.store_id == capability.store_id,
        command.session_id == envelope.lease_id,
        command.expected_generation == capability.owner_generation,
        command.fence_epoch == capability.fence_epoch,
        parameters == expected_parameters,
        envelope.board_namespace == capability.board_namespace,
        envelope.shard_id == capability.shard_id,
        envelope.owner_principal_did == capability.owner_principal_did,
        authorization_policy.policy_cid == capability.authorization_policy_cid,
        authorization_policy.owner_generation == capability.owner_generation,
        authorization_policy.fence_epoch == capability.fence_epoch,
        authorization_policy.store_id == capability.store_id,
        authorization_policy.shard_id == capability.shard_id,
    )
    if not all(checks):
        raise QuackDaemonGatewayError(
            "daemon operation capability/envelope/CAS identity join failed"
        )
    return MappingProxyType(
        {
            "operation": operation,
            "arguments": dict(arguments),
            "intent_cid": claimed_intent,
            "fencing_token": int(fencing_token),
            "expected_version": command.expected_revision,
            "idempotency_key": command.idempotency_key,
            "one_use_nonce": envelope.one_use_nonce,
            "lease_id": envelope.lease_id,
            "scope_id": envelope.scope_id,
            "effect": envelope.effect,
        }
    )


class QuackDaemonOwnerDispatcher:
    """One closed client-side dispatcher for all five daemon proxy families.

    ``authorization_provider`` obtains a freshly signed
    AuthorizedStateCommand from an external authority service.  This class
    neither owns nor loads signing keys.  ``owner_submit`` is the narrow local
    QuackCommandFabric owner method; the callback is never exposed through a
    component and receives no SQL or database location.
    """

    INTERFACE: ClassVar[str] = QUACK_DAEMON_OWNER_DISPATCHER_INTERFACE

    def __init__(
        self,
        *,
        capability: QuackDaemonGatewayCapability,
        operational_capability: Mapping[str, Any],
        authorization_policy: QuackCommandAuthorizationPolicy,
        authorization_provider: Callable[[Mapping[str, Any]], AuthorizedStateCommand],
        owner_submit: Callable[
            [AuthorizedStateCommand, Mapping[str, Any]], Any
        ],
        clock_ms: Callable[[], int],
    ) -> None:
        if not capability.production_admitted:
            raise QuackDaemonGatewayError("owner dispatcher requires production admission")
        if not isinstance(authorization_policy, QuackCommandAuthorizationPolicy):
            raise QuackDaemonGatewayError("owner dispatcher requires a typed authorization policy")
        if not callable(authorization_provider) or not callable(owner_submit):
            raise QuackDaemonGatewayError(
                "owner dispatcher requires closed authorization and owner-submit callables"
            )
        owner_gateway = getattr(owner_submit, "__self__", None)
        if (
            type(owner_gateway) is not QuackDaemonOwnerGateway
            or getattr(owner_submit, "__name__", "")
            != "submit_authorized_daemon_operation"
            or owner_gateway.production_capability_cid
            != capability.operational_capability_cid
            or owner_gateway.command_fabric_qualification_cid
            != capability.command_fabric_qualification_cid
        ):
            raise QuackDaemonGatewayError(
                "owner_submit must be the exact admitted QuackCommandFabric owner gateway"
            )
        if not callable(clock_ms):
            raise QuackDaemonGatewayError("owner dispatcher requires a clock")
        if str(operational_capability.get("capability_cid") or "") != (
            capability.operational_capability_cid
        ):
            raise QuackDaemonGatewayError("dispatcher capability identity mismatch")
        policy_checks = (
            authorization_policy.policy_cid == capability.authorization_policy_cid,
            authorization_policy.board_namespace == capability.board_namespace,
            authorization_policy.shard_id == capability.shard_id,
            authorization_policy.store_id == capability.store_id,
            authorization_policy.owner_principal_did
            == capability.owner_principal_did,
            authorization_policy.owner_generation == capability.owner_generation,
            authorization_policy.fence_epoch == capability.fence_epoch,
        )
        if not all(policy_checks):
            raise QuackDaemonGatewayError(
                "dispatcher authorization policy differs from operational capability"
            )
        self._capability = capability
        self._operational_capability = MappingProxyType(dict(operational_capability))
        self._authorization_policy = authorization_policy
        self._authorization_provider = authorization_provider
        self._owner_submit = owner_submit
        self._clock_ms = clock_ms
        self._lock = threading.RLock()
        self._attach_count = 0

    @property
    def gateway_binding_cid(self) -> str:
        return self._capability.content_id

    @property
    def attached(self) -> bool:
        with self._lock:
            return self._attach_count > 0

    def attach(self) -> None:
        with self._lock:
            self._attach_count += 1

    def close(self) -> None:
        with self._lock:
            if self._attach_count > 0:
                self._attach_count -= 1

    def dispatch(self, operation: str, arguments: Mapping[str, Any]) -> Any:
        with self._lock:
            if self._attach_count < 1:
                raise QuackDaemonGatewayError("daemon owner dispatcher is not attached")
        now_ms = int(self._clock_ms())
        if now_ms >= int(self._operational_capability["expires_at_ms"]):
            raise QuackDaemonGatewayError("daemon operational capability expired")
        intent = quack_daemon_operation_intent(
            gateway_binding_cid=self.gateway_binding_cid,
            operational_capability_cid=self._capability.operational_capability_cid,
            operation=operation,
            arguments=arguments,
        )
        envelope = self._authorization_provider(intent)
        verify_quack_daemon_operation_submission(
            envelope,
            intent,
            capability=self._capability,
            authorization_policy=self._authorization_policy,
            now_ms=now_ms,
        )
        return self._owner_submit(envelope, intent)

    def evidence(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "interface": self.INTERFACE,
                "gateway_binding_cid": self.gateway_binding_cid,
                "operational_capability_cid": (
                    self._capability.operational_capability_cid
                ),
                "authorization_policy_cid": self._authorization_policy.policy_cid,
                "operations": sorted(REQUIRED_QUACK_DAEMON_OPERATIONS),
                "attached": self.attached,
                "database_path_exposed": False,
                "arbitrary_sql_exposed": False,
            }
        )


class QuackDaemonRemoteCommandTransport:
    """Process-remote append/poll transport for signed daemon intents.

    This closes the transport gap without claiming the complete owner handler
    is qualified.  The full intent is embedded in the already signed
    AuthorizedStateCommand and appended through QuackCommandClient's sole
    relation; results are observed through QuackReadClient's fixed receipt
    query.  Rejected or unavailable owner operations remain typed failures.
    """

    INTERFACE: ClassVar[str] = "QuackDaemonRemoteCommandTransport@1"
    QUALIFICATION_STATUS: ClassVar[str] = (
        "remote_transport_implemented_partial_owner_handler_fail_closed"
    )
    MAXIMUM_RECEIPT_WAIT_MS: ClassVar[int] = 60_000

    def __init__(
        self,
        *,
        capability: QuackDaemonGatewayCapability,
        operational_capability: Mapping[str, Any],
        authorization_policy: QuackCommandAuthorizationPolicy,
        authorization_provider: Callable[[Mapping[str, Any]], AuthorizedStateCommand],
        command_client: QuackCommandClient,
        read_client: QuackReadClient,
        clock_ms: Callable[[], int],
        maximum_wait_ms: int = 30_000,
        poll_interval_ms: int = 10,
    ) -> None:
        if capability.production_admitted:
            raise QuackDaemonGatewayError(
                "remote transport alone cannot carry production admission"
            )
        if (
            not capability.operational_capability_cid
            or operational_capability.get("capability_cid")
            != capability.operational_capability_cid
        ):
            raise QuackDaemonGatewayError(
                "remote transport requires the verified capability identity"
            )
        if not isinstance(authorization_policy, QuackCommandAuthorizationPolicy):
            raise QuackDaemonGatewayError("remote transport authorization policy is untyped")
        if type(command_client) is not QuackCommandClient or type(read_client) is not QuackReadClient:
            raise QuackDaemonGatewayError(
                "remote transport requires the closed Quack command/read clients"
            )
        if (
            command_client.endpoint != capability.command_endpoint
            or read_client.endpoint != capability.state_endpoint
        ):
            raise QuackDaemonGatewayError("remote transport endpoint identity mismatch")
        if not callable(authorization_provider) or not callable(clock_ms):
            raise QuackDaemonGatewayError(
                "remote transport requires an external authorizer and clock"
            )
        self._capability = capability
        self._operational_capability = MappingProxyType(dict(operational_capability))
        self._authorization_policy = authorization_policy
        self._authorization_provider = authorization_provider
        self._command_client = command_client
        self._read_client = read_client
        self._clock_ms = clock_ms
        self._maximum_wait_ms = _strict_positive(maximum_wait_ms, "maximum_wait_ms")
        if self._maximum_wait_ms > self.MAXIMUM_RECEIPT_WAIT_MS:
            raise QuackDaemonGatewayError(
                "maximum receipt wait exceeds the fixed transport bound"
            )
        self._poll_interval_ms = _strict_positive(poll_interval_ms, "poll_interval_ms")
        if self._poll_interval_ms > self._maximum_wait_ms:
            raise QuackDaemonGatewayError("poll interval exceeds maximum wait")
        self._closed = False

    def dispatch(self, operation: str, arguments: Mapping[str, Any]) -> Any:
        if self._closed:
            raise QuackDaemonGatewayError("remote daemon transport is closed")
        now_ms = int(self._clock_ms())
        if now_ms >= int(self._operational_capability["expires_at_ms"]):
            raise QuackDaemonGatewayError("daemon operational capability expired")
        intent = quack_daemon_operation_intent(
            gateway_binding_cid=self._capability.content_id,
            operational_capability_cid=self._capability.operational_capability_cid,
            operation=operation,
            arguments=arguments,
        )
        envelope = self._authorization_provider(intent)
        verify_quack_daemon_operation_submission(
            envelope,
            intent,
            capability=self._capability,
            authorization_policy=self._authorization_policy,
            now_ms=now_ms,
        )
        transported = quack_daemon_operation_intent_from_envelope(envelope)
        if dict(transported) != dict(intent):
            raise QuackDaemonGatewayError("authorized envelope changed the operation intent")
        self._command_client.append(envelope)
        authorized_wait_ms = min(
            self._maximum_wait_ms,
            max(0, int(envelope.deadline_ms) - now_ms),
        )
        monotonic_stop_ms = time.monotonic_ns() // 1_000_000 + authorized_wait_ms
        while time.monotonic_ns() // 1_000_000 < monotonic_stop_ms:
            for receipt in self._read_client.list_recent_receipts():
                if receipt.get("submission_id") != envelope.submission_id:
                    continue
                if receipt.get("envelope_cid") != envelope.envelope_cid:
                    raise QuackDaemonGatewayError(
                        "remote daemon receipt is bound to a divergent envelope"
                    )
                outcome = str(receipt.get("outcome") or "")
                if outcome not in {
                    CommandOutcome.ACCEPTED.value,
                    CommandOutcome.IDEMPOTENT_REPLAY.value,
                }:
                    error = str(receipt.get("error") or "owner rejected operation")
                    raise QuackDaemonGatewayError(
                        f"remote daemon operation rejected: {error}"
                    )
                try:
                    result = json.loads(str(receipt.get("result_json") or "{}"))
                except (TypeError, ValueError) as exc:
                    raise QuackDaemonGatewayError(
                        "remote daemon receipt result is corrupt"
                    ) from exc
                if (
                    not isinstance(result, Mapping)
                    or result.get("daemon_operation") != operation
                    or result.get("intent_cid") != intent["intent_cid"]
                    or set(result) != {"daemon_operation", "intent_cid", "value"}
                ):
                    raise QuackDaemonGatewayError(
                        "remote daemon receipt does not bind the exact operation intent"
                    )
                return result["value"]
            remaining_ms = monotonic_stop_ms - time.monotonic_ns() // 1_000_000
            if remaining_ms > 0:
                time.sleep(min(self._poll_interval_ms, remaining_ms) / 1000)
        raise QuackDaemonGatewayError("remote daemon operation receipt deadline expired")

    def close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        for client in (self._read_client, self._command_client):
            try:
                client.close()
            except BaseException as exc:
                errors.append(exc)
        self._closed = True
        if errors:
            raise QuackDaemonGatewayError(
                f"remote daemon transport failed close: {type(errors[0]).__name__}"
            ) from errors[0]

    def evidence(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "interface": self.INTERFACE,
                "qualification_status": self.QUALIFICATION_STATUS,
                "gateway_binding_cid": self._capability.content_id,
                "operational_capability_cid": (
                    self._capability.operational_capability_cid
                ),
                "command_endpoint": self._capability.command_endpoint,
                "state_endpoint": self._capability.state_endpoint,
                "append_only_authorized_state_command": True,
                "fixed_receipt_query": True,
                "owner_handler_qualification_status": (
                    QUACK_DAEMON_HANDLER_QUALIFICATION_STATUS
                ),
                "owner_transaction_admitted_operations": sorted(
                    _OWNER_TRANSACTION_ADMITTED_OPERATIONS
                ),
                "typed_no_go_operation_count": len(
                    _OWNER_TRANSACTION_NO_GO_REASONS
                ),
                "production_admitted": False,
                "blocker": "canonical_39_operation_owner_handler_unqualified",
            }
        )


class _GatewayRecord:
    __slots__ = ("_record",)

    def __init__(self, record: Mapping[str, Any]) -> None:
        self._record = dict(record)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._record[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> dict[str, Any]:
        return dict(self._record)


class _GatewayTaskPage:
    __slots__ = ("tasks",)

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.tasks = tuple(_GatewayRecord(record) for record in records)


class _OwnerProxy:
    GATEWAY_COMPONENT_INTERFACE = QUACK_DAEMON_GATEWAY_COMPONENT_INTERFACE
    __slots__ = ("_dispatcher", "gateway_binding_cid", "_attached")

    def __init__(self, dispatcher: QuackDaemonOwnerDispatcher) -> None:
        self._dispatcher = dispatcher
        self.gateway_binding_cid = dispatcher.gateway_binding_cid
        self._attached = False

    def attach(self) -> None:
        if not self._attached:
            self._dispatcher.attach()
            self._attached = True

    def close(self) -> None:
        if self._attached:
            self._dispatcher.close()
            self._attached = False

    def _call(self, operation: str, arguments: Mapping[str, Any]) -> Any:
        if not self._attached:
            raise QuackDaemonGatewayError("daemon gateway component is not attached")
        return self._dispatcher.dispatch(operation, arguments)


class QuackDaemonTaskSourceProxy(_OwnerProxy):
    def materialize(self, population: Mapping[str, Any], **kwargs: Any) -> Mapping[str, Any]:
        result = self._call("task.materialize", {"population": population, **kwargs})
        return MappingProxyType(dict(result or {}))

    def list_tasks(self, *, limit: int) -> _GatewayTaskPage:
        result = self._call("task.list", {"limit": int(limit)})
        records = result.get("tasks", ()) if isinstance(result, Mapping) else result
        return _GatewayTaskPage(tuple(records or ()))

    def ready_tasks(self, *, limit: int) -> _GatewayTaskPage:
        result = self._call("task.ready", {"limit": int(limit)})
        records = result.get("tasks", ()) if isinstance(result, Mapping) else result
        return _GatewayTaskPage(tuple(records or ()))

    def get(self, task_cid: str) -> _GatewayRecord | None:
        result = self._call("task.get", {"task_cid": task_cid})
        return None if result is None else _GatewayRecord(result)

    def compare_and_set_status(self, task_cid: str, **kwargs: Any) -> _GatewayRecord:
        result = self._call("task.cas_status", {"task_cid": task_cid, **kwargs})
        return _GatewayRecord(result)

    def record_validation_result(self, **kwargs: Any) -> Mapping[str, Any]:
        result = self._call("task.record_validation", kwargs)
        return MappingProxyType(dict(result or {}))


class QuackDaemonCoordinatorProxy(_OwnerProxy):
    def register_task(self, **kwargs: Any) -> Any:
        return self._call("coordination.register_task", kwargs)

    def claim_ready_task(self, **kwargs: Any) -> _GatewayRecord | None:
        result = self._call("coordination.claim_ready", kwargs)
        return None if result is None else _GatewayRecord(result)

    def get_task_claim(self, claim_id: str) -> _GatewayRecord | None:
        result = self._call("coordination.get_claim", {"claim_id": claim_id})
        return None if result is None else _GatewayRecord(result)

    def protect_task_claim(self, claim: Any, **kwargs: Any) -> _GatewayRecord:
        return _GatewayRecord(
            self._call("coordination.protect_claim", {"claim": claim, **kwargs})
        )

    def renew(self, lease: Any, **kwargs: Any) -> _GatewayRecord:
        return _GatewayRecord(
            self._call("coordination.renew_lease", {"lease": lease, **kwargs})
        )

    def prepare_task_completion(self, claim: Any, **kwargs: Any) -> Any:
        return self._call(
            "coordination.prepare_completion", {"claim": claim, **kwargs}
        )

    def get_prepared_task_completion(self, task_cid: str) -> Any:
        return self._call(
            "coordination.get_prepared_completion", {"task_cid": task_cid}
        )

    def complete_task_claim(self, claim: Any, **kwargs: Any) -> Any:
        return self._call("coordination.complete_claim", {"claim": claim, **kwargs})

    def settle_task_claim(self, claim: Any, **kwargs: Any) -> Any:
        return self._call("coordination.settle_claim", {"claim": claim, **kwargs})

    def list_unsettled_task_completions(self, **kwargs: Any) -> Any:
        return self._call("coordination.list_unsettled_completions", kwargs)

    def reconcile_promoted_task_completion(self, task_cid: str, **kwargs: Any) -> Any:
        return self._call(
            "coordination.reconcile_promoted_completion",
            {"task_cid": task_cid, **kwargs},
        )

    def recover_prepared_task_completion(self, task_cid: str, **kwargs: Any) -> Any:
        return self._call(
            "coordination.recover_prepared_completion",
            {"task_cid": task_cid, **kwargs},
        )

    def abort_prepared_task_completion(self, task_cid: str, **kwargs: Any) -> Any:
        return self._call(
            "coordination.abort_prepared_completion",
            {"task_cid": task_cid, **kwargs},
        )

    def expire_task_claim(self, claim: Any, **kwargs: Any) -> Any:
        return self._call("coordination.expire_claim", {"claim": claim, **kwargs})


class QuackDaemonExecutionRepositoryProxy(_OwnerProxy):
    def bind_daemon(self, metadata: Mapping[str, Any]) -> Any:
        return self._call("execution.bind_daemon", {"metadata": metadata})

    def record_event(self, **kwargs: Any) -> Any:
        return self._call("execution.record_event", kwargs)

    def ensure_attempt(self, **kwargs: Any) -> Any:
        return self._call("execution.ensure_attempt", kwargs)

    def get_attempt(self, attempt_id: str) -> Any:
        return self._call("execution.get_attempt", {"attempt_id": attempt_id})

    def list_running_attempts(self, **kwargs: Any) -> Any:
        return self._call("execution.list_running_attempts", kwargs)

    def commit_phase(self, **kwargs: Any) -> Any:
        return self._call("execution.commit_phase", kwargs)

    def commit_reconciled_attempt(self, **kwargs: Any) -> Any:
        return self._call("execution.commit_reconciled_attempt", kwargs)

    def phase_history(self, attempt_id: str) -> Any:
        return self._call("execution.phase_history", {"attempt_id": attempt_id})

    def get_idempotent_result(self, **kwargs: Any) -> Any:
        kind = str(kwargs.get("kind") or "")
        operation = "provider.reserve" if kind == "provider" else "effect.reserve"
        return self._call(operation, kwargs)

    def record_idempotent_result(self, **kwargs: Any) -> Any:
        kind = str(kwargs.get("kind") or "")
        operation = "provider.commit" if kind == "provider" else "effect.commit"
        return self._call(operation, kwargs)

    def reserve_provider(self, **kwargs: Any) -> Any:
        return self._call("provider.reserve", kwargs)

    def commit_provider(self, **kwargs: Any) -> Any:
        return self._call("provider.commit", kwargs)

    def reserve_effect(self, **kwargs: Any) -> Any:
        return self._call("effect.reserve", kwargs)

    def commit_effect(self, **kwargs: Any) -> Any:
        return self._call("effect.commit", kwargs)

    def record_validation(self, **kwargs: Any) -> Any:
        return self._call("validation.record", kwargs)


class QuackDaemonMergeRepositoryProxy(_OwnerProxy):
    def enqueue(self, **kwargs: Any) -> Any:
        return self._call("merge.enqueue", kwargs)

    def observe(self, **kwargs: Any) -> Any:
        return self._call("merge.observe", kwargs)

    def accept(self, **kwargs: Any) -> Any:
        return self._call("merge.accept", kwargs)


class QuackDaemonPlanRepositoryProxy(_OwnerProxy):
    def prepare(self, **kwargs: Any) -> Any:
        return self._call("plan_r2.prepare", kwargs)

    def apply(self, **kwargs: Any) -> Any:
        return self._call("plan_r2.apply", kwargs)

    def observe(self, **kwargs: Any) -> Any:
        return self._call("plan_r2.observe", kwargs)


class QuackDaemonCommandGateway:
    """Composition root for five closed proxies sharing one command owner.

    The proxies are implemented by the admitted transport package.  This
    wrapper merely prevents a daemon from mixing task, coordination, or
    execution components from different shards or generations.
    """

    INTERFACE: ClassVar[str] = QUACK_DAEMON_COMMAND_GATEWAY_INTERFACE

    def __init__(
        self,
        *,
        capability: QuackDaemonGatewayCapability,
        task_source: Any = None,
        coordinator: Any = None,
        execution_repository: Any = None,
        merge_repository: Any = None,
        plan_repository: Any = None,
        owner_dispatcher: QuackDaemonOwnerDispatcher | None = None,
        operational_capability: Mapping[str, Any] | None = None,
        trusted_operational_capability_reviewer_dids: Sequence[str] = (),
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if not isinstance(capability, QuackDaemonGatewayCapability):
            raise QuackDaemonGatewayError("typed gateway capability is required")
        self.capability = capability
        self._clock_ms = clock_ms or (lambda: 1)
        self._operational_capability = (
            None
            if operational_capability is None
            else MappingProxyType(dict(operational_capability))
        )
        self._trusted_operational_capability_reviewer_dids = tuple(
            str(value) for value in trusted_operational_capability_reviewer_dids
        )
        self._owner_dispatcher = owner_dispatcher
        if capability.production_admitted:
            if self._operational_capability is None or owner_dispatcher is None:
                raise QuackDaemonGatewayError(
                    "production gateway requires its signed capability and one owner dispatcher"
                )
            verified = verify_quack_daemon_operational_capability(
                self._operational_capability,
                trusted_reviewer_dids=(
                    self._trusted_operational_capability_reviewer_dids
                ),
                now_ms=int(self._clock_ms()),
            )
            self._assert_operational_capability_matches(verified)
            if (
                type(owner_dispatcher) is not QuackDaemonOwnerDispatcher
                or owner_dispatcher.gateway_binding_cid != capability.content_id
            ):
                raise QuackDaemonGatewayError(
                    "production gateway components require one exact local owner dispatcher"
                )
            supplied = (
                task_source,
                coordinator,
                execution_repository,
                merge_repository,
                plan_repository,
            )
            if any(item is not None for item in supplied):
                raise QuackDaemonGatewayError(
                    "production gateway components are derived from its one owner dispatcher"
                )
            task_source = QuackDaemonTaskSourceProxy(owner_dispatcher)
            coordinator = QuackDaemonCoordinatorProxy(owner_dispatcher)
            execution_repository = QuackDaemonExecutionRepositoryProxy(owner_dispatcher)
            merge_repository = QuackDaemonMergeRepositoryProxy(owner_dispatcher)
            plan_repository = QuackDaemonPlanRepositoryProxy(owner_dispatcher)
        elif owner_dispatcher is not None or operational_capability is not None:
            raise QuackDaemonGatewayError(
                "structural gateway cannot carry an operational dispatcher or capability"
            )
        self.task_source = task_source
        self.coordinator = coordinator
        self.execution_repository = execution_repository
        self.merge_repository = merge_repository
        self.plan_repository = plan_repository
        self._attached = False
        self._validate_components()

    @classmethod
    def from_operational_capability(
        cls,
        operational_capability: Mapping[str, Any],
        *,
        trusted_reviewer_dids: Sequence[str],
        authorization_policy: QuackCommandAuthorizationPolicy,
        authorization_provider: Callable[[Mapping[str, Any]], AuthorizedStateCommand],
        clock_ms: Callable[[], int],
    ) -> QuackDaemonCommandGateway:
        """Verify the external record, then fail until all owner ops are sealed.

        The implementation daemon is a separate process.  An in-process bound
        method is not a transport to the sole Quack owner, so accepting it here
        would make a structurally signed fixture look deployable.  The verified
        record, append/poll transport, and three transaction-safe task reads
        remain useful evidence.  Broad gateway construction is deliberately
        unavailable while any operation has a typed no-go disposition and
        until provider/effect ordering is independently qualified.
        """

        verified = verify_quack_daemon_operational_capability(
            operational_capability,
            trusted_reviewer_dids=trusted_reviewer_dids,
            now_ms=int(clock_ms()),
        )
        del verified, authorization_policy, authorization_provider
        raise QuackDaemonGatewayError(
            "canonical_39_operation_owner_handler_unqualified: the signed "
            "append/poll transport and fail-closed 39-operation handler exist, "
            "but only task.get/task.list/task.ready are admitted owner "
            "transactions and external-effect execution remains unqualified"
        )

    @property
    def attached(self) -> bool:
        return self._attached

    def _assert_operational_capability_matches(
        self, verified: Mapping[str, Any]
    ) -> None:
        comparisons = {
            "board_namespace": self.capability.board_namespace,
            "shard_id": self.capability.shard_id,
            "store_id": self.capability.store_id,
            "control_plane_schema_version": (
                self.capability.control_plane_schema_version
            ),
            "state_schema_revision": self.capability.state_schema_revision,
            "command_endpoint": self.capability.command_endpoint,
            "state_endpoint": self.capability.state_endpoint,
            "owner_principal_did": self.capability.owner_principal_did,
            "owner_generation": self.capability.owner_generation,
            "fence_epoch": self.capability.fence_epoch,
            "authorization_policy_cid": self.capability.authorization_policy_cid,
            "command_fabric_qualification_cid": (
                self.capability.command_fabric_qualification_cid
            ),
            "capability_cid": self.capability.operational_capability_cid,
        }
        mismatched = sorted(
            name for name, expected in comparisons.items() if verified.get(name) != expected
        )
        if mismatched:
            raise QuackDaemonGatewayError(
                "operational capability differs from gateway binding: "
                + ", ".join(mismatched)
            )

    def require_production_admission(self) -> Mapping[str, Any]:
        """Reverify external authority and lifetime before real execution."""

        if (
            not self.capability.production_admitted
            or self._operational_capability is None
            or self._owner_dispatcher is None
        ):
            raise QuackDaemonGatewayError(
                "Quack daemon command gateway is not production-admitted: "
                "task.get/task.list/task.ready have owner transactions, but "
                "provider/effect reservation-before-launch and the remaining "
                "typed operations are unqualified"
            )
        verified = verify_quack_daemon_operational_capability(
            self._operational_capability,
            trusted_reviewer_dids=self._trusted_operational_capability_reviewer_dids,
            now_ms=int(self._clock_ms()),
        )
        self._assert_operational_capability_matches(verified)
        return verified

    def _validate_components(self) -> None:
        binding = self.capability.content_id
        for name, component, required_methods in (
            ("task_source", self.task_source, _TASK_COMPONENT_METHODS),
            ("coordinator", self.coordinator, _COORDINATION_COMPONENT_METHODS),
            (
                "execution_repository",
                self.execution_repository,
                _EXECUTION_COMPONENT_METHODS,
            ),
            ("merge_repository", self.merge_repository, _MERGE_COMPONENT_METHODS),
            ("plan_repository", self.plan_repository, _PLAN_COMPONENT_METHODS),
        ):
            if component is None:
                raise QuackDaemonGatewayError(f"{name} is required")
            if (
                getattr(component, "GATEWAY_COMPONENT_INTERFACE", "")
                != QUACK_DAEMON_GATEWAY_COMPONENT_INTERFACE
            ):
                raise QuackDaemonGatewayError(f"{name} is not a closed command-gateway component")
            if str(getattr(component, "gateway_binding_cid", "") or "") != binding:
                raise QuackDaemonGatewayError(f"{name} is bound to a different gateway capability")
            missing_methods = sorted(
                method
                for method in required_methods
                if not callable(getattr(component, method, None))
            )
            if missing_methods:
                raise QuackDaemonGatewayError(
                    f"{name} lacks closed gateway methods: " + ", ".join(missing_methods)
                )
            for forbidden in (
                "database_path",
                "connection",
                "execute",
                "execute_sql",
                "transaction",
            ):
                if hasattr(component, forbidden):
                    raise QuackDaemonGatewayError(
                        f"{name} exposes forbidden {forbidden!r} authority surface"
                    )
        # A concrete Plan-R2 component may join this structural composition
        # only through the narrow owner gateway created by a command fabric
        # holding an independently signed PlanR2OperationalCapability@1.  This
        # does not promote the other daemon operations or turn this structural
        # capability into a production claim.
        if (
            getattr(self.plan_repository, "INTERFACE", "")
            == "AuthorizedPlanR2TransitionRepository@1"
        ):
            owner_gateway = getattr(self.plan_repository, "owner_gateway", None)
            production_cid = str(getattr(owner_gateway, "production_capability_cid", "") or "")
            command_qualification = str(
                getattr(
                    owner_gateway,
                    "command_fabric_qualification_cid",
                    "",
                )
                or ""
            )
            if (
                type(owner_gateway) is not QuackPlanR2OwnerGateway
                or getattr(owner_gateway, "INTERFACE", "")
                != "AuthorizedStateCommandPlanR2OwnerGateway@1"
                or not production_cid.startswith("sha256:")
                or len(production_cid) != 71
                or command_qualification != self.capability.command_fabric_qualification_cid
            ):
                raise QuackDaemonGatewayError(
                    "Plan-R2 component lacks its explicit signed production "
                    "owner-dispatch capability"
                )
        if self.capability.production_admitted:
            dispatcher = self._owner_dispatcher
            expected_types = (
                QuackDaemonTaskSourceProxy,
                QuackDaemonCoordinatorProxy,
                QuackDaemonExecutionRepositoryProxy,
                QuackDaemonMergeRepositoryProxy,
                QuackDaemonPlanRepositoryProxy,
            )
            for component, expected_type in zip(
                (
                    self.task_source,
                    self.coordinator,
                    self.execution_repository,
                    self.merge_repository,
                    self.plan_repository,
                ),
                expected_types,
                strict=True,
            ):
                if type(component) is not expected_type or component._dispatcher is not dispatcher:
                    raise QuackDaemonGatewayError(
                        "production gateway proxies do not share one exact owner dispatcher"
                    )

    def attach(self) -> None:
        if self._attached:
            return
        if self.capability.production_admitted:
            self.require_production_admission()
        attached: list[Any] = []
        try:
            for component in (
                self.task_source,
                self.coordinator,
                self.execution_repository,
                self.merge_repository,
                self.plan_repository,
            ):
                attach = getattr(component, "attach", None)
                if not callable(attach):
                    raise QuackDaemonGatewayError("gateway components must implement attach()")
                attach()
                attached.append(component)
        except Exception:
            for component in reversed(attached):
                close = getattr(component, "close", None)
                if callable(close):
                    close()
            raise
        self._attached = True

    def close(self) -> None:
        errors: list[BaseException] = []
        for component in (
            self.plan_repository,
            self.merge_repository,
            self.execution_repository,
            self.coordinator,
            self.task_source,
        ):
            close = getattr(component, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as exc:
                    errors.append(exc)
        self._attached = False
        if errors:
            raise QuackDaemonGatewayError(
                f"gateway component failed close: {type(errors[0]).__name__}"
            ) from errors[0]

    def evidence(self) -> Mapping[str, Any]:
        production = bool(self.capability.production_admitted)
        return MappingProxyType(
            {
                **self.capability.to_dict(),
                "gateway_binding_cid": self.capability.content_id,
                "attached": self.attached,
                "remote_operation_intent_transport_implemented": True,
                "remote_operation_intent_transport_admitted": production,
                "canonical_39_operation_owner_handler_admitted": production,
                "canonical_owner_handler_qualification_status": (
                    QUACK_DAEMON_HANDLER_QUALIFICATION_STATUS
                ),
                "owner_transaction_admitted_operations": sorted(
                    _OWNER_TRANSACTION_ADMITTED_OPERATIONS
                ),
                "typed_no_go_operation_count": len(
                    _OWNER_TRANSACTION_NO_GO_REASONS
                ),
                "production_blockers": (
                    []
                    if production
                    else [
                        "canonical_39_operation_owner_handler_unqualified",
                    ]
                ),
            }
        )


def require_quack_daemon_command_gateway(
    value: Any,
    *,
    expected_command_endpoint: str,
) -> QuackDaemonCommandGateway:
    """Validate the exact gateway and endpoint before any daemon file access."""

    if not isinstance(value, QuackDaemonCommandGateway):
        raise QuackDaemonGatewayError(
            "quack authority requires QuackDaemonCommandGateway@1; "
            "legacy remote SQL and local execution/coordination sidecars are forbidden"
        )
    value._validate_components()
    expected = _loopback_quack_uri(expected_command_endpoint, "expected_command_endpoint")
    if value.capability.command_endpoint != expected:
        raise QuackDaemonGatewayError(
            "daemon Quack endpoint does not match the admitted command endpoint"
        )
    return value


__all__ = [
    "QUACK_DAEMON_COMMAND_GATEWAY_INTERFACE",
    "QUACK_DAEMON_COMMAND_GATEWAY_SCHEMA",
    "QUACK_DAEMON_GATEWAY_COMPONENT_INTERFACE",
    "QUACK_DAEMON_CANONICAL_HANDLER_SCHEMA",
    "QUACK_DAEMON_HANDLER_QUALIFICATION_STATUS",
    "QUACK_DAEMON_OPERATION_DISPOSITION_SCHEMA",
    "QUACK_DAEMON_OPERATIONAL_CAPABILITY_INTERFACE",
    "QUACK_DAEMON_OPERATIONAL_CAPABILITY_SCHEMA",
    "QUACK_DAEMON_OPERATION_INTENT_INTERFACE",
    "QUACK_DAEMON_OPERATION_INTENT_SCHEMA",
    "QUACK_DAEMON_OWNER_DISPATCHER_INTERFACE",
    "REQUIRED_QUACK_DAEMON_OPERATIONS",
    "QuackDaemonCommandGateway",
    "QuackDaemonCanonicalOwnerOperationHandler",
    "QuackDaemonCoordinatorProxy",
    "QuackDaemonExecutionRepositoryProxy",
    "QuackDaemonGatewayCapability",
    "QuackDaemonGatewayError",
    "QuackDaemonMergeRepositoryProxy",
    "QuackDaemonOwnerDispatcher",
    "QuackDaemonOwnerOperationNoGo",
    "QuackDaemonPlanRepositoryProxy",
    "QuackDaemonRemoteCommandTransport",
    "QuackDaemonTaskSourceProxy",
    "quack_daemon_operation_intent",
    "quack_daemon_operation_intent_from_envelope",
    "quack_daemon_operation_command_vocabulary",
    "quack_daemon_owner_operation_dispositions",
    "quack_daemon_operational_capability_signing_payload",
    "quack_daemon_state_command_parameters",
    "require_quack_daemon_command_gateway",
    "seal_quack_daemon_operational_capability",
    "verify_quack_daemon_operation_submission",
    "verify_quack_daemon_operational_capability",
]
