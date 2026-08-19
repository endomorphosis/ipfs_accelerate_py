"""Closed EAAEF bootstrap-daemon capability and gateway contract.

This is deliberately narrower than ``QuackDaemonCommandGateway@1``.  It
describes only the operations needed by the EAAEF-001..EAAEF-009 bootstrap
trace and does not reinterpret, replace, or weaken the generic 39-operation
daemon protocol.  Offline task materialization, broad task enumeration, host
merge authority, and the independently promoted Plan-R2 dispatcher are not
members of this capability.

The contract is source evidence, not production authority.  Two read-only
task operations already have canonical owner-transaction handlers; the other
29 operations remain typed no-go until they have reviewed borrowed-
transaction adapters over the sole Quack owner's active transaction.  No
component below exposes transport, a database path, Portal, SQL, or an
injectable operation callback.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .control_plane_contracts import content_identity
from .quack_daemon_gateway import (
    QUACK_DAEMON_COMMAND_GATEWAY_INTERFACE,
    REQUIRED_QUACK_DAEMON_OPERATIONS,
    QuackDaemonGatewayCapability,
    quack_daemon_operation_command_vocabulary,
    quack_daemon_owner_operation_dispositions,
)

EAAEF_BOOTSTRAP_DAEMON_CAPABILITY_INTERFACE: Final = "EAAEFBootstrapDaemonCapability@1"
EAAEF_BOOTSTRAP_DAEMON_CAPABILITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-bootstrap-daemon-capability@1"
)
EAAEF_BOOTSTRAP_DAEMON_GATEWAY_INTERFACE: Final = "EAAEFBootstrapDaemonGateway@1"
EAAEF_BOOTSTRAP_DAEMON_GATEWAY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-bootstrap-daemon-gateway@1"
)
EAAEF_BOOTSTRAP_DAEMON_COMPONENT_INTERFACE: Final = "EAAEFBootstrapDaemonComponent@1"
EAAEF_BOOTSTRAP_DAEMON_DISPOSITION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "eaaef-bootstrap-daemon-operation-disposition@1"
)
EAAEF_BOOTSTRAP_DAEMON_QUALIFICATION_STATUS: Final = (
    "29_borrowed_transaction_handlers_missing_fail_closed"
)
EAAEF_BOOTSTRAP_DAEMON_PRODUCTION_BLOCKER: Final = (
    "eaaef_bootstrap_29_borrowed_transaction_handlers_unqualified"
)

EAAEF_BOOTSTRAP_TASK_OPERATIONS: Final = frozenset(
    {
        "task.ready",
        "task.get",
        "task.cas_status",
        "task.record_validation",
    }
)
EAAEF_BOOTSTRAP_COORDINATION_OPERATIONS: Final = frozenset(
    {
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
    }
)
EAAEF_BOOTSTRAP_EXECUTION_OPERATIONS: Final = frozenset(
    {
        "execution.bind_daemon",
        "execution.record_event",
        "execution.ensure_attempt",
        "execution.get_attempt",
        "execution.list_running_attempts",
        "execution.commit_phase",
        "execution.commit_reconciled_attempt",
        "execution.phase_history",
    }
)
EAAEF_BOOTSTRAP_PROVIDER_OPERATIONS: Final = frozenset(
    {"provider.reserve", "provider.commit"}
)
EAAEF_BOOTSTRAP_EFFECT_OPERATIONS: Final = frozenset(
    {"effect.reserve", "effect.commit"}
)
EAAEF_BOOTSTRAP_VALIDATION_OPERATIONS: Final = frozenset({"validation.record"})

EAAEF_BOOTSTRAP_DAEMON_COMPONENT_OPERATIONS: Final[Mapping[str, frozenset[str]]] = (
    MappingProxyType(
        {
            "task": EAAEF_BOOTSTRAP_TASK_OPERATIONS,
            "coordination": EAAEF_BOOTSTRAP_COORDINATION_OPERATIONS,
            "execution": EAAEF_BOOTSTRAP_EXECUTION_OPERATIONS,
            "provider": EAAEF_BOOTSTRAP_PROVIDER_OPERATIONS,
            "effect": EAAEF_BOOTSTRAP_EFFECT_OPERATIONS,
            "validation": EAAEF_BOOTSTRAP_VALIDATION_OPERATIONS,
        }
    )
)
EAAEF_BOOTSTRAP_DAEMON_OPERATIONS: Final = frozenset().union(
    *EAAEF_BOOTSTRAP_DAEMON_COMPONENT_OPERATIONS.values()
)
EAAEF_BOOTSTRAP_DAEMON_EXCLUDED_OPERATIONS: Final = frozenset(
    {
        "task.materialize",
        "task.list",
        "merge.enqueue",
        "merge.observe",
        "merge.accept",
        "plan_r2.prepare",
        "plan_r2.apply",
        "plan_r2.observe",
    }
)
EAAEF_BOOTSTRAP_DAEMON_ADMITTED_OPERATIONS: Final = frozenset(
    {"task.ready", "task.get"}
)
EAAEF_BOOTSTRAP_DAEMON_MISSING_OPERATIONS: Final = (
    EAAEF_BOOTSTRAP_DAEMON_OPERATIONS - EAAEF_BOOTSTRAP_DAEMON_ADMITTED_OPERATIONS
)

if (
    len(EAAEF_BOOTSTRAP_DAEMON_OPERATIONS) != 31
    or len(EAAEF_BOOTSTRAP_DAEMON_EXCLUDED_OPERATIONS) != 8
    or len(EAAEF_BOOTSTRAP_DAEMON_MISSING_OPERATIONS) != 29
    or EAAEF_BOOTSTRAP_DAEMON_OPERATIONS & EAAEF_BOOTSTRAP_DAEMON_EXCLUDED_OPERATIONS
    or EAAEF_BOOTSTRAP_DAEMON_OPERATIONS | EAAEF_BOOTSTRAP_DAEMON_EXCLUDED_OPERATIONS
    != REQUIRED_QUACK_DAEMON_OPERATIONS
):
    raise RuntimeError("EAAEF bootstrap daemon vocabulary differs from generic v1")

_GENERIC_COMMAND_KINDS: Final = quack_daemon_operation_command_vocabulary()
_GENERIC_DISPOSITIONS: Final = quack_daemon_owner_operation_dispositions()
if frozenset(_GENERIC_COMMAND_KINDS) != REQUIRED_QUACK_DAEMON_OPERATIONS:
    raise RuntimeError("generic daemon command vocabulary is not the frozen v1 set")
if frozenset(_GENERIC_DISPOSITIONS) != REQUIRED_QUACK_DAEMON_OPERATIONS:
    raise RuntimeError("generic daemon disposition vocabulary is not the frozen v1 set")
if (
    frozenset(
        operation
        for operation in EAAEF_BOOTSTRAP_DAEMON_OPERATIONS
        if _GENERIC_DISPOSITIONS[operation]["disposition"]
        == "admitted_owner_transaction"
    )
    != EAAEF_BOOTSTRAP_DAEMON_ADMITTED_OPERATIONS
):
    raise RuntimeError(
        "EAAEF bootstrap admission differs from canonical owner evidence"
    )

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,511}\Z")


class EAAEFBootstrapDaemonGatewayError(RuntimeError):
    """The bootstrap gateway is malformed, cross-bound, or unqualified."""


class EAAEFBootstrapDaemonOperationNoGo(EAAEFBootstrapDaemonGatewayError):
    """A reviewed bootstrap operation still lacks its owner adapter."""

    def __init__(self, operation: str, reason_code: str) -> None:
        self.operation = operation
        self.reason_code = reason_code
        super().__init__(
            "eaaef_bootstrap_daemon_operation_no_go:"
            f"operation={operation};reason_code={reason_code}"
        )


def _safe_id(value: Any, noun: str) -> str:
    text = str(value or "")
    if not _SAFE_ID.fullmatch(text):
        raise EAAEFBootstrapDaemonGatewayError(f"{noun} is not a bounded identifier")
    return text


def _sha(value: Any, noun: str) -> str:
    text = str(value or "")
    if not _SHA256.fullmatch(text):
        raise EAAEFBootstrapDaemonGatewayError(f"{noun} is not a full sha256 identity")
    return text


def _store_id(value: Any) -> str:
    text = _safe_id(value, "store_id")
    if "/" in text or "\\" in text or text.lower().endswith((".duckdb", ".db")):
        raise EAAEFBootstrapDaemonGatewayError(
            "store_id must be an opaque owner identity, not a database path"
        )
    return text


def eaaef_bootstrap_daemon_operation_dispositions() -> Mapping[str, Mapping[str, Any]]:
    """Return all and only the reviewed 31 bootstrap operation dispositions."""

    return MappingProxyType(
        {
            operation: MappingProxyType(
                {
                    "schema": EAAEF_BOOTSTRAP_DAEMON_DISPOSITION_SCHEMA,
                    "operation": operation,
                    "command_kind": _GENERIC_COMMAND_KINDS[operation],
                    "disposition": _GENERIC_DISPOSITIONS[operation]["disposition"],
                    "reason_code": _GENERIC_DISPOSITIONS[operation]["reason_code"],
                    "borrowed_transaction_required": (
                        operation in EAAEF_BOOTSTRAP_DAEMON_MISSING_OPERATIONS
                    ),
                }
            )
            for operation in sorted(EAAEF_BOOTSTRAP_DAEMON_OPERATIONS)
        }
    )


@dataclass(frozen=True, slots=True)
class EAAEFBootstrapDaemonCapability:
    """Source-bound structural capability for the bounded bootstrap trace."""

    board_namespace: str
    shard_id: str
    store_id: str
    owner_principal_did: str
    owner_generation: int
    fence_epoch: int
    authorization_policy_cid: str
    command_fabric_qualification_cid: str
    operations: frozenset[str] = EAAEF_BOOTSTRAP_DAEMON_OPERATIONS
    excluded_operations: frozenset[str] = EAAEF_BOOTSTRAP_DAEMON_EXCLUDED_OPERATIONS
    owner_transaction_admitted_operations: frozenset[str] = (
        EAAEF_BOOTSTRAP_DAEMON_ADMITTED_OPERATIONS
    )
    missing_borrowed_transaction_operations: frozenset[str] = (
        EAAEF_BOOTSTRAP_DAEMON_MISSING_OPERATIONS
    )
    qualification_status: str = EAAEF_BOOTSTRAP_DAEMON_QUALIFICATION_STATUS
    production_admitted: bool = False
    portal_fallback: bool = False
    direct_database_open: bool = False
    local_sidecar_writes: bool = False
    arbitrary_sql_enabled: bool = False

    SCHEMA: ClassVar[str] = EAAEF_BOOTSTRAP_DAEMON_CAPABILITY_SCHEMA
    INTERFACE: ClassVar[str] = EAAEF_BOOTSTRAP_DAEMON_CAPABILITY_INTERFACE

    def __post_init__(self) -> None:
        for field_name in ("board_namespace", "shard_id"):
            object.__setattr__(
                self, field_name, _safe_id(getattr(self, field_name), field_name)
            )
        object.__setattr__(self, "store_id", _store_id(self.store_id))
        owner = str(self.owner_principal_did or "")
        if not owner.startswith("did:key:z") or len(owner) > 512:
            raise EAAEFBootstrapDaemonGatewayError(
                "owner_principal_did must be an Ed25519 did:key"
            )
        object.__setattr__(self, "owner_principal_did", owner)
        for field_name in ("owner_generation", "fence_epoch"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise EAAEFBootstrapDaemonGatewayError(
                    f"{field_name} must be a positive integer"
                )
        for field_name in (
            "authorization_policy_cid",
            "command_fabric_qualification_cid",
        ):
            object.__setattr__(
                self, field_name, _sha(getattr(self, field_name), field_name)
            )
        exact_sets = {
            "operations": EAAEF_BOOTSTRAP_DAEMON_OPERATIONS,
            "excluded_operations": EAAEF_BOOTSTRAP_DAEMON_EXCLUDED_OPERATIONS,
            "owner_transaction_admitted_operations": (
                EAAEF_BOOTSTRAP_DAEMON_ADMITTED_OPERATIONS
            ),
            "missing_borrowed_transaction_operations": (
                EAAEF_BOOTSTRAP_DAEMON_MISSING_OPERATIONS
            ),
        }
        for field_name, expected in exact_sets.items():
            supplied = frozenset(str(item) for item in getattr(self, field_name))
            if supplied != expected:
                missing = sorted(expected - supplied)
                extra = sorted(supplied - expected)
                detail = []
                if missing:
                    detail.append("missing=" + ",".join(missing))
                if extra:
                    detail.append("extra=" + ",".join(extra))
                raise EAAEFBootstrapDaemonGatewayError(
                    f"bootstrap {field_name} is not exact: " + ";".join(detail)
                )
            object.__setattr__(self, field_name, supplied)
        if self.qualification_status != EAAEF_BOOTSTRAP_DAEMON_QUALIFICATION_STATUS:
            raise EAAEFBootstrapDaemonGatewayError(
                "bootstrap qualification status is not the closed v1 value"
            )
        unsafe = {
            "production_admitted": self.production_admitted,
            "portal_fallback": self.portal_fallback,
            "direct_database_open": self.direct_database_open,
            "local_sidecar_writes": self.local_sidecar_writes,
            "arbitrary_sql_enabled": self.arbitrary_sql_enabled,
        }
        enabled = sorted(name for name, value in unsafe.items() if value is not False)
        if enabled:
            raise EAAEFBootstrapDaemonGatewayError(
                "bootstrap capability enables unavailable authority: "
                + ",".join(enabled)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "generic_daemon_interface": QUACK_DAEMON_COMMAND_GATEWAY_INTERFACE,
            "board_namespace": self.board_namespace,
            "shard_id": self.shard_id,
            "store_id": self.store_id,
            "owner_principal_did": self.owner_principal_did,
            "owner_generation": self.owner_generation,
            "fence_epoch": self.fence_epoch,
            "authorization_policy_cid": self.authorization_policy_cid,
            "command_fabric_qualification_cid": (self.command_fabric_qualification_cid),
            "operations": sorted(self.operations),
            "excluded_operations": sorted(self.excluded_operations),
            "component_operations": {
                name: sorted(operations)
                for name, operations in EAAEF_BOOTSTRAP_DAEMON_COMPONENT_OPERATIONS.items()
            },
            "owner_transaction_admitted_operations": sorted(
                self.owner_transaction_admitted_operations
            ),
            "missing_borrowed_transaction_operations": sorted(
                self.missing_borrowed_transaction_operations
            ),
            "qualification_status": self.qualification_status,
            "production_admitted": self.production_admitted,
            "portal_fallback": self.portal_fallback,
            "direct_database_open": self.direct_database_open,
            "local_sidecar_writes": self.local_sidecar_writes,
            "arbitrary_sql_enabled": self.arbitrary_sql_enabled,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())


class EAAEFBootstrapDaemonComponent:
    """Non-executing description of one closed bootstrap operation family."""

    __slots__ = ("component", "gateway_binding_cid", "operations")
    INTERFACE: ClassVar[str] = EAAEF_BOOTSTRAP_DAEMON_COMPONENT_INTERFACE

    def __init__(
        self,
        *,
        component: str,
        gateway_binding_cid: str,
        _construction_token: object,
    ) -> None:
        if _construction_token is not _COMPONENT_CONSTRUCTION_TOKEN:
            raise EAAEFBootstrapDaemonGatewayError(
                "bootstrap components are constructed only by their gateway"
            )
        if component not in EAAEF_BOOTSTRAP_DAEMON_COMPONENT_OPERATIONS:
            raise EAAEFBootstrapDaemonGatewayError(
                "bootstrap component family is unsupported"
            )
        self.component = component
        # ``content_identity`` uses the repository's canonical CID encoding,
        # which is deliberately distinct from the sha256 receipt identities
        # bound by the capability itself.
        self.gateway_binding_cid = _safe_id(gateway_binding_cid, "gateway_binding_cid")
        self.operations = EAAEF_BOOTSTRAP_DAEMON_COMPONENT_OPERATIONS[component]

    def disposition(self, operation: str) -> Mapping[str, Any]:
        name = str(operation or "")
        if name not in self.operations:
            raise EAAEFBootstrapDaemonGatewayError(
                f"operation is outside bootstrap {self.component} component"
            )
        return eaaef_bootstrap_daemon_operation_dispositions()[name]


_COMPONENT_CONSTRUCTION_TOKEN: Final = object()


class EAAEFBootstrapDaemonGateway:
    """Non-injectable composition root for the six bootstrap components."""

    __slots__ = (
        "capability",
        "task",
        "coordination",
        "execution",
        "provider",
        "effect",
        "validation",
    )
    SCHEMA: ClassVar[str] = EAAEF_BOOTSTRAP_DAEMON_GATEWAY_SCHEMA
    INTERFACE: ClassVar[str] = EAAEF_BOOTSTRAP_DAEMON_GATEWAY_INTERFACE

    def __init__(self, *, capability: EAAEFBootstrapDaemonCapability) -> None:
        if type(capability) is not EAAEFBootstrapDaemonCapability:
            if isinstance(capability, QuackDaemonGatewayCapability):
                raise EAAEFBootstrapDaemonGatewayError(
                    "generic 39-operation capability cannot substitute for the "
                    "EAAEF bootstrap capability"
                )
            raise EAAEFBootstrapDaemonGatewayError(
                "exact EAAEFBootstrapDaemonCapability@1 is required"
            )
        self.capability = capability
        binding = capability.content_id
        for name in EAAEF_BOOTSTRAP_DAEMON_COMPONENT_OPERATIONS:
            setattr(
                self,
                name,
                EAAEFBootstrapDaemonComponent(
                    component=name,
                    gateway_binding_cid=binding,
                    _construction_token=_COMPONENT_CONSTRUCTION_TOKEN,
                ),
            )

    def disposition(self, operation: str) -> Mapping[str, Any]:
        name = str(operation or "")
        if name in EAAEF_BOOTSTRAP_DAEMON_EXCLUDED_OPERATIONS:
            raise EAAEFBootstrapDaemonGatewayError(
                "operation is explicitly excluded from the bootstrap capability"
            )
        try:
            return eaaef_bootstrap_daemon_operation_dispositions()[name]
        except KeyError as exc:
            raise EAAEFBootstrapDaemonGatewayError(
                "operation is outside the closed 31-operation bootstrap vocabulary"
            ) from exc

    def require_operation(self, operation: str) -> None:
        record = self.disposition(operation)
        if record["disposition"] != "admitted_owner_transaction":
            raise EAAEFBootstrapDaemonOperationNoGo(
                str(record["operation"]), str(record["reason_code"])
            )

    def require_production_admission(self) -> None:
        raise EAAEFBootstrapDaemonGatewayError(
            EAAEF_BOOTSTRAP_DAEMON_PRODUCTION_BLOCKER
        )

    def attach(self) -> None:
        """Never attach while the fixed 29-operation qualification gap exists."""

        self.require_production_admission()

    def evidence(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": self.SCHEMA,
                "interface": self.INTERFACE,
                "capability_cid": self.capability.content_id,
                "operation_count": len(self.capability.operations),
                "excluded_operation_count": len(self.capability.excluded_operations),
                "component_operations": {
                    name: sorted(operations)
                    for name, operations in (
                        EAAEF_BOOTSTRAP_DAEMON_COMPONENT_OPERATIONS.items()
                    )
                },
                "owner_transaction_admitted_operations": sorted(
                    self.capability.owner_transaction_admitted_operations
                ),
                "missing_borrowed_transaction_operations": sorted(
                    self.capability.missing_borrowed_transaction_operations
                ),
                "production_admitted": False,
                "production_blockers": [EAAEF_BOOTSTRAP_DAEMON_PRODUCTION_BLOCKER],
                "portal_fallback": False,
                "direct_database_open": False,
                "local_sidecar_writes": False,
                "arbitrary_sql_enabled": False,
            }
        )


def require_eaaef_bootstrap_daemon_gateway(
    value: object,
    *,
    expected_capability_cid: str,
) -> EAAEFBootstrapDaemonGateway:
    """Reject generic/cross-bound gateways without touching any state path."""

    if type(value) is not EAAEFBootstrapDaemonGateway:
        raise EAAEFBootstrapDaemonGatewayError(
            "exact EAAEFBootstrapDaemonGateway@1 is required; generic, Portal, "
            "and direct-file fallbacks are forbidden"
        )
    expected = _safe_id(expected_capability_cid, "expected_capability_cid")
    if value.capability.content_id != expected:
        raise EAAEFBootstrapDaemonGatewayError(
            "bootstrap gateway is bound to another capability"
        )
    return value


__all__ = (
    "EAAEF_BOOTSTRAP_COORDINATION_OPERATIONS",
    "EAAEF_BOOTSTRAP_DAEMON_ADMITTED_OPERATIONS",
    "EAAEF_BOOTSTRAP_DAEMON_CAPABILITY_INTERFACE",
    "EAAEF_BOOTSTRAP_DAEMON_CAPABILITY_SCHEMA",
    "EAAEF_BOOTSTRAP_DAEMON_COMPONENT_INTERFACE",
    "EAAEF_BOOTSTRAP_DAEMON_COMPONENT_OPERATIONS",
    "EAAEF_BOOTSTRAP_DAEMON_EXCLUDED_OPERATIONS",
    "EAAEF_BOOTSTRAP_DAEMON_GATEWAY_INTERFACE",
    "EAAEF_BOOTSTRAP_DAEMON_GATEWAY_SCHEMA",
    "EAAEF_BOOTSTRAP_DAEMON_MISSING_OPERATIONS",
    "EAAEF_BOOTSTRAP_DAEMON_OPERATIONS",
    "EAAEF_BOOTSTRAP_DAEMON_PRODUCTION_BLOCKER",
    "EAAEF_BOOTSTRAP_DAEMON_QUALIFICATION_STATUS",
    "EAAEF_BOOTSTRAP_EFFECT_OPERATIONS",
    "EAAEF_BOOTSTRAP_EXECUTION_OPERATIONS",
    "EAAEF_BOOTSTRAP_PROVIDER_OPERATIONS",
    "EAAEF_BOOTSTRAP_TASK_OPERATIONS",
    "EAAEF_BOOTSTRAP_VALIDATION_OPERATIONS",
    "EAAEFBootstrapDaemonCapability",
    "EAAEFBootstrapDaemonComponent",
    "EAAEFBootstrapDaemonGateway",
    "EAAEFBootstrapDaemonGatewayError",
    "EAAEFBootstrapDaemonOperationNoGo",
    "eaaef_bootstrap_daemon_operation_dispositions",
    "require_eaaef_bootstrap_daemon_gateway",
)
