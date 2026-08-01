"""Extract native agent-supervisor control and goal/task contracts.

Interface: ``SupervisorContractCatalog@1``

This module is the SCA-G174 authority for joining SwissKnife console
capabilities to the closed native ``agent_supervisor_*`` control surface.

Normative rules:

* Every SwissKnife console capability either maps to **one exact** native
  operation identity (tool name, request/result schema IDs, dispatcher ID,
  function identity, policy flags, and effect authority) or is **explicitly
  refuted** with a typed reason.
* Generic proxy selection is rejected.  That includes the multi-operation
  ``agent_supervisor_control`` adapter, usage tooling, workflow/data/storage
  backends that only share a UI label, and bare ``mcp++/`` method strings.
* Goal completion requires current child, evidence, analyzer-health, and
  exhaustion-quorum closure through the objective completion gate.
* Mutation paths are policy dominated: dry-run/preview, authorization,
  permit, and receipt requirements dominate every mutating effect.

The extractor prefers live native catalog/tool publication and optional
SwissKnife gateway/schema source text.  It never treats display names or
backend ownership alone as native-operation reachability.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, Iterable, Mapping, Sequence

from ..control.control_contracts import (
    MUTATION_OPERATIONS,
    OPERATION_CATALOG_V2,
    PROPOSAL_OPERATIONS,
    READ_OPERATIONS,
    ControlOperationDescriptor,
    Operation,
    OperationAuthority,
    OperationCatalog,
    get_operation_catalog,
)
from ..control.control_plane import (
    DIRECT_CONTROL_SERVICE_DISPATCHER_ID,
    control_operation_behavior_id,
)
from ..objectives.goal_completion import evaluate_completion_gate
from .content_identity_bridge import identify_strict_artifact


SUPERVISOR_CONTRACT_EXTRACTOR_INTERFACE: Final = "SupervisorContractExtractor@1"
SUPERVISOR_CONTRACT_CATALOG_INTERFACE: Final = "SupervisorContractCatalog@1"
SUPERVISOR_CONTRACT_CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/supervisor-contract-catalog@1"
)
NATIVE_OPERATION_IDENTITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/native-operation-identity@1"
)
CAPABILITY_MAPPING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/supervisor-capability-mapping@1"
)
GOAL_COMPLETION_CONTRACT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/goal-completion-contract@1"
)
MUTATION_POLICY_CONTRACT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mutation-policy-contract@1"
)
SUPERVISOR_CONTRACT_EXTRACTOR_VERSION: Final = "1"

NATIVE_TOOL_PREFIX: Final = "agent_supervisor_"
NATIVE_TOOLS_MODULE: Final = (
    "ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools"
    ".native_agent_supervisor_tools"
)
NATIVE_EXECUTOR_IDENTITY: Final = (
    f"{NATIVE_TOOLS_MODULE}:execute_agent_supervisor_operation"
)
GENERIC_CONTROL_ADAPTER_TOOL: Final = "agent_supervisor_control"
GENERIC_USAGE_TOOL: Final = "agent_supervisor_usage"

GOAL_COMPLETION_EVALUATOR_IDENTITY: Final = (
    "ipfs_accelerate_py.agent_supervisor.objectives.goal_completion"
    ":evaluate_completion_gate"
)

# Mandatory closure members for GoalTaskClosure (plan claim family).
REQUIRED_COMPLETION_CLOSURE: Final[tuple[str, ...]] = (
    "child_goals",
    "required_validations",
    "analyzer_health",
    "exhaustion_quorum",
)

# Policy path that must dominate every mutation-capable effect.
MUTATION_POLICY_PATH: Final[tuple[str, ...]] = (
    "dry_run_or_preview",
    "authorization",
    "execution_permit",
    "backend_dispatch",
    "audit_receipt",
)

SWISSKNIFE_CONSOLE_SCHEMA_REL: Final = (
    "contracts/agent-supervisor-console.schema.json"
)
SWISSKNIFE_CONSOLE_GATEWAY_REL: Final = (
    "src/services/mcp/agent-supervisor-console-gateway.ts"
)

# Exact capability → native operation.  One capability, one Operation value.
# Capabilities that only proxy datasets/kit/mcp++ surfaces are refuted below.
_EXACT_CAPABILITY_OPERATIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "supervisor.health.read": Operation.HEALTH.value,
        "supervisor.queue.read": Operation.TASKS.value,
        "supervisor.goals.read": Operation.GOALS.value,
        "supervisor.subgoals.read": Operation.GOALS.value,
        "supervisor.logs.read": Operation.EVENTS.value,
        "supervisor.receipts.read": Operation.RECEIPTS.value,
        # Prompt steering is a governed mutation of objective/workflow state.
        "supervisor.prompt-steering.request": Operation.OBJECTIVE_REFINE.value,
    }
)

# Task-control console actions that resolve to one native lifecycle operation.
_TASK_CONTROL_ACTION_OPERATIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "pause": Operation.PAUSE.value,
        "resume": Operation.RESUME.value,
        "retry": Operation.RETRY.value,
        "cancel": Operation.CANCEL.value,
    }
)

# Task-control actions that are not native agent_supervisor_* control ops.
_TASK_CONTROL_REFUTED_ACTIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "claim": "scheduler_lease_action_not_native_control_op",
        "release": "scheduler_lease_action_not_native_control_op",
    }
)

# Capability IDs that are intentionally not native supervisor operations.
_REFUTED_CAPABILITIES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "supervisor.taskboard.links.read": "datasets_search_proxy",
        "supervisor.run-history.search": "datasets_search_proxy",
        "supervisor.policy.assist": "datasets_search_proxy",
        "supervisor.semantic-goal.assist": "datasets_search_proxy",
        "supervisor.receipts.persist": "kit_storage_proxy",
        "supervisor.content.retrieve": "kit_storage_proxy",
        "supervisor.event-dag.checkpoint": "kit_storage_proxy",
        "supervisor.profile-g.read": "generic_mcp_plus_plus_proxy",
        "supervisor.schedule.frontier.read": "generic_mcp_plus_plus_proxy",
        "supervisor.neighborhood.read": "generic_mcp_plus_plus_proxy",
        "supervisor.schedule.claims.read": "generic_mcp_plus_plus_proxy",
        "supervisor.risk.read": "generic_mcp_plus_plus_proxy",
        "supervisor.goal.decompose": "generic_mcp_plus_plus_proxy",
        "supervisor.schedule.propose": "generic_mcp_plus_plus_proxy",
        "supervisor.schedule.claim": "generic_mcp_plus_plus_proxy",
        "supervisor.schedule.renew": "generic_mcp_plus_plus_proxy",
        "supervisor.schedule.release": "generic_mcp_plus_plus_proxy",
        "supervisor.schedule.reconcile": "generic_mcp_plus_plus_proxy",
    }
)

# Baseline capability inventory used when SwissKnife sources are absent.
# Must stay aligned with contracts/agent-supervisor-console.schema.json enums.
_BASELINE_CAPABILITIES: Final[tuple[Mapping[str, str], ...]] = (
    {
        "id": "supervisor.health.read",
        "method": "agent_supervisor.health.read",
        "owner": "ipfs_accelerate_py",
        "access": "read",
        "policy_class": "read",
    },
    {
        "id": "supervisor.queue.read",
        "method": "agent_supervisor.queue.read",
        "owner": "ipfs_accelerate_py",
        "access": "read",
        "policy_class": "read",
    },
    {
        "id": "supervisor.goals.read",
        "method": "agent_supervisor.goals.read",
        "owner": "ipfs_accelerate_py",
        "access": "read",
        "policy_class": "read",
    },
    {
        "id": "supervisor.subgoals.read",
        "method": "agent_supervisor.subgoals.read",
        "owner": "ipfs_accelerate_py",
        "access": "read",
        "policy_class": "read",
    },
    {
        "id": "supervisor.taskboard.links.read",
        "method": "agent_supervisor.taskboard.links.read",
        "owner": "ipfs_datasets_py",
        "access": "read",
        "policy_class": "read",
    },
    {
        "id": "supervisor.logs.read",
        "method": "agent_supervisor.logs.read",
        "owner": "ipfs_accelerate_py",
        "access": "read",
        "policy_class": "read",
    },
    {
        "id": "supervisor.receipts.read",
        "method": "agent_supervisor.receipts.read",
        "owner": "ipfs_kit_py",
        "access": "read",
        "policy_class": "read",
    },
    {
        "id": "supervisor.policy.assist",
        "method": "agent_supervisor.policy.assist",
        "owner": "ipfs_datasets_py",
        "access": "read",
        "policy_class": "read",
    },
    {
        "id": "supervisor.semantic-goal.assist",
        "method": "agent_supervisor.semantic_goal.assist",
        "owner": "ipfs_datasets_py",
        "access": "read",
        "policy_class": "read",
    },
    {
        "id": "supervisor.receipts.persist",
        "method": "agent_supervisor.receipts.persist",
        "owner": "ipfs_kit_py",
        "access": "governed-write",
        "policy_class": "confirm",
    },
    {
        "id": "supervisor.content.retrieve",
        "method": "agent_supervisor.content.retrieve",
        "owner": "ipfs_kit_py",
        "access": "read",
        "policy_class": "read",
    },
    {
        "id": "supervisor.event-dag.checkpoint",
        "method": "agent_supervisor.event_dag.checkpoint",
        "owner": "ipfs_kit_py",
        "access": "governed-write",
        "policy_class": "confirm",
    },
    {
        "id": "supervisor.run-history.search",
        "method": "agent_supervisor.run_history.search",
        "owner": "ipfs_datasets_py",
        "access": "read",
        "policy_class": "read",
    },
    {
        "id": "supervisor.prompt-steering.request",
        "method": "agent_supervisor.prompt_steering.request",
        "owner": "ipfs_accelerate_py",
        "access": "governed-write",
        "policy_class": "confirm",
    },
    {
        "id": "supervisor.task-control.request",
        "method": "agent_supervisor.task_control.request",
        "owner": "ipfs_accelerate_py",
        "access": "governed-write",
        "policy_class": "privileged-control",
    },
    {
        "id": "supervisor.profile-g.read",
        "method": "mcp++/risk/profile",
        "owner": "ipfs_accelerate_py",
        "access": "read",
        "policy_class": "read",
    },
    {
        "id": "supervisor.schedule.frontier.read",
        "method": "mcp++/schedule/frontier",
        "owner": "ipfs_accelerate_py",
        "access": "read",
        "policy_class": "read",
    },
    {
        "id": "supervisor.neighborhood.read",
        "method": "mcp++/neighborhood/query",
        "owner": "ipfs_accelerate_py",
        "access": "read",
        "policy_class": "read",
    },
    {
        "id": "supervisor.schedule.claims.read",
        "method": "mcp++/schedule/status",
        "owner": "ipfs_accelerate_py",
        "access": "read",
        "policy_class": "read",
    },
    {
        "id": "supervisor.risk.read",
        "method": "mcp++/risk/history",
        "owner": "ipfs_accelerate_py",
        "access": "read",
        "policy_class": "read",
    },
    {
        "id": "supervisor.goal.decompose",
        "method": "mcp++/goals/decompose",
        "owner": "ipfs_accelerate_py",
        "access": "governed-write",
        "policy_class": "confirm",
    },
    {
        "id": "supervisor.schedule.propose",
        "method": "mcp++/schedule/propose",
        "owner": "ipfs_accelerate_py",
        "access": "governed-write",
        "policy_class": "confirm",
    },
    {
        "id": "supervisor.schedule.claim",
        "method": "mcp++/schedule/claim",
        "owner": "ipfs_accelerate_py",
        "access": "governed-write",
        "policy_class": "privileged-control",
    },
    {
        "id": "supervisor.schedule.renew",
        "method": "mcp++/schedule/renew",
        "owner": "ipfs_accelerate_py",
        "access": "governed-write",
        "policy_class": "privileged-control",
    },
    {
        "id": "supervisor.schedule.release",
        "method": "mcp++/schedule/release",
        "owner": "ipfs_accelerate_py",
        "access": "governed-write",
        "policy_class": "confirm",
    },
    {
        "id": "supervisor.schedule.reconcile",
        "method": "mcp++/schedule/reconcile",
        "owner": "ipfs_accelerate_py",
        "access": "governed-write",
        "policy_class": "privileged-control",
    },
)

_GENERIC_PROXY_REASON_CODES: Final[frozenset[str]] = frozenset(
    {
        "generic_control_adapter",
        "generic_usage_tool",
        "generic_mcp_plus_plus_proxy",
        "datasets_search_proxy",
        "kit_storage_proxy",
        "workflow_data_storage_proxy",
        "unknown_tool_name",
        "non_native_tool_prefix",
        "operation_not_in_catalog",
        "ambiguous_multi_operation_selection",
        "scheduler_lease_action_not_native_control_op",
        "task_control_action_required",
        "task_control_unknown_action",
    }
)


class SupervisorContractExtractorError(ValueError):
    """Malformed extractor input or fail-closed mapping violation."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "supervisor_contract_extractor_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = dict(details or {})


class GenericProxySelectionError(SupervisorContractExtractorError):
    """Raised when a generic proxy is offered as a native operation identity."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "generic_proxy_selection_rejected",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, reason_code=reason_code, details=details)


class MappingDisposition(str, Enum):
    """Whether a SwissKnife capability reaches a native operation."""

    MAPPED = "mapped"
    REFUTED = "refuted"
    ACTION_MAPPED = "action_mapped"


class CapabilityAccess(str, Enum):
    READ = "read"
    GOVERNED_WRITE = "governed-write"


def _cid(payload: Mapping[str, Any]) -> str:
    return identify_strict_artifact(payload).cid


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SupervisorContractExtractorError(
            f"{field_name} must be a nonempty string",
            reason_code="invalid_field",
            details={"field": field_name},
        )
    return value


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SupervisorContractExtractorError(
            f"{field_name} must be an object",
            reason_code="invalid_field",
            details={"field": field_name},
        )
    return value


def _sequence(value: object, field_name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SupervisorContractExtractorError(
            f"{field_name} must be an array",
            reason_code="invalid_field",
            details={"field": field_name},
        )
    return value


def native_tool_name(operation: Operation | str) -> str:
    """Return the exact MCP tool name for a catalog operation."""

    if isinstance(operation, Operation):
        value = operation.value
    else:
        value = str(operation).strip()
    if not value:
        raise SupervisorContractExtractorError(
            "operation is required",
            reason_code="invalid_operation",
        )
    return f"{NATIVE_TOOL_PREFIX}{value}"


def native_function_identity(operation: Operation | str) -> str:
    return f"{NATIVE_TOOLS_MODULE}:{native_tool_name(operation)}"


def parse_native_tool_name(tool_name: str) -> Operation:
    """Parse an exact ``agent_supervisor_<operation>`` tool name.

    Generic adapters and non-catalog names fail closed.
    """

    name = _text(tool_name, "tool_name")
    if name in {GENERIC_CONTROL_ADAPTER_TOOL, GENERIC_USAGE_TOOL}:
        raise GenericProxySelectionError(
            f"generic proxy tool {name!r} is not an exact native operation",
            reason_code=(
                "generic_control_adapter"
                if name == GENERIC_CONTROL_ADAPTER_TOOL
                else "generic_usage_tool"
            ),
            details={"tool_name": name},
        )
    if name.startswith("mcp++/") or "/" in name:
        raise GenericProxySelectionError(
            f"generic mcp++/proxy method {name!r} is not a native tool",
            reason_code="generic_mcp_plus_plus_proxy",
            details={"tool_name": name},
        )
    if not name.startswith(NATIVE_TOOL_PREFIX):
        raise GenericProxySelectionError(
            f"tool {name!r} does not use the native agent_supervisor_ prefix",
            reason_code="non_native_tool_prefix",
            details={"tool_name": name},
        )
    op_value = name[len(NATIVE_TOOL_PREFIX) :]
    try:
        operation = Operation(op_value)
    except ValueError as exc:
        raise GenericProxySelectionError(
            f"tool {name!r} is not a catalog operation",
            reason_code="operation_not_in_catalog",
            details={"tool_name": name, "operation": op_value},
        ) from exc
    # Reject alias confusion: only the primary enum value spelling is accepted.
    if native_tool_name(operation) != name:
        raise GenericProxySelectionError(
            f"tool {name!r} is not the canonical agent_supervisor_* spelling",
            reason_code="operation_not_in_catalog",
            details={"tool_name": name, "canonical": native_tool_name(operation)},
        )
    return operation


@dataclass(frozen=True, slots=True)
class NativeOperationIdentity:
    """Exact native identity for one catalog operation."""

    operation: str
    tool_name: str
    request_schema_id: str
    result_schema_id: str
    behavior_id: str
    dispatcher_id: str
    function_identity: str
    executor_identity: str
    authority: str
    backend_capability: str
    family: str
    requires_authorization: bool
    requires_idempotency: bool
    supports_dry_run: bool
    requires_lease: bool
    requires_fencing: bool
    policy_dominated: bool
    identity_cid: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": NATIVE_OPERATION_IDENTITY_SCHEMA,
            "operation": self.operation,
            "tool_name": self.tool_name,
            "request_schema_id": self.request_schema_id,
            "result_schema_id": self.result_schema_id,
            "behavior_id": self.behavior_id,
            "dispatcher_id": self.dispatcher_id,
            "function_identity": self.function_identity,
            "executor_identity": self.executor_identity,
            "authority": self.authority,
            "backend_capability": self.backend_capability,
            "family": self.family,
            "requires_authorization": self.requires_authorization,
            "requires_idempotency": self.requires_idempotency,
            "supports_dry_run": self.supports_dry_run,
            "requires_lease": self.requires_lease,
            "requires_fencing": self.requires_fencing,
            "policy_dominated": self.policy_dominated,
        }
        identity_cid = self.identity_cid or _cid(payload)
        payload["identity_cid"] = identity_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NativeOperationIdentity:
        payload = _mapping(data, "native_operation_identity")
        return cls(
            operation=_text(payload.get("operation"), "operation"),
            tool_name=_text(payload.get("tool_name"), "tool_name"),
            request_schema_id=_text(
                payload.get("request_schema_id"), "request_schema_id"
            ),
            result_schema_id=_text(
                payload.get("result_schema_id"), "result_schema_id"
            ),
            behavior_id=_text(payload.get("behavior_id"), "behavior_id"),
            dispatcher_id=_text(payload.get("dispatcher_id"), "dispatcher_id"),
            function_identity=_text(
                payload.get("function_identity"), "function_identity"
            ),
            executor_identity=_text(
                payload.get("executor_identity"), "executor_identity"
            ),
            authority=_text(payload.get("authority"), "authority"),
            backend_capability=_text(
                payload.get("backend_capability"), "backend_capability"
            ),
            family=str(payload.get("family") or ""),
            requires_authorization=bool(payload.get("requires_authorization")),
            requires_idempotency=bool(payload.get("requires_idempotency")),
            supports_dry_run=bool(payload.get("supports_dry_run")),
            requires_lease=bool(payload.get("requires_lease")),
            requires_fencing=bool(payload.get("requires_fencing")),
            policy_dominated=bool(payload.get("policy_dominated")),
            identity_cid=str(payload.get("identity_cid") or ""),
        )


@dataclass(frozen=True, slots=True)
class SwissKnifeCapabilityRecord:
    """One SwissKnife console capability as declared to the browser."""

    capability_id: str
    method: str
    owner: str
    access: str
    policy_class: str
    source_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "method": self.method,
            "owner": self.owner,
            "access": self.access,
            "policy_class": self.policy_class,
            "source_path": self.source_path,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SwissKnifeCapabilityRecord:
        payload = _mapping(data, "capability")
        return cls(
            capability_id=_text(payload.get("capability_id") or payload.get("id"), "capability_id"),
            method=_text(payload.get("method"), "method"),
            owner=_text(payload.get("owner"), "owner"),
            access=_text(payload.get("access"), "access"),
            policy_class=_text(payload.get("policy_class"), "policy_class"),
            source_path=str(payload.get("source_path") or ""),
        )


@dataclass(frozen=True, slots=True)
class ActionNativeBinding:
    """Action-parameterized binding for task-control style capabilities."""

    action: str
    disposition: MappingDisposition
    operation: str = ""
    identity: NativeOperationIdentity | None = None
    reason_code: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "disposition": self.disposition.value,
            "operation": self.operation,
            "identity": None if self.identity is None else self.identity.to_dict(),
            "reason_code": self.reason_code,
            "details": _json_value(dict(self.details)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ActionNativeBinding:
        payload = _mapping(data, "action_binding")
        identity_raw = payload.get("identity")
        identity = (
            NativeOperationIdentity.from_dict(identity_raw)
            if isinstance(identity_raw, Mapping)
            else None
        )
        return cls(
            action=_text(payload.get("action"), "action"),
            disposition=MappingDisposition(
                _text(payload.get("disposition"), "disposition")
            ),
            operation=str(payload.get("operation") or ""),
            identity=identity,
            reason_code=str(payload.get("reason_code") or ""),
            details=dict(payload.get("details") or {}),
        )


@dataclass(frozen=True, slots=True)
class CapabilityNativeMapping:
    """Disposition of one SwissKnife capability against native identities."""

    capability_id: str
    method: str
    owner: str
    access: str
    policy_class: str
    disposition: MappingDisposition
    operation: str = ""
    identity: NativeOperationIdentity | None = None
    reason_code: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)
    action_bindings: tuple[ActionNativeBinding, ...] = ()
    mapping_cid: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": CAPABILITY_MAPPING_SCHEMA,
            "capability_id": self.capability_id,
            "method": self.method,
            "owner": self.owner,
            "access": self.access,
            "policy_class": self.policy_class,
            "disposition": self.disposition.value,
            "operation": self.operation,
            "identity": None if self.identity is None else self.identity.to_dict(),
            "reason_code": self.reason_code,
            "details": _json_value(dict(self.details)),
            "action_bindings": [item.to_dict() for item in self.action_bindings],
        }
        mapping_cid = self.mapping_cid or _cid(payload)
        payload["mapping_cid"] = mapping_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CapabilityNativeMapping:
        payload = _mapping(data, "capability_mapping")
        identity_raw = payload.get("identity")
        identity = (
            NativeOperationIdentity.from_dict(identity_raw)
            if isinstance(identity_raw, Mapping)
            else None
        )
        action_raw = _sequence(payload.get("action_bindings") or (), "action_bindings")
        return cls(
            capability_id=_text(payload.get("capability_id"), "capability_id"),
            method=_text(payload.get("method"), "method"),
            owner=_text(payload.get("owner"), "owner"),
            access=_text(payload.get("access"), "access"),
            policy_class=_text(payload.get("policy_class"), "policy_class"),
            disposition=MappingDisposition(
                _text(payload.get("disposition"), "disposition")
            ),
            operation=str(payload.get("operation") or ""),
            identity=identity,
            reason_code=str(payload.get("reason_code") or ""),
            details=dict(payload.get("details") or {}),
            action_bindings=tuple(
                ActionNativeBinding.from_dict(item)  # type: ignore[arg-type]
                for item in action_raw
            ),
            mapping_cid=str(payload.get("mapping_cid") or ""),
        )


@dataclass(frozen=True, slots=True)
class GoalCompletionContract:
    """Goal/task completion requires current child/evidence/health/exhaustion."""

    evaluator_identity: str
    required_closure: tuple[str, ...]
    gate_check_names: tuple[str, ...]
    claim_family: str
    fail_closed: bool
    contract_cid: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": GOAL_COMPLETION_CONTRACT_SCHEMA,
            "evaluator_identity": self.evaluator_identity,
            "required_closure": list(self.required_closure),
            "gate_check_names": list(self.gate_check_names),
            "claim_family": self.claim_family,
            "fail_closed": self.fail_closed,
        }
        contract_cid = self.contract_cid or _cid(payload)
        payload["contract_cid"] = contract_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GoalCompletionContract:
        payload = _mapping(data, "goal_completion_contract")
        closure = tuple(
            str(item)
            for item in _sequence(
                payload.get("required_closure") or (), "required_closure"
            )
        )
        checks = tuple(
            str(item)
            for item in _sequence(
                payload.get("gate_check_names") or (), "gate_check_names"
            )
        )
        return cls(
            evaluator_identity=_text(
                payload.get("evaluator_identity"), "evaluator_identity"
            ),
            required_closure=closure,
            gate_check_names=checks,
            claim_family=_text(payload.get("claim_family"), "claim_family"),
            fail_closed=bool(payload.get("fail_closed", True)),
            contract_cid=str(payload.get("contract_cid") or ""),
        )


@dataclass(frozen=True, slots=True)
class MutationPolicyContract:
    """Policy dominance contract for one mutation operation."""

    operation: str
    tool_name: str
    authority: str
    requires_authorization: bool
    requires_idempotency: bool
    supports_dry_run: bool
    policy_path: tuple[str, ...]
    policy_dominated: bool
    identity: NativeOperationIdentity
    contract_cid: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": MUTATION_POLICY_CONTRACT_SCHEMA,
            "operation": self.operation,
            "tool_name": self.tool_name,
            "authority": self.authority,
            "requires_authorization": self.requires_authorization,
            "requires_idempotency": self.requires_idempotency,
            "supports_dry_run": self.supports_dry_run,
            "policy_path": list(self.policy_path),
            "policy_dominated": self.policy_dominated,
            "identity": self.identity.to_dict(),
        }
        contract_cid = self.contract_cid or _cid(payload)
        payload["contract_cid"] = contract_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MutationPolicyContract:
        payload = _mapping(data, "mutation_policy_contract")
        identity = NativeOperationIdentity.from_dict(
            _mapping(payload.get("identity"), "identity")
        )
        return cls(
            operation=_text(payload.get("operation"), "operation"),
            tool_name=_text(payload.get("tool_name"), "tool_name"),
            authority=_text(payload.get("authority"), "authority"),
            requires_authorization=bool(payload.get("requires_authorization")),
            requires_idempotency=bool(payload.get("requires_idempotency")),
            supports_dry_run=bool(payload.get("supports_dry_run")),
            policy_path=tuple(
                str(item)
                for item in _sequence(payload.get("policy_path") or (), "policy_path")
            ),
            policy_dominated=bool(payload.get("policy_dominated")),
            identity=identity,
            contract_cid=str(payload.get("contract_cid") or ""),
        )


@dataclass(frozen=True, slots=True)
class SupervisorContractCatalog:
    """CID-bound catalog of native ops, capability maps, and goal/mutation contracts."""

    interface: str
    version: str
    native_operations: tuple[NativeOperationIdentity, ...]
    capabilities: tuple[SwissKnifeCapabilityRecord, ...]
    mappings: tuple[CapabilityNativeMapping, ...]
    goal_completion: GoalCompletionContract
    mutation_policies: tuple[MutationPolicyContract, ...]
    dispatcher_id: str
    executor_identity: str
    catalog_cid: str = ""
    source_paths: tuple[str, ...] = ()

    def identity_for(self, operation: Operation | str) -> NativeOperationIdentity:
        op_value = operation.value if isinstance(operation, Operation) else str(operation)
        for item in self.native_operations:
            if item.operation == op_value:
                return item
        raise SupervisorContractExtractorError(
            f"operation {op_value!r} is not in the catalog",
            reason_code="operation_not_in_catalog",
            details={"operation": op_value},
        )

    def mapping_for(self, capability_id: str) -> CapabilityNativeMapping:
        for item in self.mappings:
            if item.capability_id == capability_id:
                return item
        raise SupervisorContractExtractorError(
            f"capability {capability_id!r} is not mapped",
            reason_code="capability_not_found",
            details={"capability_id": capability_id},
        )

    def mapped_capability_ids(self) -> tuple[str, ...]:
        return tuple(
            item.capability_id
            for item in self.mappings
            if item.disposition is MappingDisposition.MAPPED
        )

    def refuted_capability_ids(self) -> tuple[str, ...]:
        return tuple(
            item.capability_id
            for item in self.mappings
            if item.disposition is MappingDisposition.REFUTED
        )

    def action_mapped_capability_ids(self) -> tuple[str, ...]:
        return tuple(
            item.capability_id
            for item in self.mappings
            if item.disposition is MappingDisposition.ACTION_MAPPED
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": SUPERVISOR_CONTRACT_CATALOG_SCHEMA,
            "interface": self.interface,
            "version": self.version,
            "dispatcher_id": self.dispatcher_id,
            "executor_identity": self.executor_identity,
            "native_operations": [item.to_dict() for item in self.native_operations],
            "capabilities": [item.to_dict() for item in self.capabilities],
            "mappings": [item.to_dict() for item in self.mappings],
            "goal_completion": self.goal_completion.to_dict(),
            "mutation_policies": [item.to_dict() for item in self.mutation_policies],
            "source_paths": list(self.source_paths),
        }
        catalog_cid = self.catalog_cid or _cid(payload)
        payload["catalog_cid"] = catalog_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SupervisorContractCatalog:
        payload = _mapping(data, "supervisor_contract_catalog")
        native = tuple(
            NativeOperationIdentity.from_dict(item)  # type: ignore[arg-type]
            for item in _sequence(
                payload.get("native_operations") or (), "native_operations"
            )
        )
        capabilities = tuple(
            SwissKnifeCapabilityRecord.from_dict(item)  # type: ignore[arg-type]
            for item in _sequence(payload.get("capabilities") or (), "capabilities")
        )
        mappings = tuple(
            CapabilityNativeMapping.from_dict(item)  # type: ignore[arg-type]
            for item in _sequence(payload.get("mappings") or (), "mappings")
        )
        mutations = tuple(
            MutationPolicyContract.from_dict(item)  # type: ignore[arg-type]
            for item in _sequence(
                payload.get("mutation_policies") or (), "mutation_policies"
            )
        )
        return cls(
            interface=_text(payload.get("interface"), "interface"),
            version=_text(payload.get("version"), "version"),
            native_operations=native,
            capabilities=capabilities,
            mappings=mappings,
            goal_completion=GoalCompletionContract.from_dict(
                _mapping(payload.get("goal_completion"), "goal_completion")
            ),
            mutation_policies=mutations,
            dispatcher_id=_text(payload.get("dispatcher_id"), "dispatcher_id"),
            executor_identity=_text(
                payload.get("executor_identity"), "executor_identity"
            ),
            catalog_cid=str(payload.get("catalog_cid") or ""),
            source_paths=tuple(
                str(item)
                for item in _sequence(payload.get("source_paths") or (), "source_paths")
            ),
        )


def build_native_operation_identity(
    operation: Operation | str,
    *,
    catalog: OperationCatalog | None = None,
) -> NativeOperationIdentity:
    """Build the exact identity record for one catalog operation."""

    selected_catalog = catalog if catalog is not None else get_operation_catalog()
    if not isinstance(selected_catalog, OperationCatalog):
        raise SupervisorContractExtractorError(
            "catalog must be an OperationCatalog",
            reason_code="invalid_catalog",
        )
    op = operation if isinstance(operation, Operation) else Operation(str(operation))
    descriptor = selected_catalog.operation(op)
    if not isinstance(descriptor, ControlOperationDescriptor):
        raise SupervisorContractExtractorError(
            f"missing descriptor for {op.value}",
            reason_code="missing_descriptor",
            details={"operation": op.value},
        )
    authority = descriptor.authority
    authority_value = (
        authority.value if isinstance(authority, OperationAuthority) else str(authority)
    )
    policy_dominated = authority is OperationAuthority.MUTATION and bool(
        descriptor.requires_authorization
        and descriptor.supports_dry_run
        and descriptor.requires_idempotency
    )
    # Proposal and read operations are policy-safe without mutation dominance.
    if authority is not OperationAuthority.MUTATION:
        policy_dominated = True
    identity = NativeOperationIdentity(
        operation=op.value,
        tool_name=native_tool_name(op),
        request_schema_id=str(descriptor.request_schema_id),
        result_schema_id=str(descriptor.result_schema_id),
        behavior_id=control_operation_behavior_id(descriptor),
        dispatcher_id=DIRECT_CONTROL_SERVICE_DISPATCHER_ID,
        function_identity=native_function_identity(op),
        executor_identity=NATIVE_EXECUTOR_IDENTITY,
        authority=authority_value,
        backend_capability=str(descriptor.backend_capability),
        family=str(getattr(descriptor, "family", "") or op.value),
        requires_authorization=bool(descriptor.requires_authorization),
        requires_idempotency=bool(descriptor.requires_idempotency),
        supports_dry_run=bool(descriptor.supports_dry_run),
        requires_lease=bool(descriptor.requires_lease),
        requires_fencing=bool(descriptor.requires_fencing),
        policy_dominated=policy_dominated,
    )
    return NativeOperationIdentity.from_dict(identity.to_dict())


def extract_native_operations(
    *,
    catalog: OperationCatalog | None = None,
) -> tuple[NativeOperationIdentity, ...]:
    """Extract one identity per catalog operation, sorted by operation value."""

    selected = catalog if catalog is not None else get_operation_catalog()
    operations = sorted(
        {item if isinstance(item, Operation) else Operation(str(item)) for item in selected.operations},
        key=lambda item: item.value,
    )
    return tuple(
        build_native_operation_identity(operation, catalog=selected)
        for operation in operations
    )


def select_native_identity(
    tool_or_operation: str,
    *,
    catalog: OperationCatalog | None = None,
) -> NativeOperationIdentity:
    """Resolve a tool/operation name to one exact identity or reject proxies."""

    text = _text(tool_or_operation, "tool_or_operation")
    if text in {GENERIC_CONTROL_ADAPTER_TOOL, GENERIC_USAGE_TOOL}:
        raise GenericProxySelectionError(
            f"generic proxy selection {text!r} is rejected",
            reason_code=(
                "generic_control_adapter"
                if text == GENERIC_CONTROL_ADAPTER_TOOL
                else "generic_usage_tool"
            ),
            details={"selection": text},
        )
    if text.startswith("mcp++/") or text.startswith("workflow/") or text.startswith(
        "data/"
    ) or text.startswith("storage/"):
        raise GenericProxySelectionError(
            f"generic workflow/data/storage/mcp++ proxy {text!r} is rejected",
            reason_code="workflow_data_storage_proxy",
            details={"selection": text},
        )
    if text.startswith(NATIVE_TOOL_PREFIX) or text == GENERIC_CONTROL_ADAPTER_TOOL:
        operation = parse_native_tool_name(text)
    else:
        try:
            operation = Operation(text)
        except ValueError as exc:
            # Last chance: dotted backend capability agent_supervisor.<op>
            if text.startswith("agent_supervisor.") and text.count(".") == 1:
                try:
                    operation = Operation(text.split(".", 1)[1])
                except ValueError:
                    raise GenericProxySelectionError(
                        f"selection {text!r} is not an exact native operation",
                        reason_code="unknown_tool_name",
                        details={"selection": text},
                    ) from exc
            else:
                raise GenericProxySelectionError(
                    f"selection {text!r} is not an exact native operation",
                    reason_code="unknown_tool_name",
                    details={"selection": text},
                ) from exc
    return build_native_operation_identity(operation, catalog=catalog)


def extract_goal_completion_contract() -> GoalCompletionContract:
    """Materialize the goal/task completion closure contract."""

    if evaluate_completion_gate is None:  # pragma: no cover - import guard
        raise SupervisorContractExtractorError(
            "goal completion evaluator is unavailable",
            reason_code="goal_completion_unavailable",
        )
    # Gate check names are part of the stable completion surface.
    gate_checks = (
        "artifact_binding",
        "mandatory_coverage",
        "producer_channel_binding",
        "required_validations",
        "analyzer_health",
        "exhaustion_quorum",
        "analysis_terminal_state",
        "child_goals",
    )
    for required in REQUIRED_COMPLETION_CLOSURE:
        if required not in gate_checks and required != "required_validations":
            raise SupervisorContractExtractorError(
                f"required completion closure member {required!r} missing",
                reason_code="completion_closure_incomplete",
            )
    contract = GoalCompletionContract(
        evaluator_identity=GOAL_COMPLETION_EVALUATOR_IDENTITY,
        required_closure=REQUIRED_COMPLETION_CLOSURE,
        gate_check_names=gate_checks,
        claim_family="GoalTaskClosure",
        fail_closed=True,
    )
    return GoalCompletionContract.from_dict(contract.to_dict())


def extract_mutation_policy_contracts(
    *,
    catalog: OperationCatalog | None = None,
) -> tuple[MutationPolicyContract, ...]:
    """Extract policy-dominance contracts for every mutation operation."""

    identities = {
        item.operation: item for item in extract_native_operations(catalog=catalog)
    }
    contracts: list[MutationPolicyContract] = []
    for operation in sorted(MUTATION_OPERATIONS, key=lambda item: item.value):
        identity = identities[operation.value]
        if not (
            identity.requires_authorization
            and identity.supports_dry_run
            and identity.requires_idempotency
            and identity.policy_dominated
        ):
            raise SupervisorContractExtractorError(
                f"mutation operation {operation.value!r} is not policy dominated",
                reason_code="mutation_policy_not_dominated",
                details={
                    "operation": operation.value,
                    "requires_authorization": identity.requires_authorization,
                    "supports_dry_run": identity.supports_dry_run,
                    "requires_idempotency": identity.requires_idempotency,
                    "policy_dominated": identity.policy_dominated,
                },
            )
        contract = MutationPolicyContract(
            operation=operation.value,
            tool_name=identity.tool_name,
            authority=identity.authority,
            requires_authorization=identity.requires_authorization,
            requires_idempotency=identity.requires_idempotency,
            supports_dry_run=identity.supports_dry_run,
            policy_path=MUTATION_POLICY_PATH,
            policy_dominated=True,
            identity=identity,
        )
        contracts.append(MutationPolicyContract.from_dict(contract.to_dict()))
    return tuple(contracts)


def _baseline_capabilities() -> tuple[SwissKnifeCapabilityRecord, ...]:
    return tuple(
        SwissKnifeCapabilityRecord(
            capability_id=str(item["id"]),
            method=str(item["method"]),
            owner=str(item["owner"]),
            access=str(item["access"]),
            policy_class=str(item["policy_class"]),
            source_path="baseline:supervisor-console-inventory",
        )
        for item in _BASELINE_CAPABILITIES
    )


def _parse_schema_capability_ids(schema_path: Path) -> tuple[str, ...]:
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    capability_id = (
        payload.get("$defs", {}).get("capabilityId", {}) if isinstance(payload, dict) else {}
    )
    enum_values = capability_id.get("enum") if isinstance(capability_id, dict) else None
    if not isinstance(enum_values, list) or not enum_values:
        raise SupervisorContractExtractorError(
            "console schema is missing capabilityId enum",
            reason_code="swissknife_schema_incomplete",
            details={"path": str(schema_path)},
        )
    return tuple(str(item) for item in enum_values)


def _parse_gateway_capabilities(
    gateway_path: Path,
) -> tuple[SwissKnifeCapabilityRecord, ...]:
    """Cold-parse CAPABILITIES object literals from the TypeScript gateway."""

    text = gateway_path.read_text(encoding="utf-8")
    # Match compact or expanded capability object literals with id/method/owner.
    pattern = re.compile(
        r"\{\s*id:\s*'(?P<id>[^']+)'\s*,"
        r"(?:[^}]*?title:\s*'[^']*'\s*,)?"
        r"\s*access:\s*'(?P<access>[^']+)'\s*,"
        r"\s*owner:\s*'(?P<owner>[^']+)'\s*,"
        r"\s*policy_class:\s*'(?P<policy_class>[^']+)'\s*,"
        r"(?:[^}]*?transports:\s*[^,]+,)?"
        r"\s*method:\s*'(?P<method>[^']+)'",
        re.DOTALL,
    )
    found: list[SwissKnifeCapabilityRecord] = []
    seen: set[str] = set()
    for match in pattern.finditer(text):
        capability_id = match.group("id")
        if capability_id in seen:
            continue
        seen.add(capability_id)
        found.append(
            SwissKnifeCapabilityRecord(
                capability_id=capability_id,
                method=match.group("method"),
                owner=match.group("owner"),
                access=match.group("access"),
                policy_class=match.group("policy_class"),
                source_path=SWISSKNIFE_CONSOLE_GATEWAY_REL,
            )
        )
    if not found:
        raise SupervisorContractExtractorError(
            "gateway source did not yield capability descriptors",
            reason_code="swissknife_gateway_unparsed",
            details={"path": str(gateway_path)},
        )
    return tuple(found)


def extract_swissknife_capabilities(
    swissknife_root: str | Path | None = None,
) -> tuple[SwissKnifeCapabilityRecord, ...]:
    """Extract SwissKnife console capabilities from schema/gateway when present."""

    if swissknife_root is None:
        return _baseline_capabilities()
    root = Path(swissknife_root)
    if not root.is_dir():
        raise SupervisorContractExtractorError(
            "swissknife_root must be an existing directory",
            reason_code="invalid_swissknife_root",
            details={"path": str(root)},
        )
    schema_path = root / SWISSKNIFE_CONSOLE_SCHEMA_REL
    gateway_path = root / SWISSKNIFE_CONSOLE_GATEWAY_REL
    if not schema_path.is_file():
        raise SupervisorContractExtractorError(
            "SwissKnife console schema is missing",
            reason_code="swissknife_schema_missing",
            details={"path": str(schema_path)},
        )
    schema_ids = set(_parse_schema_capability_ids(schema_path))
    if gateway_path.is_file():
        capabilities = _parse_gateway_capabilities(gateway_path)
    else:
        # Fall back to baseline filtered by schema enum.
        capabilities = tuple(
            item
            for item in _baseline_capabilities()
            if item.capability_id in schema_ids
        )
    capability_ids = {item.capability_id for item in capabilities}
    # Schema enum is the closed set of capability IDs; gateway may be a superset
    # of optional expanded capabilities.  Every schema ID must be present.
    missing = sorted(schema_ids.difference(capability_ids))
    if missing:
        # Merge baseline records for any schema IDs the gateway parser missed.
        baseline_by_id = {item.capability_id: item for item in _baseline_capabilities()}
        extras = []
        still_missing = []
        for capability_id in missing:
            if capability_id in baseline_by_id:
                extras.append(baseline_by_id[capability_id])
            else:
                still_missing.append(capability_id)
        if still_missing:
            raise SupervisorContractExtractorError(
                "console capabilities incomplete relative to schema enum",
                reason_code="swissknife_capability_gap",
                details={"missing": still_missing},
            )
        capabilities = tuple(
            sorted(
                list(capabilities) + extras,
                key=lambda item: item.capability_id,
            )
        )
    # Keep only known schema capabilities plus reviewed expanded IDs already
    # present in the baseline inventory (gateway may declare expanded ops).
    allowed = schema_ids.union({item["id"] for item in _BASELINE_CAPABILITIES})
    filtered = tuple(
        item for item in capabilities if item.capability_id in allowed
    )
    return tuple(sorted(filtered, key=lambda item: item.capability_id))


def _refute_mapping(
    capability: SwissKnifeCapabilityRecord,
    reason_code: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> CapabilityNativeMapping:
    mapping = CapabilityNativeMapping(
        capability_id=capability.capability_id,
        method=capability.method,
        owner=capability.owner,
        access=capability.access,
        policy_class=capability.policy_class,
        disposition=MappingDisposition.REFUTED,
        reason_code=reason_code,
        details=dict(details or {}),
    )
    return CapabilityNativeMapping.from_dict(mapping.to_dict())


def _mapped_identity(
    capability: SwissKnifeCapabilityRecord,
    operation: Operation,
    *,
    catalog: OperationCatalog | None = None,
) -> CapabilityNativeMapping:
    identity = build_native_operation_identity(operation, catalog=catalog)
    mapping = CapabilityNativeMapping(
        capability_id=capability.capability_id,
        method=capability.method,
        owner=capability.owner,
        access=capability.access,
        policy_class=capability.policy_class,
        disposition=MappingDisposition.MAPPED,
        operation=operation.value,
        identity=identity,
    )
    return CapabilityNativeMapping.from_dict(mapping.to_dict())


def map_swissknife_capability(
    capability: SwissKnifeCapabilityRecord | Mapping[str, Any],
    *,
    catalog: OperationCatalog | None = None,
) -> CapabilityNativeMapping:
    """Map one capability to an exact native identity or a typed refutation."""

    record = (
        capability
        if isinstance(capability, SwissKnifeCapabilityRecord)
        else SwissKnifeCapabilityRecord.from_dict(capability)
    )
    capability_id = record.capability_id
    method = record.method

    # Explicit refutation table (datasets/kit/mcp++ proxies).
    if capability_id in _REFUTED_CAPABILITIES:
        return _refute_mapping(
            record,
            _REFUTED_CAPABILITIES[capability_id],
            details={"method": method, "owner": record.owner},
        )

    # Generic method forms never become native reachability.
    if method.startswith("mcp++/") or method.startswith("workflow/"):
        return _refute_mapping(
            record,
            "generic_mcp_plus_plus_proxy",
            details={"method": method},
        )

    # Task-control is action-parameterized: each action maps or refutes alone.
    if capability_id == "supervisor.task-control.request":
        bindings: list[ActionNativeBinding] = []
        for action, op_value in _TASK_CONTROL_ACTION_OPERATIONS.items():
            identity = build_native_operation_identity(op_value, catalog=catalog)
            binding = ActionNativeBinding(
                action=action,
                disposition=MappingDisposition.MAPPED,
                operation=op_value,
                identity=identity,
            )
            bindings.append(ActionNativeBinding.from_dict(binding.to_dict()))
        for action, reason in _TASK_CONTROL_REFUTED_ACTIONS.items():
            binding = ActionNativeBinding(
                action=action,
                disposition=MappingDisposition.REFUTED,
                reason_code=reason,
                details={"method": method},
            )
            bindings.append(ActionNativeBinding.from_dict(binding.to_dict()))
        mapping = CapabilityNativeMapping(
            capability_id=capability_id,
            method=method,
            owner=record.owner,
            access=record.access,
            policy_class=record.policy_class,
            disposition=MappingDisposition.ACTION_MAPPED,
            details={
                "requires_action_selector": True,
                "generic_control_adapter_rejected": True,
                "note": (
                    "task-control requires an exact lifecycle action; "
                    "agent_supervisor_control is not a substitute"
                ),
            },
            action_bindings=tuple(bindings),
        )
        return CapabilityNativeMapping.from_dict(mapping.to_dict())

    op_value = _EXACT_CAPABILITY_OPERATIONS.get(capability_id)
    if op_value is not None:
        return _mapped_identity(record, Operation(op_value), catalog=catalog)

    # Owner alone never grants native reachability.
    if record.owner != "ipfs_accelerate_py":
        return _refute_mapping(
            record,
            "workflow_data_storage_proxy",
            details={
                "owner": record.owner,
                "method": method,
                "note": "UI backend ownership is not native-operation reachability",
            },
        )

    return _refute_mapping(
        record,
        "unknown_tool_name",
        details={"method": method},
    )


def resolve_task_control_action(
    action: str,
    *,
    catalog: OperationCatalog | None = None,
) -> NativeOperationIdentity:
    """Resolve a task-control action to one exact native identity or refute."""

    selected = _text(action, "action")
    if selected in _TASK_CONTROL_REFUTED_ACTIONS:
        raise GenericProxySelectionError(
            f"task-control action {selected!r} is not a native control operation",
            reason_code=_TASK_CONTROL_REFUTED_ACTIONS[selected],
            details={"action": selected},
        )
    op_value = _TASK_CONTROL_ACTION_OPERATIONS.get(selected)
    if op_value is None:
        raise GenericProxySelectionError(
            f"unknown task-control action {selected!r}",
            reason_code="task_control_unknown_action",
            details={"action": selected},
        )
    return build_native_operation_identity(op_value, catalog=catalog)


def map_all_capabilities(
    capabilities: Sequence[SwissKnifeCapabilityRecord | Mapping[str, Any]],
    *,
    catalog: OperationCatalog | None = None,
) -> tuple[CapabilityNativeMapping, ...]:
    mappings = [
        map_swissknife_capability(item, catalog=catalog) for item in capabilities
    ]
    # Fail closed: every capability must have a terminal disposition.
    for mapping in mappings:
        if mapping.disposition is MappingDisposition.MAPPED:
            if mapping.identity is None or not mapping.operation:
                raise SupervisorContractExtractorError(
                    f"mapped capability {mapping.capability_id!r} lacks identity",
                    reason_code="incomplete_mapping",
                )
            if mapping.identity.tool_name != native_tool_name(mapping.operation):
                raise SupervisorContractExtractorError(
                    f"mapped capability {mapping.capability_id!r} tool mismatch",
                    reason_code="tool_identity_mismatch",
                )
        elif mapping.disposition is MappingDisposition.REFUTED:
            if not mapping.reason_code:
                raise SupervisorContractExtractorError(
                    f"refuted capability {mapping.capability_id!r} lacks reason",
                    reason_code="incomplete_refutation",
                )
        elif mapping.disposition is MappingDisposition.ACTION_MAPPED:
            if not mapping.action_bindings:
                raise SupervisorContractExtractorError(
                    f"action-mapped capability {mapping.capability_id!r} empty",
                    reason_code="incomplete_action_mapping",
                )
        else:  # pragma: no cover - enum exhaustiveness
            raise SupervisorContractExtractorError(
                f"unsupported disposition for {mapping.capability_id!r}",
                reason_code="unsupported_disposition",
            )
    return tuple(sorted(mappings, key=lambda item: item.capability_id))


def build_supervisor_contract_catalog(
    *,
    swissknife_root: str | Path | None = None,
    catalog: OperationCatalog | None = None,
) -> SupervisorContractCatalog:
    """Build the complete supervisor contract catalog for SCA-G174."""

    selected = catalog if catalog is not None else get_operation_catalog()
    # Prefer the canonical v2 catalog when the default factory returns it.
    if selected is None:  # pragma: no cover
        selected = OPERATION_CATALOG_V2

    native_ops = extract_native_operations(catalog=selected)
    # Publication completeness: one identity per Operation, no aliases double-counted.
    op_values = [item.operation for item in native_ops]
    if len(op_values) != len(set(op_values)):
        raise SupervisorContractExtractorError(
            "native operation identities are not unique",
            reason_code="duplicate_native_identity",
        )
    expected_ops = {item.value for item in selected.operations}
    if set(op_values) != expected_ops:
        raise SupervisorContractExtractorError(
            "native identities do not cover the operation catalog",
            reason_code="native_catalog_incomplete",
            details={
                "missing": sorted(expected_ops.difference(op_values)),
                "extra": sorted(set(op_values).difference(expected_ops)),
            },
        )

    capabilities = extract_swissknife_capabilities(swissknife_root)
    mappings = map_all_capabilities(capabilities, catalog=selected)
    if len(mappings) != len(capabilities):
        raise SupervisorContractExtractorError(
            "capability mapping count mismatch",
            reason_code="mapping_coverage_gap",
        )
    mapped_ids = {item.capability_id for item in mappings}
    capability_ids = {item.capability_id for item in capabilities}
    if mapped_ids != capability_ids:
        raise SupervisorContractExtractorError(
            "not every capability received a mapping disposition",
            reason_code="mapping_coverage_gap",
            details={
                "missing": sorted(capability_ids.difference(mapped_ids)),
                "extra": sorted(mapped_ids.difference(capability_ids)),
            },
        )

    goal_completion = extract_goal_completion_contract()
    for member in REQUIRED_COMPLETION_CLOSURE:
        if member not in goal_completion.required_closure:
            raise SupervisorContractExtractorError(
                f"goal completion missing required closure {member!r}",
                reason_code="completion_closure_incomplete",
            )

    mutation_policies = extract_mutation_policy_contracts(catalog=selected)
    if len(mutation_policies) != len(MUTATION_OPERATIONS):
        raise SupervisorContractExtractorError(
            "mutation policy contracts incomplete",
            reason_code="mutation_policy_incomplete",
        )

    source_paths: list[str] = []
    if swissknife_root is not None:
        source_paths.extend(
            [
                SWISSKNIFE_CONSOLE_SCHEMA_REL,
                SWISSKNIFE_CONSOLE_GATEWAY_REL,
            ]
        )
    source_paths.extend(
        [
            "ipfs_accelerate_py/agent_supervisor/control/control_contracts.py",
            "ipfs_accelerate_py/agent_supervisor/control/control_plane.py",
            "ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/"
            "native_agent_supervisor_tools.py",
            "ipfs_accelerate_py/agent_supervisor/objectives/goal_completion.py",
        ]
    )

    result = SupervisorContractCatalog(
        interface=SUPERVISOR_CONTRACT_CATALOG_INTERFACE,
        version=SUPERVISOR_CONTRACT_EXTRACTOR_VERSION,
        native_operations=native_ops,
        capabilities=tuple(sorted(capabilities, key=lambda item: item.capability_id)),
        mappings=mappings,
        goal_completion=goal_completion,
        mutation_policies=mutation_policies,
        dispatcher_id=DIRECT_CONTROL_SERVICE_DISPATCHER_ID,
        executor_identity=NATIVE_EXECUTOR_IDENTITY,
        source_paths=tuple(source_paths),
    )
    return SupervisorContractCatalog.from_dict(result.to_dict())


def load_supervisor_contract_catalog(
    path: str | Path,
) -> SupervisorContractCatalog:
    """Load a previously materialized catalog JSON document."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SupervisorContractExtractorError(
            "catalog file must contain a JSON object",
            reason_code="invalid_catalog_file",
            details={"path": str(path)},
        )
    return SupervisorContractCatalog.from_dict(payload)


def write_supervisor_contract_catalog(
    catalog: SupervisorContractCatalog,
    path: str | Path,
) -> str:
    """Write the catalog as canonical JSON and return its CID."""

    payload = catalog.to_dict()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return str(payload["catalog_cid"])


def assert_no_generic_proxy_selection(selection: str) -> None:
    """Public fail-closed guard used by callers and tests."""

    select_native_identity(selection)


def is_generic_proxy_reason(reason_code: str) -> bool:
    return reason_code in _GENERIC_PROXY_REASON_CODES


def summarize_catalog(catalog: SupervisorContractCatalog) -> dict[str, Any]:
    """Compact summary for evidence and operator diagnostics."""

    return {
        "interface": catalog.interface,
        "version": catalog.version,
        "catalog_cid": catalog.catalog_cid or catalog.to_dict()["catalog_cid"],
        "native_operation_count": len(catalog.native_operations),
        "capability_count": len(catalog.capabilities),
        "mapped_count": len(catalog.mapped_capability_ids()),
        "refuted_count": len(catalog.refuted_capability_ids()),
        "action_mapped_count": len(catalog.action_mapped_capability_ids()),
        "mutation_policy_count": len(catalog.mutation_policies),
        "required_completion_closure": list(catalog.goal_completion.required_closure),
        "dispatcher_id": catalog.dispatcher_id,
        "executor_identity": catalog.executor_identity,
        "read_operation_count": len(READ_OPERATIONS),
        "proposal_operation_count": len(PROPOSAL_OPERATIONS),
        "mutation_operation_count": len(MUTATION_OPERATIONS),
    }


__all__ = [
    "ActionNativeBinding",
    "CapabilityAccess",
    "CapabilityNativeMapping",
    "GENERIC_CONTROL_ADAPTER_TOOL",
    "GENERIC_USAGE_TOOL",
    "GOAL_COMPLETION_EVALUATOR_IDENTITY",
    "GenericProxySelectionError",
    "GoalCompletionContract",
    "MUTATION_POLICY_PATH",
    "MappingDisposition",
    "NATIVE_EXECUTOR_IDENTITY",
    "NATIVE_TOOL_PREFIX",
    "NATIVE_TOOLS_MODULE",
    "NativeOperationIdentity",
    "REQUIRED_COMPLETION_CLOSURE",
    "SUPERVISOR_CONTRACT_CATALOG_INTERFACE",
    "SUPERVISOR_CONTRACT_CATALOG_SCHEMA",
    "SUPERVISOR_CONTRACT_EXTRACTOR_INTERFACE",
    "SUPERVISOR_CONTRACT_EXTRACTOR_VERSION",
    "SupervisorContractCatalog",
    "SupervisorContractExtractorError",
    "SwissKnifeCapabilityRecord",
    "MutationPolicyContract",
    "assert_no_generic_proxy_selection",
    "build_native_operation_identity",
    "build_supervisor_contract_catalog",
    "extract_goal_completion_contract",
    "extract_mutation_policy_contracts",
    "extract_native_operations",
    "extract_swissknife_capabilities",
    "is_generic_proxy_reason",
    "load_supervisor_contract_catalog",
    "map_all_capabilities",
    "map_swissknife_capability",
    "native_function_identity",
    "native_tool_name",
    "parse_native_tool_name",
    "resolve_task_control_action",
    "select_native_identity",
    "summarize_catalog",
    "write_supervisor_contract_catalog",
]
