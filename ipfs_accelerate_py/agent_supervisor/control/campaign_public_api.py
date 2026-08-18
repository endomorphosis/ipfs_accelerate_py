"""Transport-neutral O2 operational campaign public API.

Discovery is import-isolation critical: this module must not start daemons,
open network connections, load providers, or import campaign execution.
Python, CLI, and MCP share one closed operation map over the existing
control catalog.  Start/resume reuse ``Operation.START`` / ``Operation.RESUME``
and never expand ``Operation``.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

from .control_contracts import (
    MUTATION_OPERATIONS,
    PROPOSAL_OPERATIONS,
    READ_OPERATIONS,
    ControlSurface,
    Operation,
    OperationAuthority,
    get_operation_catalog,
)


OPERATIONAL_CAMPAIGN_API_INTERFACE: Final = "OperationalCampaignAPI@1"
OPERATIONAL_CAMPAIGN_API_SCHEMA_VERSION: Final = "pgir-operational-campaign-api/v1"
OPERATIONAL_CAMPAIGN_API_VERSION: Final = "1.0.0"
OPERATIONAL_CAMPAIGN_API_REQUIREMENT_ID: Final = (
    "requirement:agent-supervisor/operational-campaign-public-api@1"
)

OPERATIONAL_CAMPAIGN_OPERATION_NAMES: Final[tuple[str, ...]] = (
    "create",
    "plan",
    "start",
    "resume",
    "status",
    "steer",
    "refill",
    "proof-replay",
    "compare",
    "promote",
    "reject",
    "report",
)

# Closed campaign-to-control map.  Values must already exist on Operation.
_OPERATIONAL_CAMPAIGN_CONTROL_MAP: Final[dict[str, Operation]] = {
    "create": Operation.WORKFLOW_MATERIALIZE,
    "plan": Operation.PLAN,
    "start": Operation.START,
    "resume": Operation.RESUME,
    "status": Operation.STATUS,
    "steer": Operation.OBJECTIVE_REFINE,
    "refill": Operation.BACKLOG_REFILL,
    "proof-replay": Operation.VALIDATION_REPLAY,
    "compare": Operation.RECEIPTS,
    "promote": Operation.OBJECTIVE_RECONCILE,
    "reject": Operation.QUARANTINE,
    "report": Operation.METRICS,
}
OPERATIONAL_CAMPAIGN_CONTROL_MAP: Final[Mapping[str, Operation]] = MappingProxyType(
    _OPERATIONAL_CAMPAIGN_CONTROL_MAP
)

OPERATIONAL_CAMPAIGN_CLI_COMMANDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "create": "workflow-create",
        "plan": "plan",
        "start": "start",
        "resume": "resume",
        "status": "status",
        "steer": "refine",
        "refill": "refill",
        "proof-replay": "validation-replay",
        "compare": "receipts",
        "promote": "reconcile",
        "reject": "quarantine",
        "report": "metrics",
    }
)

_LEASE_REQUIRING: Final[frozenset[str]] = frozenset(
    {
        "create",
        "start",
        "resume",
        "steer",
        "refill",
        "proof-replay",
        "promote",
        "reject",
    }
)

_PROMPT_SELECTABLE: Final[frozenset[str]] = frozenset(
    {"plan", "status", "compare", "report"}
)


def _authority_for(operation: Operation) -> OperationAuthority:
    if operation in READ_OPERATIONS:
        return OperationAuthority.READ
    if operation in PROPOSAL_OPERATIONS:
        return OperationAuthority.PROPOSAL
    if operation in MUTATION_OPERATIONS:
        return OperationAuthority.MUTATION
    return operation.authority


def operational_campaign_control_operation(name: str) -> Operation:
    """Return the existing control operation for one O2 campaign verb."""

    key = str(name or "").strip()
    try:
        return OPERATIONAL_CAMPAIGN_CONTROL_MAP[key]
    except KeyError as exc:
        raise ValueError("unknown operational campaign operation %r" % name) from exc


def prompt_may_select_campaign_operation(name: str) -> bool:
    """True only for read/proposal verbs. Prompts cannot pick mutation authority."""

    return str(name or "").strip() in _PROMPT_SELECTABLE


def discover_operational_campaign_api() -> dict[str, Any]:
    """Return the frozen O2 catalog without constructing a service or campaign."""

    catalog = get_operation_catalog()
    operations: dict[str, dict[str, Any]] = {}
    for name in OPERATIONAL_CAMPAIGN_OPERATION_NAMES:
        control = OPERATIONAL_CAMPAIGN_CONTROL_MAP[name]
        descriptor = catalog.operation(control)
        operations[name] = {
            "name": name,
            "control_operation": control.value,
            "authority": _authority_for(control).value,
            "requires_lease": name in _LEASE_REQUIRING,
            "prompt_selectable": name in _PROMPT_SELECTABLE,
            "cli_command": OPERATIONAL_CAMPAIGN_CLI_COMMANDS[name],
            "mcp_tool": control.value,
            "request_schema_id": descriptor.request_schema_id,
            "result_schema_id": descriptor.result_schema_id,
            "in_read_catalog": control in READ_OPERATIONS,
            "in_proposal_catalog": control in PROPOSAL_OPERATIONS,
            "in_mutation_catalog": control in MUTATION_OPERATIONS,
        }
    mapped = {item["control_operation"] for item in operations.values()}
    return {
        "schema": OPERATIONAL_CAMPAIGN_API_INTERFACE,
        "schema_version": OPERATIONAL_CAMPAIGN_API_SCHEMA_VERSION,
        "api_version": OPERATIONAL_CAMPAIGN_API_VERSION,
        "requirement_id": OPERATIONAL_CAMPAIGN_API_REQUIREMENT_ID,
        "operations": operations,
        "operation_names": list(OPERATIONAL_CAMPAIGN_OPERATION_NAMES),
        "surfaces": [item.value for item in ControlSurface],
        "expands_control_catalog": False,
        "import_side_effects": "none",
        "prompt_selected_authority": False,
        "mapped_control_operations": sorted(mapped),
        "catalog_id": catalog.content_id,
    }


def assert_operational_campaign_control_parity() -> None:
    """Fail closed if the O2 map would invent or re-rank a control operation."""

    catalog_ops = set(get_operation_catalog().operations)
    if set(OPERATIONAL_CAMPAIGN_OPERATION_NAMES) != set(OPERATIONAL_CAMPAIGN_CONTROL_MAP):
        raise ValueError("operational campaign catalog names drifted from the control map")
    if set(OPERATIONAL_CAMPAIGN_CLI_COMMANDS) != set(OPERATIONAL_CAMPAIGN_OPERATION_NAMES):
        raise ValueError("operational campaign CLI command map is incomplete")
    for name, operation in OPERATIONAL_CAMPAIGN_CONTROL_MAP.items():
        if operation not in catalog_ops:
            raise ValueError(
                "operational campaign operation %s expands the control catalog" % name
            )
        if operation not in Operation:
            raise ValueError(
                "operational campaign operation %s is not a closed Operation" % name
            )


assert_operational_campaign_control_parity()


__all__ = (
    "OPERATIONAL_CAMPAIGN_API_INTERFACE",
    "OPERATIONAL_CAMPAIGN_API_REQUIREMENT_ID",
    "OPERATIONAL_CAMPAIGN_API_SCHEMA_VERSION",
    "OPERATIONAL_CAMPAIGN_API_VERSION",
    "OPERATIONAL_CAMPAIGN_CLI_COMMANDS",
    "OPERATIONAL_CAMPAIGN_CONTROL_MAP",
    "OPERATIONAL_CAMPAIGN_OPERATION_NAMES",
    "assert_operational_campaign_control_parity",
    "discover_operational_campaign_api",
    "operational_campaign_control_operation",
    "prompt_may_select_campaign_operation",
)
