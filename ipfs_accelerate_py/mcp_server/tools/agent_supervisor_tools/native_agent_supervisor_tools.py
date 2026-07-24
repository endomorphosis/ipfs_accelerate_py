"""Policy-controlled MCP adapters for the agent-supervisor control service.

Registration is deliberately static and side-effect free.  A control service
is resolved only when a tool is invoked; listing categories, tools, or schemas
does not inspect a repository, initialize an optional provider, or start a
supervisor process.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from threading import RLock
from typing import Any

from ....agent_supervisor.control_contracts import (
    Operation,
    OperationRequest,
    operation_request_json_schema,
    operation_result_json_schema,
)
from ....agent_supervisor.control_plane import SupervisorControlService


AGENT_SUPERVISOR_MCP_CATEGORY = "agent_supervisor"
AGENT_SUPERVISOR_REPOSITORY_ALLOWLIST_ENV = (
    "IPFS_ACCELERATE_AGENT_REPOSITORY_ALLOWLIST"
)
AGENT_SUPERVISOR_STATE_ALLOWLIST_ENV = "IPFS_ACCELERATE_AGENT_STATE_ALLOWLIST"

ServiceFactory = Callable[[OperationRequest], SupervisorControlService]

_configuration_lock = RLock()
_configured_service: SupervisorControlService | None = None
_configured_factory: ServiceFactory | None = None


class AgentSupervisorMCPConfigurationError(RuntimeError):
    """Raised when MCP control policy has not supplied explicit allowlists."""


def configure_agent_supervisor_control(
    *,
    service: SupervisorControlService | None = None,
    service_factory: ServiceFactory | None = None,
) -> None:
    """Configure the service used by later tool invocations.

    Passing neither argument resets the adapter to its environment-backed,
    fail-closed configuration.  Supplying both is rejected so there is only
    one authority source.
    """

    if service is not None and service_factory is not None:
        raise ValueError("supply service or service_factory, not both")
    if service is not None and not isinstance(service, SupervisorControlService):
        raise TypeError("service must be a SupervisorControlService")
    if service_factory is not None and not callable(service_factory):
        raise TypeError("service_factory must be callable")
    global _configured_service, _configured_factory
    with _configuration_lock:
        _configured_service = service
        _configured_factory = service_factory


def _environment_allowlist(name: str) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in str(os.environ.get(name, "")).split(os.pathsep)
        if item.strip()
    )


def _environment_service(_request: OperationRequest) -> SupervisorControlService:
    repositories = _environment_allowlist(
        AGENT_SUPERVISOR_REPOSITORY_ALLOWLIST_ENV
    )
    states = _environment_allowlist(AGENT_SUPERVISOR_STATE_ALLOWLIST_ENV)
    if not repositories or not states:
        raise AgentSupervisorMCPConfigurationError(
            "agent-supervisor MCP tools require server-configured repository "
            "and state allowlists"
        )
    return SupervisorControlService(
        repository_allowlist=repositories,
        state_allowlist=states,
    )


def _resolve_service(request: OperationRequest) -> SupervisorControlService:
    with _configuration_lock:
        service = _configured_service
        factory = _configured_factory
    selected = service or (factory or _environment_service)(request)
    if not isinstance(selected, SupervisorControlService):
        raise AgentSupervisorMCPConfigurationError(
            "agent-supervisor service factory returned an invalid service"
        )
    return selected


async def execute_agent_supervisor_operation(
    request: Mapping[str, Any],
    operation: Operation | str,
) -> dict[str, Any]:
    """Decode, dispatch, and return the canonical shared result record."""

    selected = operation if isinstance(operation, Operation) else Operation(operation)
    decoded = OperationRequest.from_dict(request)
    if decoded.operation is not selected:
        raise ValueError(
            "request operation does not match the selected MCP tool"
        )
    service = _resolve_service(decoded)
    return service.execute(decoded).to_record()


async def agent_supervisor_control(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Generic canonical adapter, useful for direct embedding and tests."""

    decoded = OperationRequest.from_dict(request)
    return _resolve_service(decoded).execute(decoded).to_record()


def _operation_tool(operation: Operation) -> Callable[..., Any]:
    async def tool(request: Mapping[str, Any]) -> dict[str, Any]:
        return await execute_agent_supervisor_operation(request, operation)

    tool.__name__ = f"agent_supervisor_{operation.value}"
    tool.__qualname__ = tool.__name__
    tool.__doc__ = (
        f"Execute the canonical agent-supervisor {operation.value} operation."
    )
    return tool


AGENT_SUPERVISOR_OPERATION_TOOLS: dict[Operation, Callable[..., Any]] = {
    operation: _operation_tool(operation)
    for operation in sorted(Operation, key=lambda item: item.value)
}
for _operation, _tool in AGENT_SUPERVISOR_OPERATION_TOOLS.items():
    globals()[_tool.__name__] = _tool


def _tool_input_schema(operation: Operation) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "request": operation_request_json_schema(operation),
        },
        "required": ["request"],
        "additionalProperties": False,
        "x-output-schema": operation_result_json_schema(operation),
    }


def register_native_agent_supervisor_tools(manager: Any) -> None:
    """Register all closed-vocabulary operations without resolving a service."""

    for operation, tool in AGENT_SUPERVISOR_OPERATION_TOOLS.items():
        manager.register_tool(
            category=AGENT_SUPERVISOR_MCP_CATEGORY,
            name=operation.value,
            func=tool,
            description=(
                f"Execute agent-supervisor {operation.value} through the "
                "shared typed control service."
            ),
            input_schema=_tool_input_schema(operation),
            runtime="fastapi",
            tags=[
                "native",
                "agent-supervisor",
                operation.authority.value,
                "policy-controlled",
            ],
        )


__all__ = [
    "AGENT_SUPERVISOR_MCP_CATEGORY",
    "AGENT_SUPERVISOR_OPERATION_TOOLS",
    "AGENT_SUPERVISOR_REPOSITORY_ALLOWLIST_ENV",
    "AGENT_SUPERVISOR_STATE_ALLOWLIST_ENV",
    "AgentSupervisorMCPConfigurationError",
    "agent_supervisor_control",
    "configure_agent_supervisor_control",
    "execute_agent_supervisor_operation",
    "register_native_agent_supervisor_tools",
    *[
        f"agent_supervisor_{operation.value}"
        for operation in sorted(Operation, key=lambda item: item.value)
    ],
]
