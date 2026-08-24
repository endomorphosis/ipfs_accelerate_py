"""MCP adapters for admitted causal-federation control.

This compatibility-path module deliberately exposes only the canonical
post-admission command catalog.  Tool input is decoded before service
resolution; no MCP payload can choose a database, shell command, state-owner
implementation, or fallback mode.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from threading import RLock
from types import MappingProxyType
from typing import Any, Final

from ...agent_supervisor.control.service import execute_federation_command
from ...agent_supervisor.federation.cli import federation_control_response_record
from ...agent_supervisor.federation.contracts import (
    FederationCommand,
    FederationContractError,
    FederationOperation,
)
from ...agent_supervisor.federation.control_service import (
    POST_ADMISSION_OPERATIONS,
    FederationControlResponse,
    FederationControlService,
    FederationControlServiceError,
)

FEDERATION_CONTROL_MCP_CATEGORY: Final[str] = "agent_supervisor"
FEDERATION_CONTROL_MCP_INTERFACE: Final[str] = "FederationControlMCP@1"
FEDERATION_CONTROL_MCP_DISPATCH_MODE: Final[str] = "direct_typed_service"
FEDERATION_CONTROL_MCP_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/mcp-result@1"
)

ServiceFactory = Callable[[FederationCommand], FederationControlService]
_configuration_lock = RLock()
_configured_service: FederationControlService | None = None
_configured_factory: ServiceFactory | None = None
_service_resolution_count = 0


class FederationControlMCPConfigurationError(RuntimeError):
    """The MCP process has no qualified state-owner control service."""


def configure_federation_control(
    *,
    service: FederationControlService | None = None,
    service_factory: ServiceFactory | None = None,
) -> None:
    """Set the sole injected live-control authority, or reset it to absent."""

    if service is not None and service_factory is not None:
        raise ValueError("supply service or service_factory, not both")
    if service is not None and not isinstance(service, FederationControlService):
        raise TypeError("service must be a FederationControlService")
    if service_factory is not None and not callable(service_factory):
        raise TypeError("service_factory must be callable")
    global _configured_service, _configured_factory
    with _configuration_lock:
        _configured_service = service
        _configured_factory = service_factory


def _resolve_service(command: FederationCommand) -> FederationControlService:
    global _service_resolution_count
    with _configuration_lock:
        _service_resolution_count += 1
        service = _configured_service
        factory = _configured_factory
    selected = service if service is not None else factory(command) if factory else None
    if not isinstance(selected, FederationControlService):
        raise FederationControlMCPConfigurationError(
            "federation MCP tools require a configured qualified FederationControlService"
        )
    return selected


def _decode_command(
    request: Mapping[str, Any],
    operation: FederationOperation | str,
) -> FederationCommand:
    if not isinstance(request, Mapping):
        raise FederationContractError("MCP command request must be an object")
    selected = (
        operation if isinstance(operation, FederationOperation) else FederationOperation(operation)
    )
    if selected not in POST_ADMISSION_OPERATIONS:
        raise FederationControlMCPConfigurationError(
            "federation.create is accepted only by the authenticated trigger gateway"
        )
    command = FederationCommand.from_dict(request)
    if command.operation is not selected:
        raise FederationContractError("request operation does not match the selected MCP tool")
    if command.operation not in POST_ADMISSION_OPERATIONS:
        raise FederationContractError("federation.create is accepted only by the trigger gateway")
    return command


def execute_federation_control(
    request: Mapping[str, Any],
    operation: FederationOperation | str,
) -> dict[str, Any]:
    """Decode and dispatch one bounded command without transport adaptation."""

    command = _decode_command(request, operation)
    response = execute_federation_command(_resolve_service(command), command)
    if not isinstance(response, FederationControlResponse):
        raise FederationControlServiceError("control service returned no typed response")
    record = federation_control_response_record(command, response)
    record["schema"] = FEDERATION_CONTROL_MCP_RESULT_SCHEMA
    record["interface"] = FEDERATION_CONTROL_MCP_INTERFACE
    return record


def federation_control(request: Mapping[str, Any]) -> dict[str, Any]:
    """Generic direct MCP adapter; the command itself selects its closed operation."""

    command = FederationCommand.from_dict(request)
    return execute_federation_control(command.to_dict(), command.operation)


def _tool_name(operation: FederationOperation) -> str:
    return "federation_" + operation.value.removeprefix("federation.")


def _operation_tool(
    operation: FederationOperation,
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    def tool(request: Mapping[str, Any]) -> dict[str, Any]:
        return execute_federation_control(request, operation)

    tool.__name__ = _tool_name(operation)
    tool.__qualname__ = tool.__name__
    tool.__doc__ = f"Execute the canonical {operation.value} federation command."
    tool.__federation_operation__ = operation  # type: ignore[attr-defined]
    tool.__federation_dispatch_mode__ = FEDERATION_CONTROL_MCP_DISPATCH_MODE  # type: ignore[attr-defined]
    tool.__federation_executor__ = execute_federation_control  # type: ignore[attr-defined]
    return tool


FEDERATION_CONTROL_OPERATION_TOOLS: Mapping[
    FederationOperation, Callable[[Mapping[str, Any]], dict[str, Any]]
] = MappingProxyType(
    {
        operation: _operation_tool(operation)
        for operation in sorted(POST_ADMISSION_OPERATIONS, key=lambda item: item.value)
    }
)
for _operation, _tool in FEDERATION_CONTROL_OPERATION_TOOLS.items():
    globals()[_tool.__name__] = _tool


def federation_control_mcp_discovery_manifest() -> dict[str, Any]:
    """Return static tool contracts without resolving process-local authority."""

    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/causal-federation/mcp-discovery@1",
        "interface": FEDERATION_CONTROL_MCP_INTERFACE,
        "category": FEDERATION_CONTROL_MCP_CATEGORY,
        "dispatch": FEDERATION_CONTROL_MCP_DISPATCH_MODE,
        "tools": {
            tool.__name__: operation.value
            for operation, tool in FEDERATION_CONTROL_OPERATION_TOOLS.items()
        },
        "request_schema": FederationCommand.SCHEMA,
        "result_schema": FEDERATION_CONTROL_MCP_RESULT_SCHEMA,
        "shell_out": False,
        "embedded_fallback": False,
        "create_via_trigger_gateway": True,
    }


def federation_control_service_resolution_count() -> int:
    """Return invocation-only service resolution count for cold-path tests."""

    with _configuration_lock:
        return _service_resolution_count


def register_federation_control_tools(mcp: Any) -> None:
    """Register exactly the closed post-admission tool catalog on an MCP host."""

    register = getattr(mcp, "register_tool", None)
    if not callable(register):
        raise TypeError("MCP host must provide register_tool")
    for operation, tool in FEDERATION_CONTROL_OPERATION_TOOLS.items():
        register(
            name=tool.__name__,
            function=tool,
            description=tool.__doc__,
            input_schema={
                "type": "object",
                "properties": {"request": {"type": "object"}},
                "required": ["request"],
                "additionalProperties": False,
                "x-federation-command-schema": FederationCommand.SCHEMA,
                "x-federation-operation": operation.value,
                "x-federation-result-schema": FEDERATION_CONTROL_MCP_RESULT_SCHEMA,
            },
            execution_context="server",
        )


# Familiar registration spelling for legacy MCP module discovery.
register_tools = register_federation_control_tools


__all__ = [
    "FEDERATION_CONTROL_MCP_CATEGORY",
    "FEDERATION_CONTROL_MCP_DISPATCH_MODE",
    "FEDERATION_CONTROL_MCP_INTERFACE",
    "FEDERATION_CONTROL_MCP_RESULT_SCHEMA",
    "FEDERATION_CONTROL_OPERATION_TOOLS",
    "FederationControlMCPConfigurationError",
    "configure_federation_control",
    "execute_federation_control",
    "federation_control",
    "federation_control_mcp_discovery_manifest",
    "federation_control_service_resolution_count",
    "register_federation_control_tools",
    "register_tools",
    *[tool.__name__ for tool in FEDERATION_CONTROL_OPERATION_TOOLS.values()],
]
