"""MCP adapters for causal-federation control.

This compatibility-path module exposes the closed federation catalog through
the canonical hierarchical MCP manager.  CREATE resolves only an authenticated
gateway; post-admission commands resolve only a qualified control service.
Input is bounded and decoded before either authority is resolved, and no MCP
payload can choose a database, path, shell command, owner, or fallback mode.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from threading import RLock
from types import MappingProxyType
from typing import Any, Final

from ...agent_supervisor.control.service import execute_federation_command
from ...agent_supervisor.federation.cli import (
    FEDERATION_CONTROL_MAX_CANONICAL_BYTES,
    FEDERATION_CREATE_DISPATCH_MODE,
    FEDERATION_POST_ADMISSION_DISPATCH_MODE,
    FEDERATION_TRANSPORT_INVALID_CODE,
    FEDERATION_TRANSPORT_INVALID_MESSAGE,
    FEDERATION_TRANSPORT_UNAVAILABLE_CODE,
    FEDERATION_TRANSPORT_UNAVAILABLE_MESSAGE,
    FederationCreateTransport,
    decode_federation_control_request,
    federation_control_response_record,
    federation_create_response_record,
)
from ...agent_supervisor.federation.contracts import (
    FederationCommand,
    FederationOperation,
)
from ...agent_supervisor.federation.control_service import (
    FederationControlResponse,
    FederationControlService,
    FederationControlServiceError,
)
from ...agent_supervisor.federation.trigger import FederationControlGateway

FEDERATION_CONTROL_MCP_CATEGORY: Final[str] = "agent_supervisor"
FEDERATION_CONTROL_MCP_INTERFACE: Final[str] = "FederationControlMCP@1"
FEDERATION_CONTROL_MCP_DISPATCH_MODE: Final[str] = "direct_typed_authority"
FEDERATION_CONTROL_MCP_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/mcp-result@1"
)
FEDERATION_CONTROL_MCP_ERROR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/mcp-error@1"
)

ServiceFactory = Callable[[FederationCommand], FederationControlService]
GatewayFactory = Callable[[FederationCreateTransport], FederationControlGateway]
_configuration_lock = RLock()
_configured_service: FederationControlService | None = None
_configured_factory: ServiceFactory | None = None
_configured_gateway: FederationControlGateway | None = None
_configured_gateway_factory: GatewayFactory | None = None
_service_resolution_count = 0
_gateway_resolution_count = 0


class FederationControlMCPConfigurationError(RuntimeError):
    """The MCP process has no qualified state-owner control service."""


def configure_federation_control(
    *,
    service: FederationControlService | None = None,
    service_factory: ServiceFactory | None = None,
    gateway: FederationControlGateway | None = None,
    gateway_factory: GatewayFactory | None = None,
) -> None:
    """Set injected post-admission and CREATE authorities, or reset them."""

    if service is not None and service_factory is not None:
        raise ValueError("supply service or service_factory, not both")
    if gateway is not None and gateway_factory is not None:
        raise ValueError("supply gateway or gateway_factory, not both")
    if service is not None and not isinstance(service, FederationControlService):
        raise TypeError("service must be a FederationControlService")
    if service_factory is not None and not callable(service_factory):
        raise TypeError("service_factory must be callable")
    if gateway is not None and not isinstance(gateway, FederationControlGateway):
        raise TypeError("gateway must be a FederationControlGateway")
    if gateway_factory is not None and not callable(gateway_factory):
        raise TypeError("gateway_factory must be callable")
    global _configured_service, _configured_factory
    global _configured_gateway, _configured_gateway_factory
    with _configuration_lock:
        _configured_service = service
        _configured_factory = service_factory
        _configured_gateway = gateway
        _configured_gateway_factory = gateway_factory


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


def _resolve_gateway(
    transport: FederationCreateTransport,
) -> FederationControlGateway:
    global _gateway_resolution_count
    with _configuration_lock:
        _gateway_resolution_count += 1
        gateway = _configured_gateway
        factory = _configured_gateway_factory
    selected = gateway if gateway is not None else factory(transport) if factory else None
    if not isinstance(selected, FederationControlGateway):
        raise FederationControlMCPConfigurationError(
            "federation MCP CREATE requires a configured qualified gateway"
        )
    return selected


def _mcp_error_record(*, invalid: bool) -> dict[str, Any]:
    return {
        "schema": FEDERATION_CONTROL_MCP_ERROR_SCHEMA,
        "interface": FEDERATION_CONTROL_MCP_INTERFACE,
        "status": "invalid_request" if invalid else "unavailable",
        "code": (
            FEDERATION_TRANSPORT_INVALID_CODE
            if invalid
            else FEDERATION_TRANSPORT_UNAVAILABLE_CODE
        ),
        "message": (
            FEDERATION_TRANSPORT_INVALID_MESSAGE
            if invalid
            else FEDERATION_TRANSPORT_UNAVAILABLE_MESSAGE
        ),
    }


def execute_federation_control(
    request: Mapping[str, Any],
    operation: FederationOperation | str,
) -> dict[str, Any]:
    """Bound, decode, dispatch, and publish a redacted MCP transport record."""

    try:
        decoded = decode_federation_control_request(request, operation)
    except Exception:
        # Contract errors can contain caller-selected unknown key names or
        # secret/path-shaped values.  They never cross the MCP boundary.
        return _mcp_error_record(invalid=True)

    try:
        if isinstance(decoded, FederationCreateTransport):
            record = federation_create_response_record(
                decoded,
                _resolve_gateway(decoded).create(
                    decoded.request,
                    decoded.authentication,
                ),
            )
        else:
            response = execute_federation_command(_resolve_service(decoded), decoded)
            if not isinstance(response, FederationControlResponse):
                raise FederationControlServiceError(
                    "control service returned no typed response"
                )
            record = federation_control_response_record(decoded, response)
        record["schema"] = FEDERATION_CONTROL_MCP_RESULT_SCHEMA
        record["interface"] = FEDERATION_CONTROL_MCP_INTERFACE
        return record
    except Exception:
        # Factory, gateway, service, owner, and response-validation details are
        # internal.  Return one stable unavailable result without echoing them.
        return _mcp_error_record(invalid=False)


def federation_control(request: Mapping[str, Any]) -> dict[str, Any]:
    """Generic direct MCP adapter; the command itself selects its closed operation."""

    try:
        operation = (
            FederationOperation.CREATE
            if request.get("schema") == FederationCreateTransport.SCHEMA
            else FederationOperation(str(request.get("operation", "")))
        )
    except Exception:
        return _mcp_error_record(invalid=True)
    return execute_federation_control(request, operation)


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
    tool.__federation_dispatch_mode__ = (  # type: ignore[attr-defined]
        FEDERATION_CREATE_DISPATCH_MODE
        if operation is FederationOperation.CREATE
        else FEDERATION_POST_ADMISSION_DISPATCH_MODE
    )
    tool.__federation_executor__ = execute_federation_control  # type: ignore[attr-defined]
    return tool


FEDERATION_CONTROL_OPERATION_TOOLS: Mapping[
    FederationOperation, Callable[[Mapping[str, Any]], dict[str, Any]]
] = MappingProxyType(
    {
        operation: _operation_tool(operation)
        for operation in sorted(FederationOperation, key=lambda item: item.value)
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
        "dispatch": {
            FederationOperation.CREATE.value: FEDERATION_CREATE_DISPATCH_MODE,
            "post_admission": FEDERATION_POST_ADMISSION_DISPATCH_MODE,
        },
        "tools": {
            tool.__name__: operation.value
            for operation, tool in FEDERATION_CONTROL_OPERATION_TOOLS.items()
        },
        "request_schemas": {
            FederationOperation.CREATE.value: FederationCreateTransport.SCHEMA,
            "post_admission": FederationCommand.SCHEMA,
        },
        "result_schema": FEDERATION_CONTROL_MCP_RESULT_SCHEMA,
        "error_schema": FEDERATION_CONTROL_MCP_ERROR_SCHEMA,
        "shell_out": False,
        "embedded_fallback": False,
        "create_via_trigger_gateway": FederationControlGateway.__name__,
        "max_canonical_bytes": FEDERATION_CONTROL_MAX_CANONICAL_BYTES,
    }


def federation_control_service_resolution_count() -> int:
    """Return invocation-only service resolution count for cold-path tests."""

    with _configuration_lock:
        return _service_resolution_count


def federation_control_gateway_resolution_count() -> int:
    """Return invocation-only authenticated-gateway resolution count."""

    with _configuration_lock:
        return _gateway_resolution_count


def register_federation_control_tools(mcp: Any) -> None:
    """Register the closed catalog on the canonical hierarchical MCP manager."""

    register = getattr(mcp, "register_tool", None)
    if not callable(register):
        raise TypeError("MCP host must provide register_tool")
    for operation, tool in FEDERATION_CONTROL_OPERATION_TOOLS.items():
        register(
            category=FEDERATION_CONTROL_MCP_CATEGORY,
            name=tool.__name__,
            func=tool,
            description=tool.__doc__,
            input_schema={
                "type": "object",
                "properties": {"request": {"type": "object"}},
                "required": ["request"],
                "additionalProperties": False,
                "x-federation-request-schema": (
                    FederationCreateTransport.SCHEMA
                    if operation is FederationOperation.CREATE
                    else FederationCommand.SCHEMA
                ),
                "x-federation-operation": operation.value,
                "x-federation-result-schema": FEDERATION_CONTROL_MCP_RESULT_SCHEMA,
                "x-federation-error-schema": FEDERATION_CONTROL_MCP_ERROR_SCHEMA,
                "x-federation-max-canonical-bytes": (
                    FEDERATION_CONTROL_MAX_CANONICAL_BYTES
                ),
            },
            runtime="fastapi",
            tags=[
                "native",
                "agent-supervisor",
                "causal-federation",
                "bounded",
                "redacted",
                "authenticated-gateway"
                if operation is FederationOperation.CREATE
                else "typed-state-owner",
            ],
        )


# Familiar registration spelling for legacy MCP module discovery.
register_tools = register_federation_control_tools


__all__ = [
    "FEDERATION_CONTROL_MCP_CATEGORY",
    "FEDERATION_CONTROL_MCP_DISPATCH_MODE",
    "FEDERATION_CONTROL_MCP_ERROR_SCHEMA",
    "FEDERATION_CONTROL_MCP_INTERFACE",
    "FEDERATION_CONTROL_MCP_RESULT_SCHEMA",
    "FEDERATION_CONTROL_OPERATION_TOOLS",
    "FederationControlMCPConfigurationError",
    "configure_federation_control",
    "execute_federation_control",
    "federation_control",
    "federation_control_gateway_resolution_count",
    "federation_control_mcp_discovery_manifest",
    "federation_control_service_resolution_count",
    "register_federation_control_tools",
    "register_tools",
    *[tool.__name__ for tool in FEDERATION_CONTROL_OPERATION_TOOLS.values()],
]
