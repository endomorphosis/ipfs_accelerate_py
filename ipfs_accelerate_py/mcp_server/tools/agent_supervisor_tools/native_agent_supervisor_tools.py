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
from types import MappingProxyType
from typing import Any

from ....agent_supervisor.control_contracts import (
    ControlContractError,
    ControlDiscoveryManifest,
    ControlSurface,
    Operation,
    OperationCatalog,
    OperationResult,
    OperationRequest,
    decode_operation_request,
    get_operation_catalog,
    operation_request_json_schema,
    operation_result_json_schema,
)
from ....agent_supervisor.control_plane import (
    DIRECT_CONTROL_SERVICE_DISPATCHER_ID,
    ControlSurfacePublication,
    SupervisorControlService,
    control_operation_behavior_id,
    validate_control_surface_publication,
)
from ....agent_supervisor.formal_verification_contracts import content_identity


AGENT_SUPERVISOR_MCP_CATEGORY = "agent_supervisor"
AGENT_SUPERVISOR_REPOSITORY_ALLOWLIST_ENV = (
    "IPFS_ACCELERATE_AGENT_REPOSITORY_ALLOWLIST"
)
AGENT_SUPERVISOR_STATE_ALLOWLIST_ENV = "IPFS_ACCELERATE_AGENT_STATE_ALLOWLIST"
AGENT_SUPERVISOR_MCP_DISPATCH_MODE = "direct_service"

ServiceFactory = Callable[[OperationRequest], SupervisorControlService]

_configuration_lock = RLock()
_configured_service: SupervisorControlService | None = None
_configured_factory: ServiceFactory | None = None
_service_resolution_count = 0


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
    global _service_resolution_count
    with _configuration_lock:
        _service_resolution_count += 1
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
    # Decode before resolving server policy/service state. Unsafe mutation
    # payloads therefore cannot trigger a service factory or backend.
    decoded = decode_operation_request(request)
    if decoded.operation is not selected:
        raise ValueError(
            "request operation does not match the selected MCP tool"
        )
    service = _resolve_service(decoded)
    result = service.execute(decoded)
    if not isinstance(result, OperationResult):
        raise ControlContractError(
            "shared control service returned a non-canonical result"
        )
    result.validate_against(decoded)
    record = result.to_record()
    # Re-decode the transport record before publishing it.  This catches a
    # service override which returns an object whose serializer drifted from
    # the catalog even when the in-memory result itself looked well formed.
    canonical = OperationResult.from_dict(record)
    canonical.validate_against(decoded)
    if canonical.canonical_bytes() != result.canonical_bytes():
        raise ControlContractError(
            "shared control service result is not canonically stable"
        )
    return record


async def agent_supervisor_control(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Generic canonical adapter, useful for direct embedding and tests."""

    decoded = decode_operation_request(request)
    return await execute_agent_supervisor_operation(
        decoded.to_record(), decoded.operation
    )


def _operation_tool(operation: Operation) -> Callable[..., Any]:
    async def tool(request: Mapping[str, Any]) -> dict[str, Any]:
        return await execute_agent_supervisor_operation(request, operation)

    tool.__name__ = f"agent_supervisor_{operation.value}"
    tool.__qualname__ = tool.__name__
    tool.__doc__ = (
        f"Execute the canonical agent-supervisor {operation.value} operation."
    )
    # Publication validation uses immutable semantic markers instead of
    # trusting a callable's display name.  This proves that every generated
    # MCP endpoint is bound to the selected catalog operation and goes
    # directly to the shared Python control service.
    tool.__agent_supervisor_operation__ = operation  # type: ignore[attr-defined]
    tool.__agent_supervisor_dispatch_mode__ = (  # type: ignore[attr-defined]
        AGENT_SUPERVISOR_MCP_DISPATCH_MODE
    )
    tool.__agent_supervisor_executor__ = (  # type: ignore[attr-defined]
        execute_agent_supervisor_operation
    )
    return tool


AGENT_SUPERVISOR_OPERATION_TOOLS: Mapping[
    Operation, Callable[..., Any]
] = MappingProxyType(
    {
        operation: _operation_tool(operation)
        for operation in sorted(Operation, key=lambda item: item.value)
    }
)
for _operation, _tool in AGENT_SUPERVISOR_OPERATION_TOOLS.items():
    globals()[_tool.__name__] = _tool


def _tool_input_schema(operation: Operation) -> dict[str, Any]:
    request_schema = operation_request_json_schema(operation)
    result_schema = operation_result_json_schema(operation)
    return {
        "type": "object",
        "properties": {
            "request": request_schema,
        },
        "required": ["request"],
        "additionalProperties": False,
        "x-output-schema": result_schema,
        # These canonical identities let clients and completion analyzers prove
        # that discovery described the same transport-neutral schemas as the
        # Python and CLI surfaces without trusting a tool name or description.
        "x-agent-supervisor-contract": {
            "surface": ControlSurface.MCP.value,
            "operation": operation.value,
            "request_schema_id": content_identity(request_schema),
            "result_schema_id": content_identity(result_schema),
        },
    }


def _validated_mcp_operations(
    catalog: OperationCatalog | None = None,
) -> tuple[Operation, ...]:
    """Fail closed unless the complete MCP publication matches the catalog.

    Validation happens before registration starts, so a missing, extra,
    schema-drifted, or incorrectly dispatched tool cannot produce a partially
    published catalog.
    """

    selected_catalog = (
        get_operation_catalog() if catalog is None else catalog
    )
    if not isinstance(selected_catalog, OperationCatalog):
        raise ControlContractError(
            "MCP publication catalog must be an OperationCatalog"
        )
    expected = selected_catalog.operations
    actual_keys = tuple(AGENT_SUPERVISOR_OPERATION_TOOLS)
    actual_operations = tuple(
        item for item in actual_keys if isinstance(item, Operation)
    )
    missing = sorted(
        operation.value
        for operation in set(expected).difference(actual_operations)
    )
    extra = sorted(
        str(getattr(operation, "value", operation))
        for operation in set(actual_keys).difference(expected)
    )
    if (
        len(actual_operations) != len(actual_keys)
        or len(actual_keys) != len(expected)
        or missing
        or extra
    ):
        raise ControlContractError(
            "MCP publication does not exactly cover the operation catalog; "
            f"missing={missing}, extra={extra}"
        )

    seen_tools: set[int] = set()
    for operation in expected:
        descriptor = selected_catalog.operation(operation)
        tool = AGENT_SUPERVISOR_OPERATION_TOOLS[operation]
        if not callable(tool):
            raise ControlContractError(
                f"MCP tool for {operation.value} is not callable"
            )
        if id(tool) in seen_tools:
            raise ControlContractError(
                f"MCP tool for {operation.value} reuses another operation's "
                "callable"
            )
        seen_tools.add(id(tool))
        if (
            tool.__name__ != f"agent_supervisor_{operation.value}"
            or getattr(tool, "__agent_supervisor_operation__", None)
            is not operation
            or getattr(tool, "__agent_supervisor_dispatch_mode__", None)
            != AGENT_SUPERVISOR_MCP_DISPATCH_MODE
            or getattr(tool, "__agent_supervisor_executor__", None)
            is not execute_agent_supervisor_operation
        ):
            raise ControlContractError(
                f"MCP tool behavior drift for {operation.value}"
            )

        schema = _tool_input_schema(operation)
        request_schema = schema["properties"]["request"]
        result_schema = schema["x-output-schema"]
        if content_identity(request_schema) != descriptor.request_schema_id:
            raise ControlContractError(
                f"MCP request schema drift for {operation.value}"
            )
        if content_identity(result_schema) != descriptor.result_schema_id:
            raise ControlContractError(
                f"MCP result schema drift for {operation.value}"
            )
    return expected


def validate_agent_supervisor_mcp_catalog(
    catalog: OperationCatalog | None = None,
) -> ControlDiscoveryManifest:
    """Validate the static MCP catalog without resolving runtime state."""

    selected_catalog = (
        get_operation_catalog() if catalog is None else catalog
    )
    publication = mcp_control_surface_publication(selected_catalog)
    return ControlDiscoveryManifest(
        surface=ControlSurface.MCP,
        operations=publication.operations,
    )


def mcp_control_surface_publication(
    catalog: OperationCatalog | None = None,
) -> ControlSurfacePublication:
    """Return the validated, side-effect-free MCP surface publication."""

    selected_catalog = (
        get_operation_catalog() if catalog is None else catalog
    )
    operations = _validated_mcp_operations(selected_catalog)
    publication = ControlSurfacePublication(
        surface=ControlSurface.MCP,
        catalog_id=selected_catalog.catalog_id,
        operations=operations,
        request_schema_ids={
            descriptor.operation: descriptor.request_schema_id
            for descriptor in selected_catalog
        },
        result_schema_ids={
            descriptor.operation: descriptor.result_schema_id
            for descriptor in selected_catalog
        },
        behavior_ids={
            descriptor.operation: control_operation_behavior_id(descriptor)
            for descriptor in selected_catalog
        },
        dispatcher_ids={
            operation: DIRECT_CONTROL_SERVICE_DISPATCHER_ID
            for operation in operations
        },
        dispatch_mode=AGENT_SUPERVISOR_MCP_DISPATCH_MODE,
        catalog_version=selected_catalog.catalog_version,
        provider_free=True,
        process_free=True,
    )
    return validate_control_surface_publication(
        publication, catalog=selected_catalog
    )


def agent_supervisor_service_resolution_count() -> int:
    """Return the cumulative invocation-only service resolution count."""

    with _configuration_lock:
        return _service_resolution_count


def agent_supervisor_discovery_manifest() -> ControlDiscoveryManifest:
    """Validate and return static MCP discovery metadata.

    This path examines only already-constructed callables and canonical schema
    dictionaries.  It deliberately does not resolve environment policy or a
    control service.
    """

    manifest = validate_agent_supervisor_mcp_catalog()
    operations = manifest.operations
    for operation in operations:
        schema = _tool_input_schema(operation)
        request_schema = schema["properties"]["request"]
        result_schema = schema["x-output-schema"]
        if content_identity(request_schema) != manifest.request_schema_ids[
            operation.value
        ]:
            raise ControlContractError(
                f"MCP request schema drift for {operation.value}"
            )
        if content_identity(result_schema) != manifest.result_schema_ids[
            operation.value
        ]:
            raise ControlContractError(
                f"MCP result schema drift for {operation.value}"
            )
    return manifest


def register_native_agent_supervisor_tools(manager: Any) -> None:
    """Register all closed-vocabulary operations without resolving a service."""

    operations = mcp_control_surface_publication().operations
    definitions: list[dict[str, Any]] = []
    for operation in operations:
        tool = AGENT_SUPERVISOR_OPERATION_TOOLS[operation]
        tags = [
            "native",
            "agent-supervisor",
            operation.authority.value,
            "policy-controlled",
            "bounded",
            "redacted",
        ]
        if operation.mutating:
            tags.extend(
                [
                    "authorization-required",
                    "audit-receipt",
                    "dry-run",
                    "idempotent",
                    "lease-fenced",
                ]
            )
        definitions.append(
            {
                "category": AGENT_SUPERVISOR_MCP_CATEGORY,
                "name": operation.value,
                "func": tool,
                "description": (
                    f"Execute agent-supervisor {operation.value} through the "
                    "shared typed control service."
                ),
                "input_schema": _tool_input_schema(operation),
                "runtime": "fastapi",
                "tags": tags,
            }
        )
    for definition in definitions:
        manager.register_tool(**definition)


# Stable generation-2 spellings are aliases, not wrappers.  Keeping the
# callable objects identical preserves the reviewed discovery and publication
# behavior (including their operation/schema identities) while retaining the
# original names for generation-1 callers.
agent_supervisor_v2_discovery_manifest = agent_supervisor_discovery_manifest
mcp_v2_control_surface_publication = mcp_control_surface_publication


__all__ = [
    "AGENT_SUPERVISOR_MCP_CATEGORY",
    "AGENT_SUPERVISOR_MCP_DISPATCH_MODE",
    "AGENT_SUPERVISOR_OPERATION_TOOLS",
    "AGENT_SUPERVISOR_REPOSITORY_ALLOWLIST_ENV",
    "AGENT_SUPERVISOR_STATE_ALLOWLIST_ENV",
    "AgentSupervisorMCPConfigurationError",
    "agent_supervisor_discovery_manifest",
    "agent_supervisor_v2_discovery_manifest",
    "agent_supervisor_service_resolution_count",
    "agent_supervisor_control",
    "configure_agent_supervisor_control",
    "execute_agent_supervisor_operation",
    "mcp_control_surface_publication",
    "mcp_v2_control_surface_publication",
    "register_native_agent_supervisor_tools",
    "validate_agent_supervisor_mcp_catalog",
    *[tool.__name__ for tool in AGENT_SUPERVISOR_OPERATION_TOOLS.values()],
]
