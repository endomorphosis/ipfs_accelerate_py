"""Native MCP-IDL tools for unified mcp_server runtime."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from ipfs_accelerate_py.mcp_server.mcplusplus.idl_registry import (
    InterfaceDescriptorRegistry,
    build_descriptor,
)


def _make_default_registry(supported_capabilities: Iterable[str]) -> InterfaceDescriptorRegistry:
    """Create a registry seeded with baseline MCP-IDL interface descriptors."""
    registry = InterfaceDescriptorRegistry(supported_capabilities=supported_capabilities)

    registry.register_descriptor(
        build_descriptor(
            name="interfaces",
            namespace="ipfs_accelerate_py.mcp_server",
            version="1.0.0",
            methods=[
                {"name": "interfaces/list", "input_schema": {"type": "object", "properties": {}, "required": []}, "output_schema": {"type": "array", "items": {"type": "string"}}},
                {"name": "interfaces/get", "input_schema": {"type": "object", "properties": {"interface_cid": {"type": "string"}}, "required": ["interface_cid"]}, "output_schema": {"type": "object"}},
                {"name": "interfaces/compat", "input_schema": {"type": "object", "properties": {"interface_cid": {"type": "string"}}, "required": ["interface_cid"]}, "output_schema": {"type": "object"}},
                {"name": "interfaces/select", "input_schema": {"type": "object", "properties": {"task_hint_cid": {"type": "string"}, "budget": {"type": "integer", "default": 20}}, "required": []}, "output_schema": {"type": "array", "items": {"type": "string"}}},
            ],
            errors=[{"name": "NotFound"}, {"name": "ValidationError"}],
            requires=["mcp++/profile-a-idl"],
            semantic_tags=["idl", "repository", "compatibility"],
            observability={"trace": True, "provenance": True},
        )
    )
    return registry


def register_native_idl_tools(manager: Any, *, supported_capabilities: Iterable[str]) -> None:
    """Register `interfaces/*` MCP-IDL tools in the unified manager."""
    registry = _make_default_registry(supported_capabilities=supported_capabilities)

    # Expose registry for bootstrap/test introspection.
    setattr(manager, "_unified_idl_registry", registry)

    async def interfaces_list() -> Dict[str, Any]:
        interface_cids = registry.list_interfaces()
        return {"interface_cids": interface_cids, "count": len(interface_cids)}

    async def interfaces_get(interface_cid: str) -> Dict[str, Any]:
        descriptor = registry.get_descriptor(interface_cid)
        if descriptor is None:
            return {"found": False, "error": "interface_not_found", "interface_cid": interface_cid}
        return {"found": True, "interface_cid": interface_cid, "descriptor": descriptor}

    async def interfaces_compat(interface_cid: str) -> Dict[str, Any]:
        verdict = registry.compat(interface_cid)
        payload = verdict.to_dict()
        payload["interface_cid"] = interface_cid
        return payload

    async def interfaces_select(task_hint_cid: str = "", budget: int = 20) -> Dict[str, Any]:
        selected = registry.select(task_hint_cid=task_hint_cid, budget=budget)
        return {
            "task_hint_cid": task_hint_cid,
            "budget": int(budget),
            "selected_interface_cids": selected,
            "count": len(selected),
        }

    manager.register_tool(
        category="idl",
        name="interfaces_list",
        func=interfaces_list,
        description="List known interface descriptors by `interface_cid`.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcp-idl", "profile-a"],
    )
    manager.register_tool(
        category="idl",
        name="interfaces_get",
        func=interfaces_get,
        description="Fetch descriptor payload for one `interface_cid`.",
        input_schema={
            "type": "object",
            "properties": {"interface_cid": {"type": "string"}},
            "required": ["interface_cid"],
        },
        runtime="fastapi",
        tags=["native", "mcp-idl", "profile-a"],
    )
    manager.register_tool(
        category="idl",
        name="interfaces_compat",
        func=interfaces_compat,
        description="Evaluate local compatibility for one `interface_cid`.",
        input_schema={
            "type": "object",
            "properties": {"interface_cid": {"type": "string"}},
            "required": ["interface_cid"],
        },
        runtime="fastapi",
        tags=["native", "mcp-idl", "profile-a"],
    )
    manager.register_tool(
        category="idl",
        name="interfaces_select",
        func=interfaces_select,
        description="Return budgeted interface subset for a task hint CID.",
        input_schema={
            "type": "object",
            "properties": {
                "task_hint_cid": {"type": "string", "default": ""},
                "budget": {"type": "integer", "default": 20},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcp-idl", "profile-a"],
    )


def load_idl_tools(manager: Any, *, supported_capabilities: Iterable[str]) -> None:
    """Category loader entrypoint for MCP-IDL tools."""
    register_native_idl_tools(manager, supported_capabilities=supported_capabilities)


__all__ = [
    "load_idl_tools",
    "register_native_idl_tools",
]
