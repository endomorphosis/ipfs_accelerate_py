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
                {
                    "name": "interfaces/list",
                    "input_schema": {"type": "object", "properties": {}, "required": []},
                    "output_schema": {"type": "array", "items": {"type": "string"}},
                },
                {
                    "name": "interfaces/get",
                    "input_schema": {
                        "type": "object",
                        "properties": {"interface_cid": {"type": "string"}},
                        "required": ["interface_cid"],
                    },
                    "output_schema": {"type": "object"},
                },
                {
                    "name": "interfaces/compat",
                    "input_schema": {
                        "type": "object",
                        "properties": {"interface_cid": {"type": "string"}},
                        "required": ["interface_cid"],
                    },
                    "output_schema": {"type": "object"},
                },
                {
                    "name": "interfaces/select",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "task_hint_cid": {"type": "string"},
                            "budget": {"type": "integer", "default": 20},
                        },
                        "required": [],
                    },
                    "output_schema": {"type": "array", "items": {"type": "string"}},
                },
            ],
            errors=[{"name": "NotFound"}, {"name": "ValidationError"}],
            requires=["mcp++/profile-a-idl"],
            semantic_tags=["idl", "repository", "compatibility"],
            observability={"trace": True, "provenance": True},
        )
    )
    return registry


def _capability_requirements_for_category(category: str) -> list[str]:
    """Return deterministic base capability requirements for a category descriptor."""
    req = ["mcp++/profile-a-idl"]
    if category == "p2p":
        req.append("mcp++/profile-e-mcp-p2p")
    return req


def _register_loaded_category_descriptors(registry: InterfaceDescriptorRegistry, manager: Any) -> int:
    """Register descriptors for categories already loaded in the manager.

    This intentionally avoids forcing lazy category loaders, and only captures
    categories currently present in the manager's in-memory registry.
    """

    categories = getattr(manager, "_categories", None)
    if not isinstance(categories, dict):
        return 0

    added = 0
    for category in sorted(str(k) for k in categories.keys()):
        if category == "idl":
            continue
        tools = categories.get(category)
        if not isinstance(tools, dict) or not tools:
            continue

        methods: List[Dict[str, Any]] = []
        for tool_name in sorted(str(n) for n in tools.keys()):
            schema: Dict[str, Any] = {}
            description = ""
            tool = tools.get(tool_name)
            if tool is not None:
                description = str(getattr(tool, "description", "") or "")
                maybe_schema = getattr(tool, "input_schema", None)
                if isinstance(maybe_schema, dict):
                    schema = maybe_schema
            methods.append(
                {
                    "name": f"{category}/{tool_name}",
                    "input_schema": schema or {"type": "object", "properties": {}, "required": []},
                    "output_schema": {"type": "object"},
                    "description": description,
                }
            )

        if not methods:
            continue

        registry.register_descriptor(
            build_descriptor(
                name=f"{category}_tools",
                namespace=f"ipfs_accelerate_py.mcp_server.{category}",
                version="1.0.0",
                methods=methods,
                requires=_capability_requirements_for_category(category),
                semantic_tags=["idl", "category", category],
                observability={"trace": True, "provenance": True},
            )
        )
        added += 1

    return added


def register_native_idl_tools(manager: Any, *, supported_capabilities: Iterable[str]) -> None:
    """Register `interfaces/*` MCP-IDL tools in the unified manager."""
    registry = _make_default_registry(supported_capabilities=supported_capabilities)
    _register_loaded_category_descriptors(registry, manager)

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
