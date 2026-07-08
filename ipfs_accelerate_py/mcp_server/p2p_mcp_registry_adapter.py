"""Canonical P2P MCP registry adapter for tool-manifest compatibility."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Set

RUNTIME_FASTAPI = "fastapi"
RUNTIME_TRIO = "trio"
RUNTIME_UNKNOWN = "unknown"


class P2PMCPRegistryAdapter:
    """Expose host tools in the dict shape expected by `tool_manifest`."""

    def __init__(
        self,
        host_server: Any,
        default_runtime: str = RUNTIME_FASTAPI,
        enable_runtime_detection: bool = True,
    ) -> None:
        self._host = host_server
        self._default_runtime = str(default_runtime or RUNTIME_FASTAPI)
        self._enable_runtime_detection = bool(enable_runtime_detection)

        self._runtime_metadata: Dict[str, str] = {}
        self._trio_tools: Set[str] = set()
        self._fastapi_tools: Set[str] = set()

    @property
    def accelerate_instance(self) -> Any:
        return self._host

    @property
    def tools(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        host_tools = getattr(self._host, "tools", None)
        if not isinstance(host_tools, dict):
            return out

        for name, entry in host_tools.items():
            fn = None
            description = ""
            input_schema: Dict[str, Any] = {}

            if callable(entry):
                fn = entry
                description = str(getattr(entry, "__doc__", "") or "")
            elif isinstance(entry, dict):
                fn = entry.get("function") if callable(entry.get("function")) else None
                description = str(entry.get("description", "") or "")
                schema = entry.get("input_schema")
                input_schema = schema if isinstance(schema, dict) else {}

            if not callable(fn):
                continue

            runtime = self._get_tool_runtime(str(name), fn)
            out[str(name)] = {
                "function": fn,
                "description": description,
                "input_schema": input_schema,
                "runtime": runtime,
                "runtime_metadata": {
                    "is_async": self._is_async_function(fn),
                    "is_trio_native": runtime == RUNTIME_TRIO,
                    "requires_trio_context": runtime == RUNTIME_TRIO,
                },
            }
        return out

    def _get_tool_runtime(self, name: str, fn: Any) -> str:
        if name in self._runtime_metadata:
            return self._runtime_metadata[name]
        if name in self._trio_tools:
            runtime = RUNTIME_TRIO
        elif name in self._fastapi_tools:
            runtime = RUNTIME_FASTAPI
        elif self._enable_runtime_detection:
            runtime = self._detect_runtime(fn)
        else:
            runtime = self._default_runtime
        self._runtime_metadata[name] = runtime
        return runtime

    def _detect_runtime(self, fn: Any) -> str:
        marker = getattr(fn, "_mcp_runtime", None)
        if marker in {RUNTIME_FASTAPI, RUNTIME_TRIO}:
            return marker
        module = str(getattr(fn, "__module__", "") or "").lower()
        if "trio" in module or "mcplusplus" in module:
            return RUNTIME_TRIO
        return RUNTIME_FASTAPI

    def _is_async_function(self, fn: Any) -> bool:
        try:
            return inspect.iscoroutinefunction(fn)
        except Exception:
            return False

    async def validate_p2p_message(self, msg: dict) -> bool:
        fn = getattr(self._host, "validate_p2p_message", None)
        if not callable(fn):
            return False
        result = fn(msg)
        if hasattr(result, "__await__"):
            return bool(await result)
        return bool(result)

    def register_trio_tool(self, name: str) -> None:
        key = str(name)
        self._trio_tools.add(key)
        self._runtime_metadata[key] = RUNTIME_TRIO
        self._fastapi_tools.discard(key)

    def register_fastapi_tool(self, name: str) -> None:
        key = str(name)
        self._fastapi_tools.add(key)
        self._runtime_metadata[key] = RUNTIME_FASTAPI
        self._trio_tools.discard(key)

    def get_tools_by_runtime(self, runtime: str) -> Dict[str, Dict[str, Any]]:
        return {name: item for name, item in self.tools.items() if item.get("runtime") == str(runtime)}

    def get_trio_tools(self) -> Dict[str, Dict[str, Any]]:
        return self.get_tools_by_runtime(RUNTIME_TRIO)

    def get_fastapi_tools(self) -> Dict[str, Dict[str, Any]]:
        return self.get_tools_by_runtime(RUNTIME_FASTAPI)

    def get_runtime_stats(self) -> Dict[str, Any]:
        all_tools = self.tools
        return {
            "total_tools": len(all_tools),
            "trio_tools": sum(1 for v in all_tools.values() if v.get("runtime") == RUNTIME_TRIO),
            "fastapi_tools": sum(1 for v in all_tools.values() if v.get("runtime") == RUNTIME_FASTAPI),
            "unknown_tools": sum(1 for v in all_tools.values() if v.get("runtime") == RUNTIME_UNKNOWN),
            "runtime_detection_enabled": self._enable_runtime_detection,
            "default_runtime": self._default_runtime,
        }

    def is_trio_tool(self, name: str) -> bool:
        key = str(name)
        return key in self._trio_tools or self._runtime_metadata.get(key) == RUNTIME_TRIO

    def clear_runtime_cache(self) -> None:
        self._runtime_metadata.clear()


__all__ = [
    "RUNTIME_FASTAPI",
    "RUNTIME_TRIO",
    "RUNTIME_UNKNOWN",
    "P2PMCPRegistryAdapter",
]
