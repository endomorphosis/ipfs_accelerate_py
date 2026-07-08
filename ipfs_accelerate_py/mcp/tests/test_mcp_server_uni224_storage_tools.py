#!/usr/bin/env python3
"""UNI-224 unavailable_reasons schema parity tests for storage backend status."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.storage_tools.native_storage_tools import (
    get_storage_backend_status,
    register_native_storage_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI224StorageTools(unittest.TestCase):
    def test_backend_status_schemas_enforce_unavailable_reasons_property_names(self) -> None:
        manager = _DummyManager()
        register_native_storage_tools(manager)

        by_name = {c["name"]: c for c in manager.calls}

        backend_schema = by_name["get_storage_backend_status"]["input_schema"]
        backend_props = backend_schema.get("properties", {})
        backend_unavailable = backend_props.get("unavailable_reasons") or {}
        self.assertEqual(
            (backend_unavailable.get("propertyNames") or {}).get("minLength"),
            1,
        )

        manage_schema = by_name["manage_collections"]["input_schema"]
        manage_props = manage_schema.get("properties", {})
        manage_unavailable = manage_props.get("unavailable_reasons") or {}
        self.assertEqual(
            (manage_unavailable.get("propertyNames") or {}).get("minLength"),
            1,
        )

    def test_get_storage_backend_status_rejects_empty_unavailable_reason_key(self) -> None:
        async def _run() -> None:
            result = await get_storage_backend_status(
                backend_types=["memory"],
                unavailable_reasons={"": "invalid"},
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn(
                "unavailable_reasons must be an object with non-empty string keys/values",
                str(result.get("error", "")),
            )

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
