#!/usr/bin/env python3
"""UNI-166 deterministic list-action parity tests for native vector store tools."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.vector_store_tools.native_vector_store_tools import (
    register_native_vector_store_tools,
    vector_index,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI166VectorStoreTools(unittest.TestCase):
    def test_registration_schema_includes_list_action_and_optional_index_name(self) -> None:
        manager = _DummyManager()
        register_native_vector_store_tools(manager)
        schemas = {call["name"]: call["input_schema"] for call in manager.calls}

        index_schema = schemas["vector_index"]
        action_schema = index_schema["properties"]["action"]
        index_name_schema = index_schema["properties"]["index_name"]

        self.assertIn("list", action_schema.get("enum", []))
        self.assertEqual(index_name_schema.get("type"), ["string", "null"])
        self.assertEqual(index_schema.get("required"), ["action"])

    def test_vector_index_requires_index_name_for_non_list_actions(self) -> None:
        async def _run() -> None:
            result = await vector_index(action="create", index_name=None)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("index_name must be provided", str(result.get("message", "")))

        anyio.run(_run)

    def test_vector_index_list_action_allows_missing_index_name(self) -> None:
        async def _run() -> None:
            result = await vector_index(action="list", index_name=None)
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("action"), "list")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
