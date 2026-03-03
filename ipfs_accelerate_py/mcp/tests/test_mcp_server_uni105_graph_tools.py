#!/usr/bin/env python3
"""UNI-105 graph tools parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.graph_tools.native_graph_tools import (
    graph_add_relationship,
    register_native_graph_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI105GraphTools(unittest.TestCase):
    def test_register_includes_graph_add_relationship(self) -> None:
        manager = _DummyManager()
        register_native_graph_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("graph_add_relationship", names)

    def test_graph_add_relationship_rejects_missing_required_fields(self) -> None:
        async def _run() -> None:
            result = await graph_add_relationship(source_id="", target_id="b", relationship_type="KNOWS")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("must be provided", str(result.get("message", "")))

        anyio.run(_run)

    def test_graph_add_relationship_fallback_shape(self) -> None:
        async def _run() -> None:
            result = await graph_add_relationship(
                source_id="alice",
                target_id="bob",
                relationship_type="KNOWS",
            )
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("source_id"), "alice")
            self.assertEqual(result.get("target_id"), "bob")
            self.assertEqual(result.get("relationship_type"), "KNOWS")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
