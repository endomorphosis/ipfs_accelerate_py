#!/usr/bin/env python3
"""UNI-159 deterministic parity tests for native graph tools."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.graph_tools import native_graph_tools


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI159GraphTools(unittest.TestCase):
    def test_registration_schema_contracts_are_tightened(self) -> None:
        manager = _DummyManager()
        native_graph_tools.register_native_graph_tools(manager)
        schemas = {call["name"]: call["input_schema"] for call in manager.calls}

        add_entity_props = schemas["graph_add_entity"]["properties"]
        self.assertEqual(add_entity_props["entity_id"].get("minLength"), 1)

        query_props = schemas["graph_query_cypher"]["properties"]
        self.assertEqual(query_props["query"].get("minLength"), 1)

    def test_graph_add_entity_validates_and_wraps_exceptions(self) -> None:
        async def _boom(**_: object) -> dict:
            raise RuntimeError("entity boom")

        async def _run() -> None:
            invalid_properties = await native_graph_tools.graph_add_entity(
                entity_id="alice",
                entity_type="Person",
                properties=["bad"],  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_properties.get("status"), "error")
            self.assertIn("properties must be an object", str(invalid_properties.get("error", "")))

            with patch.dict(native_graph_tools._API, {"graph_add_entity": _boom}, clear=False):
                result = await native_graph_tools.graph_add_entity(
                    entity_id="alice",
                    entity_type="Person",
                )
                self.assertEqual(result.get("status"), "error")
                self.assertIn("graph_add_entity failed", str(result.get("error", "")))

        anyio.run(_run)

    def test_graph_add_relationship_and_query_validate_contracts(self) -> None:
        async def _run() -> None:
            invalid_rel = await native_graph_tools.graph_add_relationship(
                source_id="alice",
                target_id="bob",
                relationship_type="KNOWS",
                properties=["bad"],  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_rel.get("status"), "error")
            self.assertIn("properties must be an object", str(invalid_rel.get("error", "")))

            invalid_query_params = await native_graph_tools.graph_query_cypher(
                query="MATCH (n) RETURN n",
                parameters=["bad"],  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_query_params.get("status"), "error")
            self.assertIn("parameters must be an object", str(invalid_query_params.get("error", "")))

        anyio.run(_run)

    def test_graph_query_wraps_exceptions(self) -> None:
        async def _boom(**_: object) -> dict:
            raise RuntimeError("query boom")

        async def _run() -> None:
            with patch.dict(native_graph_tools._API, {"graph_query_cypher": _boom}, clear=False):
                result = await native_graph_tools.graph_query_cypher(query="MATCH (n) RETURN n")
                self.assertEqual(result.get("status"), "error")
                self.assertIn("graph_query_cypher failed", str(result.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
