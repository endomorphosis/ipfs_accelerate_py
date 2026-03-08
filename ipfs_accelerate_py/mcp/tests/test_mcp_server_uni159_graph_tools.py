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

        self.assertIn("query_knowledge_graph", schemas)
        self.assertEqual(
            schemas["graph_search_hybrid"]["properties"]["search_type"].get("enum"),
            ["hybrid", "keyword", "semantic"],
        )
        self.assertEqual(
            schemas["graph_visualize"]["properties"]["format"].get("enum"),
            ["ascii", "d3_json", "dot", "mermaid"],
        )
        self.assertEqual(
            schemas["graph_explain"]["properties"]["explain_type"].get("enum"),
            ["entity", "path", "relationship", "why_connected"],
        )

        tx_commit_props = schemas["graph_transaction_commit"]["properties"]
        self.assertEqual(tx_commit_props["transaction_id"].get("minLength"), 1)

        index_props = schemas["graph_index_create"]["properties"]
        self.assertEqual(index_props["properties"].get("minItems"), 1)

        constraint_props = schemas["graph_constraint_add"]["properties"]
        self.assertIn("unique", constraint_props["constraint_type"].get("enum", []))

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

    def test_expanded_graph_wrappers_validate_contracts(self) -> None:
        async def _run() -> None:
            invalid_search = await native_graph_tools.graph_search_hybrid(
                query="find people",
                search_type="vector",
            )
            self.assertEqual(invalid_search.get("status"), "error")
            self.assertIn("search_type must be one of", str(invalid_search.get("error", "")))

            invalid_ir_ops = await native_graph_tools.query_knowledge_graph(
                query="find people",
                ir_ops=["bad"],  # type: ignore[list-item]
            )
            self.assertEqual(invalid_ir_ops.get("status"), "error")
            self.assertIn("ir_ops must be null or a list of objects", str(invalid_ir_ops.get("error", "")))

            invalid_format = await native_graph_tools.graph_visualize(format="svg")
            self.assertEqual(invalid_format.get("status"), "error")
            self.assertIn("format must be one of", str(invalid_format.get("error", "")))

            invalid_explain = await native_graph_tools.graph_explain(explain_type="relationship")
            self.assertEqual(invalid_explain.get("status"), "error")
            self.assertIn("relationship_id is required", str(invalid_explain.get("error", "")))

        anyio.run(_run)

    def test_expanded_graph_wrappers_success_shapes(self) -> None:
        async def _run() -> None:
            query_result = await native_graph_tools.query_knowledge_graph(query="find regulations")
            self.assertIn(query_result.get("status"), ["success", "error"])
            self.assertEqual(query_result.get("query"), "find regulations")

            provenance_result = await native_graph_tools.graph_provenance_verify()
            self.assertIn(provenance_result.get("status"), ["success", "error"])

            suggestions_result = await native_graph_tools.graph_complete_suggestions(min_score=0.4)
            self.assertIn(suggestions_result.get("status"), ["success", "error"])
            self.assertIn("suggestions", suggestions_result)

        anyio.run(_run)

    def test_graph_transactions_indexes_and_constraints_validate_and_normalize(self) -> None:
        async def _run() -> None:
            invalid_tx = await native_graph_tools.graph_transaction_commit(transaction_id="   ")
            self.assertEqual(invalid_tx.get("status"), "error")
            self.assertIn("transaction_id must be a non-empty string", str(invalid_tx.get("error", "")))

            invalid_index = await native_graph_tools.graph_index_create(
                index_name="people-name",
                entity_type="Person",
                properties=[],
            )
            self.assertEqual(invalid_index.get("status"), "error")
            self.assertIn("properties must be a non-empty array", str(invalid_index.get("error", "")))

            invalid_constraint = await native_graph_tools.graph_constraint_add(
                constraint_name="person-email",
                constraint_type="primary",
                entity_type="Person",
                properties=["email"],
            )
            self.assertEqual(invalid_constraint.get("status"), "error")
            self.assertIn("constraint_type must be one of", str(invalid_constraint.get("error", "")))

            begun = await native_graph_tools.graph_transaction_begin()
            self.assertEqual(begun.get("status"), "success")
            transaction_id = begun.get("transaction_id")
            self.assertIsInstance(transaction_id, str)

            committed = await native_graph_tools.graph_transaction_commit(transaction_id=transaction_id)
            self.assertEqual(committed.get("status"), "success")

            rolled_back_missing = await native_graph_tools.graph_transaction_rollback(transaction_id="tx-missing")
            self.assertEqual(rolled_back_missing.get("status"), "error")

            created_index = await native_graph_tools.graph_index_create(
                index_name="people-name",
                entity_type="Person",
                properties=["name"],
            )
            self.assertEqual(created_index.get("status"), "success")
            self.assertEqual(created_index.get("index_name"), "people-name")

            created_constraint = await native_graph_tools.graph_constraint_add(
                constraint_name="person-email",
                constraint_type="unique",
                entity_type="Person",
                properties=["email"],
            )
            self.assertEqual(created_constraint.get("status"), "success")
            self.assertEqual(created_constraint.get("constraint_name"), "person-email")

        anyio.run(_run)

    def test_graph_wrappers_infer_error_status_from_contradictory_delegate_payload(self) -> None:
        async def _contradictory_failure(**_: object) -> dict:
            return {"status": "success", "success": False, "error": "delegate failed"}

        async def _run() -> None:
            with patch.dict(
                native_graph_tools._API,
                {
                    "graph_add_entity": _contradictory_failure,
                    "query_knowledge_graph": _contradictory_failure,
                    "graph_visualize": _contradictory_failure,
                },
                clear=False,
            ):
                added = await native_graph_tools.graph_add_entity(
                    entity_id="alice",
                    entity_type="Person",
                )
                searched = await native_graph_tools.query_knowledge_graph(query="find people")
                visualized = await native_graph_tools.graph_visualize(format="dot")

            self.assertEqual(added.get("status"), "error")
            self.assertEqual(added.get("error"), "delegate failed")

            self.assertEqual(searched.get("status"), "error")
            self.assertEqual(searched.get("error"), "delegate failed")

            self.assertEqual(visualized.get("status"), "error")
            self.assertEqual(visualized.get("error"), "delegate failed")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
