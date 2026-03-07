#!/usr/bin/env python3
"""UNI-141 bespoke_tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.bespoke_tools.native_bespoke_tools import (
    cache_stats,
    create_vector_store,
    delete_index,
    execute_workflow,
    list_indices,
    register_native_bespoke_tools,
    system_health,
    system_status,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI141BespokeTools(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_bespoke_tools(manager)
        by_name = {call["name"]: call for call in manager.calls}

        cache_schema = by_name["cache_stats"]["input_schema"]
        self.assertEqual(cache_schema["properties"]["namespace"]["minLength"], 1)
        self.assertEqual(by_name["execute_workflow"]["input_schema"]["required"], ["workflow_id"])
        self.assertEqual(
            by_name["create_vector_store"]["input_schema"]["properties"]["dimension"]["minimum"],
            1,
        )
        self.assertIn("delete_index", by_name)
        self.assertIn("list_indices", by_name)
        self.assertIn("system_status", by_name)

    def test_cache_stats_validation(self) -> None:
        async def _run() -> None:
            result = await cache_stats(namespace="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("namespace", str(result.get("error", "")))

        anyio.run(_run)

    def test_bespoke_validation_failures(self) -> None:
        async def _run() -> None:
            workflow_error = await execute_workflow("audit_report", parameters=["bad"])  # type: ignore[arg-type]
            self.assertEqual(workflow_error.get("status"), "error")
            self.assertIn("parameters", str(workflow_error.get("error", "")))

            list_error = await list_indices(store_type="sqlite")
            self.assertEqual(list_error.get("status"), "error")
            self.assertIn("store_type", str(list_error.get("error", "")))

            delete_error = await delete_index("idx_1", confirm="yes")  # type: ignore[arg-type]
            self.assertEqual(delete_error.get("status"), "error")
            self.assertIn("confirm", str(delete_error.get("error", "")))

            create_error = await create_vector_store("demo", metric="angular")
            self.assertEqual(create_error.get("status"), "error")
            self.assertIn("metric", str(create_error.get("error", "")))

        anyio.run(_run)

    def test_success_envelope_shapes(self) -> None:
        async def _run() -> None:
            health_result = await system_health()
            self.assertIn(health_result.get("status"), ["success", "error"])

            status_result = await system_status()
            self.assertIn(status_result.get("status"), ["success", "error"])

            cache_result = await cache_stats(namespace="primary")
            self.assertIn(cache_result.get("status"), ["success", "error"])
            self.assertEqual(cache_result.get("namespace"), "primary")

            workflow_result = await execute_workflow("audit_report", dry_run=True)
            self.assertIn(workflow_result.get("status"), ["success", "error"])
            self.assertEqual(workflow_result.get("workflow_id"), "audit_report")
            self.assertTrue(workflow_result.get("dry_run"))

            indices_result = await list_indices(include_stats=True)
            self.assertIn(indices_result.get("status"), ["success", "error"])
            self.assertIn("filters_applied", indices_result)

            create_result = await create_vector_store("Demo Store", configuration={"replicas": 1})
            self.assertIn(create_result.get("status"), ["success", "error"])
            store_info = create_result.get("store_info") or {}
            self.assertEqual(store_info.get("store_name"), "Demo Store")

            delete_result = await delete_index("idx_embeddings_001", confirm=True)
            self.assertIn(delete_result.get("status"), ["success", "error"])
            self.assertEqual(delete_result.get("index_id"), "idx_embeddings_001")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
