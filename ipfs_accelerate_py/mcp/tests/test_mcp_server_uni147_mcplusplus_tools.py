#!/usr/bin/env python3
"""UNI-147 mcplusplus parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.mcplusplus import native_mcplusplus_tools as mcpp_mod


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI147McplusplusTools(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        mcpp_mod.register_native_mcplusplus_tools(manager)
        by_name = {call["name"]: call for call in manager.calls}

        task_schema = by_name["mcplusplus_taskqueue_get_status"]["input_schema"]
        task_props = task_schema["properties"]
        self.assertEqual(task_props["task_id"]["minLength"], 1)
        self.assertEqual(task_props["include_logs"]["default"], False)

        workflow_schema = by_name["mcplusplus_workflow_get_status"]["input_schema"]
        workflow_props = workflow_schema["properties"]
        self.assertEqual(workflow_props["workflow_id"]["minLength"], 1)
        self.assertEqual(workflow_props["include_steps"]["default"], True)

        peer_schema = by_name["mcplusplus_peer_list"]["input_schema"]
        peer_props = peer_schema["properties"]
        self.assertEqual(peer_props["limit"]["minimum"], 1)

    def test_validation_error_envelopes(self) -> None:
        async def _run() -> None:
            invalid_task = await mcpp_mod.mcplusplus_taskqueue_get_status("   ")
            self.assertEqual(invalid_task.get("status"), "error")
            self.assertIn("task_id must be a non-empty string", str(invalid_task.get("error", "")))

            invalid_workflow = await mcpp_mod.mcplusplus_workflow_get_status("", include_steps="yes")
            self.assertEqual(invalid_workflow.get("status"), "error")
            self.assertIn(
                "workflow_id must be a non-empty string",
                str(invalid_workflow.get("error", "")),
            )

            invalid_include_steps = await mcpp_mod.mcplusplus_workflow_get_status(
                "wf-1",
                include_steps="yes",
            )
            self.assertEqual(invalid_include_steps.get("status"), "error")
            self.assertIn(
                "include_steps must be a boolean",
                str(invalid_include_steps.get("error", "")),
            )

            invalid_limit = await mcpp_mod.mcplusplus_peer_list(limit=0)
            self.assertEqual(invalid_limit.get("status"), "error")
            self.assertIn("limit must be an integer >= 1", str(invalid_limit.get("error", "")))

        anyio.run(_run)

    def test_success_envelope_shapes(self) -> None:
        async def _run() -> None:
            status = await mcpp_mod.mcplusplus_engine_status()
            self.assertIn(status.get("status"), ["success", "error"])
            self.assertIn("available", status)

            engines = await mcpp_mod.mcplusplus_list_engines()
            self.assertEqual(engines.get("status"), "success")
            self.assertIn("engines", engines)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
