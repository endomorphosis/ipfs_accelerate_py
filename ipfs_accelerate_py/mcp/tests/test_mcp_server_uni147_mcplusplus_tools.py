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

        task_submit_schema = by_name["mcplusplus_taskqueue_submit"]["input_schema"]
        task_submit_props = task_submit_schema["properties"]
        self.assertEqual(task_submit_props["task_type"]["minLength"], 1)
        self.assertEqual(task_submit_props["priority"]["exclusiveMinimum"], 0)

        workflow_schema = by_name["mcplusplus_workflow_get_status"]["input_schema"]
        workflow_props = workflow_schema["properties"]
        self.assertEqual(workflow_props["workflow_id"]["minLength"], 1)
        self.assertEqual(workflow_props["include_steps"]["default"], True)

        workflow_submit_schema = by_name["mcplusplus_workflow_submit"]["input_schema"]
        workflow_submit_props = workflow_submit_schema["properties"]
        self.assertEqual(workflow_submit_props["name"]["minLength"], 1)
        self.assertEqual(workflow_submit_props["steps"]["minItems"], 1)

        peer_schema = by_name["mcplusplus_peer_list"]["input_schema"]
        peer_props = peer_schema["properties"]
        self.assertEqual(peer_props["limit"]["minimum"], 1)

        peer_discover_schema = by_name["mcplusplus_peer_discover"]["input_schema"]
        peer_discover_props = peer_discover_schema["properties"]
        self.assertEqual(peer_discover_props["max_peers"]["minimum"], 1)
        self.assertEqual(peer_discover_props["timeout"]["minimum"], 1)

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

            invalid_submit_payload = await mcpp_mod.mcplusplus_taskqueue_submit(
                task_id="task-1",
                task_type="demo",
                payload="not-an-object",  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_submit_payload.get("status"), "error")
            self.assertIn("payload must be an object", str(invalid_submit_payload.get("error", "")))

            invalid_workflow_steps = await mcpp_mod.mcplusplus_workflow_submit(
                workflow_id="wf-1",
                name="demo",
                steps=[],
            )
            self.assertEqual(invalid_workflow_steps.get("status"), "error")
            self.assertIn("steps must be a non-empty array of objects", str(invalid_workflow_steps.get("error", "")))

            invalid_peer_discover = await mcpp_mod.mcplusplus_peer_discover(max_peers=0)
            self.assertEqual(invalid_peer_discover.get("status"), "error")
            self.assertIn("max_peers must be an integer >= 1", str(invalid_peer_discover.get("error", "")))

        anyio.run(_run)

    def test_success_envelope_shapes(self) -> None:
        async def _run() -> None:
            status = await mcpp_mod.mcplusplus_engine_status()
            self.assertIn(status.get("status"), ["success", "error"])
            self.assertIn("available", status)

            engines = await mcpp_mod.mcplusplus_list_engines()
            self.assertEqual(engines.get("status"), "success")
            self.assertIn("engines", engines)

            submit = await mcpp_mod.mcplusplus_taskqueue_submit(
                task_id="task-1",
                task_type="demo",
                payload={"x": 1},
            )
            self.assertIn(submit.get("status"), ["success", "error"])

            workflow_submit = await mcpp_mod.mcplusplus_workflow_submit(
                workflow_id="wf-1",
                name="demo",
                steps=[{"step_id": "s1", "action": "noop"}],
            )
            self.assertIn(workflow_submit.get("status"), ["success", "error"])

            discover = await mcpp_mod.mcplusplus_peer_discover(max_peers=2)
            self.assertIn(discover.get("status"), ["success", "error"])

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
