#!/usr/bin/env python3
"""UNI-301 native workflow dispatch compatibility tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp.server import create_mcp_server


class TestMCPServerUNI301WorkflowDispatchCompat(unittest.TestCase):
    def _assert_dispatch_success_envelope(self, response: dict) -> dict:
        self.assertIsInstance(response, dict)
        self.assertTrue(response.get("ok"), response)
        self.assertIn("result", response)
        self.assertIsInstance(response["result"], dict)
        return response["result"]

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_workflow_dispatch_preserves_validation_and_success_contracts(self, mock_wrapper) -> None:
        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        class _FakeWorkflow:
            workflow_id = "wf-1"
            name = "Demo Workflow"
            description = "Workflow from dispatch compat"
            status = "created"
            created_at = "2026-03-09T00:00:00Z"
            started_at = None
            completed_at = None
            error = None
            metadata = {"source": "uni301"}

            def __init__(self) -> None:
                self.tasks = [
                    _FakeTask(
                        task_id="task-1",
                        name="ingest",
                        task_type="pipeline",
                        status="pending",
                    )
                ]

            def get_progress(self) -> dict:
                return {"completed": 0, "total": len(self.tasks)}

        class _FakeTask:
            def __init__(self, task_id: str, name: str, task_type: str, status: str) -> None:
                self.task_id = task_id
                self.name = name
                self.type = task_type
                self.status = status
                self.config = {"step": 1}
                self.result = None
                self.error = None
                self.started_at = None
                self.completed_at = None
                self.dependencies = []

        class _FakeManager:
            def list_workflows(self, status=None):
                _ = status
                return [_FakeWorkflow()]

            def get_workflow(self, workflow_id):
                _ = workflow_id
                return _FakeWorkflow()

            def create_workflow(self, name, description, tasks):
                _ = name, description, tasks
                return _FakeWorkflow()

            def update_workflow(self, **kwargs):
                _ = kwargs
                return _FakeWorkflow()

            def delete_workflow(self, workflow_id):
                _ = workflow_id

            def start_workflow(self, workflow_id):
                _ = workflow_id

            def pause_workflow(self, workflow_id):
                _ = workflow_id

            def stop_workflow(self, workflow_id):
                _ = workflow_id

        mock_wrapper.return_value = DummyServer()

        async def _run_flow() -> None:
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                },
                clear=False,
            ), patch(
                "ipfs_accelerate_py.mcp_server.tools.workflow.native_workflow_tools._get_workflow_manager",
                return_value=_FakeManager(),
            ):
                server = create_mcp_server(name="workflow-dispatch-compat")
                dispatch = server.tools["tools_dispatch"]["function"]

                templates = self._assert_dispatch_success_envelope(
                    await dispatch("workflow", "get_workflow_templates", {})
                )
                self.assertEqual(templates.get("status"), "success")
                self.assertTrue(templates.get("success"))
                self.assertIn("templates", templates)
                self.assertEqual(templates.get("total"), len(templates.get("templates", {})))

                listed = self._assert_dispatch_success_envelope(
                    await dispatch("workflow", "list_workflows", {"status": "created"})
                )
                self.assertEqual(listed.get("status"), "success")
                self.assertTrue(listed.get("success"))
                self.assertEqual(listed.get("total"), 1)
                self.assertEqual((listed.get("workflows") or [{}])[0].get("workflow_id"), "wf-1")

                detailed = self._assert_dispatch_success_envelope(
                    await dispatch("workflow", "get_workflow", {"workflow_id": "wf-1"})
                )
                self.assertEqual(detailed.get("status"), "success")
                self.assertTrue(detailed.get("success"))
                self.assertEqual((detailed.get("workflow") or {}).get("workflow_id"), "wf-1")
                self.assertEqual(
                    ((detailed.get("workflow") or {}).get("tasks") or [{}])[0].get("task_id"),
                    "task-1",
                )

                created = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "workflow",
                        "create_workflow",
                        {
                            "name": "Demo Workflow",
                            "description": "Workflow from dispatch compat",
                            "tasks": [{"name": "ingest"}],
                        },
                    )
                )
                self.assertEqual(created.get("status"), "success")
                self.assertTrue(created.get("success"))
                self.assertEqual(created.get("workflow_id"), "wf-1")
                self.assertEqual(created.get("task_count"), 1)

                updated = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "workflow",
                        "update_workflow",
                        {"workflow_id": "wf-1", "name": "Updated Workflow"},
                    )
                )
                self.assertEqual(updated.get("status"), "success")
                self.assertTrue(updated.get("success"))
                self.assertEqual(updated.get("workflow_id"), "wf-1")

                started = self._assert_dispatch_success_envelope(
                    await dispatch("workflow", "start_workflow", {"workflow_id": "wf-1"})
                )
                self.assertEqual(started.get("status"), "success")
                self.assertTrue(started.get("success"))
                self.assertEqual(started.get("workflow_id"), "wf-1")

                invalid = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "workflow",
                        "create_workflow",
                        {
                            "name": " ",
                            "description": "Workflow from dispatch compat",
                            "tasks": [],
                        },
                    )
                )
                self.assertEqual(invalid.get("status"), "error")
                self.assertFalse(invalid.get("success"))
                self.assertEqual(invalid.get("error"), "name must be a non-empty string")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_workflow_dispatch_infers_error_status_from_contradictory_delegate_payloads(self, mock_wrapper) -> None:
        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(self, name, function, description, input_schema, execution_context=None, tags=None):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        class _ContradictoryManager:
            def list_workflows(self, status=None):
                _ = status
                return {"status": "success", "success": False, "error": "list failed"}

            def get_workflow(self, workflow_id):
                _ = workflow_id
                return {"status": "success", "success": False, "error": "detail failed"}

            def create_workflow(self, name, description, tasks):
                _ = name, description, tasks
                return {"status": "success", "success": False, "error": "create failed"}

            def update_workflow(self, **kwargs):
                _ = kwargs
                return {"status": "success", "success": False, "error": "update failed"}

            def delete_workflow(self, workflow_id):
                _ = workflow_id
                return {"status": "success", "success": False, "error": "delete failed"}

            def start_workflow(self, workflow_id):
                _ = workflow_id
                return {"status": "success", "success": False, "error": "start failed"}

            def pause_workflow(self, workflow_id):
                _ = workflow_id
                return {"status": "success", "success": False, "error": "pause failed"}

            def stop_workflow(self, workflow_id):
                _ = workflow_id
                return {"status": "success", "success": False, "error": "stop failed"}

        mock_wrapper.return_value = DummyServer()

        async def _run_flow() -> None:
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                },
                clear=False,
            ), patch(
                "ipfs_accelerate_py.mcp_server.tools.workflow.native_workflow_tools._get_workflow_manager",
                return_value=_ContradictoryManager(),
            ):
                server = create_mcp_server(name="workflow-dispatch-compat-errors")
                dispatch = server.tools["tools_dispatch"]["function"]

                listed = self._assert_dispatch_success_envelope(
                    await dispatch("workflow", "list_workflows", {})
                )
                self.assertEqual(listed.get("status"), "error")
                self.assertFalse(listed.get("success"))
                self.assertEqual(listed.get("error"), "list failed")

                detailed = self._assert_dispatch_success_envelope(
                    await dispatch("workflow", "get_workflow", {"workflow_id": "wf-1"})
                )
                self.assertEqual(detailed.get("status"), "error")
                self.assertFalse(detailed.get("success"))
                self.assertEqual(detailed.get("error"), "detail failed")

                created = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "workflow",
                        "create_workflow",
                        {"name": "Demo Workflow", "description": "desc", "tasks": []},
                    )
                )
                self.assertEqual(created.get("status"), "error")
                self.assertFalse(created.get("success"))
                self.assertEqual(created.get("error"), "create failed")

                updated = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "workflow",
                        "update_workflow",
                        {"workflow_id": "wf-1", "name": "Updated Workflow"},
                    )
                )
                self.assertEqual(updated.get("status"), "error")
                self.assertFalse(updated.get("success"))
                self.assertEqual(updated.get("error"), "update failed")

                deleted = self._assert_dispatch_success_envelope(
                    await dispatch("workflow", "delete_workflow", {"workflow_id": "wf-1"})
                )
                self.assertEqual(deleted.get("status"), "error")
                self.assertFalse(deleted.get("success"))
                self.assertEqual(deleted.get("error"), "delete failed")

                started = self._assert_dispatch_success_envelope(
                    await dispatch("workflow", "start_workflow", {"workflow_id": "wf-1"})
                )
                self.assertEqual(started.get("status"), "error")
                self.assertFalse(started.get("success"))
                self.assertEqual(started.get("error"), "start failed")

                paused = self._assert_dispatch_success_envelope(
                    await dispatch("workflow", "pause_workflow", {"workflow_id": "wf-1"})
                )
                self.assertEqual(paused.get("status"), "error")
                self.assertFalse(paused.get("success"))
                self.assertEqual(paused.get("error"), "pause failed")

                stopped = self._assert_dispatch_success_envelope(
                    await dispatch("workflow", "stop_workflow", {"workflow_id": "wf-1"})
                )
                self.assertEqual(stopped.get("status"), "error")
                self.assertFalse(stopped.get("success"))
                self.assertEqual(stopped.get("error"), "stop failed")

        anyio.run(_run_flow)


if __name__ == "__main__":
    unittest.main()
