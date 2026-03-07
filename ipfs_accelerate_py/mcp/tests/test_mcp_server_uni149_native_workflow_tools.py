#!/usr/bin/env python3
"""UNI-149 native workflow category parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from ipfs_accelerate_py.mcp_server.tools.workflow import native_workflow_tools as workflow_mod


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI149NativeWorkflowTools(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        workflow_mod.register_native_workflow_tools(manager)
        by_name = {call["name"]: call for call in manager.calls}

        get_schema = by_name["get_workflow"]["input_schema"]
        self.assertEqual(get_schema["properties"]["workflow_id"]["minLength"], 1)

        create_schema = by_name["create_workflow"]["input_schema"]
        self.assertEqual(create_schema["properties"]["name"]["minLength"], 1)
        self.assertEqual(create_schema["properties"]["description"]["minLength"], 1)

        update_schema = by_name["update_workflow"]["input_schema"]
        self.assertEqual(update_schema["properties"]["workflow_id"]["minLength"], 1)

    def test_validation_error_envelopes(self) -> None:
        self.assertEqual(
            workflow_mod.get_workflow("   ").get("error"),
            "workflow_id must be a non-empty string",
        )
        self.assertEqual(
            workflow_mod.create_workflow("", "desc", []).get("error"),
            "name must be a non-empty string",
        )
        self.assertEqual(
            workflow_mod.create_workflow("name", "", []).get("error"),
            "description must be a non-empty string",
        )
        self.assertEqual(
            workflow_mod.create_workflow("name", "desc", {}),  # type: ignore[arg-type]
            {"status": "error", "success": False, "error": "tasks must be a list"},
        )
        self.assertEqual(
            workflow_mod.update_workflow("wf-1", name="   ").get("error"),
            "name must be null or a non-empty string",
        )
        self.assertEqual(
            workflow_mod.list_workflows(status=123).get("error"),  # type: ignore[arg-type]
            "status must be a string or null",
        )

    def test_success_shapes_with_mock_manager(self) -> None:
        class _FakeWorkflow:
            workflow_id = "wf-1"
            name = "n"
            description = "d"
            status = "created"
            created_at = "now"
            started_at = None
            completed_at = None
            tasks = []
            error = None
            metadata = {}

            def get_progress(self):
                return {"completed": 0, "total": 0}

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

        with patch.object(workflow_mod, "_get_workflow_manager", return_value=_FakeManager()):
            self.assertEqual(workflow_mod.list_workflows().get("status"), "success")
            self.assertTrue(workflow_mod.list_workflows().get("success"))
            self.assertEqual(workflow_mod.get_workflow("wf-1").get("status"), "success")
            self.assertTrue(workflow_mod.get_workflow("wf-1").get("success"))
            self.assertEqual(workflow_mod.create_workflow("n", "d", []).get("status"), "success")
            self.assertTrue(workflow_mod.create_workflow("n", "d", []).get("success"))
            self.assertEqual(workflow_mod.update_workflow("wf-1", name="n").get("status"), "success")
            self.assertTrue(workflow_mod.update_workflow("wf-1", name="n").get("success"))
            self.assertEqual(workflow_mod.delete_workflow("wf-1").get("status"), "success")
            self.assertTrue(workflow_mod.delete_workflow("wf-1").get("success"))
            self.assertEqual(workflow_mod.start_workflow("wf-1").get("status"), "success")
            self.assertTrue(workflow_mod.start_workflow("wf-1").get("success"))
            self.assertEqual(workflow_mod.pause_workflow("wf-1").get("status"), "success")
            self.assertTrue(workflow_mod.pause_workflow("wf-1").get("success"))
            self.assertEqual(workflow_mod.stop_workflow("wf-1").get("status"), "success")
            self.assertTrue(workflow_mod.stop_workflow("wf-1").get("success"))

    def test_templates_include_explicit_success(self) -> None:
        result = workflow_mod.get_workflow_templates()
        self.assertEqual(result.get("status"), "success")
        self.assertTrue(result.get("success"))
        self.assertIn("templates", result)
        self.assertEqual(result.get("total"), len(result.get("templates", {})))

    def test_manager_unavailable_error_has_success_false(self) -> None:
        with patch.object(workflow_mod, "_get_workflow_manager", return_value=None):
            result = workflow_mod.list_workflows()

        self.assertEqual(result.get("status"), "error")
        self.assertFalse(result.get("success"))
        self.assertIn("Workflow manager not available", str(result.get("error", "")))


if __name__ == "__main__":
    unittest.main()
