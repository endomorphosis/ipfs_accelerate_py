#!/usr/bin/env python3
"""UNI-147 mcplusplus parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

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
        self.assertIn("retry_policy", task_submit_props)
        self.assertIn("metadata", task_submit_props)

        task_priority_schema = by_name["mcplusplus_taskqueue_priority"]["input_schema"]
        self.assertEqual(task_priority_schema["properties"]["new_priority"]["exclusiveMinimum"], 0)

        task_cancel_schema = by_name["mcplusplus_taskqueue_cancel"]["input_schema"]
        self.assertEqual(task_cancel_schema["properties"]["force"]["default"], False)

        task_list_schema = by_name["mcplusplus_taskqueue_list"]["input_schema"]
        self.assertEqual(task_list_schema["properties"]["limit"]["minimum"], 1)

        task_result_schema = by_name["mcplusplus_taskqueue_result"]["input_schema"]
        self.assertEqual(task_result_schema["properties"]["include_output"]["default"], True)

        task_cancel_schema = by_name["mcplusplus_taskqueue_cancel"]["input_schema"]
        task_cancel_props = task_cancel_schema["properties"]
        self.assertEqual(task_cancel_props["task_id"]["minLength"], 1)
        self.assertEqual(task_cancel_props["force"]["default"], False)

        task_list_schema = by_name["mcplusplus_taskqueue_list"]["input_schema"]
        task_list_props = task_list_schema["properties"]
        self.assertEqual(task_list_props["limit"]["minimum"], 1)
        self.assertEqual(task_list_props["offset"]["minimum"], 0)

        task_priority_schema = by_name["mcplusplus_taskqueue_priority"]["input_schema"]
        task_priority_props = task_priority_schema["properties"]
        self.assertEqual(task_priority_props["task_id"]["minLength"], 1)
        self.assertEqual(task_priority_props["new_priority"]["exclusiveMinimum"], 0)
        self.assertEqual(task_priority_props["requeue"]["default"], True)

        task_set_priority_schema = by_name["mcplusplus_taskqueue_set_priority"]["input_schema"]
        task_set_priority_props = task_set_priority_schema["properties"]
        self.assertEqual(task_set_priority_props["task_id"]["minLength"], 1)
        self.assertEqual(task_set_priority_props["new_priority"]["exclusiveMinimum"], 0)
        self.assertEqual(task_set_priority_props["requeue"]["default"], True)

        task_stats_schema = by_name["mcplusplus_taskqueue_stats"]["input_schema"]
        task_stats_props = task_stats_schema["properties"]
        self.assertEqual(task_stats_props["include_worker_stats"]["default"], False)

        task_retry_schema = by_name["mcplusplus_taskqueue_retry"]["input_schema"]
        task_retry_props = task_retry_schema["properties"]
        self.assertEqual(task_retry_props["task_id"]["minLength"], 1)

        task_pause_schema = by_name["mcplusplus_taskqueue_pause"]["input_schema"]
        self.assertIn("reason", task_pause_schema["properties"])

        task_resume_schema = by_name["mcplusplus_taskqueue_resume"]["input_schema"]
        self.assertEqual(task_resume_schema["properties"]["reorder_by_priority"]["default"], True)

        task_clear_schema = by_name["mcplusplus_taskqueue_clear"]["input_schema"]
        self.assertEqual(task_clear_schema["properties"]["confirm"]["default"], False)

        worker_register_schema = by_name["mcplusplus_worker_register"]["input_schema"]
        self.assertEqual(worker_register_schema["properties"]["max_concurrent_tasks"]["minimum"], 1)

        worker_unregister_schema = by_name["mcplusplus_worker_unregister"]["input_schema"]
        self.assertEqual(worker_unregister_schema["properties"]["timeout"]["minimum"], 1)

        worker_status_schema = by_name["mcplusplus_worker_status"]["input_schema"]
        self.assertEqual(worker_status_schema["properties"]["include_tasks"]["default"], False)

        task_pause_schema = by_name["mcplusplus_taskqueue_pause"]["input_schema"]
        task_pause_props = task_pause_schema["properties"]
        self.assertEqual(task_pause_props["reason"]["type"], ["string", "null"])

        task_resume_schema = by_name["mcplusplus_taskqueue_resume"]["input_schema"]
        task_resume_props = task_resume_schema["properties"]
        self.assertEqual(task_resume_props["reorder_by_priority"]["default"], True)

        task_clear_schema = by_name["mcplusplus_taskqueue_clear"]["input_schema"]
        task_clear_props = task_clear_schema["properties"]
        self.assertEqual(task_clear_props["confirm"]["default"], False)

        workflow_schema = by_name["mcplusplus_workflow_get_status"]["input_schema"]
        workflow_props = workflow_schema["properties"]
        self.assertEqual(workflow_props["workflow_id"]["minLength"], 1)
        self.assertEqual(workflow_props["include_steps"]["default"], True)

        workflow_submit_schema = by_name["mcplusplus_workflow_submit"]["input_schema"]
        workflow_submit_props = workflow_submit_schema["properties"]
        self.assertEqual(workflow_submit_props["name"]["minLength"], 1)
        self.assertEqual(workflow_submit_props["steps"]["minItems"], 1)
        self.assertIn("dependencies", workflow_submit_props)
        self.assertIn("metadata", workflow_submit_props)

        workflow_cancel_schema = by_name["mcplusplus_workflow_cancel"]["input_schema"]
        self.assertEqual(workflow_cancel_schema["properties"]["force"]["default"], False)

        workflow_deps_schema = by_name["mcplusplus_workflow_dependencies"]["input_schema"]
        self.assertEqual(workflow_deps_schema["properties"]["fmt"]["default"], "json")

        workflow_result_schema = by_name["mcplusplus_workflow_result"]["input_schema"]
        self.assertEqual(workflow_result_schema["properties"]["include_outputs"]["default"], True)

        workflow_list_schema = by_name["mcplusplus_workflow_list"]["input_schema"]
        workflow_list_props = workflow_list_schema["properties"]
        self.assertEqual(workflow_list_props["limit"]["minimum"], 1)
        self.assertEqual(workflow_list_props["offset"]["minimum"], 0)

        workflow_dependencies_schema = by_name["mcplusplus_workflow_dependencies"]["input_schema"]
        workflow_dependencies_props = workflow_dependencies_schema["properties"]
        self.assertEqual(workflow_dependencies_props["workflow_id"]["minLength"], 1)
        self.assertEqual(workflow_dependencies_props["fmt"]["default"], "json")

        workflow_result_schema = by_name["mcplusplus_workflow_result"]["input_schema"]
        workflow_result_props = workflow_result_schema["properties"]
        self.assertEqual(workflow_result_props["workflow_id"]["minLength"], 1)
        self.assertEqual(workflow_result_props["include_outputs"]["default"], True)

        peer_schema = by_name["mcplusplus_peer_list"]["input_schema"]
        peer_props = peer_schema["properties"]
        self.assertEqual(peer_props["limit"]["minimum"], 1)
        self.assertEqual(peer_props["offset"]["minimum"], 0)

        peer_discover_schema = by_name["mcplusplus_peer_discover"]["input_schema"]
        peer_discover_props = peer_discover_schema["properties"]
        self.assertEqual(peer_discover_props["max_peers"]["minimum"], 1)
        self.assertEqual(peer_discover_props["timeout"]["minimum"], 1)

        peer_connect_schema = by_name["mcplusplus_peer_connect"]["input_schema"]
        self.assertEqual(peer_connect_schema["properties"]["retry_count"]["minimum"], 0)

        peer_metrics_schema = by_name["mcplusplus_peer_metrics"]["input_schema"]
        self.assertEqual(peer_metrics_schema["properties"]["history_hours"]["minimum"], 1)

        peer_bootstrap_schema = by_name["mcplusplus_peer_bootstrap_network"]["input_schema"]
        self.assertEqual(peer_bootstrap_schema["properties"]["max_connections"]["minimum"], 1)

        peer_connect_schema = by_name["mcplusplus_peer_connect"]["input_schema"]
        peer_connect_props = peer_connect_schema["properties"]
        self.assertEqual(peer_connect_props["peer_id"]["minLength"], 1)
        self.assertEqual(peer_connect_props["retry_count"]["minimum"], 0)

        peer_metrics_schema = by_name["mcplusplus_peer_metrics"]["input_schema"]
        peer_metrics_props = peer_metrics_schema["properties"]
        self.assertEqual(peer_metrics_props["peer_id"]["minLength"], 1)
        self.assertEqual(peer_metrics_props["history_hours"]["minimum"], 1)

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

            invalid_cancel_reason = await mcpp_mod.mcplusplus_taskqueue_cancel(
                task_id="task-1",
                reason="   ",
            )
            self.assertEqual(invalid_cancel_reason.get("status"), "error")
            self.assertIn("reason must be a non-empty string", str(invalid_cancel_reason.get("error", "")))

            invalid_taskqueue_offset = await mcpp_mod.mcplusplus_taskqueue_list(offset=-1)
            self.assertEqual(invalid_taskqueue_offset.get("status"), "error")
            self.assertIn("offset must be an integer >= 0", str(invalid_taskqueue_offset.get("error", "")))

            invalid_taskqueue_stats = await mcpp_mod.mcplusplus_taskqueue_stats(
                include_historical="yes",  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_taskqueue_stats.get("status"), "error")
            self.assertIn("include_historical must be a boolean", str(invalid_taskqueue_stats.get("error", "")))

            invalid_taskqueue_set_priority = await mcpp_mod.mcplusplus_taskqueue_set_priority(
                task_id="task-1",
                new_priority=0,
            )
            self.assertEqual(invalid_taskqueue_set_priority.get("status"), "error")
            self.assertIn("new_priority must be > 0", str(invalid_taskqueue_set_priority.get("error", "")))

            invalid_taskqueue_priority = await mcpp_mod.mcplusplus_taskqueue_priority(
                task_id="task-1",
                new_priority=0,
            )
            self.assertEqual(invalid_taskqueue_priority.get("status"), "error")
            self.assertIn("new_priority must be > 0", str(invalid_taskqueue_priority.get("error", "")))

            invalid_taskqueue_retry = await mcpp_mod.mcplusplus_taskqueue_retry(
                task_id="task-1",
                retry_config="not-a-dict",  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_taskqueue_retry.get("status"), "error")
            self.assertIn("retry_config must be an object", str(invalid_taskqueue_retry.get("error", "")))

            invalid_taskqueue_priority = await mcpp_mod.mcplusplus_taskqueue_priority(
                task_id="task-1",
                new_priority=0,
            )
            self.assertEqual(invalid_taskqueue_priority.get("status"), "error")
            self.assertIn("new_priority must be > 0", str(invalid_taskqueue_priority.get("error", "")))

            invalid_taskqueue_resume = await mcpp_mod.mcplusplus_taskqueue_resume(
                reorder_by_priority="yes",  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_taskqueue_resume.get("status"), "error")
            self.assertIn(
                "reorder_by_priority must be a boolean",
                str(invalid_taskqueue_resume.get("error", "")),
            )

            invalid_taskqueue_clear = await mcpp_mod.mcplusplus_taskqueue_clear(
                confirm="true",  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_taskqueue_clear.get("status"), "error")
            self.assertIn("confirm must be a boolean", str(invalid_taskqueue_clear.get("error", "")))

            invalid_worker_register = await mcpp_mod.mcplusplus_worker_register(
                worker_id="worker-1",
                capabilities=[],
            )
            self.assertEqual(invalid_worker_register.get("status"), "error")
            self.assertIn("capabilities must be a non-empty array", str(invalid_worker_register.get("error", "")))

            invalid_worker_unregister = await mcpp_mod.mcplusplus_worker_unregister(
                worker_id="worker-1",
                timeout=0,
            )
            self.assertEqual(invalid_worker_unregister.get("status"), "error")
            self.assertIn("timeout must be an integer >= 1", str(invalid_worker_unregister.get("error", "")))

            invalid_worker_status = await mcpp_mod.mcplusplus_worker_status(
                worker_id="worker-1",
                include_tasks="yes",  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_worker_status.get("status"), "error")
            self.assertIn("include_tasks must be a boolean", str(invalid_worker_status.get("error", "")))

            invalid_taskqueue_pause = await mcpp_mod.mcplusplus_taskqueue_pause(reason="   ")
            self.assertEqual(invalid_taskqueue_pause.get("status"), "error")
            self.assertIn("reason must be a non-empty string", str(invalid_taskqueue_pause.get("error", "")))

            invalid_taskqueue_resume = await mcpp_mod.mcplusplus_taskqueue_resume(
                reorder_by_priority="yes",  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_taskqueue_resume.get("status"), "error")
            self.assertIn(
                "reorder_by_priority must be a boolean",
                str(invalid_taskqueue_resume.get("error", "")),
            )

            invalid_taskqueue_clear = await mcpp_mod.mcplusplus_taskqueue_clear(
                confirm="yes",  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_taskqueue_clear.get("status"), "error")
            self.assertIn("confirm must be a boolean", str(invalid_taskqueue_clear.get("error", "")))

            invalid_workflow_limit = await mcpp_mod.mcplusplus_workflow_list(limit=0)
            self.assertEqual(invalid_workflow_limit.get("status"), "error")
            self.assertIn("limit must be an integer >= 1", str(invalid_workflow_limit.get("error", "")))

            invalid_workflow_submit_dependencies = await mcpp_mod.mcplusplus_workflow_submit(
                workflow_id="wf-1",
                name="demo",
                steps=[{"step": "a"}],
                dependencies=[""],
            )
            self.assertEqual(invalid_workflow_submit_dependencies.get("status"), "error")
            self.assertIn("dependencies must be an array of non-empty strings", str(invalid_workflow_submit_dependencies.get("error", "")))

            invalid_workflow_fmt = await mcpp_mod.mcplusplus_workflow_dependencies(
                workflow_id="wf-1",
                fmt="yaml",
            )
            self.assertEqual(invalid_workflow_fmt.get("status"), "error")
            self.assertIn("fmt must be one of", str(invalid_workflow_fmt.get("error", "")))

            invalid_workflow_result = await mcpp_mod.mcplusplus_workflow_result(
                workflow_id="wf-1",
                include_outputs="yes",  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_workflow_result.get("status"), "error")
            self.assertIn("include_outputs must be a boolean", str(invalid_workflow_result.get("error", "")))

            invalid_peer_discover = await mcpp_mod.mcplusplus_peer_discover(max_peers=0)
            self.assertEqual(invalid_peer_discover.get("status"), "error")
            self.assertIn("max_peers must be an integer >= 1", str(invalid_peer_discover.get("error", "")))

            invalid_peer_list_offset = await mcpp_mod.mcplusplus_peer_list(offset=-1)
            self.assertEqual(invalid_peer_list_offset.get("status"), "error")
            self.assertIn("offset must be an integer >= 0", str(invalid_peer_list_offset.get("error", "")))

            invalid_task_offset = await mcpp_mod.mcplusplus_taskqueue_list(offset=-1)
            self.assertEqual(invalid_task_offset.get("status"), "error")
            self.assertIn("offset must be an integer >= 0", str(invalid_task_offset.get("error", "")))

            invalid_workflow_fmt = await mcpp_mod.mcplusplus_workflow_dependencies("wf-1", fmt="yaml")
            self.assertEqual(invalid_workflow_fmt.get("status"), "error")
            self.assertIn("fmt must be one of: json, dot, mermaid", str(invalid_workflow_fmt.get("error", "")))

            invalid_peer_connect = await mcpp_mod.mcplusplus_peer_connect("peer-1", "", timeout=10)
            self.assertEqual(invalid_peer_connect.get("status"), "error")
            self.assertIn("multiaddr must be a non-empty string", str(invalid_peer_connect.get("error", "")))

            invalid_bootstrap_range = await mcpp_mod.mcplusplus_peer_bootstrap_network(
                min_connections=5,
                max_connections=2,
            )
            self.assertEqual(invalid_bootstrap_range.get("status"), "error")
            self.assertIn(
                "min_connections must be <= max_connections",
                str(invalid_bootstrap_range.get("error", "")),
            )

            invalid_peer_connect = await mcpp_mod.mcplusplus_peer_connect(
                peer_id="peer-1",
                multiaddr="/ip4/127.0.0.1/tcp/4001",
                retry_count=-1,
            )
            self.assertEqual(invalid_peer_connect.get("status"), "error")
            self.assertIn("retry_count must be an integer >= 0", str(invalid_peer_connect.get("error", "")))

            invalid_peer_metrics = await mcpp_mod.mcplusplus_peer_metrics(
                peer_id="peer-1",
                history_hours=0,
            )
            self.assertEqual(invalid_peer_metrics.get("status"), "error")
            self.assertIn("history_hours must be an integer >= 1", str(invalid_peer_metrics.get("error", "")))

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

            task_priority = await mcpp_mod.mcplusplus_taskqueue_priority(
                task_id="task-1",
                new_priority=2,
            )
            self.assertIn(task_priority.get("status"), ["success", "error"])

            task_list = await mcpp_mod.mcplusplus_taskqueue_list(limit=5)
            self.assertIn(task_list.get("status"), ["success", "error"])

            task_set_priority = await mcpp_mod.mcplusplus_taskqueue_set_priority(
                task_id="task-1",
                new_priority=2.0,
                requeue=False,
            )
            self.assertIn(task_set_priority.get("status"), ["success", "error"])

            task_priority = await mcpp_mod.mcplusplus_taskqueue_priority(
                task_id="task-1",
                new_priority=1.5,
            )
            self.assertIn(task_priority.get("status"), ["success", "error"])

            task_stats = await mcpp_mod.mcplusplus_taskqueue_stats()
            self.assertIn(task_stats.get("status"), ["success", "error"])

            task_retry = await mcpp_mod.mcplusplus_taskqueue_retry(task_id="task-1")
            self.assertIn(task_retry.get("status"), ["success", "error"])

            task_pause = await mcpp_mod.mcplusplus_taskqueue_pause(reason="maintenance")
            self.assertIn(task_pause.get("status"), ["success", "error"])

            task_resume = await mcpp_mod.mcplusplus_taskqueue_resume(reorder_by_priority=True)
            self.assertIn(task_resume.get("status"), ["success", "error"])

            task_clear = await mcpp_mod.mcplusplus_taskqueue_clear(confirm=False)
            self.assertIn(task_clear.get("status"), ["success", "error"])

            task_pause = await mcpp_mod.mcplusplus_taskqueue_pause()
            self.assertIn(task_pause.get("status"), ["success", "error"])

            task_resume = await mcpp_mod.mcplusplus_taskqueue_resume()
            self.assertIn(task_resume.get("status"), ["success", "error"])

            task_clear = await mcpp_mod.mcplusplus_taskqueue_clear(confirm=True)
            self.assertIn(task_clear.get("status"), ["success", "error"])

            task_result = await mcpp_mod.mcplusplus_taskqueue_result("task-1")
            self.assertIn(task_result.get("status"), ["success", "error"])

            workflow_submit = await mcpp_mod.mcplusplus_workflow_submit(
                workflow_id="wf-1",
                name="demo",
                steps=[{"step_id": "s1", "action": "noop"}],
                dependencies=["wf-0"],
            )
            self.assertIn(workflow_submit.get("status"), ["success", "error"])

            workflow_list = await mcpp_mod.mcplusplus_workflow_list(limit=5)
            self.assertIn(workflow_list.get("status"), ["success", "error"])

            workflow_dependencies = await mcpp_mod.mcplusplus_workflow_dependencies(
                workflow_id="wf-1",
            )
            self.assertIn(workflow_dependencies.get("status"), ["success", "error"])

            workflow_result = await mcpp_mod.mcplusplus_workflow_result(workflow_id="wf-1")
            self.assertIn(workflow_result.get("status"), ["success", "error"])

            workflow_result = await mcpp_mod.mcplusplus_workflow_result("wf-1")
            self.assertIn(workflow_result.get("status"), ["success", "error"])

            workflow_list = await mcpp_mod.mcplusplus_workflow_list(limit=5)
            self.assertIn(workflow_list.get("status"), ["success", "error"])

            task_cancel = await mcpp_mod.mcplusplus_taskqueue_cancel(task_id="task-1")
            self.assertIn(task_cancel.get("status"), ["success", "error"])

            task_list = await mcpp_mod.mcplusplus_taskqueue_list(limit=5)
            self.assertIn(task_list.get("status"), ["success", "error"])

            discover = await mcpp_mod.mcplusplus_peer_discover(max_peers=2)
            self.assertIn(discover.get("status"), ["success", "error"])

            peer_connect = await mcpp_mod.mcplusplus_peer_connect(
                peer_id="peer-1",
                multiaddr="/ip4/127.0.0.1/tcp/4001",
            )
            self.assertIn(peer_connect.get("status"), ["success", "error"])

            peer_bootstrap = await mcpp_mod.mcplusplus_peer_bootstrap_network(max_connections=5)
            self.assertIn(peer_bootstrap.get("status"), ["success", "error"])

            peer_list = await mcpp_mod.mcplusplus_peer_list(
                capability_filter=["inference"],
                sort_by="last_seen",
                offset=0,
            )
            self.assertIn(peer_list.get("status"), ["success", "error"])

            peer_connect = await mcpp_mod.mcplusplus_peer_connect(
                peer_id="peer-1",
                multiaddr="/ip4/127.0.0.1/tcp/4001",
            )
            self.assertIn(peer_connect.get("status"), ["success", "error"])

            peer_metrics = await mcpp_mod.mcplusplus_peer_metrics(peer_id="peer-1")
            self.assertIn(peer_metrics.get("status"), ["success", "error"])

            worker_register = await mcpp_mod.mcplusplus_worker_register(
                worker_id="worker-1",
                capabilities=["inference"],
            )
            self.assertIn(worker_register.get("status"), ["success", "error"])

            worker_status = await mcpp_mod.mcplusplus_worker_status(worker_id="worker-1")
            self.assertIn(worker_status.get("status"), ["success", "error"])

            worker_unregister = await mcpp_mod.mcplusplus_worker_unregister(worker_id="worker-1")
            self.assertIn(worker_unregister.get("status"), ["success", "error"])

        anyio.run(_run)

    def test_taskqueue_get_status_minimal_success_defaults(self) -> None:
        async def _run() -> None:
            class _TaskQueueEngine:
                def get_status(self, **_kwargs):
                    return {"status": "success"}

            with patch.object(mcpp_mod, "_API", new={"TaskQueueEngine": _TaskQueueEngine}):
                result = await mcpp_mod.mcplusplus_taskqueue_get_status("task-1")

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("engine"), "TaskQueueEngine")
            self.assertEqual(result.get("method"), "get_status")

        anyio.run(_run)

    def test_taskqueue_get_status_error_only_payload_infers_error(self) -> None:
        async def _run() -> None:
            class _TaskQueueEngine:
                def get_status(self, **_kwargs):
                    return {"error": "task missing"}

            with patch.object(mcpp_mod, "_API", new={"TaskQueueEngine": _TaskQueueEngine}):
                result = await mcpp_mod.mcplusplus_taskqueue_get_status("task-1")

            self.assertEqual(result.get("status"), "error")
            self.assertEqual(result.get("success"), False)
            self.assertEqual(result.get("engine"), "TaskQueueEngine")
            self.assertEqual(result.get("method"), "get_status")
            self.assertIn("task missing", str(result.get("error", "")))

        anyio.run(_run)

    def test_list_and_stats_wrappers_apply_sparse_success_defaults(self) -> None:
        async def _run() -> None:
            class _TaskQueueEngine:
                def list_tasks(self, **_kwargs):
                    return {"status": "success"}

                def get_stats(self, **_kwargs):
                    return {"status": "success"}

            class _WorkflowEngine:
                def list_workflows(self, **_kwargs):
                    return {"status": "success"}

            class _PeerEngine:
                def list_peers(self, **_kwargs):
                    return {"status": "success"}

            with patch.object(
                mcpp_mod,
                "_API",
                new={
                    "TaskQueueEngine": _TaskQueueEngine,
                    "WorkflowEngine": _WorkflowEngine,
                    "PeerEngine": _PeerEngine,
                },
            ):
                task_list = await mcpp_mod.mcplusplus_taskqueue_list(limit=5, offset=2)
                task_stats = await mcpp_mod.mcplusplus_taskqueue_stats(include_worker_stats=True)
                workflow_list = await mcpp_mod.mcplusplus_workflow_list(limit=4, offset=1)
                peer_list = await mcpp_mod.mcplusplus_peer_list(limit=3, offset=1, sort_by="latency")

            self.assertEqual(task_list.get("status"), "success")
            self.assertEqual(task_list.get("tasks"), [])
            self.assertEqual(task_list.get("limit"), 5)
            self.assertEqual(task_list.get("offset"), 2)

            self.assertEqual(task_stats.get("status"), "success")
            self.assertEqual(task_stats.get("stats"), {})
            self.assertEqual(task_stats.get("include_worker_stats"), True)
            self.assertEqual(task_stats.get("include_historical"), False)

            self.assertEqual(workflow_list.get("status"), "success")
            self.assertEqual(workflow_list.get("workflows"), [])
            self.assertEqual(workflow_list.get("limit"), 4)
            self.assertEqual(workflow_list.get("offset"), 1)

            self.assertEqual(peer_list.get("status"), "success")
            self.assertEqual(peer_list.get("peers"), [])
            self.assertEqual(peer_list.get("limit"), 3)
            self.assertEqual(peer_list.get("offset"), 1)
            self.assertEqual(peer_list.get("sort_by"), "latency")

        anyio.run(_run)

    def test_status_result_and_peer_wrappers_apply_sparse_success_defaults(self) -> None:
        async def _run() -> None:
            class _TaskQueueEngine:
                def get_worker_status(self, **_kwargs):
                    return {"status": "success"}

                def get_result(self, **_kwargs):
                    return {"status": "success"}

            class _WorkflowEngine:
                def get_result(self, **_kwargs):
                    return {"status": "success"}

            class _PeerEngine:
                def discover(self, **_kwargs):
                    return {"status": "success"}

                def get_metrics(self, **_kwargs):
                    return {"status": "success"}

            with patch.object(
                mcpp_mod,
                "_API",
                new={
                    "TaskQueueEngine": _TaskQueueEngine,
                    "WorkflowEngine": _WorkflowEngine,
                    "PeerEngine": _PeerEngine,
                },
            ):
                worker_status = await mcpp_mod.mcplusplus_worker_status(
                    "worker-1",
                    include_tasks=True,
                    include_metrics=True,
                )
                task_result = await mcpp_mod.mcplusplus_taskqueue_result(
                    "task-1",
                    include_output=True,
                    include_logs=True,
                )
                workflow_result = await mcpp_mod.mcplusplus_workflow_result(
                    "wf-1",
                    include_outputs=True,
                    include_logs=True,
                )
                peer_discover = await mcpp_mod.mcplusplus_peer_discover(
                    capability_filter=["inference"],
                    max_peers=2,
                    timeout=15,
                    include_metrics=True,
                )
                peer_metrics = await mcpp_mod.mcplusplus_peer_metrics(
                    "peer-1",
                    include_history=True,
                    history_hours=12,
                )

            self.assertEqual(worker_status.get("status"), "success")
            self.assertEqual(worker_status.get("worker_id"), "worker-1")
            self.assertEqual(worker_status.get("include_tasks"), True)
            self.assertEqual(worker_status.get("include_metrics"), True)
            self.assertEqual(worker_status.get("tasks"), [])
            self.assertEqual(worker_status.get("metrics"), {})

            self.assertEqual(task_result.get("status"), "success")
            self.assertEqual(task_result.get("task_id"), "task-1")
            self.assertEqual(task_result.get("include_output"), True)
            self.assertEqual(task_result.get("include_logs"), True)
            self.assertIsNone(task_result.get("output"))
            self.assertEqual(task_result.get("logs"), [])

            self.assertEqual(workflow_result.get("status"), "success")
            self.assertEqual(workflow_result.get("workflow_id"), "wf-1")
            self.assertEqual(workflow_result.get("include_outputs"), True)
            self.assertEqual(workflow_result.get("include_logs"), True)
            self.assertEqual(workflow_result.get("outputs"), {})
            self.assertEqual(workflow_result.get("logs"), [])

            self.assertEqual(peer_discover.get("status"), "success")
            self.assertEqual(peer_discover.get("peers"), [])
            self.assertEqual(peer_discover.get("capability_filter"), ["inference"])
            self.assertEqual(peer_discover.get("max_peers"), 2)
            self.assertEqual(peer_discover.get("timeout"), 15)
            self.assertEqual(peer_discover.get("include_metrics"), True)

            self.assertEqual(peer_metrics.get("status"), "success")
            self.assertEqual(peer_metrics.get("peer_id"), "peer-1")
            self.assertEqual(peer_metrics.get("include_history"), True)
            self.assertEqual(peer_metrics.get("history_hours"), 12)
            self.assertEqual(peer_metrics.get("metrics"), {})
            self.assertEqual(peer_metrics.get("history"), [])

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
