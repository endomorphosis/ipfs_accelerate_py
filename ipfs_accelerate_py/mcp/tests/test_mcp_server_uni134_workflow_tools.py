#!/usr/bin/env python3
"""UNI-134 workflow tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.workflow_tools import native_workflow_tools_category
from ipfs_accelerate_py.mcp_server.tools.workflow_tools.native_workflow_tools_category import (
    calculate_peer_distance,
    create_template,
    enhanced_batch_processing,
    enhanced_data_pipeline,
    enhanced_workflow_management,
    get_workflow_tags,
    merge_merkle_clock,
    register_native_workflow_tools_category,
    resume_workflow,
    schedule_p2p_workflow,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI134WorkflowTools(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_workflow_tools_category(manager)
        by_name = {c["name"]: c for c in manager.calls}

        template_schema = by_name["create_template"]["input_schema"]
        self.assertEqual(
            template_schema["properties"]["template"].get("minProperties"),
            1,
        )

        schedule_schema = by_name["schedule_p2p_workflow"]["input_schema"]
        props = schedule_schema["properties"]
        self.assertEqual(props["workflow_id"].get("minLength"), 1)
        self.assertEqual(props["tags"].get("minItems"), 1)
        self.assertEqual(props["priority"].get("minimum"), 0)

        enhanced_management_schema = by_name["enhanced_workflow_management"]["input_schema"]
        self.assertIn("create", enhanced_management_schema["properties"]["action"].get("enum", []))

        enhanced_batch_schema = by_name["enhanced_batch_processing"]["input_schema"]
        self.assertEqual(enhanced_batch_schema["properties"]["operation_type"].get("minLength"), 1)

        enhanced_pipeline_schema = by_name["enhanced_data_pipeline"]["input_schema"]
        self.assertEqual(enhanced_pipeline_schema["properties"]["pipeline_config"].get("minProperties"), 1)

    def test_create_template_rejects_empty_template(self) -> None:
        async def _run() -> None:
            result = await create_template(template={})
            self.assertEqual(result.get("status"), "error")
            self.assertIn("template must be a non-empty object", str(result.get("error", "")))

        anyio.run(_run)

    def test_resume_workflow_rejects_blank_workflow_id(self) -> None:
        async def _run() -> None:
            result = await resume_workflow(workflow_id="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("workflow_id must be a non-empty string", str(result.get("error", "")))

        anyio.run(_run)

    def test_calculate_peer_distance_rejects_blank_hash(self) -> None:
        async def _run() -> None:
            result = await calculate_peer_distance(hash1="", hash2="abc123")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("hash1 must be a non-empty string", str(result.get("error", "")))

        anyio.run(_run)

    def test_merge_merkle_clock_rejects_negative_counter(self) -> None:
        async def _run() -> None:
            result = await merge_merkle_clock(other_peer_id="peer-a", other_counter=-1)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("other_counter must be an integer >= 0", str(result.get("error", "")))

        anyio.run(_run)

    def test_schedule_p2p_workflow_success_envelope_shape(self) -> None:
        async def _run() -> None:
            result = await schedule_p2p_workflow(
                workflow_id="wf-1",
                name="demo",
                tags=["smoke"],
            )
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("workflow_id"), "wf-1")
            self.assertEqual(result.get("name"), "demo")

        anyio.run(_run)

    def test_schedule_p2p_workflow_minimal_success_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.workflow_tools.native_workflow_tools_category._API"
            ) as mock_api:
                mock_api.__getitem__.return_value = lambda **_: {"status": "success"}

                result = await schedule_p2p_workflow(
                    workflow_id="wf-1",
                    name="demo",
                    tags=["smoke"],
                    priority=2,
                )

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("workflow_id"), "wf-1")
            self.assertEqual(result.get("name"), "demo")
            self.assertEqual(result.get("tags"), ["smoke"])
            self.assertEqual(result.get("priority"), 2.0)
            self.assertEqual(result.get("metadata"), {})

        anyio.run(_run)

    def test_calculate_peer_distance_minimal_success_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.workflow_tools.native_workflow_tools_category._API"
            ) as mock_api:
                mock_api.__getitem__.return_value = lambda **_: {"status": "success"}

                result = await calculate_peer_distance(hash1="abc", hash2="def")

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("hash1"), "abc")
            self.assertEqual(result.get("hash2"), "def")
            self.assertEqual(result.get("distance"), 0)

        anyio.run(_run)

    def test_merge_merkle_clock_minimal_success_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.workflow_tools.native_workflow_tools_category._API"
            ) as mock_api:
                mock_api.__getitem__.return_value = lambda **_: {"status": "success"}

                result = await merge_merkle_clock(
                    other_peer_id="peer-a",
                    other_counter=3,
                    other_parent_hash="h1",
                    other_timestamp=123.0,
                )

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("other_peer_id"), "peer-a")
            self.assertEqual(result.get("other_counter"), 3)
            self.assertEqual(result.get("other_parent_hash"), "h1")
            self.assertEqual(result.get("other_timestamp"), 123.0)

        anyio.run(_run)

    def test_get_workflow_tags_minimal_success_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.workflow_tools.native_workflow_tools_category._API"
            ) as mock_api:
                mock_api.__getitem__.return_value = lambda: {"success": True}

                result = await get_workflow_tags()

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("tags"), [])

        anyio.run(_run)

    def test_enhanced_workflow_management_rejects_missing_definition_for_create(self) -> None:
        async def _run() -> None:
            result = await enhanced_workflow_management(action="create")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("workflow_definition must be a non-empty object", str(result.get("error", "")))

        anyio.run(_run)

    def test_enhanced_batch_processing_rejects_empty_data_source(self) -> None:
        async def _run() -> None:
            result = await enhanced_batch_processing(
                operation_type="reindex",
                data_source={},
                output_config={"destination": "/tmp/out"},
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("data_source must be a non-empty object", str(result.get("error", "")))

        anyio.run(_run)

    def test_enhanced_data_pipeline_rejects_missing_extract(self) -> None:
        async def _run() -> None:
            result = await enhanced_data_pipeline(
                pipeline_config={"name": "demo", "load": {"destination_type": "file"}},
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("must include 'extract'", str(result.get("error", "")))

        anyio.run(_run)

    def test_enhanced_workflow_success_shapes(self) -> None:
        async def _run() -> None:
            manage_result = await enhanced_workflow_management(
                action="list",
                status_filter="active",
            )
            self.assertEqual(manage_result.get("status"), "success")
            self.assertIn("workflows", manage_result)

            batch_result = await enhanced_batch_processing(
                operation_type="reindex",
                data_source={"source_type": "dataset", "name": "demo"},
                output_config={"destination": "/tmp/out"},
            )
            self.assertEqual(batch_result.get("status"), "success")
            self.assertIn("processing_completed", batch_result)

            pipeline_result = await enhanced_data_pipeline(
                pipeline_config={
                    "name": "demo-pipeline",
                    "extract": {"source_type": "dataset"},
                    "load": {"destination_type": "file"},
                },
            )
            self.assertEqual(pipeline_result.get("status"), "success")
            self.assertIn("pipeline_name", pipeline_result)

        anyio.run(_run)

    def test_workflow_wrappers_infer_error_status_from_contradictory_delegate_payloads(self) -> None:
        async def _contradictory_failure(**_: object) -> dict:
            return {"status": "success", "success": False, "error": "delegate failed"}

        async def _run() -> None:
            with patch.dict(
                native_workflow_tools_category._API,
                {
                    "create_template": _contradictory_failure,
                    "schedule_p2p_workflow": _contradictory_failure,
                    "calculate_peer_distance": _contradictory_failure,
                    "merge_merkle_clock": _contradictory_failure,
                    "get_workflow_tags": _contradictory_failure,
                },
                clear=False,
            ):
                created = await create_template(template={"name": "demo"})
                scheduled = await schedule_p2p_workflow(
                    workflow_id="wf-1",
                    name="demo",
                    tags=["smoke"],
                )
                distance = await calculate_peer_distance(hash1="abc", hash2="def")
                merged = await merge_merkle_clock(other_peer_id="peer-a", other_counter=1)
                tags = await get_workflow_tags()

            self.assertEqual(created.get("status"), "error")
            self.assertEqual(created.get("error"), "delegate failed")

            self.assertEqual(scheduled.get("status"), "error")
            self.assertEqual(scheduled.get("error"), "delegate failed")

            self.assertEqual(distance.get("status"), "error")
            self.assertEqual(distance.get("error"), "delegate failed")

            self.assertEqual(merged.get("status"), "error")
            self.assertEqual(merged.get("error"), "delegate failed")

            self.assertEqual(tags.get("status"), "error")
            self.assertEqual(tags.get("error"), "delegate failed")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
