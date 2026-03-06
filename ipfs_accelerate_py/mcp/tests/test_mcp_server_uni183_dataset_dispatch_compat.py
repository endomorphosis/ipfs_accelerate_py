#!/usr/bin/env python3
"""UNI-183 dataset dispatch compatibility tests for focused parity coverage."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp.server import create_mcp_server
from ipfs_accelerate_py.mcp_server.tools.dataset_tools import native_dataset_tools


class TestMCPServerUNI183DatasetDispatchCompat(unittest.TestCase):
    def _assert_dispatch_success_envelope(self, response: dict) -> dict:
        self.assertIsInstance(response, dict)
        self.assertTrue(response.get("ok"), response)
        self.assertIn("result", response)
        self.assertIsInstance(response["result"], dict)
        return response["result"]

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_dataset_dispatch_preserves_schema_and_passthrough_fields(self, mock_wrapper) -> None:
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

        mock_wrapper.return_value = DummyServer()

        async def _process_dataset(**kwargs: object) -> dict:
            operations = kwargs.get("operations") or []
            return {
                "status": "success",
                "dataset_id": kwargs.get("output_id") or "processed-1",
                "num_operations": len(operations),
                "operations_summary": [item.get("type", "unknown") for item in operations if isinstance(item, dict)],
            }

        async def _convert_dataset_format(**kwargs: object) -> dict:
            dataset_id = str(kwargs.get("dataset_id"))
            target_format = str(kwargs.get("target_format"))
            return {
                "status": "success",
                "original_dataset_id": dataset_id,
                "dataset_id": f"converted-{dataset_id}-{target_format}",
                "target_format": target_format,
            }

        async def _run_flow() -> None:
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                },
                clear=False,
            ), patch.dict(
                native_dataset_tools._API,
                {
                    "process_dataset": _process_dataset,
                    "convert_dataset_format": _convert_dataset_format,
                },
                clear=False,
            ):
                server = create_mcp_server(name="dataset-dispatch-compat")

                get_schema = server.tools["tools_get_schema"]["function"]
                dispatch = server.tools["tools_dispatch"]["function"]

                process_schema = await get_schema("dataset_tools", "process_dataset")
                process_props = (process_schema.get("input_schema") or {}).get("properties", {})
                self.assertIn("array", (process_props.get("operations") or {}).get("type", []))

                convert_schema = await get_schema("dataset_tools", "convert_dataset_format")
                convert_props = (convert_schema.get("input_schema") or {}).get("properties", {})
                self.assertEqual((convert_props.get("dataset_id") or {}).get("minLength"), 1)

                processed = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "dataset_tools",
                        "process_dataset",
                        {
                            "dataset_source": "dataset://demo",
                            "operations": [{"type": "filter"}],
                            "output_id": "processed-compat",
                        },
                    )
                )
                self.assertEqual(processed.get("status"), "success")
                self.assertEqual(processed.get("dataset_id"), "processed-compat")
                self.assertEqual(processed.get("num_operations"), 1)
                self.assertEqual(processed.get("operations_summary"), ["filter"])

                converted = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "dataset_tools",
                        "convert_dataset_format",
                        {
                            "dataset_id": "dataset-1",
                            "target_format": "parquet",
                        },
                    )
                )
                self.assertEqual(converted.get("status"), "success")
                self.assertEqual(converted.get("original_dataset_id"), "dataset-1")
                self.assertEqual(converted.get("target_format"), "parquet")

        anyio.run(_run_flow)


if __name__ == "__main__":
    unittest.main()