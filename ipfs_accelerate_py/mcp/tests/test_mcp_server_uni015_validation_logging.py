#!/usr/bin/env python3
"""Validation/logging compatibility tests for UNI-015 surfaces."""

from __future__ import annotations

import logging
import os
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp.server import create_mcp_server
from ipfs_accelerate_py.mcp_server.exceptions import ValidationError
from ipfs_accelerate_py.mcp_server.logger import configure_root_logging, get_logger
from ipfs_accelerate_py.mcp_server.validators import EnhancedParameterValidator, validate_dispatch_inputs


class TestMCPServerUNI015ValidationLogging(unittest.TestCase):
    def test_validate_dispatch_inputs_rejects_invalid_category(self) -> None:
        with self.assertRaisesRegex(ValidationError, "category"):
            validate_dispatch_inputs(category=123, tool_name="echo", parameters={})

    def test_validate_dispatch_inputs_normalizes_payload_shape(self) -> None:
        category, tool_name, payload = validate_dispatch_inputs(
            category=" smoke ",
            tool_name=" echo ",
            parameters=["not", "a", "dict"],
        )
        self.assertEqual(category, "smoke")
        self.assertEqual(tool_name, "echo")
        self.assertEqual(payload, {})

    def test_enhanced_parameter_validator_collection_cache_contract(self) -> None:
        instance = EnhancedParameterValidator()
        self.assertEqual(instance.validate_collection_name("dataset_cache_1"), "dataset_cache_1")
        self.assertEqual(instance.validate_collection_name("dataset_cache_1"), "dataset_cache_1")
        self.assertGreaterEqual(instance.performance_metrics["cache_hits"], 1)

    def test_get_logger_returns_python_logger(self) -> None:
        logger = get_logger("ipfs_accelerate_py.mcp_server.tests.uni015")
        self.assertIsInstance(logger, logging.Logger)

    @patch("ipfs_accelerate_py.mcp_server.logger.logging.basicConfig")
    @patch("ipfs_accelerate_py.mcp_server.logger.logging.FileHandler", side_effect=OSError("no file"))
    @patch("ipfs_accelerate_py.mcp_server.logger.logging.getLogger")
    def test_configure_root_logging_falls_back_to_stream_handler(
        self,
        mock_get_logger,
        _mock_file_handler,
        mock_basic_config,
    ) -> None:
        mock_get_logger.return_value = type("Root", (), {"handlers": []})()

        configure_root_logging()

        kwargs = mock_basic_config.call_args.kwargs
        handlers = kwargs.get("handlers", [])
        self.assertEqual(len(handlers), 1)
        self.assertIsInstance(handlers[0], logging.StreamHandler)

    @patch("ipfs_accelerate_py.mcp_server.logger.logging.basicConfig")
    @patch("ipfs_accelerate_py.mcp_server.logger.logging.getLogger")
    def test_configure_root_logging_skips_when_handlers_present(self, mock_get_logger, mock_basic_config) -> None:
        mock_get_logger.return_value = type("Root", (), {"handlers": [object()]})()

        configure_root_logging()

        mock_basic_config.assert_not_called()

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_tools_dispatch_rejects_non_string_category(self, mock_wrapper) -> None:
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

        with patch.dict(
            os.environ,
            {
                "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
            },
            clear=False,
        ):
            server = create_mcp_server(name="dispatch-invalid-category")

        async def _run_flow() -> None:
            async def echo(value: str):
                return {"echo": value}

            server._unified_tool_manager.register_tool("smoke", "echo", echo, description="echo smoke")
            dispatch = server.tools["tools_dispatch"]["function"]

            response = await dispatch(123, "echo", {"value": "ok"})

            self.assertFalse(response["ok"])
            self.assertEqual(response["error"], "invalid_dispatch_parameter")
            self.assertIn("category", response.get("details", ""))

        anyio.run(_run_flow)


if __name__ == "__main__":
    unittest.main()
