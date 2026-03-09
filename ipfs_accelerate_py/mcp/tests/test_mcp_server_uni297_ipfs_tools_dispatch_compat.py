#!/usr/bin/env python3
"""UNI-297 ipfs-tools dispatch compatibility tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import anyio

import ipfs_accelerate_py.mcp_server.tools.ipfs_tools.native_ipfs_tools_category as native_ipfs_tools

from ipfs_accelerate_py.mcp.server import create_mcp_server


class TestMCPServerUNI297IPFSToolsDispatchCompat(unittest.TestCase):
    def _assert_dispatch_success_envelope(self, response: dict) -> dict:
        self.assertIsInstance(response, dict)
        self.assertTrue(response.get("ok"), response)
        self.assertIn("result", response)
        self.assertIsInstance(response["result"], dict)
        return response["result"]

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_ipfs_tools_dispatch_preserves_validation_and_passthrough_contracts(self, mock_wrapper) -> None:
        class DummyServer:
            def __init__(self):
                self.tools = {}
                self.mcp = None

            def register_tool(
                self,
                name,
                function,
                description,
                input_schema,
                execution_context=None,
                tags=None,
            ):
                self.tools[name] = {
                    "function": function,
                    "description": description,
                    "input_schema": input_schema,
                    "execution_context": execution_context,
                    "tags": tags,
                }

        mock_wrapper.return_value = DummyServer()

        captured_pin: dict = {}
        captured_get: dict = {}

        async def _fake_pin_to_ipfs(**kwargs):
            captured_pin.update(kwargs)
            return {
                "status": "success",
                "cid": "QmPinnedFromDispatch",
                "content_path": str(kwargs.get("content_source")),
            }

        async def _fake_get_from_ipfs(**kwargs):
            captured_get.update(kwargs)
            return {
                "status": "success",
                "cid": kwargs.get("cid"),
                "output_path": kwargs.get("output_path"),
            }

        async def _run_flow() -> None:
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                },
                clear=False,
            ):
                with patch.dict(
                    native_ipfs_tools._API,
                    {
                        "pin_to_ipfs": _fake_pin_to_ipfs,
                        "get_from_ipfs": _fake_get_from_ipfs,
                    },
                    clear=False,
                ):
                    server = create_mcp_server(name="ipfs-tools-dispatch-compat")
                    dispatch = server.tools["tools_dispatch"]["function"]

                    pinned = self._assert_dispatch_success_envelope(
                        await dispatch(
                            "ipfs_tools",
                            "pin_to_ipfs",
                            {
                                "content_source": "  /tmp/dispatch-demo.txt  ",
                                "recursive": False,
                                "wrap_with_directory": True,
                                "hash_algo": " blake3 ",
                            },
                        )
                    )
                    self.assertEqual(pinned.get("status"), "success")
                    self.assertEqual(pinned.get("cid"), "QmPinnedFromDispatch")
                    self.assertEqual(captured_pin.get("content_source"), "/tmp/dispatch-demo.txt")
                    self.assertFalse(captured_pin.get("recursive"))
                    self.assertTrue(captured_pin.get("wrap_with_directory"))
                    self.assertEqual(captured_pin.get("hash_algo"), "blake3")

                    fetched = self._assert_dispatch_success_envelope(
                        await dispatch(
                            "ipfs_tools",
                            "get_from_ipfs",
                            {
                                "cid": "  QmDispatchCid  ",
                                "output_path": "  /tmp/fetched.bin  ",
                                "timeout_seconds": 12,
                                "gateway": "https://ipfs.io/",
                            },
                        )
                    )
                    self.assertEqual(fetched.get("status"), "success")
                    self.assertEqual(fetched.get("cid"), "QmDispatchCid")
                    self.assertEqual(captured_get.get("cid"), "QmDispatchCid")
                    self.assertEqual(captured_get.get("output_path"), "/tmp/fetched.bin")
                    self.assertEqual(captured_get.get("timeout_seconds"), 12)
                    self.assertEqual(captured_get.get("gateway"), "https://ipfs.io")

                    invalid_gateway = self._assert_dispatch_success_envelope(
                        await dispatch(
                            "ipfs_tools",
                            "get_from_ipfs",
                            {
                                "cid": "QmDispatchCid",
                                "gateway": "ipfs://gateway.invalid",
                            },
                        )
                    )
                    self.assertEqual(invalid_gateway.get("status"), "error")
                    self.assertIn("must start with", str(invalid_gateway.get("message", "")))

                    invalid_pin = self._assert_dispatch_success_envelope(
                        await dispatch(
                            "ipfs_tools",
                            "pin_to_ipfs",
                            {
                                "content_source": 123,
                            },
                        )
                    )
                    self.assertEqual(invalid_pin.get("status"), "error")
                    self.assertIn("'content_source'", str(invalid_pin.get("message", "")))

        anyio.run(_run_flow)


if __name__ == "__main__":
    unittest.main()
