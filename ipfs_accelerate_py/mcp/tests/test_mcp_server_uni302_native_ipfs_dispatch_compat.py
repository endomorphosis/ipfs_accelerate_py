#!/usr/bin/env python3
"""UNI-302 native IPFS dispatch compatibility tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp.server import create_mcp_server


class _Result:
    def __init__(self, success: bool = True, data=None, error=None) -> None:
        self.success = success
        self.data = {} if data is None else data
        self.error = error


class _FakeKit:
    def validate_cid(self, cid: str):
        return _Result(success=True, data={"cid": cid, "valid": True})

    def list_files(self, path: str):
        return _Result(success=True, data={"path": path, "entries": []})

    def add_file(self, path: str, pin: bool):
        return _Result(success=True, data={"path": path, "pin": pin})

    def pin_file(self, cid: str):
        return _Result(success=False, data=None, error=f"pin failed for {cid}")

    def unpin_file(self, cid: str):
        return _Result(success=True, data={"cid": cid, "pinned": False})

    def get_file(self, cid: str, output_path: str):
        return _Result(success=True, data={"cid": cid, "output_path": output_path})

    def cat_file(self, cid: str):
        return _Result(success=True, data={"cid": cid, "content": "hello"})


class TestMCPServerUNI302NativeIPFSDispatchCompat(unittest.TestCase):
    def _assert_dispatch_success_envelope(self, response: dict) -> dict:
        self.assertIsInstance(response, dict)
        self.assertTrue(response.get("ok"), response)
        self.assertIn("result", response)
        self.assertIsInstance(response["result"], dict)
        return response["result"]

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_ipfs_dispatch_preserves_success_validation_and_failed_kit_contracts(
        self, mock_wrapper
    ) -> None:
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

        async def _run_flow() -> None:
            with patch.dict(
                os.environ,
                {
                    "IPFS_MCP_ENABLE_UNIFIED_BRIDGE": "1",
                    "IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP": "1",
                },
                clear=False,
            ), patch(
                "ipfs_accelerate_py.mcp_server.tools.ipfs.native_ipfs_tools.get_ipfs_files_kit",
                return_value=_FakeKit(),
            ):
                server = create_mcp_server(name="native-ipfs-dispatch-compat")
                dispatch = server.tools["tools_dispatch"]["function"]

                validated = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "ipfs",
                        "ipfs_files_validate_cid",
                        {"cid": "bafy-dispatch-demo"},
                    )
                )
                self.assertEqual(validated.get("status"), "success")
                self.assertTrue(validated.get("success"))
                self.assertEqual((validated.get("data") or {}).get("cid"), "bafy-dispatch-demo")

                listed = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "ipfs",
                        "ipfs_files_list_files",
                        {"path": "/"},
                    )
                )
                self.assertEqual(listed.get("status"), "success")
                self.assertTrue(listed.get("success"))
                self.assertEqual((listed.get("data") or {}).get("path"), "/")

                fetched = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "ipfs",
                        "ipfs_files_get_file",
                        {"cid": "bafy-dispatch-demo", "output_path": "/tmp/out.txt"},
                    )
                )
                self.assertEqual(fetched.get("status"), "success")
                self.assertTrue(fetched.get("success"))
                self.assertEqual((fetched.get("data") or {}).get("output_path"), "/tmp/out.txt")

                failed_pin = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "ipfs",
                        "ipfs_files_pin_file",
                        {"cid": "bafy-dispatch-demo"},
                    )
                )
                self.assertEqual(failed_pin.get("status"), "error")
                self.assertFalse(failed_pin.get("success"))
                self.assertEqual(failed_pin.get("error"), "pin failed for bafy-dispatch-demo")

                invalid_add = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "ipfs",
                        "ipfs_files_add_file",
                        {"path": "/tmp/in.txt", "pin": "yes"},
                    )
                )
                self.assertEqual(invalid_add.get("status"), "error")
                self.assertFalse(invalid_add.get("success"))
                self.assertEqual(invalid_add.get("error"), "pin must be a boolean")

        anyio.run(_run_flow)

    @patch("ipfs_accelerate_py.mcp.server.MCPServerWrapper")
    def test_native_ipfs_dispatch_infers_error_status_from_contradictory_delegate_payloads(
        self, mock_wrapper
    ) -> None:
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

        class _ContradictoryKit:
            def validate_cid(self, cid: str):
                return {"status": "success", "success": False, "error": f"invalid {cid}"}

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
                "ipfs_accelerate_py.mcp_server.tools.ipfs.native_ipfs_tools.get_ipfs_files_kit",
                return_value=_ContradictoryKit(),
            ):
                server = create_mcp_server(name="native-ipfs-dispatch-contradictory")
                dispatch = server.tools["tools_dispatch"]["function"]

                failed_validate = self._assert_dispatch_success_envelope(
                    await dispatch(
                        "ipfs",
                        "ipfs_files_validate_cid",
                        {"cid": "bafy-dispatch-demo"},
                    )
                )
                self.assertEqual(failed_validate.get("status"), "error")
                self.assertFalse(failed_validate.get("success"))
                self.assertEqual(failed_validate.get("error"), "invalid bafy-dispatch-demo")

        anyio.run(_run_flow)


if __name__ == "__main__":
    unittest.main()
