#!/usr/bin/env python3
"""UNI-101 security tool parity hardening tests."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.security_tools.native_security_tools import (
    check_access_permission,
    register_native_security_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI101SecurityTools(unittest.TestCase):
    def test_register_schema_requires_user_id(self) -> None:
        manager = _DummyManager()
        register_native_security_tools(manager)
        self.assertEqual(len(manager.calls), 2)

        tool_by_name = {call["name"]: call for call in manager.calls}

        schema = tool_by_name["check_access_permission"]["input_schema"]
        self.assertIn("required", schema)
        self.assertEqual(schema["required"], ["resource_id", "user_id"])
        self.assertEqual(schema["properties"]["permission_type"]["default"], "read")

    def test_check_access_permission_rejects_missing_user(self) -> None:
        async def _run() -> None:
            result = await check_access_permission(resource_id="abc", user_id=None)
            self.assertEqual(result.get("status"), "error")
            self.assertEqual(result.get("allowed"), False)
            self.assertIn("user_id must be provided", str(result.get("error", "")))

        anyio.run(_run)

    def test_check_access_permission_rejects_invalid_permission_type(self) -> None:
        async def _run() -> None:
            result = await check_access_permission(
                resource_id="abc",
                user_id="user-1",
                permission_type="godmode",
            )
            self.assertEqual(result.get("status"), "error")
            self.assertEqual(result.get("allowed"), False)
            self.assertIn("permission_type must be one of", str(result.get("error", "")))

        anyio.run(_run)

    def test_check_access_permission_normalizes_and_delegates(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.security_tools.native_security_tools._CHECK_ACCESS_PERMISSION"
            ) as mock_impl:
                async def _impl(**kwargs):
                    return {"status": "success", **kwargs}

                mock_impl.side_effect = _impl

                result = await check_access_permission(
                    resource_id="  abc  ",
                    user_id="  user-1  ",
                    permission_type="READ",
                )

                self.assertEqual(result.get("status"), "success")
                self.assertEqual(result.get("resource_id"), "abc")
                self.assertEqual(result.get("user_id"), "user-1")
                self.assertEqual(result.get("permission_type"), "read")

        anyio.run(_run)

    def test_check_access_permission_supports_json_string_entrypoint(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.security_tools.native_security_tools._CHECK_ACCESS_PERMISSION"
            ) as mock_impl:
                async def _impl(**kwargs):
                    return {"status": "success", "allowed": True, **kwargs}

                mock_impl.side_effect = _impl

                result = await check_access_permission(
                    json.dumps(
                        {
                            "resource_id": "resource-1",
                            "user_id": "user-1",
                            "permission_type": "write",
                            "resource_type": "dataset",
                        }
                    )
                )

            payload = json.loads(result["content"][0]["text"])
            self.assertEqual(payload.get("status"), "success")
            self.assertEqual(payload.get("allowed"), True)
            self.assertEqual(payload.get("resource_id"), "resource-1")
            self.assertEqual(payload.get("user_id"), "user-1")
            self.assertEqual(payload.get("permission_type"), "write")

        anyio.run(_run)

    def test_check_access_permission_json_string_requires_fields(self) -> None:
        async def _run() -> None:
            result = await check_access_permission(json.dumps({"resource_id": "resource-1"}))
            payload = json.loads(result["content"][0]["text"])
            self.assertEqual(payload.get("status"), "error")
            self.assertIn("Missing required field: user_id", str(payload.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
