#!/usr/bin/env python3
"""UNI-127 security tool parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.security_tools.native_security_tools import (
    check_access_permission,
    check_access_permissions_batch,
    register_native_security_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI127SecurityTools(unittest.TestCase):
    def test_register_schema_enforces_non_empty_strings(self) -> None:
        manager = _DummyManager()
        register_native_security_tools(manager)
        self.assertEqual(len(manager.calls), 2)

        tool_by_name = {call["name"]: call for call in manager.calls}

        schema = tool_by_name["check_access_permission"]["input_schema"]
        properties = schema["properties"]
        self.assertEqual(properties["resource_id"].get("minLength"), 1)
        self.assertEqual(properties["user_id"].get("minLength"), 1)

        resource_type_anyof = properties["resource_type"]["anyOf"]
        self.assertEqual(resource_type_anyof[0].get("minLength"), 1)

        batch_schema = tool_by_name["check_access_permissions_batch"]["input_schema"]
        self.assertEqual((batch_schema["properties"]["requests"]).get("minItems"), 1)

    def test_check_access_permission_rejects_blank_resource_type(self) -> None:
        async def _run() -> None:
            result = await check_access_permission(
                resource_id="resource-1",
                user_id="user-1",
                permission_type="read",
                resource_type="   ",
            )
            self.assertEqual(result.get("status"), "error")
            self.assertEqual(result.get("allowed"), False)
            self.assertIn("resource_type must be a non-empty string", str(result.get("error", "")))

        anyio.run(_run)

    def test_check_access_permission_normalizes_delegate_dict_payload(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.security_tools.native_security_tools._CHECK_ACCESS_PERMISSION"
            ) as mock_impl:
                async def _impl(**kwargs):
                    return {"allowed": True, "backend": "stub", **kwargs}

                mock_impl.side_effect = _impl

                result = await check_access_permission(
                    resource_id=" resource-1 ",
                    user_id=" user-1 ",
                    permission_type="READ",
                    resource_type="dataset",
                )

                self.assertEqual(result.get("status"), "success")
                self.assertEqual(result.get("allowed"), True)
                self.assertEqual(result.get("resource_id"), "resource-1")
                self.assertEqual(result.get("user_id"), "user-1")
                self.assertEqual(result.get("permission_type"), "read")
                self.assertEqual(result.get("resource_type"), "dataset")

        anyio.run(_run)

    def test_check_access_permission_wraps_non_dict_delegate_payload(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.security_tools.native_security_tools._CHECK_ACCESS_PERMISSION"
            ) as mock_impl:
                async def _impl(**kwargs):
                    _ = kwargs
                    return "granted"

                mock_impl.side_effect = _impl

                result = await check_access_permission(
                    resource_id="resource-1",
                    user_id="user-1",
                )

                self.assertEqual(result.get("status"), "success")
                self.assertEqual(result.get("allowed"), False)
                self.assertEqual(result.get("result"), "granted")
                self.assertEqual(result.get("resource_id"), "resource-1")
                self.assertEqual(result.get("user_id"), "user-1")

        anyio.run(_run)

    def test_check_access_permission_handles_delegate_exception(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.security_tools.native_security_tools._CHECK_ACCESS_PERMISSION"
            ) as mock_impl:
                async def _impl(**kwargs):
                    _ = kwargs
                    raise RuntimeError("backend failure")

                mock_impl.side_effect = _impl

                result = await check_access_permission(
                    resource_id="resource-1",
                    user_id="user-1",
                    permission_type="write",
                )

                self.assertEqual(result.get("status"), "error")
                self.assertEqual(result.get("allowed"), False)
                self.assertIn("backend failure", str(result.get("error", "")))
                self.assertEqual(result.get("permission_type"), "write")

        anyio.run(_run)

    def test_check_access_permissions_batch_requires_non_empty_array(self) -> None:
        async def _run() -> None:
            result = await check_access_permissions_batch(requests=[])
            self.assertEqual(result.get("status"), "error")
            self.assertIn("non-empty array", str(result.get("error", "")))

        anyio.run(_run)

    def test_check_access_permissions_batch_aggregates_results(self) -> None:
        async def _run() -> None:
            result = await check_access_permissions_batch(
                requests=[
                    {"resource_id": "resource-1", "user_id": "user-1", "permission_type": "read"},
                    {"resource_id": "", "user_id": "user-2", "permission_type": "read"},
                ],
                fail_fast=False,
            )
            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("processed"), 2)
            self.assertIn("allowed_count", result)
            self.assertIn("error_count", result)
            self.assertIn("results", result)
            self.assertEqual(len(result["results"]), 2)

        anyio.run(_run)

    def test_check_access_permissions_batch_honors_fail_fast(self) -> None:
        async def _run() -> None:
            result = await check_access_permissions_batch(
                requests=[
                    {"resource_id": "", "user_id": "user-1", "permission_type": "read"},
                    {"resource_id": "resource-2", "user_id": "user-2", "permission_type": "read"},
                ],
                fail_fast=True,
            )
            self.assertEqual(result.get("status").lower(), "success")
            self.assertEqual(result.get("processed"), 1)
            self.assertEqual(result.get("requested"), 2)
            self.assertGreaterEqual(int(result.get("error_count", 0)), 1)

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
