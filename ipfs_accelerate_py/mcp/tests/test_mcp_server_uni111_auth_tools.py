#!/usr/bin/env python3
"""UNI-111 auth tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.auth_tools.native_auth_tools import (
    authenticate_user,
    get_user_info,
    register_native_auth_tools,
    validate_token,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI111AuthTools(unittest.TestCase):
    def test_register_includes_auth_tools(self) -> None:
        manager = _DummyManager()
        register_native_auth_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("authenticate_user", names)
        self.assertIn("validate_token", names)
        self.assertIn("get_user_info", names)

    def test_validate_token_schema_contract(self) -> None:
        manager = _DummyManager()
        register_native_auth_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}

        auth_schema = by_name["authenticate_user"]["input_schema"]
        self.assertEqual(auth_schema["properties"]["remember_me"]["default"], False)

        schema = by_name["validate_token"]["input_schema"]
        action = schema["properties"]["action"]
        self.assertEqual(action.get("default"), "validate")
        self.assertIn("decode", action.get("enum", []))
        self.assertEqual(schema["properties"]["strict"]["default"], False)

        user_info_schema = by_name["get_user_info"]["input_schema"]
        self.assertEqual(user_info_schema["properties"]["include_permissions"]["default"], True)
        self.assertEqual(user_info_schema["properties"]["include_profile"]["default"], True)

    def test_authenticate_user_rejects_invalid_username(self) -> None:
        async def _run() -> None:
            result = await authenticate_user(username="", password="pw")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("Username is required", str(result.get("message", "")))

        anyio.run(_run)

    def test_validate_token_rejects_invalid_action_and_permission(self) -> None:
        async def _run() -> None:
            bad_action = await validate_token(token="tok", action="bad")
            self.assertEqual(bad_action.get("status"), "error")
            self.assertEqual(bad_action.get("valid"), False)
            self.assertIn("Invalid action", str(bad_action.get("message", "")))

            bad_permission = await validate_token(token="tok", required_permission="admin")
            self.assertEqual(bad_permission.get("status"), "error")
            self.assertEqual(bad_permission.get("valid"), False)
            self.assertIn("Invalid required_permission", str(bad_permission.get("message", "")))

            bad_strict = await validate_token(token="tok", strict="yes")
            self.assertEqual(bad_strict.get("status"), "error")
            self.assertEqual(bad_strict.get("valid"), False)
            self.assertIn("strict must be a boolean", str(bad_strict.get("message", "")))

        anyio.run(_run)

    def test_get_user_info_requires_token(self) -> None:
        async def _run() -> None:
            result = await get_user_info(token=" ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("Token is required", str(result.get("message", "")))

            bad_permissions = await get_user_info(token="tok", include_permissions="yes")
            self.assertEqual(bad_permissions.get("status"), "error")
            self.assertIn("include_permissions must be a boolean", str(bad_permissions.get("message", "")))

        anyio.run(_run)

    def test_validate_token_decode_shape(self) -> None:
        async def _run() -> None:
            result = await validate_token(token="dummy", action="decode")
            self.assertIn(result.get("status"), ["success", "error"])
            if result.get("status") == "success":
                self.assertIn("message", result)

            auth = await authenticate_user(username="demo", password="pw", remember_me=True)
            self.assertIn(auth.get("status"), ["success", "error"])
            self.assertEqual(auth.get("remember_me"), True)

            info = await get_user_info(token="dummy", include_permissions=False, include_profile=False)
            self.assertIn(info.get("status"), ["success", "error"])
            self.assertEqual(info.get("include_permissions"), False)
            self.assertEqual(info.get("include_profile"), False)

            strict_validation = await validate_token(token="dummy", strict=True)
            self.assertIn(strict_validation.get("status"), ["success", "error"])
            self.assertEqual(strict_validation.get("strict"), True)

        anyio.run(_run)

    def test_authenticate_user_sets_enhanced_authentication_payload(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.auth_tools.native_auth_tools._API"
            ) as mock_api:
                async def _authenticate(**kwargs):
                    self.assertEqual(kwargs["username"], "demo")
                    self.assertEqual(kwargs["password"], "pw")
                    return {
                        "status": "success",
                        "username": "demo",
                        "access_token": "token-123",
                        "token_type": "bearer",
                        "role": "user",
                        "expires_in": 3600,
                    }

                mock_api.__getitem__.side_effect = {
                    "authenticate_user": _authenticate,
                    "validate_token": None,
                    "get_user_info": None,
                }.__getitem__

                result = await authenticate_user(username="demo", password="pw", remember_me=True)

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("expires_in"), 86400 * 7)
            self.assertEqual(result.get("message"), "Authentication successful")
            self.assertEqual((result.get("authentication") or {}).get("access_token"), "token-123")
            self.assertEqual((result.get("authentication") or {}).get("expires_in"), 86400 * 7)

        anyio.run(_run)

    def test_validate_token_sets_enhanced_nested_payloads(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.auth_tools.native_auth_tools._API"
            ) as mock_api:
                async def _validate(**kwargs):
                    action = kwargs["action"]
                    if action == "refresh":
                        return {
                            "status": "success",
                            "access_token": "new-token",
                            "refresh_token": "refresh-token",
                            "expires_in": 3600,
                            "token_type": "bearer",
                        }
                    if action == "decode":
                        return {
                            "status": "success",
                            "user_id": "user-1",
                            "username": "demo",
                            "exp": 12345,
                            "role": "admin",
                        }
                    return {
                        "status": "success",
                        "valid": True,
                        "username": "demo",
                        "role": "admin",
                        "permissions": ["manage"],
                        "expires_in": 7200,
                        "has_required_permission": True,
                    }

                mock_api.__getitem__.side_effect = {
                    "authenticate_user": None,
                    "validate_token": _validate,
                    "get_user_info": None,
                }.__getitem__

                refreshed = await validate_token(token="tok", action="refresh")
                decoded = await validate_token(token="tok", action="decode")
                validated = await validate_token(token="tok", required_permission="manage", strict=True)

            self.assertEqual((refreshed.get("refresh_result") or {}).get("access_token"), "new-token")
            self.assertEqual(decoded.get("message"), "Token decoded successfully")
            self.assertEqual((decoded.get("decoded_token") or {}).get("user_id"), "user-1")
            self.assertEqual((validated.get("validation_result") or {}).get("username"), "demo")
            self.assertEqual((validated.get("validation_result") or {}).get("has_required_permission"), True)
            self.assertEqual(validated.get("strict"), True)

        anyio.run(_run)

    def test_auth_tools_apply_sparse_success_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.auth_tools.native_auth_tools._API"
            ) as mock_api:
                async def _authenticate(**_kwargs):
                    return {"status": "success"}

                async def _validate(**kwargs):
                    return {"status": "success"}

                async def _user_info(**_kwargs):
                    return {"status": "success", "username": "demo", "role": "admin"}

                mock_api.__getitem__.side_effect = {
                    "authenticate_user": _authenticate,
                    "validate_token": _validate,
                    "get_user_info": _user_info,
                }.__getitem__

                auth_result = await authenticate_user(username="demo", password="pw", remember_me=True)
                refresh_result = await validate_token(token="tok", action="refresh")
                decode_result = await validate_token(token="tok", action="decode")
                validate_result = await validate_token(token="tok", action="validate")
                user_info_result = await get_user_info(token="tok", include_permissions=True, include_profile=True)

            self.assertEqual(auth_result.get("status"), "success")
            self.assertEqual(auth_result.get("username"), "demo")
            self.assertEqual(auth_result.get("token_type"), "bearer")
            self.assertEqual(auth_result.get("role"), "user")
            self.assertEqual(auth_result.get("expires_in"), 86400 * 7)
            self.assertEqual((auth_result.get("authentication") or {}).get("username"), "demo")

            self.assertEqual(refresh_result.get("status"), "success")
            self.assertEqual(refresh_result.get("message"), "Token refreshed successfully")
            self.assertEqual(refresh_result.get("token_type"), "bearer")
            self.assertEqual(refresh_result.get("expires_in"), 3600)
            self.assertEqual(refresh_result.get("refresh_result"), {"expires_in": 3600, "token_type": "bearer"})

            self.assertEqual(decode_result.get("status"), "success")
            self.assertEqual(decode_result.get("message"), "Token decoded successfully")
            self.assertEqual(decode_result.get("decoded_token"), {})

            self.assertEqual(validate_result.get("status"), "success")
            self.assertTrue(validate_result.get("valid"))
            self.assertEqual(validate_result.get("permissions"), [])
            self.assertEqual((validate_result.get("validation_result") or {}).get("valid"), True)

            self.assertEqual(user_info_result.get("status"), "success")
            self.assertEqual((user_info_result.get("user_info") or {}).get("permissions"), [])
            self.assertEqual((user_info_result.get("user_info") or {}).get("profile"), {})

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
