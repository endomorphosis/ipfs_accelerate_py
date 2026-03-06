#!/usr/bin/env python3
"""UNI-109 session tools parity tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.session_tools.native_session_tools import (
    create_session,
    get_session_state,
    manage_session,
    register_native_session_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI109SessionTools(unittest.TestCase):
    def test_register_includes_enhanced_session_wrappers(self) -> None:
        manager = _DummyManager()
        register_native_session_tools(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("manage_session", names)
        self.assertIn("get_session_state", names)

    def test_manage_session_get_requires_session_id(self) -> None:
        async def _run() -> None:
            result = await manage_session(action="get", session_id=None)
            self.assertEqual(result.get("status"), "error")
            self.assertEqual(result.get("code"), "MISSING_SESSION_ID")

        anyio.run(_run)

    def test_manage_session_unknown_action_contract(self) -> None:
        async def _run() -> None:
            result = await manage_session(action="invalid")
            self.assertEqual(result.get("status"), "error")
            self.assertEqual(result.get("code"), "UNKNOWN_ACTION")
            self.assertIn("valid_actions", result)

        anyio.run(_run)

    def test_get_session_state_rejects_invalid_uuid(self) -> None:
        async def _run() -> None:
            result = await get_session_state(session_id="not-a-uuid")
            self.assertEqual(result.get("status"), "error")
            self.assertEqual(result.get("code"), "INVALID_SESSION_ID")

        anyio.run(_run)

    def test_get_session_state_success_shape(self) -> None:
        async def _run() -> None:
            created = await create_session(session_name="uni109-state")
            self.assertEqual(created.get("status"), "success")
            session_id = str(created.get("session_id"))

            result = await get_session_state(session_id=session_id)
            self.assertEqual(result.get("status"), "success")
            self.assertIn("session_state", result)
            state = result["session_state"]
            self.assertIn("basic_info", state)
            self.assertIn("session_id", state)

        anyio.run(_run)

    def test_create_session_accepts_enhanced_creation_fields(self) -> None:
        async def _run() -> None:
            created = await create_session(
                session_name="uni109-enhanced",
                user_id="uni109-user",
                session_type="batch",
                resource_limits={"memory_limit_mb": 512},
                metadata={"purpose": "test"},
                tags=["uni", "session"],
            )
            self.assertEqual(created.get("status"), "success")
            self.assertEqual(created.get("session_type"), "batch")
            self.assertEqual(created.get("metadata", {}).get("purpose"), "test")
            self.assertIn("session", created.get("tags", []))

        anyio.run(_run)

    def test_create_session_rejects_invalid_tags_shape(self) -> None:
        async def _run() -> None:
            created = await create_session(
                session_name="uni109-invalid-tags",
                tags=["ok", "   "],
            )
            self.assertEqual(created.get("status"), "error")
            self.assertIn("tags must be a list of non-empty strings", str(created.get("message", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
