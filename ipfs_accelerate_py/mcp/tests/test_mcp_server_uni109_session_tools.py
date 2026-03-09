#!/usr/bin/env python3
"""UNI-109 session tools parity tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

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

    def test_session_wrappers_apply_sparse_success_defaults(self) -> None:
        class _SparseManager:
            async def create_session(self, **kwargs):
                return {"session_id": "11111111-1111-1111-1111-111111111111", **kwargs}

            async def get_session(self, session_id: str):
                return {"session_id": session_id}

            async def update_session(self, session_id: str, **kwargs):
                return {"session_id": session_id, **kwargs}

        async def _run() -> None:
            created = await create_session(
                session_name="uni109-sparse",
                user_id="uni109-user",
                session_type="batch",
                metadata={"purpose": "sparse"},
                tags=["uni109"],
                session_manager=_SparseManager(),
            )
            self.assertEqual(created.get("status"), "success")
            self.assertIn("session", created)
            self.assertEqual(created["session"].get("session_name"), "uni109-sparse")
            self.assertEqual(created["session"].get("session_type"), "batch")
            self.assertEqual(created["session"].get("metadata", {}).get("purpose"), "sparse")
            self.assertEqual(created["session"].get("tags"), ["uni109"])

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.session_tools.native_session_tools._get_session_manager",
                return_value=_SparseManager(),
            ):
                managed = await manage_session(
                    action="get",
                    session_id="11111111-1111-1111-1111-111111111111",
                )
                self.assertEqual(managed.get("status"), "success")
                self.assertEqual(managed["session"].get("session_id"), "11111111-1111-1111-1111-111111111111")
                self.assertEqual(managed["session"].get("status"), "unknown")
                self.assertIn("created_at", managed["session"])
                self.assertIn("last_activity", managed["session"])

                state = await get_session_state(session_id="11111111-1111-1111-1111-111111111111")
                self.assertEqual(state.get("status"), "success")
                session_state = state.get("session_state", {})
                self.assertEqual(session_state.get("session_id"), "11111111-1111-1111-1111-111111111111")
                self.assertIn("metrics", session_state)
                self.assertIn("resource_usage", session_state)
                self.assertIn("health_info", session_state)

        anyio.run(_run)

    def test_session_wrappers_infer_error_status_from_contradictory_delegate_payloads(self) -> None:
        class _ContradictoryManager:
            async def create_session(self, **kwargs):
                return {"status": "success", "success": False, "error": "create failed"}

            async def get_session(self, session_id: str):
                return {"status": "success", "success": False, "error": "lookup failed"}

            async def update_session(self, session_id: str, **kwargs):
                return {"status": "success", "success": False, "error": "update failed"}

            async def delete_session(self, session_id: str):
                return {"status": "success", "success": False, "error": "delete failed"}

            async def list_sessions(self, **filters):
                return {"status": "success", "success": False, "error": "list failed"}

            async def cleanup_expired_sessions(self, max_age_hours: int = 24):
                return {"status": "success", "success": False, "error": "cleanup failed"}

        async def _run() -> None:
            manager = _ContradictoryManager()

            created = await create_session(session_name="uni109-contradictory", session_manager=manager)
            self.assertEqual(created.get("status"), "error")
            self.assertFalse(created.get("success"))
            self.assertEqual(created.get("error"), "create failed")

            with patch(
                "ipfs_accelerate_py.mcp_server.tools.session_tools.native_session_tools._get_session_manager",
                return_value=manager,
            ):
                managed_get = await manage_session(
                    action="get",
                    session_id="11111111-1111-1111-1111-111111111111",
                )
                self.assertEqual(managed_get.get("status"), "error")
                self.assertFalse(managed_get.get("success"))
                self.assertEqual(managed_get.get("error"), "lookup failed")

                managed_update = await manage_session(
                    action="update",
                    session_id="11111111-1111-1111-1111-111111111111",
                    updates={"status": "paused"},
                )
                self.assertEqual(managed_update.get("status"), "error")
                self.assertFalse(managed_update.get("success"))
                self.assertEqual(managed_update.get("error"), "update failed")

                managed_list = await manage_session(action="list")
                self.assertEqual(managed_list.get("status"), "error")
                self.assertFalse(managed_list.get("success"))
                self.assertEqual(managed_list.get("error"), "list failed")

                managed_cleanup = await manage_session(
                    action="cleanup",
                    cleanup_options={"max_age_hours": 12},
                )
                self.assertEqual(managed_cleanup.get("status"), "error")
                self.assertFalse(managed_cleanup.get("success"))
                self.assertEqual(managed_cleanup.get("error"), "cleanup failed")

                state = await get_session_state(session_id="11111111-1111-1111-1111-111111111111")
                self.assertEqual(state.get("status"), "error")
                self.assertFalse(state.get("success"))
                self.assertEqual(state.get("error"), "lookup failed")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
