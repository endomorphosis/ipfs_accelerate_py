#!/usr/bin/env python3
"""UNI-153 p2p_workflow_tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.p2p_workflow_tools import native_p2p_workflow_tools as module


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI153P2PWorkflowTools(unittest.TestCase):
    def test_register_schema_contracts_are_hardened(self) -> None:
        manager = _DummyManager()
        module.register_native_p2p_workflow_tools(manager)

        by_name = {call["name"]: call for call in manager.calls}
        schedule_props = by_name["schedule_p2p_workflow"]["input_schema"]["properties"]
        self.assertEqual(schedule_props["workflow_id"].get("minLength"), 1)
        self.assertEqual(schedule_props["name"].get("minLength"), 1)
        self.assertGreater(schedule_props["priority"].get("minimum", 0), 0)

    def test_initialize_scheduler_exception_yields_error_envelope(self) -> None:
        async def _run() -> None:
            with patch.dict(
                module._API,  # type: ignore[attr-defined]
                {"initialize_p2p_scheduler": lambda **_: (_ for _ in ()).throw(RuntimeError("boom"))},
            ):
                result = await module.initialize_p2p_scheduler(peer_id="peer-1")
                self.assertEqual(result.get("status"), "error")
                self.assertIn("initialize_p2p_scheduler failed", str(result.get("message", "")))

        anyio.run(_run)

    def test_schedule_workflow_exception_yields_error_envelope(self) -> None:
        async def _run() -> None:
            with patch.dict(
                module._API,  # type: ignore[attr-defined]
                {"schedule_p2p_workflow": lambda **_: (_ for _ in ()).throw(RuntimeError("boom"))},
            ):
                result = await module.schedule_p2p_workflow(
                    workflow_id="wf-1",
                    name="demo",
                    tags=["p2p_eligible"],
                )
                self.assertEqual(result.get("status"), "error")
                self.assertIn("schedule_p2p_workflow failed", str(result.get("message", "")))

        anyio.run(_run)

    def test_status_wrapper_exception_yields_error_envelope(self) -> None:
        async def _run() -> None:
            with patch.dict(
                module._API,  # type: ignore[attr-defined]
                {"get_p2p_scheduler_status": lambda: (_ for _ in ()).throw(RuntimeError("boom"))},
            ):
                result = await module.get_p2p_scheduler_status()
                self.assertEqual(result.get("status"), "error")
                self.assertIn("get_p2p_scheduler_status failed", str(result.get("message", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
