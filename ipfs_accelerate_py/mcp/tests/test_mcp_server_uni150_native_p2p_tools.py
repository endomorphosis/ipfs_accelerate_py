#!/usr/bin/env python3
"""UNI-150 native p2p category parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.p2p import native_p2p_tools as p2p_mod


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI150NativeP2PTools(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        p2p_mod.register_native_p2p_tools(manager)
        by_name = {call["name"]: call for call in manager.calls}

        status_schema = by_name["p2p_taskqueue_status"]["input_schema"]
        self.assertGreater(status_schema["properties"]["timeout_s"]["minimum"], 0)

        submit_schema = by_name["p2p_taskqueue_submit"]["input_schema"]
        self.assertEqual(submit_schema["properties"]["task_type"]["minLength"], 1)
        self.assertEqual(submit_schema["properties"]["model_name"]["minLength"], 1)

        wait_schema = by_name["p2p_taskqueue_wait_task"]["input_schema"]
        self.assertEqual(wait_schema["properties"]["task_id"]["minLength"], 1)

    def test_validation_error_envelopes(self) -> None:
        async def _run() -> None:
            invalid_status = await p2p_mod.p2p_taskqueue_status(timeout_s=0)
            self.assertFalse(invalid_status.get("ok"))
            self.assertIn("timeout_s must be a number > 0", str(invalid_status.get("error", "")))

            invalid_submit = await p2p_mod.p2p_taskqueue_submit("", "model", {})
            self.assertFalse(invalid_submit.get("ok"))
            self.assertIn("task_type must be a non-empty string", str(invalid_submit.get("error", "")))

            invalid_get = await p2p_mod.p2p_taskqueue_get_task("   ")
            self.assertFalse(invalid_get.get("ok"))
            self.assertIn("task_id must be a non-empty string", str(invalid_get.get("error", "")))

            invalid_wait = await p2p_mod.p2p_taskqueue_wait_task("task-1", timeout_s=0)
            self.assertFalse(invalid_wait.get("ok"))
            self.assertIn("timeout_s must be a number > 0", str(invalid_wait.get("error", "")))

        anyio.run(_run)

    def test_exception_path_envelope_shape(self) -> None:
        async def _run() -> None:
            result = await p2p_mod.p2p_taskqueue_submit(
                "demo",
                "model",
                {},
                remote_multiaddr="",
                peer_id="",
            )
            self.assertIn(result.get("ok"), [True, False])
            if result.get("ok") is False:
                self.assertIn("error", result)

        anyio.run(_run)

    def test_status_minimal_success_defaults(self) -> None:
        async def _run() -> None:
            with patch.object(p2p_mod, "_request_status") as mock_request:
                async def _impl(**_kwargs):
                    return {"ok": True}

                mock_request.side_effect = _impl
                result = await p2p_mod.p2p_taskqueue_status(detail=True)

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("ok"), True)
            self.assertEqual(result.get("detail"), True)

        anyio.run(_run)

    def test_submit_error_only_payload_infers_error(self) -> None:
        async def _run() -> None:
            with patch.object(p2p_mod, "_submit_task_with_info") as mock_submit:
                async def _impl(**_kwargs):
                    return {"error": "remote queue unavailable"}

                mock_submit.side_effect = _impl
                result = await p2p_mod.p2p_taskqueue_submit("demo", "model", {})

            self.assertEqual(result.get("status"), "error")
            self.assertEqual(result.get("success"), False)
            self.assertEqual(result.get("ok"), False)
            self.assertIn("remote queue unavailable", str(result.get("error", "")))

        anyio.run(_run)

    def test_status_infers_error_from_contradictory_delegate_payload(self) -> None:
        async def _run() -> None:
            with patch.object(p2p_mod, "_request_status") as mock_request:
                async def _impl(**_kwargs):
                    return {"status": "success", "success": False, "error": "delegate failure"}

                mock_request.side_effect = _impl
                result = await p2p_mod.p2p_taskqueue_status(detail=True)

            self.assertEqual(result.get("status"), "error")
            self.assertEqual(result.get("success"), False)
            self.assertEqual(result.get("ok"), False)
            self.assertEqual(result.get("error"), "delegate failure")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
