#!/usr/bin/env python3
"""UNI-126 IPFS cluster tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.ipfs_cluster_tools.native_ipfs_cluster_tools import (
    manage_ipfs_cluster,
    manage_ipfs_content,
    register_native_ipfs_cluster_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI126IPFSClusterTools(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_ipfs_cluster_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}

        cluster_schema = by_name["manage_ipfs_cluster"]["input_schema"]
        self.assertIn("pin_content", cluster_schema["properties"]["action"].get("enum", []))

        content_schema = by_name["manage_ipfs_content"]["input_schema"]
        self.assertIn("upload", content_schema["properties"]["action"].get("enum", []))

    def test_manage_cluster_rejects_invalid_action(self) -> None:
        async def _run() -> None:
            result = await manage_ipfs_cluster(action="heal")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("action must be one of", str(result.get("message", "")))

        anyio.run(_run)

    def test_manage_cluster_requires_cid_for_pin(self) -> None:
        async def _run() -> None:
            result = await manage_ipfs_cluster(action="pin_content")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("cid is required for pin_content action", str(result.get("message", "")))

        anyio.run(_run)

    def test_manage_content_requires_content_for_upload(self) -> None:
        async def _run() -> None:
            result = await manage_ipfs_content(action="upload", content="")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("content is required for upload action", str(result.get("message", "")))

        anyio.run(_run)

    def test_manage_content_requires_cid_for_download(self) -> None:
        async def _run() -> None:
            result = await manage_ipfs_content(action="download")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("cid is required for download action", str(result.get("message", "")))

        anyio.run(_run)

    def test_manage_content_success_envelope_shape(self) -> None:
        async def _run() -> None:
            result = await manage_ipfs_content(action="list_content")
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("action"), "list_content")

        anyio.run(_run)

    def test_manage_cluster_defaults_with_minimal_success_payload(self) -> None:
        async def _minimal_cluster(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.ipfs_cluster_tools.native_ipfs_cluster_tools._API",
                {
                    "manage_ipfs_cluster": _minimal_cluster,
                    "manage_ipfs_content": None,
                },
            ):
                result = await manage_ipfs_cluster(
                    action="status",
                    replication_factor=5,
                    cluster_config={"mode": "test"},
                    filters={"scope": "all"},
                )

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("action"), "status")
            self.assertEqual(result.get("replication_factor"), 5)
            self.assertEqual(result.get("cluster_config"), {"mode": "test"})
            self.assertEqual(result.get("filters"), {"scope": "all"})
            self.assertEqual(result.get("cluster_operation"), True)
            self.assertEqual(result.get("result"), {})

        anyio.run(_run)

    def test_manage_content_defaults_with_minimal_success_payload(self) -> None:
        async def _minimal_content(**_: object) -> dict:
            return {"status": "success"}

        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.ipfs_cluster_tools.native_ipfs_cluster_tools._API",
                {
                    "manage_ipfs_cluster": None,
                    "manage_ipfs_content": _minimal_content,
                },
            ):
                result = await manage_ipfs_content(
                    action="upload",
                    content="hello",
                    metadata={"source": "unit"},
                    pin=False,
                    content_type="text/plain",
                )

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("action"), "upload")
            self.assertEqual(result.get("pin"), False)
            self.assertEqual(result.get("content_type"), "text/plain")
            self.assertEqual(result.get("metadata"), {"source": "unit"})
            self.assertEqual(result.get("content"), "hello")
            self.assertEqual(result.get("result"), {})

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
