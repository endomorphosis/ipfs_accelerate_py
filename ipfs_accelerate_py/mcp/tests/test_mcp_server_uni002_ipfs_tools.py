#!/usr/bin/env python3
"""UNI-002 ipfs_tools parity tests."""

from __future__ import annotations

import unittest
import json

import anyio

import ipfs_accelerate_py.mcp_server.tools.ipfs_tools.native_ipfs_tools_category as native_ipfs_tools

from ipfs_accelerate_py.mcp_server.tools.ipfs_tools.native_ipfs_tools_category import (
    get_from_ipfs,
    pin_to_ipfs,
    register_native_ipfs_tools_category,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI002IPFSTools(unittest.TestCase):
    def test_register_includes_ipfs_tools(self) -> None:
        manager = _DummyManager()
        register_native_ipfs_tools_category(manager)
        names = [c["name"] for c in manager.calls]
        self.assertIn("pin_to_ipfs", names)
        self.assertIn("get_from_ipfs", names)

    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_ipfs_tools_category(manager)
        by_name = {c["name"]: c for c in manager.calls}

        pin_schema = by_name["pin_to_ipfs"]["input_schema"]
        self.assertEqual(pin_schema.get("required"), ["content_source"])
        self.assertEqual(pin_schema["properties"]["hash_algo"].get("default"), "sha2-256")

        get_schema = by_name["get_from_ipfs"]["input_schema"]
        self.assertEqual(get_schema.get("required"), ["cid"])
        self.assertEqual(get_schema["properties"]["timeout_seconds"].get("minimum"), 1)

    def test_pin_to_ipfs_validation_and_shape(self) -> None:
        async def _run() -> None:
            missing_source = await pin_to_ipfs(content_source="")
            self.assertEqual(missing_source.get("status"), "error")
            self.assertIn("'content_source'", str(missing_source.get("message", "")))

            invalid_source_type = await pin_to_ipfs(content_source=123)  # type: ignore[arg-type]
            self.assertEqual(invalid_source_type.get("status"), "error")
            self.assertIn("'content_source'", str(invalid_source_type.get("message", "")))

            string_source = await pin_to_ipfs(content_source="/tmp/demo.txt")
            self.assertIn(string_source.get("status"), ["success", "error"])

            object_source = await pin_to_ipfs(content_source={"data": "hello"})
            self.assertIn(object_source.get("status"), ["success", "error"])

        anyio.run(_run)

    def test_get_from_ipfs_validation_and_shape(self) -> None:
        async def _run() -> None:
            missing_cid = await get_from_ipfs(cid="")
            self.assertEqual(missing_cid.get("status"), "error")
            self.assertIn("'cid' is required", str(missing_cid.get("message", "")))

            invalid_timeout = await get_from_ipfs(cid="QmDemoHash", timeout_seconds=0)
            self.assertEqual(invalid_timeout.get("status"), "error")
            self.assertIn("'timeout_seconds'", str(invalid_timeout.get("message", "")))

            invalid_timeout_type = await get_from_ipfs(cid="QmDemoHash", timeout_seconds="bad")  # type: ignore[arg-type]
            self.assertEqual(invalid_timeout_type.get("status"), "error")
            self.assertIn("must be an integer", str(invalid_timeout_type.get("message", "")))

            invalid_output = await get_from_ipfs(cid="QmDemoHash", output_path="   ")
            self.assertEqual(invalid_output.get("status"), "error")
            self.assertIn("'output_path'", str(invalid_output.get("message", "")))

            invalid_gateway = await get_from_ipfs(cid="QmDemoHash", gateway="  ")
            self.assertEqual(invalid_gateway.get("status"), "error")
            self.assertIn("'gateway'", str(invalid_gateway.get("message", "")))

            invalid_gateway_scheme = await get_from_ipfs(
                cid="QmDemoHash",
                gateway="ipfs://localhost:8080",
            )
            self.assertEqual(invalid_gateway_scheme.get("status"), "error")
            self.assertIn("must start with", str(invalid_gateway_scheme.get("message", "")))

            result = await get_from_ipfs(cid="QmDemoHash", timeout_seconds=5)
            self.assertIn(result.get("status"), ["success", "error"])

        anyio.run(_run)

    def test_get_from_ipfs_gateway_normalization_passthrough(self) -> None:
        async def _run() -> None:
            original_get = native_ipfs_tools._API["get_from_ipfs"]
            captured: dict = {}

            async def _fake_get_from_ipfs(**kwargs):
                captured.update(kwargs)
                return {"status": "success", "cid": kwargs.get("cid")}

            native_ipfs_tools._API["get_from_ipfs"] = _fake_get_from_ipfs
            try:
                result = await get_from_ipfs(
                    cid="QmDemoHash",
                    gateway="https://ipfs.io/",
                    timeout_seconds=15,
                )
            finally:
                native_ipfs_tools._API["get_from_ipfs"] = original_get

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(captured.get("cid"), "QmDemoHash")
            self.assertEqual(captured.get("timeout_seconds"), 15)
            self.assertEqual(captured.get("gateway"), "https://ipfs.io")

        anyio.run(_run)

    def test_pin_to_ipfs_accepts_json_string_entrypoint(self) -> None:
        async def _run() -> None:
            payload = json.dumps(
                {
                    "content_source": "/tmp/demo.txt",
                    "recursive": True,
                    "wrap_with_directory": False,
                    "hash_algo": "sha2-256",
                }
            )
            result = await pin_to_ipfs(content_source=payload)
            self.assertIn("content", result)
            self.assertIsInstance(result["content"], list)
            self.assertEqual(result["content"][0].get("type"), "text")
            parsed = json.loads(result["content"][0].get("text", "{}"))
            self.assertIn(parsed.get("status"), ["success", "error"])

        anyio.run(_run)

    def test_get_from_ipfs_json_string_missing_cid_validation(self) -> None:
        async def _run() -> None:
            payload = json.dumps({"output_path": "/tmp/out.txt"})
            result = await get_from_ipfs(cid=payload)
            self.assertIn("content", result)
            parsed = json.loads(result["content"][0].get("text", "{}"))
            self.assertEqual(parsed.get("status"), "error")
            self.assertIn("Missing required field: cid", str(parsed.get("error", "")))

        anyio.run(_run)

    def test_get_from_ipfs_json_string_invalid_timeout_validation(self) -> None:
        async def _run() -> None:
            payload = json.dumps({"cid": "QmDemoHash", "timeout_seconds": "bad"})
            result = await get_from_ipfs(cid=payload)
            self.assertIn("content", result)
            parsed = json.loads(result["content"][0].get("text", "{}"))
            self.assertEqual(parsed.get("status"), "error")
            self.assertIn("must be an integer", str(parsed.get("message", "")))

        anyio.run(_run)

    def test_pin_to_ipfs_json_string_invalid_json_validation(self) -> None:
        async def _run() -> None:
            result = await pin_to_ipfs(content_source="{not-json")
            self.assertIn("content", result)
            parsed = json.loads(result["content"][0].get("text", "{}"))
            self.assertEqual(parsed.get("status"), "error")
            self.assertEqual(parsed.get("error_type"), "validation")
            self.assertIn("Invalid JSON", str(parsed.get("error", "")))

        anyio.run(_run)

    def test_get_from_ipfs_json_string_non_object_validation(self) -> None:
        async def _run() -> None:
            result = await get_from_ipfs(cid='["not-an-object"]')
            self.assertIn("content", result)
            parsed = json.loads(result["content"][0].get("text", "{}"))
            self.assertEqual(parsed.get("status"), "error")
            self.assertEqual(parsed.get("error_type"), "validation")
            self.assertIn("must be an object", str(parsed.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
