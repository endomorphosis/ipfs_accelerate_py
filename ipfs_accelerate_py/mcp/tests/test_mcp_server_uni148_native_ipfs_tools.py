#!/usr/bin/env python3
"""UNI-148 native ipfs category parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from ipfs_accelerate_py.mcp_server.tools.ipfs import native_ipfs_tools as ipfs_mod


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class _Result:
    def __init__(self, success=True, data=None, error=None) -> None:
        self.success = success
        self.data = data if data is not None else {}
        self.error = error


class _FakeKit:
    def validate_cid(self, cid: str):
        return _Result(success=True, data={"cid": cid, "valid": True})

    def list_files(self, path: str):
        return _Result(success=True, data={"path": path, "entries": []})

    def add_file(self, path: str, pin: bool):
        return _Result(success=True, data={"path": path, "pin": pin})

    def pin_file(self, cid: str):
        return _Result(success=True, data={"cid": cid, "pinned": True})

    def unpin_file(self, cid: str):
        return _Result(success=True, data={"cid": cid, "pinned": False})

    def get_file(self, cid: str, output_path: str):
        return _Result(success=True, data={"cid": cid, "output_path": output_path})

    def cat_file(self, cid: str):
        return _Result(success=True, data={"cid": cid, "content": "hello"})


class TestMCPServerUNI148NativeIPFSTools(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        ipfs_mod.register_native_ipfs_tools(manager)
        by_name = {call["name"]: call for call in manager.calls}

        validate_schema = by_name["ipfs_files_validate_cid"]["input_schema"]
        self.assertEqual(validate_schema["properties"]["cid"]["minLength"], 1)

        add_schema = by_name["ipfs_files_add_file"]["input_schema"]
        self.assertEqual(add_schema["properties"]["path"]["minLength"], 1)
        self.assertEqual(add_schema["properties"]["pin"]["default"], True)

        get_schema = by_name["ipfs_files_get_file"]["input_schema"]
        self.assertEqual(get_schema["properties"]["output_path"]["minLength"], 1)

    def test_validation_error_envelopes(self) -> None:
        self.assertFalse(ipfs_mod.ipfs_files_validate_cid("   ")["success"])
        self.assertIn("non-empty string", str(ipfs_mod.ipfs_files_validate_cid("   ").get("error", "")))

        self.assertFalse(ipfs_mod.ipfs_files_list_files("   ")["success"])
        self.assertFalse(ipfs_mod.ipfs_files_add_file("", pin=True)["success"])
        self.assertFalse(ipfs_mod.ipfs_files_add_file("/tmp/a", pin="yes")["success"])  # type: ignore[arg-type]
        self.assertFalse(ipfs_mod.ipfs_files_pin_file(" ")["success"])
        self.assertFalse(ipfs_mod.ipfs_files_unpin_file(" ")["success"])
        self.assertFalse(ipfs_mod.ipfs_files_get_file("cid", " ")["success"])
        self.assertFalse(ipfs_mod.ipfs_files_cat(" ")["success"])

    def test_success_shapes_with_fake_kit(self) -> None:
        with patch.object(ipfs_mod, "get_ipfs_files_kit", return_value=_FakeKit()):
            validate = ipfs_mod.ipfs_files_validate_cid("bafy-demo")
            self.assertTrue(validate.get("success"))
            self.assertEqual(validate.get("data", {}).get("cid"), "bafy-demo")

            listed = ipfs_mod.ipfs_files_list_files("/")
            self.assertTrue(listed.get("success"))

            added = ipfs_mod.ipfs_files_add_file("/tmp/file.txt", pin=False)
            self.assertTrue(added.get("success"))
            self.assertEqual(added.get("data", {}).get("pin"), False)

            fetched = ipfs_mod.ipfs_files_get_file("bafy-demo", "/tmp/out.txt")
            self.assertTrue(fetched.get("success"))

    def test_error_shape_when_kit_raises(self) -> None:
        class _ExplodingKit:
            def validate_cid(self, cid: str):
                raise RuntimeError(f"boom:{cid}")

        with patch.object(ipfs_mod, "get_ipfs_files_kit", return_value=_ExplodingKit()):
            result = ipfs_mod.ipfs_files_validate_cid("bafy-demo")

        self.assertFalse(result.get("success"))
        self.assertIn("boom:bafy-demo", str(result.get("error", "")))

    def test_minimal_success_defaults_with_sparse_kit_result(self) -> None:
        class _SparseResult:
            success = True
            data = None
            error = None

        class _SparseKit:
            def validate_cid(self, cid: str):
                return _SparseResult()

        with patch.object(ipfs_mod, "get_ipfs_files_kit", return_value=_SparseKit()):
            result = ipfs_mod.ipfs_files_validate_cid("bafy-demo")

        self.assertEqual(result.get("status"), "success")
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("data"), {})
        self.assertIsNone(result.get("error"))

    def test_error_status_inferred_from_failed_kit_result(self) -> None:
        class _FailedResult:
            success = False
            data = None
            error = "invalid cid"

        class _FailedKit:
            def validate_cid(self, cid: str):
                return _FailedResult()

        with patch.object(ipfs_mod, "get_ipfs_files_kit", return_value=_FailedKit()):
            result = ipfs_mod.ipfs_files_validate_cid("bafy-demo")

        self.assertEqual(result.get("status"), "error")
        self.assertFalse(result.get("success"))
        self.assertEqual(result.get("error"), "invalid cid")


if __name__ == "__main__":
    unittest.main()
