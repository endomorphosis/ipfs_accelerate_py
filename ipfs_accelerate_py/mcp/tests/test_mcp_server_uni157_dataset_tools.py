#!/usr/bin/env python3
"""UNI-157 deterministic parity tests for native dataset tools."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.dataset_tools import native_dataset_tools


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI157DatasetTools(unittest.TestCase):
    def test_registration_schema_contracts_are_tightened(self) -> None:
        manager = _DummyManager()
        native_dataset_tools.register_native_dataset_tools(manager)
        schemas = {call["name"]: call["input_schema"] for call in manager.calls}

        load_props = schemas["load_dataset"]["properties"]
        self.assertEqual(load_props["source"].get("minLength"), 1)

        text_to_fol_props = schemas["text_to_fol"]["properties"]
        self.assertEqual(text_to_fol_props["confidence_threshold"].get("minimum"), 0)
        self.assertEqual(text_to_fol_props["confidence_threshold"].get("maximum"), 1)

        self.assertEqual(schemas["dataset_tools_claudes"].get("type"), "object")

    def test_load_and_save_validate_and_wrap_exceptions(self) -> None:
        async def _boom(**_: object) -> dict:
            raise RuntimeError("dataset boom")

        async def _run() -> None:
            invalid_source = await native_dataset_tools.load_dataset(source="   ")
            self.assertEqual(invalid_source.get("status"), "error")
            self.assertIn("source must be a non-empty string", str(invalid_source.get("error", "")))

            invalid_dataset = await native_dataset_tools.save_dataset(dataset_data="   ")
            self.assertEqual(invalid_dataset.get("status"), "error")
            self.assertIn("dataset_data must be non-empty", str(invalid_dataset.get("error", "")))

            with patch.dict(native_dataset_tools._API, {"load_dataset": _boom}, clear=False):
                result = await native_dataset_tools.load_dataset(source="dataset://id")
                self.assertEqual(result.get("status"), "error")
                self.assertIn("load_dataset failed", str(result.get("error", "")))

        anyio.run(_run)

    def test_text_to_fol_validates_threshold_and_predicates(self) -> None:
        async def _run() -> None:
            invalid_predicates = await native_dataset_tools.text_to_fol(
                text_input="All humans are mortal",
                domain_predicates=["Human", ""],
            )
            self.assertEqual(invalid_predicates.get("status"), "error")
            self.assertIn("domain_predicates must be an array of non-empty strings", str(invalid_predicates.get("error", "")))

            invalid_threshold = await native_dataset_tools.text_to_fol(
                text_input="All humans are mortal",
                confidence_threshold=1.5,
            )
            self.assertEqual(invalid_threshold.get("status"), "error")
            self.assertIn("confidence_threshold must be between 0 and 1", str(invalid_threshold.get("error", "")))

        anyio.run(_run)

    def test_legal_text_to_deontic_validates_flags_and_wraps_exceptions(self) -> None:
        async def _boom(**_: object) -> dict:
            raise RuntimeError("deontic boom")

        async def _run() -> None:
            invalid_flag = await native_dataset_tools.legal_text_to_deontic(
                text_input="Drivers must stop at red lights",
                extract_obligations="yes",  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_flag.get("status"), "error")
            self.assertIn("extract_obligations must be a boolean", str(invalid_flag.get("error", "")))

            with patch.dict(native_dataset_tools._API, {"legal_text_to_deontic": _boom}, clear=False):
                result = await native_dataset_tools.legal_text_to_deontic(
                    text_input="Drivers must stop at red lights",
                )
                self.assertEqual(result.get("status"), "error")
                self.assertIn("legal_text_to_deontic failed", str(result.get("error", "")))

        anyio.run(_run)

    def test_dataset_tools_claudes_success_defaults_and_exception_wrapping(self) -> None:
        async def _minimal() -> dict:
            return {"status": "success"}

        async def _boom() -> dict:
            raise RuntimeError("claudes boom")

        async def _run() -> None:
            with patch.dict(native_dataset_tools._API, {"dataset_tools_claudes": _minimal}, clear=False):
                result = await native_dataset_tools.dataset_tools_claudes()
                self.assertEqual(result.get("status"), "success")
                self.assertEqual(result.get("tool_type"), "Dataset processing tool")
                self.assertEqual(result.get("available_methods"), ["process_data"])

            with patch.dict(native_dataset_tools._API, {"dataset_tools_claudes": _boom}, clear=False):
                failed = await native_dataset_tools.dataset_tools_claudes()
                self.assertEqual(failed.get("status"), "error")
                self.assertIn("dataset_tools_claudes failed", str(failed.get("error", "")))

        anyio.run(_run)

    def test_dataset_crud_tools_support_json_string_entrypoints(self) -> None:
        async def _load(**kwargs: object) -> dict:
            return {"status": "success", "source": kwargs.get("source"), "dataset_id": "loaded-1"}

        async def _save(**kwargs: object) -> dict:
            return {
                "status": "success",
                "destination": kwargs.get("destination"),
                "dataset_id": "saved-1",
            }

        async def _process(**kwargs: object) -> dict:
            operations = kwargs.get("operations") or []
            return {
                "status": "success",
                "dataset_id": kwargs.get("output_id") or "processed-1",
                "num_operations": len(operations),
            }

        async def _convert(**kwargs: object) -> dict:
            return {
                "status": "success",
                "dataset_id": f"converted-{kwargs.get('dataset_id')}",
                "target_format": kwargs.get("target_format"),
            }

        async def _run() -> None:
            with patch.dict(
                native_dataset_tools._API,
                {
                    "load_dataset": _load,
                    "save_dataset": _save,
                    "process_dataset": _process,
                    "convert_dataset_format": _convert,
                },
                clear=False,
            ):
                load_result = await native_dataset_tools.load_dataset(
                    source=json.dumps({"source": "dataset://demo", "format": "json"})
                )
                load_payload = json.loads(load_result["content"][0]["text"])
                self.assertEqual(load_payload.get("status"), "success")
                self.assertEqual(load_payload.get("source"), "dataset://demo")

                save_result = await native_dataset_tools.save_dataset(
                    dataset_data=json.dumps(
                        {
                            "dataset_data": {"records": [1, 2]},
                            "destination": "/tmp/output.json",
                            "format": "json",
                        }
                    )
                )
                save_payload = json.loads(save_result["content"][0]["text"])
                self.assertEqual(save_payload.get("status"), "success")
                self.assertEqual(save_payload.get("destination"), "/tmp/output.json")

                process_result = await native_dataset_tools.process_dataset(
                    dataset_source=json.dumps(
                        {
                            "dataset_source": "dataset://demo",
                            "operations": [{"type": "filter"}],
                            "output_id": "processed-compat",
                        }
                    )
                )
                process_payload = json.loads(process_result["content"][0]["text"])
                self.assertEqual(process_payload.get("status"), "success")
                self.assertEqual(process_payload.get("dataset_id"), "processed-compat")
                self.assertEqual(process_payload.get("num_operations"), 1)

                convert_result = await native_dataset_tools.convert_dataset_format(
                    dataset_id=json.dumps({"dataset_id": "dataset-1", "target_format": "parquet"})
                )
                convert_payload = json.loads(convert_result["content"][0]["text"])
                self.assertEqual(convert_payload.get("status"), "success")
                self.assertEqual(convert_payload.get("dataset_id"), "converted-dataset-1")
                self.assertEqual(convert_payload.get("target_format"), "parquet")

        anyio.run(_run)

    def test_dataset_json_string_entrypoints_require_source_fields(self) -> None:
        async def _run() -> None:
            save_result = await native_dataset_tools.save_dataset(dataset_data=json.dumps({"dataset_data": {}}))
            save_payload = json.loads(save_result["content"][0]["text"])
            self.assertEqual(save_payload.get("status"), "error")
            self.assertIn("Missing required field: destination", str(save_payload.get("error", "")))

            process_result = await native_dataset_tools.process_dataset(dataset_source=json.dumps({"dataset_source": "x"}))
            process_payload = json.loads(process_result["content"][0]["text"])
            self.assertEqual(process_payload.get("status"), "error")
            self.assertIn("Missing required fields", str(process_payload.get("error", "")))

            convert_result = await native_dataset_tools.convert_dataset_format(dataset_id=json.dumps({"dataset_id": "x"}))
            convert_payload = json.loads(convert_result["content"][0]["text"])
            self.assertEqual(convert_payload.get("status"), "error")
            self.assertIn("Missing required field: target_format", str(convert_payload.get("error", "")))

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
