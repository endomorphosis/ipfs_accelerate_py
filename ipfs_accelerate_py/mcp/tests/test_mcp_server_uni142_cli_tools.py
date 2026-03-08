#!/usr/bin/env python3
"""UNI-142 cli parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.cli import native_cli_tools
from ipfs_accelerate_py.mcp_server.tools.cli.native_cli_tools import (
    discover_biomolecules_rag_cli,
    discover_enzyme_inhibitors_cli,
    discover_protein_binders_cli,
    execute_command,
    register_native_cli_tools,
    scrape_clinical_trials_cli,
    scrape_pubmed_cli,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI142CliTools(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_cli_tools(manager)
        by_name = {call["name"]: call for call in manager.calls}

        schema = by_name["execute_command"]["input_schema"]
        props = schema["properties"]

        self.assertEqual(props["command"]["minLength"], 1)
        self.assertEqual(props["timeout_seconds"]["minimum"], 1)
        self.assertIn("scrape_pubmed_cli", by_name)
        self.assertIn("discover_biomolecules_rag_cli", by_name)
        self.assertEqual(
            by_name["discover_biomolecules_rag_cli"]["input_schema"]["required"],
            ["target", "type"],
        )

    def test_execute_command_validation(self) -> None:
        async def _run() -> None:
            result = await execute_command(command="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("command", str(result.get("error", "")))

            result = await execute_command(command="echo", args=[""])
            self.assertEqual(result.get("status"), "error")
            self.assertIn("args", str(result.get("error", "")))

            result = await execute_command(command="echo", timeout_seconds=0)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("timeout_seconds", str(result.get("error", "")))

        anyio.run(_run)

    def test_medical_cli_validation(self) -> None:
        async def _run() -> None:
            result = await scrape_pubmed_cli(query="topic", research_type="case_report")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("research_type", str(result.get("error", "")))

            result = await scrape_clinical_trials_cli()
            self.assertEqual(result.get("status"), "error")
            self.assertIn("query or condition is required", str(result.get("error", "")))

            result = await discover_protein_binders_cli(target="PD-L1", min_confidence=1.5)
            self.assertEqual(result.get("status"), "error")
            self.assertIn("min_confidence", str(result.get("error", "")))

            result = await discover_enzyme_inhibitors_cli(target="ACE2", format="yaml")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("format", str(result.get("error", "")))

            result = await discover_biomolecules_rag_cli(target="mTOR", type="proteins")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("type", str(result.get("error", "")))

        anyio.run(_run)

    def test_success_envelope_shapes(self) -> None:
        async def _run() -> None:
            result = await execute_command(
                command="echo",
                args=["hello"],
                timeout_seconds=3,
            )
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("command"), "echo")
            self.assertEqual(result.get("args"), ["hello"])
            self.assertEqual(result.get("timeout_seconds"), 3)

            pubmed = await scrape_pubmed_cli(query="COVID-19", format="table")
            self.assertIn(pubmed.get("status"), ["success", "error"])
            self.assertEqual(pubmed.get("query"), "COVID-19")
            self.assertEqual(pubmed.get("format"), "table")

            trials = await scrape_clinical_trials_cli(condition="diabetes")
            self.assertIn(trials.get("status"), ["success", "error"])
            self.assertEqual(trials.get("query"), "diabetes")

            binders = await discover_protein_binders_cli(target="PD-L1", interaction="binding")
            self.assertIn(binders.get("status"), ["success", "error"])
            self.assertEqual(binders.get("target"), "PD-L1")

            inhibitors = await discover_enzyme_inhibitors_cli(target="ACE2", enzyme_class="protease")
            self.assertIn(inhibitors.get("status"), ["success", "error"])
            self.assertEqual(inhibitors.get("target"), "ACE2")

            rag = await discover_biomolecules_rag_cli(target="mTOR signaling", type="pathway")
            self.assertIn(rag.get("status"), ["success", "error"])
            self.assertEqual(rag.get("type"), "pathway")

        anyio.run(_run)

    def test_execute_command_minimal_success_defaults(self) -> None:
        async def _run() -> None:
            with patch("ipfs_accelerate_py.mcp_server.tools.cli.native_cli_tools._API") as mock_api:
                async def _impl(**_: object) -> dict:
                    return {"status": "success"}

                mock_api.__getitem__.return_value = _impl
                result = await execute_command(command="echo", args=["hello"], timeout_seconds=3)

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("message"), "Command execution request processed")
            self.assertEqual(result.get("command"), "echo")
            self.assertEqual(result.get("args"), ["hello"])
            self.assertEqual(result.get("timeout_seconds"), 3)

        anyio.run(_run)

    def test_scrape_pubmed_error_only_payload_infers_error(self) -> None:
        async def _run() -> None:
            with patch("ipfs_accelerate_py.mcp_server.tools.cli.native_cli_tools._API") as mock_api:
                async def _impl(**_: object) -> dict:
                    return {"error": "pubmed unavailable"}

                mock_api.__getitem__.return_value = _impl
                result = await scrape_pubmed_cli(query="COVID-19")

            self.assertEqual(result.get("status"), "error")
            self.assertIn("pubmed unavailable", str(result.get("error", "")))

        anyio.run(_run)

    def test_cli_wrappers_infer_error_status_from_contradictory_delegate_payloads(self) -> None:
        async def _contradictory_failure(**_: object) -> dict:
            return {"status": "success", "success": False, "error": "delegate failed"}

        async def _run() -> None:
            with patch.dict(
                native_cli_tools._API,
                {
                    "execute_command": _contradictory_failure,
                    "scrape_pubmed_cli": _contradictory_failure,
                    "scrape_clinical_trials_cli": _contradictory_failure,
                    "discover_protein_binders_cli": _contradictory_failure,
                    "discover_biomolecules_rag_cli": _contradictory_failure,
                },
                clear=False,
            ):
                executed = await execute_command(command="echo", args=["hello"], timeout_seconds=3)
                pubmed = await scrape_pubmed_cli(query="COVID-19")
                trials = await scrape_clinical_trials_cli(condition="diabetes")
                binders = await discover_protein_binders_cli(target="PD-L1")
                rag = await discover_biomolecules_rag_cli(target="mTOR signaling", type="pathway")

            self.assertEqual(executed.get("status"), "error")
            self.assertEqual(executed.get("error"), "delegate failed")

            self.assertEqual(pubmed.get("status"), "error")
            self.assertEqual(pubmed.get("error"), "delegate failed")

            self.assertEqual(trials.get("status"), "error")
            self.assertEqual(trials.get("error"), "delegate failed")

            self.assertEqual(binders.get("status"), "error")
            self.assertEqual(binders.get("error"), "delegate failed")

            self.assertEqual(rag.get("status"), "error")
            self.assertEqual(rag.get("error"), "delegate failed")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
