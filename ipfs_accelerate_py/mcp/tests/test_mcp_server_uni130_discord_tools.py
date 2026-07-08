#!/usr/bin/env python3
"""UNI-130 discord tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.discord_tools import native_discord_tools
from ipfs_accelerate_py.mcp_server.tools.discord_tools.native_discord_tools import (
    discord_analyze_export,
    discord_batch_convert_exports,
    discord_convert_export,
    discord_export_channel,
    discord_list_channels,
    discord_list_dm_channels,
    discord_list_guilds,
    register_native_discord_tools,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI130DiscordTools(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_discord_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}

        channels_schema = by_name["discord_list_channels"]["input_schema"]
        self.assertEqual(channels_schema["properties"]["guild_id"].get("minLength"), 1)

        self.assertIn("discord_list_dm_channels", by_name)
        self.assertIn("discord_export_channel", by_name)
        self.assertIn("discord_analyze_export", by_name)
        self.assertIn("discord_convert_export", by_name)
        self.assertIn("discord_batch_convert_exports", by_name)

        guilds_schema = by_name["discord_list_guilds"]["input_schema"]
        token_anyof = guilds_schema["properties"]["token"]["anyOf"]
        self.assertEqual(token_anyof[0].get("minLength"), 1)

        export_schema = by_name["discord_export_channel"]["input_schema"]
        self.assertEqual(export_schema["properties"]["channel_id"].get("minLength"), 1)

        convert_schema = by_name["discord_convert_export"]["input_schema"]
        self.assertIn("jsonld", convert_schema["properties"]["to_format"].get("enum", []))

    def test_discord_list_guilds_rejects_blank_token(self) -> None:
        async def _run() -> None:
            result = await discord_list_guilds(token="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("token must be a non-empty string", str(result.get("error", "")))

        anyio.run(_run)

    def test_discord_list_channels_rejects_blank_guild_id(self) -> None:
        async def _run() -> None:
            result = await discord_list_channels(guild_id="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("guild_id is required", str(result.get("error", "")))

        anyio.run(_run)

    def test_discord_list_channels_rejects_blank_token(self) -> None:
        async def _run() -> None:
            result = await discord_list_channels(guild_id="g-1", token="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("token must be a non-empty string", str(result.get("error", "")))

        anyio.run(_run)

    def test_discord_list_channels_success_envelope_shape(self) -> None:
        async def _run() -> None:
            result = await discord_list_channels(guild_id="g-1")
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("tool"), "discord_list_channels")
            self.assertEqual(result.get("guild_id"), "g-1")

        anyio.run(_run)

    def test_discord_list_dm_channels_rejects_blank_token(self) -> None:
        async def _run() -> None:
            result = await discord_list_dm_channels(token="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("token must be a non-empty string", str(result.get("error", "")))

        anyio.run(_run)

    def test_discord_export_channel_validates_channel_and_flags(self) -> None:
        async def _run() -> None:
            missing_channel = await discord_export_channel(channel_id="   ")
            self.assertEqual(missing_channel.get("status"), "error")
            self.assertIn("channel_id is required", str(missing_channel.get("error", "")))

            invalid_flag = await discord_export_channel(channel_id="c-1", download_media="yes")  # type: ignore[arg-type]
            self.assertEqual(invalid_flag.get("status"), "error")
            self.assertIn("download_media must be a boolean", str(invalid_flag.get("error", "")))

        anyio.run(_run)

    def test_discord_analyze_export_validates_analysis_types(self) -> None:
        async def _run() -> None:
            result = await discord_analyze_export(export_path="/tmp/export.json", analysis_types=[""])  # type: ignore[list-item]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("analysis_types must be an array of non-empty strings", str(result.get("error", "")))

        anyio.run(_run)

    def test_discord_convert_export_validates_format_and_paths(self) -> None:
        async def _run() -> None:
            invalid_format = await discord_convert_export(input_path="in.json", output_path="out.json", to_format="xml")
            self.assertEqual(invalid_format.get("status"), "error")
            self.assertIn("to_format must be one of", str(invalid_format.get("error", "")))

            invalid_context = await discord_convert_export(
                input_path="in.json",
                output_path="out.json",
                context=["bad"],  # type: ignore[arg-type]
            )
            self.assertEqual(invalid_context.get("status"), "error")
            self.assertIn("context must be an object", str(invalid_context.get("error", "")))

        anyio.run(_run)

    def test_discord_batch_convert_exports_validates_pattern(self) -> None:
        async def _run() -> None:
            result = await discord_batch_convert_exports(input_dir="/tmp/in", output_dir="/tmp/out", file_pattern="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("file_pattern must be a non-empty string", str(result.get("error", "")))

        anyio.run(_run)

    def test_discord_wrappers_infer_error_status_from_contradictory_delegate_payloads(self) -> None:
        async def _contradictory_failure(**_: object) -> dict:
            return {"status": "success", "success": False, "error": "delegate failed"}

        async def _run() -> None:
            with patch.dict(
                native_discord_tools._API,
                {
                    "discord_list_guilds": _contradictory_failure,
                    "discord_list_channels": _contradictory_failure,
                    "discord_list_dm_channels": _contradictory_failure,
                    "discord_export_channel": _contradictory_failure,
                    "discord_analyze_export": _contradictory_failure,
                    "discord_convert_export": _contradictory_failure,
                    "discord_batch_convert_exports": _contradictory_failure,
                },
                clear=False,
            ):
                guilds = await discord_list_guilds()
                channels = await discord_list_channels(guild_id="g-1")
                dms = await discord_list_dm_channels()
                exported = await discord_export_channel(channel_id="c-1")
                analyzed = await discord_analyze_export(export_path="/tmp/export.json")
                converted = await discord_convert_export(input_path="in.json", output_path="out.json")
                batched = await discord_batch_convert_exports(input_dir="/tmp/in", output_dir="/tmp/out")

            self.assertEqual(guilds.get("status"), "error")
            self.assertEqual(guilds.get("error"), "delegate failed")

            self.assertEqual(channels.get("status"), "error")
            self.assertEqual(channels.get("error"), "delegate failed")

            self.assertEqual(dms.get("status"), "error")
            self.assertEqual(dms.get("error"), "delegate failed")

            self.assertEqual(exported.get("status"), "error")
            self.assertEqual(exported.get("error"), "delegate failed")

            self.assertEqual(analyzed.get("status"), "error")
            self.assertEqual(analyzed.get("error"), "delegate failed")

            self.assertEqual(converted.get("status"), "error")
            self.assertEqual(converted.get("error"), "delegate failed")

            self.assertEqual(batched.get("status"), "error")
            self.assertEqual(batched.get("error"), "delegate failed")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
