#!/usr/bin/env python3
"""UNI-130 discord tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.discord_tools.native_discord_tools import (
    discord_list_channels,
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

        guilds_schema = by_name["discord_list_guilds"]["input_schema"]
        token_anyof = guilds_schema["properties"]["token"]["anyOf"]
        self.assertEqual(token_anyof[0].get("minLength"), 1)

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


if __name__ == "__main__":
    unittest.main()
