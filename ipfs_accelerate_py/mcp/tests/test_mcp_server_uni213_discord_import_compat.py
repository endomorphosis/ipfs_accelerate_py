#!/usr/bin/env python3
"""UNI-213 discord import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.discord_tools import (
    discord_analyze_export,
    discord_batch_convert_exports,
    discord_convert_export,
    discord_export_channel,
    discord_list_channels,
    discord_list_dm_channels,
    discord_list_guilds,
)
from ipfs_accelerate_py.mcp_server.tools.discord_tools import native_discord_tools


def test_discord_package_exports_supported_native_functions() -> None:
    assert discord_list_guilds is native_discord_tools.discord_list_guilds
    assert discord_list_channels is native_discord_tools.discord_list_channels
    assert discord_list_dm_channels is native_discord_tools.discord_list_dm_channels
    assert discord_export_channel is native_discord_tools.discord_export_channel
    assert discord_analyze_export is native_discord_tools.discord_analyze_export
    assert discord_convert_export is native_discord_tools.discord_convert_export
    assert discord_batch_convert_exports is native_discord_tools.discord_batch_convert_exports