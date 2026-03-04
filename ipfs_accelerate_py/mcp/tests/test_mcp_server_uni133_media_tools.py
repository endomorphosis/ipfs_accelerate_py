#!/usr/bin/env python3
"""UNI-133 media tools parity hardening tests."""

from __future__ import annotations

import unittest

import anyio

from ipfs_accelerate_py.mcp_server.tools.media_tools.native_media_tools import (
    ffmpeg_analyze,
    register_native_media_tools,
    ytdlp_extract_info,
)


class _DummyManager:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_tool(self, **kwargs) -> None:
        self.calls.append(kwargs)


class TestMCPServerUNI133MediaTools(unittest.TestCase):
    def test_register_schema_contracts(self) -> None:
        manager = _DummyManager()
        register_native_media_tools(manager)
        by_name = {c["name"]: c for c in manager.calls}

        ffmpeg_schema = by_name["ffmpeg_analyze"]["input_schema"]
        input_one_of = ffmpeg_schema["properties"]["input_file"]["oneOf"]
        self.assertEqual(input_one_of[0].get("minLength"), 1)
        self.assertEqual(input_one_of[1].get("minProperties"), 1)

        yt_schema = by_name["ytdlp_extract_info"]["input_schema"]
        self.assertEqual(yt_schema["properties"]["url"].get("minLength"), 1)

    def test_ffmpeg_analyze_rejects_blank_string(self) -> None:
        async def _run() -> None:
            result = await ffmpeg_analyze(input_file="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("input_file must be a non-empty string or object", str(result.get("error", "")))

        anyio.run(_run)

    def test_ytdlp_extract_info_rejects_invalid_url(self) -> None:
        async def _run() -> None:
            result = await ytdlp_extract_info(url="example.com/video")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("url must start with http:// or https://", str(result.get("error", "")))

        anyio.run(_run)

    def test_ytdlp_extract_info_rejects_non_boolean_flags(self) -> None:
        async def _run() -> None:
            result = await ytdlp_extract_info(url="https://example.com/video", flat_playlist=1)  # type: ignore[arg-type]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("flat_playlist must be a boolean", str(result.get("error", "")))

        anyio.run(_run)

    def test_ytdlp_extract_info_success_envelope_shape(self) -> None:
        async def _run() -> None:
            result = await ytdlp_extract_info(url="https://example.com/video")
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("url"), "https://example.com/video")

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
