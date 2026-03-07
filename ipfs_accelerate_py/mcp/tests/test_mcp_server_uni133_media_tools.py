#!/usr/bin/env python3
"""UNI-133 media tools parity hardening tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import anyio

from ipfs_accelerate_py.mcp_server.tools.media_tools.native_media_tools import (
    ffmpeg_analyze,
    ffmpeg_batch_process,
    ffmpeg_cut,
    ffmpeg_mux,
    ffmpeg_stream_output,
    register_native_media_tools,
    ytdlp_batch_download,
    ytdlp_download_playlist,
    ytdlp_extract_info,
    ytdlp_search_videos,
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

        self.assertEqual(
            set(by_name),
            {
                "ffmpeg_probe",
                "ffmpeg_analyze",
                "ffmpeg_convert",
                "ffmpeg_mux",
                "ffmpeg_demux",
                "ffmpeg_stream_input",
                "ffmpeg_stream_output",
                "ffmpeg_cut",
                "ffmpeg_splice",
                "ffmpeg_concat",
                "ffmpeg_apply_filters",
                "ffmpeg_batch_process",
                "ytdlp_download_video",
                "ytdlp_download_playlist",
                "ytdlp_extract_info",
                "ytdlp_search_videos",
                "ytdlp_batch_download",
            },
        )

        ffmpeg_schema = by_name["ffmpeg_analyze"]["input_schema"]
        input_one_of = ffmpeg_schema["properties"]["input_file"]["oneOf"]
        self.assertEqual(input_one_of[0].get("minLength"), 1)
        self.assertEqual(input_one_of[1].get("minProperties"), 1)

        mux_schema = by_name["ffmpeg_mux"]["input_schema"]
        self.assertEqual((mux_schema["properties"]["output_file"] or {}).get("minLength"), 1)
        self.assertEqual((mux_schema["properties"]["metadata"] or {}).get("minProperties"), 1)

        batch_schema = by_name["ffmpeg_batch_process"]["input_schema"]
        batch_one_of = batch_schema["properties"]["input_files"]["oneOf"]
        self.assertEqual(batch_one_of[0].get("minItems"), 1)
        self.assertEqual(batch_one_of[1].get("minProperties"), 1)

        yt_schema = by_name["ytdlp_extract_info"]["input_schema"]
        self.assertEqual(yt_schema["properties"]["url"].get("minLength"), 1)

        playlist_schema = by_name["ytdlp_download_playlist"]["input_schema"]
        self.assertEqual((playlist_schema["properties"]["playlist_url"] or {}).get("minLength"), 1)

        batch_download_schema = by_name["ytdlp_batch_download"]["input_schema"]
        self.assertEqual((batch_download_schema["properties"]["urls"] or {}).get("minItems"), 1)

        stream_output_schema = by_name["ffmpeg_stream_output"]["input_schema"]
        self.assertEqual((stream_output_schema["properties"]["stream_url"] or {}).get("minLength"), 1)

        search_schema = by_name["ytdlp_search_videos"]["input_schema"]
        self.assertEqual((search_schema["properties"]["query"] or {}).get("minLength"), 1)

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

    def test_ffmpeg_mux_requires_at_least_one_input_stream(self) -> None:
        async def _run() -> None:
            result = await ffmpeg_mux(output_file="/tmp/output.mp4")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("At least one input stream must be provided", str(result.get("error", "")))

        anyio.run(_run)

    def test_ffmpeg_cut_requires_exactly_one_end_boundary(self) -> None:
        async def _run() -> None:
            result = await ffmpeg_cut(
                input_file="/tmp/video.mp4",
                output_file="/tmp/clip.mp4",
                start_time="00:00:01",
                end_time="00:00:02",
                duration="1",
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("Specify exactly one of end_time or duration", str(result.get("error", "")))

        anyio.run(_run)

    def test_ffmpeg_batch_process_rejects_invalid_parallelism(self) -> None:
        async def _run() -> None:
            result = await ffmpeg_batch_process(
                input_files=["/tmp/a.mp4"],
                output_directory="/tmp/out",
                max_parallel=0,
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("max_parallel must be an integer greater than or equal to 1", str(result.get("error", "")))

        anyio.run(_run)

    def test_ytdlp_extract_info_rejects_non_boolean_flags(self) -> None:
        async def _run() -> None:
            result = await ytdlp_extract_info(url="https://example.com/video", flat_playlist=1)  # type: ignore[arg-type]
            self.assertEqual(result.get("status"), "error")
            self.assertIn("flat_playlist must be a boolean", str(result.get("error", "")))

        anyio.run(_run)

    def test_ytdlp_download_playlist_rejects_invalid_start_end_window(self) -> None:
        async def _run() -> None:
            result = await ytdlp_download_playlist(
                playlist_url="https://example.com/playlist",
                start_index=5,
                end_index=3,
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn(
                "end_index must be an integer greater than or equal to start_index when provided",
                str(result.get("error", "")),
            )

        anyio.run(_run)

    def test_ytdlp_batch_download_rejects_invalid_url_entries(self) -> None:
        async def _run() -> None:
            result = await ytdlp_batch_download(urls=["https://example.com/video", "bad-url"])
            self.assertEqual(result.get("status"), "error")
            self.assertIn("url must start with http:// or https://", str(result.get("error", "")))

        anyio.run(_run)

    def test_ffmpeg_stream_output_rejects_blank_stream_url(self) -> None:
        async def _run() -> None:
            result = await ffmpeg_stream_output(
                input_file="/tmp/video.mp4",
                stream_url="   ",
            )
            self.assertEqual(result.get("status"), "error")
            self.assertIn("stream_url must be a non-empty string", str(result.get("error", "")))

        anyio.run(_run)

    def test_ytdlp_search_videos_rejects_blank_query(self) -> None:
        async def _run() -> None:
            result = await ytdlp_search_videos(query="   ")
            self.assertEqual(result.get("status"), "error")
            self.assertIn("query must be a non-empty string", str(result.get("error", "")))

        anyio.run(_run)

    def test_ytdlp_extract_info_success_envelope_shape(self) -> None:
        async def _run() -> None:
            result = await ytdlp_extract_info(url="https://example.com/video")
            self.assertIn(result.get("status"), ["success", "error"])
            self.assertEqual(result.get("url"), "https://example.com/video")

        anyio.run(_run)

    def test_ffmpeg_analyze_minimal_success_payload_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.media_tools.native_media_tools._API"
            ) as mock_api:
                mock_api.__getitem__.return_value = lambda **_: {"status": "success"}

                result = await ffmpeg_analyze(input_file="/tmp/video.mp4")

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("input_file"), "/tmp/video.mp4")
            self.assertEqual(result.get("metadata"), {})

        anyio.run(_run)

    def test_ytdlp_extract_info_minimal_success_payload_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.media_tools.native_media_tools._API"
            ) as mock_api:
                mock_api.__getitem__.return_value = lambda **_: {"status": "success"}

                result = await ytdlp_extract_info(url="https://example.com/video")

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("success"), True)
            self.assertEqual(result.get("url"), "https://example.com/video")
            self.assertEqual(result.get("info"), {})

        anyio.run(_run)

    def test_ffmpeg_mux_minimal_success_payload_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.media_tools.native_media_tools._API"
            ) as mock_api:
                mock_api.__getitem__.return_value = lambda **_: {"status": "success"}

                result = await ffmpeg_mux(video_input="/tmp/video.mp4", output_file="/tmp/output.mp4")

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("output_file"), "/tmp/output.mp4")
            self.assertEqual(result.get("inputs"), {"video": "/tmp/video.mp4", "audio": [], "subtitle": []})

        anyio.run(_run)

    def test_ytdlp_batch_download_minimal_success_payload_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.media_tools.native_media_tools._API"
            ) as mock_api:
                mock_api.__getitem__.return_value = lambda **_: {"status": "success"}

                result = await ytdlp_batch_download(urls=["https://example.com/video"])

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("urls"), ["https://example.com/video"])
            self.assertEqual(result.get("results"), [])

        anyio.run(_run)

    def test_ytdlp_search_videos_minimal_success_payload_defaults(self) -> None:
        async def _run() -> None:
            with patch(
                "ipfs_accelerate_py.mcp_server.tools.media_tools.native_media_tools._API"
            ) as mock_api:
                mock_api.__getitem__.return_value = lambda **_: {"status": "success"}

                result = await ytdlp_search_videos(query="unit test")

            self.assertEqual(result.get("status"), "success")
            self.assertEqual(result.get("query"), "unit test")
            self.assertEqual(result.get("results"), [])

        anyio.run(_run)


if __name__ == "__main__":
    unittest.main()
