#!/usr/bin/env python3
"""UNI-203 media import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.media_tools import (
    ffmpeg_analyze,
    ffmpeg_apply_filters,
    ffmpeg_batch_process,
    ffmpeg_concat,
    ffmpeg_convert,
    ffmpeg_cut,
    ffmpeg_demux,
    ffmpeg_mux,
    ffmpeg_probe,
    ffmpeg_splice,
    ffmpeg_stream_input,
    ffmpeg_stream_output,
    ytdlp_batch_download,
    ytdlp_download_playlist,
    ytdlp_download_video,
    ytdlp_extract_info,
    ytdlp_search_videos,
)
from ipfs_accelerate_py.mcp_server.tools.media_tools import native_media_tools


def test_media_package_exports_source_compatible_functions() -> None:
    assert ffmpeg_convert is native_media_tools.ffmpeg_convert
    assert ffmpeg_mux is native_media_tools.ffmpeg_mux
    assert ffmpeg_demux is native_media_tools.ffmpeg_demux
    assert ffmpeg_stream_input is native_media_tools.ffmpeg_stream_input
    assert ffmpeg_stream_output is native_media_tools.ffmpeg_stream_output
    assert ffmpeg_cut is native_media_tools.ffmpeg_cut
    assert ffmpeg_splice is native_media_tools.ffmpeg_splice
    assert ffmpeg_concat is native_media_tools.ffmpeg_concat
    assert ffmpeg_probe is native_media_tools.ffmpeg_probe
    assert ffmpeg_analyze is native_media_tools.ffmpeg_analyze
    assert ffmpeg_apply_filters is native_media_tools.ffmpeg_apply_filters
    assert ffmpeg_batch_process is native_media_tools.ffmpeg_batch_process
    assert ytdlp_download_video is native_media_tools.ytdlp_download_video
    assert ytdlp_download_playlist is native_media_tools.ytdlp_download_playlist
    assert ytdlp_extract_info is native_media_tools.ytdlp_extract_info
    assert ytdlp_search_videos is native_media_tools.ytdlp_search_videos
    assert ytdlp_batch_download is native_media_tools.ytdlp_batch_download