"""Focused tests for the reusable Hugging Face Space compatibility client."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

import ipfs_accelerate_py
from ipfs_accelerate_py.hf_space_inference import (
    HFBucketBackend,
    HFSpaceClient,
)


def _response(payload: object = None) -> Mock:
    response = Mock()
    response.json.return_value = payload
    response.headers = {}
    return response


def _indextts_config() -> dict[str, object]:
    return {
        "dependencies": [
            {
                "id": 5,
                "api_name": False,
                "label": "update prompt audio",
                "inputs": [],
            },
            {
                "id": 6,
                "api_name": "gen_single",
                "label": "Generate a single response",
                "component_name": "button",
                "inputs": list(range(24)),
            },
        ]
    }


def test_package_exports_compatibility_client() -> None:
    assert ipfs_accelerate_py.HFSpaceClient is HFSpaceClient
    assert ipfs_accelerate_py.HFBucketBackend is HFBucketBackend


def test_endpoint_contract_resolution_and_arity() -> None:
    client = HFSpaceClient("https://example.hf.space", session=Mock())
    config = _indextts_config()

    assert client.dependency_api_names(config) == ["/gen_single"]
    assert client.resolve_fn_index("/gen_single", config) == 6
    assert client.resolve_fn_index(
        "/missing",
        config,
        fallback_markers=("generate",),
    ) == 6
    assert client.lookup_dependency_input_count(6, config) == 24


def test_config_is_cached_but_can_be_refreshed() -> None:
    session = Mock()
    session.get.return_value = _response(_indextts_config())
    client = HFSpaceClient("https://example.hf.space", session=session)

    first = client.get_config()
    second = client.get_config()
    refreshed = client.get_config(use_cache=False)

    assert first is second
    assert refreshed == first
    assert session.get.call_count == 2
    assert session.get.call_args_list[0].args == (
        "https://example.hf.space/config",
    )


def test_upload_uses_gradio_api_and_preserves_multipart_boundary() -> None:
    session = Mock()
    session.post.return_value = _response(
        [{"path": "/tmp/reference.wav", "orig_name": "reference.wav"}]
    )
    client = HFSpaceClient(
        "https://example.hf.space/",
        headers_factory=lambda: {
            "Authorization": "Bearer test",
            "Content-Type": "application/json",
        },
        session=session,
    )

    uploaded = client.upload_file(
        "reference.wav",
        b"RIFF",
        "audio/wav",
    )

    assert uploaded[0]["path"] == "/tmp/reference.wav"
    assert session.post.call_args.args[0] == (
        "https://example.hf.space/gradio_api/upload"
    )
    headers = session.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer test"
    assert not any(key.lower() == "content-type" for key in headers)
    assert session.post.call_args.kwargs["files"]["files"] == (
        "reference.wav",
        b"RIFF",
        "audio/wav",
    )


def test_queue_join_and_sse_completion() -> None:
    session = Mock()
    session.post.return_value = _response({"event_id": "event-1"})
    stream_response = _response()
    stream_response.iter_lines.return_value = [
        'data: {"msg":"estimation","rank":0}',
        (
            'data: {"msg":"process_completed","success":true,'
            '"output":{"data":[{"path":"/tmp/result.wav"}]}}'
        ),
    ]
    session.get.return_value = stream_response
    client = HFSpaceClient("https://example.hf.space", session=session)

    session_hash = client.queue_join(
        6,
        ["hello"],
        session_hash="session with spaces",
    )
    result = client.wait_for_queue_result(
        session_hash,
        timeout_seconds=2,
        poll_interval_seconds=0,
    )

    assert session_hash == "session with spaces"
    join_payload = session.post.call_args.kwargs["json"]
    assert join_payload == {
        "data": ["hello"],
        "fn_index": 6,
        "session_hash": "session with spaces",
    }
    assert result["data"][0]["path"] == "/tmp/result.wav"
    assert session.get.call_args.args[0].endswith(
        "session_hash=session%20with%20spaces"
    )


def test_queue_failure_is_not_treated_as_a_cacheable_result() -> None:
    with pytest.raises(RuntimeError, match="worker failed"):
        HFSpaceClient._resolve_terminal_event(
            {
                "msg": "process_completed",
                "success": False,
                "output": {"error": "worker failed"},
            },
            operation="queue",
        )


def test_fetch_file_supports_inline_and_relative_gradio_urls() -> None:
    session = Mock()
    client = HFSpaceClient("https://example.hf.space", session=session)

    inline = client.fetch_file(
        {"name": "result.wav", "_inline_bytes": b"RIFF"}
    )
    assert inline == (b"RIFF", "audio/x-wav")
    session.get.assert_not_called()

    file_response = _response()
    file_response.content = b"audio"
    file_response.headers = {"Content-Type": "audio/mpeg"}
    session.get.return_value = file_response
    downloaded = client.fetch_file(
        {"url": "/gradio_api/file=/tmp/result.mp3"}
    )
    assert downloaded == (b"audio", "audio/mpeg")
    assert session.get.call_args.args[0] == (
        "https://example.hf.space/gradio_api/file=/tmp/result.mp3"
    )


def test_bucket_exists_uses_current_read_only_hf_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                [
                    {
                        "type": "file",
                        "path": "run/audio/response.mp3",
                        "size": 100,
                    }
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(
        "ipfs_accelerate_py.hf_space_inference.subprocess.run",
        fake_run,
    )
    backend = HFBucketBackend(
        "hf://buckets/Publicus/abby-voice/run",
        hf_token="test-token",
    )

    assert backend.exists("audio/response.mp3") is True
    assert calls == [
        [
            "hf",
            "buckets",
            "list",
            (
                "hf://buckets/Publicus/abby-voice/run/"
                "audio/response.mp3"
            ),
            "--json",
        ]
    ]
    assert "ls-lh" not in calls[0]


def test_bucket_missing_object_and_recursive_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(
                    [
                        {"type": "file", "path": "run/audio/b.mp3"},
                        {"type": "directory", "path": "run/audio"},
                        {"type": "file", "path": "run/audio/a.mp3"},
                    ]
                ),
                stderr="",
            ),
        ]
    )

    def fake_run(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        result = next(responses)
        result.args = command
        return result

    monkeypatch.setattr(
        "ipfs_accelerate_py.hf_space_inference.subprocess.run",
        fake_run,
    )
    backend = HFBucketBackend("hf://buckets/Publicus/abby-voice/run")

    assert backend.exists("audio/missing.mp3") is False
    assert backend.list_files("audio") == [
        "run/audio/a.mp3",
        "run/audio/b.mp3",
    ]


def test_bucket_upload_uses_buckets_cp_without_touching_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "ipfs_accelerate_py.hf_space_inference.subprocess.run",
        fake_run,
    )
    source = tmp_path / "response.mp3"
    source.write_bytes(b"audio")
    backend = HFBucketBackend("hf://buckets/Publicus/abby-voice/run")

    assert backend.put_file(source, "audio/response.mp3") is True
    assert calls[0][:3] == ["hf", "buckets", "cp"]
    assert calls[0][-1].endswith("/audio/response.mp3")
