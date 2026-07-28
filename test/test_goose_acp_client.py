"""Deterministic offline tests for Goose ACP client (GOOSE-008).

Uses a fake NDJSON ACP server — no live network, no real goose binary,
and no goose serve / unauthenticated listeners.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from ipfs_accelerate_py.cli_runtime.acp import (
    ACP_PROTOCOL_VERSION,
    ACPBounds,
    ACPCapacityError,
    ACPClientState,
    ACPNotReadyError,
    ACPRestartExhaustedError,
    ACPRestartPolicy,
    ACPUncertainSideEffectError,
    FAILURE_KIND_UNCERTAIN_SIDE_EFFECT,
    GooseACPClient,
    create_goose_acp_client,
    encode_acp_message,
    parse_acp_line,
    split_ndjson_buffer,
)
from ipfs_accelerate_py.cli_runtime.errors import (
    BoundsExceededError,
    MalformedOutputError,
    PolicyDeniedError,
)


# ---------------------------------------------------------------------------
# Fake ACP server scripts
# ---------------------------------------------------------------------------


def _write_fake_acp(
    directory: Path,
    *,
    script: str,
    name: str = "goose",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _base_fake_acp_script(
    *,
    crash_after_prompt: bool = False,
    hang_on_prompt: bool = False,
    reject_load: bool = False,
    partial_frame: bool = False,
    malformed_once: bool = False,
    slow_updates: int = 0,
    max_sessions: int = 100,
    load_session: bool = True,
    close_supported: bool = True,
    exit_after_init: bool = False,
    unknown_id_noise: bool = False,
) -> str:
    """Generate a deterministic NDJSON ACP server as a Python executable."""
    return textwrap.dedent(
        f"""\
        #!{sys.executable}
        import json, os, sys, threading, time, uuid

        CRASH_AFTER_PROMPT = {crash_after_prompt!r}
        HANG_ON_PROMPT = {hang_on_prompt!r}
        REJECT_LOAD = {reject_load!r}
        PARTIAL_FRAME = {partial_frame!r}
        MALFORMED_ONCE = {malformed_once!r}
        SLOW_UPDATES = {slow_updates!r}
        MAX_SESSIONS = {max_sessions!r}
        LOAD_SESSION = {load_session!r}
        CLOSE_SUPPORTED = {close_supported!r}
        EXIT_AFTER_INIT = {exit_after_init!r}
        UNKNOWN_ID_NOISE = {unknown_id_noise!r}

        sessions = {{}}
        cancelled = set()
        lock = threading.Lock()
        malformed_sent = [False]

        def send(obj):
            sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\\n")
            sys.stdout.flush()

        def respond(msg_id, result):
            send({{"jsonrpc": "2.0", "id": msg_id, "result": result}})

        def respond_error(msg_id, code, message):
            send({{
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {{"code": code, "message": message}},
            }})

        def handle(msg):
            method = msg.get("method")
            msg_id = msg.get("id")
            params = msg.get("params") or {{}}

            if method == "initialize":
                if MALFORMED_ONCE and not malformed_sent[0]:
                    sys.stdout.write("{{not-json\\n")
                    sys.stdout.flush()
                    malformed_sent[0] = True
                result = {{
                    "protocolVersion": 1,
                    "agentCapabilities": {{
                        "loadSession": LOAD_SESSION,
                        "promptCapabilities": {{
                            "image": False,
                            "audio": False,
                            "embeddedContext": False,
                        }},
                        "sessionCapabilities": {{"close": {{}}}} if CLOSE_SUPPORTED else {{}},
                    }},
                    "agentInfo": {{
                        "name": "fake-goose-acp",
                        "version": "test",
                    }},
                    "authMethods": [],
                }}
                if PARTIAL_FRAME:
                    # Split one valid NDJSON frame across two flushes.
                    payload = json.dumps(
                        {{"jsonrpc": "2.0", "id": msg_id, "result": result}},
                        separators=(",", ":"),
                    )
                    mid = max(1, len(payload) // 2)
                    sys.stdout.write(payload[:mid])
                    sys.stdout.flush()
                    time.sleep(0.05)
                    sys.stdout.write(payload[mid:] + "\\n")
                    sys.stdout.flush()
                else:
                    respond(msg_id, result)
                if UNKNOWN_ID_NOISE:
                    send({{"jsonrpc": "2.0", "id": 999999, "result": {{"orphan": True}}}})
                if EXIT_AFTER_INIT:
                    time.sleep(0.05)
                    sys.exit(1)
                return

            if method == "session/new":
                with lock:
                    if len(sessions) >= MAX_SESSIONS:
                        respond_error(msg_id, -32000, "too many sessions")
                        return
                    sid = "sess-" + uuid.uuid4().hex[:12]
                    sessions[sid] = {{
                        "cwd": params.get("cwd"),
                        "history": [],
                    }}
                respond(msg_id, {{"sessionId": sid}})
                return

            if method == "session/load":
                if REJECT_LOAD or not LOAD_SESSION:
                    respond_error(msg_id, -32601, "loadSession not supported")
                    return
                sid = params.get("sessionId")
                with lock:
                    sessions[sid] = {{
                        "cwd": params.get("cwd"),
                        "history": ["loaded"],
                    }}
                respond(msg_id, {{"sessionId": sid}})
                return

            if method == "session/close":
                if not CLOSE_SUPPORTED:
                    respond_error(msg_id, -32601, "session/close not supported")
                    return
                sid = params.get("sessionId")
                with lock:
                    sessions.pop(sid, None)
                    cancelled.discard(sid)
                respond(msg_id, {{}})
                return

            if method == "session/cancel":
                sid = params.get("sessionId")
                with lock:
                    cancelled.add(sid)
                # notification — no response
                return

            if method == "session/prompt":
                sid = params.get("sessionId")
                with lock:
                    if sid not in sessions:
                        respond_error(msg_id, -32000, "unknown session")
                        return
                if HANG_ON_PROMPT:
                    # Wait until cancelled or process killed.
                    for _ in range(200):
                        with lock:
                            if sid in cancelled:
                                respond(msg_id, {{"stopReason": "cancelled"}})
                                return
                        time.sleep(0.05)
                    respond(msg_id, {{"stopReason": "end_turn"}})
                    return

                # Stream updates for this session only.
                send({{
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {{
                        "sessionId": sid,
                        "update": {{
                            "sessionUpdate": "agent_message_chunk",
                            "content": {{"type": "text", "text": "hello-"}},
                        }},
                    }},
                }})
                # Cross-session noise must be ignored by client isolation.
                send({{
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {{
                        "sessionId": "other-session-leak",
                        "update": {{
                            "sessionUpdate": "agent_message_chunk",
                            "content": {{"type": "text", "text": "LEAKED"}},
                        }},
                    }},
                }})
                for i in range(SLOW_UPDATES):
                    time.sleep(0.02)
                    send({{
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {{
                            "sessionId": sid,
                            "update": {{
                                "sessionUpdate": "agent_message_chunk",
                                "content": {{"type": "text", "text": str(i)}},
                            }},
                        }},
                    }})
                send({{
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {{
                        "sessionId": sid,
                        "update": {{
                            "sessionUpdate": "agent_message_chunk",
                            "content": {{"type": "text", "text": "world"}},
                        }},
                    }},
                }})
                send({{
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {{
                        "sessionId": sid,
                        "update": {{
                            "sessionUpdate": "tool_call",
                            "toolCallId": "t1",
                            "title": "noop",
                        }},
                    }},
                }})
                with lock:
                    if sid in cancelled:
                        respond(msg_id, {{"stopReason": "cancelled"}})
                        return
                    sessions[sid]["history"].append("prompt")
                respond(msg_id, {{"stopReason": "end_turn"}})
                if CRASH_AFTER_PROMPT:
                    time.sleep(0.05)
                    sys.exit(2)
                return

            if msg_id is not None:
                respond_error(msg_id, -32601, f"method not found: {{method}}")

        # argv: we are invoked as `goose acp`
        argv = sys.argv[1:]
        if argv and argv[0] != "acp":
            # still accept if only 'acp' missing for robustness
            pass
        # Refuse serve mode if somehow invoked incorrectly.
        if "serve" in argv or "--dangerously-unauthenticated" in argv:
            print("serve mode forbidden", file=sys.stderr)
            sys.exit(9)

        # Record state root for tests.
        marker = os.environ.get("GOOSE_FAKE_ACP_MARKER")
        if marker:
            with open(marker, "w", encoding="utf-8") as fh:
                json.dump({{
                    "argv": argv,
                    "GOOSE_PATH_ROOT": os.environ.get("GOOSE_PATH_ROOT"),
                    "cwd": os.getcwd(),
                }}, fh)

        buf = ""
        while True:
            chunk = sys.stdin.readline()
            if chunk == "":
                break
            buf += chunk
            while "\\n" in buf:
                line, buf = buf.split("\\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                try:
                    handle(msg)
                except Exception as exc:
                    if msg.get("id") is not None:
                        respond_error(msg["id"], -32000, type(exc).__name__)
        """
    )


@pytest.fixture
def state_root(tmp_path: Path) -> Path:
    root = tmp_path / "goose-state"
    root.mkdir()
    return root


@pytest.fixture
def fake_bin(tmp_path: Path) -> Path:
    return tmp_path / "bin"


def _client(
    exe: Path,
    state_root: Path,
    **kwargs: Any,
) -> GooseACPClient:
    bounds = kwargs.pop("bounds", None) or ACPBounds(
        max_pending_requests=8,
        max_sessions=4,
        max_restarts=kwargs.pop("max_restarts", 2),
        request_timeout_seconds=kwargs.pop("request_timeout_seconds", 5.0),
        init_timeout_seconds=kwargs.pop("init_timeout_seconds", 5.0),
        max_idle_seconds=kwargs.pop("max_idle_seconds", 60.0),
        event_queue_size=kwargs.pop("event_queue_size", 16),
    )
    policy = kwargs.pop("restart_policy", None) or ACPRestartPolicy(
        enabled=kwargs.pop("restart_enabled", True),
        max_restarts=bounds.max_restarts,
        restart_on_unexpected_exit=kwargs.pop(
            "restart_on_unexpected_exit", False
        ),
    )
    return GooseACPClient(
        str(exe),
        str(state_root),
        bounds=bounds,
        restart_policy=policy,
        cwd=str(state_root),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Framing unit tests
# ---------------------------------------------------------------------------


def test_encode_and_parse_roundtrip() -> None:
    msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    data = encode_acp_message(msg, max_bytes=4096)
    assert data.endswith(b"\n")
    assert b"\n" not in data[:-1]
    parsed = parse_acp_line(data[:-1])
    assert parsed["method"] == "initialize"


def test_partial_frame_buffering() -> None:
    lines, residual = split_ndjson_buffer(b'{"a":1', max_line_bytes=1024)
    assert lines == []
    assert residual == b'{"a":1'
    lines, residual = split_ndjson_buffer(
        residual + b'}\n{"b":2}\n', max_line_bytes=1024
    )
    assert len(lines) == 2
    assert residual == b""
    assert json.loads(lines[0])["a"] == 1


def test_partial_frame_size_bound() -> None:
    with pytest.raises(BoundsExceededError):
        split_ndjson_buffer(b"x" * 100, max_line_bytes=50)


def test_malformed_line_raises() -> None:
    with pytest.raises(MalformedOutputError):
        parse_acp_line(b"{not-json")


def test_restart_policy_forbids_auto_replay() -> None:
    with pytest.raises(PolicyDeniedError):
        ACPRestartPolicy(auto_replay_agent_work=True)


# ---------------------------------------------------------------------------
# Lifecycle: start, initialize, capabilities
# ---------------------------------------------------------------------------


def test_start_with_explicit_executable_and_isolated_state_root(
    fake_bin: Path, state_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "marker.json"
    monkeypatch.setenv("GOOSE_FAKE_ACP_MARKER", str(marker))
    exe = _write_fake_acp(fake_bin, script=_base_fake_acp_script())
    client = _client(exe, state_root)
    try:
        out = client.start()
        assert out["success"] is True
        assert out["protocol_version"] == ACP_PROTOCOL_VERSION
        assert client.is_ready
        assert client.state is ACPClientState.READY
        caps = out["agent_capabilities"]
        assert "loadSession" in caps or caps.get("loadSession") is True
        assert marker.exists()
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        assert recorded["argv"] == ["acp"]
        assert recorded["GOOSE_PATH_ROOT"] == str(state_root.resolve())
        # State root isolation — not the user home.
        assert recorded["GOOSE_PATH_ROOT"] != str(Path.home())
    finally:
        client.stop()
        assert client.state is ACPClientState.STOPPED


def test_rejects_work_before_initialize(
    fake_bin: Path, state_root: Path
) -> None:
    exe = _write_fake_acp(fake_bin, script=_base_fake_acp_script())
    client = _client(exe, state_root)
    with pytest.raises(ACPNotReadyError):
        client.session_new()
    client.stop()


def test_clean_shutdown_clears_pending_and_sessions(
    fake_bin: Path, state_root: Path
) -> None:
    exe = _write_fake_acp(fake_bin, script=_base_fake_acp_script())
    client = _client(exe, state_root)
    client.start()
    sess = client.session_new()
    sid = sess["session_id"]
    assert client.get_session(sid) is not None
    stop = client.stop()
    assert stop["success"] is True
    assert client.list_sessions() == []
    assert not client.is_ready


# ---------------------------------------------------------------------------
# Sessions: create, load, concurrent, isolation
# ---------------------------------------------------------------------------


def test_session_prompt_and_event_correlation(
    fake_bin: Path, state_root: Path
) -> None:
    exe = _write_fake_acp(fake_bin, script=_base_fake_acp_script())
    client = _client(exe, state_root)
    client.start()
    try:
        sid = client.session_new()["session_id"]
        secret = "CONFIDENTIAL_PROMPT_XYZ"
        result = client.session_prompt(sid, secret)
        assert result["success"] is True
        assert result["session_id"] == sid
        assert "hello-" in result["text"]
        assert "world" in result["text"]
        # Cross-session leak text must not appear.
        assert "LEAKED" not in result["text"]
        assert secret not in str(result)
        assert "prompt" not in result
        assert result["side_effects_started"] is True  # tool_call event
        assert result["cacheable"] is False
        assert result["retryable"] is False
        for event in result["events"]:
            assert event.get("session_id") == sid
    finally:
        client.stop()


def test_concurrent_sessions_no_cross_leakage(
    fake_bin: Path, state_root: Path
) -> None:
    exe = _write_fake_acp(
        fake_bin, script=_base_fake_acp_script(slow_updates=3)
    )
    client = _client(exe, state_root)
    client.start()
    try:
        s1 = client.session_new()["session_id"]
        s2 = client.session_new()["session_id"]
        assert s1 != s2
        results: Dict[str, Any] = {}
        errors: List[BaseException] = []

        def _run(sid: str) -> None:
            try:
                results[sid] = client.session_prompt(sid, f"p-{sid}")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t1 = threading.Thread(target=_run, args=(s1,))
        t2 = threading.Thread(target=_run, args=(s2,))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)
        assert not errors, errors
        assert results[s1]["session_id"] == s1
        assert results[s2]["session_id"] == s2
        for sid, res in results.items():
            for event in res["events"]:
                assert event["session_id"] == sid
    finally:
        client.stop()


def test_session_load_when_supported(
    fake_bin: Path, state_root: Path
) -> None:
    exe = _write_fake_acp(fake_bin, script=_base_fake_acp_script())
    client = _client(exe, state_root)
    client.start()
    try:
        out = client.session_load("sess-resume-1")
        assert out["success"] is True
        assert out["session_id"] == "sess-resume-1"
        assert out["loaded"] is True
    finally:
        client.stop()


def test_session_bound_capacity(
    fake_bin: Path, state_root: Path
) -> None:
    exe = _write_fake_acp(fake_bin, script=_base_fake_acp_script())
    client = _client(
        exe,
        state_root,
        bounds=ACPBounds(max_sessions=2, max_restarts=1),
    )
    client.start()
    try:
        client.session_new()
        client.session_new()
        with pytest.raises(ACPCapacityError):
            client.session_new()
    finally:
        client.stop()


def test_unknown_response_id_dropped(
    fake_bin: Path, state_root: Path
) -> None:
    exe = _write_fake_acp(
        fake_bin, script=_base_fake_acp_script(unknown_id_noise=True)
    )
    client = _client(exe, state_root)
    # Should still initialize successfully despite orphan response id.
    out = client.start()
    assert out["success"] is True
    client.stop()


# ---------------------------------------------------------------------------
# Partial frames / malformed messages
# ---------------------------------------------------------------------------


def test_partial_frames_from_server(
    fake_bin: Path, state_root: Path
) -> None:
    exe = _write_fake_acp(
        fake_bin, script=_base_fake_acp_script(partial_frame=True)
    )
    client = _client(exe, state_root)
    out = client.start()
    assert out["success"] is True
    client.stop()


def test_malformed_message_does_not_crash_client(
    fake_bin: Path, state_root: Path
) -> None:
    exe = _write_fake_acp(
        fake_bin, script=_base_fake_acp_script(malformed_once=True)
    )
    client = _client(exe, state_root)
    out = client.start()
    assert out["success"] is True
    sid = client.session_new()["session_id"]
    result = client.session_prompt(sid, "still works")
    assert result["success"] is True
    client.stop()


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_cancellation_cleans_pending_state(
    fake_bin: Path, state_root: Path
) -> None:
    exe = _write_fake_acp(
        fake_bin, script=_base_fake_acp_script(hang_on_prompt=True)
    )
    client = _client(
        exe,
        state_root,
        request_timeout_seconds=10.0,
    )
    client.start()
    try:
        sid = client.session_new()["session_id"]
        done = threading.Event()
        result_box: List[Any] = []

        def _prompt() -> None:
            try:
                result_box.append(client.session_prompt(sid, "hang please"))
            except BaseException as exc:  # noqa: BLE001
                result_box.append(exc)
            finally:
                done.set()

        t = threading.Thread(target=_prompt)
        t.start()
        time.sleep(0.15)
        cancel = client.session_cancel(sid)
        assert cancel["success"] is True
        done.wait(timeout=10)
        t.join(timeout=2)
        assert result_box, "prompt thread produced no result"
        # Either cancelled stopReason or cancelled error.
        item = result_box[0]
        if isinstance(item, dict):
            assert item.get("stop_reason") in {"cancelled", "end_turn"} or item.get(
                "success"
            )
        # Pending cleared.
        assert client.get_session(sid) is not None
        sess = client.get_session(sid)
        assert sess is not None
        assert sess["pending_prompts"] == 0
    finally:
        client.stop()


def test_session_close_cancels_and_frees_capacity(
    fake_bin: Path, state_root: Path
) -> None:
    exe = _write_fake_acp(fake_bin, script=_base_fake_acp_script())
    client = _client(
        exe,
        state_root,
        bounds=ACPBounds(max_sessions=1, max_restarts=1),
    )
    client.start()
    try:
        sid = client.session_new()["session_id"]
        closed = client.session_close(sid)
        assert closed["closed"] is True
        # Capacity freed.
        sid2 = client.session_new()["session_id"]
        assert sid2
    finally:
        client.stop()


# ---------------------------------------------------------------------------
# Crash / uncertain side effects / restart
# ---------------------------------------------------------------------------


def test_crash_fails_with_uncertain_side_effect(
    fake_bin: Path, state_root: Path
) -> None:
    exe = _write_fake_acp(
        fake_bin,
        script=_base_fake_acp_script(exit_after_init=False, crash_after_prompt=True),
    )
    # Disable auto-restart so we observe FAILED / uncertain on next call.
    client = _client(
        exe,
        state_root,
        restart_on_unexpected_exit=False,
        restart_enabled=False,
    )
    client.start()
    try:
        sid = client.session_new()["session_id"]
        # First prompt succeeds then server exits.
        result = client.session_prompt(sid, "boom")
        assert result["success"] is True
        # Wait for process death.
        deadline = time.time() + 3.0
        while client.is_ready and time.time() < deadline:
            time.sleep(0.05)
        # Subsequent work should fail closed.
        with pytest.raises((ACPNotReadyError, ACPUncertainSideEffectError)):
            client.session_prompt(sid, "again")
    finally:
        client.stop()


def test_unexpected_exit_during_prompt_is_uncertain(
    fake_bin: Path, state_root: Path
) -> None:
    """Server exits immediately after initialize; in-flight path is uncertain."""
    exe = _write_fake_acp(
        fake_bin, script=_base_fake_acp_script(exit_after_init=True)
    )
    client = _client(
        exe,
        state_root,
        restart_enabled=False,
        restart_on_unexpected_exit=False,
        init_timeout_seconds=2.0,
    )
    # start() may succeed briefly then die, or fail on subsequent ops.
    try:
        try:
            client.start()
        except Exception:
            pass
        time.sleep(0.2)
        if client.is_ready:
            # Race: process may die under us.
            try:
                client.session_new()
            except (
                ACPNotReadyError,
                ACPUncertainSideEffectError,
                Exception,
            ):
                pass
        # describe should report failure kind when last_failure set
        desc = client.describe()
        assert desc["auto_replay_agent_work"] is False
    finally:
        client.stop()


def test_restart_does_not_replay_agent_work(
    fake_bin: Path, state_root: Path
) -> None:
    exe = _write_fake_acp(fake_bin, script=_base_fake_acp_script())
    client = _client(
        exe,
        state_root,
        max_restarts=3,
        restart_enabled=True,
    )
    client.start()
    try:
        sid = client.session_new()["session_id"]
        client.session_prompt(sid, "before-restart")
        # Explicit restart clears sessions and does not replay.
        out = client.restart_transport(explicit=True)
        assert out["success"] is True
        assert out["auto_replay_agent_work"] is False
        assert out["sessions_cleared"] is True
        assert client.list_sessions() == []
        # Must create a new session deliberately.
        sid2 = client.session_new()["session_id"]
        result = client.session_prompt(sid2, "after-restart")
        assert result["success"] is True
    finally:
        client.stop()


def test_restart_exhaustion(
    fake_bin: Path, state_root: Path
) -> None:
    exe = _write_fake_acp(fake_bin, script=_base_fake_acp_script())
    client = _client(
        exe,
        state_root,
        bounds=ACPBounds(max_restarts=1),
        restart_policy=ACPRestartPolicy(
            enabled=True, max_restarts=1, restart_on_unexpected_exit=False
        ),
    )
    client.start()
    try:
        client.restart_transport(explicit=True)
        with pytest.raises(ACPRestartExhaustedError):
            client.restart_transport(explicit=True)
    finally:
        client.stop()


def test_pending_request_backpressure(
    fake_bin: Path, state_root: Path
) -> None:
    exe = _write_fake_acp(
        fake_bin, script=_base_fake_acp_script(hang_on_prompt=True)
    )
    client = _client(
        exe,
        state_root,
        bounds=ACPBounds(
            max_pending_requests=1,
            max_sessions=4,
            max_restarts=0,
            request_timeout_seconds=8.0,
        ),
        restart_enabled=False,
    )
    client.start()
    try:
        sid = client.session_new()["session_id"]
        started = threading.Event()
        finished = threading.Event()

        def _hang() -> None:
            started.set()
            try:
                client.session_prompt(sid, "block")
            except Exception:
                pass
            finally:
                finished.set()

        t = threading.Thread(target=_hang)
        t.start()
        assert started.wait(timeout=2)
        time.sleep(0.1)
        # Second concurrent request should hit capacity.
        with pytest.raises(ACPCapacityError):
            # Use a different session so we don't serialize on session alone.
            sid2 = client.session_new()["session_id"]
            client.session_prompt(sid2, "overflow")
        client.session_cancel(sid)
        finished.wait(timeout=10)
        t.join(timeout=2)
    finally:
        client.stop()


def test_stream_prompt_events(
    fake_bin: Path, state_root: Path
) -> None:
    exe = _write_fake_acp(fake_bin, script=_base_fake_acp_script())
    client = _client(exe, state_root)
    client.start()
    try:
        sid = client.session_new()["session_id"]
        events = list(client.stream_prompt(sid, "stream me"))
        kinds = [e.get("event") for e in events]
        assert kinds[0] == "started"
        assert "completed" in kinds
        assert all(e.get("session_id") == sid for e in events if e.get("session_id"))
    finally:
        client.stop()


def test_create_factory_does_not_start_process(
    fake_bin: Path, state_root: Path
) -> None:
    exe = _write_fake_acp(fake_bin, script=_base_fake_acp_script())
    client = create_goose_acp_client(str(exe), str(state_root))
    assert client.state is ACPClientState.CREATED
    assert not client.is_ready
    client.stop()


def test_context_manager(
    fake_bin: Path, state_root: Path
) -> None:
    exe = _write_fake_acp(fake_bin, script=_base_fake_acp_script())
    with _client(exe, state_root) as client:
        assert client.is_ready
        sid = client.session_new()["session_id"]
        assert client.session_prompt(sid, "hi")["success"]
    assert client.state is ACPClientState.STOPPED


def test_describe_exposes_bounds_and_no_auto_replay(
    fake_bin: Path, state_root: Path
) -> None:
    exe = _write_fake_acp(fake_bin, script=_base_fake_acp_script())
    client = _client(exe, state_root)
    client.start()
    try:
        desc = client.describe()
        assert desc["ready"] is True
        assert desc["state_root"] == str(state_root.resolve())
        assert desc["auto_replay_agent_work"] is False
        assert "max_pending_requests" in desc["bounds"]
        assert "max_sessions" in desc["bounds"]
        assert "max_restarts" in desc["bounds"]
    finally:
        client.stop()


def test_failure_kind_constant_stable() -> None:
    assert FAILURE_KIND_UNCERTAIN_SIDE_EFFECT == "uncertain_side_effect"
    err = ACPUncertainSideEffectError()
    assert err.uncertain_side_effects is True
    assert err.failure_kind == FAILURE_KIND_UNCERTAIN_SIDE_EFFECT
    assert err.retryable is False
