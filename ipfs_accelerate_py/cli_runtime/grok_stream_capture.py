"""Bounded Grok 1.0.46 ``streaming-json`` transport, without a raw-output tee.

The pinned CLI's shipped headless reference defines LF-delimited ACP events and
an explicit final ``end`` event. That event is not an OS-process exit. This
module observes both independently; existing grok_native_usage projections
remain responsible for accounting and model-outcome interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import errno
import json
import math
import os
import selectors
import subprocess
import time
from typing import Callable, Mapping, Sequence

from .grok_stream_usage import GrokUsageEventObserver, validate_grok_usage_events

LEGACY_SCHEMA = "native-grok-stream-progress@1"
SCHEMA = "native-grok-stream-progress@2"
MAX_FRAME_BYTES = 256 * 1024
MAX_EXPLICIT_FRAME_BYTES = 8 * 1024 * 1024
MAX_TEXT_BYTES = 2 * 1024 * 1024
CALLBACK_INTERVAL_SECONDS = 5.0
TERMINAL_DRAIN_SECONDS = 3.0
_MAX_COUNT = (1 << 63) - 1
_EVENTS = ("text", "thought", "tool_call", "tool_call_update", "usage", "plan",
           "available_commands", "end", "error", "max_turns_reached", "auto_compact", "unknown")
_TOOLS = ("read_file", "search_replace", "grep", "list_dir", "todo_write", "run_terminal_cmd", "unknown")
_STATUSES = ("pending", "in_progress", "completed", "failed", "unknown")
_TERMINALS = frozenset({"end", "error", "max_turns_reached"})
_REASONS = frozenset({"end_turn", "max_turns", "max_tokens", "refusal", "cancelled",
                      "structured_output_retries", "execution_error", "process_error",
                      "schema_rejected", "timeout", "other_stop", "unknown"})
_COUNTERS = ("stdout_bytes", "stderr_bytes", "frames", "malformed_frames", "oversized_frames",
             "non_object_frames", "unknown_events", "events_after_final", "text_bytes",
             "discarded_text_bytes", "retained_text_bytes", "callback_errors")
_TIMES = ("first_stdout_seconds", "last_stdout_seconds", "first_stderr_seconds", "last_stderr_seconds",
          "first_event_seconds", "last_event_seconds", "final_event_seconds", "process_exit_seconds",
          "duration_seconds")
_BOOLS = ("stdout_eof", "stderr_eof", "process_exit_observed", "timed_out", "final_envelope_observed",
          "text_truncated", "stream_tainted", "stream_complete", "response_usable")
_CONSTANTS = {"schema": SCHEMA, "transport": "streaming-json",
              "text_scope": "last_usage_delimited_text_segment", "observation_only": True,
              "completion_authority": False, "execution_authority": False,
              "proof_authority": False, "raw_provider_data_exported": False, "event_origin_authenticated": False}
_KEYS = frozenset((*_CONSTANTS, *_COUNTERS, *_TIMES, *_BOOLS, "returncode", "timeout_phase",
                   "final_event_type", "final_reason_code", "event_counts", "tool_name_counts", "tool_status_counts"))


def _number(value):
    return type(value) in (int, float) and 0 <= value <= _MAX_COUNT and math.isfinite(value)


def validate_grok_stream_progress(value: object) -> dict:
    """Copy one closed diagnostic snapshot; reject extra fields and authority."""
    if type(value) is not dict:
        raise ValueError("invalid Grok progress fields")
    schema = value.get("schema")
    if type(schema) is not str or schema not in {LEGACY_SCHEMA, SCHEMA}:
        raise ValueError("invalid Grok progress schema")
    keys = _KEYS if schema == LEGACY_SCHEMA else _KEYS | {"usage_observations"}
    if set(value) != keys:
        raise ValueError("invalid Grok progress fields")
    for key, expected in _CONSTANTS.items():
        if key == "schema":
            expected = schema
        if type(value[key]) is not type(expected) or value[key] != expected:
            raise ValueError("invalid Grok progress constants")
    for key in _COUNTERS:
        if type(value[key]) is not int or not 0 <= value[key] <= _MAX_COUNT:
            raise ValueError("invalid Grok progress counter")
    for key in _TIMES:
        if value[key] is not None and not _number(value[key]):
            raise ValueError("invalid Grok progress time")
    for key in _BOOLS:
        if type(value[key]) is not bool:
            raise ValueError("invalid Grok progress flag")
    for key, names in (("event_counts", _EVENTS), ("tool_name_counts", _TOOLS), ("tool_status_counts", _STATUSES)):
        rows = value[key]
        if (type(rows) is not dict or set(rows) != set(names)
                or any(type(n) is not int or not 0 <= n <= _MAX_COUNT for n in rows.values())):
            raise ValueError("invalid Grok progress buckets")
    code = value["returncode"]
    if code is not None and (type(code) is not int or not -(1 << 31) <= code < (1 << 31)):
        raise ValueError("invalid Grok progress exit code")
    if type(value["timeout_phase"]) is not str or value["timeout_phase"] not in {"none", "process_wait", "pipe_drain"}:
        raise ValueError("invalid Grok progress timeout phase")
    terminal = value["final_event_type"]
    if terminal is not None and (type(terminal) is not str or terminal not in _TERMINALS):
        raise ValueError("invalid Grok progress final type")
    if type(value["final_reason_code"]) is not str or value["final_reason_code"] not in _REASONS:
        raise ValueError("invalid Grok progress final reason")
    if value["timed_out"] != (value["timeout_phase"] != "none"):
        raise ValueError("inconsistent Grok progress timeout")
    if value["process_exit_observed"] != (code is not None) or value["process_exit_observed"] != (value["process_exit_seconds"] is not None):
        raise ValueError("inconsistent Grok progress process exit")
    if value["final_envelope_observed"] != (terminal is not None) or value["final_envelope_observed"] != (value["final_event_seconds"] is not None):
        raise ValueError("inconsistent Grok progress final event")
    complete = (value["final_envelope_observed"] and not value["stream_tainted"] and not value["text_truncated"]
                and value["stdout_eof"] and value["stderr_eof"] and value["process_exit_observed"] and not value["timed_out"])
    if value["stream_complete"] != complete or value["response_usable"] != (complete and code == 0 and terminal == "end"
            and value["final_reason_code"] == "end_turn" and value["retained_text_bytes"] > 0):
        raise ValueError("inconsistent Grok progress completion")
    # Historical receipts have unknown event values; never synthesize zero rows.
    result = {key: dict(item) if type(item) is dict else item for key, item in value.items()}
    if schema == SCHEMA:
        result["usage_observations"] = validate_grok_usage_events(value["usage_observations"],
            observed_events=value["event_counts"]["usage"], frames=value["frames"],
            duration_seconds=value["duration_seconds"], final_event_seconds=value["final_event_seconds"],
            events_after_final=value["events_after_final"])
    return result


def grok_stream_progress_summary(value: object) -> dict:
    """Project count-only callback telemetry, retaining event rows only at exit.

    Repeating the entire bounded history every five seconds can exceed the
    outer runtime's log retention bound. The historical @1 contract precisely
    describes this projection; it does not invent missing per-event values.
    """
    result = validate_grok_stream_progress(value)
    result.pop("usage_observations", None)
    result["schema"] = LEGACY_SCHEMA
    return validate_grok_stream_progress(result)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("nonfinite JSON number")
    return parsed


def _nonfinite(_value):
    raise ValueError("nonfinite JSON number")


def validate_max_frame_bytes(value: object) -> int:
    """Validate a caller-selected bounded frame budget before process launch.

    Ordinary text calls retain MAX_FRAME_BYTES. Image readers may explicitly
    account for verified image payload bytes without disabling overflow checks.
    """
    if type(value) is not int or not 1 <= value <= MAX_EXPLICIT_FRAME_BYTES:
        raise ValueError("Grok frame bound must be an integer from 1 through 8 MiB")
    return value


class GrokStreamParser:
    """Incremental LF parser. Snapshots contain counts/times, never model data."""

    def __init__(self, *, max_frame_bytes: int = MAX_FRAME_BYTES, max_text_bytes: int = MAX_TEXT_BYTES):
        if any(type(n) is not int or n < 1 for n in (max_frame_bytes, max_text_bytes)):
            raise ValueError("positive stream bounds required")
        validate_max_frame_bytes(max_frame_bytes)
        if max_text_bytes > MAX_TEXT_BYTES:
            raise ValueError("stream bounds cannot exceed supported limits")
        self._max_frame = max_frame_bytes
        self._max_text = max_text_bytes
        self._pending = bytearray()
        self._discard_line = False
        self._text = bytearray()
        self._segment_closed = False
        self._final_payload = None
        self._usage_observer = GrokUsageEventObserver()
        self._state = {**_CONSTANTS, **dict.fromkeys(_COUNTERS, 0), **dict.fromkeys(_TIMES),
                       **dict.fromkeys(_BOOLS, False), "returncode": None, "timeout_phase": "none",
                       "final_event_type": None, "final_reason_code": "unknown", "event_counts": dict.fromkeys(_EVENTS, 0),
                       "tool_name_counts": dict.fromkeys(_TOOLS, 0), "tool_status_counts": dict.fromkeys(_STATUSES, 0)}

    def _inc(self, key, count=1, *, bucket=None):
        target = self._state if bucket is None else self._state[bucket]
        target[key] = min(_MAX_COUNT, target[key] + count)

    def _at(self, elapsed_seconds):
        if not _number(elapsed_seconds):
            raise ValueError("invalid stream observation time")
        self._state["duration_seconds"] = max(elapsed_seconds, self._state["duration_seconds"] or 0)

    def _activity(self, kind, elapsed_seconds):
        self._at(elapsed_seconds)
        if self._state[f"first_{kind}_seconds"] is None:
            self._state[f"first_{kind}_seconds"] = elapsed_seconds
        self._state[f"last_{kind}_seconds"] = elapsed_seconds

    def _taint(self):
        self._state["stream_tainted"] = True
        self._final_payload = None

    def _clear_text(self):
        self._inc("discarded_text_bytes", len(self._text))
        self._text.clear()

    def feed(self, chunk: bytes, *, elapsed_seconds: float = 0.0):
        if type(chunk) is not bytes:
            raise TypeError("stream input must be bytes")
        if self._state["stdout_eof"]:
            raise ValueError("stdout already closed")
        if not chunk:
            return
        self._activity("stdout", elapsed_seconds)
        self._inc("stdout_bytes", len(chunk))
        offset = 0
        while offset < len(chunk):
            end = chunk.find(b"\n", offset)
            stop = end if end >= 0 else len(chunk)
            length = stop - offset
            if not self._discard_line:
                if len(self._pending) + length > self._max_frame:
                    self._pending.clear()
                    self._discard_line = True
                    self._inc("oversized_frames")
                    self._taint()
                else:
                    self._pending.extend(chunk[offset:stop])
            if end < 0:
                break
            if not self._discard_line:
                self._frame(bytes(self._pending), elapsed_seconds)
            self._pending.clear()
            self._discard_line = False
            offset = end + 1

    def stderr(self, byte_count: int, *, elapsed_seconds: float):
        if type(byte_count) is not int or byte_count < 0:
            raise ValueError("invalid stderr byte count")
        if byte_count:
            self._activity("stderr", elapsed_seconds)
            self._inc("stderr_bytes", byte_count)

    def eof(self, channel: str, *, elapsed_seconds: float = 0.0):
        if channel not in {"stdout", "stderr"}:
            raise ValueError("invalid stream channel")
        self._at(elapsed_seconds)
        if self._state[f"{channel}_eof"]:
            return
        if channel == "stdout":
            if not self._discard_line and self._pending:
                self._frame(bytes(self._pending), elapsed_seconds)
            self._pending.clear()
        self._state[f"{channel}_eof"] = True

    def _frame(self, line, elapsed_seconds):
        if not line.strip():
            return
        self._inc("frames")
        if self._state["final_envelope_observed"]:
            self._inc("events_after_final")
            self._taint()
        try:
            event = json.loads(line.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_nonfinite, parse_float=_finite_float)
        except (ValueError, UnicodeError, RecursionError):
            self._inc("malformed_frames")
            self._taint()
            return
        if type(event) is not dict:
            self._inc("non_object_frames")
            self._taint()
            return
        self._activity("event", elapsed_seconds)
        kind = event.get("type")
        bucket = kind if type(kind) is str and kind in _EVENTS[:-1] else (
            "auto_compact" if type(kind) is str and len(kind) <= 128 and kind.startswith("auto_compact_") else "unknown")
        self._inc(bucket, bucket="event_counts")
        if bucket == "unknown":
            self._inc("unknown_events")
            self._taint()
        elif bucket == "text":
            data = event.get("data")
            try:
                encoded = data.encode("utf-8") if type(data) is str else None
            except UnicodeError:
                encoded = None
            if encoded is None:
                self._inc("malformed_frames")
                self._taint()
                return
            if self._segment_closed:
                self._clear_text()
                self._segment_closed = False
            self._inc("text_bytes", len(encoded))
            if len(self._text) + len(encoded) > self._max_text:
                self._state["text_truncated"] = True
                self._taint()
                self._inc("discarded_text_bytes", len(encoded))
                self._clear_text()
            elif not self._state["text_truncated"]:
                self._text.extend(encoded)
            else:
                self._inc("discarded_text_bytes", len(encoded))
        elif bucket in {"tool_call", "tool_call_update"}:
            if bucket == "tool_call":
                name = event.get("toolName")
                self._inc(name if type(name) is str and name in _TOOLS else "unknown", bucket="tool_name_counts")
            status = event.get("status")
            self._inc(status if type(status) is str and status in _STATUSES else "unknown", bucket="tool_status_counts")
        elif bucket == "usage":
            self._usage_observer.observe(event, frame_ordinal=self._state["frames"],
                elapsed_seconds=elapsed_seconds, after_final=self._state["final_envelope_observed"])
            self._segment_closed = True
            if event.get("stopReason") == "tool_use":
                self._clear_text()
        elif bucket in _TERMINALS:
            if not self._state["final_envelope_observed"]:
                self._state["final_envelope_observed"] = True
                self._state["final_event_type"] = bucket
                self._state["final_event_seconds"] = elapsed_seconds
                from .grok_native_usage import native_grok_observation
                self._state["final_reason_code"] = native_grok_observation(event, exit_code=0)["native_grok_reason_code"]
                if not self._state["stream_tainted"]:
                    self._final_payload = event

    def process_exit(self, returncode: int, *, elapsed_seconds: float):
        if type(returncode) is not int or not -(1 << 31) <= returncode < (1 << 31):
            raise ValueError("invalid process exit")
        self._at(elapsed_seconds)
        if not self._state["process_exit_observed"]:
            self._state.update(process_exit_observed=True, returncode=returncode, process_exit_seconds=elapsed_seconds)

    def timeout(self, *, process_running: bool, elapsed_seconds: float):
        self._at(elapsed_seconds)
        self._state.update(timed_out=True, timeout_phase="process_wait" if process_running else "pipe_drain")

    def callback_error(self):
        self._inc("callback_errors")

    @property
    def final_payload(self):
        return self._final_payload

    @property
    def response_text(self):
        return self._text.decode("utf-8")

    def snapshot(self, *, elapsed_seconds: float | None = None) -> dict:
        if elapsed_seconds is not None:
            self._at(elapsed_seconds)
        state = self._state
        state["usage_observations"] = self._usage_observer.snapshot()
        state["retained_text_bytes"] = len(self._text)
        state["stream_complete"] = bool(state["final_envelope_observed"] and not state["stream_tainted"]
            and not state["text_truncated"] and state["stdout_eof"] and state["stderr_eof"]
            and state["process_exit_observed"] and not state["timed_out"])
        state["response_usable"] = bool(state["stream_complete"] and state["returncode"] == 0 and state["final_event_type"] == "end"
            and state["final_reason_code"] == "end_turn" and state["retained_text_bytes"] > 0)
        return validate_grok_stream_progress(state)


@dataclass(frozen=True)
class GrokStreamResult:
    args: Sequence[str] = field(repr=False)
    returncode: int | None
    stdout: str = field(repr=False)
    stderr: str = field(repr=False)
    timed_out: bool
    progress: dict
    final_payload: dict | None = field(repr=False)
    response_text: str = field(repr=False)


def _terminal_drain_allowed(parser: GrokStreamParser, timeout: float) -> bool:
    """Permit only OS/pipe cleanup of a complete response observed on time."""
    state = parser.snapshot()
    return bool(state["final_event_type"] == "end" and state["final_reason_code"] == "end_turn"
                and state["final_event_seconds"] is not None and state["final_event_seconds"] < timeout
                and state["retained_text_bytes"] > 0 and not state["stream_tainted"]
                and not state["text_truncated"] and not parser._pending.strip()
                and not parser._discard_line and state["returncode"] in (None, 0))


def capture_grok_stream(command: Sequence[str], *, env: Mapping[str, str] | None,
                        timeout: float, cwd: str | os.PathLike | None = None,
                        progress_callback: Callable[[dict], None] | None = None,
                        max_frame_bytes: int = MAX_FRAME_BYTES) -> GrokStreamResult:
    """Keep the model deadline; bound terminal-only OS/pipe cleanup separately.

    Descendant custody belongs to the calling runtime. An inherited descendant
    pipe cannot extend the finite cleanup bound or authorize a partial response.
    Only a clean completed end_turn observed before the model deadline permits
    at most min(3 seconds, timeout) extra for natural exit and both pipe EOFs.
    It does not admit late model output, another turn, or a new invocation.
    Callback exceptions are diagnostic failures; cancellation still propagates
    through owned-child cleanup. Stream event origin is not authenticated.
    """
    if not _number(timeout) or timeout <= 0:
        raise ValueError("positive finite Grok stream timeout required")
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("Grok progress callback must be callable")
    parser = GrokStreamParser(max_frame_bytes=validate_max_frame_bytes(max_frame_bytes))
    started = time.monotonic()
    deadline = started + timeout
    terminal_deadline = deadline + min(TERMINAL_DRAIN_SECONDS, timeout)
    last_callback = -CALLBACK_INTERVAL_SECONDS
    process = None
    selector = selectors.DefaultSelector()

    def elapsed():
        return max(0.0, time.monotonic() - started)

    def callback(*, final=False):
        nonlocal last_callback
        now = elapsed()
        if progress_callback is not None and (final or now - last_callback >= CALLBACK_INTERVAL_SECONDS):
            last_callback = now
            try:
                progress_callback(grok_stream_progress_summary(parser.snapshot(elapsed_seconds=now)))
            except Exception:
                parser.callback_error()

    def observe_exit():
        code = process.poll()
        if code is not None:
            parser.process_exit(code, elapsed_seconds=elapsed())
        return code

    def reap_direct():
        if process is not None:
            if process.poll() is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            process.wait(timeout=5)
            observe_exit()

    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env=env, cwd=cwd, close_fds=True)
        for channel in ("stdout", "stderr"):
            pipe = getattr(process, channel)
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, channel)
        callback()
        while True:
            code = observe_exit()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # The terminal must already have been observed before the
                # original deadline. Neither late output nor process exit can
                # bypass that boundary, including when both pipes just closed.
                remaining = terminal_deadline - time.monotonic() if _terminal_drain_allowed(parser, timeout) else 0
                if remaining <= 0:
                    parser.timeout(process_running=code is None, elapsed_seconds=elapsed())
                    reap_direct()
                    break
            if code is not None and not selector.get_map():
                break
            for key, _mask in selector.select(min(0.1, remaining)):
                try:
                    chunk = os.read(key.fd, 64 * 1024)
                except OSError as exc:
                    if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                        continue
                    raise
                now = elapsed()
                if not chunk:
                    parser.eof(key.data, elapsed_seconds=now)
                    selector.unregister(key.fileobj)
                elif key.data == "stdout":
                    parser.feed(chunk, elapsed_seconds=now)
                else:
                    parser.stderr(len(chunk), elapsed_seconds=now)
            callback()
        callback(final=True)
        progress = parser.snapshot(elapsed_seconds=elapsed())
        payload = parser.final_payload
        text = parser.response_text if progress["response_usable"] else ""
        # Never synthesize success from malformed/incomplete output. A clean
        # native error envelope is retained for existing outcome projection.
        stdout = ""
        if progress["stream_complete"] and payload is not None:
            stdout = json.dumps({**payload, "text": text}, ensure_ascii=True, allow_nan=False)
        return GrokStreamResult(command, process.returncode, stdout, "", progress["timed_out"],
                                progress, payload, text)
    finally:
        try:
            if process is not None and process.poll() is None:
                reap_direct()
        finally:
            selector.close()
            if process is not None:
                for pipe in (process.stdout, process.stderr):
                    if pipe is not None:
                        pipe.close()
