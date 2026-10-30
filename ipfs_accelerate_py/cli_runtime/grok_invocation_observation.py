"""Invocation-local Grok identity observations without provider bodies or global last state.

These observations describe the native process/stream boundary, not authenticated
server identity, billing, task completion, or permission to retry. Only the final
native end envelope contributes modelUsage keys; usage-event/model/tool text does not.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
import re
import uuid

_MODEL = re.compile(r"grok-[0-9]{1,2}(?:\.[0-9]{1,2}){0,2}(?:-build)?\Z")
_CURRENT: ContextVar[object | None] = ContextVar("grok_generation_observation", default=None)


@dataclass(frozen=True)
class GrokInvocationObservation:
    invocation_id: str
    request_id: str
    native_dispatch_count: int = 0
    selected_provider_name: str = ""
    selected_model_name: str = ""
    provider_reported_model: str = ""
    process_returncode: int | None = None
    stream_complete: bool = False
    wire_model_observed: bool = False
    evidence_class: str = "absent"
    schema: str = "native-grok-invocation-observation@1"
    invocation_scope: str = "current_call"
    observation_only: bool = True
    raw_provider_data_exported: bool = False
    event_origin_authenticated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ObservedTextGeneration:
    text: str = field(repr=False)
    observation: GrokInvocationObservation


class _Capture:
    def __init__(self, request_id=None):
        self.identity = uuid.uuid4().hex
        self.request_id = request_id or self.identity
        self.count = 0
        self.observation = GrokInvocationObservation(self.identity, self.request_id)
        self.closed = False

    def begin(self):
        if self.closed:
            raise ValueError("closed Grok invocation observation")
        self.count += 1
        self.observation = GrokInvocationObservation(self.identity, self.request_id, self.count)
        self.dispatch_token = object()
        self.recorded = False
        return self.dispatch_token

    def record(self, payload, *, model, returncode, progress, dispatch_token):
        if self.closed:
            raise ValueError("closed Grok invocation observation")
        if (dispatch_token is None or dispatch_token is not getattr(self, "dispatch_token", None)
                or getattr(self, "recorded", False)):
            self.observation = GrokInvocationObservation(self.identity, self.request_id, self.count)
            self.recorded = True
            return
        self.recorded = True
        # A second launch cannot reuse the first launch's identity evidence,
        # even if an outer caller caught that launch's timeout/exception.
        if self.count != 1:
            return
        from .grok_stream_capture import validate_grok_stream_progress
        try:
            snapshot = validate_grok_stream_progress(progress)
        except (TypeError, ValueError, KeyError):
            return
        if (type(returncode) is not int or returncode != 0
                or snapshot["returncode"] != returncode
                or not snapshot["response_usable"]
                or snapshot["event_counts"]["end"] != 1
                or snapshot["event_counts"]["error"] != 0
                or snapshot["event_counts"]["max_turns_reached"] != 0
                or type(payload) is not dict or payload.get("type") != "end"
                or type(model) is not str or _MODEL.fullmatch(model) is None):
            return
        reasons = [payload[key] for key in ("stopReason", "stop_reason") if key in payload]
        if not reasons or any(type(reason) is not str or reason != "end_turn" for reason in reasons):
            return
        from .grok_native_usage import native_grok_observation
        outcome = native_grok_observation(payload, exit_code=returncode)
        if (outcome["native_grok_error_observed"] or outcome["native_grok_reason_code"] != "end_turn"
                or any(payload.get(key) not in (None, False, "", [], {})
                       for key in ("error", "errors", "exception", "error_message"))):
            return
        model_usage = payload.get("modelUsage")
        reported = ""
        if type(model_usage) is dict and len(model_usage) == 1:
            key = next(iter(model_usage))
            if type(key) is str and _MODEL.fullmatch(key) is not None and type(model_usage[key]) is dict:
                reported = key
        self.observation = GrokInvocationObservation(
            invocation_id=self.identity, request_id=self.request_id, native_dispatch_count=1,
            selected_provider_name="grok_cli", selected_model_name=model,
            provider_reported_model=reported, process_returncode=0,
            stream_complete=True, wire_model_observed=bool(reported),
            evidence_class="native_final_end_model_usage" if reported else "native_stream_without_model_identity",
        )


@contextmanager
def capture_grok_invocation(request_id=None):
    """Private native wrapper scope; never fall back to a prior thread/global run."""
    if _CURRENT.get() is not None:
        raise ValueError("nested Grok observation invocations are not supported")
    if request_id is not None and (type(request_id) is not str or re.fullmatch(r"[0-9a-f]{32}", request_id) is None):
        raise ValueError("Grok observation request id must be 32 lowercase hexadecimal characters")
    capture = _Capture(request_id)
    token = _CURRENT.set(capture)
    try:
        yield capture
    finally:
        capture.closed = True
        _CURRENT.reset(token)


def begin_native_grok_dispatch():
    """Count the native attempt before launch, including launch errors/timeouts."""
    capture = _CURRENT.get()
    return capture.begin() if isinstance(capture, _Capture) else None


def record_native_grok_result(payload, *, model, returncode, progress, dispatch_token):
    """Called only at the existing built-in Grok process finalization boundary."""
    capture = _CURRENT.get()
    if isinstance(capture, _Capture):
        capture.record(payload, model=model, returncode=returncode, progress=progress, dispatch_token=dispatch_token)
