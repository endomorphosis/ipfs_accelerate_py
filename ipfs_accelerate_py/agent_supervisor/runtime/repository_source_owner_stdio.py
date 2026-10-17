"""Worker stdio adapter for the independently owned read-only repository MCP.

The worker receives a public, hash-bound descriptor, never a native lease or
private index path. It relays the supported MCP newline-JSON subset through the
existing owner protocol, without retries, model calls or fallback providers.
Repository refresh can change the owner's private index; it cannot edit source.
Transport failures are not evidence that an owner-side operation did not occur.

Direct-file execution under Python ``-I`` imports only standard-library modules
and adjacent protocol/optional receipt helpers. This module does not import the
accelerator package or any datasets/resource-scheduler implementation.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import threading
import uuid

if __package__:
    from . import repository_source_owner_protocol as protocol
else:
    _spec = importlib.util.spec_from_file_location(
        "_repository_source_owner_protocol", Path(__file__).with_name("repository_source_owner_protocol.py"))
    protocol = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(protocol)

MAX_REQUEST_BYTES = protocol.MAX_MESSAGE_REQUEST_BYTES
MAX_RESPONSE_BYTES = protocol.MAX_MESSAGE_RESPONSE_BYTES
UNAVAILABLE = "repository_source_owner_stdio_unavailable"
FORWARDING_SCHEMA = "repository-source-worker-forwarding@1"
MAX_FORWARDING_EVENTS = 64
MAX_FORWARDING_RECEIPT_BYTES = 65536


class ForwardingObservation:
    """One-use, bounded worker-reported evidence; no retention or authority.

    Only hashes and sizes of validated owner replies and planned stdout enter
    this observation. A completed write means the existing write method
    returned, including its final checkpoint. It never proves provider receipt
    or consumption. The caller may explicitly retain ``receipt()`` elsewhere.
    Local parse/protocol errors are outside this owner-response observation.
    """
    def __init__(self):
        self._bridge_id = uuid.uuid4().hex
        self._used = self._started = self._finished = self._failed = self._interrupted = False
        self._use_lock = threading.Lock()
        self._session_id = self._descriptor_sha256 = self._returncode = None
        self._attempts = self._unconfirmed = 0
        self._active_exchange = False
        self._events = []

    def _start(self):
        self._started = True

    def _bind(self, descriptor, digest):
        if self._session_id is not None:
            raise ValueError("forwarding_observation_already_bound")
        self._session_id, self._descriptor_sha256 = descriptor["session_id"], digest

    def _exchange_started(self):
        if self._active_exchange or self._attempts >= MAX_FORWARDING_EVENTS:
            raise ValueError("forwarding_observation_exchange_limit")
        self._attempts += 1
        self._active_exchange = True

    def _response_validated(self, request, response):
        if not self._active_exchange or len(self._events) >= MAX_FORWARDING_EVENTS:
            raise ValueError("forwarding_observation_event_limit")
        output = (response["message"] if response["ok"] else
                  _error(request["message"], code=-32000, reason=response["reason_code"]))
        raw = None if output is None else protocol.canonical_json(output) + b"\n"
        owner_raw = protocol.canonical_json(response)
        event = dict(exchange_ordinal=self._attempts,
            request_sha256=hashlib.sha256(protocol.canonical_json(request)).hexdigest(),
            owner_response_sha256=hashlib.sha256(owner_raw).hexdigest(), owner_response_bytes=len(owner_raw),
            owner_ok=response["ok"], owner_reason_code=response["reason_code"],
            output_kind="no_output" if raw is None else "owner_message" if response["ok"] else "local_error",
            stdout_sha256=None if raw is None else hashlib.sha256(raw).hexdigest(),
            stdout_bytes=0 if raw is None else len(raw),
            forwarding_status="no_output" if raw is None else "unconfirmed")
        self._events.append(event)
        self._active_exchange = False
        return event

    def _exchange_unconfirmed(self):
        if self._active_exchange:
            self._unconfirmed += 1
            self._active_exchange = False

    def _output_finished(self, event, *, emitted):
        if event is None:
            return
        if not any(event is candidate for candidate in self._events):
            raise ValueError("forwarding_observation_event_differs")
        event["forwarding_status"] = ("budget_refused" if not emitted else
            "no_output" if event["output_kind"] == "no_output" else "write_completed")

    def _finish(self, returncode):
        self._exchange_unconfirmed()
        self._returncode = returncode if type(returncode) is int and returncode in (0, 2) else None
        self._interrupted = self._returncode is None
        self._finished = True

    def receipt(self):
        """Return an independent content-free snapshot, never a mutable handle."""
        complete = (self._used and self._finished and not self._failed and not self._interrupted
                    and self._session_id is not None)
        reason = ("not_started" if not self._used else "observation_failed" if self._failed
                  else "descriptor_unavailable" if self._session_id is None
                  else None if complete else "observation_incomplete")
        known = self._started and not self._failed
        result = dict(schema=FORWARDING_SCHEMA,
            status="observed" if complete else "not_started" if not self._used else "incomplete",
            complete=complete, reason=reason, bridge_id=self._bridge_id,
            session_id=self._session_id, descriptor_sha256=self._descriptor_sha256,
            returncode=self._returncode,
            owner_exchange_attempts=self._attempts if known else None,
            owner_exchange_unconfirmed=self._unconfirmed if known else None,
            observed_responses=len(self._events) if known else None, events=self._events,
            worker_reported=True, independently_verified=False, provider_ingestion_verified=False,
            token_savings_measured=False, proof_authority=False, execution_authority=False,
            completion_authority=False, raw_content_exported=False, observer_provider_calls=0)
        raw = protocol.canonical_json(result)
        if len(raw) > MAX_FORWARDING_RECEIPT_BYTES:
            raise ValueError("forwarding_observation_receipt_limit")
        return protocol.strict_json_loads(raw, max_bytes=MAX_FORWARDING_RECEIPT_BYTES)


def _observe(observation, method, *args, **kwargs):
    if observation is None:
        return None
    try:
        return getattr(observation, method)(*args, **kwargs)
    except Exception:
        # Observation is advisory. Ordinary accounting failures cannot change
        # an otherwise successful MCP exchange or reveal private exception text.
        observation._failed = True
        return None
    except BaseException:
        # A control-flow interruption, including one after finalization has
        # assigned a return code, cannot leave a complete observation behind.
        observation._failed = observation._interrupted = True
        observation._returncode = None
        raise


def _check(deadline, cancel_event):
    return protocol._check(deadline, cancel_event)


def _caller_deadline(value):
    if value is None:
        return None
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise protocol.RepositorySourceOwnerProtocolError("deadline_exceeded")
    return value


class _Stdio:
    """One synchronous reader/writer; no thread can outlive the adapter."""
    def __init__(self, input_fd, output_fd, *, deadline, cancel_event):
        if input_fd == output_fd:
            raise protocol.RepositorySourceOwnerProtocolError("transport_error")
        self.input_fd, self.output_fd = input_fd, output_fd
        self.deadline, self.cancel_event = deadline, cancel_event
        self.pending = bytearray()

    def checkpoint(self):
        _check(self.deadline, self.cancel_event)

    def readline(self):
        limit = MAX_REQUEST_BYTES + 2
        while True:
            self.checkpoint()
            newline = self.pending.find(b"\n", 0, limit)
            size = newline + 1 if newline >= 0 else limit if len(self.pending) >= limit else None
            if size is not None:
                raw = bytes(self.pending[:size])
                del self.pending[:size]
                return raw
            protocol._ready(self.input_fd, writing=False, deadline=self.deadline,
                            cancel_event=self.cancel_event)
            try:
                part = os.read(self.input_fd, min(65536, limit - len(self.pending)))
            except (BlockingIOError, InterruptedError):
                continue
            if not part:
                raw = bytes(self.pending)
                self.pending.clear()
                return raw
            self.pending.extend(part)

    def write(self, raw):
        if type(raw) is not bytes or not raw.endswith(b"\n") or len(raw) > MAX_RESPONSE_BYTES + 1:
            raise protocol.RepositorySourceOwnerProtocolError("invalid_response")
        self.checkpoint()
        offset, view = 0, memoryview(raw)
        while offset < len(view):
            protocol._ready(self.output_fd, writing=True, deadline=self.deadline,
                            cancel_event=self.cancel_event)
            try:
                count = os.write(self.output_fd, view[offset:offset + 65536])
            except (BlockingIOError, InterruptedError):
                continue
            if count <= 0:
                raise protocol.RepositorySourceOwnerProtocolError("transport_error")
            offset += count
        self.checkpoint()


def _error(message, *, code, reason):
    """Produce closed errors, preserving a valid request ID without echoing input."""
    identity = None
    if type(message) is dict:
        if "id" not in message and type(message.get("method")) is str:
            return None  # Notifications never acquire a synthetic response ID.
        candidate = message.get("id")
        if protocol._id(candidate):
            identity = candidate
    return {"jsonrpc": "2.0", "id": identity, "error": {"code": code, "message": reason}}


class _Budget:
    """Conservative local accounting; the owner independently enforces its budget."""
    def __init__(self, maximum):
        self.maximum, self.accounted = maximum, 0

    def can_request(self, descriptor, message):
        # Every valid owner response must at least fit one closed envelope.
        # This lower bound does not promise that the actual reply will fit;
        # emit() checks its complete size after the owner returns.
        minimum = {"schema": protocol.RESPONSE_SCHEMA, "session_id": descriptor["session_id"],
            "request_id": "0" * 32, "ok": False, "message": None,
            "reason_code": min(protocol.REASON_CODES, key=len)}
        size = len(protocol.canonical_json(minimum))
        if "id" not in message:
            minimum.update(ok=True, reason_code=None)
            size = min(size, len(protocol.canonical_json(minimum)))
        return size <= self.maximum - self.accounted

    def emit(self, streams, message, *, owner_response=None):
        # Owner accounting counts canonical envelopes, including notifications.
        # Local errors additionally consume their actual newline-output bytes.
        # Normal output is a subset of its envelope, so it is not counted twice.
        encoded = None if message is None else protocol.canonical_json(message) + b"\n"
        if encoded is not None and len(encoded) > MAX_RESPONSE_BYTES + 1:
            raise protocol.RepositorySourceOwnerProtocolError("invalid_response")
        envelope_bytes = 0 if owner_response is None else len(protocol.canonical_json(owner_response))
        charge = max(envelope_bytes, 0 if encoded is None else len(encoded))
        if charge > self.maximum - self.accounted:
            return False
        self.accounted += charge
        if encoded is not None:
            streams.write(encoded)
        return True


def _serve(descriptor_path, descriptor_sha256, *, workspace, descriptor, streams, observation=None):
    budget = _Budget(descriptor["max_total_response_bytes"])
    for _ in range(descriptor["max_requests"]):
        raw = streams.readline()
        if not raw:
            return 0
        if len(raw) > MAX_REQUEST_BYTES + 1 or not raw.endswith(b"\n"):
            budget.emit(streams, _error(None, code=-32600, reason="request_limit"))
            return 2
        try:
            message = protocol.strict_json_loads(raw[:-1], max_bytes=MAX_REQUEST_BYTES)
        except (ValueError, UnicodeError, RecursionError):
            if not budget.emit(streams, _error(None, code=-32700, reason="invalid_json")):
                return 2
            continue
        try:
            protocol.validate_message(message)
        except (ValueError, UnicodeError, RecursionError):
            if not budget.emit(streams, _error(message, code=-32600, reason="invalid_request")):
                return 2
            continue
        streams.checkpoint()
        if not budget.can_request(descriptor, message):
            return 2
        _observe(observation, "_exchange_started")
        try:
            response = protocol.request_message(descriptor_path, descriptor_sha256,
                message=message, workspace=workspace, deadline_monotonic=streams.deadline,
                cancel_event=streams.cancel_event)
            # request_message has already verified request-ID and SO_PEERCRED.
            # Recheck the public-session/message binding before emitting stdout.
            request = protocol.make_request(descriptor["session_id"], message,
                request_id=response.get("request_id") if type(response) is dict else None)
            protocol.validate_response(response, request=request, descriptor=descriptor)
            event = _observe(observation, "_response_validated", request, response)
        except Exception:
            _observe(observation, "_exchange_unconfirmed")
            streams.checkpoint()  # Cancellation/deadline must not attempt output.
            budget.emit(streams, _error(message, code=-32000, reason="owner_unavailable"))
            return 2
        streams.checkpoint()
        if response["ok"]:
            emitted = budget.emit(streams, response["message"], owner_response=response)
            _observe(observation, "_output_finished", event, emitted=emitted)
            if not emitted:
                return 2
        else:
            # Refusals carry a closed reason code; never print exception text,
            # source, owner context or any private descriptor/helper contents.
            emitted = budget.emit(streams, _error(message, code=-32000, reason=response["reason_code"]),
                                  owner_response=response)
            _observe(observation, "_output_finished", event, emitted=emitted)
            return 2
    return 0


def _serve_checked(descriptor_path, descriptor_sha256, *, workspace, incoming, outgoing,
                   deadline_monotonic=None, cancel_event=None, observation=None):
    """Relay one bounded session; an absent caller deadline uses descriptor expiry.

    The deadline is absolute across idle input, partial lines, owner calls and
    blocked output. Cancellation is polled at most every 0.1 seconds while
    waiting on an FD. Native work remains solely within the existing owner.
    EOF exits cleanly; protocol/transport failures never trigger a retry.
    """
    requested_deadline = _caller_deadline(deadline_monotonic)
    if requested_deadline is not None:
        _check(requested_deadline, cancel_event)
    descriptor = protocol.load_descriptor(descriptor_path, descriptor_sha256, workspace=workspace)
    deadline = descriptor["expires_monotonic_ns"] / 1_000_000_000
    if requested_deadline is not None:
        deadline = min(deadline, requested_deadline)
    _check(deadline, cancel_event)
    _observe(observation, "_bind", descriptor, descriptor_sha256)
    with ExitStack() as resources:
        input_fd = resources.enter_context(protocol._nonblocking(incoming))
        output_fd = resources.enter_context(protocol._nonblocking(outgoing))
        streams = _Stdio(input_fd, output_fd, deadline=deadline, cancel_event=cancel_event)
        return _serve(descriptor_path, descriptor_sha256, workspace=workspace,
                      descriptor=descriptor, streams=streams, observation=observation)


def serve(descriptor_path, descriptor_sha256, *, workspace, incoming, outgoing,
          deadline_monotonic=None, cancel_event=None, observation=None):
    """Return 0 on clean EOF/fulfilled local count, 2 on refusal or stop.

    Cancellation and absolute-deadline failures return 2 without an attempted
    late stdout write. The CLI emits only a fixed diagnostic; neither API
    surfaces private exception text or retries an owner operation.
    """
    if observation is not None:
        if type(observation) is not ForwardingObservation:
            raise ValueError("exact_forwarding_observation_required")
        with observation._use_lock:
            if observation._used:
                raise ValueError("forwarding_observation_already_used")
            observation._used = True
    result = None
    try:
        _observe(observation, "_start")
        result = _serve_checked(descriptor_path, descriptor_sha256, workspace=workspace,
            incoming=incoming, outgoing=outgoing, deadline_monotonic=deadline_monotonic,
            cancel_event=cancel_event, observation=observation)
        return result
    except Exception:
        result = 2
        return 2
    finally:
        _observe(observation, "_finish", result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(protocol.DESCRIPTOR_FLAG, type=Path, required=True)
    parser.add_argument(protocol.DIGEST_FLAG, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--worker-forwarding-target")
    args = parser.parse_args(argv)
    forwarding = target = observation = None
    if args.worker_forwarding_target is not None:
        try:
            if __package__:
                from . import repository_source_worker_receipts as forwarding
            else:
                spec = importlib.util.spec_from_file_location("_repository_source_worker_receipts",
                    Path(__file__).with_name("repository_source_worker_receipts.py"))
                forwarding = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(forwarding)
            target = forwarding.validate_target(forwarding.parse_target(args.worker_forwarding_target),
                workspace=args.workspace, descriptor_sha256=args.repository_source_owner_descriptor_sha256)
            observation = ForwardingObservation()
        except Exception:
            # Missing advisory storage must not prevent a source exchange.
            forwarding = target = observation = None
    pending_error = None
    try:
        result = serve(args.repository_source_owner_descriptor,
            args.repository_source_owner_descriptor_sha256, workspace=args.workspace,
            incoming=sys.stdin.buffer, outgoing=sys.stdout.buffer,
            **({"observation": observation} if observation is not None else {}))
        if result:
            print(UNAVAILABLE, file=sys.stderr)
        return result
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
        print(UNAVAILABLE, file=sys.stderr)
        return 2
    except BaseException as error:
        pending_error = error
        raise
    finally:
        if forwarding is not None and target is not None and observation is not None:
            try:
                forwarding.publish_receipt(target, observation.receipt())
            except Exception:
                # Worker-writable telemetry is advisory; no added stdout,
                # retry, owner request, or change to the bridge's exit status.
                pass
            except BaseException:
                # Preserve the original control-flow failure if shutdown was
                # already interrupted. A fresh interruption still propagates.
                if pending_error is None:
                    raise


if __name__ == "__main__":
    raise SystemExit(main())
