"""A host-native owner admission joined to one candidate read/dispatch session.

Only the actual issued native gate can construct this service. Its public V2
descriptor remains observation-only; source reads preserve the old bounded
grammar, while dispatch consumes one authenticated child nomination globally.
"""
from __future__ import annotations

import hashlib
import threading
import weakref

from . import candidate_dispatch_client as protocol
from .native_candidate_round_admission import NativeCandidateRoundAdmission
from .numerical_obligation_broker import MAX_TOTAL_RESPONSE_BYTES
from .worker_candidate_round_broker import WorkerCandidateRoundBroker
from .worker_source_broker import WorkerSourceBroker

_SESSIONS = weakref.WeakKeyDictionary()
_SESSIONS_LOCK = threading.Lock()


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


class CandidateDispatchBroker(WorkerCandidateRoundBroker):
    """One host-only dispatch exchange and candidate reads for the native child."""
    protocol = protocol
    file_prefix = "candidate-dispatch"
    audit_schema = "supervisor-worker-candidate-dispatch-session@1"

    def __init__(self, *, admission, provider, model, timeout, max_output_tokens, reasoning_effort="high"):
        _need(type(admission) is NativeCandidateRoundAdmission, "exact live native candidate admission required")
        policy = protocol.validate_policy(provider=provider, model=model, timeout=timeout,
            max_output_tokens=max_output_tokens, reasoning_effort=reasoning_effort)
        observation = NativeCandidateRoundAdmission.revalidate(admission)
        _need(observation["spawn_started"] is False and admission.process is None,
            "candidate dispatch broker must precede its admitted child")
        candidate_prompt = admission.candidate_prompt
        prompt_metadata = candidate_prompt.revalidate()
        _need(candidate_prompt._snapshot is admission._snapshot
            and admission.worker_attempt is admission._worker_attempt
            and observation["candidate_prompt_sha256"] == prompt_metadata["provider_prompt_sha256"],
            "candidate dispatch requires the same native snapshot, prompt and attempt")
        self._admission = admission
        self._admission_binding = admission._binding
        self._candidate_prompt = candidate_prompt
        self._candidate_prompt_bytes = candidate_prompt.provider_prompt
        self._policy = protocol.canonical_json(policy)
        self._dispatch_pins = dict(admission_id=observation["admission_id"],
            admission_binding_sha256=_sha(self._admission_binding),
            worker_attempt_id=observation["worker_attempt_id"],
            candidate_prompt_sha256=prompt_metadata["provider_prompt_sha256"],
            candidate_prompt_bytes=prompt_metadata["provider_prompt_bytes"],
            original_requirements_sha256=prompt_metadata["original_requirements_sha256"],
            original_native_prompt_sha256=prompt_metadata["native_prompt_sha256"],
            public_instruction_block_sha256=prompt_metadata["public_instruction_block_sha256"],
            absolute_deadline_ns=observation["absolute_deadline_ns"], **policy)
        self._dispatch_pins_bytes = protocol.canonical_json(self._dispatch_pins)
        self._dispatch_state = dict(spent=False, invocation_id=None, broker=weakref.ref(self),
            literal=(self._admission_binding, self._candidate_prompt_bytes, self._policy,
                self._dispatch_pins_bytes), references=(candidate_prompt, candidate_prompt._snapshot,
                admission.worker_attempt))
        with _SESSIONS_LOCK:
            _need(admission not in _SESSIONS, "native candidate admission already bound to a dispatch endpoint")
            # Even uncertain endpoint construction cannot create a second route.
            _SESSIONS[admission] = self._dispatch_state
        super().__init__(candidate_prompt._snapshot)

    def _start_session(self, *, identity, boundary_sha, timeout_seconds, descriptor_schema=None):
        self._descriptor_schema = protocol.DESCRIPTOR_SCHEMA
        self._session_binding = dict(identity, **self._dispatch_pins)
        return super()._start_session(identity=self._session_binding, boundary_sha=boundary_sha,
            timeout_seconds=timeout_seconds, descriptor_schema=protocol.DESCRIPTOR_SCHEMA)

    def _current_context(self, *, working=None, generation=None):
        super()._current_context(working=working, generation=generation)
        _need(type(self._admission) is NativeCandidateRoundAdmission
            and self._admission.candidate_prompt is self._candidate_prompt
            and self._candidate_prompt._snapshot is self._snapshot
            and self._admission.worker_attempt is self._admission._worker_attempt
            and self._admission._binding == self._admission_binding
            and self._candidate_prompt.provider_prompt == self._candidate_prompt_bytes
            and protocol.canonical_json(self._dispatch_pins) == self._dispatch_pins_bytes
            and protocol.canonical_json({key: self._dispatch_pins[key] for key in
                ("provider", "model", "timeout", "max_output_tokens", "reasoning_effort")}) == self._policy,
            "candidate dispatch retained admission or policy changed")
        with _SESSIONS_LOCK:
            state = _SESSIONS.get(self._admission)
            _need(state is self._dispatch_state and state["broker"]() is self
                and state["literal"] == (self._admission_binding, self._candidate_prompt_bytes,
                    self._policy, self._dispatch_pins_bytes)
                and all(a is b for a, b in zip(state["references"],
                    (self._candidate_prompt, self._snapshot, self._admission.worker_attempt))),
                "candidate dispatch endpoint lost its original issuance")
        current = NativeCandidateRoundAdmission.revalidate(self._admission)
        _need(current["admission_id"] == self._dispatch_pins["admission_id"]
            and current["worker_attempt_id"] == self._dispatch_pins["worker_attempt_id"]
            and current["candidate_prompt_sha256"] == self._dispatch_pins["candidate_prompt_sha256"]
            and current["absolute_deadline_ns"] == self._dispatch_pins["absolute_deadline_ns"],
            "candidate dispatch native binding changed")
        return current

    def _answer(self, request, channel):
        try:
            protocol._request_binding(request, self.descriptor)
            if request["schema"] in (protocol.round_client.REQUEST_SCHEMA_V2, protocol.round_client.REQUEST_SCHEMA_V3):
                cursor = request["query"]["cursor"]
                _need(cursor is None or self._source_continuations.get(cursor["token"])
                    == protocol.canonical_json(cursor), "candidate continuation was not delivered")
        except (ValueError, TypeError, KeyError):
            return self._response(request, reason="invalid_request")
        if request["schema"] != protocol.REQUEST_SCHEMA:
            # Skip the parent's V1 descriptor check, retaining its actual source
            # handler, two peer checks, native currentness and bounded pages.
            return WorkerSourceBroker._answer(self, request, channel)
        if not self._live():
            return self._response(request, reason="session_closed")
        if request["request_id"] in self._seen:
            return self._response(request, reason="invalid_request")
        self._seen.add(request["request_id"])
        if self._response_bytes + request["query"]["max_bytes"] > MAX_TOTAL_RESPONSE_BYTES:
            return self._response(request, reason="budget_exhausted")
        try:
            first_peer = self._peer(channel)
            _need(self._admission.process is self._process and self._process is not None,
                "dispatch peer must belong to the exact native spawned process")
        except (OSError, ValueError, RuntimeError):
            return self._response(request, reason="peer_refused")
        operation = request["query"]["operation"]
        with _SESSIONS_LOCK:
            if self._dispatch_state["spent"]:
                return self._response(request, reason="invalid_request")
            if operation == "consume":
                # Spend before final native observers and transport send. Failure
                # or lost acknowledgement never resurrects the nomination.
                self._dispatch_state["spent"] = True
                self._dispatch_state["invocation_id"] = request["query"]["invocation_id"]
        try:
            current = self._current_context()
            _need(current["spawn_completed"] is True and self._admission.process is self._process
                and self._process.poll() is None and self._peer(channel) == first_peer,
                "native candidate child or peer is not current")
            timeout_seconds = protocol._remaining(self.descriptor)
        except (OSError, ValueError, RuntimeError):
            return self._response(request, reason="stale_context")
        page = dict(schema=protocol.PAGE_SCHEMA, session_id=self.descriptor["session_id"],
            invocation_id=request["query"]["invocation_id"], operation=operation,
            workspace_generation=self._generation, boundary_manifest_sha256=self.descriptor["boundary_manifest_sha256"],
            dispatch_consumed=operation == "consume", timeout_seconds=timeout_seconds,
            host_native_admission_only=True, serialized_observation_grants_authority=False,
            provider_dispatch_observed=False, provider_attempted=None,
            proof_authority=False, completion_authority=False, retry_authority=False,
            production_container_dispatch_qualified=False, **self._dispatch_pins)
        response = self._response(request, page=page)
        protocol.validate_response(response, request=request, descriptor=self.descriptor)
        return response

    def _request_details(self, request=None):
        if request is not None and request.get("schema") == protocol.REQUEST_SCHEMA:
            return dict(candidate_dispatch_request=True, candidate_dispatch_operation=request["query"]["operation"],
                candidate_dispatch_invocation_id=request["query"]["invocation_id"])
        result = super()._request_details(request)
        return dict(result, candidate_dispatch_request=False, candidate_dispatch_operation=None,
            candidate_dispatch_invocation_id=None)

    def _record_details(self, response=None):
        page = response.get("page") if type(response) is dict and response.get("ok") is True else None
        dispatch = type(page) is dict and page.get("schema") == protocol.PAGE_SCHEMA
        result = super()._record_details(None if dispatch else response)
        return dict(result, candidate_dispatch_response=dispatch,
            candidate_dispatch_consumed=self._dispatch_state["spent"],
            provider_dispatch_observed=False, provider_attempted=None)


__all__ = ["CandidateDispatchBroker"]
