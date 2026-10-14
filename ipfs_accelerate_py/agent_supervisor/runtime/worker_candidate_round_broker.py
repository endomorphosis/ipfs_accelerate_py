"""A fresh read session for one retained owner candidate snapshot.

Only an in-memory owner snapshot can create this service. Serialized checkpoint
metadata cannot admit it. The descriptor routes advisory reads to one separately
registered child; it is neither an implementation-launch nor retry authority.
Baseline evidence stays separate, and this service cannot refresh its candidate
generation after edits or publish an accepted index.
"""
from __future__ import annotations

import hashlib
import math
import os
import time

from . import worker_candidate_round_client as protocol
from .numerical_obligation_broker import MAX_SESSION_SECONDS
from .owner_candidate_round_context import OwnerCandidateRoundContext
from .worker_candidate_context import WorkerCandidateContext
from .worker_source_broker import WorkerSourceBroker


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


class WorkerCandidateRoundBroker(WorkerSourceBroker):
    """One candidate-only source/artifact session; ordinary routes stay closed."""

    protocol = protocol
    file_prefix = "round-view"
    audit_schema = "supervisor-worker-candidate-round-session@1"

    def __init__(self, snapshot):
        _need(type(snapshot) is OwnerCandidateRoundContext,
              "exact retained owner candidate snapshot required")
        metadata = snapshot.revalidate()
        _need(type(metadata["round_number"]) is int and metadata["round_number"] == 2,
              "candidate source session requires round two")
        self._snapshot = snapshot
        self._snapshot_metadata = snapshot.metadata_bytes
        self._generation = metadata["workspace_generation"]
        # Shared descriptor timestamps are signed 63-bit nanoseconds. Clamp
        # before multiplication so a valid distant finite lease cannot overflow
        # a float; the shorter session never extends the original deadline.
        maximum_ns = (1 << 63) - 1
        self._deadline_ns = (maximum_ns if snapshot._absolute_deadline >= maximum_ns / 1_000_000_000
            else min(maximum_ns, int(snapshot._absolute_deadline * 1_000_000_000)))
        original = snapshot._source_broker
        self.repository, self.workspace = original.repository, original.workspace
        self.boundary = original.boundary
        self.owner_uid, self.worker_uid = original.owner_uid, original.worker_uid
        _need(self.owner_uid > 0 and self.worker_uid > 0
              and os.getuid() == os.geteuid() == self.owner_uid
              and os.getgid() == self.owner_uid,
              "candidate source broker requires the isolated nonroot owner group")
        self.prompt = snapshot.native_prompt.decode("utf-8")
        self._table_bytes = original._table_bytes
        self._binding = dict(metadata["baseline_binding"])
        self._view_admission = original._view_admission
        self.views = original.views
        self.source_edit_mode = self.semantic_evidence_mode = "disabled"
        self.source_edits = self.semantic_evidence = None
        self._edit_confirmation_blocked = False
        self._edit_confirmation_request = None
        self._source_continuations = {}
        self._descriptor_schema = protocol.DESCRIPTOR_SCHEMA
        self._session_binding = dict(self._binding,
            view_policy_binding_sha256=original._session_binding.get("view_policy_binding_sha256"),
            source_scope_sha256=metadata["baseline_scope_sha256"],
            workspace_generation=self._generation,
            round_context_sha256=_sha(self._snapshot_metadata),
            prior_session_id=metadata["prior_session_id"],
            prior_descriptor_sha256=metadata["descriptor_sha256"],
            round_number=2, candidate_scope_cid=metadata["candidate_scope_cid"],
            candidate_semantic_root_cid=metadata["semantic_root_cid"])
        self.candidate_context = WorkerCandidateContext(self.views,
            tuple(metadata["program_paths"]), builder=self._snapshot_artifacts)
        # No fresh lease: both advertised expiry and every actual read are
        # bounded by the original owner's remaining monotonic deadline.
        remaining = snapshot._absolute_deadline - time.monotonic()
        _need(remaining > 1, "candidate source session has less than one second remaining")
        timeout_seconds = min(MAX_SESSION_SECONDS, math.floor(remaining))
        self._start_session(identity=self._session_binding,
            boundary_sha=original.descriptor["boundary_manifest_sha256"],
            timeout_seconds=timeout_seconds, descriptor_schema=self._descriptor_schema)
        try:
            self._current_context()
        except BaseException:
            self.close()
            raise

    def _live(self):
        return super()._live() and time.monotonic_ns() < self._deadline_ns

    def _current_context(self, *, working=None, generation=None):
        # The shared session constructor calls without a query; actual baseline
        # requests call with False and remain forbidden even before any edit.
        _need(working is None or working is True, "candidate session refuses baseline evidence")
        _need(self._live(), "candidate source session expired")
        _need(self.descriptor.get("schema") == self._descriptor_schema
              and all(self.descriptor.get(key) == value for key, value in self._session_binding.items())
              and self.descriptor.get("boundary_manifest_sha256")
                  == self._snapshot._source_broker.descriptor["boundary_manifest_sha256"]
              and self.descriptor["expires_monotonic_ns"] <= self._deadline_ns,
              "candidate source descriptor or deadline changed")
        if getattr(self, "descriptor_sha256", None) is not None:
            _need(_sha(protocol.canonical_json(self.descriptor)) == self.descriptor_sha256,
                  "candidate source descriptor bytes changed")
        _need(generation is None or generation == self._generation,
              "candidate source generation differs")
        metadata = self._snapshot.revalidate()
        _need(protocol.canonical_json(metadata) == self._snapshot_metadata
              and metadata["workspace_generation"] == self._generation,
              "retained candidate snapshot changed")
        self.views.working_admission(expected_workspace_generation=self._generation)

    def _snapshot_artifacts(self, *, sources, program_paths, workspace_generation):
        metadata = self._snapshot.revalidate()
        _need(workspace_generation == self._generation
              and program_paths == tuple(metadata["program_paths"])
              and {name: _sha(raw) for name, raw in sources.items()} == metadata["source_hashes"],
              "candidate artifact request differs from retained source snapshot")
        artifacts = dict(self._snapshot.artifacts)
        _need(all(metadata["artifacts"][name] == dict(sha256=_sha(raw), bytes=len(raw))
                  for name, raw in artifacts.items()),
              "retained candidate artifact pins changed")
        return artifacts

    def _answer(self, request, channel):
        try:
            protocol._request_binding(request, self.descriptor)
            if request["schema"] in (protocol.REQUEST_SCHEMA_V2, protocol.REQUEST_SCHEMA_V3):
                cursor = request["query"]["cursor"]
                _need(cursor is None or self._source_continuations.get(cursor["token"])
                      == protocol.canonical_json(cursor),
                      "source continuation was not delivered in this candidate session")
        except (ValueError, TypeError, KeyError):
            return self._response(request, reason="invalid_request")
        return super()._answer(request, channel)

    def _response_finished(self, request, response, *, delivered):
        super()._response_finished(request, response, delivered=delivered)
        if (request is not None and request["schema"] in (
                protocol.REQUEST_SCHEMA_V2, protocol.REQUEST_SCHEMA_V3)
                and response is not None and response.get("ok") is True):
            cursor = request["query"]["cursor"]
            if cursor is not None:
                self._source_continuations.pop(cursor["token"], None)
            successor = response["page"]["next_cursor"]
            if delivered and not self._closed and successor is not None:
                self._source_continuations[successor["token"]] = protocol.canonical_json(successor)

    def close(self):
        try:
            super().close()
        finally:
            self._source_continuations.clear()


__all__ = ["WorkerCandidateRoundBroker"]
