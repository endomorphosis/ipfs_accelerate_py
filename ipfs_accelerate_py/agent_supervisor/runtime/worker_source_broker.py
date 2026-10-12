"""Task-scoped source views for the existing isolated coding session.

The verified semantic producer admits the public population. This read-only
adapter shares the numerical reader's socket/child lifecycle, creates no
database owner, and does not impersonate the typed owner's source.get grant.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import worker_source_client as protocol
from .numerical_obligation_broker import (
    MAX_SESSION_SECONDS, MAX_TOTAL_RESPONSE_BYTES, NumericalObligationBroker, _argument,
)


class WorkerSourceBroker(NumericalObligationBroker):
    protocol = protocol
    audit_schema = "supervisor-worker-source-session@1"
    file_prefix = "source-view"

    def __init__(self, *, repository, workspace, boundary, prompt, profile, timeout_seconds):
        from . import source_formalization_context as guard
        from .semantic_router_translation import encode_semantic_router_prompt
        from .worker_source_views import WorkerSourceViews

        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= MAX_SESSION_SECONDS:
            raise ValueError("bounded source-view session required")
        self.repository, self.workspace = guard._workspace(repository, workspace)
        self.boundary = Path(boundary).absolute()
        boundary_value, boundary_sha = guard._owner_boundary(self.boundary)
        self.owner_uid, self.worker_uid = boundary_value["owner_uid"], boundary_value["worker_uid"]
        if not (self.owner_uid > 0 and self.worker_uid > 0 and os.getgid() == self.owner_uid):
            raise ValueError("source-view broker requires the isolated nonroot owner group")
        self.prompt = prompt
        encoded = encode_semantic_router_prompt(prompt=prompt, repository=self.repository)
        table = encoded.table.to_dict()
        self._table_bytes = protocol.canonical_json(table)
        self._binding = protocol.semantic_binding(encoded, profile)
        self.views = WorkerSourceViews(repository=self.repository, workspace=self.workspace,
            source_hashes={name: binding["sha256"] for name, binding in table["source_manifest"].items()},
            **self._binding)
        self._view_admission = protocol.canonical_json(self.views.admission())
        self._start_session(identity=self._binding, boundary_sha=boundary_sha,
                            timeout_seconds=timeout_seconds)

    def _current_context(self):
        from . import source_formalization_context as guard
        from .semantic_router_translation import encode_semantic_router_prompt

        if not self._live():
            raise ValueError("source-view session expired")
        if guard._owner_boundary(self.boundary)[1] != self.descriptor["boundary_manifest_sha256"]:
            raise ValueError("source-view boundary changed")
        if any(self.descriptor.get(key) != value for key, value in self._binding.items()):
            raise ValueError("source-view descriptor binding changed")
        encoded = encode_semantic_router_prompt(prompt=self.prompt, repository=self.repository)
        if (protocol.canonical_json(encoded.table.to_dict()) != self._table_bytes
                or protocol.canonical_json(self.views.admission()) != self._view_admission):
            raise ValueError("source-view semantic admission changed")

    def _answer(self, request, channel):
        if not self._live():
            return self._response(request, reason="session_closed")
        if request["session_id"] != self.descriptor["session_id"] or request["request_id"] in self._seen:
            return self._response(request, reason="invalid_request")
        self._seen.add(request["request_id"])
        if self._response_bytes + request["query"]["max_bytes"] > MAX_TOTAL_RESPONSE_BYTES:
            return self._response(request, reason="budget_exhausted")
        try:
            self._peer(channel)
        except (OSError, ValueError):
            return self._response(request, reason="peer_refused")
        try:
            self._current_context()
        except (OSError, ValueError, RuntimeError):
            return self._response(request, reason="stale_context")
        overhead = len(protocol.canonical_json(self._response(request, page={}))) - 2
        query = dict(request["query"], max_bytes=request["query"]["max_bytes"] - overhead)
        try:
            page = self.views.page(**query)
        except (OSError, ValueError, RuntimeError):
            return self._response(request, reason="retrieval_refused")
        try:
            self._current_context()
            self._peer(channel)
        except (OSError, ValueError, RuntimeError):
            return self._response(request, reason="stale_context")
        response = self._response(request, page=page)
        if len(protocol.canonical_json(response)) > request["query"]["max_bytes"]:
            return self._response(request, reason="retrieval_refused")
        return response

    def _record_details(self, response=None):
        return dict(returned_source_bytes=(response["page"]["end_offset"] - response["page"]["offset"])
            if response is not None and response["ok"] else 0)


def prepare_worker_source_broker(*, argv, boundary, workspace, input_fd=0):
    """Opt in using an owner-sealed semantic prompt, without numerical context."""
    if (type(argv) is not list or any(type(arg) is not str for arg in argv)
            or any(arg == flag or arg.startswith(flag + "=") for arg in argv
                   for flag in (protocol.DESCRIPTOR_FLAG, protocol.DIGEST_FLAG))):
        raise ValueError("owner creates source-view descriptors independently")
    profile = _argument(argv, "--source-view-profile", "disabled")
    if profile == "disabled":
        return None
    if profile not in protocol.PROFILES:
        raise ValueError("unsupported source-view profile")
    if _argument(argv, "--purpose", "coding") != "coding" or "--preflight" in argv:
        raise ValueError("source-view broker requires an isolated coding invocation")
    repository = _argument(argv, "--semantic-repository")
    if repository is None:
        raise ValueError("source-view broker requires the selected semantic repository")
    import fcntl

    required = sum(getattr(fcntl, name, value) for name, value in (
        ("F_SEAL_SEAL", 1), ("F_SEAL_SHRINK", 2), ("F_SEAL_GROW", 4), ("F_SEAL_WRITE", 8)))
    if fcntl.fcntl(input_fd, getattr(fcntl, "F_GET_SEALS", 1034)) & required != required:
        raise ValueError("source-view broker requires the sealed owner prompt")
    size = os.fstat(input_fd).st_size
    if size > 2 * 1024 * 1024:
        raise ValueError("source-view prompt exceeds bound")
    prompt = os.pread(input_fd, size + 1, 0).decode("utf-8")
    return WorkerSourceBroker(repository=repository, workspace=workspace, boundary=boundary,
        prompt=prompt, profile=profile, timeout_seconds=int(_argument(argv, "--timeout", "90")))
