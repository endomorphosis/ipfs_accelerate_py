"""Read-only mapping and postimage checks for one worker-owned compact edit.

The broker supplies its exact source-view owner and independently verified
program paths. This service never writes a file, constructs a worktree owner,
or grants a lease. The isolated worker may apply a prepared span using its
existing workspace permissions. Confirmation observes that expected postimage;
ordinary task validation and publication remain separate native boundaries.

One successful preparation and one confirmation attempt are allowed. A failed
confirmation can follow a real worker write and must never invite reapplication.
Checks are sequential observations, not exclusion of concurrent writers.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import threading

from .source_view_edits import map_source_view_edit
from .worker_source_views import WorkerSourceViews, MAX_SOURCE_BYTES, canonical_json, _digest, _path

MODE = "mapped-span@1"
PREPARED_SCHEMA = "supervisor-worker-source-edit-prepared@1"
CONFIRMED_SCHEMA = "supervisor-worker-source-edit-confirmed@1"
MAX_EDIT_BYTES = 4096
MAX_REQUEST_BYTES = 8192
MIN_RESPONSE_BYTES = 4096
MAX_RESPONSE_BYTES = 8192
RESPONSE_RESERVE_BYTES = 1024
REASON_CODES = frozenset({"invalid_request", "edit_unavailable", "proposal_consumed",
                          "mapping_refused", "confirmation_refused"})
_PREPARE_FIELDS = frozenset({"operation", "path", "expected_workspace_generation",
    "expected_source_sha256", "expected_view_sha256", "expected_manifest_sha256",
    "view_start", "view_end", "expected_view_preimage", "replacement",
    "view_policy_binding_sha256", "max_bytes"})
_CONFIRM_FIELDS = frozenset({"operation", "proposal_sha256", "view_policy_binding_sha256", "max_bytes"})
_FLAGS = dict(validation_pending=True, owner_writes_performed=False,
    proof_authority=False, execution_authority=False, publication_authority=False,
    completion_authority=False, mutation_authority=False, semantic_equivalence_claimed=False,
    source_obligations_discharged=0, provider_calls=0, prover_calls=0)


class WorkerSourceEditError(ValueError):
    """A closed reason only; no source, path, or exception detail is exported."""

    def __init__(self, reason_code):
        if reason_code not in REASON_CODES:
            reason_code = "edit_unavailable"
        self.reason_code = reason_code
        super().__init__(reason_code)


def _need(condition, reason="invalid_request"):
    if not condition:
        raise WorkerSourceEditError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _query(value, operation):
    """Detach a closed bounded scalar query before any retained-state access."""
    fields = _PREPARE_FIELDS if operation == "prepare" else _CONFIRM_FIELDS
    _need(type(value) is dict and len(value) == len(fields) and set(value) == fields
          and type(value.get("operation")) is str and value["operation"] == operation)
    try:
        query = value
        _need(type(query["max_bytes"]) is int
              and MIN_RESPONSE_BYTES <= query["max_bytes"] <= MAX_RESPONSE_BYTES)
        policy = query["view_policy_binding_sha256"]
        _need(policy is None or _digest(policy))
        if operation == "prepare":
            _need(type(query["path"]) is str and len(query["path"]) <= 512
                  and len(query["path"].encode()) <= 512)
            _path(query["path"])
            for key in ("expected_workspace_generation", "expected_source_sha256",
                        "expected_view_sha256", "expected_manifest_sha256"):
                _need(_digest(query[key]))
            _need(type(query["view_start"]) is int and type(query["view_end"]) is int
                  and 0 <= query["view_start"] <= query["view_end"] <= MAX_SOURCE_BYTES)
            for key in ("expected_view_preimage", "replacement"):
                _need(type(query[key]) is str and len(query[key]) <= MAX_EDIT_BYTES and "\0" not in query[key]
                      and len(query[key].encode("utf-8", "strict")) <= MAX_EDIT_BYTES)
            _need(len(query["expected_view_preimage"].encode()) == query["view_end"] - query["view_start"])
        else:
            _need(_digest(query["proposal_sha256"]))
        raw = canonical_json(query)
        _need(len(raw) <= MAX_REQUEST_BYTES)
        return json.loads(raw)
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
        raise WorkerSourceEditError("invalid_request") from None


class WorkerSourceEditService:
    """A single broker session's pure mapping followed by a checked observation."""

    def __init__(self, *, views, allowed_paths):
        _need(type(views) is WorkerSourceViews, "edit_unavailable")
        _need(type(allowed_paths) in (tuple, list) and 0 < len(allowed_paths) <= 64,
              "edit_unavailable")
        try:
            _need(all(type(path) is str and len(path.encode()) <= 512 for path in allowed_paths),
                  "edit_unavailable")
            paths = tuple(_path(path) for path in allowed_paths)
            _need(len(set(paths)) == len(paths) and set(paths) <= set(views._hashes), "edit_unavailable")
            baseline = views.baseline_admission()
        except (OSError, ValueError, TypeError, UnicodeError, RuntimeError):
            raise WorkerSourceEditError("edit_unavailable") from None
        self.views = views
        self._allowed_paths = frozenset(paths)
        self._baseline = canonical_json(baseline)
        policy = baseline.get("view_policy_binding")
        self._policy_sha256 = None if policy is None else _sha(canonical_json(policy))
        self._lock = threading.RLock()
        self._proposal = None
        self._confirmation_attempted = False

    def _current(self, query):
        _need(query["view_policy_binding_sha256"] == self._policy_sha256, "invalid_request")
        _need(canonical_json(self.views.baseline_admission()) == self._baseline, "edit_unavailable")

    @staticmethod
    def _response(value, maximum):
        raw = canonical_json(value)
        _need(len(raw) <= maximum - RESPONSE_RESERVE_BYTES, "invalid_request")
        return json.loads(raw)

    def handle(self, query, *, request_sha256):
        """The authenticated broker computes the digest of the full request."""
        _need(type(query) is dict and type(query.get("operation")) is str
              and query["operation"] in {"prepare", "confirm"})
        return (self.prepare if query["operation"] == "prepare" else self.confirm)(
            query, request_sha256=request_sha256)

    def prepare(self, query, *, request_sha256):
        query = _query(query, "prepare")
        _need(_digest(request_sha256))
        with self._lock:
            _need(self._proposal is None, "proposal_consumed")
            try:
                self._current(query)
                path = query["path"]
                _need(path in self._allowed_paths, "mapping_refused")
                generation = query["expected_workspace_generation"]
                sources = self.views.working_sources(expected_workspace_generation=generation)
                original = sources[path]
                _need(b"\0" not in original, "mapping_refused")
                original.decode("utf-8", "strict")
                projection = self.views.working_projection(path=path, expected_workspace_generation=generation)
                _need(_sha(original) == query["expected_source_sha256"]
                      and _sha(projection.view) == query["expected_view_sha256"]
                      and projection.manifest_sha256 == query["expected_manifest_sha256"], "mapping_refused")
                preimage, replacement = (query[key].encode() for key in ("expected_view_preimage", "replacement"))
                mapped = map_source_view_edit(original=original, view=projection.view,
                    manifest=projection.manifest, expected_manifest_sha256=projection.manifest_sha256,
                    view_start=query["view_start"], view_end=query["view_end"],
                    expected_view_preimage=preimage, replacement=replacement)
                receipt = mapped.receipt
                _need(original[receipt["original_start"]:receipt["original_end"]] == preimage,
                      "mapping_refused")
                expected = dict(sources)
                expected[path] = mapped.candidate
                _need(sum(map(len, expected.values())) <= MAX_SOURCE_BYTES, "mapping_refused")
                result = dict(schema=PREPARED_SCHEMA, status="prepared_validation_pending",
                    prepare_request_sha256=request_sha256, path=path, source_generation=generation,
                    before_source_sha256=_sha(original), before_view_sha256=_sha(projection.view),
                    before_manifest_sha256=projection.manifest_sha256,
                    **{key: receipt[key] for key in ("original_start", "original_end", "preimage_sha256",
                        "preimage_bytes", "replacement_sha256", "replacement_bytes", "candidate_sha256",
                        "candidate_bytes", "candidate_view_sha256", "candidate_view_bytes", "candidate_manifest_sha256")},
                    view_policy_binding_sha256=self._policy_sha256, source_postimage_verified=False, **_FLAGS)
                result["proposal_sha256"] = _sha(canonical_json(dict(prepared=result,
                    session_nonce=secrets.token_hex(32), baseline_sha256=_sha(self._baseline))))
                result = self._response(result, query["max_bytes"])
                # Do not consume a failed preparation or silently adopt source drift.
                self._current(query)
                _need(self.views.working_sources(expected_workspace_generation=generation) == sources,
                      "mapping_refused")
                self._proposal = (canonical_json(result), expected)
                return result
            except WorkerSourceEditError:
                raise
            except (OSError, ValueError, TypeError, UnicodeError, RuntimeError):
                raise WorkerSourceEditError("mapping_refused") from None

    def confirm(self, query, *, request_sha256):
        query = _query(query, "confirm")
        _need(_digest(request_sha256))
        with self._lock:
            _need(self._proposal is not None and not self._confirmation_attempted, "proposal_consumed")
            # Even a refused well-formed confirmation consumes this observation.
            # The caller may already have written. Never invite a retrying edit.
            self._confirmation_attempted = True
            try:
                self._current(query)
                prepared_raw, expected = self._proposal
                prepared = json.loads(prepared_raw)
                _need(query["proposal_sha256"] == prepared["proposal_sha256"], "confirmation_refused")
                with self.views._lock:
                    _need(self.views._capture(self.views._workspace) == expected, "confirmation_refused")
                    self.views._discard_working()
                page = self.views.working_page(path=prepared["path"], max_bytes=MAX_RESPONSE_BYTES)
                generation = page["workspace_generation"]
                _need(generation != prepared["source_generation"]
                      and page["source_sha256"] == prepared["candidate_sha256"]
                      and page["view_sha256"] == prepared["candidate_view_sha256"]
                      and page["manifest_sha256"] == prepared["candidate_manifest_sha256"], "confirmation_refused")
                result = dict(prepared, schema=CONFIRMED_SCHEMA, status="applied_validation_pending",
                    source_postimage_verified=True, workspace_generation=generation,
                    source_population_sha256=_sha(canonical_json({path: _sha(raw) for path, raw in expected.items()})),
                    baseline_evidence_applicable=False)
                result = self._response(result, query["max_bytes"])
                self._current(query)
                _need(self.views.working_sources(expected_workspace_generation=generation) == expected,
                      "confirmation_refused")
                return result
            except (OSError, ValueError, TypeError, UnicodeError, RuntimeError):
                with self.views._lock:
                    self.views._discard_working()
                raise WorkerSourceEditError("confirmation_refused") from None


__all__ = ["WorkerSourceEditService", "WorkerSourceEditError", "MODE", "PREPARED_SCHEMA",
           "CONFIRMED_SCHEMA", "MAX_EDIT_BYTES", "MAX_REQUEST_BYTES", "MIN_RESPONSE_BYTES",
           "MAX_RESPONSE_BYTES", "RESPONSE_RESERVE_BYTES", "REASON_CODES"]
