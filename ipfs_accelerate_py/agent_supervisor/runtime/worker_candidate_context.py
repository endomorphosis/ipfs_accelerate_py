"""Bounded session-local candidate artifacts over owner-captured source bytes.

This service owns no database or proof index. It observes an already captured
working generation and never refreshes source, baseline evidence or proofs.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import threading

from .worker_source_views import MAX_FILES, MAX_SOURCE_BYTES, canonical_json, _path

PAGE_SCHEMA = "supervisor-worker-candidate-context-page@1"
CURSOR_SCHEMA = "supervisor-worker-candidate-context-cursor@1"
ARTIFACTS = frozenset({"summary", "capsules", "formal"})
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_CONTINUATIONS = 256
MAX_PAGE_BYTES = 65536
BINDING_FIELDS = ("task_id", "scope_cid", "semantic_root_cid", "semantic_sha256",
                  "native_prompt_sha256", "source_manifest_sha256", "profile")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _need(value, message):
    if not value:
        raise ValueError(message)


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


class WorkerCandidateContext:
    """One bounded candidate cache, served only while every source stays current."""

    def __init__(self, views, program_paths, builder=None):
        _need(type(program_paths) in (tuple, list) and len(program_paths) <= MAX_FILES,
            "bounded candidate program paths required")
        for name in program_paths:
            _path(name)
        _need(len(set(program_paths)) == len(program_paths), "duplicate candidate program path")
        _need(builder is None or callable(builder), "candidate artifact builder must be callable")
        self.views = views
        self.program_paths = tuple(sorted(program_paths))
        self._builder = builder
        admission = views.baseline_admission()
        self._baseline = canonical_json(admission)
        self._scope_sha256 = admission["scope_sha256"]
        self._binding = {key: admission[key] for key in BINDING_FIELDS if key != "profile"}
        self._binding["profile"] = admission["requested_profile"]
        self._generation = None
        self._producers = None
        self._artifacts = {}
        self._continuations = {}
        self._lock = threading.Lock()

    def _discard(self):
        self._generation = None
        self._producers = None
        self._artifacts.clear()
        self._continuations.clear()

    def clear(self):
        """Drop this session's candidate artifacts after a surrounding refusal."""
        with self._lock:
            self._discard()

    def _sources(self, generation):
        _need(canonical_json(self.views.baseline_admission()) == self._baseline,
            "candidate baseline admission changed")
        sources = self.views.working_sources(expected_workspace_generation=generation)
        _need(type(sources) is dict and 0 < len(sources) <= MAX_FILES
            and set(self.program_paths) <= set(sources), "candidate source population differs")
        _need(all(type(raw) is bytes for raw in sources.values())
            and sum(map(len, sources.values())) <= MAX_SOURCE_BYTES,
            "bounded captured candidate bytes required")
        return sources

    def _build(self, sources, generation):
        builder = self._builder
        if builder is None:
            from .worker_candidate_artifacts import build_candidate_artifacts
            builder = build_candidate_artifacts
        artifacts = builder(sources=dict(sources), program_paths=self.program_paths,
            workspace_generation=generation)
        _need(type(artifacts) is dict and set(artifacts) == ARTIFACTS,
            "exact candidate artifact set required")
        _need(all(type(raw) is bytes and raw for raw in artifacts.values())
            and sum(map(len, artifacts.values())) <= MAX_ARTIFACT_BYTES,
            "bounded candidate artifacts required")
        for raw in artifacts.values():
            try:
                value = json.loads(raw)
            except (ValueError, UnicodeError, RecursionError) as exc:
                raise ValueError("candidate artifact must be canonical JSON") from exc
            _need(type(value) is dict and canonical_json(value) == raw,
                "candidate artifact must be a canonical JSON object")
        return dict(artifacts)

    def _native_producers(self):
        if self._builder is not None:
            return None
        from .supervisor_meta_index import public_replay_without_metadata
        with public_replay_without_metadata():
            from .worker_candidate_artifacts import _producer_binding
            producers = _producer_binding()
        _need(type(producers) is dict and producers
            and all(type(name) is str and _digest(digest) for name, digest in producers.items()),
            "exact native candidate producer binding required")
        return canonical_json(producers)

    def page(self, *, artifact, expected_workspace_generation, cursor=None, max_bytes=8192):
        with self._lock:
            try:
                return self._page(artifact=artifact, generation=expected_workspace_generation,
                    cursor=cursor, max_bytes=max_bytes)
            except Exception:
                # Even malformed continuations discard the cache. No refusal
                # can leave a previous candidate available for accidental reuse.
                self._discard()
                raise

    def _page(self, *, artifact, generation, cursor, max_bytes):
        _need(type(artifact) is str and artifact in ARTIFACTS, "unsupported candidate artifact")
        _need(_digest(generation), "exact working source generation required")
        _need(type(max_bytes) is int and 1024 <= max_bytes <= MAX_PAGE_BYTES,
            "invalid complete candidate page byte bound")
        sources = self._sources(generation)
        producers = self._native_producers()
        if self._generation != generation:
            self._discard()
            _need(cursor is None, "candidate continuation requires its cached generation")
            artifacts = self._build(sources, generation)
            if producers is not None:
                summary = json.loads(artifacts["summary"])
                _need(canonical_json(summary.get("producer_sources")) == producers,
                    "candidate artifact producer binding differs")
            _need(self._sources(generation) == sources, "candidate source changed during build")
            self._artifacts = artifacts
            self._generation = generation
            self._producers = producers
        _need(self._producers == producers, "candidate producer sources changed")
        raw = self._artifacts[artifact]
        artifact_sha256 = _sha(raw)
        query = dict(baseline_scope_sha256=self._scope_sha256, workspace_generation=generation,
            artifact=artifact, artifact_sha256=artifact_sha256, max_bytes=max_bytes)
        request_sha256 = _sha(canonical_json(query))
        offset = 0
        if cursor is not None:
            _need(type(cursor) is dict and set(cursor) == {"schema", "request_sha256", "offset", "token"}
                and cursor["schema"] == CURSOR_SCHEMA and cursor["request_sha256"] == request_sha256
                and type(cursor["offset"]) is int and 0 < cursor["offset"] < len(raw)
                and type(cursor["token"]) is str and re.fullmatch(r"[0-9a-f]{32}", cursor["token"])
                and self._continuations.get(cursor["token"]) == canonical_json(cursor),
                "candidate continuation is foreign, forged or consumed")
            offset = cursor["offset"]
        token = secrets.token_hex(16)

        def render(size):
            end = offset + size
            part = raw[offset:end]
            try:
                content, encoding = part.decode("utf-8"), "utf-8"
            except UnicodeDecodeError:
                content, encoding = base64.b64encode(part).decode("ascii"), "base64"
            successor = None if end == len(raw) else dict(schema=CURSOR_SCHEMA,
                request_sha256=request_sha256, offset=end, token=token)
            return dict(schema=PAGE_SCHEMA, binding=dict(self._binding), workspace_generation=generation,
                source_state="working-candidate", baseline_evidence_applicable=False,
                proof_authority=False, completion_authority=False, source_obligations_discharged=0,
                artifact=artifact, artifact_sha256=artifact_sha256, artifact_bytes=len(raw), offset=offset,
                end_offset=end, encoding=encoding, content=content, slice_sha256=_sha(part),
                exhausted=successor is None, next_cursor=successor)

        low, high = 0, min(len(raw) - offset, max_bytes)
        complete = render(high)
        if high == len(raw) - offset and len(canonical_json(complete)) <= max_bytes:
            page = complete
        else:
            while low < high:
                middle = (low + high + 1) // 2
                if len(canonical_json(render(middle))) <= max_bytes:
                    low = middle
                else:
                    high = middle - 1
            _need(low > 0, "candidate page cannot fit one artifact byte")
            page = render(low)
        _need(len(canonical_json(page)) <= max_bytes, "candidate page exceeds byte bound")
        _need(self._sources(generation) == sources, "candidate source changed during retrieval")
        _need(self._native_producers() == producers, "candidate producer sources changed during retrieval")
        successor = page["next_cursor"]
        _need(successor is None or len(self._continuations) - (cursor is not None) < MAX_CONTINUATIONS,
            "candidate continuation population exceeds bound")
        if cursor is not None:
            del self._continuations[cursor["token"]]
        if successor is not None:
            self._continuations[successor["token"]] = canonical_json(successor)
        return page


__all__ = ["WorkerCandidateContext", "PAGE_SCHEMA", "CURSOR_SCHEMA", "ARTIFACTS"]
