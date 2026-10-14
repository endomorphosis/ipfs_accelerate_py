"""Exact selected baseline capsules retained by the existing source broker.

The caller has already verified the native semantic producer. This bounded
pager owns no source, index, freshness or proof authority; the broker must check
its ordinary baseline admission both before and after every page.
"""
from __future__ import annotations

import base64
import hashlib
import secrets
import threading

from . import worker_semantic_evidence_client as protocol


class WorkerSemanticEvidence:
    def __init__(self, *, capsules, binding):
        protocol._need(type(capsules) is list and len(capsules) <= protocol.MAX_SELECTED_CAPSULES
            and all(type(row) is dict for row in capsules), "bounded selected native capsules required")
        protocol.validate_binding(binding)
        self._binding_bytes = protocol.canonical_json(binding)
        self._capsules = tuple(protocol.canonical_json(row) for row in capsules)
        protocol._need(sum(map(len, self._capsules)) <= protocol.MAX_SELECTED_CAPSULE_BYTES,
            "selected capsule population exceeds byte bound")
        self._continuations = {}
        self._lock = threading.Lock()

    def clear(self):
        """Revoke continuations after a failed freshness or delivery check."""
        with self._lock:
            self._continuations.clear()

    def page(self, *, capsule_index, cursor=None, max_bytes=8192):
        with self._lock:
            try:
                return self._page(capsule_index=capsule_index, cursor=cursor, max_bytes=max_bytes)
            except Exception:
                self._continuations.clear()
                raise

    def _page(self, *, capsule_index, cursor, max_bytes):
        need = protocol._need
        need(type(capsule_index) is int and 0 <= capsule_index < len(self._capsules),
            "capsule index is outside the selected baseline population")
        need(type(max_bytes) is int and 1024 <= max_bytes <= protocol.MAX_FRAME_BYTES,
            "invalid complete semantic evidence page bound")
        raw = self._capsules[capsule_index]
        artifact_sha256 = hashlib.sha256(raw).hexdigest()
        request_sha256 = hashlib.sha256(protocol.canonical_json(dict(
            binding=protocol.strict_json_loads(self._binding_bytes),
            capsule_index=capsule_index, artifact_sha256=artifact_sha256,
            max_bytes=max_bytes))).hexdigest()
        offset = 0
        if cursor is not None:
            protocol.validate_cursor(cursor)
            need(cursor["request_sha256"] == request_sha256
                and 0 < cursor["offset"] < len(raw)
                and self._continuations.get(cursor["token"]) == protocol.canonical_json(cursor),
                "semantic evidence continuation is foreign, forged or consumed")
            offset = cursor["offset"]
        token = secrets.token_hex(16)

        def render(size):
            end = offset + size
            part = raw[offset:end]
            try:
                content, encoding = part.decode("utf-8"), "utf-8"
            except UnicodeDecodeError:
                content, encoding = base64.b64encode(part).decode("ascii"), "base64"
            successor = None if end == len(raw) else dict(schema=protocol.CURSOR_SCHEMA,
                request_sha256=request_sha256, offset=end, token=token)
            return dict(schema=protocol.PAGE_SCHEMA,
                binding=protocol.strict_json_loads(self._binding_bytes),
                artifact="selected-capsules", capsule_index=capsule_index,
                artifact_sha256=artifact_sha256, artifact_bytes=len(raw),
                source_state="baseline", canonical_source_current=True, workspace_source_current=True,
                read_only=True, **{key: False for key in protocol.AUTHORITY_FLAGS},
                source_obligations_discharged=0, offset=offset, end_offset=end,
                encoding=encoding, content=content, slice_sha256=hashlib.sha256(part).hexdigest(),
                exhausted=successor is None, next_cursor=successor)

        low, high = 0, min(len(raw) - offset, max_bytes)
        complete = render(high)
        if high == len(raw) - offset and len(protocol.canonical_json(complete)) <= max_bytes:
            page = complete
        else:
            while low < high:
                middle = (low + high + 1) // 2
                if len(protocol.canonical_json(render(middle))) <= max_bytes:
                    low = middle
                else:
                    high = middle - 1
            need(low > 0, "semantic evidence page cannot fit one capsule byte")
            page = render(low)
        need(len(protocol.canonical_json(page)) <= max_bytes, "semantic evidence page exceeds bound")
        successor = page["next_cursor"]
        need(successor is None or len(self._continuations) - (cursor is not None) < protocol.MAX_CONTINUATIONS,
            "semantic evidence continuation population exceeds bound")
        if cursor is not None:
            del self._continuations[cursor["token"]]
        if successor is not None:
            self._continuations[successor["token"]] = protocol.canonical_json(successor)
        return page
