"""Dedicated candidate-context CAS for two-stage warm lookup (PTR-145).

Retains exact canonical component bytes keyed by locator CID.  Index hits are
hints only — callers must rehash retained bytes before trusting content.
This store never authorizes SKIP.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Final

from .formal_verification_contracts import canonical_json_bytes, content_identity

TEST_CANDIDATE_CONTEXT_STORE_INTERFACE: Final = "TestCandidateContextStore@1"
TEST_CANDIDATE_CONTEXT_STORE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/test-candidate-context-store@1"
)

REQUIRED_COMPONENT_KEYS: Final[tuple[str, ...]] = (
    "execution_key",
    "static_trace",
    "runtime_trace",
    "repository_forest",
    "environment",
    "policy",
    "pass_receipt",
    "test_ast",
)

_CID_SAFE_RE: Final = re.compile(r"^[a-z0-9]+$")
_DEFAULT_MAX_BLOB_BYTES: Final = 2 * 1_048_576


class CandidateContextStoreError(RuntimeError):
    """Operational failure for the candidate-context store."""


@dataclass(frozen=True, slots=True)
class CandidateContextPublishResult:
    """Result of :meth:`TestCandidateContextStore.publish`."""

    stored: bool
    may_authorize_skip: bool = False
    reason_code: str = ""
    locator_cid: str = ""
    candidate_context_cid: str = ""


@dataclass(frozen=True, slots=True)
class CandidateContextLookupResult:
    """Result of :meth:`TestCandidateContextStore.lookup`."""

    hit: bool
    may_authorize_skip: bool = False
    reason_code: str = ""
    locator_cid: str = ""
    descriptor_bytes: bytes = b""
    component_bytes: Mapping[str, bytes] = field(default_factory=dict)
    candidate_context_cid: str = ""


class TestCandidateContextStore:
    """Filesystem-backed candidate context store with rehash-on-read semantics."""

    __test__: ClassVar[bool] = False
    interface: ClassVar[str] = TEST_CANDIDATE_CONTEXT_STORE_INTERFACE

    def __init__(
        self,
        root: str | Path,
        *,
        clock: Callable[[], float] | None = None,
        max_blob_bytes: int = _DEFAULT_MAX_BLOB_BYTES,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._clock = clock or time.time
        self._max_blob_bytes = int(max_blob_bytes)
        self._lock = threading.RLock()
        (self.root / "blobs").mkdir(exist_ok=True, mode=0o700)
        (self.root / "index").mkdir(exist_ok=True, mode=0o700)

    def publish(
        self,
        candidate: Any,
        components: Mapping[str, bytes],
        *,
        locator_cid: str | None = None,
    ) -> CandidateContextPublishResult:
        """Persist descriptor + component bytes for one locator."""

        loc = str(
            locator_cid
            or getattr(candidate, "locator_cid", "")
            or ""
        ).strip()
        if not loc or not _CID_SAFE_RE.fullmatch(loc):
            return CandidateContextPublishResult(
                stored=False,
                reason_code="invalid_locator_cid",
            )
        if not isinstance(components, Mapping) or not components:
            return CandidateContextPublishResult(
                stored=False,
                reason_code="components_required",
                locator_cid=loc,
            )
        retained: dict[str, bytes] = {}
        for key, value in components.items():
            name = str(key).strip()
            if not name or not isinstance(value, (bytes, bytearray)):
                return CandidateContextPublishResult(
                    stored=False,
                    reason_code="invalid_component",
                    locator_cid=loc,
                )
            data = bytes(value)
            if not data or len(data) > self._max_blob_bytes:
                return CandidateContextPublishResult(
                    stored=False,
                    reason_code="component_size_out_of_bounds",
                    locator_cid=loc,
                )
            retained[name] = data

        try:
            if hasattr(candidate, "to_dict"):
                descriptor = dict(candidate.to_dict())
            elif isinstance(candidate, Mapping):
                descriptor = dict(candidate)
            else:
                descriptor = {
                    "locator_cid": loc,
                    "type_name": type(candidate).__name__,
                }
            descriptor_bytes = canonical_json_bytes(descriptor)
            context_cid = content_identity(descriptor)
        except Exception as exc:
            return CandidateContextPublishResult(
                stored=False,
                reason_code=f"descriptor_encode_failed:{type(exc).__name__}",
                locator_cid=loc,
            )

        with self._lock:
            try:
                blob_dir = self.root / "blobs" / loc
                blob_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                self._atomic_write(blob_dir / "descriptor.blob", descriptor_bytes)
                component_index: dict[str, str] = {}
                for name, data in retained.items():
                    cid = content_identity(
                        {
                            "schema": TEST_CANDIDATE_CONTEXT_STORE_SCHEMA + "/component",
                            "name": name,
                            "sha256": __import__("hashlib")
                            .sha256(data)
                            .hexdigest(),
                        }
                    )
                    # Retain exact bytes under a content-safe filename.
                    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:128]
                    path = blob_dir / f"{safe}.blob"
                    self._atomic_write(path, data)
                    component_index[name] = path.name
                index_payload = {
                    "schema": TEST_CANDIDATE_CONTEXT_STORE_SCHEMA,
                    "interface": TEST_CANDIDATE_CONTEXT_STORE_INTERFACE,
                    "locator_cid": loc,
                    "candidate_context_cid": context_cid,
                    "component_files": component_index,
                    "retained_at": float(self._clock()),
                    "may_authorize_skip": False,
                }
                self._atomic_write(
                    self.root / "index" / f"{loc}.json",
                    json.dumps(index_payload, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    ),
                )
            except Exception as exc:
                return CandidateContextPublishResult(
                    stored=False,
                    reason_code=f"publish_failed:{type(exc).__name__}",
                    locator_cid=loc,
                )
        return CandidateContextPublishResult(
            stored=True,
            may_authorize_skip=False,
            locator_cid=loc,
            candidate_context_cid=context_cid,
        )

    def lookup(self, locator_cid: str, *, max_candidates: int = 1) -> CandidateContextLookupResult:
        """Load retained descriptor/components for a locator CID."""

        del max_candidates  # single-candidate v1 store
        loc = str(locator_cid or "").strip()
        if not loc or not _CID_SAFE_RE.fullmatch(loc):
            return CandidateContextLookupResult(
                hit=False, reason_code="invalid_locator_cid"
            )
        index_path = self.root / "index" / f"{loc}.json"
        blob_dir = self.root / "blobs" / loc
        with self._lock:
            if not index_path.is_file() or not blob_dir.is_dir():
                return CandidateContextLookupResult(
                    hit=False, reason_code="miss", locator_cid=loc
                )
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                if not isinstance(index, dict):
                    return CandidateContextLookupResult(
                        hit=False, reason_code="corrupt_index", locator_cid=loc
                    )
                descriptor_path = blob_dir / "descriptor.blob"
                descriptor_bytes = descriptor_path.read_bytes()
                # Rehash descriptor for integrity.
                expected = str(index.get("candidate_context_cid") or "")
                actual = content_identity(json.loads(descriptor_bytes.decode("utf-8")))
                if expected and expected != actual:
                    return CandidateContextLookupResult(
                        hit=False,
                        reason_code="descriptor_rehash_mismatch",
                        locator_cid=loc,
                    )
                component_files = index.get("component_files") or {}
                component_bytes: dict[str, bytes] = {}
                if isinstance(component_files, Mapping):
                    for name, filename in component_files.items():
                        path = blob_dir / str(filename)
                        if not path.is_file():
                            continue
                        data = path.read_bytes()
                        if 0 < len(data) <= self._max_blob_bytes:
                            component_bytes[str(name)] = data
            except Exception as exc:
                return CandidateContextLookupResult(
                    hit=False,
                    reason_code=f"lookup_failed:{type(exc).__name__}",
                    locator_cid=loc,
                )
        return CandidateContextLookupResult(
            hit=True,
            may_authorize_skip=False,
            locator_cid=loc,
            descriptor_bytes=descriptor_bytes,
            component_bytes=component_bytes,
            candidate_context_cid=str(index.get("candidate_context_cid") or ""),
        )

    def _atomic_write(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = path.with_name(f".tmp.{os.getpid()}.{threading.get_ident()}.{path.name}")
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)


__all__ = (
    "REQUIRED_COMPONENT_KEYS",
    "TEST_CANDIDATE_CONTEXT_STORE_INTERFACE",
    "CandidateContextLookupResult",
    "CandidateContextPublishResult",
    "CandidateContextStoreError",
    "TestCandidateContextStore",
)
