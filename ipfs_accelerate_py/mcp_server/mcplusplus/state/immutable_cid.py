"""ImmutableCidState@1 — append-only, CID-verified immutable state provider.

Implements :class:`StateProvider` for consistency mode ``immutable`` (ADR-0004).
Values are content-addressed; identity is the CID of canonical payload bytes.
Writes never mutate an existing CID: identical replays are idempotent, and any
attempt to store different bytes under a known CID fails closed.

Storage reuses MCP++ artifact persistence patterns:

* canonical JSON via :func:`canonicalize_artifact` for mapping payloads;
* CIDv1 raw/sha2-256 via :func:`cid_for_bytes` (same helper as Kubo-compatible
  integration tests);
* optional :class:`ArtifactStore` side-index for dict payloads keyed by CID.
"""

from __future__ import annotations

import copy
import json
import threading
from typing import Any, Mapping, MutableMapping, Optional

from ipfs_accelerate_py.mcp_server.mcplusplus.artifacts import (
    ArtifactStore,
    canonicalize_artifact,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes
from ipfs_accelerate_py.mcp_server.mcplusplus.state.provider import (
    STATE_REF_SCHEMA,
    StateIntegrityError,
    StateMutationError,
    StateNotFoundError,
    StateProvider,
    StateRefError,
    StateWriteResult,
    require_portable_cid,
    validate_state_ref,
)

IMMUTABLE_CID_STATE_INTERFACE = "ImmutableCidState@1"
IMMUTABLE_MODE = "immutable"
IMMUTABLE_PROVIDER_ID = "immutable-cid"


def canonicalize_payload(payload: Mapping[str, Any] | bytes) -> bytes:
    """Return deterministic bytes for a mapping or raw byte payload."""

    if isinstance(payload, (bytes, bytearray, memoryview)):
        return bytes(payload)
    if isinstance(payload, Mapping):
        return canonicalize_artifact(dict(payload))
    raise TypeError("payload must be a mapping or bytes")


def cid_for_payload(payload: Mapping[str, Any] | bytes) -> str:
    """Compute the immutable content address for ``payload``."""

    return cid_for_bytes(canonicalize_payload(payload))


def verify_cid_bytes(cid: str, data: bytes) -> None:
    """Fail closed when ``data`` does not hash to ``cid``."""

    actual = cid_for_bytes(data)
    if actual != cid:
        raise StateIntegrityError(
            f"bytes do not match cid: expected {cid}, computed {actual}"
        )


class ImmutableCidState(StateProvider):
    """Append-only immutable state backend keyed by CID.

    Thread-safe. The authoritative store is a CID → bytes map. An optional
    :class:`ArtifactStore` receives successful dict writes for reuse by other
    MCP++ artifact tooling; that side-index is also treated as append-only.
    """

    def __init__(
        self,
        *,
        artifact_store: Optional[ArtifactStore] = None,
        blocks: Optional[MutableMapping[str, bytes]] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._blocks: MutableMapping[str, bytes] = blocks if blocks is not None else {}
        self._artifact_store = artifact_store
        # Logical StateRef heads: id -> normalized StateRef mapping.
        self._refs: dict[str, dict[str, Any]] = {}
        self._write_count = 0
        self._idempotent_hits = 0
        self._mutation_rejects = 0

    @property
    def mode(self) -> str:
        return IMMUTABLE_MODE

    @property
    def provider_id(self) -> str:
        return IMMUTABLE_PROVIDER_ID

    @property
    def interface(self) -> str:
        return IMMUTABLE_CID_STATE_INTERFACE

    @property
    def artifact_store(self) -> Optional[ArtifactStore]:
        """Optional shared artifact side-index (may be None)."""

        return self._artifact_store

    def write(
        self,
        payload: Mapping[str, Any] | bytes,
        *,
        expected_cid: Optional[str] = None,
        state_id: Optional[str] = None,
    ) -> StateWriteResult:
        """Append canonical payload bytes under their content CID.

        * New CID: stored and returned with ``created=True``.
        * Existing CID with identical bytes: idempotent ``created=False``.
        * Existing CID with different bytes: :class:`StateMutationError`.
        * ``expected_cid`` set but not equal to the content CID:
          :class:`StateIntegrityError`.
        """

        data = canonicalize_payload(payload)
        cid = cid_for_bytes(data)
        if expected_cid is not None:
            expected = require_portable_cid(expected_cid, field="expected_cid")
            if expected != cid:
                raise StateIntegrityError(
                    f"payload cid {cid} does not match expected_cid {expected}"
                )

        with self._lock:
            existing = self._blocks.get(cid)
            if existing is not None:
                if existing != data:
                    self._mutation_rejects += 1
                    raise StateMutationError(
                        f"immutable cid mutation rejected for {cid}"
                    )
                self._idempotent_hits += 1
                self._maybe_index_artifact(cid, payload, data)
                self._advance_ref_head(state_id, cid)
                return StateWriteResult(
                    cid=cid,
                    created=False,
                    byte_length=len(data),
                    mode=self.mode,
                    provider=self.provider_id,
                )

            self._blocks[cid] = data
            self._write_count += 1
            self._maybe_index_artifact(cid, payload, data)
            self._advance_ref_head(state_id, cid)
            return StateWriteResult(
                cid=cid,
                created=True,
                byte_length=len(data),
                mode=self.mode,
                provider=self.provider_id,
            )

    def put_bytes(
        self,
        data: bytes,
        *,
        expected_cid: Optional[str] = None,
        state_id: Optional[str] = None,
    ) -> StateWriteResult:
        """Append raw bytes (alias of :meth:`write` for byte payloads)."""

        return self.write(data, expected_cid=expected_cid, state_id=state_id)

    def fetch(self, cid: str) -> bytes:
        """Return stored bytes after verifying they still hash to ``cid``."""

        key = require_portable_cid(cid, field="cid")
        with self._lock:
            data = self._blocks.get(key)
            if data is None:
                raise StateNotFoundError(key)
            # Defensive copy so callers cannot mutate the store.
            out = bytes(data)
        verify_cid_bytes(key, out)
        return out

    def fetch_json(self, cid: str) -> dict[str, Any]:
        """Fetch and parse a JSON object, re-verifying the CID over canonical form."""

        data = self.fetch(cid)
        try:
            value = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StateIntegrityError(f"{cid} is not valid JSON") from exc
        if not isinstance(value, dict):
            raise StateIntegrityError(f"{cid} is not a JSON object")
        # Re-canonicalize so alternate key ordering cannot pass verification.
        canonical = canonicalize_artifact(value)
        verify_cid_bytes(cid, canonical)
        if canonical != data:
            raise StateIntegrityError(f"{cid} is not canonical JSON")
        return copy.deepcopy(value)

    def has(self, cid: str) -> bool:
        key = str(cid or "").strip()
        if not key:
            return False
        with self._lock:
            return key in self._blocks

    def bind_ref(self, state_ref: Mapping[str, Any]) -> dict[str, Any]:
        """Validate a StateRef@1 with mode ``immutable`` and track its head.

        When ``root_cid`` is present it must already exist and verify. Binding
        never rewrites block bytes.
        """

        normalized = validate_state_ref(state_ref, require_mode=IMMUTABLE_MODE)
        root_cid = normalized.get("root_cid")
        if root_cid is not None:
            # Ensures fetch-path CID verification for the declared head.
            self.fetch(str(root_cid))

        bound = dict(normalized)
        bound.setdefault("schema", STATE_REF_SCHEMA)
        bound["mode"] = IMMUTABLE_MODE
        bound["provider"] = self.provider_id
        bound.setdefault("authority", {"kind": "none"})
        bound.setdefault("parents", list(normalized.get("parents") or []))

        with self._lock:
            self._refs[bound["id"]] = copy.deepcopy(bound)
        return copy.deepcopy(bound)

    def get_ref(self, state_id: str) -> Optional[dict[str, Any]]:
        key = str(state_id or "").strip()
        if not key:
            return None
        with self._lock:
            ref = self._refs.get(key)
            return copy.deepcopy(ref) if ref is not None else None

    def open_ref(self, state_ref: Mapping[str, Any]) -> dict[str, Any]:
        """Alias of :meth:`bind_ref` for callers that prefer open/read naming."""

        return self.bind_ref(state_ref)

    def publish(
        self,
        state_id: str,
        payload: Mapping[str, Any] | bytes,
        *,
        expected_cid: Optional[str] = None,
        parents: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Write a new immutable value and advance the logical StateRef head.

        Previous CIDs remain stored and immutable. The returned StateRef points
        at the newly minted ``root_cid``.
        """

        sid = str(state_id or "").strip()
        if not sid:
            raise StateRefError("state_id must be a non-empty string")

        result = self.write(payload, expected_cid=expected_cid, state_id=None)
        parent_list: list[str] = []
        with self._lock:
            prior = self._refs.get(sid)
            if parents is not None:
                parent_list = [require_portable_cid(p, field="parents") for p in parents]
            elif prior is not None and prior.get("root_cid"):
                parent_list = [str(prior["root_cid"])]

            ref = {
                "schema": STATE_REF_SCHEMA,
                "id": sid,
                "mode": IMMUTABLE_MODE,
                "root_cid": result.cid,
                "authority": {"kind": "none"},
                "provider": self.provider_id,
                "version": int((prior or {}).get("version") or 0) + (1 if result.created else 0),
                "parents": parent_list,
            }
            if prior and prior.get("schema_cid"):
                ref["schema_cid"] = prior["schema_cid"]
            self._refs[sid] = copy.deepcopy(ref)
            return copy.deepcopy(ref)

    def mutate_existing(
        self,
        cid: str,
        payload: Mapping[str, Any] | bytes,
    ) -> StateWriteResult:
        """Explicitly attempt to replace bytes under ``cid``.

        Always fails closed when content differs from the stored (or expected)
        identity. Provided so callers and tests exercise mutation rejection
        without going through accidental same-CID paths.
        """

        key = require_portable_cid(cid, field="cid")
        data = canonicalize_payload(payload)
        computed = cid_for_bytes(data)
        if computed != key:
            # Content does not address to the target CID — integrity failure
            # before any store mutation is considered.
            raise StateIntegrityError(
                f"cannot place payload (cid={computed}) under foreign cid {key}"
            )
        # Content hashes to key: normal write path (idempotent or collision).
        return self.write(data, expected_cid=key)

    def stats(self) -> MutableMapping[str, Any]:
        with self._lock:
            return {
                "interface": self.interface,
                "provider": self.provider_id,
                "mode": self.mode,
                "block_count": len(self._blocks),
                "ref_count": len(self._refs),
                "write_count": self._write_count,
                "idempotent_hits": self._idempotent_hits,
                "mutation_rejects": self._mutation_rejects,
                "artifact_store_attached": self._artifact_store is not None,
            }

    def export_blocks(self) -> dict[str, bytes]:
        """Return a defensive copy of all stored blocks keyed by CID."""

        with self._lock:
            return {cid: bytes(data) for cid, data in self._blocks.items()}

    def _maybe_index_artifact(
        self,
        cid: str,
        payload: Mapping[str, Any] | bytes,
        data: bytes,
    ) -> None:
        """Append-only side-index into ArtifactStore for mapping payloads."""

        if self._artifact_store is None or not isinstance(payload, Mapping):
            return
        existing = self._artifact_store.get(cid)
        body = copy.deepcopy(dict(payload))
        if existing is not None:
            if existing != body:
                # Side-index must not diverge from immutable semantics.
                self._mutation_rejects += 1
                raise StateMutationError(
                    f"immutable cid mutation rejected for artifact store {cid}"
                )
            return
        self._artifact_store.put(cid, body)

    def _advance_ref_head(self, state_id: Optional[str], cid: str) -> None:
        if not state_id:
            return
        sid = str(state_id).strip()
        if not sid:
            return
        prior = self._refs.get(sid)
        parents = [str(prior["root_cid"])] if prior and prior.get("root_cid") else []
        version = int((prior or {}).get("version") or 0)
        if prior is None or prior.get("root_cid") != cid:
            version += 1
        self._refs[sid] = {
            "schema": STATE_REF_SCHEMA,
            "id": sid,
            "mode": IMMUTABLE_MODE,
            "root_cid": cid,
            "authority": {"kind": "none"},
            "provider": self.provider_id,
            "version": version,
            "parents": parents,
        }


def create_immutable_cid_state(
    *,
    artifact_store: Optional[ArtifactStore] = None,
) -> ImmutableCidState:
    """Factory for a fresh in-memory immutable CID state provider."""

    return ImmutableCidState(artifact_store=artifact_store)


__all__ = [
    "IMMUTABLE_CID_STATE_INTERFACE",
    "IMMUTABLE_MODE",
    "IMMUTABLE_PROVIDER_ID",
    "ImmutableCidState",
    "canonicalize_payload",
    "cid_for_payload",
    "create_immutable_cid_state",
    "verify_cid_bytes",
]
