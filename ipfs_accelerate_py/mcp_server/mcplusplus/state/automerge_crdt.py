"""Automerge CRDT state adapter (``AutomergeCrdtState@1``).

Mandatory ``crdt``-mode backend for MCP++ 1.0 (ADR-0004 / KD-10). This adapter
wraps the real Automerge document CRDT. It does **not** invent last-write-wins
maps, timestamp winners, or informal JSON merges.

Wire identity
-------------
- Interface: ``AutomergeCrdtState@1``
- Consistency mode: ``crdt``
- Backend label: ``automerge``

Concurrency model
-----------------
Replicas apply local writes under distinct Automerge actor IDs. Concurrent
offline edits are exchanged by document merge or the Automerge sync protocol.
After a partition heals, isolated replicas converge to the same Automerge
heads and value snapshot. Replaying a previously applied document or change
digest is a no-op (idempotent duplicates).
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Union

try:
    from automerge.core import (
        ROOT,
        Document,
        Message,
        ObjType,
        ScalarType,
        SyncState,
        extract,
    )
except ImportError as _exc:  # pragma: no cover - fail closed without Automerge
    raise ImportError(
        "AutomergeCrdtState@1 requires the 'automerge' package "
        "(real Automerge CRDT bindings; not an informal LWW shim). "
        "Install with: pip install 'automerge>=0.2.0.dev1'"
    ) from _exc


INTERFACE_ID = "AutomergeCrdtState@1"
STATE_MODE = "crdt"
BACKEND_ID = "automerge"
STATE_REF_SCHEMA = "mcp++/state/state-ref@1"

# Root Automerge map key holding the user-visible key/value map.
_KV_ROOT = "kv"


class AutomergeCrdtError(Exception):
    """Base error for the Automerge CRDT state adapter."""


class AutomergeDependencyError(AutomergeCrdtError):
    """Raised when Automerge cannot satisfy an operation."""


class AutomergeValueError(AutomergeCrdtError, ValueError):
    """Invalid key, value, or Automerge payload."""


@dataclass(frozen=True, slots=True)
class ApplyResult:
    """Outcome of assimilating remote Automerge material."""

    applied: int
    skipped_duplicates: int
    heads: tuple[bytes, ...]


def _normalize_actor_id(actor_id: Optional[Union[bytes, bytearray, str]]) -> bytes:
    if actor_id is None:
        return secrets.token_bytes(16)
    if isinstance(actor_id, str):
        raw = actor_id.encode("utf-8")
        if len(raw) == 16:
            return raw
        # Stable 16-byte actor derived from the string label.
        return hashlib.sha256(raw).digest()[:16]
    if isinstance(actor_id, (bytes, bytearray)):
        raw = bytes(actor_id)
        if len(raw) == 16:
            return raw
        if not raw:
            raise AutomergeValueError("actor_id must not be empty")
        return hashlib.sha256(raw).digest()[:16]
    raise AutomergeValueError("actor_id must be bytes, str, or None")


def _digest(blob: bytes) -> bytes:
    return hashlib.sha256(blob).digest()


def _require_key(key: object) -> str:
    if not isinstance(key, str) or not key:
        raise AutomergeValueError("key must be a non-empty string")
    if key.startswith("__"):
        raise AutomergeValueError("key must not use reserved __ prefix")
    return key


def _encode_scalar(tx: Any, obj_id: bytes, prop: Union[str, int], value: Any) -> None:
    if value is None:
        tx.put(obj_id, prop, ScalarType.Null, None)
    elif isinstance(value, bool):
        tx.put(obj_id, prop, ScalarType.Boolean, value)
    elif isinstance(value, int) and not isinstance(value, bool):
        tx.put(obj_id, prop, ScalarType.Int, value)
    elif isinstance(value, float):
        tx.put(obj_id, prop, ScalarType.F64, value)
    elif isinstance(value, str):
        # Immutable scalar strings (not collaborative Text) for map values.
        tx.put(obj_id, prop, ScalarType.Str, value)
    elif isinstance(value, (bytes, bytearray)):
        tx.put(obj_id, prop, ScalarType.Bytes, bytes(value))
    else:
        raise AutomergeValueError(
            f"unsupported scalar type {type(value).__name__}; "
            "use str/int/float/bool/bytes/None or nested dict/list"
        )


def _insert_list_value(tx: Any, list_id: bytes, index: int, value: Any) -> None:
    if isinstance(value, Mapping):
        child = tx.insert_object(list_id, index, ObjType.Map)
        for nested_key, nested_val in value.items():
            if not isinstance(nested_key, str):
                raise AutomergeValueError("map keys must be strings")
            _write_value(tx, child, nested_key, nested_val)
        return
    if isinstance(value, list):
        child = tx.insert_object(list_id, index, ObjType.List)
        for idx, item in enumerate(value):
            _insert_list_value(tx, child, idx, item)
        return
    if isinstance(value, tuple):
        _insert_list_value(tx, list_id, index, list(value))
        return
    if value is None:
        tx.insert(list_id, index, ScalarType.Null, None)
    elif isinstance(value, bool):
        tx.insert(list_id, index, ScalarType.Boolean, value)
    elif isinstance(value, int) and not isinstance(value, bool):
        tx.insert(list_id, index, ScalarType.Int, value)
    elif isinstance(value, float):
        tx.insert(list_id, index, ScalarType.F64, value)
    elif isinstance(value, str):
        tx.insert(list_id, index, ScalarType.Str, value)
    elif isinstance(value, (bytes, bytearray)):
        tx.insert(list_id, index, ScalarType.Bytes, bytes(value))
    else:
        raise AutomergeValueError(
            f"unsupported list element type {type(value).__name__}"
        )


def _write_value(tx: Any, obj_id: bytes, prop: Union[str, int], value: Any) -> None:
    """Write ``value`` at ``prop`` under Automerge object ``obj_id``."""
    if isinstance(value, Mapping):
        child = tx.put_object(obj_id, prop, ObjType.Map)
        for nested_key, nested_val in value.items():
            if not isinstance(nested_key, str):
                raise AutomergeValueError("map keys must be strings")
            _write_value(tx, child, nested_key, nested_val)
        return
    if isinstance(value, list):
        child = tx.put_object(obj_id, prop, ObjType.List)
        for idx, item in enumerate(value):
            # put_object creates empty list; fill by insert for correct CRDT list ops.
            _insert_list_value(tx, child, idx, item)
        return
    if isinstance(value, tuple):
        _write_value(tx, obj_id, prop, list(value))
        return
    _encode_scalar(tx, obj_id, prop, value)


class AutomergeCrdtState:
    """Multi-writer Automerge document adapter for ``StateRef`` mode ``crdt``.

    Each instance is one replica. Concurrent offline writes on different
    replicas are merged with Automerge document merge / sync — not by comparing
    wall-clock timestamps.
    """

    interface_id = INTERFACE_ID
    mode = STATE_MODE
    backend = BACKEND_ID

    def __init__(
        self,
        state_id: str,
        *,
        actor_id: Optional[Union[bytes, bytearray, str]] = None,
        document: Optional[Document] = None,
    ) -> None:
        if not isinstance(state_id, str) or not state_id:
            raise AutomergeValueError("state_id must be a non-empty string")
        self._state_id = state_id
        self._actor = _normalize_actor_id(actor_id)
        if document is None:
            self._doc = Document(self._actor)
            self._ensure_kv_root()
        else:
            self._doc = document
            self._doc.set_actor(self._actor)
            self._ensure_kv_root()
        # Content digests of Automerge document saves / change blobs already
        # incorporated. Used for explicit duplicate idempotency on the wire.
        self._applied_digests: set[bytes] = set()
        self._remember_local_history()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def open(
        cls,
        state_id: str,
        *,
        actor_id: Optional[Union[bytes, bytearray, str]] = None,
    ) -> "AutomergeCrdtState":
        """Create an empty replica for ``state_id``."""
        return cls(state_id, actor_id=actor_id)

    @classmethod
    def load(
        cls,
        data: bytes,
        *,
        state_id: str,
        actor_id: Optional[Union[bytes, bytearray, str]] = None,
    ) -> "AutomergeCrdtState":
        """Load a replica from an Automerge document save blob."""
        if not isinstance(data, (bytes, bytearray)) or not data:
            raise AutomergeValueError("document bytes must be non-empty")
        try:
            doc = Document.load(bytes(data))
        except Exception as exc:  # noqa: BLE001 - normalize binding errors
            raise AutomergeValueError("invalid Automerge document bytes") from exc
        return cls(state_id, actor_id=actor_id, document=doc)

    def fork(self, *, actor_id: Optional[Union[bytes, bytearray, str]] = None) -> "AutomergeCrdtState":
        """Fork this replica's document into a new actor-isolated replica."""
        forked = self._doc.fork()
        return AutomergeCrdtState(
            self._state_id,
            actor_id=actor_id,
            document=forked,
        )

    def _ensure_kv_root(self) -> None:
        existing = self._doc.get(ROOT, _KV_ROOT)
        if existing is not None:
            return
        with self._doc.transaction() as tx:
            tx.put_object(ROOT, _KV_ROOT, ObjType.Map)

    def _kv_id(self) -> bytes:
        found = self._doc.get(ROOT, _KV_ROOT)
        if found is None:
            self._ensure_kv_root()
            found = self._doc.get(ROOT, _KV_ROOT)
        assert found is not None
        return found[1]

    def _remember_local_history(self) -> None:
        for change in self._doc.get_changes([]):
            self._applied_digests.add(_digest(change.bytes))
            self._applied_digests.add(change.hash)

    # ------------------------------------------------------------------
    # Identity / inspection
    # ------------------------------------------------------------------

    @property
    def state_id(self) -> str:
        return self._state_id

    @property
    def actor_id(self) -> bytes:
        return bytes(self._doc.get_actor())

    @property
    def document(self) -> Document:
        """Underlying Automerge document (advanced use / diagnostics)."""
        return self._doc

    def heads(self) -> list[bytes]:
        """Current Automerge heads (sorted for deterministic comparison)."""
        return sorted(self._doc.get_heads())

    def heads_hex(self) -> list[str]:
        return [h.hex() for h in self.heads()]

    def save(self) -> bytes:
        """Serialize the full Automerge document (merge evidence)."""
        return self._doc.save()

    def change_hashes(self) -> list[bytes]:
        """All Automerge change hashes currently in this replica."""
        return [c.hash for c in self._doc.get_changes([])]

    def change_evidence(self) -> dict[str, Any]:
        """Structured merge evidence for receipts / StateRef parents."""
        changes = self._doc.get_changes([])
        return {
            "backend": BACKEND_ID,
            "interface": INTERFACE_ID,
            "mode": STATE_MODE,
            "state_id": self._state_id,
            "actor_id": self.actor_id.hex(),
            "heads": self.heads_hex(),
            "change_hashes": [c.hash.hex() for c in changes],
            "change_count": len(changes),
            "document_sha256": hashlib.sha256(self.save()).hexdigest(),
        }

    def state_ref(self, **extra: Any) -> dict[str, Any]:
        """Build a ``StateRef@1``-shaped handle for this replica.

        ``root_cid`` is left for callers that mint content-addressed roots;
        Automerge heads are carried under ``clocks.automerge_heads`` and
        ``metadata.automerge`` so merge evidence is never a bare timestamp.
        """
        ref: dict[str, Any] = {
            "schema": STATE_REF_SCHEMA,
            "id": self._state_id,
            "mode": STATE_MODE,
            "provider": BACKEND_ID,
            "version": len(self._doc.get_changes([])),
            "clocks": {
                "automerge_heads": self.heads_hex(),
            },
            "metadata": {
                "automerge": self.change_evidence(),
                "interface": INTERFACE_ID,
            },
        }
        for key, value in extra.items():
            if key in {"schema", "id", "mode"}:
                raise AutomergeValueError(f"cannot override StateRef field {key!r}")
            ref[key] = value
        return ref

    # ------------------------------------------------------------------
    # Local reads / writes
    # ------------------------------------------------------------------

    def put(self, key: str, value: Any) -> bytes:
        """Write ``key`` → ``value`` as a local Automerge change.

        Returns the hash of the local change produced by this write (empty
        bytes if Automerge coalesced to no new change).
        """
        key = _require_key(key)
        kv = self._kv_id()
        with self._doc.transaction() as tx:
            _write_value(tx, kv, key, value)
        change = self._doc.get_last_local_change()
        if change is None:
            return b""
        self._applied_digests.add(_digest(change.bytes))
        self._applied_digests.add(change.hash)
        return change.hash

    def get(self, key: str, default: Any = None) -> Any:
        """Return the current value for ``key``, or ``default`` if absent."""
        key = _require_key(key)
        kv = self._kv_id()
        found = self._doc.get(kv, key)
        if found is None:
            return default
        value, obj_id = found
        if isinstance(value, ObjType):
            return extract(self._doc, obj_id)
        return value[1]

    def delete(self, key: str) -> bool:
        """Delete ``key`` if present. Returns whether a value was removed."""
        key = _require_key(key)
        kv = self._kv_id()
        if self._doc.get(kv, key) is None:
            return False
        with self._doc.transaction() as tx:
            tx.delete(kv, key)
        change = self._doc.get_last_local_change()
        if change is not None:
            self._applied_digests.add(_digest(change.bytes))
            self._applied_digests.add(change.hash)
        return True

    def keys(self) -> list[str]:
        return list(self._doc.keys(self._kv_id()))

    def snapshot(self) -> dict[str, Any]:
        """Materialize the user key/value map as plain Python data."""
        raw = extract(self._doc, self._kv_id())
        if not isinstance(raw, dict):
            return {}
        return raw

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str) or not key:
            return False
        return self._doc.get(self._kv_id(), key) is not None

    def __getitem__(self, key: str) -> Any:
        if key not in self:
            raise KeyError(key)
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.put(key, value)

    def __delitem__(self, key: str) -> None:
        if not self.delete(key):
            raise KeyError(key)

    # ------------------------------------------------------------------
    # Merge / sync (partition heal)
    # ------------------------------------------------------------------

    def merge(self, other: Union["AutomergeCrdtState", Document, bytes]) -> list[bytes]:
        """Merge another Automerge document into this replica.

        Accepts a peer adapter, a raw ``Document``, or ``Document.save()``
        bytes. Returns the list of Automerge heads introduced by the merge
        (empty when the peer contributed nothing new). Re-merging the same
        peer document is idempotent.
        """
        peer_doc, blob = self._coerce_document(other)
        digest = _digest(blob) if blob is not None else None
        if digest is not None and digest in self._applied_digests:
            # Still merge: Automerge itself is idempotent; skip only digest bookkeeping.
            pass
        new_heads = list(self._doc.merge(peer_doc))
        self._remember_local_history()
        if digest is not None:
            self._applied_digests.add(digest)
        if blob is not None:
            # Mark every peer change hash present after merge.
            for change in peer_doc.get_changes([]):
                self._applied_digests.add(change.hash)
                self._applied_digests.add(_digest(change.bytes))
        return new_heads

    def _coerce_document(
        self, other: Union["AutomergeCrdtState", Document, bytes]
    ) -> tuple[Document, Optional[bytes]]:
        if isinstance(other, AutomergeCrdtState):
            if other.state_id != self._state_id:
                raise AutomergeValueError(
                    f"state_id mismatch: {other.state_id!r} != {self._state_id!r}"
                )
            blob = other.save()
            return Document.load(blob), blob
        if isinstance(other, Document):
            blob = other.save()
            return Document.load(blob), blob
        if isinstance(other, (bytes, bytearray)):
            blob = bytes(other)
            try:
                return Document.load(blob), blob
            except Exception as exc:  # noqa: BLE001
                raise AutomergeValueError("invalid Automerge document bytes") from exc
        raise AutomergeValueError("merge target must be AutomergeCrdtState, Document, or bytes")

    def export_changes(self, have_heads: Optional[Sequence[bytes]] = None) -> list[dict[str, bytes]]:
        """Export Automerge changes not covered by ``have_heads``.

        Each item is ``{"hash": <32-byte change hash>, "bytes": <change blob>}``.
        These blobs are Automerge change evidence; full causal incorporation on
        a peer uses :meth:`merge` or :meth:`sync_with` (Automerge document
        merge / sync protocol).
        """
        heads = list(have_heads) if have_heads is not None else []
        out: list[dict[str, bytes]] = []
        for change in self._doc.get_changes(heads):
            out.append({"hash": change.hash, "bytes": change.bytes})
        return out

    def apply_document(self, document_bytes: bytes) -> ApplyResult:
        """Idempotently merge an Automerge document save into this replica."""
        if not isinstance(document_bytes, (bytes, bytearray)) or not document_bytes:
            raise AutomergeValueError("document bytes must be non-empty")
        blob = bytes(document_bytes)
        digest = _digest(blob)
        heads_before = self.heads()
        seen_before = digest in self._applied_digests
        self.merge(blob)
        heads_after = self.heads()
        if heads_after != heads_before:
            return ApplyResult(
                applied=1,
                skipped_duplicates=0,
                heads=tuple(heads_after),
            )
        return ApplyResult(
            applied=0,
            skipped_duplicates=1 if seen_before else 0,
            heads=tuple(heads_after),
        )

    def apply_changes(
        self,
        changes: Sequence[Union[bytes, Mapping[str, bytes], "AutomergeCrdtState"]],
    ) -> ApplyResult:
        """Idempotently assimilate remote Automerge material.

        Accepted elements:
        - full Automerge document save bytes
        - mappings with ``bytes`` (change or document blob) and optional ``hash``
        - peer :class:`AutomergeCrdtState` instances (merged by document save)

        Duplicate digests / change hashes already present in history are
        skipped. Individual Automerge *change* blobs that are not full document
        saves are recognized when already in history; new change-only blobs
        require incorporation via a peer document merge or :meth:`sync_with`
        (the Python Automerge binding exposes merge/sync, not ``apply_changes``).
        """
        applied = 0
        skipped = 0
        for item in changes:
            if isinstance(item, AutomergeCrdtState):
                result = self.apply_document(item.save())
                applied += result.applied
                skipped += result.skipped_duplicates
                continue
            blob: bytes
            declared_hash: Optional[bytes] = None
            if isinstance(item, Mapping):
                raw = item.get("bytes")
                if not isinstance(raw, (bytes, bytearray)):
                    raise AutomergeValueError("change mapping requires bytes")
                blob = bytes(raw)
                h = item.get("hash")
                if isinstance(h, (bytes, bytearray)):
                    declared_hash = bytes(h)
            elif isinstance(item, (bytes, bytearray)):
                blob = bytes(item)
            else:
                raise AutomergeValueError("unsupported change element type")

            digest = _digest(blob)
            known_change_hashes = {c.hash for c in self._doc.get_changes([])}
            known_change_bytes = {c.bytes for c in self._doc.get_changes([])}

            if (
                digest in self._applied_digests
                or (declared_hash is not None and declared_hash in self._applied_digests)
                or (declared_hash is not None and declared_hash in known_change_hashes)
                or blob in known_change_bytes
            ):
                self._applied_digests.add(digest)
                if declared_hash is not None:
                    self._applied_digests.add(declared_hash)
                skipped += 1
                continue

            # Prefer full-document merge when the blob is a document save.
            try:
                Document.load(blob)
            except Exception:
                raise AutomergeValueError(
                    "change blob is not an Automerge document save and is not "
                    "already present in history; exchange peer documents via "
                    "merge/sync_with for new Automerge ops"
                )
            result = self.apply_document(blob)
            applied += result.applied
            skipped += result.skipped_duplicates
            if declared_hash is not None:
                self._applied_digests.add(declared_hash)

        return ApplyResult(
            applied=applied,
            skipped_duplicates=skipped,
            heads=tuple(self.heads()),
        )

    def new_sync_state(self) -> SyncState:
        """Allocate a fresh Automerge ``SyncState`` for a peer session."""
        return SyncState()

    def generate_sync_message(self, sync_state: SyncState) -> Optional[bytes]:
        """Produce the next Automerge sync message bytes for ``sync_state``."""
        msg = self._doc.generate_sync_message(sync_state)
        if msg is None:
            return None
        return msg.encode()

    def receive_sync_message(self, sync_state: SyncState, message: bytes) -> None:
        """Receive one Automerge sync message into this replica."""
        if not isinstance(message, (bytes, bytearray)) or not message:
            raise AutomergeValueError("sync message must be non-empty bytes")
        try:
            decoded = Message.decode(bytes(message))
        except Exception as exc:  # noqa: BLE001
            raise AutomergeValueError("invalid Automerge sync message") from exc
        self._doc.receive_sync_message(sync_state, decoded)
        self._remember_local_history()

    def sync_with(self, peer: "AutomergeCrdtState", *, max_rounds: int = 64) -> int:
        """Bidirectional Automerge sync until quiescent (partition heal).

        Returns the number of sync messages exchanged. After return, both
        replicas share the same heads and value snapshot for concurrent
        commutative/associative Automerge merges.
        """
        if not isinstance(peer, AutomergeCrdtState):
            raise AutomergeValueError("peer must be an AutomergeCrdtState")
        if peer.state_id != self._state_id:
            raise AutomergeValueError(
                f"state_id mismatch: {peer.state_id!r} != {self._state_id!r}"
            )
        local_state = SyncState()
        peer_state = SyncState()
        exchanged = 0
        for _ in range(max_rounds):
            progressed = False
            msg_local = self._doc.generate_sync_message(local_state)
            if msg_local is not None:
                peer._doc.receive_sync_message(peer_state, msg_local)
                exchanged += 1
                progressed = True
            msg_peer = peer._doc.generate_sync_message(peer_state)
            if msg_peer is not None:
                self._doc.receive_sync_message(local_state, msg_peer)
                exchanged += 1
                progressed = True
            if not progressed:
                break
        else:
            raise AutomergeDependencyError(
                f"Automerge sync did not quiesce within {max_rounds} rounds"
            )
        self._remember_local_history()
        peer._remember_local_history()
        return exchanged

    def converged_with(self, peer: "AutomergeCrdtState") -> bool:
        """Return True when heads and snapshots match ``peer``."""
        if peer.state_id != self._state_id:
            return False
        return self.heads() == peer.heads() and self.snapshot() == peer.snapshot()


def open_automerge_crdt_state(
    state_id: str,
    *,
    actor_id: Optional[Union[bytes, bytearray, str]] = None,
) -> AutomergeCrdtState:
    """Factory for :class:`AutomergeCrdtState` (``AutomergeCrdtState@1``)."""
    return AutomergeCrdtState.open(state_id, actor_id=actor_id)


__all__ = [
    "INTERFACE_ID",
    "STATE_MODE",
    "BACKEND_ID",
    "STATE_REF_SCHEMA",
    "ApplyResult",
    "AutomergeCrdtError",
    "AutomergeDependencyError",
    "AutomergeValueError",
    "AutomergeCrdtState",
    "open_automerge_crdt_state",
]
