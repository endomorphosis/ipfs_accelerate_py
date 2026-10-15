"""Bounded terminal-attempt discovery on an existing native Kit owner.

This storage adapter never opens a connection, starts a provider, or grants
semantic acceptance. The launcher supplies a native DuckDBCoordinationStore in
its own dedicated campaign/shard namespace. Each slot has at most two pointer
transitions: a pending reservation and a terminal publication. There is no
expiry, deletion, eviction or rebinding. Pending attempts require explicit
recovery rather than automatic redispatch after an uncertain interruption.

Kit verifies the complete namespace transition history on every root read.
Consequently this adapter is bounded by 2 * max_entries transitions only when
the trusted owner writes this dedicated namespace exclusively through this
adapter. It cannot bound hostile history introduced through direct owner CAS.
It is a local owner integration, not a remote Quack qualification.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .contracts import RootRef, opaque_cid_validation_without_mirroring
from .durable_state import _operation_id, verify_durable_root_cas_result
from .wire import canonicalize_artifact

INDEX_SCHEMA = "ipfs-accelerate.terminal-attempt-index@1"
RESERVATION_SCHEMA = "ipfs-accelerate.terminal-attempt-reservation@1"
TICKET_SCHEMA = "ipfs-accelerate.terminal-attempt-ticket@1"
PREFLIGHT_SCHEMA = "ipfs-accelerate.terminal-attempt-preflight@1"
RESOLVE_SCHEMA = "ipfs-accelerate.terminal-attempt-resolve@1"
PUBLICATION_SCHEMA = "ipfs-accelerate.terminal-attempt-publication@1"
ATTEMPT_SCHEMA = "ipfs-accelerate.semantic-harness-attempt@1"
NAMESPACE_PREFIX = "terminal-attempt-index/"


class TerminalAttemptIndexError(RuntimeError):
    reason_code = "attempt_index_unavailable"

    def __init__(self, message: str, *, envelope_cid: str | None = None):
        super().__init__(message)
        self.envelope_cid = envelope_cid


class TerminalAttemptIndexIntegrityError(TerminalAttemptIndexError):
    reason_code = "attempt_index_integrity_failed"


class TerminalAttemptIndexCapacityError(TerminalAttemptIndexError):
    reason_code = "attempt_index_capacity_exhausted"


class TerminalAttemptIndexConflictError(TerminalAttemptIndexError):
    reason_code = "attempt_context_changed"


class TerminalAttemptIndexPendingError(TerminalAttemptIndexError):
    reason_code = "attempt_recovery_required"

    def __init__(self, identity: str, replay_context_cid: str, pending_cid: str):
        super().__init__("pending attempt requires explicit recovery; redispatch refused")
        self.attempt_identity = identity
        self.replay_context_cid = replay_context_cid
        self.pending_cid = pending_cid


def _identity(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > 512
        or len(value.encode()) > 512
        or any(char in value for char in "\x00\r\n")
    ):
        raise TerminalAttemptIndexIntegrityError(f"invalid {name}")
    return value


def _bounded_object(
    value: Any, limit: int, *, canonicalize=canonicalize_artifact
) -> tuple[dict, bytes]:
    if type(value) is not dict:
        raise TerminalAttemptIndexIntegrityError("artifact must be a plain JSON object")
    stack, count, byte_cost = [(value, 0)], 0, 0

    def charge(cost: int) -> None:
        nonlocal byte_cost
        byte_cost += cost
        if byte_cost > limit:
            raise TerminalAttemptIndexCapacityError("artifact exceeds aggregate byte bound")

    def charge_string(text: str) -> None:
        # Count native ensure_ascii=False JSON bytes without constructing an
        # escaped string or UTF-8 copy. Wire ASCII escaping is never shorter.
        # Every occurrence is charged, including aliases and repeated values.
        if len(text) + 2 > limit - byte_cost:
            raise TerminalAttemptIndexCapacityError("artifact exceeds aggregate byte bound")
        charge(2)
        for char in text:
            code = ord(char)
            if char in '"\\\b\f\n\r\t':
                charge(2)
            elif code < 0x20:
                charge(6)
            elif code < 0x80:
                charge(1)
            elif code < 0x800:
                charge(2)
            elif 0xD800 <= code <= 0xDFFF:
                raise TerminalAttemptIndexIntegrityError("artifact contains invalid UTF-8 text")
            elif code < 0x10000:
                charge(3)
            else:
                charge(4)

    while stack:
        child, depth = stack.pop()
        count += 1
        if depth > 32 or count > 50_000:
            raise TerminalAttemptIndexCapacityError("artifact exceeds structure bound")
        if type(child) is dict:
            if count + len(stack) + len(child) > 50_000:
                raise TerminalAttemptIndexCapacityError("artifact exceeds structure bound")
            charge(2 + len(child) + max(0, len(child) - 1))
            for key, item in child.items():
                if type(key) is not str:
                    raise TerminalAttemptIndexIntegrityError("invalid artifact key")
                charge_string(key)
                stack.append((item, depth + 1))
        elif type(child) is list:
            if count + len(stack) + len(child) > 50_000:
                raise TerminalAttemptIndexCapacityError("artifact exceeds structure bound")
            charge(2 + max(0, len(child) - 1))
            stack.extend((item, depth + 1) for item in child)
        elif type(child) is str:
            charge_string(child)
        elif type(child) is int:
            # floor((bits-1)*log10(2))+1 is a decimal-digit lower bound.
            # This rational is slightly below log10(2), so valid JSON within
            # the byte limit cannot be rejected. No huge decimal conversion
            # is attempted before its guaranteed cost fits the budget.
            bits = child.bit_length()
            digits = 1 if bits == 0 else ((bits - 1) * 30_102 // 100_000) + 1
            charge(digits + (child < 0))
        elif type(child) is float:
            if not math.isfinite(child):
                raise TerminalAttemptIndexIntegrityError("artifact contains a nonfinite number")
            charge(len(repr(child)))
        elif type(child) is bool:
            charge(4 if child else 5)
        elif child is None:
            charge(4)
        else:
            raise TerminalAttemptIndexIntegrityError("artifact contains a non-JSON value")
    try:
        raw = canonicalize(value)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise TerminalAttemptIndexIntegrityError("invalid canonical artifact") from exc
    if len(raw) > limit:
        raise TerminalAttemptIndexCapacityError("artifact exceeds byte bound")
    return json.loads(raw), raw


class TerminalAttemptIndex:
    """Insert-only bounded shard with fresh reservation tickets.

    Use preflight and reserve before worktree/provider effects. A returned
    ticket authorizes only this adapter's storage transition, not execution.
    Persisting a terminal envelope precedes index CAS; publication failures
    expose its immutable CID for explicit recovery. No automatic retry of an
    interrupted model invocation is authorized by a cache miss or pending row.
    """

    def __init__(
        self,
        store: Any,
        *,
        namespace: str,
        max_entries: int = 64,
        max_manifest_bytes: int = 65_536,
        max_record_bytes: int = 262_144,
        max_cas_attempts: int = 2,
    ):
        self.namespace = _identity(namespace, "namespace")
        if not self.namespace.startswith(NAMESPACE_PREFIX) or self.namespace == NAMESPACE_PREFIX:
            raise TerminalAttemptIndexIntegrityError(
                "dedicated terminal-attempt-index namespace required"
            )
        for value, maximum, name in (
            (max_entries, 256, "max_entries"),
            (max_manifest_bytes, 65_536, "max_manifest_bytes"),
            (max_record_bytes, 262_144, "max_record_bytes"),
            (max_cas_attempts, 3, "max_cas_attempts"),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise TerminalAttemptIndexIntegrityError(f"invalid {name}")
        # Dynamic source-forest loading gives native Kit classes private module
        # names. Verify their actual native source/contract rather than a bare
        # import's isinstance test which would reject the admitted class.
        native_type = type(store)
        module = sys.modules.get(native_type.__module__)
        try:
            filename = inspect.getsourcefile(native_type)
        except TypeError as exc:
            raise TerminalAttemptIndexIntegrityError("native Kit source is unavailable") from exc
        required = (
            "put",
            "get",
            "get_bytes",
            "has",
            "current_state_root",
            "compare_and_swap_state_root",
        )
        if (
            native_type.__name__ != "DuckDBCoordinationStore"
            or module is None
            or getattr(module, "DuckDBCoordinationStore", None) is not native_type
            or not filename
            or Path(filename).name != "duckdb_coordination_storage.py"
            or not all(callable(getattr(store, name, None)) for name in required)
            or getattr(store, "namespace", None) != self.namespace
            or not isinstance(getattr(store, "_identity", None), Mapping)
            or not hasattr(store, "_lock")
            or not callable(getattr(module, "cid_for_bytes", None))
            or not callable(getattr(module, "validate_transport_cid", None))
            or not callable(getattr(module, "_canonical_json", None))
        ):
            raise TerminalAttemptIndexIntegrityError(
                "an existing launcher-bound native Kit store is required"
            )
        self.store = store
        self._kit_module = module
        self.max_entries, self.max_manifest_bytes = max_entries, max_manifest_bytes
        self.max_record_bytes, self.max_cas_attempts = max_record_bytes, max_cas_attempts
        self._source = Path(filename).resolve(strict=True)
        self._source_sha256 = hashlib.sha256(self._source.read_bytes()).hexdigest()
        try:
            leaf = Path(inspect.getsourcefile(module._canonical_json)).resolve(strict=True)
            if leaf != self._source.with_name("coordination_storage.py") or any(
                Path(inspect.getsourcefile(getattr(module, name))).resolve(strict=True) != leaf
                for name in ("cid_for_bytes", "validate_transport_cid")
            ):
                raise ValueError("native byte authority source differs")
        except (TypeError, ValueError, OSError) as exc:
            raise TerminalAttemptIndexIntegrityError(
                "native byte authority source differs"
            ) from exc
        self._leaf_source = leaf
        self._leaf_source_sha256 = hashlib.sha256(leaf.read_bytes()).hexdigest()
        self.source_binding = {
            "path": str(self._source),
            "sha256": self._source_sha256,
            "canonical_path": str(leaf),
            "canonical_sha256": self._leaf_source_sha256,
        }

    def _check_source(self) -> None:
        if any(
            hashlib.sha256(path.read_bytes()).hexdigest() != expected
            for path, expected in (
                (self._source, self._source_sha256),
                (self._leaf_source, self._leaf_source_sha256),
            )
        ):
            raise TerminalAttemptIndexError("bound native Kit source changed")

    def _bounded(self, value: Any, limit: int) -> tuple[dict, bytes]:
        self._check_source()
        return _bounded_object(value, limit, canonicalize=self._kit_module._canonical_json)

    def _cid(self, body: dict, *, codec: str = "dag-json") -> str:
        # Native Kit and harness envelopes use different explicit transport
        # codecs. Preserve the envelope's raw CID; never silently reinterpret
        # an expected raw identity as a DAG-JSON identity.
        self._check_source()
        return self._kit_module.cid_for_bytes(self._kit_module._canonical_json(body), codec=codec)

    def record_cid(self, record: dict) -> str:
        """Actual native raw transport CID, including native Unicode encoding.

        The harness wire codec escapes non-ASCII text while native Kit stores
        literal UTF-8. Indexed envelopes therefore use this bound native byte
        authority, not an incompatible recomputation through the wire codec.
        This computation does not write, publish, or accept the record.
        """
        body, _ = self._bounded(record, self.max_record_bytes)
        return self._cid(body, codec="raw")

    def _call(self, method: str, *args, **kwargs):
        try:
            self._check_source()
            return getattr(self.store, method)(*args, **kwargs)
        except TerminalAttemptIndexError:
            raise
        except Exception as exc:
            if type(exc).__name__ in {"ArtifactNotFound", "ArtifactIntegrityError"}:
                raise TerminalAttemptIndexIntegrityError(
                    "indexed native content is missing or corrupt"
                ) from exc
            raise TerminalAttemptIndexError("native owner storage operation unavailable") from exc

    def _read(self, cid: str, limit: int) -> dict:
        _identity(cid, "record CID")
        raw = self._call("get_bytes", cid)
        if type(raw) is not bytes or len(raw) > limit:
            raise TerminalAttemptIndexIntegrityError("indexed native block exceeds its bound")
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise TerminalAttemptIndexIntegrityError(
                "indexed native block is invalid JSON"
            ) from exc
        body, canonical = self._bounded(payload, limit)
        codec = self._kit_module.validate_transport_cid(cid)
        if canonical != raw or self._kit_module.cid_for_bytes(canonical, codec=codec) != cid:
            raise TerminalAttemptIndexIntegrityError("indexed native block identity differs")
        return body

    def _put(self, body: dict, limit: int, *, codec: str = "dag-json") -> str:
        body, _ = self._bounded(body, limit)
        expected = self._cid(body, codec=codec)
        receipt = self._call("put", body, expected_cid=expected, codec=codec, replicate=False)
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("cid") != expected
            or receipt.get("durable") is not True
            or receipt.get("replicated") is not False
        ):
            raise TerminalAttemptIndexIntegrityError("native immutable storage receipt differs")
        try:
            if self._read(expected, limit) != body:
                raise TerminalAttemptIndexIntegrityError(
                    "native immutable storage readback differs"
                )
        except TerminalAttemptIndexError as exc:
            exc.envelope_cid = expected
            raise
        return expected

    def _manifest(self, entries: list[dict], revision: int) -> dict:
        return {
            "schema": INDEX_SCHEMA,
            "namespace": self.namespace,
            "max_entries": self.max_entries,
            "max_manifest_bytes": self.max_manifest_bytes,
            "max_record_bytes": self.max_record_bytes,
            "revision": revision,
            "entries": entries,
        }

    def _pending(self, entry: dict) -> dict:
        pending = self._read(entry["pending_cid"], self.max_record_bytes)
        if (
            set(pending)
            != {"schema", "namespace", "attempt_identity", "replay_context_cid", "nonce"}
            or pending.get("schema") != RESERVATION_SCHEMA
            or pending.get("namespace") != self.namespace
            or any(
                pending.get(key) != entry[key] for key in ("attempt_identity", "replay_context_cid")
            )
            or type(pending.get("nonce")) is not str
            or len(pending["nonce"]) != 32
        ):
            raise TerminalAttemptIndexIntegrityError("pending reservation binding differs")
        try:
            parsed = uuid.UUID(hex=pending["nonce"])
            if parsed.hex != pending["nonce"] or parsed.version != 4:
                raise ValueError()
        except ValueError as exc:
            raise TerminalAttemptIndexIntegrityError("pending nonce is invalid") from exc
        return pending

    def _envelope(self, identity: str, body: dict, replay_context_cid: str) -> None:
        if (
            not {"schema", "attempt_identity", "outcome", "terminal", "replay_context_cid"}
            <= set(body)
            or set(body)
            - {
                "schema",
                "interface",
                "attempt_identity",
                "outcome",
                "terminal",
                "replay_context_cid",
            }
            or ("interface" in body and body["interface"] != "SemanticCompressionHarness@1")
            or body.get("schema") != ATTEMPT_SCHEMA
            or body.get("attempt_identity") != identity
            or body.get("terminal") is not True
            or body.get("replay_context_cid") != replay_context_cid
            or type(body.get("outcome")) is not dict
        ):
            raise TerminalAttemptIndexIntegrityError("terminal envelope binding differs")
        outcome = body["outcome"]
        if outcome.get("attempt_identity") != identity:
            raise TerminalAttemptIndexIntegrityError("terminal outcome identity differs")
        if type(outcome.get("result")) is not dict:
            raise TerminalAttemptIndexIntegrityError(
                "terminal result must remain a structured body"
            )
        for flag in ("simulated", "root_conflict", "human_review_required"):
            if type(outcome.get(flag)) is not bool:
                raise TerminalAttemptIndexIntegrityError("terminal outcome boolean is not typed")
        if "historical_replay" in outcome and type(outcome["historical_replay"]) is not bool:
            raise TerminalAttemptIndexIntegrityError("terminal historical replay is not typed")
        if outcome.get("historical_replay", False) is not False:
            raise TerminalAttemptIndexIntegrityError(
                "historical replay cannot be published as a new terminal"
            )

    def _snapshot(self, *, capture: tuple[str, list[dict]] | None = None) -> tuple[dict, dict]:
        if capture is not None:
            capture[1].clear()
        root = self._call("current_state_root", self.namespace)
        if (
            not isinstance(root, Mapping)
            or set(root) != {"namespace", "root_cid", "revision", "transition_cid"}
            or root.get("namespace") != self.namespace
            or type(root.get("revision")) is not int
            or not 0 <= root["revision"] <= 2 * self.max_entries
        ):
            raise TerminalAttemptIndexIntegrityError(
                "native index root exceeds campaign history or binding"
            )
        root = dict(root)
        if root["revision"] == 0:
            if root.get("root_cid") is not None or root.get("transition_cid") is not None:
                raise TerminalAttemptIndexIntegrityError("empty index root is inconsistent")
            return root, self._manifest([], 0)
        manifest = self._read(root["root_cid"], self.max_manifest_bytes)
        self._validate_manifest(manifest, root["revision"], capture=capture)
        return root, manifest

    def _validate_manifest(
        self, manifest: dict, revision: int, *, capture: tuple[str, list[dict]] | None = None
    ) -> None:
        if type(revision) is not int or not 1 <= revision <= 2 * self.max_entries:
            raise TerminalAttemptIndexIntegrityError("immutable index manifest revision differs")
        fields = {
            "schema",
            "namespace",
            "max_entries",
            "max_manifest_bytes",
            "max_record_bytes",
            "revision",
            "entries",
        }
        expected = self._manifest([], revision)
        if (
            set(manifest) != fields
            or any(
                type(manifest.get(key)) is not type(expected[key])
                or manifest.get(key) != expected[key]
                for key in fields - {"entries"}
            )
            or type(manifest["entries"]) is not list
            or not 0 < len(manifest["entries"]) <= self.max_entries
        ):
            raise TerminalAttemptIndexIntegrityError("immutable index manifest policy differs")
        identities, completed = [], 0
        for entry in manifest["entries"]:
            if (
                type(entry) is not dict
                or set(entry)
                != {
                    "attempt_identity",
                    "replay_context_cid",
                    "pending_cid",
                    "envelope_cid",
                    "state",
                }
                or entry.get("state") not in {"pending", "terminal"}
            ):
                raise TerminalAttemptIndexIntegrityError("immutable index entry is malformed")
            identity = _identity(entry["attempt_identity"], "attempt_identity")
            _identity(entry["replay_context_cid"], "replay_context_cid")
            self._pending(entry)
            if entry["state"] == "pending":
                if entry["envelope_cid"] is not None:
                    raise TerminalAttemptIndexIntegrityError(
                        "pending entry claims a terminal envelope"
                    )
            else:
                envelope = self._read(entry["envelope_cid"], self.max_record_bytes)
                self._envelope(
                    identity,
                    envelope,
                    entry["replay_context_cid"],
                )
                if capture is not None and identity == capture[0]:
                    # Retain at most this one requested child. Every other
                    # child is still authenticated, and duplicate identities
                    # still fail the complete manifest check below.
                    capture[1][:] = [envelope]
                completed += 1
                del envelope
            identities.append(identity)
        if identities != sorted(set(identities)) or revision != len(identities) + completed:
            raise TerminalAttemptIndexIntegrityError(
                "immutable index order or transition count differs"
            )

    @staticmethod
    def _entry(manifest: dict, identity: str) -> dict | None:
        if type(manifest.get("entries")) is not list or any(
            type(entry) is not dict or type(entry.get("attempt_identity")) is not str
            for entry in manifest["entries"]
        ):
            raise TerminalAttemptIndexIntegrityError("immutable index entries are malformed")
        return next(
            (entry for entry in manifest["entries"] if entry["attempt_identity"] == identity), None
        )

    @staticmethod
    def _pending_error(entry: dict) -> TerminalAttemptIndexPendingError:
        return TerminalAttemptIndexPendingError(
            entry["attempt_identity"], entry["replay_context_cid"], entry["pending_cid"]
        )

    def preflight(self, identity: str, *, replay_context_cid: str | None = None) -> dict:
        """Validate the bounded complete shard before worktree/provider effects."""
        identity = _identity(identity, "attempt_identity")
        if replay_context_cid is not None:
            _identity(replay_context_cid, "replay_context_cid")
        root, manifest = self._snapshot()
        return self._preflight_receipt(identity, replay_context_cid, root, manifest)

    def _preflight_receipt(
        self, identity: str, replay_context_cid: str | None, root: dict, manifest: dict
    ) -> dict:
        entry = self._entry(manifest, identity)
        if entry is not None:
            if replay_context_cid is not None and entry["replay_context_cid"] != replay_context_cid:
                raise TerminalAttemptIndexConflictError(
                    "attempt context changed; use a new attempt key"
                )
            if entry["state"] == "pending":
                raise self._pending_error(entry)
        elif len(manifest["entries"]) >= self.max_entries:
            raise TerminalAttemptIndexCapacityError(
                "campaign shard is full; explicit new shard required"
            )
        return {
            "schema": PREFLIGHT_SCHEMA,
            "status": "terminal" if entry else "available",
            "namespace": self.namespace,
            "attempt_identity": identity,
            "replay_context_cid": None if entry is None else entry["replay_context_cid"],
            "envelope_cid": None if entry is None else entry["envelope_cid"],
            "index_cid": root["root_cid"],
            "revision": root["revision"],
            "entries": len(manifest["entries"]),
            "remaining_capacity": self.max_entries - len(manifest["entries"]),
            "durable": True,
            "completion_authority": False,
        }

    def resolve(self, identity: str, *, replay_context_cid: str) -> dict:
        """Return receipt and selected envelope from one full authenticated read.

        Native calls retain the injected owner's own lock and fence checks.
        A captured root and its immutable children define one historical
        observation without requiring the owner's lock to be reentrant. Only
        the selected envelope is retained, with no cache or partial-validation
        path. An available observation authorizes no work; reserve still
        checks the latest root and owns the concurrent race.
        """
        identity = _identity(identity, "attempt_identity")
        context = _identity(replay_context_cid, "replay_context_cid")
        captured: list[dict] = []
        root, manifest = self._snapshot(capture=(identity, captured))
        receipt = self._preflight_receipt(identity, context, root, manifest)
        envelope = captured[0] if captured else None
        if (receipt["status"] == "terminal") != (envelope is not None):
            raise TerminalAttemptIndexIntegrityError("resolved envelope and receipt differ")
        return {"schema": RESOLVE_SCHEMA, "receipt": receipt, "envelope": envelope}

    def lookup(self, identity: str) -> dict | None:
        identity = _identity(identity, "attempt_identity")
        _, manifest = self._snapshot()
        entry = self._entry(manifest, identity)
        if entry is None:
            return None
        if entry["state"] == "pending":
            raise self._pending_error(entry)
        return self._read(entry["envelope_cid"], self.max_record_bytes)

    def _cas(self, root: dict, entries: list[dict]) -> tuple[dict, str]:
        entries = sorted(entries, key=lambda item: item["attempt_identity"])
        successor = self._manifest(entries, root["revision"] + 1)
        index_cid = self._put(successor, self.max_manifest_bytes)
        result = self._call(
            "compare_and_swap_state_root",
            self.namespace,
            expected_revision=root["revision"],
            expected_root_cid=root["root_cid"],
            new_root_cid=index_cid,
            operation_id=_operation_id(root["revision"], index_cid),
        )
        if (
            not isinstance(result, Mapping)
            or result.get("status") not in {"updated", "unchanged", "conflict"}
            or result.get("local_durable") is not True
            or result.get("replicated") is not False
        ):
            raise TerminalAttemptIndexIntegrityError("native index CAS receipt is malformed")
        if result["status"] in {"updated", "unchanged"}:
            after = result.get("after")
            if (
                not isinstance(after, Mapping)
                or after.get("namespace") != self.namespace
                or after.get("root_cid") != index_cid
                or after.get("revision") != root["revision"] + 1
            ):
                raise TerminalAttemptIndexIntegrityError("native index CAS publication differs")
            if self._read(index_cid, self.max_manifest_bytes) != successor:
                raise TerminalAttemptIndexIntegrityError(
                    "native index publication readback differs"
                )
            try:
                # Exact CID and transition checks remain active. This owner
                # already has its fenced writer; optional shape observations
                # must not activate an ambient metadata owner on this path.
                with opaque_cid_validation_without_mirroring():
                    verify_durable_root_cas_result(
                        self.namespace,
                        RootRef(root_cid=root["root_cid"], generation=root["revision"])
                        if root["revision"]
                        else None,
                        index_cid,
                        result,
                        read_artifact=lambda cid: self._read(cid, self.max_record_bytes),
                        mirror_metadata=False,
                    )
            except Exception as exc:
                raise TerminalAttemptIndexIntegrityError(
                    "native immutable CAS transition differs"
                ) from exc
        return dict(result), index_cid

    def reserve(self, identity: str, *, replay_context_cid: str) -> dict:
        identity, context = (
            _identity(identity, "attempt_identity"),
            _identity(replay_context_cid, "replay_context_cid"),
        )
        pending = {
            "schema": RESERVATION_SCHEMA,
            "namespace": self.namespace,
            "attempt_identity": identity,
            "replay_context_cid": context,
            "nonce": uuid.uuid4().hex,
        }
        for _attempt in range(self.max_cas_attempts):
            root, manifest = self._snapshot()
            entry = self._entry(manifest, identity)
            if entry is not None:
                if entry["replay_context_cid"] != context:
                    raise TerminalAttemptIndexConflictError(
                        "attempt context changed; use a new attempt key"
                    )
                if entry["state"] == "pending":
                    raise self._pending_error(entry)
                raise TerminalAttemptIndexConflictError("attempt already has a terminal record")
            if len(manifest["entries"]) >= self.max_entries:
                raise TerminalAttemptIndexCapacityError(
                    "campaign shard is full; explicit new shard required"
                )
            # Reserve only if the worst-case terminal entry (CID of the same
            # fixed encoding length) will also fit the fixed manifest budget.
            pending_cid = self._cid(pending)
            entry = {
                "attempt_identity": identity,
                "replay_context_cid": context,
                "pending_cid": pending_cid,
                "envelope_cid": None,
                "state": "pending",
            }
            worst = dict(entry, envelope_cid=pending_cid, state="terminal")
            worst_entries = [
                dict(item, envelope_cid=item["pending_cid"], state="terminal")
                for item in manifest["entries"]
            ] + [worst]
            self._bounded(
                self._manifest(worst_entries, 2 * (len(worst_entries))), self.max_manifest_bytes
            )
            self._put(pending, self.max_record_bytes)
            result, index_cid = self._cas(root, [*manifest["entries"], entry])
            if result["status"] != "conflict":
                return {
                    "schema": TICKET_SCHEMA,
                    **{
                        key: pending[key]
                        for key in ("namespace", "attempt_identity", "replay_context_cid", "nonce")
                    },
                    "pending_cid": pending_cid,
                    "index_cid": index_cid,
                    "revision": root["revision"] + 1,
                    "transition_cid": result["after"]["transition_cid"],
                }
        raise TerminalAttemptIndexConflictError("bounded reservation CAS retries exhausted")

    def publish(
        self,
        identity: str,
        record: dict,
        *,
        reservation: Mapping | None = None,
        ticket: Mapping | None = None,
    ) -> dict:
        """Complete exactly the caller's reservation; never overwrite a terminal."""
        if reservation is not None and ticket is not None:
            raise TerminalAttemptIndexIntegrityError("supply one reservation ticket")
        reservation = reservation if reservation is not None else ticket
        identity = _identity(identity, "attempt_identity")
        if not isinstance(reservation, Mapping) or set(reservation) != {
            "schema",
            "namespace",
            "attempt_identity",
            "replay_context_cid",
            "nonce",
            "pending_cid",
            "index_cid",
            "revision",
            "transition_cid",
        }:
            raise TerminalAttemptIndexIntegrityError("exact reservation ticket is required")
        if (
            reservation["schema"] != TICKET_SCHEMA
            or reservation["namespace"] != self.namespace
            or reservation["attempt_identity"] != identity
            or type(reservation["revision"]) is not int
        ):
            raise TerminalAttemptIndexIntegrityError("reservation ticket binding differs")
        context = _identity(reservation["replay_context_cid"], "replay_context_cid")
        # The ticket pins its original immutable reservation manifest, while
        # publication CAS uses the current shard head. Unrelated callers may
        # advance that head without invalidating this caller's ticket.
        origin = self._read(reservation["index_cid"], self.max_manifest_bytes)
        self._validate_manifest(origin, reservation["revision"])
        origin_entry = (
            self._entry(origin, identity) if type(origin.get("entries")) is list else None
        )
        if (
            origin.get("schema") != INDEX_SCHEMA
            or origin.get("namespace") != self.namespace
            or type(origin.get("revision")) is not int
            or origin["revision"] != reservation["revision"]
            or origin_entry is None
            or origin_entry.get("state") != "pending"
            or origin_entry.get("pending_cid") != reservation["pending_cid"]
            or origin_entry.get("replay_context_cid") != context
            or origin_entry.get("envelope_cid") is not None
        ):
            raise TerminalAttemptIndexIntegrityError("ticket reservation manifest differs")
        transition = self._read(reservation["transition_cid"], self.max_record_bytes)
        previous_revision = reservation["revision"] - 1
        previous_cid = transition.get("expected_root_cid")
        expected_transition = {
            "schema": "mcp++/coordination/state-root-transition@1",
            "namespace": self.namespace,
            "operation_id": _operation_id(previous_revision, reservation["index_cid"]),
            "expected_root_cid": previous_cid,
            "expected_revision": previous_revision,
            "new_root_cid": reservation["index_cid"],
            "new_revision": reservation["revision"],
        }
        if (
            set(transition) != set(expected_transition) | {"created_at_ms"}
            or type(transition.get("created_at_ms")) is not int
            or transition["created_at_ms"] < 0
            or any(
                type(transition.get(key)) is not type(value) or transition.get(key) != value
                for key, value in expected_transition.items()
            )
            or (previous_revision == 0 and previous_cid is not None)
        ):
            raise TerminalAttemptIndexIntegrityError("ticket reservation transition differs")
        if previous_revision:
            previous = self._read(previous_cid, self.max_manifest_bytes)
            self._validate_manifest(previous, previous_revision)
        else:
            previous = self._manifest([], 0)
        if self._entry(previous, identity) is not None or origin["entries"] != sorted(
            [*previous["entries"], origin_entry], key=lambda item: item["attempt_identity"]
        ):
            raise TerminalAttemptIndexIntegrityError(
                "ticket reservation is not an insert transition"
            )
        body, _ = self._bounded(record, self.max_record_bytes)
        self._envelope(identity, body, context)
        envelope_cid = None
        try:
            for _attempt in range(self.max_cas_attempts):
                root, manifest = self._snapshot()
                entry = self._entry(manifest, identity)
                if entry is None or entry["pending_cid"] != reservation["pending_cid"]:
                    raise TerminalAttemptIndexConflictError(
                        "reservation is absent or belongs to another caller"
                    )
                pending = self._pending(entry)
                if any(
                    reservation[key] != pending[key]
                    for key in ("namespace", "attempt_identity", "replay_context_cid", "nonce")
                ):
                    raise TerminalAttemptIndexConflictError(
                        "reservation ticket belongs to another caller"
                    )
                actual_cid = self._cid(body, codec="raw")
                if entry["state"] == "terminal":
                    if entry["envelope_cid"] != actual_cid:
                        raise TerminalAttemptIndexConflictError(
                            "terminal identity cannot be rebound"
                        )
                    return self._receipt(
                        identity,
                        context,
                        actual_cid,
                        entry["pending_cid"],
                        root["root_cid"],
                        root["revision"],
                        root["transition_cid"],
                        unchanged=True,
                    )
                envelope_cid = self._put(body, self.max_record_bytes, codec="raw")
                updated = [
                    dict(item, state="terminal", envelope_cid=envelope_cid)
                    if item is entry
                    else item
                    for item in manifest["entries"]
                ]
                result, index_cid = self._cas(root, updated)
                if result["status"] != "conflict":
                    return self._receipt(
                        identity,
                        context,
                        envelope_cid,
                        entry["pending_cid"],
                        index_cid,
                        root["revision"] + 1,
                        result.get("transition_cid"),
                        unchanged=False,
                    )
            raise TerminalAttemptIndexConflictError("bounded publication CAS retries exhausted")
        except TerminalAttemptIndexError as exc:
            if envelope_cid is not None:
                exc.envelope_cid = envelope_cid
            raise

    def _receipt(
        self,
        identity: str,
        context: str,
        cid: str,
        pending_cid: str,
        index_cid: str,
        revision: int,
        transition_cid: str | None,
        *,
        unchanged: bool,
    ) -> dict:
        return {
            "schema": PUBLICATION_SCHEMA,
            "status": "unchanged" if unchanged else "published",
            "namespace": self.namespace,
            "attempt_identity": identity,
            "replay_context_cid": context,
            "envelope_cid": cid,
            "pending_cid": pending_cid,
            "index_cid": index_cid,
            "revision": revision,
            "transition_cid": transition_cid,
            "idempotent_replay": unchanged,
            "durable": True,
            "completion_authority": False,
        }
