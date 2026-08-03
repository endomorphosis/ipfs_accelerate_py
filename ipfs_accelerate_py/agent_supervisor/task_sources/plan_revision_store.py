"""Append-only plan revision store with atomic Markdown/DuckDB apply (PDR-031).

:class:`PlanRevisionStore` is the durable authority for create/steer apply:

* re-observes authority roots before any mutation;
* journals intent (base root/revision/cursor and expected effects) before
  effects;
* appends revision, delta, supersession, record, and event history without
  rewriting accepted or claimed specifications;
* prepares Markdown and DuckDB projections, verifies exact CIDs and round-trip
  parity, then atomically commits the active pointer or restores the prior
  projection;
* recovers at every crash boundary from CAS/store state rather than process
  dictionaries; and
* quarantines irreconcilable split-brain between backends.

Task-source backends remain lossless projections.  They do not redefine plan
authority; this store owns revision ancestry and the active plan root.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Final, Iterable

from ..planning.plan_revision_contracts import (
    LifecycleState,
    PlanAuthorityRoots,
    PlanDelta,
    PlanDeltaItem,
    PlanDeltaOperation,
    PlanOrigin,
    PlanRevision,
    PlanRevisionAuthorityError,
    PlanRevisionContractError,
    PlanRevisionLifecycleError,
    PlanRevisionStaleRootError,
    PopulationKind,
    assert_delta_preserves_history,
    assert_population_history_intact,
    plan_revision_cid,
)
from ..proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)


PLAN_REVISION_STORE_INTERFACE: Final[str] = "PlanRevisionStore@1"
PLAN_REVISION_STORE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-revision-store@1"
)
PLAN_REVISION_INTENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-revision-intent@1"
)
PLAN_REVISION_APPLY_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-revision-apply-receipt@1"
)
PLAN_REVISION_ACTIVE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-revision-active@1"
)
PLAN_REVISION_EVENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-revision-event@1"
)
PLAN_REVISION_SUPERSESSION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-revision-supersession@1"
)
PLAN_REVISION_CONTINUATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-revision-continuation@1"
)
PLAN_REVISION_INDEX_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-revision-index@1"
)

MAX_INTENT_BYTES: Final[int] = 1_048_576
MAX_CAS_BYTES: Final[int] = 1_048_576
MAX_INDEX_ENTRIES: Final[int] = 65_536
MAX_EVENTS: Final[int] = 100_000
MAX_CONTINUATION_BYTES: Final[int] = 1_048_576

_CID_RE_PREFIX = "b"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PlanRevisionStoreError(RuntimeError):
    """Base error for durable plan-revision store failures."""


class PlanRevisionStoreConflictError(PlanRevisionStoreError):
    """CAS, fence, root, or expected-effects conflict during apply."""


class PlanRevisionStoreQuarantinedError(PlanRevisionStoreError):
    """Store is quarantined after split-brain or unrecoverable corruption."""


class PlanRevisionStoreIntegrityError(PlanRevisionStoreError):
    """CAS payload, index, or active pointer failed integrity checks."""


class PlanRevisionStoreStaleError(PlanRevisionStoreConflictError):
    """Observed roots/revision/cursor no longer match the expected fence."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class PlanRevisionApplyState(str, Enum):
    """Durable states of one apply transaction."""

    INTENT_JOURNALED = "intent_journaled"
    PREPARED = "prepared"
    VERIFIED = "verified"
    COMMITTED = "committed"
    RESTORED = "restored"
    QUARANTINED = "quarantined"
    BLOCKED = "blocked"
    REPLAYED = "replayed"


class PlanRevisionEventType(str, Enum):
    INTENT_JOURNALED = "intent_journaled"
    REVISION_APPENDED = "revision_appended"
    DELTA_APPENDED = "delta_appended"
    SUPERSESSION_APPENDED = "supersession_appended"
    PROJECTION_PREPARED = "projection_prepared"
    PROJECTION_VERIFIED = "projection_verified"
    PROJECTION_COMMITTED = "projection_committed"
    PROJECTION_RESTORED = "projection_restored"
    DEFERRED_ACTIVATED = "deferred_activated"
    SPLIT_BRAIN_QUARANTINED = "split_brain_quarantined"
    RECOVERED = "recovered"


class PlanRevisionBackendKind(str, Enum):
    MARKDOWN = "markdown"
    DUCKDB = "duckdb"
    BOTH = "both"
    NONE = "none"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_bytes(value: Any) -> bytes:
    try:
        return canonical_json_bytes(value)
    except Exception:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")


def _cid_for(value: Any) -> str:
    try:
        return content_identity(value)
    except Exception:
        digest = hashlib.sha256(_canonical_bytes(value)).hexdigest()
        # Stable non-multiformats fallback for pure-dict fixtures in unit tests.
        return plan_revision_cid({"sha256": digest, "payload": value})


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(dict(value)) + b"\n"
    if len(payload) > MAX_CAS_BYTES:
        raise PlanRevisionStoreIntegrityError(
            f"{path.name} exceeds the plan-revision persistence bound"
        )
    _atomic_write_bytes(path, payload)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise PlanRevisionStoreIntegrityError(f"missing store file: {path}") from exc
    except OSError as exc:
        raise PlanRevisionStoreIntegrityError(f"cannot read {path}: {exc}") from exc
    if len(raw) > MAX_CAS_BYTES:
        raise PlanRevisionStoreIntegrityError(
            f"{path.name} exceeds the plan-revision persistence bound"
        )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlanRevisionStoreIntegrityError(
            f"{path.name} is not valid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise PlanRevisionStoreIntegrityError(f"{path.name} must be a JSON object")
    return value


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        if required:
            raise PlanRevisionStoreError(f"{name} is required")
        return ""
    if not isinstance(value, str):
        raise PlanRevisionStoreError(f"{name} must be a string")
    if "\x00" in value:
        raise PlanRevisionStoreError(f"{name} must not contain NUL")
    text = value.strip()
    if required and not text:
        raise PlanRevisionStoreError(f"{name} must not be empty")
    return text


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PlanRevisionStoreError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _sequence_ids(values: Any, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(
        values, Sequence
    ):
        raise PlanRevisionStoreError(f"{name} must be a sequence of strings")
    items: list[str] = []
    seen: set[str] = set()
    for raw in values:
        item = _text(raw, name)
        if item not in seen:
            seen.add(item)
            items.append(item)
    return tuple(items)


def _decode_revision(value: Any) -> PlanRevision:
    if isinstance(value, PlanRevision):
        return value
    if isinstance(value, Mapping):
        return PlanRevision.from_dict(value)
    raise PlanRevisionStoreError("revision must be a PlanRevision or mapping")


def _decode_delta(value: Any) -> PlanDelta | None:
    if value is None:
        return None
    if isinstance(value, PlanDelta):
        return value
    if isinstance(value, Mapping):
        return PlanDelta.from_dict(value)
    raise PlanRevisionStoreError("delta must be a PlanDelta or mapping")


def _decode_roots(value: Any) -> PlanAuthorityRoots:
    if isinstance(value, PlanAuthorityRoots):
        return value
    if isinstance(value, Mapping):
        return PlanAuthorityRoots.from_dict(value)
    raise PlanRevisionStoreError("roots must be PlanAuthorityRoots or a mapping")


def _protected_cids(revision: PlanRevision) -> set[str]:
    protected: set[str] = set()
    for population in (
        revision.claimed_population,
        revision.completed_population,
    ):
        protected.update(population.member_cids)
    return protected


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanRevisionIntent:
    """One intent-journal entry written before projection effects."""

    intent_cid: str
    idempotency_key: str
    origin: PlanOrigin
    base_plan_root: str
    base_plan_revision: int
    base_event_cursor: str
    candidate_plan_root: str
    candidate_revision_cid: str
    delta_cid: str
    expected_effects: tuple[str, ...]
    observed_roots: PlanAuthorityRoots
    expected_roots: PlanAuthorityRoots
    markdown_path: str = ""
    duckdb_path: str = ""
    fencing_token: int = 1
    lease_id: str = ""
    deferred_item_keys: tuple[str, ...] = ()
    retained_task_cids: tuple[str, ...] = ()
    claimed_task_cids: tuple[str, ...] = ()
    accepted_task_cids: tuple[str, ...] = ()
    state: PlanRevisionApplyState = PlanRevisionApplyState.INTENT_JOURNALED
    created_at_ns: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLAN_REVISION_INTENT_SCHEMA,
            "intent_cid": self.intent_cid,
            "idempotency_key": self.idempotency_key,
            "origin": self.origin.value,
            "base_plan_root": self.base_plan_root,
            "base_plan_revision": self.base_plan_revision,
            "base_event_cursor": self.base_event_cursor,
            "candidate_plan_root": self.candidate_plan_root,
            "candidate_revision_cid": self.candidate_revision_cid,
            "delta_cid": self.delta_cid,
            "expected_effects": list(self.expected_effects),
            "observed_roots": self.observed_roots.to_dict(),
            "expected_roots": self.expected_roots.to_dict(),
            "markdown_path": self.markdown_path,
            "duckdb_path": self.duckdb_path,
            "fencing_token": self.fencing_token,
            "lease_id": self.lease_id,
            "deferred_item_keys": list(self.deferred_item_keys),
            "retained_task_cids": list(self.retained_task_cids),
            "claimed_task_cids": list(self.claimed_task_cids),
            "accepted_task_cids": list(self.accepted_task_cids),
            "state": self.state.value,
            "created_at_ns": self.created_at_ns,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanRevisionIntent":
        if payload.get("schema") != PLAN_REVISION_INTENT_SCHEMA:
            raise PlanRevisionStoreIntegrityError("unsupported intent schema")
        return cls(
            intent_cid=_text(payload.get("intent_cid"), "intent_cid"),
            idempotency_key=_text(
                payload.get("idempotency_key"), "idempotency_key"
            ),
            origin=PlanOrigin(str(payload.get("origin"))),
            base_plan_root=_text(
                payload.get("base_plan_root") or "",
                "base_plan_root",
                required=False,
            ),
            base_plan_revision=int(payload.get("base_plan_revision") or 0),
            base_event_cursor=_text(
                payload.get("base_event_cursor") or "",
                "base_event_cursor",
                required=False,
            ),
            candidate_plan_root=_text(
                payload.get("candidate_plan_root"), "candidate_plan_root"
            ),
            candidate_revision_cid=_text(
                payload.get("candidate_revision_cid"),
                "candidate_revision_cid",
            ),
            delta_cid=_text(
                payload.get("delta_cid") or "", "delta_cid", required=False
            ),
            expected_effects=_sequence_ids(
                payload.get("expected_effects"), "expected_effects"
            ),
            observed_roots=_decode_roots(payload.get("observed_roots")),
            expected_roots=_decode_roots(payload.get("expected_roots")),
            markdown_path=_text(
                payload.get("markdown_path") or "",
                "markdown_path",
                required=False,
            ),
            duckdb_path=_text(
                payload.get("duckdb_path") or "", "duckdb_path", required=False
            ),
            fencing_token=int(payload.get("fencing_token") or 1),
            lease_id=_text(
                payload.get("lease_id") or "", "lease_id", required=False
            ),
            deferred_item_keys=_sequence_ids(
                payload.get("deferred_item_keys"), "deferred_item_keys"
            ),
            retained_task_cids=_sequence_ids(
                payload.get("retained_task_cids"), "retained_task_cids"
            ),
            claimed_task_cids=_sequence_ids(
                payload.get("claimed_task_cids"), "claimed_task_cids"
            ),
            accepted_task_cids=_sequence_ids(
                payload.get("accepted_task_cids"), "accepted_task_cids"
            ),
            state=PlanRevisionApplyState(
                str(payload.get("state") or PlanRevisionApplyState.INTENT_JOURNALED.value)
            ),
            created_at_ns=int(payload.get("created_at_ns") or 0),
        )


@dataclass(frozen=True)
class PlanRevisionApplyReceipt:
    """Body-free receipt for one create/steer apply attempt."""

    receipt_cid: str
    intent_cid: str
    state: PlanRevisionApplyState
    revision_cid: str
    plan_root_cid: str
    delta_cid: str = ""
    markdown_projection_cid: str = ""
    duckdb_projection_cid: str = ""
    prior_active_cid: str = ""
    event_cursor: str = ""
    expected_effects: tuple[str, ...] = ()
    observed_effects: tuple[str, ...] = ()
    deferred_item_keys: tuple[str, ...] = ()
    activated_deferred_keys: tuple[str, ...] = ()
    resumed: bool = False
    quarantined: bool = False
    reason_codes: tuple[str, ...] = ()
    markdown_path: str = ""
    duckdb_path: str = ""

    @property
    def committed(self) -> bool:
        return self.state in {
            PlanRevisionApplyState.COMMITTED,
            PlanRevisionApplyState.REPLAYED,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLAN_REVISION_APPLY_RECEIPT_SCHEMA,
            "receipt_cid": self.receipt_cid,
            "intent_cid": self.intent_cid,
            "state": self.state.value,
            "revision_cid": self.revision_cid,
            "plan_root_cid": self.plan_root_cid,
            "delta_cid": self.delta_cid,
            "markdown_projection_cid": self.markdown_projection_cid,
            "duckdb_projection_cid": self.duckdb_projection_cid,
            "prior_active_cid": self.prior_active_cid,
            "event_cursor": self.event_cursor,
            "expected_effects": list(self.expected_effects),
            "observed_effects": list(self.observed_effects),
            "deferred_item_keys": list(self.deferred_item_keys),
            "activated_deferred_keys": list(self.activated_deferred_keys),
            "resumed": self.resumed,
            "quarantined": self.quarantined,
            "committed": self.committed,
            "reason_codes": list(self.reason_codes),
            "markdown_path": self.markdown_path,
            "duckdb_path": self.duckdb_path,
        }


@dataclass(frozen=True)
class PlanRevisionActiveProjection:
    """Atomically published active plan projection pointer."""

    active_cid: str
    plan_root_cid: str
    revision_cid: str
    semantic_revision: int
    event_cursor: str
    markdown_projection_cid: str = ""
    duckdb_projection_cid: str = ""
    markdown_path: str = ""
    duckdb_path: str = ""
    intent_cid: str = ""
    prior_active_cid: str = ""
    deferred_item_keys: tuple[str, ...] = ()
    quarantined: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLAN_REVISION_ACTIVE_SCHEMA,
            "active_cid": self.active_cid,
            "plan_root_cid": self.plan_root_cid,
            "revision_cid": self.revision_cid,
            "semantic_revision": self.semantic_revision,
            "event_cursor": self.event_cursor,
            "markdown_projection_cid": self.markdown_projection_cid,
            "duckdb_projection_cid": self.duckdb_projection_cid,
            "markdown_path": self.markdown_path,
            "duckdb_path": self.duckdb_path,
            "intent_cid": self.intent_cid,
            "prior_active_cid": self.prior_active_cid,
            "deferred_item_keys": list(self.deferred_item_keys),
            "quarantined": self.quarantined,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlanRevisionActiveProjection":
        if payload.get("schema") != PLAN_REVISION_ACTIVE_SCHEMA:
            raise PlanRevisionStoreIntegrityError("unsupported active projection schema")
        return cls(
            active_cid=_text(payload.get("active_cid"), "active_cid"),
            plan_root_cid=_text(payload.get("plan_root_cid"), "plan_root_cid"),
            revision_cid=_text(payload.get("revision_cid"), "revision_cid"),
            semantic_revision=int(payload.get("semantic_revision") or 0),
            event_cursor=_text(
                payload.get("event_cursor") or "", "event_cursor", required=False
            ),
            markdown_projection_cid=_text(
                payload.get("markdown_projection_cid") or "",
                "markdown_projection_cid",
                required=False,
            ),
            duckdb_projection_cid=_text(
                payload.get("duckdb_projection_cid") or "",
                "duckdb_projection_cid",
                required=False,
            ),
            markdown_path=_text(
                payload.get("markdown_path") or "",
                "markdown_path",
                required=False,
            ),
            duckdb_path=_text(
                payload.get("duckdb_path") or "", "duckdb_path", required=False
            ),
            intent_cid=_text(
                payload.get("intent_cid") or "", "intent_cid", required=False
            ),
            prior_active_cid=_text(
                payload.get("prior_active_cid") or "",
                "prior_active_cid",
                required=False,
            ),
            deferred_item_keys=_sequence_ids(
                payload.get("deferred_item_keys"), "deferred_item_keys"
            ),
            quarantined=bool(payload.get("quarantined")),
        )


@dataclass(frozen=True)
class PlanRevisionApplyRequest:
    """Inputs for one authorized create or steer apply."""

    revision: PlanRevision
    observed_roots: PlanAuthorityRoots
    idempotency_key: str
    expected_effects: Sequence[str] = ()
    delta: PlanDelta | Mapping[str, Any] | None = None
    admission: Any = None
    goal_graph: Any = None
    aliases: Mapping[str, str] | None = None
    markdown_source: Any = None
    duckdb_source: Any = None
    repository_tree_id: str = ""
    fencing_token: int = 1
    lease_id: str = ""
    base_event_cursor: str = ""
    expected_active_plan_root: str = ""
    expected_active_revision_cid: str = ""
    activate_deferred_keys: Sequence[str] = ()
    fault_injector: Callable[[str], None] | None = None
    records: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "revision", _decode_revision(self.revision))
        object.__setattr__(
            self, "observed_roots", _decode_roots(self.observed_roots)
        )
        object.__setattr__(
            self,
            "idempotency_key",
            _text(self.idempotency_key, "idempotency_key"),
        )
        object.__setattr__(
            self,
            "expected_effects",
            _sequence_ids(self.expected_effects, "expected_effects"),
        )
        object.__setattr__(self, "delta", _decode_delta(self.delta))
        object.__setattr__(
            self,
            "activate_deferred_keys",
            _sequence_ids(
                self.activate_deferred_keys, "activate_deferred_keys"
            ),
        )
        object.__setattr__(
            self,
            "repository_tree_id",
            _text(
                self.repository_tree_id or "",
                "repository_tree_id",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "lease_id",
            _text(self.lease_id or "", "lease_id", required=False),
        )
        object.__setattr__(
            self,
            "base_event_cursor",
            _text(
                self.base_event_cursor or "",
                "base_event_cursor",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "expected_active_plan_root",
            _text(
                self.expected_active_plan_root or "",
                "expected_active_plan_root",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "expected_active_revision_cid",
            _text(
                self.expected_active_revision_cid or "",
                "expected_active_revision_cid",
                required=False,
            ),
        )
        if (
            isinstance(self.fencing_token, bool)
            or not isinstance(self.fencing_token, int)
            or self.fencing_token < 1
        ):
            raise PlanRevisionStoreError("fencing_token must be a positive integer")
        if self.aliases is not None and not isinstance(self.aliases, Mapping):
            raise PlanRevisionStoreError("aliases must be a mapping when provided")
        if self.records is not None and not isinstance(self.records, Mapping):
            raise PlanRevisionStoreError("records must be a mapping when provided")


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class PlanRevisionStore:
    """Crash-safe append-only store for plan revisions and dual projections.

    Directory layout under ``root``::

        cas/<cid>                 content-addressed records
        intents/<intent_cid>.json intent journal entries
        continuations/<key>.json  durable apply continuations (not process dicts)
        events.jsonl              append-only store events
        supersessions.jsonl       append-only supersession links
        index.json                revision/delta index
        active.json               current active projection pointer
        prior_active.json         previous active projection for restore
        quarantine/               split-brain and corrupt payloads
        projection_backups/       prior Markdown/DuckDB bytes for restore
        .lock                     exclusive apply/recover lock
    """

    INTERFACE: Final[str] = PLAN_REVISION_STORE_INTERFACE

    def __init__(
        self,
        root: Path | str,
        *,
        recover: bool = True,
        clock_ns: Callable[[], int] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.cas_dir = self.root / "cas"
        self.intents_dir = self.root / "intents"
        self.continuations_dir = self.root / "continuations"
        self.quarantine_dir = self.root / "quarantine"
        self.backups_dir = self.root / "projection_backups"
        self.events_path = self.root / "events.jsonl"
        self.supersessions_path = self.root / "supersessions.jsonl"
        self.index_path = self.root / "index.json"
        self.active_path = self.root / "active.json"
        self.prior_active_path = self.root / "prior_active.json"
        self.lock_path = self.root / ".plan-revision-store.lock"
        self._clock_ns = clock_ns or time.time_ns
        self._thread_lock = threading.RLock()
        for directory in (
            self.cas_dir,
            self.intents_dir,
            self.continuations_dir,
            self.quarantine_dir,
            self.backups_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            _atomic_write_json(
                self.index_path,
                {
                    "schema": PLAN_REVISION_INDEX_SCHEMA,
                    "revisions": [],
                    "deltas": [],
                    "latest_revision_cid": "",
                    "latest_intent_cid": "",
                },
            )
        if not self.events_path.exists():
            _atomic_write_bytes(self.events_path, b"")
        if not self.supersessions_path.exists():
            _atomic_write_bytes(self.supersessions_path, b"")
        if recover:
            self.recover()

    # -- locking -------------------------------------------------------------

    def _guard(self):
        from .duckdb_state import exclusive_file_lock

        return exclusive_file_lock(self.lock_path, timeout_seconds=30.0)

    def _fault(self, injector: Callable[[str], None] | None, point: str) -> None:
        if callable(injector):
            injector(point)

    # -- CAS / continuation --------------------------------------------------

    def put_cas(self, value: Any, *, media_type: str = "application/json") -> str:
        """Persist one content-addressed record and return its CID."""

        if hasattr(value, "to_dict") and callable(value.to_dict):
            payload = value.to_dict()
        elif isinstance(value, Mapping):
            payload = dict(value)
        else:
            payload = value
        cid = _cid_for(payload)
        path = self.cas_dir / cid
        record = {
            "schema": PLAN_REVISION_STORE_SCHEMA,
            "cid": cid,
            "media_type": media_type,
            "payload": payload,
        }
        if path.exists():
            existing = _read_json(path)
            if (
                existing.get("cid") != cid
                or existing.get("payload") != record["payload"]
            ):
                raise PlanRevisionStoreIntegrityError(
                    f"CAS collision for {cid}"
                )
            return cid
        _atomic_write_json(path, record)
        return cid

    def get_cas(self, cid: str) -> dict[str, Any]:
        path = self.cas_dir / _text(cid, "cid")
        record = _read_json(path)
        if record.get("cid") != cid:
            raise PlanRevisionStoreIntegrityError("CAS cid mismatch")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise PlanRevisionStoreIntegrityError("CAS payload must be an object")
        return payload

    def put_continuation(
        self,
        idempotency_key: str,
        payload: Mapping[str, Any],
    ) -> str:
        """Persist apply continuation under a durable key (not a process dict)."""

        key = _text(idempotency_key, "idempotency_key")
        body = {
            "schema": PLAN_REVISION_CONTINUATION_SCHEMA,
            "idempotency_key": key,
            "payload": dict(payload),
            "updated_at_ns": self._clock_ns(),
        }
        cid = _cid_for(body)
        body["continuation_cid"] = cid
        path = self.continuations_dir / f"{_safe_filename(key)}.json"
        _atomic_write_json(path, body)
        self.put_cas(body)
        return cid

    def load_continuation(
        self, idempotency_key: str
    ) -> Mapping[str, Any] | None:
        """Reload continuation from the store; never from process memory."""

        key = _text(idempotency_key, "idempotency_key")
        path = self.continuations_dir / f"{_safe_filename(key)}.json"
        if not path.exists():
            return None
        record = _read_json(path)
        if record.get("schema") != PLAN_REVISION_CONTINUATION_SCHEMA:
            raise PlanRevisionStoreIntegrityError(
                "unsupported continuation schema"
            )
        if record.get("idempotency_key") != key:
            raise PlanRevisionStoreIntegrityError(
                "continuation key mismatch"
            )
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            raise PlanRevisionStoreIntegrityError(
                "continuation payload must be a mapping"
            )
        return MappingProxyType(dict(payload))

    def clear_continuation(self, idempotency_key: str) -> None:
        key = _text(idempotency_key, "idempotency_key")
        path = self.continuations_dir / f"{_safe_filename(key)}.json"
        try:
            path.unlink()
        except FileNotFoundError:
            return

    # -- indexes / events ----------------------------------------------------

    def _read_index(self) -> dict[str, Any]:
        index = _read_json(self.index_path)
        if index.get("schema") != PLAN_REVISION_INDEX_SCHEMA:
            raise PlanRevisionStoreIntegrityError("unsupported revision index schema")
        revisions = index.get("revisions")
        deltas = index.get("deltas")
        if not isinstance(revisions, list) or not isinstance(deltas, list):
            raise PlanRevisionStoreIntegrityError(
                "revision index collections must be lists"
            )
        if len(revisions) > MAX_INDEX_ENTRIES or len(deltas) > MAX_INDEX_ENTRIES:
            raise PlanRevisionStoreIntegrityError(
                "revision index exceeds its entry bound"
            )
        return index

    def _write_index(self, index: Mapping[str, Any]) -> None:
        _atomic_write_json(self.index_path, dict(index))

    def _append_jsonl(self, path: Path, record: Mapping[str, Any]) -> None:
        line = _canonical_bytes(dict(record)) + b"\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())

    def _append_event(
        self,
        event_type: PlanRevisionEventType | str,
        *,
        intent_cid: str = "",
        revision_cid: str = "",
        plan_root_cid: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> str:
        payload = {
            "schema": PLAN_REVISION_EVENT_SCHEMA,
            "event_type": (
                event_type.value
                if isinstance(event_type, PlanRevisionEventType)
                else str(event_type)
            ),
            "intent_cid": intent_cid,
            "revision_cid": revision_cid,
            "plan_root_cid": plan_root_cid,
            "body": dict(body or {}),
            "created_at_ns": self._clock_ns(),
        }
        event_cid = _cid_for(payload)
        payload["event_cid"] = event_cid
        self._append_jsonl(self.events_path, payload)
        self.put_cas(payload)
        return event_cid

    def list_events(self, *, limit: int = MAX_EVENTS) -> tuple[dict[str, Any], ...]:
        if limit < 1:
            raise PlanRevisionStoreError("limit must be positive")
        if not self.events_path.exists():
            return ()
        rows: list[dict[str, Any]] = []
        with self.events_path.open("rb") as stream:
            for raw in stream:
                if not raw.strip():
                    continue
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise PlanRevisionStoreIntegrityError(
                        "event log is corrupt"
                    ) from exc
                if isinstance(value, dict):
                    rows.append(value)
                if len(rows) > MAX_EVENTS:
                    raise PlanRevisionStoreIntegrityError(
                        "event log exceeds its bound"
                    )
        return tuple(rows[-limit:])

    def append_supersession(
        self,
        *,
        prior_cid: str,
        next_cid: str,
        kind: str,
        body: Mapping[str, Any] | None = None,
    ) -> str:
        record = {
            "schema": PLAN_REVISION_SUPERSESSION_SCHEMA,
            "prior_cid": _text(prior_cid, "prior_cid"),
            "next_cid": _text(next_cid, "next_cid"),
            "kind": _text(kind, "kind"),
            "body": dict(body or {}),
            "created_at_ns": self._clock_ns(),
        }
        supersession_cid = _cid_for(record)
        record["supersession_cid"] = supersession_cid
        self._append_jsonl(self.supersessions_path, record)
        self.put_cas(record)
        self._append_event(
            PlanRevisionEventType.SUPERSESSION_APPENDED,
            revision_cid=next_cid,
            body={"supersession_cid": supersession_cid, "kind": kind},
        )
        return supersession_cid

    def list_supersessions(self) -> tuple[dict[str, Any], ...]:
        if not self.supersessions_path.exists():
            return ()
        rows: list[dict[str, Any]] = []
        with self.supersessions_path.open("rb") as stream:
            for raw in stream:
                if not raw.strip():
                    continue
                value = json.loads(raw)
                if isinstance(value, dict):
                    rows.append(value)
        return tuple(rows)

    # -- active projection ---------------------------------------------------

    def get_active(self) -> PlanRevisionActiveProjection | None:
        if not self.active_path.exists():
            return None
        return PlanRevisionActiveProjection.from_dict(_read_json(self.active_path))

    def get_prior_active(self) -> PlanRevisionActiveProjection | None:
        if not self.prior_active_path.exists():
            return None
        return PlanRevisionActiveProjection.from_dict(
            _read_json(self.prior_active_path)
        )

    def load_revision(self, revision_cid: str) -> PlanRevision:
        payload = self.get_cas(revision_cid)
        return PlanRevision.from_dict(payload)

    def list_revision_cids(self) -> tuple[str, ...]:
        index = self._read_index()
        return tuple(str(item) for item in index.get("revisions") or [])

    def is_quarantined(self) -> bool:
        active = self.get_active()
        return bool(active and active.quarantined)

    def _publish_active(
        self,
        active: PlanRevisionActiveProjection,
        *,
        retain_prior: bool = True,
    ) -> None:
        if retain_prior and self.active_path.exists():
            prior = _read_json(self.active_path)
            _atomic_write_json(self.prior_active_path, prior)
        _atomic_write_json(self.active_path, active.to_dict())

    def _restore_prior_active(self) -> PlanRevisionActiveProjection | None:
        """Restore the last committed active pointer after a failed apply.

        When no prior pointer exists, the current active projection is left
        intact: the failed apply never successfully published a replacement.
        """

        if not self.prior_active_path.exists():
            return self.get_active()
        prior = PlanRevisionActiveProjection.from_dict(
            _read_json(self.prior_active_path)
        )
        _atomic_write_json(self.active_path, prior.to_dict())
        return prior

    # -- observation / authority --------------------------------------------

    def reobserve_roots(
        self,
        expected: PlanAuthorityRoots | Mapping[str, Any],
        observed: PlanAuthorityRoots | Mapping[str, Any],
    ) -> PlanAuthorityRoots:
        """Fail closed when any bound authority root drifted."""

        expected_roots = _decode_roots(expected)
        observed_roots = _decode_roots(observed)
        try:
            observed_roots.require_current(expected_roots)
        except PlanRevisionStaleRootError as exc:
            raise PlanRevisionStoreStaleError(str(exc)) from exc
        return observed_roots

    def _assert_lifecycle_safe(
        self,
        revision: PlanRevision,
        delta: PlanDelta | None,
        *,
        prior: PlanRevision | None,
    ) -> None:
        if delta is not None:
            assert_delta_preserves_history(delta)
            for item in delta.items:
                if item.expected_target_lifecycle in {
                    LifecycleState.COMPLETED,
                    LifecycleState.ACCEPTED,
                    LifecycleState.CLAIMED,
                    LifecycleState.RUNNING,
                    LifecycleState.SETTLING,
                } and item.operation in {
                    PlanDeltaOperation.SUPERSEDE_UNSTARTED_TASK,
                    PlanDeltaOperation.AMEND_UNSTARTED_GOAL,
                    PlanDeltaOperation.SPLIT_UNSTARTED_TASK,
                    PlanDeltaOperation.COALESCE_UNSTARTED_TASKS,
                    PlanDeltaOperation.REWIRE_UNSTARTED_DEPENDENCY,
                    PlanDeltaOperation.REPRIORITIZE_UNSTARTED_TASK,
                }:
                    raise PlanRevisionLifecycleError(
                        f"delta item {item.item_key!r} would edit "
                        f"{item.expected_target_lifecycle.value} specs"
                    )
        if prior is not None:
            assert_population_history_intact(
                prior_completed=prior.completed_population.member_cids,
                prior_accepted=(),
                prior_claimed=prior.claimed_population.member_cids,
                next_completed=revision.completed_population.member_cids,
                next_accepted=(),
                next_claimed=revision.claimed_population.member_cids,
            )
            protected = _protected_cids(prior)
            # Spec-bearing retained members of protected populations must still
            # appear in the candidate revision task population.
            retained = set(revision.task_population.member_cids)
            missing = protected - retained - set(
                revision.superseded_population.member_cids
            )
            # Claimed/completed members may remain only in claimed/completed
            # digests; ensure they were not deleted from history digests.
            if missing & set(prior.task_population.member_cids):
                # Only enforce when prior tracked them in task_population.
                still_tracked = protected & set(prior.task_population.member_cids)
                lost = still_tracked - set(revision.task_population.member_cids)
                # Supersede of claimed is forbidden; lost claimed/completed is error.
                claimed_lost = lost & set(prior.claimed_population.member_cids)
                completed_lost = lost & set(prior.completed_population.member_cids)
                if claimed_lost or completed_lost:
                    raise PlanRevisionLifecycleError(
                        "accepted/claimed task specifications cannot be removed"
                    )

    # -- intent journal ------------------------------------------------------

    def journal_intent(
        self,
        request: PlanRevisionApplyRequest,
        *,
        markdown_path: str = "",
        duckdb_path: str = "",
    ) -> PlanRevisionIntent:
        revision = request.revision
        expected_roots = revision.roots
        observed = self.reobserve_roots(expected_roots, request.observed_roots)
        delta = request.delta
        if revision.origin is PlanOrigin.STEER and delta is None:
            raise PlanRevisionStoreError("steer apply requires a PlanDelta")
        if revision.origin is PlanOrigin.CREATE and revision.semantic_revision != 1:
            raise PlanRevisionStoreError("create apply requires semantic_revision == 1")
        if delta is not None and delta.base_plan_root and revision.parent_plan_root:
            if delta.base_plan_root != revision.parent_plan_root:
                raise PlanRevisionStoreConflictError(
                    "delta base_plan_root must match revision parent_plan_root"
                )
        prior_revision = None
        active = self.get_active()
        if active is not None:
            if request.expected_active_plan_root and (
                active.plan_root_cid != request.expected_active_plan_root
            ):
                raise PlanRevisionStoreStaleError(
                    "active plan root does not match expected fence"
                )
            if request.expected_active_revision_cid and (
                active.revision_cid != request.expected_active_revision_cid
            ):
                raise PlanRevisionStoreStaleError(
                    "active revision cid does not match expected fence"
                )
            if request.base_event_cursor and active.event_cursor:
                if request.base_event_cursor != active.event_cursor:
                    raise PlanRevisionStoreStaleError(
                        "event cursor does not match expected fence"
                    )
            if active.quarantined:
                raise PlanRevisionStoreQuarantinedError(
                    "plan revision store is quarantined"
                )
            prior_revision = self.load_revision(active.revision_cid)
        self._assert_lifecycle_safe(revision, delta, prior=prior_revision)

        expected_effects = tuple(request.expected_effects) or tuple(
            delta.expected_effects if delta is not None else ()
        )
        deferred = tuple(revision.deferred_population.member_cids)
        if delta is not None:
            deferred = tuple(
                dict.fromkeys(
                    list(deferred) + list(delta.deferred_item_keys)
                )
            )
        intent_body = {
            "schema": PLAN_REVISION_INTENT_SCHEMA,
            "idempotency_key": request.idempotency_key,
            "origin": revision.origin.value,
            "base_plan_root": revision.parent_plan_root
            or (active.plan_root_cid if active else ""),
            "base_plan_revision": (
                (active.semantic_revision if active else 0)
                if revision.origin is PlanOrigin.STEER
                else 0
            ),
            "base_event_cursor": request.base_event_cursor
            or (active.event_cursor if active else ""),
            "candidate_plan_root": revision.plan_root_cid,
            "candidate_revision_cid": revision.revision_cid,
            "delta_cid": delta.delta_cid if delta is not None else revision.delta_cid,
            "expected_effects": list(expected_effects),
            "observed_roots": observed.to_dict(),
            "expected_roots": expected_roots.to_dict(),
            "markdown_path": markdown_path,
            "duckdb_path": duckdb_path,
            "fencing_token": request.fencing_token,
            "lease_id": request.lease_id,
            "deferred_item_keys": list(deferred),
            "retained_task_cids": list(revision.retained_population.member_cids),
            "claimed_task_cids": list(revision.claimed_population.member_cids),
            "accepted_task_cids": list(revision.completed_population.member_cids),
            "state": PlanRevisionApplyState.INTENT_JOURNALED.value,
            "created_at_ns": self._clock_ns(),
        }
        intent_cid = _cid_for(intent_body)
        intent_body["intent_cid"] = intent_cid
        intent = PlanRevisionIntent.from_dict(intent_body)
        path = self.intents_dir / f"{intent_cid}.json"
        if path.exists():
            existing = PlanRevisionIntent.from_dict(_read_json(path))
            if existing.to_dict() != intent.to_dict():
                raise PlanRevisionStoreConflictError(
                    "intent cid collision with different payload"
                )
            return existing
        _atomic_write_json(path, intent.to_dict())
        self.put_cas(intent.to_dict())
        self.put_continuation(
            request.idempotency_key,
            {
                "phase": PlanRevisionApplyState.INTENT_JOURNALED.value,
                "intent_cid": intent_cid,
                "revision_cid": revision.revision_cid,
                "plan_root_cid": revision.plan_root_cid,
                "delta_cid": intent.delta_cid,
            },
        )
        self._append_event(
            PlanRevisionEventType.INTENT_JOURNALED,
            intent_cid=intent_cid,
            revision_cid=revision.revision_cid,
            plan_root_cid=revision.plan_root_cid,
            body={"idempotency_key": request.idempotency_key},
        )
        return intent

    def load_intent(self, intent_cid: str) -> PlanRevisionIntent:
        path = self.intents_dir / f"{_text(intent_cid, 'intent_cid')}.json"
        return PlanRevisionIntent.from_dict(_read_json(path))

    def _update_intent_state(
        self, intent: PlanRevisionIntent, state: PlanRevisionApplyState
    ) -> PlanRevisionIntent:
        body = intent.to_dict()
        body["state"] = state.value
        updated = PlanRevisionIntent.from_dict(body)
        _atomic_write_json(self.intents_dir / f"{intent.intent_cid}.json", body)
        self.put_cas(body)
        return updated

    # -- projection helpers --------------------------------------------------

    def _backend_paths(
        self, request: PlanRevisionApplyRequest
    ) -> tuple[str, str]:
        markdown_path = ""
        duckdb_path = ""
        if request.markdown_source is not None:
            markdown_path = str(Path(request.markdown_source.path).resolve())
        if request.duckdb_source is not None:
            duckdb_path = str(Path(request.duckdb_source.database_path).resolve())
        return markdown_path, duckdb_path

    def _backup_projection(
        self, intent_cid: str, kind: str, path: Path
    ) -> Path | None:
        if not path.exists():
            return None
        destination = self.backups_dir / f"{intent_cid}.{kind}"
        if path.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)
        return destination

    def _restore_projection_file(
        self, backup: Path | None, target: Path
    ) -> None:
        if backup is None:
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        if backup.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(backup, target)
        else:
            shutil.copy2(backup, target)

    def _apply_markdown(
        self,
        request: PlanRevisionApplyRequest,
        *,
        intent: PlanRevisionIntent,
    ) -> str:
        source = request.markdown_source
        if source is None:
            return ""
        apply_fn = getattr(source, "apply_plan_revision", None)
        if not callable(apply_fn):
            raise PlanRevisionStoreError(
                "markdown_source must implement apply_plan_revision"
            )
        result = apply_fn(
            revision=request.revision,
            admission=request.admission,
            goal_graph=request.goal_graph,
            aliases=request.aliases,
            retained_task_cids=intent.retained_task_cids,
            claimed_task_cids=intent.claimed_task_cids,
            deferred_item_keys=intent.deferred_item_keys,
            origin=request.revision.origin.value,
            store_continuation=self,
            idempotency_key=request.idempotency_key,
        )
        if isinstance(result, Mapping):
            return str(
                result.get("projection_cid")
                or result.get("board_revision")
                or ""
            )
        projection_cid = getattr(result, "projection_cid", None)
        if projection_cid:
            return str(projection_cid)
        snapshot = getattr(result, "snapshot", None)
        if snapshot is not None:
            return str(
                getattr(snapshot, "board_revision", "")
                or getattr(snapshot, "projection_id", "")
                or ""
            )
        return str(result)

    def _apply_duckdb(
        self,
        request: PlanRevisionApplyRequest,
        *,
        intent: PlanRevisionIntent,
    ) -> str:
        source = request.duckdb_source
        if source is None:
            return ""
        apply_fn = getattr(source, "apply_plan_revision", None)
        if not callable(apply_fn):
            raise PlanRevisionStoreError(
                "duckdb_source must implement apply_plan_revision"
            )
        result = apply_fn(
            revision=request.revision,
            admission=request.admission,
            goal_graph=request.goal_graph,
            aliases=request.aliases,
            repository_tree_id=request.repository_tree_id
            or request.observed_roots.dirty_worktree_root,
            retained_task_cids=intent.retained_task_cids,
            claimed_task_cids=intent.claimed_task_cids,
            deferred_item_keys=intent.deferred_item_keys,
            origin=request.revision.origin.value,
            delta=request.delta,
            store_continuation=self,
            idempotency_key=request.idempotency_key,
            fencing_token=request.fencing_token,
        )
        if isinstance(result, Mapping):
            return str(
                result.get("projection_cid")
                or result.get("receipt_cid")
                or ""
            )
        return str(getattr(result, "projection_cid", result))

    def _verify_projection_cids(
        self,
        request: PlanRevisionApplyRequest,
        *,
        markdown_cid: str,
        duckdb_cid: str,
    ) -> tuple[str, str]:
        markdown_source = request.markdown_source
        duckdb_source = request.duckdb_source
        observed_md = markdown_cid
        observed_db = duckdb_cid
        if markdown_source is not None:
            verify = getattr(markdown_source, "plan_revision_projection_cid", None)
            if callable(verify):
                observed_md = str(verify())
            elif hasattr(markdown_source, "snapshot"):
                snap = markdown_source.snapshot()
                observed_md = str(
                    getattr(snap, "board_revision", "")
                    or getattr(snap, "projection_id", "")
                    or observed_md
                )
        if duckdb_source is not None:
            verify = getattr(duckdb_source, "plan_revision_projection_cid", None)
            if callable(verify):
                observed_db = str(verify())
            elif hasattr(duckdb_source, "snapshot"):
                snap = duckdb_source.snapshot()
                observed_db = str(
                    getattr(snap, "projection_cid", "") or observed_db
                )
        if markdown_cid and observed_md and markdown_cid != observed_md:
            raise PlanRevisionStoreIntegrityError(
                "Markdown projection CID failed round-trip verification"
            )
        if duckdb_cid and observed_db and duckdb_cid != observed_db:
            raise PlanRevisionStoreIntegrityError(
                "DuckDB projection CID failed round-trip verification"
            )
        # Dual parity: when both backends materialize the same admitted graph,
        # require both CIDs present; exact cross-backend equality is not always
        # possible (different encodings), so compare via backend parity helpers
        # when available.
        if markdown_source is not None and duckdb_source is not None:
            parity = getattr(markdown_source, "compare_plan_revision_parity", None)
            if callable(parity):
                report = parity(duckdb_source)
                valid = bool(
                    report is True
                    or (isinstance(report, Mapping) and report.get("valid"))
                    or getattr(report, "valid", False)
                )
                if not valid:
                    raise PlanRevisionStoreIntegrityError(
                        "Markdown/DuckDB plan revision projections disagree"
                    )
        return observed_md, observed_db

    # -- apply ---------------------------------------------------------------

    def apply(
        self, request: PlanRevisionApplyRequest | Mapping[str, Any]
    ) -> PlanRevisionApplyReceipt:
        """Re-observe, journal intent, append history, and atomically apply."""

        if not isinstance(request, PlanRevisionApplyRequest):
            if not isinstance(request, Mapping):
                raise PlanRevisionStoreError(
                    "request must be PlanRevisionApplyRequest or mapping"
                )
            request = PlanRevisionApplyRequest(**dict(request))

        with self._thread_lock:
            with self._guard():
                return self._apply_locked(request)

    def _apply_locked(
        self, request: PlanRevisionApplyRequest
    ) -> PlanRevisionApplyReceipt:
        if self.is_quarantined():
            raise PlanRevisionStoreQuarantinedError(
                "plan revision store is quarantined"
            )

        # Idempotent replay from durable continuation / committed index.
        continuation = self.load_continuation(request.idempotency_key)
        if continuation is not None:
            receipt_cid = str(continuation.get("receipt_cid") or "")
            if continuation.get("phase") == PlanRevisionApplyState.COMMITTED.value:
                if receipt_cid:
                    try:
                        payload = self.get_cas(receipt_cid)
                        return PlanRevisionApplyReceipt(
                            receipt_cid=str(payload.get("receipt_cid") or receipt_cid),
                            intent_cid=str(payload.get("intent_cid") or ""),
                            state=PlanRevisionApplyState.REPLAYED,
                            revision_cid=str(payload.get("revision_cid") or ""),
                            plan_root_cid=str(payload.get("plan_root_cid") or ""),
                            delta_cid=str(payload.get("delta_cid") or ""),
                            markdown_projection_cid=str(
                                payload.get("markdown_projection_cid") or ""
                            ),
                            duckdb_projection_cid=str(
                                payload.get("duckdb_projection_cid") or ""
                            ),
                            prior_active_cid=str(
                                payload.get("prior_active_cid") or ""
                            ),
                            event_cursor=str(payload.get("event_cursor") or ""),
                            expected_effects=_sequence_ids(
                                payload.get("expected_effects"),
                                "expected_effects",
                            ),
                            observed_effects=_sequence_ids(
                                payload.get("observed_effects"),
                                "observed_effects",
                            ),
                            deferred_item_keys=_sequence_ids(
                                payload.get("deferred_item_keys"),
                                "deferred_item_keys",
                            ),
                            activated_deferred_keys=_sequence_ids(
                                payload.get("activated_deferred_keys"),
                                "activated_deferred_keys",
                            ),
                            resumed=True,
                            reason_codes=("idempotent_replay",),
                            markdown_path=str(payload.get("markdown_path") or ""),
                            duckdb_path=str(payload.get("duckdb_path") or ""),
                        )
                    except PlanRevisionStoreError:
                        pass

        markdown_path, duckdb_path = self._backend_paths(request)
        intent = self.journal_intent(
            request,
            markdown_path=markdown_path,
            duckdb_path=duckdb_path,
        )
        self._fault(request.fault_injector, "after_intent")

        # Append revision/delta/records into CAS + indexes (history never rewritten).
        revision_payload = request.revision.to_dict()
        revision_cid = self.put_cas(revision_payload)
        if revision_cid != request.revision.revision_cid:
            # Prefer the contract identity; store under that key explicitly.
            path = self.cas_dir / request.revision.revision_cid
            if not path.exists():
                _atomic_write_json(
                    path,
                    {
                        "schema": PLAN_REVISION_STORE_SCHEMA,
                        "cid": request.revision.revision_cid,
                        "media_type": "application/json",
                        "payload": revision_payload,
                    },
                )
            revision_cid = request.revision.revision_cid

        delta_cid = ""
        if request.delta is not None:
            delta_payload = request.delta.to_dict()
            delta_cid = request.delta.delta_cid
            path = self.cas_dir / delta_cid
            if not path.exists():
                _atomic_write_json(
                    path,
                    {
                        "schema": PLAN_REVISION_STORE_SCHEMA,
                        "cid": delta_cid,
                        "media_type": "application/json",
                        "payload": delta_payload,
                    },
                )
            self._append_event(
                PlanRevisionEventType.DELTA_APPENDED,
                intent_cid=intent.intent_cid,
                revision_cid=revision_cid,
                plan_root_cid=request.revision.plan_root_cid,
                body={"delta_cid": delta_cid},
            )

        if request.records:
            for key, record in sorted(request.records.items(), key=lambda item: str(item[0])):
                self.put_cas(
                    {
                        "record_key": str(key),
                        "record": record
                        if not hasattr(record, "to_dict")
                        else record.to_dict(),
                    }
                )

        index = self._read_index()
        revisions = list(index.get("revisions") or [])
        deltas = list(index.get("deltas") or [])
        if revision_cid not in revisions:
            revisions.append(revision_cid)
        if delta_cid and delta_cid not in deltas:
            deltas.append(delta_cid)
        index.update(
            {
                "revisions": revisions,
                "deltas": deltas,
                "latest_revision_cid": revision_cid,
                "latest_intent_cid": intent.intent_cid,
            }
        )
        self._write_index(index)
        self._append_event(
            PlanRevisionEventType.REVISION_APPENDED,
            intent_cid=intent.intent_cid,
            revision_cid=revision_cid,
            plan_root_cid=request.revision.plan_root_cid,
        )
        self._fault(request.fault_injector, "after_history_append")

        # Supersession links for steer.
        if (
            request.revision.origin is PlanOrigin.STEER
            and request.revision.parent_plan_root
        ):
            for member in request.revision.superseded_population.member_cids:
                self.append_supersession(
                    prior_cid=member,
                    next_cid=request.revision.plan_root_cid,
                    kind="task_or_goal_supersession",
                    body={"revision_cid": revision_cid},
                )
            self.append_supersession(
                prior_cid=request.revision.parent_plan_root,
                next_cid=request.revision.plan_root_cid,
                kind="plan_revision",
                body={"revision_cid": revision_cid},
            )

        prior_active = self.get_active()
        prior_active_cid = prior_active.active_cid if prior_active else ""

        md_backup = None
        db_backup = None
        if request.markdown_source is not None:
            md_backup = self._backup_projection(
                intent.intent_cid,
                "markdown",
                Path(request.markdown_source.path),
            )
        if request.duckdb_source is not None:
            db_backup = self._backup_projection(
                intent.intent_cid,
                "duckdb",
                Path(request.duckdb_source.database_path),
            )

        intent = self._update_intent_state(
            intent, PlanRevisionApplyState.PREPARED
        )
        self.put_continuation(
            request.idempotency_key,
            {
                "phase": PlanRevisionApplyState.PREPARED.value,
                "intent_cid": intent.intent_cid,
                "revision_cid": revision_cid,
                "plan_root_cid": request.revision.plan_root_cid,
                "delta_cid": delta_cid,
                "markdown_backup": str(md_backup) if md_backup else "",
                "duckdb_backup": str(db_backup) if db_backup else "",
            },
        )

        markdown_cid = ""
        duckdb_cid = ""
        try:
            self._fault(request.fault_injector, "after_prepare")
            # Re-observe immediately before projection writes.
            self.reobserve_roots(request.revision.roots, request.observed_roots)
            markdown_cid = self._apply_markdown(request, intent=intent)
            self._fault(request.fault_injector, "after_markdown")
            duckdb_cid = self._apply_duckdb(request, intent=intent)
            self._fault(request.fault_injector, "after_duckdb")
            markdown_cid, duckdb_cid = self._verify_projection_cids(
                request,
                markdown_cid=markdown_cid,
                duckdb_cid=duckdb_cid,
            )
            self._append_event(
                PlanRevisionEventType.PROJECTION_VERIFIED,
                intent_cid=intent.intent_cid,
                revision_cid=revision_cid,
                plan_root_cid=request.revision.plan_root_cid,
                body={
                    "markdown_projection_cid": markdown_cid,
                    "duckdb_projection_cid": duckdb_cid,
                },
            )
            intent = self._update_intent_state(
                intent, PlanRevisionApplyState.VERIFIED
            )
            self._fault(request.fault_injector, "after_verify")
        except Exception as exc:
            # Compensate: restore prior projection bytes and active pointer.
            if request.markdown_source is not None:
                self._restore_projection_file(
                    md_backup, Path(request.markdown_source.path)
                )
            if request.duckdb_source is not None:
                self._restore_projection_file(
                    db_backup, Path(request.duckdb_source.database_path)
                )
            restored = self._restore_prior_active()
            reason = str(exc) or type(exc).__name__
            # Only quarantine true dual-backend split-brain / parity failures.
            # Ordinary integrity or conflict errors restore the prior active
            # projection and remain retryable after the operator corrects inputs.
            split_brain = (
                "projections disagree" in reason
                or "split_brain" in reason.lower()
                or "split-brain" in reason.lower()
            )
            if split_brain:
                self._quarantine(
                    intent_cid=intent.intent_cid,
                    revision_cid=revision_cid,
                    reason=reason,
                    body={
                        "markdown_projection_cid": markdown_cid,
                        "duckdb_projection_cid": duckdb_cid,
                    },
                )
                receipt = self._receipt(
                    intent=intent,
                    state=PlanRevisionApplyState.QUARANTINED,
                    revision_cid=revision_cid,
                    plan_root_cid=request.revision.plan_root_cid,
                    delta_cid=delta_cid,
                    markdown_cid=markdown_cid,
                    duckdb_cid=duckdb_cid,
                    prior_active_cid=prior_active_cid,
                    reason_codes=("split_brain_quarantined", reason),
                    quarantined=True,
                )
                self.put_continuation(
                    request.idempotency_key,
                    {
                        "phase": PlanRevisionApplyState.QUARANTINED.value,
                        "intent_cid": intent.intent_cid,
                        "receipt_cid": receipt.receipt_cid,
                    },
                )
                raise PlanRevisionStoreQuarantinedError(reason) from exc
            intent = self._update_intent_state(
                intent, PlanRevisionApplyState.RESTORED
            )
            self._append_event(
                PlanRevisionEventType.PROJECTION_RESTORED,
                intent_cid=intent.intent_cid,
                revision_cid=revision_cid,
                body={
                    "reason": reason,
                    "restored_active_cid": restored.active_cid if restored else "",
                },
            )
            receipt = self._receipt(
                intent=intent,
                state=PlanRevisionApplyState.RESTORED,
                revision_cid=revision_cid,
                plan_root_cid=(
                    restored.plan_root_cid
                    if restored is not None
                    else request.revision.plan_root_cid
                ),
                delta_cid=delta_cid,
                markdown_cid=markdown_cid,
                duckdb_cid=duckdb_cid,
                prior_active_cid=prior_active_cid,
                reason_codes=("projection_restored", reason),
            )
            self.put_continuation(
                request.idempotency_key,
                {
                    "phase": PlanRevisionApplyState.RESTORED.value,
                    "intent_cid": intent.intent_cid,
                    "receipt_cid": receipt.receipt_cid,
                },
            )
            raise PlanRevisionStoreConflictError(reason) from exc

        activated = tuple(request.activate_deferred_keys)
        remaining_deferred = tuple(
            key
            for key in intent.deferred_item_keys
            if key not in set(activated)
        )
        event_cursor = _cid_for(
            {
                "revision_cid": revision_cid,
                "intent_cid": intent.intent_cid,
                "markdown_projection_cid": markdown_cid,
                "duckdb_projection_cid": duckdb_cid,
            }
        )
        active = PlanRevisionActiveProjection(
            active_cid=_cid_for(
                {
                    "plan_root_cid": request.revision.plan_root_cid,
                    "revision_cid": revision_cid,
                    "event_cursor": event_cursor,
                    "markdown_projection_cid": markdown_cid,
                    "duckdb_projection_cid": duckdb_cid,
                }
            ),
            plan_root_cid=request.revision.plan_root_cid,
            revision_cid=revision_cid,
            semantic_revision=request.revision.semantic_revision,
            event_cursor=event_cursor,
            markdown_projection_cid=markdown_cid,
            duckdb_projection_cid=duckdb_cid,
            markdown_path=markdown_path,
            duckdb_path=duckdb_path,
            intent_cid=intent.intent_cid,
            prior_active_cid=prior_active_cid,
            deferred_item_keys=remaining_deferred,
            quarantined=False,
        )
        self._publish_active(active, retain_prior=True)
        self._fault(request.fault_injector, "after_commit_pointer")
        intent = self._update_intent_state(
            intent, PlanRevisionApplyState.COMMITTED
        )
        if activated:
            self._append_event(
                PlanRevisionEventType.DEFERRED_ACTIVATED,
                intent_cid=intent.intent_cid,
                revision_cid=revision_cid,
                plan_root_cid=request.revision.plan_root_cid,
                body={"activated_deferred_keys": list(activated)},
            )
        self._append_event(
            PlanRevisionEventType.PROJECTION_COMMITTED,
            intent_cid=intent.intent_cid,
            revision_cid=revision_cid,
            plan_root_cid=request.revision.plan_root_cid,
            body={"active_cid": active.active_cid},
        )
        receipt = self._receipt(
            intent=intent,
            state=PlanRevisionApplyState.COMMITTED,
            revision_cid=revision_cid,
            plan_root_cid=request.revision.plan_root_cid,
            delta_cid=delta_cid,
            markdown_cid=markdown_cid,
            duckdb_cid=duckdb_cid,
            prior_active_cid=prior_active_cid,
            event_cursor=event_cursor,
            observed_effects=intent.expected_effects,
            activated_deferred_keys=activated,
        )
        # Store the receipt under its stable receipt_cid (identity of the
        # body-free receipt fields), not a secondary identity of the wrapped
        # envelope that includes the cid field itself.
        receipt_path = self.cas_dir / receipt.receipt_cid
        if not receipt_path.exists():
            _atomic_write_json(
                receipt_path,
                {
                    "schema": PLAN_REVISION_STORE_SCHEMA,
                    "cid": receipt.receipt_cid,
                    "media_type": "application/json",
                    "payload": receipt.to_dict(),
                },
            )
        self.put_continuation(
            request.idempotency_key,
            {
                "phase": PlanRevisionApplyState.COMMITTED.value,
                "intent_cid": intent.intent_cid,
                "receipt_cid": receipt.receipt_cid,
                "revision_cid": revision_cid,
                "plan_root_cid": request.revision.plan_root_cid,
                "active_cid": active.active_cid,
            },
        )
        return receipt

    def _receipt(
        self,
        *,
        intent: PlanRevisionIntent,
        state: PlanRevisionApplyState,
        revision_cid: str,
        plan_root_cid: str,
        delta_cid: str = "",
        markdown_cid: str = "",
        duckdb_cid: str = "",
        prior_active_cid: str = "",
        event_cursor: str = "",
        observed_effects: Sequence[str] = (),
        activated_deferred_keys: Sequence[str] = (),
        reason_codes: Sequence[str] = (),
        quarantined: bool = False,
    ) -> PlanRevisionApplyReceipt:
        body = {
            "schema": PLAN_REVISION_APPLY_RECEIPT_SCHEMA,
            "intent_cid": intent.intent_cid,
            "state": state.value,
            "revision_cid": revision_cid,
            "plan_root_cid": plan_root_cid,
            "delta_cid": delta_cid,
            "markdown_projection_cid": markdown_cid,
            "duckdb_projection_cid": duckdb_cid,
            "prior_active_cid": prior_active_cid,
            "event_cursor": event_cursor,
            "expected_effects": list(intent.expected_effects),
            "observed_effects": list(observed_effects),
            "deferred_item_keys": list(intent.deferred_item_keys),
            "activated_deferred_keys": list(activated_deferred_keys),
            "quarantined": quarantined,
            "reason_codes": list(reason_codes),
            "markdown_path": intent.markdown_path,
            "duckdb_path": intent.duckdb_path,
        }
        receipt_cid = _cid_for(body)
        return PlanRevisionApplyReceipt(
            receipt_cid=receipt_cid,
            intent_cid=intent.intent_cid,
            state=state,
            revision_cid=revision_cid,
            plan_root_cid=plan_root_cid,
            delta_cid=delta_cid,
            markdown_projection_cid=markdown_cid,
            duckdb_projection_cid=duckdb_cid,
            prior_active_cid=prior_active_cid,
            event_cursor=event_cursor,
            expected_effects=intent.expected_effects,
            observed_effects=tuple(observed_effects),
            deferred_item_keys=intent.deferred_item_keys,
            activated_deferred_keys=tuple(activated_deferred_keys),
            quarantined=quarantined,
            reason_codes=tuple(reason_codes),
            markdown_path=intent.markdown_path,
            duckdb_path=intent.duckdb_path,
        )

    def _quarantine(
        self,
        *,
        intent_cid: str,
        revision_cid: str,
        reason: str,
        body: Mapping[str, Any] | None = None,
    ) -> None:
        record = {
            "intent_cid": intent_cid,
            "revision_cid": revision_cid,
            "reason": reason,
            "body": dict(body or {}),
            "created_at_ns": self._clock_ns(),
        }
        digest = hashlib.sha256(_canonical_bytes(record)).hexdigest()
        path = self.quarantine_dir / f"{intent_cid}.{digest}.json"
        _atomic_write_json(path, record)
        active = self.get_active()
        if active is not None:
            quarantined = PlanRevisionActiveProjection(
                active_cid=active.active_cid,
                plan_root_cid=active.plan_root_cid,
                revision_cid=active.revision_cid,
                semantic_revision=active.semantic_revision,
                event_cursor=active.event_cursor,
                markdown_projection_cid=active.markdown_projection_cid,
                duckdb_projection_cid=active.duckdb_projection_cid,
                markdown_path=active.markdown_path,
                duckdb_path=active.duckdb_path,
                intent_cid=active.intent_cid,
                prior_active_cid=active.prior_active_cid,
                deferred_item_keys=active.deferred_item_keys,
                quarantined=True,
            )
            _atomic_write_json(self.active_path, quarantined.to_dict())
        else:
            # Publish a quarantine-only active marker so subsequent applies fail.
            marker = PlanRevisionActiveProjection(
                active_cid=_cid_for(record),
                plan_root_cid=str((body or {}).get("plan_root_cid") or "quarantined"),
                revision_cid=revision_cid or "quarantined",
                semantic_revision=0,
                event_cursor="",
                intent_cid=intent_cid,
                quarantined=True,
            )
            _atomic_write_json(self.active_path, marker.to_dict())
        self._append_event(
            PlanRevisionEventType.SPLIT_BRAIN_QUARANTINED,
            intent_cid=intent_cid,
            revision_cid=revision_cid,
            body={"reason": reason, "quarantine_path": str(path)},
        )

    # -- deferred successors -------------------------------------------------

    def activate_deferred(
        self,
        item_keys: Sequence[str],
        *,
        preconditions_satisfied: Mapping[str, bool] | None = None,
    ) -> tuple[str, ...]:
        """Activate deferred successors whose preconditions are true.

        Returns the keys that were activated.  Keys whose preconditions are
        not yet satisfied remain deferred on the active projection.
        """

        with self._thread_lock:
            with self._guard():
                active = self.get_active()
                if active is None:
                    raise PlanRevisionStoreError("no active plan revision")
                if active.quarantined:
                    raise PlanRevisionStoreQuarantinedError(
                        "plan revision store is quarantined"
                    )
                satisfied = {
                    str(key): bool(value)
                    for key, value in dict(preconditions_satisfied or {}).items()
                }
                activated: list[str] = []
                remaining: list[str] = []
                requested = set(_sequence_ids(item_keys, "item_keys"))
                for key in active.deferred_item_keys:
                    if key in requested and satisfied.get(key, False):
                        activated.append(key)
                    else:
                        remaining.append(key)
                if not activated:
                    return ()
                updated = PlanRevisionActiveProjection(
                    active_cid=_cid_for(
                        {
                            "prior_active_cid": active.active_cid,
                            "activated": activated,
                            "remaining": remaining,
                        }
                    ),
                    plan_root_cid=active.plan_root_cid,
                    revision_cid=active.revision_cid,
                    semantic_revision=active.semantic_revision,
                    event_cursor=active.event_cursor,
                    markdown_projection_cid=active.markdown_projection_cid,
                    duckdb_projection_cid=active.duckdb_projection_cid,
                    markdown_path=active.markdown_path,
                    duckdb_path=active.duckdb_path,
                    intent_cid=active.intent_cid,
                    prior_active_cid=active.active_cid,
                    deferred_item_keys=tuple(remaining),
                    quarantined=False,
                )
                self._publish_active(updated, retain_prior=True)
                self._append_event(
                    PlanRevisionEventType.DEFERRED_ACTIVATED,
                    intent_cid=active.intent_cid,
                    revision_cid=active.revision_cid,
                    plan_root_cid=active.plan_root_cid,
                    body={"activated_deferred_keys": activated},
                )
                return tuple(activated)

    # -- recovery ------------------------------------------------------------

    def recover(self) -> tuple[str, ...]:
        """Resume, restore, or quarantine every interrupted apply boundary."""

        recovered: list[str] = []
        with self._thread_lock:
            with self._guard():
                # Scan durable continuations; never rely on process dictionaries.
                for path in sorted(self.continuations_dir.glob("*.json")):
                    try:
                        record = _read_json(path)
                    except PlanRevisionStoreError:
                        quarantine_path = self.quarantine_dir / path.name
                        try:
                            os.replace(path, quarantine_path)
                        except OSError:
                            pass
                        recovered.append(f"quarantined-corrupt:{path.name}")
                        continue
                    phase = str(record.get("payload", {}).get("phase") or "")
                    intent_cid = str(
                        record.get("payload", {}).get("intent_cid") or ""
                    )
                    if phase in {
                        PlanRevisionApplyState.COMMITTED.value,
                        PlanRevisionApplyState.REPLAYED.value,
                        PlanRevisionApplyState.QUARANTINED.value,
                    }:
                        continue
                    if phase in {
                        PlanRevisionApplyState.PREPARED.value,
                        PlanRevisionApplyState.VERIFIED.value,
                        PlanRevisionApplyState.INTENT_JOURNALED.value,
                    }:
                        # Incomplete apply: restore prior projections when backups exist.
                        payload = dict(record.get("payload") or {})
                        md_backup = payload.get("markdown_backup") or ""
                        db_backup = payload.get("duckdb_backup") or ""
                        active = self.get_active()
                        intent = None
                        if intent_cid:
                            intent_path = self.intents_dir / f"{intent_cid}.json"
                            if intent_path.exists():
                                intent = PlanRevisionIntent.from_dict(
                                    _read_json(intent_path)
                                )
                        if intent is not None and intent.markdown_path and md_backup:
                            self._restore_projection_file(
                                Path(md_backup), Path(intent.markdown_path)
                            )
                        if intent is not None and intent.duckdb_path and db_backup:
                            self._restore_projection_file(
                                Path(db_backup), Path(intent.duckdb_path)
                            )
                        # Only restore active pointer when the incomplete intent
                        # advanced past prepare without commit.
                        if phase in {
                            PlanRevisionApplyState.PREPARED.value,
                            PlanRevisionApplyState.VERIFIED.value,
                        }:
                            self._restore_prior_active()
                        if intent is not None:
                            self._update_intent_state(
                                intent, PlanRevisionApplyState.RESTORED
                            )
                        payload["phase"] = PlanRevisionApplyState.RESTORED.value
                        record["payload"] = payload
                        _atomic_write_json(path, record)
                        self._append_event(
                            PlanRevisionEventType.RECOVERED,
                            intent_cid=intent_cid,
                            body={
                                "phase": phase,
                                "active_plan_root": (
                                    active.plan_root_cid if active else ""
                                ),
                            },
                        )
                        recovered.append(intent_cid or path.name)
                # Detect split-brain between active pointer and prior without
                # matching continuation: quarantine.
                active = self.get_active()
                prior = self.get_prior_active()
                if (
                    active is not None
                    and prior is not None
                    and active.plan_root_cid
                    and prior.plan_root_cid
                    and active.plan_root_cid == prior.plan_root_cid
                    and active.revision_cid != prior.revision_cid
                    and not active.quarantined
                ):
                    # Same plan root with divergent revision cids is corrupt.
                    self._quarantine(
                        intent_cid=active.intent_cid,
                        revision_cid=active.revision_cid,
                        reason="active/prior revision identity split",
                    )
                    recovered.append(f"quarantined-split:{active.revision_cid}")
        return tuple(recovered)


def _safe_filename(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    cleaned = "".join(
        ch if ch.isalnum() or ch in "._-@" else "_" for ch in value
    )[:96]
    return f"{cleaned}.{digest[:16]}"


def open_plan_revision_store(
    root: Path | str,
    **kwargs: Any,
) -> PlanRevisionStore:
    """Open or create one plan revision store root."""

    return PlanRevisionStore(root, **kwargs)


__all__ = [
    "MAX_CAS_BYTES",
    "MAX_CONTINUATION_BYTES",
    "MAX_EVENTS",
    "MAX_INDEX_ENTRIES",
    "MAX_INTENT_BYTES",
    "PLAN_REVISION_ACTIVE_SCHEMA",
    "PLAN_REVISION_APPLY_RECEIPT_SCHEMA",
    "PLAN_REVISION_CONTINUATION_SCHEMA",
    "PLAN_REVISION_EVENT_SCHEMA",
    "PLAN_REVISION_INDEX_SCHEMA",
    "PLAN_REVISION_INTENT_SCHEMA",
    "PLAN_REVISION_STORE_INTERFACE",
    "PLAN_REVISION_STORE_SCHEMA",
    "PLAN_REVISION_SUPERSESSION_SCHEMA",
    "PlanRevisionActiveProjection",
    "PlanRevisionApplyReceipt",
    "PlanRevisionApplyRequest",
    "PlanRevisionApplyState",
    "PlanRevisionBackendKind",
    "PlanRevisionEventType",
    "PlanRevisionIntent",
    "PlanRevisionStore",
    "PlanRevisionStoreConflictError",
    "PlanRevisionStoreError",
    "PlanRevisionStoreIntegrityError",
    "PlanRevisionStoreQuarantinedError",
    "PlanRevisionStoreStaleError",
    "open_plan_revision_store",
]
