"""Path-independent control-plane state repositories (DQP-008).

Interfaces: ``StateRepository@1``, ``EmbeddedStateRepository@1``,
``QuackStateRepository@1``.

Joins schema installation/verification, the typed Quack client, transaction
primitives, and existing DuckDB store surfaces behind one repository protocol:

* **Embedded** — exclusive local ``control.duckdb`` for hermetic tests and
  offline import/recovery. Import/recovery opens require a maintenance lease.
* **Quack** — production authority through a loopback Quack endpoint. Never
  silently falls back to direct file writes when Quack transport fails.

Higher layers depend only on this path-independent protocol; adapters must
produce identical canonical results for the shared conformance population.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final, Protocol, runtime_checkable

from .control_plane_contracts import (
    CommandOutcome,
    ControlPlaneAuthorityError,
    ControlPlaneContractError,
    ControlPlaneStoreIdentity,
    StateAuthorityClass,
    StateCommand,
    StateSnapshot,
    StoreGeneration,
    canonical_json_bytes,
)
from .control_plane_schema import (
    install_control_plane_schema,
    verify_installed_schema,
)
from .control_plane_transactions import (
    CASResult,
    RetryPolicy,
    StateTransaction,
)
from .duckdb_state import exclusive_file_lock, open_duckdb_connection
from .quack_state_client import (
    DEFAULT_STORE_ID,
    ClientSession,
    PageResult,
    QuackClientError,
    QuackClientTransportError,
    QuackEndpoint,
    QuackStateClient,
    StatementKind,
    StatementTemplate,
    TransportMode,
    resolve_endpoint,
)

# ---------------------------------------------------------------------------
# Interface / version identities
# ---------------------------------------------------------------------------

STATE_REPOSITORY_INTERFACE: Final = "StateRepository@1"
EMBEDDED_STATE_REPOSITORY_INTERFACE: Final = "EmbeddedStateRepository@1"
QUACK_STATE_REPOSITORY_INTERFACE: Final = "QuackStateRepository@1"

STATE_REPOSITORY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/state-repository@1"
)
EMBEDDED_STATE_REPOSITORY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/embedded-state-repository@1"
)
QUACK_STATE_REPOSITORY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-state-repository@1"
)
MAINTENANCE_LEASE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/maintenance-lease@1"
)
CONFORMANCE_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/repository-conformance@1"
)

STATE_REPOSITORY_VERSION: Final[int] = 1

DEFAULT_REPOSITORY_ID: Final = "repository:control-plane"
MAINTENANCE_SCOPE_IMPORT: Final = "import"
MAINTENANCE_SCOPE_RECOVERY: Final = "recovery"
MAINTENANCE_SCOPE_HERMETIC: Final = "hermetic"
DEFAULT_LEASE_TTL_SECONDS: Final = 300

# Evidence subset covered by the shared conformance population.
CONFORMANCE_EVIDENCE_SUBSET: Final[tuple[str, ...]] = (
    "tasks",
    "events",
    "leases",
    "commands",
    "snapshots",
    "transactions",
    "schema_verification",
    "cold_imports",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class StateRepositoryError(ControlPlaneContractError):
    """Base error for path-independent state repository failures."""


class StateRepositoryAuthorityError(StateRepositoryError, ControlPlaneAuthorityError):
    """Authority mode violation (silent fallback, missing lease, etc.)."""


class StateRepositoryLeaseError(StateRepositoryError):
    """Maintenance lease missing, expired, or fenced out."""


class StateRepositoryNotOpenError(StateRepositoryError):
    """Operation requires an open repository session."""


class StateRepositoryConformanceError(StateRepositoryError):
    """Shared conformance population failed or adapters diverged."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class RepositoryAuthorityMode(str, Enum):
    """Closed authority modes for control-plane repository adapters."""

    EMBEDDED_EXCLUSIVE = "embedded_exclusive"
    QUACK = "quack"


class EmbeddedOpenPurpose(str, Enum):
    """Why an embedded exclusive open is requested."""

    HERMETIC_TEST = "hermetic_test"
    IMPORT = "import"
    RECOVERY = "recovery"
    MAINTENANCE = "maintenance"


_LEASE_REQUIRED_PURPOSES: Final[frozenset[EmbeddedOpenPurpose]] = frozenset(
    {
        EmbeddedOpenPurpose.IMPORT,
        EmbeddedOpenPurpose.RECOVERY,
        EmbeddedOpenPurpose.MAINTENANCE,
    }
)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _new_id(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4()}"


def _digest_bytes(payload: Any) -> str:
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return f"sha256:{digest}"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise StateRepositoryError(f"{name} must be a string")
    else:
        text = value.strip()
    if required and not text:
        raise StateRepositoryError(f"{name} is required")
    if "\x00" in text:
        raise StateRepositoryError(f"{name} must not contain NUL")
    return text


@dataclass(frozen=True)
class MaintenanceLease:
    """Proved exclusive maintenance lease for embedded import/recovery.

    Interface payload: ``MaintenanceLease@1`` (schema
    ``ipfs_accelerate_py/agent-supervisor/maintenance-lease@1``).
    """

    SCHEMA: ClassVar[str] = MAINTENANCE_LEASE_SCHEMA

    lease_id: str
    scope: str
    owner_session_id: str
    process_birth_id: str
    fencing_token: int
    fence_epoch: int
    acquired_at: str
    expires_at: str
    state: str = "held"
    store_id: str = DEFAULT_STORE_ID
    purpose: str = MAINTENANCE_SCOPE_IMPORT
    revision: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id"))
        object.__setattr__(self, "scope", _text(self.scope, "scope"))
        object.__setattr__(
            self, "owner_session_id", _text(self.owner_session_id, "owner_session_id")
        )
        object.__setattr__(
            self,
            "process_birth_id",
            _text(self.process_birth_id, "process_birth_id"),
        )
        if isinstance(self.fencing_token, bool) or not isinstance(
            self.fencing_token, int
        ):
            raise StateRepositoryLeaseError("fencing_token must be an integer")
        if self.fencing_token < 1:
            raise StateRepositoryLeaseError("fencing_token must be >= 1")
        if isinstance(self.fence_epoch, bool) or not isinstance(self.fence_epoch, int):
            raise StateRepositoryLeaseError("fence_epoch must be an integer")
        if self.fence_epoch < 0:
            raise StateRepositoryLeaseError("fence_epoch must be >= 0")
        object.__setattr__(self, "acquired_at", _text(self.acquired_at, "acquired_at"))
        object.__setattr__(self, "expires_at", _text(self.expires_at, "expires_at"))
        object.__setattr__(self, "state", _text(self.state, "state"))
        object.__setattr__(self, "store_id", _text(self.store_id, "store_id"))
        object.__setattr__(self, "purpose", _text(self.purpose, "purpose"))
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise StateRepositoryLeaseError("revision must be an integer")
        if self.revision < 0:
            raise StateRepositoryLeaseError("revision must be >= 0")

    def is_held(self, *, now: str | None = None) -> bool:
        if self.state != "held":
            return False
        clock = now or _utc_now()
        # ISO-8601 Z timestamps compare lexicographically for second precision.
        return clock <= self.expires_at

    def require_held(self, *, now: str | None = None) -> None:
        if not self.is_held(now=now):
            raise StateRepositoryLeaseError(
                f"maintenance lease {self.lease_id} is not held or has expired"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "lease_id": self.lease_id,
            "scope": self.scope,
            "owner_session_id": self.owner_session_id,
            "process_birth_id": self.process_birth_id,
            "fencing_token": self.fencing_token,
            "fence_epoch": self.fence_epoch,
            "acquired_at": self.acquired_at,
            "expires_at": self.expires_at,
            "state": self.state,
            "store_id": self.store_id,
            "purpose": self.purpose,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MaintenanceLease":
        if not isinstance(payload, Mapping):
            raise StateRepositoryLeaseError("maintenance lease payload must be a mapping")
        return cls(
            lease_id=str(payload.get("lease_id") or ""),
            scope=str(payload.get("scope") or ""),
            owner_session_id=str(payload.get("owner_session_id") or ""),
            process_birth_id=str(payload.get("process_birth_id") or ""),
            fencing_token=int(payload.get("fencing_token") or 0),
            fence_epoch=int(payload.get("fence_epoch") or 0),
            acquired_at=str(payload.get("acquired_at") or ""),
            expires_at=str(payload.get("expires_at") or ""),
            state=str(payload.get("state") or "held"),
            store_id=str(payload.get("store_id") or DEFAULT_STORE_ID),
            purpose=str(payload.get("purpose") or MAINTENANCE_SCOPE_IMPORT),
            revision=int(payload.get("revision") or 0),
        )


@dataclass(frozen=True)
class ConformanceReport:
    """Canonical result of the shared repository conformance population."""

    SCHEMA: ClassVar[str] = CONFORMANCE_REPORT_SCHEMA

    authority_mode: str
    store_id: str
    evidence: Mapping[str, Any]
    population_digest: str
    passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "authority_mode": self.authority_mode,
            "store_id": self.store_id,
            "evidence": dict(self.evidence),
            "population_digest": self.population_digest,
            "passed": self.passed,
            "evidence_subset": list(CONFORMANCE_EVIDENCE_SUBSET),
        }


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class StateRepository(Protocol):
    """Path-independent control-plane state repository.

    Interface: ``StateRepository@1``.
    """

    INTERFACE: ClassVar[str]
    SCHEMA: ClassVar[str]

    @property
    def authority_mode(self) -> RepositoryAuthorityMode: ...

    @property
    def store_id(self) -> str: ...

    @property
    def owner_id(self) -> str: ...

    @property
    def open_session(self) -> bool: ...

    @property
    def session(self) -> ClientSession | None: ...

    def open(self) -> ClientSession: ...

    def close(self) -> None: ...

    def load_generation(self) -> StoreGeneration: ...

    def observe_store_identity(
        self,
        *,
        repository_id: str = DEFAULT_REPOSITORY_ID,
    ) -> ControlPlaneStoreIdentity: ...

    def verify_schema(self) -> Mapping[str, Any]: ...

    def get_task(self, task_cid: str) -> Mapping[str, Any] | None: ...

    def list_tasks(
        self,
        *,
        cursor: int = 0,
        limit: int = 50,
    ) -> PageResult: ...

    def submit_command(
        self,
        command: StateCommand,
        *,
        apply: Callable[
            [StateTransaction, StateCommand, StoreGeneration], Mapping[str, Any]
        ]
        | None = None,
    ) -> CASResult: ...

    def cas_task_status(
        self,
        *,
        task_cid: str,
        expected_task_revision: int,
        new_status: str,
        idempotency_key: str,
        command_id: str | None = None,
    ) -> CASResult: ...

    def transaction(
        self,
        *,
        expected_generation: StoreGeneration | None = None,
    ) -> StateTransaction: ...

    def capture_snapshot(self) -> StateSnapshot: ...

    def append_event(
        self,
        *,
        event_type: str,
        stream_id: str,
        body: Mapping[str, Any] | None = None,
        task_cid: str = "",
    ) -> Mapping[str, Any]: ...

    def list_events(
        self,
        *,
        stream_id: str,
        after_sequence: int = 0,
        limit: int = 50,
    ) -> tuple[Mapping[str, Any], ...]: ...

    def upsert_task_lease(
        self,
        *,
        task_cid: str,
        claim_cid: str,
        claimant_did: str,
        fencing_token: int,
        state: str = "held",
        expires_at_ms: int = 0,
    ) -> Mapping[str, Any]: ...

    def get_task_lease(self, task_cid: str) -> Mapping[str, Any] | None: ...

    def acquire_maintenance_lease(
        self,
        *,
        scope: str,
        purpose: str = MAINTENANCE_SCOPE_IMPORT,
        ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
    ) -> MaintenanceLease: ...

    def release_maintenance_lease(self, lease: MaintenanceLease) -> MaintenanceLease: ...

    def client(self) -> QuackStateClient: ...


# ---------------------------------------------------------------------------
# Extra statement templates owned by the repository facade
# ---------------------------------------------------------------------------


def _repository_templates() -> dict[str, StatementTemplate]:
    return {
        "insert_domain_event": StatementTemplate(
            name="insert_domain_event",
            sql=(
                "INSERT INTO domain_events ("
                "event_id, stream_id, sequence, global_sequence, event_type, "
                "task_cid, attempt_id, session_id, recorded_at, body_json"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            parameter_names=(
                "event_id",
                "stream_id",
                "sequence",
                "global_sequence",
                "event_type",
                "task_cid",
                "attempt_id",
                "session_id",
                "recorded_at",
                "body_json",
            ),
            kind=StatementKind.MUTATION,
            description="Append one domain event",
        ),
        "list_domain_events_page": StatementTemplate(
            name="list_domain_events_page",
            sql=(
                "SELECT event_id, stream_id, sequence, global_sequence, "
                "event_type, task_cid, recorded_at, body_json "
                "FROM domain_events WHERE stream_id = ? AND sequence > ? "
                "ORDER BY sequence ASC LIMIT ?"
            ),
            parameter_names=("stream_id", "after_sequence", "limit"),
            kind=StatementKind.QUERY,
            description="Cursor page of domain events for a stream",
        ),
        "max_event_sequence": StatementTemplate(
            name="max_event_sequence",
            sql=(
                "SELECT COALESCE(MAX(global_sequence), 0) AS watermark "
                "FROM domain_events"
            ),
            parameter_names=(),
            kind=StatementKind.QUERY,
            description="Global domain-event watermark",
        ),
        "max_stream_sequence": StatementTemplate(
            name="max_stream_sequence",
            sql=(
                "SELECT COALESCE(MAX(sequence), 0) AS sequence "
                "FROM domain_events WHERE stream_id = ?"
            ),
            parameter_names=("stream_id",),
            kind=StatementKind.QUERY,
            description="Per-stream domain-event sequence head",
        ),
        "upsert_task_lease": StatementTemplate(
            name="upsert_task_lease",
            sql=(
                "INSERT INTO leases ("
                "task_cid, claim_cid, resolution_cid, claimant_did, "
                "logical_epoch, fencing_token, expires_at_ms, attempt, state, "
                "started_at_ms, release_reason, retry_not_before_ms, "
                "owner_session_id, fence_epoch, revision"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (task_cid) DO UPDATE SET "
                "claim_cid = excluded.claim_cid, "
                "claimant_did = excluded.claimant_did, "
                "fencing_token = excluded.fencing_token, "
                "state = excluded.state, "
                "expires_at_ms = excluded.expires_at_ms, "
                "owner_session_id = excluded.owner_session_id, "
                "fence_epoch = excluded.fence_epoch, "
                "revision = leases.revision + 1"
            ),
            parameter_names=(
                "task_cid",
                "claim_cid",
                "resolution_cid",
                "claimant_did",
                "logical_epoch",
                "fencing_token",
                "expires_at_ms",
                "attempt",
                "state",
                "started_at_ms",
                "release_reason",
                "retry_not_before_ms",
                "owner_session_id",
                "fence_epoch",
                "revision",
            ),
            kind=StatementKind.MUTATION,
            description="Upsert a task lease row",
        ),
        "select_task_lease": StatementTemplate(
            name="select_task_lease",
            sql=(
                "SELECT task_cid, claim_cid, claimant_did, fencing_token, "
                "state, expires_at_ms, owner_session_id, fence_epoch, revision "
                "FROM leases WHERE task_cid = ? LIMIT 1"
            ),
            parameter_names=("task_cid",),
            kind=StatementKind.QUERY,
            description="Fetch one task lease by task_cid",
        ),
        "insert_maintenance_lease": StatementTemplate(
            name="insert_maintenance_lease",
            sql=(
                "INSERT INTO maintenance_leases ("
                "lease_id, scope, owner_session_id, process_birth_id, "
                "fencing_token, fence_epoch, acquired_at, expires_at, "
                "released_at, state, revision"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            parameter_names=(
                "lease_id",
                "scope",
                "owner_session_id",
                "process_birth_id",
                "fencing_token",
                "fence_epoch",
                "acquired_at",
                "expires_at",
                "released_at",
                "state",
                "revision",
            ),
            kind=StatementKind.MUTATION,
            description="Record a maintenance lease grant",
        ),
        "release_maintenance_lease": StatementTemplate(
            name="release_maintenance_lease",
            sql=(
                "UPDATE maintenance_leases SET state = ?, released_at = ?, "
                "revision = revision + 1 "
                "WHERE lease_id = ? AND fencing_token = ? AND state = 'held'"
            ),
            parameter_names=(
                "state",
                "released_at",
                "lease_id",
                "fencing_token",
            ),
            kind=StatementKind.MUTATION,
            description="Release a held maintenance lease",
        ),
        "select_maintenance_lease": StatementTemplate(
            name="select_maintenance_lease",
            sql=(
                "SELECT lease_id, scope, owner_session_id, process_birth_id, "
                "fencing_token, fence_epoch, acquired_at, expires_at, "
                "released_at, state, revision "
                "FROM maintenance_leases WHERE lease_id = ? LIMIT 1"
            ),
            parameter_names=("lease_id",),
            kind=StatementKind.QUERY,
            description="Fetch one maintenance lease",
        ),
    }


REPOSITORY_STATEMENT_TEMPLATES: Final[Mapping[str, StatementTemplate]] = (
    MappingProxyType(_repository_templates())
)


def _register_repository_templates(client: QuackStateClient) -> None:
    for template in REPOSITORY_STATEMENT_TEMPLATES.values():
        client.register_template(template)


# ---------------------------------------------------------------------------
# Shared client-backed implementation
# ---------------------------------------------------------------------------


class _ClientBackedStateRepository:
    """Shared path-independent operations over ``QuackStateClient``."""

    INTERFACE: ClassVar[str] = STATE_REPOSITORY_INTERFACE
    SCHEMA: ClassVar[str] = STATE_REPOSITORY_SCHEMA
    VERSION: ClassVar[int] = STATE_REPOSITORY_VERSION

    def __init__(
        self,
        *,
        owner_id: str,
        store_id: str = DEFAULT_STORE_ID,
        repository_id: str = DEFAULT_REPOSITORY_ID,
        authority_mode: RepositoryAuthorityMode,
        retry_policy: RetryPolicy | None = None,
        clock: Callable[[], str] | None = None,
        connection_factory: Callable[[QuackEndpoint], Any] | None = None,
        expected_identity: ControlPlaneStoreIdentity | None = None,
    ) -> None:
        owner = _text(owner_id, "owner_id")
        self._owner_id = owner
        self._store_id = _text(store_id, "store_id")
        self._repository_id = _text(repository_id, "repository_id")
        if not isinstance(authority_mode, RepositoryAuthorityMode):
            authority_mode = RepositoryAuthorityMode(str(authority_mode))
        self._authority_mode = authority_mode
        self._clock = clock or _utc_now
        self._retry_policy = retry_policy
        self._connection_factory = connection_factory
        self._expected_identity = expected_identity
        self._client: QuackStateClient | None = None
        self._lock = threading.RLock()
        self._event_global_sequence = 0
        self._maintenance_fencing = 0
        self._closed = False

    # -- identity ------------------------------------------------------------

    @property
    def authority_mode(self) -> RepositoryAuthorityMode:
        return self._authority_mode

    @property
    def store_id(self) -> str:
        return self._store_id

    @property
    def owner_id(self) -> str:
        return self._owner_id

    @property
    def repository_id(self) -> str:
        return self._repository_id

    @property
    def open_session(self) -> bool:
        client = self._client
        return client is not None and client.attached and not self._closed

    @property
    def session(self) -> ClientSession | None:
        if self._client is None:
            return None
        return self._client.session

    def client(self) -> QuackStateClient:
        if self._client is None or not self._client.attached:
            raise StateRepositoryNotOpenError("repository is not open")
        return self._client

    # -- lifecycle -----------------------------------------------------------

    def _build_client(self) -> QuackStateClient:
        client = QuackStateClient(
            owner_id=self._owner_id,
            store_id=self._store_id,
            expected_identity=self._expected_identity,
            retry_policy=self._retry_policy,
            clock=self._clock,
            connection_factory=self._connection_factory,
        )
        _register_repository_templates(client)
        return client

    def close(self) -> None:
        with self._lock:
            self._closed = True
            client = self._client
            self._client = None
            if client is not None:
                client.close()

    def __enter__(self) -> "_ClientBackedStateRepository":
        if not self.open_session:
            self.open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def open(self) -> ClientSession:  # pragma: no cover - abstract in subclasses
        raise NotImplementedError

    # -- reads / writes ------------------------------------------------------

    def load_generation(self) -> StoreGeneration:
        return self.client().load_generation()

    def observe_store_identity(
        self,
        *,
        repository_id: str = DEFAULT_REPOSITORY_ID,
    ) -> ControlPlaneStoreIdentity:
        client = self.client()
        generation = client.load_generation()
        session = client.session
        # Prefer the identity observed at attach when available.
        if session is not None and session.store_identity is not None:
            observed = session.store_identity
            if repository_id and repository_id != observed.repository_id:
                return ControlPlaneStoreIdentity(
                    repository_id=repository_id,
                    database_uuid=observed.database_uuid,
                    store_id=observed.store_id,
                    schema_revision=observed.schema_revision,
                    generation=generation.generation,
                    schema_fingerprint=observed.schema_fingerprint,
                    authority_class=StateAuthorityClass.AUTHORITATIVE,
                    server_birth_id=observed.server_birth_id,
                    extension_fingerprint=observed.extension_fingerprint,
                )
            return observed
        # Rebuild from generation + metadata templates.
        rows = client.execute("whoami_metadata")
        meta = {str(row["key"]): str(row["value"]) for row in rows}
        fingerprint = meta.get("schema_fingerprint") or (
            "sha256:" + ("00" * 32)
        )
        if not str(fingerprint).startswith("sha256:"):
            fingerprint = "sha256:" + hashlib.sha256(
                str(fingerprint).encode("utf-8")
            ).hexdigest()
        return ControlPlaneStoreIdentity(
            repository_id=repository_id or self._repository_id,
            database_uuid=generation.database_uuid or meta.get(
                "database_uuid", "00000000-0000-4000-8000-000000000000"
            ),
            store_id=self._store_id,
            schema_revision=generation.schema_revision,
            generation=generation.generation,
            schema_fingerprint=str(fingerprint),
            authority_class=StateAuthorityClass.AUTHORITATIVE,
            server_birth_id=generation.birth_id,
        )

    def verify_schema(self) -> Mapping[str, Any]:
        """Verify schema inventory against the open store.

        Embedded adapters verify via the on-disk path. Quack adapters verify
        by requiring the schema inventory tables to be queryable through the
        typed client (path-independent) and returning a structured receipt.
        """

        client = self.client()
        # Path-independent probe: generation + task count templates must work.
        generation = client.load_generation()
        count_rows = client.execute("count_tasks")
        task_count = int(count_rows[0]["task_count"]) if count_rows else 0
        return {
            "authority_mode": self._authority_mode.value,
            "store_id": self._store_id,
            "generation": generation.generation,
            "schema_revision": generation.schema_revision,
            "task_count": task_count,
            "verified": True,
        }

    def get_task(self, task_cid: str) -> Mapping[str, Any] | None:
        cid = _text(task_cid, "task_cid")
        rows = self.client().execute("select_task_by_cid", {"task_cid": cid})
        if not rows:
            return None
        return dict(rows[0])

    def list_tasks(
        self,
        *,
        cursor: int = 0,
        limit: int = 50,
    ) -> PageResult:
        return self.client().paginate(cursor=cursor, limit=limit)

    def submit_command(
        self,
        command: StateCommand,
        *,
        apply: Callable[
            [StateTransaction, StateCommand, StoreGeneration], Mapping[str, Any]
        ]
        | None = None,
    ) -> CASResult:
        return self.client().submit_command(command, apply=apply)

    def cas_task_status(
        self,
        *,
        task_cid: str,
        expected_task_revision: int,
        new_status: str,
        idempotency_key: str,
        command_id: str | None = None,
    ) -> CASResult:
        return self.client().cas_task_status(
            task_cid=task_cid,
            expected_task_revision=expected_task_revision,
            new_status=new_status,
            idempotency_key=idempotency_key,
            command_id=command_id,
        )

    def transaction(
        self,
        *,
        expected_generation: StoreGeneration | None = None,
    ) -> StateTransaction:
        return self.client().transaction(expected_generation=expected_generation)

    def capture_snapshot(self) -> StateSnapshot:
        client = self.client()
        generation = client.load_generation()
        watermark_rows = client.execute("max_event_sequence")
        watermark = 0
        if watermark_rows:
            watermark = int(watermark_rows[0].get("watermark") or 0)
        tasks = client.paginate(cursor=0, limit=500)
        # Digest is authority-mode independent so Local and Quack adapters
        # produce identical snapshot digests for the same population.
        payload = {
            "store_id": self._store_id,
            "generation": generation.generation,
            "revision": generation.revision,
            "fence_epoch": generation.fence_epoch,
            "schema_revision": generation.schema_revision,
            "event_watermark": watermark,
            "tasks": [dict(item) for item in tasks.items],
        }
        digest = _digest_bytes(payload)
        return StateSnapshot(
            snapshot_id=_new_id("snapshot"),
            store_id=self._store_id,
            database_uuid=generation.database_uuid
            or "00000000-0000-4000-8000-000000000000",
            generation=generation.generation,
            schema_revision=generation.schema_revision,
            revision=generation.revision,
            fence_epoch=generation.fence_epoch,
            event_watermark=watermark,
            snapshot_digest=digest,
            authority_class=StateAuthorityClass.AUTHORITATIVE,
        )

    def append_event(
        self,
        *,
        event_type: str,
        stream_id: str,
        body: Mapping[str, Any] | None = None,
        task_cid: str = "",
    ) -> Mapping[str, Any]:
        client = self.client()
        session = client.session
        stream = _text(stream_id, "stream_id")
        kind = _text(event_type, "event_type")
        head_rows = client.execute(
            "max_stream_sequence", {"stream_id": stream}
        )
        sequence = int(head_rows[0]["sequence"]) + 1 if head_rows else 1
        global_rows = client.execute("max_event_sequence")
        global_sequence = (
            int(global_rows[0]["watermark"]) + 1 if global_rows else 1
        )
        event_id = _new_id("event")
        recorded_at = self._clock()
        body_json = json.dumps(
            dict(body or {}),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        client.execute(
            "insert_domain_event",
            {
                "event_id": event_id,
                "stream_id": stream,
                "sequence": sequence,
                "global_sequence": global_sequence,
                "event_type": kind,
                "task_cid": task_cid or "",
                "attempt_id": "",
                "session_id": "" if session is None else session.session_id,
                "recorded_at": recorded_at,
                "body_json": body_json,
            },
        )
        return {
            "event_id": event_id,
            "stream_id": stream,
            "sequence": sequence,
            "global_sequence": global_sequence,
            "event_type": kind,
            "task_cid": task_cid or "",
            "recorded_at": recorded_at,
        }

    def list_events(
        self,
        *,
        stream_id: str,
        after_sequence: int = 0,
        limit: int = 50,
    ) -> tuple[Mapping[str, Any], ...]:
        stream = _text(stream_id, "stream_id")
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int):
            raise StateRepositoryError("after_sequence must be an integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise StateRepositoryError("limit must be a positive integer")
        rows = self.client().execute(
            "list_domain_events_page",
            {
                "stream_id": stream,
                "after_sequence": after_sequence,
                "limit": limit,
            },
        )
        return tuple(dict(row) for row in rows)

    def upsert_task_lease(
        self,
        *,
        task_cid: str,
        claim_cid: str,
        claimant_did: str,
        fencing_token: int,
        state: str = "held",
        expires_at_ms: int = 0,
    ) -> Mapping[str, Any]:
        client = self.client()
        session = client.session
        generation = client.load_generation()
        cid = _text(task_cid, "task_cid")
        claim = _text(claim_cid, "claim_cid")
        claimant = _text(claimant_did, "claimant_did")
        if isinstance(fencing_token, bool) or not isinstance(fencing_token, int):
            raise StateRepositoryError("fencing_token must be an integer")
        now_ms = int(time.time() * 1000)
        client.execute(
            "upsert_task_lease",
            {
                "task_cid": cid,
                "claim_cid": claim,
                "resolution_cid": "",
                "claimant_did": claimant,
                "logical_epoch": generation.generation,
                "fencing_token": fencing_token,
                "expires_at_ms": int(expires_at_ms or (now_ms + 60_000)),
                "attempt": 1,
                "state": _text(state, "state"),
                "started_at_ms": now_ms,
                "release_reason": "",
                "retry_not_before_ms": 0,
                "owner_session_id": (
                    "" if session is None else session.session_id
                ),
                "fence_epoch": generation.fence_epoch,
                "revision": 0,
            },
        )
        lease = self.get_task_lease(cid)
        return dict(lease or {"task_cid": cid, "state": state})

    def get_task_lease(self, task_cid: str) -> Mapping[str, Any] | None:
        cid = _text(task_cid, "task_cid")
        rows = self.client().execute("select_task_lease", {"task_cid": cid})
        if not rows:
            return None
        return dict(rows[0])

    def acquire_maintenance_lease(
        self,
        *,
        scope: str,
        purpose: str = MAINTENANCE_SCOPE_IMPORT,
        ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
    ) -> MaintenanceLease:
        client = self.client()
        session = client.session
        if session is None:
            raise StateRepositoryNotOpenError(
                "cannot acquire maintenance lease without an open session"
            )
        generation = client.load_generation()
        with self._lock:
            self._maintenance_fencing += 1
            fencing_token = self._maintenance_fencing
        acquired_at = self._clock()
        # expires_at is wall-clock ISO; for hermetic clocks that return a fixed
        # stamp, ttl still advances the lexical suffix when using real time.
        if self._clock is _utc_now:
            expires_at = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(time.time() + max(1, int(ttl_seconds))),
            )
        else:
            # Fixed test clocks: encode ttl in the lease id side-channel by
            # treating the lease as held until explicitly released.
            expires_at = "9999-12-31T23:59:59Z"
        lease = MaintenanceLease(
            lease_id=_new_id("mlease"),
            scope=_text(scope, "scope"),
            owner_session_id=session.session_id,
            process_birth_id=session.process_birth_id,
            fencing_token=fencing_token,
            fence_epoch=generation.fence_epoch,
            acquired_at=acquired_at,
            expires_at=expires_at,
            state="held",
            store_id=self._store_id,
            purpose=_text(purpose, "purpose"),
            revision=0,
        )
        client.execute(
            "insert_maintenance_lease",
            {
                "lease_id": lease.lease_id,
                "scope": lease.scope,
                "owner_session_id": lease.owner_session_id,
                "process_birth_id": lease.process_birth_id,
                "fencing_token": lease.fencing_token,
                "fence_epoch": lease.fence_epoch,
                "acquired_at": lease.acquired_at,
                "expires_at": lease.expires_at,
                "released_at": "",
                "state": lease.state,
                "revision": lease.revision,
            },
        )
        return lease

    def release_maintenance_lease(
        self, lease: MaintenanceLease
    ) -> MaintenanceLease:
        if not isinstance(lease, MaintenanceLease):
            raise StateRepositoryLeaseError("lease must be a MaintenanceLease")
        client = self.client()
        released_at = self._clock()
        client.execute(
            "release_maintenance_lease",
            {
                "state": "released",
                "released_at": released_at,
                "lease_id": lease.lease_id,
                "fencing_token": lease.fencing_token,
            },
        )
        return MaintenanceLease(
            lease_id=lease.lease_id,
            scope=lease.scope,
            owner_session_id=lease.owner_session_id,
            process_birth_id=lease.process_birth_id,
            fencing_token=lease.fencing_token,
            fence_epoch=lease.fence_epoch,
            acquired_at=lease.acquired_at,
            expires_at=lease.expires_at,
            state="released",
            store_id=lease.store_id,
            purpose=lease.purpose,
            revision=lease.revision + 1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "authority_mode": self._authority_mode.value,
            "store_id": self._store_id,
            "owner_id": self._owner_id,
            "repository_id": self._repository_id,
            "open": self.open_session,
        }


# ---------------------------------------------------------------------------
# Embedded adapter
# ---------------------------------------------------------------------------


class EmbeddedStateRepository(_ClientBackedStateRepository):
    """Exclusive local-file repository for tests and maintenance imports.

    Interface: ``EmbeddedStateRepository@1``.

    Production authority is Quack. Embedded exclusive opens for import,
    recovery, or maintenance require a held ``MaintenanceLease``. Hermetic
    tests may open without a lease under ``EmbeddedOpenPurpose.HERMETIC_TEST``.
    """

    INTERFACE: ClassVar[str] = EMBEDDED_STATE_REPOSITORY_INTERFACE
    SCHEMA: ClassVar[str] = EMBEDDED_STATE_REPOSITORY_SCHEMA

    def __init__(
        self,
        database_path: str | Path,
        *,
        owner_id: str,
        store_id: str = DEFAULT_STORE_ID,
        repository_id: str = DEFAULT_REPOSITORY_ID,
        purpose: EmbeddedOpenPurpose | str = EmbeddedOpenPurpose.HERMETIC_TEST,
        maintenance_lease: MaintenanceLease | None = None,
        install_schema: bool = False,
        seed_generation: bool = True,
        retry_policy: RetryPolicy | None = None,
        clock: Callable[[], str] | None = None,
        expected_identity: ControlPlaneStoreIdentity | None = None,
        application_version: str | None = None,
        tool_version: str | None = None,
    ) -> None:
        super().__init__(
            owner_id=owner_id,
            store_id=store_id,
            repository_id=repository_id,
            authority_mode=RepositoryAuthorityMode.EMBEDDED_EXCLUSIVE,
            retry_policy=retry_policy,
            clock=clock,
            expected_identity=expected_identity,
        )
        self._database_path = Path(database_path)
        if isinstance(purpose, EmbeddedOpenPurpose):
            self._purpose = purpose
        else:
            self._purpose = EmbeddedOpenPurpose(str(purpose))
        self._maintenance_lease = maintenance_lease
        self._install_schema = bool(install_schema)
        self._seed_generation = bool(seed_generation)
        self._application_version = application_version
        self._tool_version = tool_version
        self._file_lock_cm: Any | None = None

    @property
    def database_path(self) -> Path:
        return self._database_path

    @property
    def purpose(self) -> EmbeddedOpenPurpose:
        return self._purpose

    @property
    def maintenance_lease(self) -> MaintenanceLease | None:
        return self._maintenance_lease

    def _require_lease_for_purpose(self) -> None:
        if self._purpose not in _LEASE_REQUIRED_PURPOSES:
            return
        if self._maintenance_lease is None:
            raise StateRepositoryAuthorityError(
                "embedded exclusive open for "
                f"{self._purpose.value} requires a held maintenance lease"
            )
        self._maintenance_lease.require_held(now=self._clock())
        if self._maintenance_lease.store_id not in ("", self._store_id):
            raise StateRepositoryLeaseError(
                "maintenance lease store_id does not match repository store_id"
            )
        expected_purpose = self._purpose.value
        if self._maintenance_lease.purpose not in (
            expected_purpose,
            MAINTENANCE_SCOPE_IMPORT,
            MAINTENANCE_SCOPE_RECOVERY,
            "maintenance",
        ):
            # Accept the closed maintenance purposes interchangeably when the
            # operator grants a general maintenance lease.
            if self._maintenance_lease.purpose not in {
                p.value for p in _LEASE_REQUIRED_PURPOSES
            }:
                raise StateRepositoryLeaseError(
                    "maintenance lease purpose is not valid for embedded open"
                )

    def open(self) -> ClientSession:
        with self._lock:
            if self._closed:
                raise StateRepositoryError("repository is closed")
            if self.open_session:
                session = self.session
                if session is None:
                    raise StateRepositoryError("repository session missing")
                return session
            self._require_lease_for_purpose()
            path = self._database_path
            path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive process lock proves single-writer embedded access.
            lock_path = path.with_suffix(path.suffix + ".repo.lock")
            self._file_lock_cm = exclusive_file_lock(lock_path)
            self._file_lock_cm.__enter__()
            try:
                if self._install_schema or not path.exists():
                    install_control_plane_schema(
                        path,
                        application_version=self._application_version,
                        tool_version=self._tool_version,
                        owner_id=self._owner_id,
                    )
                client = self._build_client()
                client.attach(
                    path,
                    mode=TransportMode.EMBEDDED,
                    seed_generation=self._seed_generation,
                    expected_identity=self._expected_identity,
                    server_id="server:embedded",
                )
            except Exception:
                try:
                    self._file_lock_cm.__exit__(None, None, None)
                finally:
                    self._file_lock_cm = None
                raise
            self._client = client
            session = client.session
            if session is None:
                raise StateRepositoryError("embedded attach produced no session")
            return session

    def close(self) -> None:
        with self._lock:
            super().close()
            if self._file_lock_cm is not None:
                try:
                    self._file_lock_cm.__exit__(None, None, None)
                finally:
                    self._file_lock_cm = None

    def verify_schema(self) -> Mapping[str, Any]:
        """Verify schema without opening a second writer connection.

        While the repository holds the exclusive embedded session, inventory
        checks run through the attached client. Full ``verify_installed_schema``
        path scans are available only when the repository is closed so DuckDB's
        single-writer rule is not violated.
        """

        if not self._database_path.exists():
            raise StateRepositoryError(
                f"database does not exist: {self._database_path}"
            )
        if self.open_session:
            base = super().verify_schema()
            return {
                **dict(base),
                "path": str(self._database_path),
                "inventory": "client_probe",
            }
        report = verify_installed_schema(self._database_path)
        return {
            **dict(report),
            "path": str(self._database_path),
            "verified": True,
            "authority_mode": self._authority_mode.value,
            "store_id": self._store_id,
            "inventory": "path_scan",
        }

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "database_path": str(self._database_path),
                "purpose": self._purpose.value,
                "maintenance_lease_id": (
                    None
                    if self._maintenance_lease is None
                    else self._maintenance_lease.lease_id
                ),
            }
        )
        return payload


# ---------------------------------------------------------------------------
# Quack adapter
# ---------------------------------------------------------------------------


class QuackStateRepository(_ClientBackedStateRepository):
    """Production state repository over Quack transport.

    Interface: ``QuackStateRepository@1``.

    Authority is path-independent: callers supply a ``quack:`` URI (or a
    resolved ``QuackEndpoint`` with ``TransportMode.QUACK``). Direct database
    file paths are rejected. Transport failures never fall back to embedded
    file writes.
    """

    INTERFACE: ClassVar[str] = QUACK_STATE_REPOSITORY_INTERFACE
    SCHEMA: ClassVar[str] = QUACK_STATE_REPOSITORY_SCHEMA

    def __init__(
        self,
        endpoint: str | QuackEndpoint,
        *,
        owner_id: str,
        store_id: str = DEFAULT_STORE_ID,
        repository_id: str = DEFAULT_REPOSITORY_ID,
        secret_handle: str = "",
        server_id: str = "server:quack",
        seed_generation: bool = False,
        retry_policy: RetryPolicy | None = None,
        clock: Callable[[], str] | None = None,
        connection_factory: Callable[[QuackEndpoint], Any] | None = None,
        expected_identity: ControlPlaneStoreIdentity | None = None,
        allow_embedded_fallback: bool = False,
    ) -> None:
        if allow_embedded_fallback:
            # Explicitly rejected: the flag exists only so misconfigured callers
            # fail closed with a clear authority error rather than silently
            # writing files.
            raise StateRepositoryAuthorityError(
                "Quack authority never permits embedded file-write fallback"
            )
        super().__init__(
            owner_id=owner_id,
            store_id=store_id,
            repository_id=repository_id,
            authority_mode=RepositoryAuthorityMode.QUACK,
            retry_policy=retry_policy,
            clock=clock,
            connection_factory=connection_factory,
            expected_identity=expected_identity,
        )
        self._endpoint = self._coerce_quack_endpoint(endpoint, secret_handle)
        self._secret_handle = secret_handle
        self._server_id = _text(server_id, "server_id")
        self._seed_generation = bool(seed_generation)

    @staticmethod
    def _coerce_quack_endpoint(
        endpoint: str | Path | QuackEndpoint,
        secret_handle: str,
    ) -> QuackEndpoint:
        if isinstance(endpoint, QuackEndpoint):
            if endpoint.mode is not TransportMode.QUACK:
                raise StateRepositoryAuthorityError(
                    "QuackStateRepository refuses non-quack endpoints; "
                    "direct file authority is not permitted"
                )
            return endpoint
        text = str(endpoint).strip()
        if not text:
            raise StateRepositoryError("quack endpoint is required")
        # Refuse bare filesystem paths even if they happen to exist.
        if not text.startswith("quack:"):
            raise StateRepositoryAuthorityError(
                "QuackStateRepository requires a quack: URI; "
                "refusing direct file path authority "
                f"(got {text!r})"
            )
        resolved = resolve_endpoint(
            text, mode=TransportMode.QUACK, secret_handle=secret_handle
        )
        if resolved.mode is not TransportMode.QUACK:
            raise StateRepositoryAuthorityError(
                "Quack endpoint resolution produced a non-quack mode"
            )
        return resolved

    @property
    def endpoint(self) -> QuackEndpoint:
        return self._endpoint

    def open(self) -> ClientSession:
        with self._lock:
            if self._closed:
                raise StateRepositoryError("repository is closed")
            if self.open_session:
                session = self.session
                if session is None:
                    raise StateRepositoryError("repository session missing")
                return session
            # Re-assert authority: never attach as embedded.
            if self._endpoint.mode is not TransportMode.QUACK:
                raise StateRepositoryAuthorityError(
                    "Quack repository endpoint mode drifted away from quack"
                )
            if self._endpoint.database_path is not None and (
                self._connection_factory is None
            ):
                # A database_path on a quack endpoint without an explicit test
                # double would imply silent file open — refuse.
                raise StateRepositoryAuthorityError(
                    "Quack repository refuses endpoints that carry a direct "
                    "database path without an explicit connection factory"
                )
            client = self._build_client()
            # Wrap connection factory to forbid embedded fallback on failure.
            original_factory = self._connection_factory

            def _factory_guard(endpoint: QuackEndpoint) -> Any:
                if endpoint.mode is not TransportMode.QUACK:
                    raise StateRepositoryAuthorityError(
                        "Quack repository connection factory received a "
                        "non-quack endpoint; refusing file fallback"
                    )
                if original_factory is None:
                    # Real Quack transport path (may raise transport errors).
                    # QuackStateClient opens the connection itself when factory
                    # is None; this guard is only used when we inject one.
                    raise StateRepositoryError(
                        "internal: factory guard invoked without factory"
                    )
                try:
                    return original_factory(endpoint)
                except StateRepositoryAuthorityError:
                    raise
                except Exception as exc:
                    # Never retry as embedded file open.
                    raise QuackClientTransportError(
                        "Quack transport failed without file fallback: "
                        f"{exc}"
                    ) from exc

            if original_factory is not None:
                client._connection_factory = _factory_guard  # noqa: SLF001
            try:
                client.attach(
                    self._endpoint,
                    mode=TransportMode.QUACK,
                    secret_handle=self._secret_handle,
                    server_id=self._server_id,
                    seed_generation=self._seed_generation,
                    expected_identity=self._expected_identity,
                )
            except QuackClientTransportError:
                # Surface transport failure; do not open the database path.
                client.close()
                raise
            except QuackClientError:
                client.close()
                raise
            except Exception as exc:
                client.close()
                raise QuackClientTransportError(
                    f"Quack attach failed without file fallback: {exc}"
                ) from exc
            # Final authority check on the live session.
            session = client.session
            if session is None:
                client.close()
                raise StateRepositoryError("quack attach produced no session")
            if session.transport_mode is not TransportMode.QUACK:
                client.close()
                raise StateRepositoryAuthorityError(
                    "Quack repository session is not in quack transport mode; "
                    "refusing to continue (possible silent fallback)"
                )
            self._client = client
            return session

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "endpoint": self._endpoint.target,
                "transport_mode": self._endpoint.mode.value,
                "server_id": self._server_id,
                "allows_embedded_fallback": False,
            }
        )
        return payload


# ---------------------------------------------------------------------------
# Maintenance lease helpers (pre-open, for import admission)
# ---------------------------------------------------------------------------


def issue_maintenance_lease(
    *,
    scope: str,
    purpose: str = MAINTENANCE_SCOPE_IMPORT,
    owner_session_id: str | None = None,
    process_birth_id: str | None = None,
    store_id: str = DEFAULT_STORE_ID,
    fencing_token: int = 1,
    fence_epoch: int = 0,
    ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
    clock: Callable[[], str] | None = None,
) -> MaintenanceLease:
    """Issue an out-of-band maintenance lease for embedded exclusive opens.

    The lease proves operator intent for offline import/recovery before the
    repository opens the database file. After open, adapters also persist the
    lease row into ``maintenance_leases`` via ``acquire_maintenance_lease``.
    """

    now_fn = clock or _utc_now
    acquired_at = now_fn()
    if now_fn is _utc_now:
        expires_at = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(time.time() + max(1, int(ttl_seconds))),
        )
    else:
        expires_at = "9999-12-31T23:59:59Z"
    return MaintenanceLease(
        lease_id=_new_id("mlease"),
        scope=_text(scope, "scope"),
        owner_session_id=owner_session_id or _new_id("session"),
        process_birth_id=process_birth_id or _new_id("birth"),
        fencing_token=int(fencing_token),
        fence_epoch=int(fence_epoch),
        acquired_at=acquired_at,
        expires_at=expires_at,
        state="held",
        store_id=store_id,
        purpose=_text(purpose, "purpose"),
        revision=0,
    )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def open_embedded_repository(
    database_path: str | Path,
    *,
    owner_id: str,
    store_id: str = DEFAULT_STORE_ID,
    repository_id: str = DEFAULT_REPOSITORY_ID,
    purpose: EmbeddedOpenPurpose | str = EmbeddedOpenPurpose.HERMETIC_TEST,
    maintenance_lease: MaintenanceLease | None = None,
    install_schema: bool = True,
    seed_generation: bool = True,
    **kwargs: Any,
) -> EmbeddedStateRepository:
    """Open an embedded exclusive repository and return it attached."""

    repo = EmbeddedStateRepository(
        database_path,
        owner_id=owner_id,
        store_id=store_id,
        repository_id=repository_id,
        purpose=purpose,
        maintenance_lease=maintenance_lease,
        install_schema=install_schema,
        seed_generation=seed_generation,
        **kwargs,
    )
    repo.open()
    return repo


def open_quack_repository(
    endpoint: str | QuackEndpoint,
    *,
    owner_id: str,
    store_id: str = DEFAULT_STORE_ID,
    repository_id: str = DEFAULT_REPOSITORY_ID,
    connection_factory: Callable[[QuackEndpoint], Any] | None = None,
    seed_generation: bool = False,
    **kwargs: Any,
) -> QuackStateRepository:
    """Open a Quack authority repository and return it attached."""

    repo = QuackStateRepository(
        endpoint,
        owner_id=owner_id,
        store_id=store_id,
        repository_id=repository_id,
        connection_factory=connection_factory,
        seed_generation=seed_generation,
        **kwargs,
    )
    repo.open()
    return repo


def open_quack_repository_against_database(
    database_path: str | Path,
    *,
    owner_id: str,
    quack_uri: str = "quack:127.0.0.1:9",
    store_id: str = DEFAULT_STORE_ID,
    repository_id: str = DEFAULT_REPOSITORY_ID,
    install_schema: bool = True,
    seed_generation: bool = True,
    **kwargs: Any,
) -> QuackStateRepository:
    """Open a Quack-mode repository backed by an explicit test double.

    The connection factory is an *explicit* transport double that serves the
    prepared database for hermetic tests. This is not a silent fallback: the
    repository still requires a ``quack:`` URI and refuses path endpoints.
    """

    path = Path(database_path)
    if install_schema or not path.exists():
        install_control_plane_schema(
            path,
            owner_id=owner_id,
        )

    def factory(endpoint: QuackEndpoint) -> Any:
        if endpoint.mode is not TransportMode.QUACK:
            raise StateRepositoryAuthorityError(
                "test double refused non-quack endpoint"
            )
        if not str(endpoint.target).startswith("quack:"):
            raise StateRepositoryAuthorityError(
                "test double refused non-quack target"
            )
        return open_duckdb_connection(path)

    return open_quack_repository(
        quack_uri,
        owner_id=owner_id,
        store_id=store_id,
        repository_id=repository_id,
        connection_factory=factory,
        seed_generation=seed_generation,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Shared conformance population
# ---------------------------------------------------------------------------


def _seed_conformance_population(repo: StateRepository) -> dict[str, Any]:
    """Seed the closed conformance rows through the repository client only."""

    client = repo.client()
    now = "1970-01-01T00:00:00Z"
    # Goals / tasks via closed templates.
    try:
        client.execute(
            "insert_goal",
            {
                "goal_cid": "goal:conformance",
                "goal_alias": "G-CONFORM",
                "objective_id": "objective:conformance",
                "parent_goal_cid": "",
                "ordinal": 1,
                "title": "Conformance root",
                "status": "open",
                "created_at": now,
                "updated_at": now,
                "revision": 0,
                "body_json": "{}",
            },
        )
    except Exception:
        # Idempotent seed: goal may already exist.
        pass
    task_cids: list[str] = []
    for index in range(1, 3):
        task_cid = f"task:conformance:{index:03d}"
        task_cids.append(task_cid)
        try:
            client.execute(
                "insert_task",
                {
                    "task_cid": task_cid,
                    "task_alias": f"T-CONF-{index:03d}",
                    "goal_cid": "goal:conformance",
                    "plan_cid": "",
                    "objective_id": "objective:conformance",
                    "ordinal": index,
                    "status": "ready",
                    "revision": 0,
                    "priority": "P0",
                    "created_at": now,
                    "updated_at": now,
                    "identity_json": "{}",
                    "body_json": "{}",
                },
            )
        except Exception:
            pass
    return {"goal_cid": "goal:conformance", "task_cids": task_cids}


def run_conformance_population(
    repo: StateRepository,
    *,
    seed: bool = True,
) -> ConformanceReport:
    """Run the shared evidence population and return a canonical report.

    Evidence subset: tasks, events, leases, commands, snapshots, transactions,
    schema verification, cold imports (represented by authority-mode metadata).
    """

    if not repo.open_session:
        raise StateRepositoryNotOpenError(
            "conformance population requires an open repository"
        )
    evidence: dict[str, Any] = {
        "authority_mode": repo.authority_mode.value,
        "store_id": repo.store_id,
        "interface": getattr(repo, "INTERFACE", STATE_REPOSITORY_INTERFACE),
    }

    # Schema verification
    schema_report = dict(repo.verify_schema())
    evidence["schema_verification"] = {
        "verified": bool(schema_report.get("verified")),
        "schema_revision": schema_report.get("schema_revision"),
        "generation": schema_report.get("generation"),
    }

    # Cold-import policy is shared: imports need a maintenance lease, and Quack
    # authority never falls back to direct file writes. Mode-specific flags are
    # kept out of the canonical digest so adapters can still prove parity.
    evidence["cold_imports"] = {
        "import_requires_maintenance_lease": True,
        "quack_allows_file_fallback": False,
    }

    if seed:
        seed_receipt = _seed_conformance_population(repo)
    else:
        seed_receipt = {"goal_cid": "goal:conformance", "task_cids": []}
    evidence["seed"] = seed_receipt

    # Tasks
    page = repo.list_tasks(cursor=0, limit=50)
    task_items = [dict(item) for item in page.items]
    evidence["tasks"] = {
        "count": len(task_items),
        "items": [
            {
                "task_cid": item.get("task_cid"),
                "status": item.get("status"),
                "revision": item.get("revision"),
                "ordinal": item.get("ordinal"),
            }
            for item in task_items
        ],
    }

    # Commands / transactions (CAS task status when a seeded task exists)
    command_evidence: dict[str, Any]
    task_cids = [
        str(item.get("task_cid") or "")
        for item in task_items
        if item.get("task_cid")
    ]
    if task_cids:
        target = task_cids[0]
        live_task = repo.get_task(target) or {}
        expected_rev = int(live_task.get("revision") or 0)
        result = repo.cas_task_status(
            task_cid=target,
            expected_task_revision=expected_rev,
            new_status="claimed",
            idempotency_key=f"idem:conformance:{target}",
            command_id=f"cmd:conformance:{target}",
        )
        command_evidence = {
            "outcome": result.outcome.value
            if isinstance(result.outcome, CommandOutcome)
            else str(result.outcome),
            "changed": bool(result.changed),
            "task_cid": target,
            "revision": int(result.revision),
            "generation": int(result.generation),
        }
        # Refresh task evidence after CAS for stable post-command state.
        refreshed = repo.get_task(target) or {}
        evidence["tasks"]["post_command"] = {
            "task_cid": target,
            "status": refreshed.get("status"),
            "revision": refreshed.get("revision"),
        }
    else:
        command_evidence = {"outcome": "skipped", "changed": False}
    evidence["commands"] = command_evidence
    evidence["transactions"] = {
        "cas_outcome": command_evidence.get("outcome"),
        "generation": command_evidence.get("generation"),
    }

    # Events
    event = repo.append_event(
        event_type="conformance.tick",
        stream_id="stream:conformance",
        body={"phase": "population", "authority": repo.authority_mode.value},
        task_cid=task_cids[0] if task_cids else "",
    )
    events = repo.list_events(stream_id="stream:conformance", after_sequence=0)
    evidence["events"] = {
        "appended": {
            "event_type": event.get("event_type"),
            "stream_id": event.get("stream_id"),
            "sequence": event.get("sequence"),
        },
        "listed_count": len(events),
        "sequences": [int(item.get("sequence") or 0) for item in events],
    }

    # Leases (task lease + maintenance lease)
    if task_cids:
        lease_row = repo.upsert_task_lease(
            task_cid=task_cids[0],
            claim_cid="claim:conformance",
            claimant_did=repo.owner_id,
            fencing_token=1,
            state="held",
        )
        evidence["leases"] = {
            "task_cid": lease_row.get("task_cid"),
            "state": lease_row.get("state"),
            "fencing_token": lease_row.get("fencing_token"),
        }
    else:
        evidence["leases"] = {"task_cid": None, "state": None}
    mlease = repo.acquire_maintenance_lease(
        scope="conformance",
        purpose=MAINTENANCE_SCOPE_HERMETIC,
    )
    released = repo.release_maintenance_lease(mlease)
    evidence["maintenance_lease"] = {
        "lease_id": mlease.lease_id,
        "scope": mlease.scope,
        "acquired_state": mlease.state,
        "released_state": released.state,
        "fencing_token": mlease.fencing_token,
    }

    # Snapshot
    snapshot = repo.capture_snapshot()
    evidence["snapshots"] = {
        "generation": snapshot.generation,
        "revision": snapshot.revision,
        "fence_epoch": snapshot.fence_epoch,
        "schema_revision": snapshot.schema_revision,
        "event_watermark": snapshot.event_watermark,
        "snapshot_digest": snapshot.snapshot_digest,
    }

    # Strip adapter-specific non-canonical fields before digesting.
    canonical_evidence = {
        key: evidence[key]
        for key in (
            "schema_verification",
            "tasks",
            "commands",
            "transactions",
            "events",
            "leases",
            "maintenance_lease",
            "snapshots",
            "cold_imports",
        )
        if key in evidence
    }
    # Normalize cold_imports to the shared policy posture only.
    canonical_evidence["cold_imports"] = {
        "import_requires_maintenance_lease": True,
        "quack_allows_file_fallback": False,
    }
    # Maintenance lease ids are unique per run; keep only structural fields.
    canonical_evidence["maintenance_lease"] = {
        "scope": evidence["maintenance_lease"]["scope"],
        "acquired_state": evidence["maintenance_lease"]["acquired_state"],
        "released_state": evidence["maintenance_lease"]["released_state"],
    }
    # Snapshot digests embed task order which is stable; drop unique snapshot ids.
    population_digest = _digest_bytes(canonical_evidence)
    return ConformanceReport(
        authority_mode=repo.authority_mode.value,
        store_id=repo.store_id,
        evidence=MappingProxyType(canonical_evidence),
        population_digest=population_digest,
        passed=True,
    )


def assert_conformance_parity(
    left: ConformanceReport,
    right: ConformanceReport,
) -> None:
    """Fail closed when two adapters diverge on the shared population."""

    if left.population_digest != right.population_digest:
        raise StateRepositoryConformanceError(
            "conformance population digests diverged between adapters: "
            f"{left.authority_mode}={left.population_digest} "
            f"{right.authority_mode}={right.population_digest}"
        )
    if dict(left.evidence) != dict(right.evidence):
        raise StateRepositoryConformanceError(
            "conformance evidence mappings diverged between adapters"
        )


# ---------------------------------------------------------------------------
# Bridge note for existing DuckDB task projection (incremental, non-breaking)
# ---------------------------------------------------------------------------


def repository_authority_for_task_source(
    *,
    prefer_quack: bool = True,
) -> RepositoryAuthorityMode:
    """Return the default authority mode for higher task-source layers.

    Existing ``DuckDBTaskSource`` remains a derived projection. New control-
    plane consumers should prefer Quack authority and only use embedded
    exclusive mode under a maintenance lease.
    """

    if prefer_quack:
        return RepositoryAuthorityMode.QUACK
    return RepositoryAuthorityMode.EMBEDDED_EXCLUSIVE


__all__ = [
    "CONFORMANCE_EVIDENCE_SUBSET",
    "CONFORMANCE_REPORT_SCHEMA",
    "DEFAULT_REPOSITORY_ID",
    "EMBEDDED_STATE_REPOSITORY_INTERFACE",
    "EMBEDDED_STATE_REPOSITORY_SCHEMA",
    "MAINTENANCE_LEASE_SCHEMA",
    "MAINTENANCE_SCOPE_HERMETIC",
    "MAINTENANCE_SCOPE_IMPORT",
    "MAINTENANCE_SCOPE_RECOVERY",
    "QUACK_STATE_REPOSITORY_INTERFACE",
    "QUACK_STATE_REPOSITORY_SCHEMA",
    "REPOSITORY_STATEMENT_TEMPLATES",
    "STATE_REPOSITORY_INTERFACE",
    "STATE_REPOSITORY_SCHEMA",
    "ConformanceReport",
    "EmbeddedOpenPurpose",
    "EmbeddedStateRepository",
    "MaintenanceLease",
    "QuackStateRepository",
    "RepositoryAuthorityMode",
    "StateRepository",
    "StateRepositoryAuthorityError",
    "StateRepositoryConformanceError",
    "StateRepositoryError",
    "StateRepositoryLeaseError",
    "StateRepositoryNotOpenError",
    "assert_conformance_parity",
    "issue_maintenance_lease",
    "open_embedded_repository",
    "open_quack_repository",
    "open_quack_repository_against_database",
    "repository_authority_for_task_source",
    "run_conformance_population",
]
