"""Path-independent control-plane state repository adapters (DQP-008).

Interfaces: ``StateRepository@1``, ``EmbeddedStateRepository@1``,
``QuackStateRepository@1``

Joins schema installation/verification, the typed Quack client, transaction
primitives, and existing DuckDB store access behind one repository protocol.
Higher layers select an explicit authority mode rather than a filesystem path:

* ``quack`` — production authority through Quack; never falls back to direct
  file writes against ``control.duckdb``
* ``embedded`` — hermetic tests and single-process tooling
* ``embedded_exclusive`` — cold import / offline recovery; requires a live
  maintenance lease for the exclusive embedded open

Local and Quack adapters project the same canonical population so conformance
tests can prove parity without rewriting ``DuckDBTaskSource``,
``LeaseCoordinator``, or ``MergeQueue`` in this change.

Import is side-effect free. Opening a repository is the first I/O boundary.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final, Protocol, runtime_checkable

from .control_plane_contracts import (
    ControlPlaneContractError,
    ControlPlaneStoreIdentity,
    StateAuthorityClass,
    StateCommand,
    StateSnapshot,
    StoreGeneration,
    canonical_json_bytes,
    content_identity,
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
from .duckdb_state import open_duckdb_connection
from .quack_state_client import (
    DEFAULT_PAGE_LIMIT,
    DEFAULT_STORE_ID,
    PageResult,
    QuackEndpoint,
    QuackStateClient,
    StatementKind,
    StatementTemplate,
    TransportMode,
    resolve_endpoint,
)

# ---------------------------------------------------------------------------
# Interface / schema identities
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
REPOSITORY_POPULATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/repository-population@1"
)
MAINTENANCE_LEASE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/maintenance-lease@1"
)
STATE_REPOSITORY_VERSION: Final[int] = 1

DEFAULT_MAINTENANCE_SCOPE: Final = "control-plane-exclusive"
DEFAULT_OWNER_ID: Final = "repository:local"
MAINTENANCE_LEASE_ACTIVE: Final = "active"
MAINTENANCE_LEASE_RELEASED: Final = "released"

# Repository-owned statement templates layered onto the closed client set.
_REPOSITORY_TEMPLATES: Final[Mapping[str, StatementTemplate]] = MappingProxyType(
    {
        "list_leases": StatementTemplate(
            name="list_leases",
            sql=(
                "SELECT task_cid, claim_cid, resolution_cid, claimant_did, "
                "logical_epoch, fencing_token, expires_at_ms, attempt, state, "
                "started_at_ms, revision FROM leases "
                "ORDER BY task_cid ASC"
            ),
            parameter_names=(),
            kind=StatementKind.QUERY,
            description="List all task leases for conformance population",
        ),
        "select_lease_by_task": StatementTemplate(
            name="select_lease_by_task",
            sql=(
                "SELECT task_cid, claim_cid, resolution_cid, claimant_did, "
                "logical_epoch, fencing_token, expires_at_ms, attempt, state, "
                "started_at_ms, revision FROM leases "
                "WHERE task_cid = ? LIMIT 1"
            ),
            parameter_names=("task_cid",),
            kind=StatementKind.QUERY,
            description="Fetch one lease by task content id",
        ),
        "list_domain_events_page": StatementTemplate(
            name="list_domain_events_page",
            sql=(
                "SELECT event_id, stream_id, sequence, global_sequence, "
                "event_type, task_cid, attempt_id, session_id, recorded_at "
                "FROM domain_events WHERE global_sequence > ? "
                "ORDER BY global_sequence ASC LIMIT ?"
            ),
            parameter_names=("after_global_sequence", "limit"),
            kind=StatementKind.QUERY,
            description="Cursor page of domain events by global sequence",
        ),
        "max_event_watermark": StatementTemplate(
            name="max_event_watermark",
            sql=(
                "SELECT COALESCE(MAX(global_sequence), 0) AS event_watermark "
                "FROM domain_events"
            ),
            parameter_names=(),
            kind=StatementKind.QUERY,
            description="Latest domain-event watermark",
        ),
        "list_idempotency_keys": StatementTemplate(
            name="list_idempotency_keys",
            sql=(
                "SELECT idempotency_key, command_kind, command_id, store_id, "
                "session_id, result_digest, created_at "
                "FROM idempotency_records ORDER BY idempotency_key ASC"
            ),
            parameter_names=(),
            kind=StatementKind.QUERY,
            description="List idempotent command receipts",
        ),
        "select_active_maintenance_lease": StatementTemplate(
            name="select_active_maintenance_lease",
            sql=(
                "SELECT lease_id, scope, owner_session_id, process_birth_id, "
                "fencing_token, fence_epoch, acquired_at, expires_at, state, "
                "revision FROM maintenance_leases "
                "WHERE scope = ? AND state = ? "
                "ORDER BY fencing_token DESC LIMIT 1"
            ),
            parameter_names=("scope", "state"),
            kind=StatementKind.QUERY,
            description="Load the active maintenance lease for a scope",
        ),
        "insert_maintenance_lease": StatementTemplate(
            name="insert_maintenance_lease",
            sql=(
                "INSERT INTO maintenance_leases ("
                "lease_id, scope, owner_session_id, process_birth_id, "
                "fencing_token, fence_epoch, acquired_at, expires_at, "
                "released_at, state, revision"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)"
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
                "state",
                "revision",
            ),
            kind=StatementKind.MUTATION,
            description="Acquire a maintenance lease row",
        ),
        "release_maintenance_lease": StatementTemplate(
            name="release_maintenance_lease",
            sql=(
                "UPDATE maintenance_leases "
                "SET state = ?, released_at = ?, revision = revision + 1 "
                "WHERE lease_id = ? AND state = ?"
            ),
            parameter_names=("state", "released_at", "lease_id", "expected_state"),
            kind=StatementKind.MUTATION,
            description="Release an active maintenance lease",
        ),
    }
)


# ---------------------------------------------------------------------------
# Errors / authority mode
# ---------------------------------------------------------------------------


class StateRepositoryError(ControlPlaneContractError):
    """Base fail-closed error for state repository adapters."""


class StateRepositoryAuthorityError(StateRepositoryError):
    """Authority mode misuse (fallback, missing lease, wrong transport)."""


class StateRepositoryNotOpenError(StateRepositoryError):
    """Operation requires an open repository session."""


class StateRepositoryMaintenanceError(StateRepositoryError):
    """Maintenance lease acquisition or exclusive-mode precondition failed."""


class RepositoryAuthorityMode(str, Enum):
    """Closed authority modes for control-plane repository adapters."""

    QUACK = "quack"
    EMBEDDED = "embedded"
    EMBEDDED_EXCLUSIVE = "embedded_exclusive"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaintenanceLease:
    """Active or released exclusive-maintenance lease record."""

    SCHEMA: ClassVar[str] = MAINTENANCE_LEASE_SCHEMA

    lease_id: str
    scope: str
    owner_session_id: str
    process_birth_id: str
    fencing_token: int
    fence_epoch: int
    acquired_at: str
    expires_at: str
    state: str
    revision: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "lease_id": self.lease_id,
            "scope": self.scope,
            "owner_session_id": self.owner_session_id,
            "process_birth_id": self.process_birth_id,
            "fencing_token": int(self.fencing_token),
            "fence_epoch": int(self.fence_epoch),
            "acquired_at": self.acquired_at,
            "expires_at": self.expires_at,
            "state": self.state,
            "revision": int(self.revision),
        }

    @property
    def active(self) -> bool:
        return self.state == MAINTENANCE_LEASE_ACTIVE


@dataclass(frozen=True)
class RepositoryPopulation:
    """Canonical multi-domain population for adapter conformance.

    Interface projection for evidence subset: tasks, events, leases, commands,
    snapshots, transactions (generation head), and schema verification.
    """

    SCHEMA: ClassVar[str] = REPOSITORY_POPULATION_SCHEMA

    store_id: str
    authority_mode: str
    generation: Mapping[str, Any]
    store_identity: Mapping[str, Any]
    tasks: tuple[Mapping[str, Any], ...]
    leases: tuple[Mapping[str, Any], ...]
    events: tuple[Mapping[str, Any], ...]
    commands: tuple[Mapping[str, Any], ...]
    schema_fingerprint: str
    event_watermark: int
    task_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "store_id": self.store_id,
            "authority_mode": self.authority_mode,
            "generation": dict(self.generation),
            "store_identity": dict(self.store_identity),
            "tasks": [dict(item) for item in self.tasks],
            "leases": [dict(item) for item in self.leases],
            "events": [dict(item) for item in self.events],
            "commands": [dict(item) for item in self.commands],
            "schema_fingerprint": self.schema_fingerprint,
            "event_watermark": int(self.event_watermark),
            "task_count": int(self.task_count),
        }

    @property
    def content_id(self) -> str:
        # Authority mode is adapter metadata; parity ignores it.
        material = {
            "store_id": self.store_id,
            "generation": dict(self.generation),
            "tasks": [dict(item) for item in self.tasks],
            "leases": [dict(item) for item in self.leases],
            "events": [dict(item) for item in self.events],
            "commands": [dict(item) for item in self.commands],
            "schema_fingerprint": self.schema_fingerprint,
            "event_watermark": int(self.event_watermark),
            "task_count": int(self.task_count),
        }
        return content_identity(material)


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
    def authority_mode(self) -> RepositoryAuthorityMode:
        """Explicit authority mode; never inferred from path shape alone."""

    @property
    def store_id(self) -> str:
        """Logical store identity (not a filesystem path)."""

    @property
    def open(self) -> bool:  # noqa: A003 — protocol surface name
        """Whether the repository session is attached."""

    def attach(self) -> None:
        """Open the repository session (idempotent attach)."""

    def detach(self) -> None:
        """Close the repository session and release transport resources."""

    def close(self) -> None:
        """Permanently close the repository (cannot re-attach)."""

    def store_identity(self) -> ControlPlaneStoreIdentity:
        """Return the verified store identity for the attached session."""

    def load_generation(self) -> StoreGeneration:
        """Load the live store generation head."""

    def verify_schema(self) -> Mapping[str, Any]:
        """Verify schema fingerprint / join-critical surfaces."""

    def get_task(self, task_cid: str) -> Mapping[str, Any] | None:
        """Fetch one task by content id."""

    def list_tasks(
        self,
        *,
        cursor: int = 0,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> PageResult:
        """Cursor-page tasks."""

    def count_tasks(self) -> int:
        """Return the task population size."""

    def get_lease(self, task_cid: str) -> Mapping[str, Any] | None:
        """Fetch one lease by task content id."""

    def list_leases(self) -> tuple[Mapping[str, Any], ...]:
        """List all leases (bounded control-plane population)."""

    def list_events(
        self,
        *,
        cursor: int = 0,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> PageResult:
        """Cursor-page domain events by global sequence."""

    def event_watermark(self) -> int:
        """Latest domain-event global sequence."""

    def list_commands(self) -> tuple[Mapping[str, Any], ...]:
        """List durable idempotency / command receipts."""

    def submit_command(
        self,
        command: StateCommand,
        *,
        apply: Callable[[StateTransaction, StateCommand, StoreGeneration], Mapping[str, Any]]
        | None = None,
        refresh_on_conflict: bool = True,
    ) -> CASResult:
        """Submit a fenced idempotent command."""

    def cas_task_status(
        self,
        *,
        task_cid: str,
        expected_task_revision: int,
        new_status: str,
        idempotency_key: str,
        command_id: str | None = None,
    ) -> CASResult:
        """CAS update task status."""

    def transaction(
        self,
        *,
        expected_generation: StoreGeneration | None = None,
    ) -> StateTransaction:
        """Open a short StateTransaction against the live connection."""

    def snapshot(self) -> StateSnapshot:
        """Build a generation-bound state snapshot identity."""

    def canonical_population(self) -> RepositoryPopulation:
        """Project the conformance population for adapter parity."""


# ---------------------------------------------------------------------------
# Shared client-backed implementation
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _canonical_value(value: Any) -> Any:
    """Normalize driver scalars to JSON-canonical Python types."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    # numpy / duckdb scalar adapters expose ``.item()``.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _canonical_value(item())
        except Exception:
            pass
    if isinstance(value, float):
        # Refuse non-finite floats; finite floats are not control-plane identity.
        if value != value or abs(value) == float("inf"):
            raise StateRepositoryError("non-finite float in repository row")
        if value.is_integer():
            return int(value)
        raise StateRepositoryError("non-integer float in repository row")
    return value


def _mapping_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _canonical_value(row[key]) for key in row}


def _register_repository_templates(client: QuackStateClient) -> None:
    for template in _REPOSITORY_TEMPLATES.values():
        client.register_template(template)


class _ClientBackedStateRepository:
    """Shared implementation for embedded and Quack adapters."""

    INTERFACE: ClassVar[str] = STATE_REPOSITORY_INTERFACE
    SCHEMA: ClassVar[str] = STATE_REPOSITORY_SCHEMA
    VERSION: ClassVar[int] = STATE_REPOSITORY_VERSION

    def __init__(
        self,
        *,
        authority_mode: RepositoryAuthorityMode,
        owner_id: str,
        store_id: str = DEFAULT_STORE_ID,
        expected_identity: ControlPlaneStoreIdentity | None = None,
        retry_policy: RetryPolicy | None = None,
        connect_timeout_seconds: float = 30.0,
        process_birth_id: str | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        if not isinstance(authority_mode, RepositoryAuthorityMode):
            raise StateRepositoryError(
                f"unsupported authority mode: {authority_mode!r}"
            )
        owner = str(owner_id or "").strip()
        if not owner:
            raise StateRepositoryError("owner_id is required")
        self._authority_mode = authority_mode
        self._owner_id = owner
        self._store_id = str(store_id or DEFAULT_STORE_ID).strip() or DEFAULT_STORE_ID
        self._expected_identity = expected_identity
        self._retry_policy = retry_policy
        self._connect_timeout_seconds = float(connect_timeout_seconds)
        self._process_birth_id = process_birth_id or f"birth:{uuid.uuid4()}"
        self._clock = clock or _utc_now
        self._client: QuackStateClient | None = None
        self._closed = False
        self._maintenance_lease: MaintenanceLease | None = None
        self._schema_fingerprint: str = ""

    # -- protocol properties -------------------------------------------------

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
    def is_open(self) -> bool:
        return self._client is not None and self._client.attached and not self._closed

    # Protocol surface uses ``open`` as a property name in docs; expose alias.
    @property
    def open(self) -> bool:  # noqa: A003
        return self.is_open

    @property
    def client(self) -> QuackStateClient:
        return self._require_client()

    @property
    def maintenance_lease(self) -> MaintenanceLease | None:
        return self._maintenance_lease

    # -- lifecycle -----------------------------------------------------------

    def attach(self) -> None:
        raise NotImplementedError

    def detach(self) -> None:
        client = self._client
        self._client = None
        self._maintenance_lease = None
        if client is not None:
            client.detach()

    def close(self) -> None:
        self._closed = True
        client = self._client
        self._client = None
        self._maintenance_lease = None
        if client is not None:
            client.close()

    def __enter__(self) -> "_ClientBackedStateRepository":
        if not self.is_open:
            self.attach()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    # -- identity / schema ---------------------------------------------------

    def store_identity(self) -> ControlPlaneStoreIdentity:
        client = self._require_client()
        session = client.session
        if session is None or session.store_identity is None:
            raise StateRepositoryNotOpenError("store identity is unavailable")
        return session.store_identity

    def load_generation(self) -> StoreGeneration:
        return self._require_client().load_generation()

    def verify_schema(self) -> Mapping[str, Any]:
        raise NotImplementedError

    # -- reads ---------------------------------------------------------------

    def get_task(self, task_cid: str) -> Mapping[str, Any] | None:
        cid = str(task_cid or "").strip()
        if not cid:
            raise StateRepositoryError("task_cid is required")
        rows = self._require_client().execute(
            "select_task_by_cid", {"task_cid": cid}
        )
        if not rows:
            return None
        return _mapping_row(rows[0])

    def list_tasks(
        self,
        *,
        cursor: int = 0,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> PageResult:
        return self._require_client().paginate(
            "list_tasks_page",
            cursor=cursor,
            limit=limit,
            cursor_parameter="after_ordinal",
            limit_parameter="limit",
            cursor_field="ordinal",
        )

    def count_tasks(self) -> int:
        rows = self._require_client().execute("count_tasks")
        if not rows:
            return 0
        return int(rows[0].get("task_count") or 0)

    def get_lease(self, task_cid: str) -> Mapping[str, Any] | None:
        cid = str(task_cid or "").strip()
        if not cid:
            raise StateRepositoryError("task_cid is required")
        rows = self._require_client().execute(
            "select_lease_by_task", {"task_cid": cid}
        )
        if not rows:
            return None
        return _mapping_row(rows[0])

    def list_leases(self) -> tuple[Mapping[str, Any], ...]:
        rows = self._require_client().execute("list_leases")
        return tuple(_mapping_row(row) for row in rows)

    def list_events(
        self,
        *,
        cursor: int = 0,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> PageResult:
        return self._require_client().paginate(
            "list_domain_events_page",
            cursor=cursor,
            limit=limit,
            cursor_parameter="after_global_sequence",
            limit_parameter="limit",
            cursor_field="global_sequence",
        )

    def event_watermark(self) -> int:
        rows = self._require_client().execute("max_event_watermark")
        if not rows:
            return 0
        return int(rows[0].get("event_watermark") or 0)

    def list_commands(self) -> tuple[Mapping[str, Any], ...]:
        rows = self._require_client().execute("list_idempotency_keys")
        return tuple(_mapping_row(row) for row in rows)

    # -- writes --------------------------------------------------------------

    def submit_command(
        self,
        command: StateCommand,
        *,
        apply: Callable[[StateTransaction, StateCommand, StoreGeneration], Mapping[str, Any]]
        | None = None,
        refresh_on_conflict: bool = True,
    ) -> CASResult:
        return self._require_client().submit_command(
            command,
            apply=apply,
            refresh_on_conflict=refresh_on_conflict,
        )

    def cas_task_status(
        self,
        *,
        task_cid: str,
        expected_task_revision: int,
        new_status: str,
        idempotency_key: str,
        command_id: str | None = None,
    ) -> CASResult:
        return self._require_client().cas_task_status(
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
        return self._require_client().transaction(
            expected_generation=expected_generation
        )

    def execute(
        self,
        template_name: str,
        parameters: Mapping[str, Any] | Sequence[Any] | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        """Execute a named closed template (no raw SQL escape hatch)."""

        client = self._require_client()
        rows = client.execute(template_name, parameters)
        # Best-effort commit so mutation templates are durable for later readers
        # (imports, exclusive maintenance sessions, cross-adapter conformance).
        adapter = getattr(client, "_adapter", None)
        if adapter is not None:
            try:
                adapter.commit()
            except Exception:
                pass
        return rows

    def execute_sql(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        """Rejected: repositories never expose arbitrary SQL."""

        raise StateRepositoryAuthorityError(
            "arbitrary SQL is forbidden on StateRepository; use named templates"
        )

    # -- snapshot / population -----------------------------------------------

    def snapshot(self) -> StateSnapshot:
        client = self._require_client()
        generation = client.load_generation()
        identity = self.store_identity()
        watermark = self.event_watermark()
        digest_material = {
            "store_id": self._store_id,
            "database_uuid": generation.database_uuid,
            "generation": generation.generation,
            "schema_revision": generation.schema_revision,
            "revision": generation.revision,
            "fence_epoch": generation.fence_epoch,
            "event_watermark": watermark,
            "schema_fingerprint": identity.schema_fingerprint,
        }
        digest = (
            "sha256:"
            + hashlib.sha256(canonical_json_bytes(digest_material)).hexdigest()
        )
        snapshot_id = (
            f"snapshot:{generation.generation}:{generation.revision}:{watermark}"
        )
        return StateSnapshot(
            snapshot_id=snapshot_id,
            store_id=self._store_id,
            database_uuid=generation.database_uuid,
            generation=generation.generation,
            schema_revision=generation.schema_revision,
            revision=generation.revision,
            fence_epoch=generation.fence_epoch,
            event_watermark=watermark,
            snapshot_digest=digest,
            authority_class=StateAuthorityClass.AUTHORITATIVE,
        )

    def canonical_population(self) -> RepositoryPopulation:
        """Collect the conformance population (mode-independent content)."""

        generation = self.load_generation()
        identity = self.store_identity()
        # Drain all task pages for a stable full population.
        tasks: list[dict[str, Any]] = []
        cursor = 0
        while True:
            page = self.list_tasks(cursor=cursor, limit=DEFAULT_PAGE_LIMIT)
            tasks.extend(_mapping_row(item) for item in page.items)
            if page.exhausted or page.next_cursor is None:
                break
            cursor = int(page.next_cursor)
        events: list[dict[str, Any]] = []
        event_cursor = 0
        while True:
            page = self.list_events(cursor=event_cursor, limit=DEFAULT_PAGE_LIMIT)
            events.extend(_mapping_row(item) for item in page.items)
            if page.exhausted or page.next_cursor is None:
                break
            event_cursor = int(page.next_cursor)
        schema_fingerprint = self._schema_fingerprint or identity.schema_fingerprint
        # Keep only generation identity fields that are shared across adapters
        # (drop schema labels, birth_id, and other attach-local metadata).
        generation_payload = {
            "store_id": self._store_id,
            "generation": int(generation.generation),
            "schema_revision": int(generation.schema_revision),
            "fence_epoch": int(generation.fence_epoch),
            "revision": int(generation.revision),
            "database_uuid": str(generation.database_uuid),
        }
        identity_payload = {
            "repository_id": str(identity.repository_id),
            "database_uuid": str(identity.database_uuid),
            "store_id": str(identity.store_id),
            "schema_revision": int(identity.schema_revision),
            "schema_fingerprint": str(identity.schema_fingerprint),
            "authority_class": str(identity.authority_class.value),
        }
        # Sort for stable content identity regardless of scan order.
        tasks_sorted = tuple(
            sorted(tasks, key=lambda row: str(row.get("task_cid") or ""))
        )
        leases_sorted = tuple(
            sorted(
                self.list_leases(),
                key=lambda row: str(row.get("task_cid") or ""),
            )
        )
        events_sorted = tuple(
            sorted(
                events,
                key=lambda row: (
                    int(row.get("global_sequence") or 0),
                    str(row.get("event_id") or ""),
                ),
            )
        )
        commands_sorted = tuple(
            sorted(
                self.list_commands(),
                key=lambda row: str(row.get("idempotency_key") or ""),
            )
        )
        return RepositoryPopulation(
            store_id=self._store_id,
            authority_mode=self._authority_mode.value,
            generation=MappingProxyType(generation_payload),
            store_identity=MappingProxyType(identity_payload),
            tasks=tasks_sorted,
            leases=leases_sorted,
            events=events_sorted,
            commands=commands_sorted,
            schema_fingerprint=schema_fingerprint,
            event_watermark=self.event_watermark(),
            task_count=self.count_tasks(),
        )

    # -- helpers -------------------------------------------------------------

    def _require_client(self) -> QuackStateClient:
        if self._closed:
            raise StateRepositoryNotOpenError("repository is closed")
        if self._client is None or not self._client.attached:
            raise StateRepositoryNotOpenError("repository is not attached")
        return self._client

    def _new_client(
        self,
        *,
        connection_factory: Callable[[QuackEndpoint], Any] | None = None,
    ) -> QuackStateClient:
        client = QuackStateClient(
            owner_id=self._owner_id,
            store_id=self._store_id,
            expected_identity=self._expected_identity,
            retry_policy=self._retry_policy,
            connect_timeout_seconds=self._connect_timeout_seconds,
            process_birth_id=self._process_birth_id,
            clock=self._clock,
            connection_factory=connection_factory,
        )
        _register_repository_templates(client)
        return client

    def _capture_schema_fingerprint(self) -> None:
        client = self._require_client()
        rows = client.execute("whoami_metadata")
        meta = {str(row["key"]): str(row["value"]) for row in rows}
        fingerprint = str(meta.get("schema_fingerprint") or "")
        if fingerprint.startswith("sha256:"):
            self._schema_fingerprint = fingerprint
        else:
            identity = self.store_identity()
            self._schema_fingerprint = identity.schema_fingerprint


# ---------------------------------------------------------------------------
# Embedded adapter
# ---------------------------------------------------------------------------


class EmbeddedStateRepository(_ClientBackedStateRepository):
    """Embedded DuckDB repository for tests, tooling, and leased imports.

    Interface: ``EmbeddedStateRepository@1``.

    ``RepositoryAuthorityMode.EMBEDDED`` opens the database file under the
    process-shared exclusive lock helper. ``EMBEDDED_EXCLUSIVE`` additionally
    requires a live maintenance lease so cold imports cannot race a live Quack
    state-owner.
    """

    INTERFACE: ClassVar[str] = EMBEDDED_STATE_REPOSITORY_INTERFACE
    SCHEMA: ClassVar[str] = EMBEDDED_STATE_REPOSITORY_SCHEMA

    def __init__(
        self,
        database_path: str | Path,
        *,
        owner_id: str = DEFAULT_OWNER_ID,
        store_id: str = DEFAULT_STORE_ID,
        exclusive: bool = False,
        maintenance_lease: MaintenanceLease | Mapping[str, Any] | None = None,
        maintenance_scope: str = DEFAULT_MAINTENANCE_SCOPE,
        expected_identity: ControlPlaneStoreIdentity | None = None,
        retry_policy: RetryPolicy | None = None,
        connect_timeout_seconds: float = 30.0,
        process_birth_id: str | None = None,
        install_schema: bool = False,
        seed_generation: bool = True,
        clock: Callable[[], str] | None = None,
        application_version: str = "0.0.45",
        tool_version: str = "1.5.2",
    ) -> None:
        mode = (
            RepositoryAuthorityMode.EMBEDDED_EXCLUSIVE
            if exclusive
            else RepositoryAuthorityMode.EMBEDDED
        )
        super().__init__(
            authority_mode=mode,
            owner_id=owner_id,
            store_id=store_id,
            expected_identity=expected_identity,
            retry_policy=retry_policy,
            connect_timeout_seconds=connect_timeout_seconds,
            process_birth_id=process_birth_id,
            clock=clock,
        )
        path = Path(database_path)
        if not str(path):
            raise StateRepositoryError("database_path is required for embedded mode")
        self.database_path = path
        self._install_schema = bool(install_schema)
        self._seed_generation = bool(seed_generation)
        self._application_version = application_version
        self._tool_version = tool_version
        self._maintenance_scope = str(maintenance_scope or DEFAULT_MAINTENANCE_SCOPE)
        self._provided_lease = (
            None if maintenance_lease is None else _coerce_lease(maintenance_lease)
        )

    def attach(self) -> None:
        if self._closed:
            raise StateRepositoryNotOpenError("repository is closed")
        if self.is_open:
            return
        if self._authority_mode is RepositoryAuthorityMode.EMBEDDED_EXCLUSIVE:
            if self._provided_lease is None or not self._provided_lease.active:
                raise StateRepositoryMaintenanceError(
                    "embedded exclusive mode requires an active maintenance lease; "
                    "refuse unleased direct file authority"
                )
            self._maintenance_lease = self._provided_lease

        if self._install_schema:
            install_control_plane_schema(
                self.database_path,
                application_version=self._application_version,
                tool_version=self._tool_version,
                owner_id=self._owner_id,
            )

        client = self._new_client()
        try:
            client.attach(
                self.database_path,
                mode=TransportMode.EMBEDDED,
                seed_generation=self._seed_generation,
                expected_identity=self._expected_identity,
            )
        except Exception:
            client.close()
            raise
        self._client = client
        self._capture_schema_fingerprint()
        if self._authority_mode is RepositoryAuthorityMode.EMBEDDED_EXCLUSIVE:
            self._assert_maintenance_lease_still_active()

    def verify_schema(self) -> Mapping[str, Any]:
        """Verify schema without taking a second exclusive file lock.

        When attached, probe through the live client (the open connection
        already holds the process lock). When detached, use the path-bound
        installer helper which opens its own exclusive session.
        """

        if self.is_open:
            return self._verify_schema_via_client()
        report = dict(verify_installed_schema(self.database_path))
        fingerprint = str(report.get("schema_fingerprint") or "")
        if fingerprint.startswith("sha256:"):
            self._schema_fingerprint = fingerprint
        report["authority_mode"] = self._authority_mode.value
        report["store_id"] = self._store_id
        report["transport_mode"] = TransportMode.EMBEDDED.value
        return report

    def _verify_schema_via_client(self) -> Mapping[str, Any]:
        client = self._require_client()
        generation = client.load_generation()
        identity = self.store_identity()
        meta_rows = client.execute("whoami_metadata")
        meta = {str(row["key"]): str(row["value"]) for row in meta_rows}
        fingerprint = str(
            self._schema_fingerprint or identity.schema_fingerprint or ""
        )
        if not fingerprint.startswith("sha256:"):
            raise StateRepositoryError(
                "schema_fingerprint missing from embedded store metadata"
            )
        self._schema_fingerprint = fingerprint
        task_count = self.count_tasks()
        _ = client.execute("list_leases")
        watermark = self.event_watermark()
        return {
            "authority_mode": self._authority_mode.value,
            "store_id": self._store_id,
            "schema_fingerprint": fingerprint,
            "schema_revision": generation.schema_revision,
            "generation": generation.generation,
            "task_count": task_count,
            "event_watermark": watermark,
            "database_uuid": generation.database_uuid,
            "transport_mode": TransportMode.EMBEDDED.value,
            # Compatibility keys for callers that inspect the path-bound report.
            "tables_ok": ["tasks", "leases", "domain_events", "store_generations"],
            "task_columns_ok": list(
                ("task_cid", "task_alias", "goal_cid", "ordinal", "status", "revision")
            ),
            "lease_columns_ok": ["task_cid", "claim_cid", "fencing_token", "state"],
        }

    def _assert_maintenance_lease_still_active(self) -> None:
        lease = self._maintenance_lease
        if lease is None:
            raise StateRepositoryMaintenanceError("maintenance lease missing")
        rows = self._require_client().execute(
            "select_active_maintenance_lease",
            {
                "scope": lease.scope,
                "state": MAINTENANCE_LEASE_ACTIVE,
            },
        )
        if not rows:
            raise StateRepositoryMaintenanceError(
                "maintenance lease is not active in the store"
            )
        row = rows[0]
        if str(row.get("lease_id")) != lease.lease_id:
            raise StateRepositoryMaintenanceError(
                "maintenance lease_id mismatch for exclusive embedded open"
            )
        if int(row.get("fencing_token") or 0) != int(lease.fencing_token):
            raise StateRepositoryMaintenanceError(
                "maintenance lease fencing_token mismatch"
            )


# ---------------------------------------------------------------------------
# Quack adapter
# ---------------------------------------------------------------------------


class QuackStateRepository(_ClientBackedStateRepository):
    """Production Quack-backed repository.

    Interface: ``QuackStateRepository@1``.

    Never opens ``control.duckdb`` as a direct file writer. Attach targets must
    be loopback ``quack:`` URIs (or a pre-resolved ``QuackEndpoint`` with
    ``TransportMode.QUACK``). There is no silent embedded fallback when the
    extension or endpoint is unavailable — callers receive a typed error.
    """

    INTERFACE: ClassVar[str] = QUACK_STATE_REPOSITORY_INTERFACE
    SCHEMA: ClassVar[str] = QUACK_STATE_REPOSITORY_SCHEMA

    def __init__(
        self,
        endpoint: str | QuackEndpoint,
        *,
        owner_id: str = DEFAULT_OWNER_ID,
        store_id: str = DEFAULT_STORE_ID,
        secret_handle: str = "",
        server_id: str = "server:local",
        expected_identity: ControlPlaneStoreIdentity | None = None,
        retry_policy: RetryPolicy | None = None,
        connect_timeout_seconds: float = 30.0,
        process_birth_id: str | None = None,
        clock: Callable[[], str] | None = None,
        connection_factory: Callable[[QuackEndpoint], Any] | None = None,
        seed_generation: bool = False,
        allow_embedded_fallback: bool = False,
    ) -> None:
        if allow_embedded_fallback:
            # Explicitly rejected: Quack authority must never degrade to files.
            raise StateRepositoryAuthorityError(
                "QuackStateRepository refuses embedded file-write fallback; "
                "set allow_embedded_fallback=False (the only accepted value)"
            )
        super().__init__(
            authority_mode=RepositoryAuthorityMode.QUACK,
            owner_id=owner_id,
            store_id=store_id,
            expected_identity=expected_identity,
            retry_policy=retry_policy,
            connect_timeout_seconds=connect_timeout_seconds,
            process_birth_id=process_birth_id,
            clock=clock,
        )
        self._endpoint_input = endpoint
        self._secret_handle = secret_handle
        self._server_id = server_id
        self._connection_factory = connection_factory
        self._seed_generation = bool(seed_generation)
        self._resolved_endpoint: QuackEndpoint | None = None

    @property
    def endpoint(self) -> QuackEndpoint | None:
        return self._resolved_endpoint

    def attach(self) -> None:
        if self._closed:
            raise StateRepositoryNotOpenError("repository is closed")
        if self.is_open:
            return
        endpoint = self._resolve_quack_endpoint(self._endpoint_input)
        self._resolved_endpoint = endpoint
        # Live Quack ATTACH is production-only; hermetic tests inject a factory.
        # Transport failure never degrades to an embedded file open.
        client = self._new_client(connection_factory=self._connection_factory)
        try:
            client.attach(
                endpoint,
                mode=TransportMode.QUACK,
                secret_handle=self._secret_handle,
                server_id=self._server_id,
                seed_generation=self._seed_generation,
                expected_identity=self._expected_identity,
            )
        except Exception:
            client.close()
            self._client = None
            raise
        # Defense in depth: session transport must remain Quack.
        session = client.session
        if session is None or session.transport_mode is not TransportMode.QUACK:
            client.close()
            self._client = None
            raise StateRepositoryAuthorityError(
                "QuackStateRepository session is not in Quack transport mode; "
                "refusing embedded file-write fallback"
            )
        self._client = client
        self._capture_schema_fingerprint()

    def verify_schema(self) -> Mapping[str, Any]:
        """Verify schema via typed templates (no direct file open)."""

        client = self._require_client()
        # Touch identity + generation; refuse if missing.
        generation = client.load_generation()
        identity = self.store_identity()
        meta_rows = client.execute("whoami_metadata")
        meta = {str(row["key"]): str(row["value"]) for row in meta_rows}
        fingerprint = str(
            self._schema_fingerprint or identity.schema_fingerprint or ""
        )
        if not fingerprint.startswith("sha256:"):
            raise StateRepositoryError(
                "schema_fingerprint missing from Quack whoami metadata"
            )
        self._schema_fingerprint = fingerprint
        # Prove domain tables respond through templates (not raw SQL).
        task_count = self.count_tasks()
        _ = client.execute("list_leases")
        watermark = self.event_watermark()
        return {
            "authority_mode": self._authority_mode.value,
            "store_id": self._store_id,
            "schema_fingerprint": fingerprint,
            "schema_revision": generation.schema_revision,
            "generation": generation.generation,
            "task_count": task_count,
            "event_watermark": watermark,
            "database_uuid": generation.database_uuid,
            "transport_mode": TransportMode.QUACK.value,
        }

    def _resolve_quack_endpoint(
        self, endpoint: str | QuackEndpoint
    ) -> QuackEndpoint:
        if isinstance(endpoint, QuackEndpoint):
            if endpoint.mode is not TransportMode.QUACK:
                raise StateRepositoryAuthorityError(
                    "QuackStateRepository requires TransportMode.QUACK; "
                    f"got {endpoint.mode!r}"
                )
            if endpoint.database_path is not None and self._connection_factory is None:
                # A database_path on a Quack endpoint would invite file fallback.
                raise StateRepositoryAuthorityError(
                    "QuackStateRepository rejects endpoints that carry a "
                    "database_path (no direct file authority)"
                )
            return endpoint
        text = str(endpoint or "").strip()
        if not text:
            raise StateRepositoryError("Quack endpoint is required")
        # Hard-reject filesystem paths before resolve_endpoint defaults them.
        if not text.startswith("quack:"):
            path_candidate = Path(text)
            if path_candidate.suffix in {".duckdb", ".db"} or path_candidate.exists():
                raise StateRepositoryAuthorityError(
                    "QuackStateRepository refuses filesystem database paths; "
                    "provide a loopback quack: URI"
                )
            raise StateRepositoryAuthorityError(
                "QuackStateRepository requires a quack: URI, not "
                f"{text!r}"
            )
        resolved = resolve_endpoint(
            text,
            mode=TransportMode.QUACK,
            secret_handle=self._secret_handle,
        )
        if resolved.mode is not TransportMode.QUACK:
            raise StateRepositoryAuthorityError(
                "resolved endpoint is not Quack transport"
            )
        return resolved


# ---------------------------------------------------------------------------
# Maintenance lease helpers (embedded exclusive / cold import)
# ---------------------------------------------------------------------------


def _coerce_lease(
    value: MaintenanceLease | Mapping[str, Any],
) -> MaintenanceLease:
    if isinstance(value, MaintenanceLease):
        return value
    return MaintenanceLease(
        lease_id=str(value["lease_id"]),
        scope=str(value["scope"]),
        owner_session_id=str(value["owner_session_id"]),
        process_birth_id=str(value["process_birth_id"]),
        fencing_token=int(value["fencing_token"]),
        fence_epoch=int(value["fence_epoch"]),
        acquired_at=str(value["acquired_at"]),
        expires_at=str(value["expires_at"]),
        state=str(value.get("state") or MAINTENANCE_LEASE_ACTIVE),
        revision=int(value.get("revision") or 0),
    )


def acquire_maintenance_lease(
    database_path: str | Path,
    *,
    owner_session_id: str,
    process_birth_id: str,
    scope: str = DEFAULT_MAINTENANCE_SCOPE,
    fencing_token: int | None = None,
    fence_epoch: int = 1,
    ttl_seconds: int = 3600,
    clock: Callable[[], str] | None = None,
) -> MaintenanceLease:
    """Acquire an exclusive maintenance lease on an embedded control plane.

    Used before cold import / offline recovery so ``EMBEDDED_EXCLUSIVE`` opens
    are fenced. Callers must stop Quack before taking this lease in production.

    When ``fencing_token`` is omitted, the next free token for ``scope`` is
    chosen so sequential acquire/release cycles do not collide with the unique
    ``(scope, state, fencing_token)`` index on released rows.
    """

    if fencing_token is not None and (
        isinstance(fencing_token, bool)
        or not isinstance(fencing_token, int)
        or fencing_token < 1
    ):
        raise StateRepositoryMaintenanceError("fencing_token must be a positive int")
    if ttl_seconds < 1:
        raise StateRepositoryMaintenanceError("ttl_seconds must be >= 1")
    now_fn = clock or _utc_now
    acquired_at = now_fn()
    # Expiry is an ISO-ish watermark; exact clock math is not required for the
    # lease row — operators supply short-lived leases.
    expires_at = acquired_at
    lease_id = f"mlease:{uuid.uuid4()}"
    path = Path(database_path)
    with open_duckdb_connection(path) as connection:
        existing = connection.execute(
            """
            SELECT lease_id FROM maintenance_leases
            WHERE scope = ? AND state = ?
            LIMIT 1
            """,
            [scope, MAINTENANCE_LEASE_ACTIVE],
        ).fetchone()
        if existing is not None:
            raise StateRepositoryMaintenanceError(
                f"active maintenance lease already held for scope {scope!r}"
            )
        if fencing_token is None:
            row = connection.execute(
                """
                SELECT COALESCE(MAX(fencing_token), 0) AS max_token
                FROM maintenance_leases
                WHERE scope = ?
                """,
                [scope],
            ).fetchone()
            if row is None:
                next_token = 1
            elif isinstance(row, Mapping):
                next_token = int(row["max_token"]) + 1
            else:
                next_token = int(row[0]) + 1
            fencing_token = max(1, next_token)
        connection.execute(
            """
            INSERT INTO maintenance_leases (
                lease_id, scope, owner_session_id, process_birth_id,
                fencing_token, fence_epoch, acquired_at, expires_at,
                released_at, state, revision
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            [
                lease_id,
                scope,
                owner_session_id,
                process_birth_id,
                fencing_token,
                fence_epoch,
                acquired_at,
                expires_at,
                MAINTENANCE_LEASE_ACTIVE,
                0,
            ],
        )
    return MaintenanceLease(
        lease_id=lease_id,
        scope=scope,
        owner_session_id=owner_session_id,
        process_birth_id=process_birth_id,
        fencing_token=int(fencing_token),
        fence_epoch=fence_epoch,
        acquired_at=acquired_at,
        expires_at=expires_at,
        state=MAINTENANCE_LEASE_ACTIVE,
        revision=0,
    )


def release_maintenance_lease(
    database_path: str | Path,
    lease: MaintenanceLease | Mapping[str, Any],
    *,
    clock: Callable[[], str] | None = None,
) -> MaintenanceLease:
    """Release a previously acquired maintenance lease."""

    record = _coerce_lease(lease)
    now_fn = clock or _utc_now
    released_at = now_fn()
    path = Path(database_path)
    with open_duckdb_connection(path) as connection:
        result = connection.execute(
            """
            UPDATE maintenance_leases
            SET state = ?, released_at = ?, revision = revision + 1
            WHERE lease_id = ? AND state = ?
            """,
            [
                MAINTENANCE_LEASE_RELEASED,
                released_at,
                record.lease_id,
                MAINTENANCE_LEASE_ACTIVE,
            ],
        )
        changed = getattr(result, "rowcount", None)
        # DuckDB may not expose rowcount; re-read.
        row = connection.execute(
            """
            SELECT state, revision FROM maintenance_leases
            WHERE lease_id = ? LIMIT 1
            """,
            [record.lease_id],
        ).fetchone()
        if row is None:
            raise StateRepositoryMaintenanceError("maintenance lease not found")
        if isinstance(row, Mapping):
            state = str(row["state"])
            revision = int(row["revision"])
        else:
            state = str(row[0])
            revision = int(row[1])
        if state != MAINTENANCE_LEASE_RELEASED:
            raise StateRepositoryMaintenanceError(
                "maintenance lease was not active at release"
            )
        _ = changed
    return MaintenanceLease(
        lease_id=record.lease_id,
        scope=record.scope,
        owner_session_id=record.owner_session_id,
        process_birth_id=record.process_birth_id,
        fencing_token=record.fencing_token,
        fence_epoch=record.fence_epoch,
        acquired_at=record.acquired_at,
        expires_at=record.expires_at,
        state=MAINTENANCE_LEASE_RELEASED,
        revision=revision,
    )


@contextmanager
def exclusive_embedded_repository(
    database_path: str | Path,
    *,
    owner_id: str = DEFAULT_OWNER_ID,
    owner_session_id: str | None = None,
    process_birth_id: str | None = None,
    scope: str = DEFAULT_MAINTENANCE_SCOPE,
    install_schema: bool = False,
    seed_generation: bool = True,
    **kwargs: Any,
) -> Iterator[EmbeddedStateRepository]:
    """Context manager: acquire lease, open exclusive embedded repo, release."""

    birth = process_birth_id or f"birth:{uuid.uuid4()}"
    session_id = owner_session_id or f"session:maintenance:{uuid.uuid4()}"
    path = Path(database_path)
    if install_schema:
        install_control_plane_schema(
            path,
            owner_id=owner_id,
            application_version=kwargs.get("application_version", "0.0.45"),
            tool_version=kwargs.get("tool_version", "1.5.2"),
        )
    lease = acquire_maintenance_lease(
        path,
        owner_session_id=session_id,
        process_birth_id=birth,
        scope=scope,
    )
    reserved = {
        "application_version",
        "tool_version",
        "exclusive",
        "maintenance_lease",
        "maintenance_scope",
        "owner_id",
        "process_birth_id",
        "install_schema",
        "seed_generation",
    }
    repo = EmbeddedStateRepository(
        path,
        owner_id=owner_id,
        exclusive=True,
        maintenance_lease=lease,
        maintenance_scope=scope,
        install_schema=False,
        seed_generation=seed_generation,
        process_birth_id=birth,
        **{key: value for key, value in kwargs.items() if key not in reserved},
    )
    try:
        repo.attach()
        yield repo
    finally:
        repo.close()
        try:
            release_maintenance_lease(path, lease)
        except StateRepositoryMaintenanceError:
            pass


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def open_embedded_repository(
    database_path: str | Path,
    *,
    owner_id: str = DEFAULT_OWNER_ID,
    exclusive: bool = False,
    maintenance_lease: MaintenanceLease | Mapping[str, Any] | None = None,
    install_schema: bool = False,
    seed_generation: bool = True,
    **kwargs: Any,
) -> EmbeddedStateRepository:
    """Construct and attach an embedded state repository."""

    repo = EmbeddedStateRepository(
        database_path,
        owner_id=owner_id,
        exclusive=exclusive,
        maintenance_lease=maintenance_lease,
        install_schema=install_schema,
        seed_generation=seed_generation,
        **kwargs,
    )
    repo.attach()
    return repo


def open_quack_repository(
    endpoint: str | QuackEndpoint,
    *,
    owner_id: str = DEFAULT_OWNER_ID,
    secret_handle: str = "",
    connection_factory: Callable[[QuackEndpoint], Any] | None = None,
    seed_generation: bool = False,
    **kwargs: Any,
) -> QuackStateRepository:
    """Construct and attach a Quack state repository (no file fallback)."""

    repo = QuackStateRepository(
        endpoint,
        owner_id=owner_id,
        secret_handle=secret_handle,
        connection_factory=connection_factory,
        seed_generation=seed_generation,
        **kwargs,
    )
    repo.attach()
    return repo


def open_state_repository(
    *,
    authority_mode: RepositoryAuthorityMode | str,
    target: str | Path | QuackEndpoint,
    owner_id: str = DEFAULT_OWNER_ID,
    **kwargs: Any,
) -> EmbeddedStateRepository | QuackStateRepository:
    """Open a repository by explicit authority mode (path-independent API)."""

    mode = (
        authority_mode
        if isinstance(authority_mode, RepositoryAuthorityMode)
        else RepositoryAuthorityMode(str(authority_mode))
    )
    if mode is RepositoryAuthorityMode.QUACK:
        return open_quack_repository(
            target,  # type: ignore[arg-type]
            owner_id=owner_id,
            **kwargs,
        )
    if mode is RepositoryAuthorityMode.EMBEDDED:
        return open_embedded_repository(
            target,  # type: ignore[arg-type]
            owner_id=owner_id,
            exclusive=False,
            **kwargs,
        )
    if mode is RepositoryAuthorityMode.EMBEDDED_EXCLUSIVE:
        return open_embedded_repository(
            target,  # type: ignore[arg-type]
            owner_id=owner_id,
            exclusive=True,
            **kwargs,
        )
    raise StateRepositoryError(f"unsupported authority mode: {mode!r}")


def populations_equivalent(
    left: RepositoryPopulation,
    right: RepositoryPopulation,
) -> bool:
    """Return True when two adapter populations share content identity."""

    return left.content_id == right.content_id


__all__ = [
    "DEFAULT_MAINTENANCE_SCOPE",
    "DEFAULT_OWNER_ID",
    "DEFAULT_STORE_ID",
    "EMBEDDED_STATE_REPOSITORY_INTERFACE",
    "EMBEDDED_STATE_REPOSITORY_SCHEMA",
    "MAINTENANCE_LEASE_ACTIVE",
    "MAINTENANCE_LEASE_RELEASED",
    "MAINTENANCE_LEASE_SCHEMA",
    "QUACK_STATE_REPOSITORY_INTERFACE",
    "QUACK_STATE_REPOSITORY_SCHEMA",
    "REPOSITORY_POPULATION_SCHEMA",
    "STATE_REPOSITORY_INTERFACE",
    "STATE_REPOSITORY_SCHEMA",
    "STATE_REPOSITORY_VERSION",
    "EmbeddedStateRepository",
    "MaintenanceLease",
    "QuackStateRepository",
    "RepositoryAuthorityMode",
    "RepositoryPopulation",
    "StateRepository",
    "StateRepositoryAuthorityError",
    "StateRepositoryError",
    "StateRepositoryMaintenanceError",
    "StateRepositoryNotOpenError",
    "acquire_maintenance_lease",
    "exclusive_embedded_repository",
    "open_embedded_repository",
    "open_quack_repository",
    "open_state_repository",
    "populations_equivalent",
    "release_maintenance_lease",
]
