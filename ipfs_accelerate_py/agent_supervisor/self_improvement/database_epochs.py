"""DuckDB-backed self-improvement epoch repository.

DQP-031 / ImprovementEpochRepository@1
======================================

:class:`ImprovementEpochRepository` stores improvement epochs, stage
transitions, challengers, rollouts, token metrics, and receipts as
transactional rows. Challengers use ordinary worktree, session, and lease
identities — the same coordination vocabulary as any other registered
worker. Self-improvement goals and tasks are planned into the same database
so they participate in the ordinary objective/task schedule.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ..task_sources.control_plane_contracts import (
    REDACTION_MARKER,
    redact_mapping,
)
from ..task_sources.duckdb_state import open_duckdb_connection
from ..task_sources.task_identity import canonical_json_bytes


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

IMPROVEMENT_EPOCH_REPOSITORY_INTERFACE: Final[str] = "ImprovementEpochRepository@1"
IMPROVEMENT_EPOCH_REPOSITORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/improvement-epoch-repository@1"
)
IMPROVEMENT_EPOCH_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/improvement-epoch@1"
)
EPOCH_TRANSITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/improvement-epoch-transition@1"
)
EPOCH_CHALLENGER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/improvement-epoch-challenger@1"
)
EPOCH_ROLLOUT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/improvement-epoch-rollout@1"
)
EPOCH_TOKEN_METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/improvement-epoch-token-metrics@1"
)
EPOCH_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/improvement-epoch-receipt@1"
)
PLANNED_GOAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/improvement-planned-goal@1"
)
PLANNED_TASK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/improvement-planned-task@1"
)

DEFAULT_SNAPSHOT_ID: Final[str] = "snapshot:improvement-epoch-repository"
AUTHORITY_CLASS: Final[str] = "database_authority"
MAX_CHALLENGERS_PER_EPOCH: Final[int] = 1
MAX_GOALS_PER_EPOCH: Final[int] = 8
MAX_TASKS_PER_EPOCH: Final[int] = 24
MAX_TRANSITIONS_PER_EPOCH: Final[int] = 256
MAX_BODY_BYTES: Final[int] = 262_144
MAX_TEXT_BYTES: Final[int] = 8_192
MAX_ID_BYTES: Final[int] = 512
MAX_RECURSION_DEPTH: Final[int] = 8
DEFAULT_PAGE_LIMIT: Final[int] = 256
MAX_PAGE_LIMIT: Final[int] = 4_096

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./:@+-]{0,511}$")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS improvement_epoch_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS improvement_epochs (
    epoch_id VARCHAR PRIMARY KEY,
    program_id VARCHAR NOT NULL DEFAULT '',
    policy_id VARCHAR NOT NULL DEFAULT '',
    mode VARCHAR NOT NULL,
    stage VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    repository_id VARCHAR NOT NULL DEFAULT '',
    tree_id VARCHAR NOT NULL DEFAULT '',
    objective_id VARCHAR NOT NULL DEFAULT '',
    baseline_digest VARCHAR NOT NULL DEFAULT '',
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}',
    content_digest VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS improvement_epochs_status_idx
    ON improvement_epochs(status, updated_at);

CREATE TABLE IF NOT EXISTS epoch_transitions (
    transition_id VARCHAR PRIMARY KEY,
    epoch_id VARCHAR NOT NULL,
    sequence BIGINT NOT NULL,
    from_stage VARCHAR NOT NULL,
    to_stage VARCHAR NOT NULL,
    reason VARCHAR NOT NULL DEFAULT '',
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}',
    content_digest VARCHAR NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS epoch_transitions_seq_uidx
    ON epoch_transitions(epoch_id, sequence);
CREATE INDEX IF NOT EXISTS epoch_transitions_epoch_idx
    ON epoch_transitions(epoch_id, recorded_at);

CREATE TABLE IF NOT EXISTS epoch_challengers (
    challenger_id VARCHAR PRIMARY KEY,
    epoch_id VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    lease_id VARCHAR NOT NULL,
    fencing_epoch BIGINT NOT NULL DEFAULT 0,
    status VARCHAR NOT NULL DEFAULT 'registered',
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}',
    content_digest VARCHAR NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS epoch_challengers_epoch_uidx
    ON epoch_challengers(epoch_id);
CREATE INDEX IF NOT EXISTS epoch_challengers_worktree_idx
    ON epoch_challengers(worktree_id);
CREATE INDEX IF NOT EXISTS epoch_challengers_session_idx
    ON epoch_challengers(session_id);
CREATE INDEX IF NOT EXISTS epoch_challengers_lease_idx
    ON epoch_challengers(lease_id);

CREATE TABLE IF NOT EXISTS epoch_rollouts (
    rollout_id VARCHAR PRIMARY KEY,
    epoch_id VARCHAR NOT NULL,
    mode VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}',
    content_digest VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS epoch_rollouts_epoch_idx
    ON epoch_rollouts(epoch_id, created_at);

CREATE TABLE IF NOT EXISTS epoch_token_metrics (
    metrics_id VARCHAR PRIMARY KEY,
    epoch_id VARCHAR NOT NULL,
    tokens_in BIGINT NOT NULL DEFAULT 0,
    tokens_out BIGINT NOT NULL DEFAULT 0,
    provider_calls BIGINT NOT NULL DEFAULT 0,
    cost_micros BIGINT NOT NULL DEFAULT 0,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}',
    content_digest VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS epoch_token_metrics_epoch_idx
    ON epoch_token_metrics(epoch_id, recorded_at);

CREATE TABLE IF NOT EXISTS epoch_receipts (
    receipt_id VARCHAR PRIMARY KEY,
    epoch_id VARCHAR NOT NULL,
    kind VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}',
    content_digest VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS epoch_receipts_epoch_idx
    ON epoch_receipts(epoch_id, recorded_at);

CREATE TABLE IF NOT EXISTS planned_goals (
    goal_cid VARCHAR PRIMARY KEY,
    epoch_id VARCHAR NOT NULL,
    title VARCHAR NOT NULL,
    status VARCHAR NOT NULL DEFAULT 'open',
    parent_goal_cid VARCHAR NOT NULL DEFAULT '',
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}',
    content_digest VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS planned_goals_epoch_idx
    ON planned_goals(epoch_id, created_at);

CREATE TABLE IF NOT EXISTS planned_tasks (
    task_cid VARCHAR PRIMARY KEY,
    epoch_id VARCHAR NOT NULL,
    goal_cid VARCHAR NOT NULL,
    title VARCHAR NOT NULL,
    status VARCHAR NOT NULL DEFAULT 'ready',
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}',
    content_digest VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS planned_tasks_epoch_idx
    ON planned_tasks(epoch_id, created_at);
CREATE INDEX IF NOT EXISTS planned_tasks_goal_idx
    ON planned_tasks(goal_cid, created_at);
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ImprovementEpochError(RuntimeError):
    """Base error for improvement-epoch repository failures."""


class ImprovementEpochNotOpenError(ImprovementEpochError):
    """Operation requires an open repository."""


class ImprovementEpochBoundsError(ImprovementEpochError, ValueError):
    """A payload or count bound was exceeded."""


class ImprovementEpochConflictError(ImprovementEpochError):
    """Stage, challenger, or identity conflict."""


class ImprovementEpochNotFoundError(ImprovementEpochError, LookupError):
    """Requested epoch or record is absent."""


class DuckDBUnavailableError(ImprovementEpochError):
    """Optional DuckDB dependency is not installed."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class EpochMode(str, Enum):
    OFF = "off"
    OBSERVE = "observe"
    SHADOW = "shadow"
    ASSIST = "assist"
    CANARY = "canary"
    AUTOMATIC = "automatic"

    @classmethod
    def coerce(cls, value: Any) -> "EpochMode":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        for item in cls:
            if item.value == raw:
                return item
        raise ImprovementEpochBoundsError(f"unsupported epoch mode: {value!r}")


class EpochStage(str, Enum):
    BASELINE = "baseline"
    PROPOSE = "propose"
    SHADOW = "shadow"
    EVALUATE = "evaluate"
    REJECT = "reject"
    RETAIN = "retain"
    CANARY = "canary"
    RECHECK = "recheck"
    PROMOTE = "promote"
    ROLLBACK = "rollback"
    REFILL = "refill"
    STOP = "stop"

    @classmethod
    def coerce(cls, value: Any) -> "EpochStage":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        for item in cls:
            if item.value == raw:
                return item
        raise ImprovementEpochBoundsError(f"unsupported epoch stage: {value!r}")


class EpochStatus(str, Enum):
    OPEN = "open"
    ACTIVE = "active"
    TERMINAL = "terminal"
    ROLLED_BACK = "rolled_back"

    @classmethod
    def coerce(cls, value: Any) -> "EpochStatus":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        for item in cls:
            if item.value == raw:
                return item
        raise ImprovementEpochBoundsError(f"unsupported epoch status: {value!r}")


class ChallengerStatus(str, Enum):
    REGISTERED = "registered"
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    ROLLED_BACK = "rolled_back"

    @classmethod
    def coerce(cls, value: Any) -> "ChallengerStatus":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        for item in cls:
            if item.value == raw:
                return item
        raise ImprovementEpochBoundsError(
            f"unsupported challenger status: {value!r}"
        )


TERMINAL_STAGES: Final[frozenset[EpochStage]] = frozenset(
    {
        EpochStage.STOP,
        EpochStage.REJECT,
        EpochStage.PROMOTE,
    }
)

_ALLOWED_TRANSITIONS: Final[Mapping[EpochStage, frozenset[EpochStage]]] = {
    EpochStage.BASELINE: frozenset(
        {EpochStage.PROPOSE, EpochStage.STOP, EpochStage.REJECT}
    ),
    EpochStage.PROPOSE: frozenset(
        {EpochStage.SHADOW, EpochStage.EVALUATE, EpochStage.REJECT, EpochStage.STOP}
    ),
    EpochStage.SHADOW: frozenset(
        {EpochStage.EVALUATE, EpochStage.RETAIN, EpochStage.REJECT, EpochStage.STOP}
    ),
    EpochStage.EVALUATE: frozenset(
        {
            EpochStage.RETAIN,
            EpochStage.CANARY,
            EpochStage.REJECT,
            EpochStage.ROLLBACK,
            EpochStage.STOP,
        }
    ),
    EpochStage.RETAIN: frozenset(
        {EpochStage.CANARY, EpochStage.PROMOTE, EpochStage.REFILL, EpochStage.STOP}
    ),
    EpochStage.CANARY: frozenset(
        {EpochStage.RECHECK, EpochStage.PROMOTE, EpochStage.ROLLBACK, EpochStage.STOP}
    ),
    EpochStage.RECHECK: frozenset(
        {EpochStage.PROMOTE, EpochStage.ROLLBACK, EpochStage.STOP}
    ),
    EpochStage.PROMOTE: frozenset({EpochStage.REFILL, EpochStage.STOP}),
    EpochStage.ROLLBACK: frozenset({EpochStage.REFILL, EpochStage.STOP}),
    EpochStage.REFILL: frozenset({EpochStage.STOP}),
    EpochStage.REJECT: frozenset({EpochStage.STOP}),
    EpochStage.STOP: frozenset(),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def duckdb_available() -> bool:
    """Return whether the optional duckdb package can be imported."""

    try:
        import duckdb  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    maximum: int = MAX_TEXT_BYTES,
) -> str:
    text = str(value or "").strip()
    if "\x00" in text:
        raise ImprovementEpochError(f"{name} contains NUL")
    if required and not text:
        raise ImprovementEpochError(f"{name} is required")
    if len(text.encode("utf-8")) > maximum:
        raise ImprovementEpochBoundsError(f"{name} exceeds {maximum} bytes")
    return text


def _safe_id(value: Any, name: str) -> str:
    text = _text(value, name, maximum=MAX_ID_BYTES)
    if not _SAFE_ID_RE.match(text):
        raise ImprovementEpochBoundsError(f"{name} is not a closed identity token")
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ImprovementEpochBoundsError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ImprovementEpochBoundsError(f"{name} must be a positive integer")
    return value


def _sha256_hex(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> str:
    try:
        return canonical_json_bytes(value).decode("utf-8")
    except ValueError:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )


def _bounded_body(
    body: Mapping[str, Any] | None,
    *,
    redact: bool,
    depth: int = 0,
    name: str = "body",
) -> dict[str, Any]:
    if depth > MAX_RECURSION_DEPTH:
        raise ImprovementEpochBoundsError(
            f"{name} exceeds recursion depth {MAX_RECURSION_DEPTH}"
        )
    raw = dict(body or {})
    cleaned = redact_mapping(raw) if redact else raw
    if not isinstance(cleaned, dict):
        raise ImprovementEpochError(f"{name} must project to an object")
    encoded = _canonical_json(cleaned).encode("utf-8")
    if len(encoded) > MAX_BODY_BYTES:
        raise ImprovementEpochBoundsError(
            f"{name} exceeds the {MAX_BODY_BYTES}-byte bound"
        )
    return cleaned


def _row_mapping(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return {str(key): row[key] for key in row}
    try:
        keys = list(row.keys())  # type: ignore[attr-defined]
    except Exception:
        return {}
    return {str(key): row[key] for key in keys}


def _split_sql_statements(sql_text: str) -> list[str]:
    statements: list[str] = []
    for chunk in str(sql_text).split(";"):
        statement = chunk.strip()
        if not statement or statement.startswith("--"):
            continue
        lines = [
            line
            for line in statement.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        if lines:
            statements.append("\n".join(lines))
    return statements


def _identity(prefix: str, material: Mapping[str, Any]) -> str:
    digest = _sha256_hex(_canonical_json(dict(material)).encode("utf-8"))
    return f"{prefix}:{digest[7:39]}"


def _load_json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    try:
        value = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImprovementEpoch:
    """One self-improvement epoch row."""

    epoch_id: str
    mode: str
    stage: str
    status: str
    created_at: str
    updated_at: str
    content_digest: str
    program_id: str = ""
    policy_id: str = ""
    repository_id: str = ""
    tree_id: str = ""
    objective_id: str = ""
    baseline_digest: str = ""
    body: Mapping[str, Any] = MappingProxyType({})
    schema: str = IMPROVEMENT_EPOCH_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "epoch_id": self.epoch_id,
            "program_id": self.program_id,
            "policy_id": self.policy_id,
            "mode": self.mode,
            "stage": self.stage,
            "status": self.status,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_id": self.objective_id,
            "baseline_digest": self.baseline_digest,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "body": dict(self.body),
            "content_digest": self.content_digest,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class EpochChallenger:
    """Isolated challenger bound to ordinary coordination identities."""

    challenger_id: str
    epoch_id: str
    worktree_id: str
    session_id: str
    lease_id: str
    created_at: str
    content_digest: str
    fencing_epoch: int = 0
    status: str = ChallengerStatus.REGISTERED.value
    body: Mapping[str, Any] = MappingProxyType({})
    schema: str = EPOCH_CHALLENGER_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "challenger_id": self.challenger_id,
            "epoch_id": self.epoch_id,
            "worktree_id": self.worktree_id,
            "session_id": self.session_id,
            "lease_id": self.lease_id,
            "fencing_epoch": self.fencing_epoch,
            "status": self.status,
            "created_at": self.created_at,
            "body": dict(self.body),
            "content_digest": self.content_digest,
            "identity_class": "ordinary_worktree_session_lease",
            "authority": AUTHORITY_CLASS,
        }


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class ImprovementEpochRepository:
    """DuckDB authority for improvement epochs and planned goals/tasks."""

    INTERFACE: Final[str] = IMPROVEMENT_EPOCH_REPOSITORY_INTERFACE

    def __init__(
        self,
        database_path: Path | str,
        *,
        snapshot_id: str = DEFAULT_SNAPSHOT_ID,
        auto_redact: bool = True,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for ImprovementEpochRepository; install "
                "the optional duckdb dependency"
            )
        self._path = Path(database_path)
        self._snapshot_id = _text(snapshot_id, "snapshot_id")
        self._auto_redact = bool(auto_redact)
        self._connection: Any | None = None
        self._lock = threading.RLock()
        self._closed = True

    # -- lifecycle -----------------------------------------------------------

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def snapshot_id(self) -> str:
        return self._snapshot_id

    @property
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    def open(self) -> "ImprovementEpochRepository":
        with self._lock:
            if self.is_open:
                return self
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = open_duckdb_connection(self._path)
            for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                connection.execute(statement)
            for key, value in (
                ("interface", IMPROVEMENT_EPOCH_REPOSITORY_INTERFACE),
                ("schema", IMPROVEMENT_EPOCH_REPOSITORY_SCHEMA),
                ("snapshot_id", self._snapshot_id),
                ("authority", AUTHORITY_CLASS),
                ("challenger_identity_class", "ordinary_worktree_session_lease"),
                ("goals_tasks_same_database", "true"),
            ):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO improvement_epoch_metadata(key, value)
                    VALUES (?, ?)
                    """,
                    [key, value],
                )
            self._connection = connection
            self._closed = False
            return self

    def close(self) -> None:
        with self._lock:
            connection = self._connection
            self._connection = None
            self._closed = True
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    def __enter__(self) -> "ImprovementEpochRepository":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require(self) -> Any:
        if not self.is_open or self._connection is None:
            raise ImprovementEpochNotOpenError(
                "ImprovementEpochRepository is not open"
            )
        return self._connection

    def _commit_if_idle(self, connection: Any) -> None:
        if getattr(connection, "in_transaction", False):
            return
        commit = getattr(connection, "commit", None)
        if callable(commit):
            try:
                commit()
            except Exception:
                pass

    def metadata(self) -> dict[str, str]:
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                "SELECT key, value FROM improvement_epoch_metadata ORDER BY key"
            ).fetchall()
            return {
                str(_row_mapping(row)["key"]): str(_row_mapping(row)["value"])
                for row in rows
            }

    # -- epochs --------------------------------------------------------------

    def create_epoch(
        self,
        *,
        mode: EpochMode | str = EpochMode.SHADOW,
        program_id: str = "",
        policy_id: str = "",
        repository_id: str = "",
        tree_id: str = "",
        objective_id: str = "",
        baseline_digest: str = "",
        body: Mapping[str, Any] | None = None,
        epoch_id: str | None = None,
        created_at: str | None = None,
        redact: bool | None = None,
    ) -> dict[str, Any]:
        """Open one improvement epoch at the baseline stage."""

        selected_mode = EpochMode.coerce(mode)
        if selected_mode is EpochMode.OFF:
            raise ImprovementEpochConflictError(
                "mode off refuses epoch creation"
            )
        do_redact = self._auto_redact if redact is None else bool(redact)
        payload = _bounded_body(body, redact=do_redact, name="body")
        stamp = _text(created_at or _utc_iso(), "created_at")
        material = {
            "mode": selected_mode.value,
            "program_id": _text(program_id, "program_id", required=False),
            "policy_id": _text(policy_id, "policy_id", required=False),
            "repository_id": _text(repository_id, "repository_id", required=False),
            "tree_id": _text(tree_id, "tree_id", required=False),
            "objective_id": _text(objective_id, "objective_id", required=False),
            "baseline_digest": _text(
                baseline_digest, "baseline_digest", required=False
            ),
            "created_at": stamp,
            "body": payload,
        }
        selected_id = _safe_id(epoch_id or _identity("epoch", material), "epoch_id")
        digest = _sha256_hex(
            _canonical_json({**material, "epoch_id": selected_id}).encode("utf-8")
        )
        with self._lock:
            connection = self._require()
            existing = connection.execute(
                "SELECT epoch_id FROM improvement_epochs WHERE epoch_id = ?",
                [selected_id],
            ).fetchone()
            if existing is not None:
                raise ImprovementEpochConflictError(
                    f"epoch already exists: {selected_id}"
                )
            connection.execute("BEGIN TRANSACTION")
            try:
                connection.execute(
                    """
                    INSERT INTO improvement_epochs(
                        epoch_id, program_id, policy_id, mode, stage, status,
                        repository_id, tree_id, objective_id, baseline_digest,
                        created_at, updated_at, body_json, content_digest
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        selected_id,
                        material["program_id"],
                        material["policy_id"],
                        selected_mode.value,
                        EpochStage.BASELINE.value,
                        EpochStatus.OPEN.value,
                        material["repository_id"],
                        material["tree_id"],
                        material["objective_id"],
                        material["baseline_digest"],
                        stamp,
                        stamp,
                        _canonical_json(payload),
                        digest,
                    ],
                )
                transition_id = self._append_transition(
                    connection,
                    epoch_id=selected_id,
                    from_stage=EpochStage.BASELINE,
                    to_stage=EpochStage.BASELINE,
                    reason="epoch_opened",
                    recorded_at=stamp,
                    body={"initial": True},
                )
                connection.execute("COMMIT")
            except Exception:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            self._commit_if_idle(connection)
            result = self.get_epoch(selected_id)
            result["initial_transition_id"] = transition_id
            return result

    def transition(
        self,
        epoch_id: str,
        to_stage: EpochStage | str,
        *,
        reason: str = "",
        body: Mapping[str, Any] | None = None,
        recorded_at: str | None = None,
        redact: bool | None = None,
    ) -> dict[str, Any]:
        """Advance an epoch through the closed stage machine."""

        selected_id = _safe_id(epoch_id, "epoch_id")
        target = EpochStage.coerce(to_stage)
        do_redact = self._auto_redact if redact is None else bool(redact)
        payload = _bounded_body(body, redact=do_redact, name="body")
        stamp = _text(recorded_at or _utc_iso(), "recorded_at")
        with self._lock:
            connection = self._require()
            epoch = self._load_epoch(connection, selected_id)
            current = EpochStage.coerce(epoch.stage)
            allowed = _ALLOWED_TRANSITIONS.get(current, frozenset())
            if target not in allowed and target is not current:
                raise ImprovementEpochConflictError(
                    f"illegal transition {current.value} -> {target.value}"
                )
            status = (
                EpochStatus.TERMINAL
                if target in TERMINAL_STAGES
                else EpochStatus.ACTIVE
            )
            if target is EpochStage.ROLLBACK:
                status = EpochStatus.ROLLED_BACK
            connection.execute("BEGIN TRANSACTION")
            try:
                transition_id = self._append_transition(
                    connection,
                    epoch_id=selected_id,
                    from_stage=current,
                    to_stage=target,
                    reason=_text(reason, "reason", required=False),
                    recorded_at=stamp,
                    body=payload,
                )
                connection.execute(
                    """
                    UPDATE improvement_epochs
                    SET stage = ?, status = ?, updated_at = ?
                    WHERE epoch_id = ?
                    """,
                    [target.value, status.value, stamp, selected_id],
                )
                connection.execute("COMMIT")
            except Exception:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            self._commit_if_idle(connection)
            result = self.get_epoch(selected_id)
            result["transition_id"] = transition_id
            return result

    def rollback_epoch(
        self,
        epoch_id: str,
        *,
        reason: str = "operator_rollback",
        body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Transition an epoch into the rollback stage and mark challenger."""

        result = self.transition(
            epoch_id,
            EpochStage.ROLLBACK,
            reason=reason,
            body=body,
        )
        with self._lock:
            connection = self._require()
            connection.execute(
                """
                UPDATE epoch_challengers
                SET status = ?
                WHERE epoch_id = ?
                """,
                [ChallengerStatus.ROLLED_BACK.value, _safe_id(epoch_id, "epoch_id")],
            )
            self._commit_if_idle(connection)
        result["challenger_status"] = ChallengerStatus.ROLLED_BACK.value
        return result

    def get_epoch(self, epoch_id: str) -> dict[str, Any]:
        selected = _safe_id(epoch_id, "epoch_id")
        with self._lock:
            connection = self._require()
            return self._load_epoch(connection, selected).to_dict()

    def list_epochs(
        self,
        *,
        status: EpochStatus | str | None = None,
        limit: int = DEFAULT_PAGE_LIMIT,
        offset: int = 0,
    ) -> tuple[dict[str, Any], ...]:
        page_limit = min(_positive_int(limit, "limit"), MAX_PAGE_LIMIT)
        page_offset = _nonneg_int(offset, "offset")
        with self._lock:
            connection = self._require()
            if status is not None:
                selected = EpochStatus.coerce(status).value
                rows = connection.execute(
                    """
                    SELECT * FROM improvement_epochs
                    WHERE status = ?
                    ORDER BY created_at ASC, epoch_id ASC
                    LIMIT ? OFFSET ?
                    """,
                    [selected, page_limit, page_offset],
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM improvement_epochs
                    ORDER BY created_at ASC, epoch_id ASC
                    LIMIT ? OFFSET ?
                    """,
                    [page_limit, page_offset],
                ).fetchall()
            return tuple(
                self._epoch_from_row(_row_mapping(row)).to_dict() for row in rows
            )

    def list_transitions(
        self,
        epoch_id: str,
        *,
        limit: int = MAX_TRANSITIONS_PER_EPOCH,
    ) -> tuple[dict[str, Any], ...]:
        selected = _safe_id(epoch_id, "epoch_id")
        page_limit = min(_positive_int(limit, "limit"), MAX_TRANSITIONS_PER_EPOCH)
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT * FROM epoch_transitions
                WHERE epoch_id = ?
                ORDER BY sequence ASC
                LIMIT ?
                """,
                [selected, page_limit],
            ).fetchall()
            results: list[dict[str, Any]] = []
            for row in rows:
                mapping = _row_mapping(row)
                results.append(
                    {
                        "schema": EPOCH_TRANSITION_SCHEMA,
                        "transition_id": mapping["transition_id"],
                        "epoch_id": mapping["epoch_id"],
                        "sequence": int(mapping["sequence"]),
                        "from_stage": mapping["from_stage"],
                        "to_stage": mapping["to_stage"],
                        "reason": mapping.get("reason") or "",
                        "recorded_at": mapping["recorded_at"],
                        "body": _load_json_object(mapping.get("body_json")),
                        "content_digest": mapping["content_digest"],
                    }
                )
            return tuple(results)

    # -- challengers ---------------------------------------------------------

    def register_challenger(
        self,
        epoch_id: str,
        *,
        worktree_id: str,
        session_id: str,
        lease_id: str,
        fencing_epoch: int = 0,
        body: Mapping[str, Any] | None = None,
        challenger_id: str | None = None,
        created_at: str | None = None,
        redact: bool | None = None,
    ) -> dict[str, Any]:
        """Register the single challenger using ordinary coordination IDs.

        Acceptance: challenger uses ordinary worktree/session/lease identities.
        Special or privileged challenger identity classes are rejected.
        """

        selected_epoch = _safe_id(epoch_id, "epoch_id")
        selected_worktree = _safe_id(worktree_id, "worktree_id")
        selected_session = _safe_id(session_id, "session_id")
        selected_lease = _safe_id(lease_id, "lease_id")
        fence = _nonneg_int(fencing_epoch, "fencing_epoch")
        do_redact = self._auto_redact if redact is None else bool(redact)
        payload = _bounded_body(body, redact=do_redact, name="body")
        # Reject non-ordinary identity markers if present in body.
        for key in ("identity_class", "privileged", "special_identity"):
            if key in payload:
                raise ImprovementEpochConflictError(
                    "challenger must use ordinary worktree/session/lease identities"
                )
        stamp = _text(created_at or _utc_iso(), "created_at")
        material = {
            "epoch_id": selected_epoch,
            "worktree_id": selected_worktree,
            "session_id": selected_session,
            "lease_id": selected_lease,
            "fencing_epoch": fence,
            "created_at": stamp,
            "body": payload,
        }
        selected_id = _safe_id(
            challenger_id or _identity("challenger", material),
            "challenger_id",
        )
        digest = _sha256_hex(
            _canonical_json({**material, "challenger_id": selected_id}).encode(
                "utf-8"
            )
        )
        with self._lock:
            connection = self._require()
            self._load_epoch(connection, selected_epoch)
            existing = connection.execute(
                "SELECT challenger_id FROM epoch_challengers WHERE epoch_id = ?",
                [selected_epoch],
            ).fetchone()
            if existing is not None:
                raise ImprovementEpochConflictError(
                    f"epoch already has a challenger (max {MAX_CHALLENGERS_PER_EPOCH})"
                )
            connection.execute(
                """
                INSERT INTO epoch_challengers(
                    challenger_id, epoch_id, worktree_id, session_id, lease_id,
                    fencing_epoch, status, created_at, body_json, content_digest
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    selected_id,
                    selected_epoch,
                    selected_worktree,
                    selected_session,
                    selected_lease,
                    fence,
                    ChallengerStatus.REGISTERED.value,
                    stamp,
                    _canonical_json(payload),
                    digest,
                ],
            )
            self._commit_if_idle(connection)
            return EpochChallenger(
                challenger_id=selected_id,
                epoch_id=selected_epoch,
                worktree_id=selected_worktree,
                session_id=selected_session,
                lease_id=selected_lease,
                fencing_epoch=fence,
                status=ChallengerStatus.REGISTERED.value,
                created_at=stamp,
                body=MappingProxyType(payload),
                content_digest=digest,
            ).to_dict()

    def get_challenger(self, epoch_id: str) -> dict[str, Any] | None:
        selected = _safe_id(epoch_id, "epoch_id")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM epoch_challengers WHERE epoch_id = ?",
                [selected],
            ).fetchone()
            if row is None:
                return None
            return self._challenger_from_row(_row_mapping(row)).to_dict()

    # -- planning goals/tasks in the same database ---------------------------

    def plan_goal(
        self,
        epoch_id: str,
        *,
        title: str,
        goal_cid: str | None = None,
        parent_goal_cid: str = "",
        status: str = "open",
        body: Mapping[str, Any] | None = None,
        created_at: str | None = None,
        redact: bool | None = None,
    ) -> dict[str, Any]:
        """Plan a self-improvement goal as an ordinary goal row."""

        selected_epoch = _safe_id(epoch_id, "epoch_id")
        selected_title = _text(title, "title")
        do_redact = self._auto_redact if redact is None else bool(redact)
        payload = _bounded_body(body, redact=do_redact, name="body")
        stamp = _text(created_at or _utc_iso(), "created_at")
        material = {
            "epoch_id": selected_epoch,
            "title": selected_title,
            "parent_goal_cid": _text(
                parent_goal_cid, "parent_goal_cid", required=False
            ),
            "status": _text(status, "status"),
            "created_at": stamp,
            "body": payload,
        }
        selected_goal = _safe_id(
            goal_cid or _identity("goal", material),
            "goal_cid",
        )
        digest = _sha256_hex(
            _canonical_json({**material, "goal_cid": selected_goal}).encode("utf-8")
        )
        with self._lock:
            connection = self._require()
            self._load_epoch(connection, selected_epoch)
            count = connection.execute(
                "SELECT COUNT(*) AS n FROM planned_goals WHERE epoch_id = ?",
                [selected_epoch],
            ).fetchone()
            count_map = _row_mapping(count)
            current = int(count_map.get("n") or count_map.get("c") or 0)
            if current >= MAX_GOALS_PER_EPOCH:
                raise ImprovementEpochBoundsError(
                    f"epoch goal bound {MAX_GOALS_PER_EPOCH} exceeded"
                )
            connection.execute(
                """
                INSERT INTO planned_goals(
                    goal_cid, epoch_id, title, status, parent_goal_cid,
                    created_at, body_json, content_digest
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    selected_goal,
                    selected_epoch,
                    selected_title,
                    material["status"],
                    material["parent_goal_cid"],
                    stamp,
                    _canonical_json(payload),
                    digest,
                ],
            )
            self._commit_if_idle(connection)
            return {
                "schema": PLANNED_GOAL_SCHEMA,
                "goal_cid": selected_goal,
                "epoch_id": selected_epoch,
                "title": selected_title,
                "status": material["status"],
                "parent_goal_cid": material["parent_goal_cid"],
                "created_at": stamp,
                "body": payload,
                "content_digest": digest,
                "same_database": True,
                "authority": AUTHORITY_CLASS,
            }

    def plan_task(
        self,
        epoch_id: str,
        *,
        goal_cid: str,
        title: str,
        task_cid: str | None = None,
        status: str = "ready",
        body: Mapping[str, Any] | None = None,
        created_at: str | None = None,
        redact: bool | None = None,
    ) -> dict[str, Any]:
        """Plan a self-improvement task bound to a planned goal."""

        selected_epoch = _safe_id(epoch_id, "epoch_id")
        selected_goal = _safe_id(goal_cid, "goal_cid")
        selected_title = _text(title, "title")
        do_redact = self._auto_redact if redact is None else bool(redact)
        payload = _bounded_body(body, redact=do_redact, name="body")
        stamp = _text(created_at or _utc_iso(), "created_at")
        material = {
            "epoch_id": selected_epoch,
            "goal_cid": selected_goal,
            "title": selected_title,
            "status": _text(status, "status"),
            "created_at": stamp,
            "body": payload,
        }
        selected_task = _safe_id(
            task_cid or _identity("task", material),
            "task_cid",
        )
        digest = _sha256_hex(
            _canonical_json({**material, "task_cid": selected_task}).encode("utf-8")
        )
        with self._lock:
            connection = self._require()
            self._load_epoch(connection, selected_epoch)
            goal = connection.execute(
                """
                SELECT goal_cid FROM planned_goals
                WHERE goal_cid = ? AND epoch_id = ?
                """,
                [selected_goal, selected_epoch],
            ).fetchone()
            if goal is None:
                raise ImprovementEpochNotFoundError(
                    f"planned goal not found for epoch: {selected_goal}"
                )
            count = connection.execute(
                "SELECT COUNT(*) AS n FROM planned_tasks WHERE epoch_id = ?",
                [selected_epoch],
            ).fetchone()
            count_map = _row_mapping(count)
            current = int(count_map.get("n") or count_map.get("c") or 0)
            if current >= MAX_TASKS_PER_EPOCH:
                raise ImprovementEpochBoundsError(
                    f"epoch task bound {MAX_TASKS_PER_EPOCH} exceeded"
                )
            connection.execute(
                """
                INSERT INTO planned_tasks(
                    task_cid, epoch_id, goal_cid, title, status,
                    created_at, body_json, content_digest
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    selected_task,
                    selected_epoch,
                    selected_goal,
                    selected_title,
                    material["status"],
                    stamp,
                    _canonical_json(payload),
                    digest,
                ],
            )
            self._commit_if_idle(connection)
            return {
                "schema": PLANNED_TASK_SCHEMA,
                "task_cid": selected_task,
                "epoch_id": selected_epoch,
                "goal_cid": selected_goal,
                "title": selected_title,
                "status": material["status"],
                "created_at": stamp,
                "body": payload,
                "content_digest": digest,
                "same_database": True,
                "authority": AUTHORITY_CLASS,
            }

    def list_planned_goals(self, epoch_id: str) -> tuple[dict[str, Any], ...]:
        selected = _safe_id(epoch_id, "epoch_id")
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT * FROM planned_goals
                WHERE epoch_id = ?
                ORDER BY created_at ASC, goal_cid ASC
                """,
                [selected],
            ).fetchall()
            results: list[dict[str, Any]] = []
            for row in rows:
                mapping = _row_mapping(row)
                results.append(
                    {
                        "schema": PLANNED_GOAL_SCHEMA,
                        "goal_cid": mapping["goal_cid"],
                        "epoch_id": mapping["epoch_id"],
                        "title": mapping["title"],
                        "status": mapping["status"],
                        "parent_goal_cid": mapping.get("parent_goal_cid") or "",
                        "created_at": mapping["created_at"],
                        "body": _load_json_object(mapping.get("body_json")),
                        "content_digest": mapping["content_digest"],
                        "same_database": True,
                    }
                )
            return tuple(results)

    def list_planned_tasks(self, epoch_id: str) -> tuple[dict[str, Any], ...]:
        selected = _safe_id(epoch_id, "epoch_id")
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT * FROM planned_tasks
                WHERE epoch_id = ?
                ORDER BY created_at ASC, task_cid ASC
                """,
                [selected],
            ).fetchall()
            results: list[dict[str, Any]] = []
            for row in rows:
                mapping = _row_mapping(row)
                results.append(
                    {
                        "schema": PLANNED_TASK_SCHEMA,
                        "task_cid": mapping["task_cid"],
                        "epoch_id": mapping["epoch_id"],
                        "goal_cid": mapping["goal_cid"],
                        "title": mapping["title"],
                        "status": mapping["status"],
                        "created_at": mapping["created_at"],
                        "body": _load_json_object(mapping.get("body_json")),
                        "content_digest": mapping["content_digest"],
                        "same_database": True,
                    }
                )
            return tuple(results)

    # -- metrics / rollouts / receipts ---------------------------------------

    def record_token_metrics(
        self,
        epoch_id: str,
        *,
        tokens_in: int = 0,
        tokens_out: int = 0,
        provider_calls: int = 0,
        cost_micros: int = 0,
        body: Mapping[str, Any] | None = None,
        recorded_at: str | None = None,
        redact: bool | None = None,
    ) -> dict[str, Any]:
        selected = _safe_id(epoch_id, "epoch_id")
        do_redact = self._auto_redact if redact is None else bool(redact)
        payload = _bounded_body(body, redact=do_redact, name="body")
        stamp = _text(recorded_at or _utc_iso(), "recorded_at")
        material = {
            "epoch_id": selected,
            "tokens_in": _nonneg_int(tokens_in, "tokens_in"),
            "tokens_out": _nonneg_int(tokens_out, "tokens_out"),
            "provider_calls": _nonneg_int(provider_calls, "provider_calls"),
            "cost_micros": _nonneg_int(cost_micros, "cost_micros"),
            "recorded_at": stamp,
            "body": payload,
        }
        metrics_id = _identity("metrics", material)
        digest = _sha256_hex(_canonical_json(material).encode("utf-8"))
        with self._lock:
            connection = self._require()
            self._load_epoch(connection, selected)
            connection.execute(
                """
                INSERT INTO epoch_token_metrics(
                    metrics_id, epoch_id, tokens_in, tokens_out,
                    provider_calls, cost_micros, recorded_at, body_json,
                    content_digest
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    metrics_id,
                    selected,
                    material["tokens_in"],
                    material["tokens_out"],
                    material["provider_calls"],
                    material["cost_micros"],
                    stamp,
                    _canonical_json(payload),
                    digest,
                ],
            )
            self._commit_if_idle(connection)
            return {
                "schema": EPOCH_TOKEN_METRICS_SCHEMA,
                "metrics_id": metrics_id,
                "epoch_id": selected,
                "tokens_in": material["tokens_in"],
                "tokens_out": material["tokens_out"],
                "provider_calls": material["provider_calls"],
                "cost_micros": material["cost_micros"],
                "recorded_at": stamp,
                "body": payload,
                "content_digest": digest,
            }

    def record_rollout(
        self,
        epoch_id: str,
        *,
        mode: EpochMode | str,
        status: str = "scheduled",
        body: Mapping[str, Any] | None = None,
        created_at: str | None = None,
        redact: bool | None = None,
    ) -> dict[str, Any]:
        selected = _safe_id(epoch_id, "epoch_id")
        selected_mode = EpochMode.coerce(mode)
        do_redact = self._auto_redact if redact is None else bool(redact)
        payload = _bounded_body(body, redact=do_redact, name="body")
        stamp = _text(created_at or _utc_iso(), "created_at")
        material = {
            "epoch_id": selected,
            "mode": selected_mode.value,
            "status": _text(status, "status"),
            "created_at": stamp,
            "body": payload,
        }
        rollout_id = _identity("rollout", material)
        digest = _sha256_hex(_canonical_json(material).encode("utf-8"))
        with self._lock:
            connection = self._require()
            self._load_epoch(connection, selected)
            connection.execute(
                """
                INSERT INTO epoch_rollouts(
                    rollout_id, epoch_id, mode, status, created_at,
                    body_json, content_digest
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    rollout_id,
                    selected,
                    selected_mode.value,
                    material["status"],
                    stamp,
                    _canonical_json(payload),
                    digest,
                ],
            )
            self._commit_if_idle(connection)
            return {
                "schema": EPOCH_ROLLOUT_SCHEMA,
                "rollout_id": rollout_id,
                "epoch_id": selected,
                "mode": selected_mode.value,
                "status": material["status"],
                "created_at": stamp,
                "body": payload,
                "content_digest": digest,
            }

    def record_receipt(
        self,
        epoch_id: str,
        *,
        kind: str,
        status: str = "recorded",
        body: Mapping[str, Any] | None = None,
        recorded_at: str | None = None,
        redact: bool | None = None,
    ) -> dict[str, Any]:
        selected = _safe_id(epoch_id, "epoch_id")
        selected_kind = _safe_id(kind, "kind")
        do_redact = self._auto_redact if redact is None else bool(redact)
        payload = _bounded_body(body, redact=do_redact, name="body")
        stamp = _text(recorded_at or _utc_iso(), "recorded_at")
        material = {
            "epoch_id": selected,
            "kind": selected_kind,
            "status": _text(status, "status"),
            "recorded_at": stamp,
            "body": payload,
        }
        receipt_id = _identity("receipt", material)
        digest = _sha256_hex(_canonical_json(material).encode("utf-8"))
        with self._lock:
            connection = self._require()
            self._load_epoch(connection, selected)
            connection.execute(
                """
                INSERT INTO epoch_receipts(
                    receipt_id, epoch_id, kind, status, recorded_at,
                    body_json, content_digest
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    receipt_id,
                    selected,
                    selected_kind,
                    material["status"],
                    stamp,
                    _canonical_json(payload),
                    digest,
                ],
            )
            self._commit_if_idle(connection)
            return {
                "schema": EPOCH_RECEIPT_SCHEMA,
                "receipt_id": receipt_id,
                "epoch_id": selected,
                "kind": selected_kind,
                "status": material["status"],
                "recorded_at": stamp,
                "body": payload,
                "content_digest": digest,
            }

    # -- internal ------------------------------------------------------------

    def _append_transition(
        self,
        connection: Any,
        *,
        epoch_id: str,
        from_stage: EpochStage,
        to_stage: EpochStage,
        reason: str,
        recorded_at: str,
        body: Mapping[str, Any],
    ) -> str:
        row = connection.execute(
            """
            SELECT COALESCE(MAX(sequence), 0) AS max_seq
            FROM epoch_transitions
            WHERE epoch_id = ?
            """,
            [epoch_id],
        ).fetchone()
        sequence = int(_row_mapping(row).get("max_seq") or 0) + 1
        if sequence > MAX_TRANSITIONS_PER_EPOCH:
            raise ImprovementEpochBoundsError(
                f"epoch transition bound {MAX_TRANSITIONS_PER_EPOCH} exceeded"
            )
        material = {
            "epoch_id": epoch_id,
            "sequence": sequence,
            "from_stage": from_stage.value,
            "to_stage": to_stage.value,
            "reason": reason,
            "recorded_at": recorded_at,
            "body": dict(body),
        }
        transition_id = _identity("transition", material)
        digest = _sha256_hex(_canonical_json(material).encode("utf-8"))
        connection.execute(
            """
            INSERT INTO epoch_transitions(
                transition_id, epoch_id, sequence, from_stage, to_stage,
                reason, recorded_at, body_json, content_digest
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                transition_id,
                epoch_id,
                sequence,
                from_stage.value,
                to_stage.value,
                reason,
                recorded_at,
                _canonical_json(dict(body)),
                digest,
            ],
        )
        return transition_id

    def _load_epoch(self, connection: Any, epoch_id: str) -> ImprovementEpoch:
        row = connection.execute(
            "SELECT * FROM improvement_epochs WHERE epoch_id = ?",
            [epoch_id],
        ).fetchone()
        if row is None:
            raise ImprovementEpochNotFoundError(f"epoch not found: {epoch_id}")
        return self._epoch_from_row(_row_mapping(row))

    def _epoch_from_row(self, row: Mapping[str, Any]) -> ImprovementEpoch:
        return ImprovementEpoch(
            epoch_id=str(row["epoch_id"]),
            program_id=str(row.get("program_id") or ""),
            policy_id=str(row.get("policy_id") or ""),
            mode=str(row["mode"]),
            stage=str(row["stage"]),
            status=str(row["status"]),
            repository_id=str(row.get("repository_id") or ""),
            tree_id=str(row.get("tree_id") or ""),
            objective_id=str(row.get("objective_id") or ""),
            baseline_digest=str(row.get("baseline_digest") or ""),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            body=MappingProxyType(_load_json_object(row.get("body_json"))),
            content_digest=str(row["content_digest"]),
        )

    def _challenger_from_row(self, row: Mapping[str, Any]) -> EpochChallenger:
        return EpochChallenger(
            challenger_id=str(row["challenger_id"]),
            epoch_id=str(row["epoch_id"]),
            worktree_id=str(row["worktree_id"]),
            session_id=str(row["session_id"]),
            lease_id=str(row["lease_id"]),
            fencing_epoch=int(row.get("fencing_epoch") or 0),
            status=str(row.get("status") or ChallengerStatus.REGISTERED.value),
            created_at=str(row["created_at"]),
            body=MappingProxyType(_load_json_object(row.get("body_json"))),
            content_digest=str(row["content_digest"]),
        )


def open_improvement_epoch_repository(
    database_path: Path | str,
    *,
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    auto_redact: bool = True,
) -> ImprovementEpochRepository:
    """Open and initialize an :class:`ImprovementEpochRepository`."""

    return ImprovementEpochRepository(
        database_path,
        snapshot_id=snapshot_id,
        auto_redact=auto_redact,
    ).open()


__all__ = (
    "AUTHORITY_CLASS",
    "ChallengerStatus",
    "DEFAULT_SNAPSHOT_ID",
    "EPOCH_CHALLENGER_SCHEMA",
    "EPOCH_RECEIPT_SCHEMA",
    "EPOCH_ROLLOUT_SCHEMA",
    "EPOCH_TOKEN_METRICS_SCHEMA",
    "EPOCH_TRANSITION_SCHEMA",
    "EpochChallenger",
    "EpochMode",
    "EpochStage",
    "EpochStatus",
    "IMPROVEMENT_EPOCH_REPOSITORY_INTERFACE",
    "IMPROVEMENT_EPOCH_REPOSITORY_SCHEMA",
    "IMPROVEMENT_EPOCH_SCHEMA",
    "ImprovementEpoch",
    "ImprovementEpochBoundsError",
    "ImprovementEpochConflictError",
    "ImprovementEpochError",
    "ImprovementEpochNotFoundError",
    "ImprovementEpochNotOpenError",
    "ImprovementEpochRepository",
    "MAX_CHALLENGERS_PER_EPOCH",
    "MAX_GOALS_PER_EPOCH",
    "MAX_TASKS_PER_EPOCH",
    "PLANNED_GOAL_SCHEMA",
    "PLANNED_TASK_SCHEMA",
    "REDACTION_MARKER",
    "TERMINAL_STAGES",
    "DuckDBUnavailableError",
    "duckdb_available",
    "open_improvement_epoch_repository",
)
