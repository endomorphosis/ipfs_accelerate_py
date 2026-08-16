"""DuckDB authority for v3 run heads and effect recovery.

JSON handles are useful portable projections, but they are deliberately not a
source of liveness or write authority.  This module keeps the mutable head and
the three-phase effect journal in one flock-serialised DuckDB transaction.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import exclusive_file_lock


RUN_REGISTRY_BACKEND_SCHEMA = "ipfs_accelerate_py/agent-supervisor/run-registry-duckdb@1"


class RunRegistryBackendError(RuntimeError):
    pass


class RunRevisionConflictError(RunRegistryBackendError):
    """A stale writer tried to advance a run revision."""


class EffectRecoveryError(RunRegistryBackendError):
    pass


@dataclass(frozen=True)
class RunRevisionCAS:
    run_id: str
    expected_revision: int
    expected_head_cid: str


@dataclass(frozen=True)
class DurableRunHead:
    run_id: str
    run_revision: int
    handle_cid: str
    state: str
    health: str
    process_cid: str = ""
    process_birth_identity: str = ""
    event_cursor: str = ""
    updated_at_ms: int = 0

    def __post_init__(self) -> None:
        if not self.run_id or self.run_revision < 1 or not self.handle_cid:
            raise RunRegistryBackendError("run head requires run_id, revision, and handle")
        # A PID is intentionally not represented: only a durable birth receipt
        # can support process adoption/liveness.
        if self.state == "running" and (
            not self.process_cid
            or not self.process_birth_identity
            or self.health in {"", "unknown"}
        ):
            raise RunRegistryBackendError(
                "running head requires process CID, birth identity, and observed health"
            )

    @property
    def content_id(self) -> str:
        return cid_for_dag_json({"schema": RUN_REGISTRY_BACKEND_SCHEMA + "/head@1", **asdict(self)})


@dataclass(frozen=True)
class ImmutableRunEpoch:
    run_id: str
    run_revision: int
    head_cid: str
    event_cursor: str
    exported_at_ms: int

    @property
    def content_id(self) -> str:
        return cid_for_dag_json({"schema": RUN_REGISTRY_BACKEND_SCHEMA + "/epoch@1", **asdict(self)})


@dataclass(frozen=True)
class EffectJournalEntry:
    run_id: str
    effect_key: str
    phase: str
    intent_cid: str
    effect_cid: str = ""
    receipt_cid: str = ""


class DuckDBRunRegistryBackend:
    """Single-writer DuckDB run truth with deterministic effect continuation."""

    def __init__(self, path: str | Path) -> None:
        candidate = Path(path)
        self.path = candidate if candidate.suffix == ".duckdb" else candidate / "run_registry.duckdb"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._initialize()

    def _connect(self) -> Any:
        try:
            import duckdb
        except ModuleNotFoundError as exc:
            raise RunRegistryBackendError("DuckDB is required for durable run authority") from exc
        return duckdb.connect(str(self.path))

    def _initialize(self) -> None:
        with exclusive_file_lock(self.lock_path):
            conn = self._connect()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS run_heads (
                      run_id VARCHAR PRIMARY KEY, run_revision BIGINT NOT NULL,
                      handle_cid VARCHAR NOT NULL, payload_json VARCHAR NOT NULL,
                      updated_at_ms BIGINT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS run_effects (
                      run_id VARCHAR NOT NULL, effect_key VARCHAR NOT NULL,
                      phase VARCHAR NOT NULL, intent_cid VARCHAR NOT NULL,
                      effect_cid VARCHAR NOT NULL, receipt_cid VARCHAR NOT NULL,
                      PRIMARY KEY (run_id, effect_key)
                    );
                """)
            finally:
                conn.close()

    def create(self, head: DurableRunHead) -> DurableRunHead:
        with exclusive_file_lock(self.lock_path):
            conn = self._connect()
            try:
                existing = conn.execute("SELECT run_id FROM run_heads WHERE run_id = ?", [head.run_id]).fetchone()
                if existing:
                    raise RunRevisionConflictError("run already has an authoritative head")
                self._insert_head(conn, head)
                return head
            finally:
                conn.close()

    def reconstruct(self, run_id: str) -> DurableRunHead:
        with exclusive_file_lock(self.lock_path):
            conn = self._connect()
            try:
                row = conn.execute("SELECT payload_json FROM run_heads WHERE run_id = ?", [run_id]).fetchone()
                if row is None:
                    raise RunRegistryBackendError("authoritative run head not found")
                return DurableRunHead(**json.loads(row[0]))
            finally:
                conn.close()

    def adopt_healthy_matching_process(
        self,
        *,
        run_id: str,
        process_cid: str,
        process_birth_identity: str,
        healthy: bool,
    ) -> DurableRunHead:
        """Adopt only the exact process recorded by durable run truth.

        Callers may use a PID to *find* a candidate, but a PID never enters
        this contract: reuse is authorized only by the persisted process CID,
        OS birth identity, and a fresh health observation.
        """
        head = self.reconstruct(run_id)
        if (
            not healthy
            or head.state != "running"
            or head.health != "healthy"
            or head.process_cid != process_cid
            or head.process_birth_identity != process_birth_identity
        ):
            raise EffectRecoveryError("process is not the healthy exact replay target")
        return head

    def compare_and_swap(self, cas: RunRevisionCAS, next_head: DurableRunHead) -> DurableRunHead:
        if next_head.run_id != cas.run_id or next_head.run_revision != cas.expected_revision + 1:
            raise RunRegistryBackendError("CAS must advance exactly one matching run revision")
        with exclusive_file_lock(self.lock_path):
            conn = self._connect()
            try:
                conn.execute("BEGIN TRANSACTION")
                row = conn.execute("SELECT run_revision, handle_cid FROM run_heads WHERE run_id = ?", [cas.run_id]).fetchone()
                if row is None or int(row[0]) != cas.expected_revision or row[1] != cas.expected_head_cid:
                    conn.execute("ROLLBACK")
                    raise RunRevisionConflictError("authoritative run revision changed")
                conn.execute("UPDATE run_heads SET run_revision=?, handle_cid=?, payload_json=?, updated_at_ms=? WHERE run_id=?", [next_head.run_revision, next_head.handle_cid, json.dumps(asdict(next_head), sort_keys=True), next_head.updated_at_ms, next_head.run_id])
                conn.execute("COMMIT")
                return next_head
            except Exception:
                try: conn.execute("ROLLBACK")
                except Exception: pass
                raise
            finally:
                conn.close()

    def _insert_head(self, conn: Any, head: DurableRunHead) -> None:
        conn.execute("INSERT INTO run_heads VALUES (?, ?, ?, ?, ?)", [head.run_id, head.run_revision, head.handle_cid, json.dumps(asdict(head), sort_keys=True), head.updated_at_ms])

    def record_intent(self, *, run_id: str, effect_key: str, intent_cid: str) -> EffectJournalEntry:
        """Persist intent before a side effect; exact replays return it unchanged."""
        return self._effect_transition(run_id, effect_key, "intent", intent_cid=intent_cid)

    def record_effect(self, *, run_id: str, effect_key: str, effect_cid: str) -> EffectJournalEntry:
        return self._effect_transition(run_id, effect_key, "effect", effect_cid=effect_cid)

    def record_receipt(self, *, run_id: str, effect_key: str, receipt_cid: str) -> EffectJournalEntry:
        return self._effect_transition(run_id, effect_key, "receipt", receipt_cid=receipt_cid)

    def _effect_transition(self, run_id: str, key: str, phase: str, **value: str) -> EffectJournalEntry:
        if not run_id or not key:
            raise EffectRecoveryError("effect journal requires run_id and idempotency key")
        with exclusive_file_lock(self.lock_path):
            conn = self._connect()
            try:
                row = conn.execute("SELECT phase, intent_cid, effect_cid, receipt_cid FROM run_effects WHERE run_id=? AND effect_key=?", [run_id, key]).fetchone()
                if row is None:
                    if phase != "intent": raise EffectRecoveryError("effect cannot precede persisted intent")
                    entry = EffectJournalEntry(run_id, key, "intent", value["intent_cid"])
                    conn.execute("INSERT INTO run_effects VALUES (?, ?, ?, ?, ?, ?)", [run_id, key, entry.phase, entry.intent_cid, "", ""])
                    return entry
                entry = EffectJournalEntry(run_id, key, *row)
                order = {"intent": 0, "effect": 1, "receipt": 2}
                if order[phase] < order[entry.phase]:
                    return entry
                if phase == entry.phase:
                    supplied = value.get(phase + "_cid", "")
                    existing = getattr(entry, phase + "_cid") if phase != "intent" else entry.intent_cid
                    if supplied and supplied != existing: raise EffectRecoveryError("idempotency key was reused for a different effect")
                    return entry
                if order[phase] != order[entry.phase] + 1: raise EffectRecoveryError("effect phases must be contiguous")
                values = {**asdict(entry), **value, "phase": phase}
                updated = EffectJournalEntry(**values)
                conn.execute("UPDATE run_effects SET phase=?, intent_cid=?, effect_cid=?, receipt_cid=? WHERE run_id=? AND effect_key=?", [updated.phase, updated.intent_cid, updated.effect_cid, updated.receipt_cid, run_id, key])
                return updated
            finally:
                conn.close()

    def continuation_for(self, *, run_id: str, effect_key: str) -> str:
        """The only restart decision: emit, observe, or finish the receipt."""
        with exclusive_file_lock(self.lock_path):
            conn = self._connect()
            try:
                row = conn.execute("SELECT phase FROM run_effects WHERE run_id=? AND effect_key=?", [run_id, effect_key]).fetchone()
                if row is None: return "persist_intent"
                return {"intent": "perform_effect", "effect": "record_receipt", "receipt": "already_complete"}[row[0]]
            finally:
                conn.close()

    def export_epoch(self, run_id: str, *, exported_at_ms: int | None = None) -> ImmutableRunEpoch:
        head = self.reconstruct(run_id)
        return ImmutableRunEpoch(run_id, head.run_revision, head.content_id, head.event_cursor, int(time.time() * 1000) if exported_at_ms is None else int(exported_at_ms))


__all__ = ["DuckDBRunRegistryBackend", "DurableRunHead", "EffectJournalEntry", "EffectRecoveryError", "ImmutableRunEpoch", "RUN_REGISTRY_BACKEND_SCHEMA", "RunRegistryBackendError", "RunRevisionCAS", "RunRevisionConflictError"]
