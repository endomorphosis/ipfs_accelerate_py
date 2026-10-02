"""Durable negative state for the Doctor gate, on its existing cache owner.

This domain stores dependency identities and reuse refusals, never proof
receipts. All connections come from FormalVerificationCache's serialized native
owner path. Root revocations also reject a late registration after invalidation;
reopening a gate cannot revive an old positive entry. No revocation is expired
or cleared by cache TTL/garbage collection.
"""
from __future__ import annotations

from contextlib import contextmanager
import json

from .formal_verification_contracts import canonical_json


class DoctorCacheStateError(ValueError):
    """Durable gate state is malformed or conflicts with its exact key."""


_TABLES = {
    "doctor_cache_control_meta": "schema_version INTEGER PRIMARY KEY",
    "doctor_cache_control_keys": "key_id VARCHAR PRIMARY KEY, key_json VARCHAR NOT NULL",
    "doctor_cache_control_roots": (
        "key_id VARCHAR NOT NULL, root_field VARCHAR NOT NULL, root_cid VARCHAR NOT NULL, "
        "PRIMARY KEY (key_id, root_field)"),
    "doctor_cache_control_revocations": (
        "root_field VARCHAR NOT NULL, root_cid VARCHAR NOT NULL, invalidated_at_ms BIGINT NOT NULL, "
        "reason VARCHAR NOT NULL, PRIMARY KEY (root_field, root_cid)"),
    "doctor_cache_control_tombstones": (
        "key_id VARCHAR NOT NULL, root_field VARCHAR NOT NULL, root_cid VARCHAR NOT NULL, "
        "invalidated_at_ms BIGINT NOT NULL, reason VARCHAR NOT NULL, "
        "PRIMARY KEY (key_id, root_field, root_cid)"),
    "doctor_cache_control_quarantine": (
        "key_id VARCHAR PRIMARY KEY, reason VARCHAR NOT NULL, recorded_at_ms BIGINT NOT NULL"),
    "doctor_cache_control_observations": (
        "key_id VARCHAR NOT NULL, receipt_id VARCHAR NOT NULL, PRIMARY KEY (key_id, receipt_id)"),
}


def _text(value):
    if type(value) is not str or not value.strip() or len(value.encode()) > 8192:
        raise DoctorCacheStateError("bounded nonempty state identity/reason required")
    return value


def _time(value):
    if type(value) is not int or value < 0:
        raise DoctorCacheStateError("nonnegative integer state timestamp required")
    return value


class DoctorCacheState:
    """Additive, transactional denial state sharing the formal cache database."""

    def __init__(self, cache):
        self._cache = cache
        with self._transaction(initialize=True) as cx:
            existing = {row[0] for row in cx.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()}
            found = existing.intersection(_TABLES)
            if found and found != set(_TABLES):
                raise DoctorCacheStateError("incomplete Doctor control schema")
            if not found:
                for name, columns in _TABLES.items():
                    cx.execute(f"CREATE TABLE {name} ({columns})")
                cx.execute("INSERT INTO doctor_cache_control_meta VALUES (1)")
                self._bootstrap_roots(cx)
            if [row[0] for row in cx.execute(
                "SELECT schema_version FROM doctor_cache_control_meta"
            ).fetchall()] != [1]:
                raise DoctorCacheStateError("unsupported Doctor control schema")
            cx.execute("""CREATE INDEX IF NOT EXISTS doctor_cache_control_root_lookup
                ON doctor_cache_control_roots(root_field, root_cid)""")

    def _bootstrap_roots(self, cx):
        """Migrate exact Doctor keys, without promoting stored receipt claims."""
        from .doctor_proof_cache import DoctorProofCacheKey

        after = ""
        while True:
            rows = cx.execute("""SELECT key_id, key_json FROM proof_cache_entries
                WHERE key_id>? ORDER BY key_id LIMIT 256""", [after]).fetchall()
            if not rows:
                break
            for row in rows:
                formal = json.loads(row[1])
                obligation = formal.get("obligation")
                if not isinstance(obligation, dict) or "doctor_semantic_key" not in obligation:
                    continue
                key = DoctorProofCacheKey.from_dict(obligation["doctor_semantic_key"])
                if (key.contains_private_material or key.to_formal_key().key_id != row[0]
                        or canonical_json(key.to_formal_key().to_dict()) != row[1]):
                    raise DoctorCacheStateError("legacy Doctor formal key failed exact reconstruction")
                self._register(cx, key)
            after = rows[-1][0]

    @contextmanager
    def _transaction(self, *, initialize=False):
        cx = self._cache._connect()
        try:
            if (getattr(cx, "_transport_mode", "") == "quack"
                    or getattr(cx, "_default_catalog", None)):
                raise DoctorCacheStateError(
                    "Doctor control requires a current transactional owner; "
                    "snapshot Quack transport needs a qualified typed owner command")
            cx.execute("BEGIN TRANSACTION")
            try:
                if not initialize:
                    versions = cx.execute("SELECT schema_version FROM doctor_cache_control_meta").fetchall()
                    if [row[0] for row in versions] != [1]:
                        raise DoctorCacheStateError("Doctor control schema changed")
                yield cx
                cx.execute("COMMIT")
            except BaseException:
                cx.execute("ROLLBACK")
                raise
        finally:
            cx.close()

    @staticmethod
    def _register(cx, key):
        if key.contains_private_material:
            raise DoctorCacheStateError("private Doctor key cannot enter durable state")
        body = canonical_json(key.to_dict())
        row = cx.execute("SELECT key_json FROM doctor_cache_control_keys WHERE key_id=?",
                         [key.key_id]).fetchone()
        roots = sorted(key.semantic_root_ids.items())
        if row is None:
            # Dangling rows must not be accepted as a new registration.
            if cx.execute("SELECT 1 FROM doctor_cache_control_roots WHERE key_id=? LIMIT 1",
                          [key.key_id]).fetchone() is not None:
                raise DoctorCacheStateError("orphaned Doctor dependency rows")
            cx.execute("INSERT INTO doctor_cache_control_keys VALUES (?, ?)", [key.key_id, body])
            for field, cid in roots:
                cx.execute("INSERT INTO doctor_cache_control_roots VALUES (?, ?, ?)",
                           [key.key_id, field, cid])
        else:
            if row[0] != body:
                raise DoctorCacheStateError("Doctor key canonical preimage differs")
            actual = [(row[0], row[1]) for row in cx.execute(
                "SELECT root_field, root_cid FROM doctor_cache_control_roots WHERE key_id=? ORDER BY root_field",
                [key.key_id]).fetchall()]
            if actual != roots:
                raise DoctorCacheStateError("Doctor dependency roots differ from canonical key")
        # Materialize late registrations against every already revoked root.
        cx.execute("""INSERT INTO doctor_cache_control_tombstones
            SELECT r.key_id, r.root_field, r.root_cid, v.invalidated_at_ms, v.reason
            FROM doctor_cache_control_roots r JOIN doctor_cache_control_revocations v
              ON r.root_field=v.root_field AND r.root_cid=v.root_cid
            WHERE r.key_id=? ON CONFLICT DO NOTHING""", [key.key_id])

    @staticmethod
    def _status(cx, key_id):
        result = {}
        row = cx.execute("SELECT reason, recorded_at_ms FROM doctor_cache_control_quarantine WHERE key_id=?",
                         [key_id]).fetchone()
        if row is not None:
            result.update(quarantine=_text(row[0]), recorded_at_ms=_time(row[1]))
        row = cx.execute("""SELECT t.root_field, t.root_cid, t.invalidated_at_ms, t.reason,
                v.invalidated_at_ms, v.reason
            FROM doctor_cache_control_tombstones t LEFT JOIN doctor_cache_control_revocations v
              ON t.root_field=v.root_field AND t.root_cid=v.root_cid
            WHERE t.key_id=? ORDER BY t.invalidated_at_ms, t.root_field, t.root_cid LIMIT 1""",
            [key_id]).fetchone()
        if row is not None:
            if row[2] != row[4] or row[3] != row[5]:
                raise DoctorCacheStateError("Doctor tombstone lost its revocation binding")
            result["tombstone"] = {"key_id": key_id, "root_field": _text(row[0]),
                "root_cid": _text(row[1]), "invalidated_at_ms": _time(row[2]), "reason": _text(row[3])}
        return result

    def check(self, key):
        with self._transaction() as cx:
            self._register(cx, key)
            return self._status(cx, key.key_id)

    def observe(self, key, receipt_id, *, now_ms, equivocation_reason):
        _text(receipt_id)
        with self._transaction() as cx:
            self._register(cx, key)
            blocked = self._status(cx, key.key_id)
            if blocked:
                return blocked
            previous = cx.execute(
                "SELECT receipt_id FROM doctor_cache_control_observations WHERE key_id=?",
                [key.key_id]).fetchall()
            if previous and any(row[0] != receipt_id for row in previous):
                self._quarantine(cx, key.key_id, equivocation_reason, now_ms)
            cx.execute("INSERT INTO doctor_cache_control_observations VALUES (?, ?) ON CONFLICT DO NOTHING",
                       [key.key_id, receipt_id])
            return self._status(cx, key.key_id)

    def invalidate(self, *, root_field, root_cid, reason, now_ms):
        from .doctor_proof_cache import DoctorProofCacheKey

        _text(root_field); _text(root_cid); _text(reason); _time(now_ms)
        with self._transaction() as cx:
            # Verify retained canonical keys before deriving descendant tombstones.
            rows = cx.execute("""SELECT k.key_id, k.key_json FROM doctor_cache_control_keys k
                JOIN doctor_cache_control_roots r ON r.key_id=k.key_id
                WHERE r.root_field=? AND r.root_cid=? ORDER BY k.key_id""",
                [root_field, root_cid]).fetchall()
            for row in rows:
                key = DoctorProofCacheKey.from_dict(json.loads(row[1]))
                if key.key_id != row[0] or canonical_json(key.to_dict()) != row[1]:
                    raise DoctorCacheStateError("Doctor dependency key failed reconstruction")
                self._register(cx, key)
            cx.execute("INSERT INTO doctor_cache_control_revocations VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
                       [root_field, root_cid, now_ms, reason])
            cx.execute("""INSERT INTO doctor_cache_control_tombstones
                SELECT r.key_id, r.root_field, r.root_cid, v.invalidated_at_ms, v.reason
                FROM doctor_cache_control_roots r JOIN doctor_cache_control_revocations v
                  ON r.root_field=v.root_field AND r.root_cid=v.root_cid
                WHERE r.root_field=? AND r.root_cid=? ON CONFLICT DO NOTHING""", [root_field, root_cid])
            return [dict(key_id=row[0], root_field=row[1], root_cid=row[2],
                         invalidated_at_ms=_time(row[3]), reason=_text(row[4])) for row in cx.execute(
                """SELECT key_id, root_field, root_cid, invalidated_at_ms, reason
                FROM doctor_cache_control_tombstones WHERE root_field=? AND root_cid=? ORDER BY key_id""",
                [root_field, root_cid]).fetchall()]

    @staticmethod
    def _quarantine(cx, key_id, reason, now_ms):
        cx.execute("INSERT INTO doctor_cache_control_quarantine VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
                   [_text(key_id), _text(reason), _time(now_ms)])

    def quarantine(self, key_id, *, reason, now_ms):
        with self._transaction() as cx:
            self._quarantine(cx, key_id, reason, now_ms)

    def state_for_id(self, key_id):
        with self._transaction() as cx:
            return self._status(cx, _text(key_id))
