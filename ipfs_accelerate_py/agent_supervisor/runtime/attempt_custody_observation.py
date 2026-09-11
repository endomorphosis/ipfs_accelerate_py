"""Bounded observations made by an admitted execution reader, never settlement.

The caller supplies its existing typed readers. This module opens no database,
resolves no credentials and creates no grant. In particular, a control owner
relaying this record does not become the execution or coordination authority.
Opaque digests identify retained input; they are not independently verified
proofs. Empty result tables do not establish that a callback never ran.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

SCHEMA = "ipfs_accelerate_py/attempt-custody-observation@1"
MAX_BYTES = 6144
MAX_PHASES = 16
IDENTITY_TEXT = ("task_cid", "claim_id", "attempt_id", "lease_id", "owner_session_id")
IDENTITY_NUMBERS = ("attempt_number", "fencing_token", "fence_epoch")
IDENTITY = (*IDENTITY_TEXT, *IDENTITY_NUMBERS)
DENIALS = ("task_authority", "completion_authority", "source_transition_authority",
           "settlement_authority", "retry_authorized", "references_verified")


class AttemptObservationUnavailable(RuntimeError):
    def __init__(self):
        super().__init__("exact attempt custody observation unavailable")


def _bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _text(value: Any) -> bool:
    return type(value) is str and 0 < len(value) <= 512 and bool(value.strip())


def _number(value: Any, *, minimum: int = 1) -> bool:
    return type(value) is int and minimum <= value < 2**63


def _hash(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_observation(value: Any) -> dict[str, Any]:
    """Validate the closed relay format, without authenticating its producer."""
    try:
        keys = {"schema", "attempt", "execution", "claim", "coordination_attempt",
                "phases", "receipt_counts", "observation_sha256", "callback_outcome",
                "reader_kind", *DENIALS}
        if (type(value) is not dict or set(value) != keys
                or len(_bytes(value)) > MAX_BYTES or value["schema"] != SCHEMA
                or value["reader_kind"] != "daemon_typed_readers"
                or value["callback_outcome"] != "unknown"
                or any(value[name] is not False for name in DENIALS)):
            raise AttemptObservationUnavailable()
        identity = value["attempt"]
        if (type(identity) is not dict or set(identity) != set(IDENTITY)
                or any(not _text(identity[k]) for k in IDENTITY_TEXT)
                or any(not _number(identity[k]) for k in IDENTITY_NUMBERS)):
            raise AttemptObservationUnavailable()
        for name, text_fields, number_fields, hashes in (
            ("execution", ("status", "committed_phase"), ("revision",), ("body_sha256",)),
            ("claim", ("state",), ("revision", "expires_at_ms"), ("body_sha256",)),
            ("coordination_attempt", ("status",), ("revision",), ()),
        ):
            row = value[name]
            if (type(row) is not dict or set(row) != {*text_fields, *number_fields, *hashes}
                    or any(not _text(row[k]) for k in text_fields)
                    or any(not _number(row[k], minimum=0 if k == "expires_at_ms" else 1)
                           for k in number_fields)
                    or any(not _hash(row[k]) for k in hashes)):
                raise AttemptObservationUnavailable()
        phases = value["phases"]
        if type(phases) is not list or not 1 <= len(phases) <= MAX_PHASES:
            raise AttemptObservationUnavailable()
        revisions = []
        for row in phases:
            if (type(row) is not dict
                    or set(row) != {"phase", "revision", "committed_at_ms",
                                    "fencing_token", "fence_epoch", "body_sha256"}
                    or not _text(row["phase"]) or not _hash(row["body_sha256"])
                    or any(not _number(row[k], minimum=0 if k == "committed_at_ms" else 1)
                           for k in ("revision", "committed_at_ms", "fencing_token", "fence_epoch"))
                    or any(row[k] != identity[k] for k in ("fencing_token", "fence_epoch"))):
                raise AttemptObservationUnavailable()
            revisions.append(row["revision"])
        if (revisions != sorted(set(revisions))
                or len({p["phase"] for p in phases}) != len(phases)
                or phases[-1]["revision"] != value["execution"]["revision"]
                or phases[-1]["phase"] != value["execution"]["committed_phase"]):
            raise AttemptObservationUnavailable()
        counts = value["receipt_counts"]
        if (type(counts) is not dict
                or set(counts) != {"provider_invocation_count", "effect_claim_count"}
                or any(not _number(n, minimum=0) for n in counts.values())):
            raise AttemptObservationUnavailable()
        if (not _hash(value["observation_sha256"])
                or value["observation_sha256"] != _digest(
                    {k: v for k, v in value.items() if k != "observation_sha256"})):
            raise AttemptObservationUnavailable()
        # Copy nested containers so later caller mutation cannot alter a relay.
        return json.loads(_bytes(value))
    except Exception:
        raise AttemptObservationUnavailable() from None


def observe_attempt(daemon: Any, attempt: Any) -> dict[str, Any]:
    """Read the exact retained attempt twice using only the lane's own APIs.

    Matching reads detect revision/body/phase changes, not a distributed
    transaction. Settlement must independently revalidate all authorities at
    its mutation boundary. A missing or inconsistent row remains unavailable.
    """
    try:
        expected = {k: getattr(attempt, k) for k in IDENTITY}
        if expected["owner_session_id"] != daemon.owner_session_id:
            raise AttemptObservationUnavailable()

        def read():
            execution = daemon.get_attempt(expected["attempt_id"])
            claim = daemon.coordinator.get_task_claim(expected["claim_id"])
            coordinated = daemon.coordinator.get_task_attempt(expected["attempt_id"])
            if execution is None or claim is None or coordinated is None:
                raise AttemptObservationUnavailable()
            records = [item.to_dict() for item in (execution, claim, coordinated)]
            for record, names in zip(records, (IDENTITY, IDENTITY,
                    tuple(k for k in IDENTITY if k not in {"claim_id", "lease_id"}))):
                if any(type(record.get(k)) is not type(expected[k]) or record.get(k) != expected[k]
                       for k in names):
                    raise AttemptObservationUnavailable()
            phases = daemon.phase_history(expected["attempt_id"])
            if type(phases) is not list or not 1 <= len(phases) <= MAX_PHASES:
                raise AttemptObservationUnavailable()
            counts = daemon._attempt_execution_evidence_counts(expected["attempt_id"])
            # Snapshot full reader results now, including bodies that are only
            # hashed on the wire. Shared mutable dicts must not hide a change.
            raw = _bytes([*records, phases, counts])
            if len(raw) > 1024 * 1024:
                raise AttemptObservationUnavailable()
            return raw

        started = time.monotonic()
        first = read()
        if first != read() or time.monotonic() - started > 5:
            raise AttemptObservationUnavailable()
        execution, claim, coordinated, phases, counts = json.loads(first)
        result = {
            "schema": SCHEMA, "attempt": expected,
            "reader_kind": "daemon_typed_readers",
            "execution": {**{k: execution[k] for k in ("status", "committed_phase", "revision")},
                          "body_sha256": _digest(execution["body"])},
            "claim": {**{k: claim[k] for k in ("state", "revision", "expires_at_ms")},
                      "body_sha256": _digest(claim["body"])},
            "coordination_attempt": {k: coordinated[k] for k in ("status", "revision")},
            "phases": [{**{k: row[k] for k in ("phase", "revision", "committed_at_ms",
                                              "fencing_token", "fence_epoch")},
                        "body_sha256": _digest(row["body"])}
                       for row in sorted(phases, key=lambda row: row["revision"])],
            "receipt_counts": counts, "callback_outcome": "unknown",
            **{name: False for name in DENIALS},
        }
        result["observation_sha256"] = _digest(result)
        return validate_observation(result)
    except Exception:
        raise AttemptObservationUnavailable() from None
