"""Bounded advisory retry guidance; never proposal or execution authority."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "supervisor-prior-seed-guidance@1"
MAX_BYTES = 65_536
MAX_GUIDANCE_BYTES = 16_384
BINDING_FIELDS = frozenset({"schema", "task_id", "canonical_task_cid", "canonical_task_key",
    "board_namespace", "attempt", "event_source"})
AUTHORITY_FIELDS = frozenset({"proof_authority", "execution_authority", "completion_authority"})
RECORD_FIELDS = BINDING_FIELDS | AUTHORITY_FIELDS | {
    "guidance", "guidance_sha256", "status", "candidate_worktree"}


def decode_guidance_record(payload: bytes, *, artifact_path: Path) -> dict:
    """Apply one bounded, unambiguous wire contract in recovery and replay."""
    if type(payload) is not bytes or not 0 < len(payload) <= MAX_BYTES:
        raise ValueError("bounded prior seed guidance bytes required")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate prior seed guidance key")
            result[key] = value
        return result

    try:
        record = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_pairs)
    except (RecursionError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("malformed prior seed guidance encoding") from error
    return validate_guidance_record(record, artifact_path=artifact_path)


def guidance_artifact_path(binding: dict) -> Path:
    if type(binding) is not dict or set(binding) != BINDING_FIELDS or binding["schema"] != SCHEMA:
        raise ValueError("closed prior seed guidance binding required")
    if type(binding["attempt"]) is not int or binding["attempt"] < 1:
        raise ValueError("exact positive prior seed attempt required")
    for field in BINDING_FIELDS - {"attempt"}:
        value = binding[field]
        if type(value) is not str or not 0 < len(value.encode("utf-8")) <= 4096:
            raise ValueError("bounded prior seed identity required")
    event_source = Path(binding["event_source"])
    if not event_source.is_absolute() or event_source.resolve() != event_source:
        raise ValueError("canonical supervisor event state required")
    name = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return event_source.parent / "seed_recovery_guidance" / (name + ".json")


def validate_guidance_record(record: dict, *, artifact_path: Path) -> dict:
    """Check a detached record; callers separately bind current task and attempt."""
    if type(record) is not dict or set(record) != RECORD_FIELDS:
        raise ValueError("closed prior seed guidance record required")
    binding = {field: record[field] for field in BINDING_FIELDS}
    expected = guidance_artifact_path(binding)
    artifact = Path(artifact_path)
    if artifact != expected or artifact.resolve() != artifact:
        raise ValueError("prior seed guidance path differs from identity binding")
    guidance = record["guidance"]
    if (type(guidance) is not str or not 0 < len(guidance.encode("utf-8")) <= MAX_GUIDANCE_BYTES
            or record["guidance_sha256"] != hashlib.sha256(guidance.encode("utf-8")).hexdigest()
            or type(record["status"]) is not str or record["status"] not in {"pending", "consumed"}
            or any(record[field] is not False for field in AUTHORITY_FIELDS)):
        raise ValueError("invalid or authoritative prior seed guidance")
    candidate = record["candidate_worktree"]
    if (type(candidate) is not str or not candidate or len(candidate.encode("utf-8")) > 4096
            or not Path(candidate).is_absolute() or Path(candidate).resolve() != Path(candidate)
            or artifact.is_relative_to(Path(candidate))
            or Path(binding["event_source"]).is_relative_to(Path(candidate))):
        raise ValueError("prior seed guidance must remain outside candidate")
    if len(json.dumps(record, sort_keys=True, allow_nan=False).encode("utf-8")) > MAX_BYTES:
        raise ValueError("prior seed guidance record exceeds its bound")
    return dict(record)
