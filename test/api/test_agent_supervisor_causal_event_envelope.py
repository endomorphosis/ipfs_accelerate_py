"""Closed envelope compatibility with the real durable event producer."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.event_log import (
    CAUSAL_EVENT_ENVELOPE_FIELDS,
    LEGACY_EVENT_ENVELOPE_FIELDS,
    append_jsonl_event,
    strict_event_envelope_fields,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalBridgeError,
    DatabasePortalExecutionBridge,
)


def _identity(event):
    body = {k: v for k, v in event.items() if k != "event_id"}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _explicit_causal_events(events):
    """Construct independent valid native records from immutable legacy inputs."""
    causal = []
    for event in events:
        row = {**event, "causal_parent_ids": [causal[-1]["event_id"]] if causal else [],
               "coalescing_key": "", "coalescing_forbidden": False}
        row["previous_event_id"] = causal[-1]["event_id"] if causal else ""
        row["event_id"] = _identity(row)
        causal.append(row)
    return causal


def test_real_producer_envelope_and_legacy_chain_keep_full_hash(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    legacy = [append_jsonl_event(path, "source", {"value": 1}),
              append_jsonl_event(path, "callback", {"value": 2})]
    raw = path.read_bytes()
    assert all(strict_event_envelope_fields(event) == LEGACY_EVENT_ENVELOPE_FIELDS for event in legacy)
    assert DatabasePortalExecutionBridge._verified_event_chain(SimpleNamespace(events=path)) == legacy
    causal = _explicit_causal_events(legacy)
    assert all(strict_event_envelope_fields(event) == LEGACY_EVENT_ENVELOPE_FIELDS | CAUSAL_EVENT_ENVELOPE_FIELDS for event in causal)
    assert causal[1]["causal_parent_ids"] == [causal[0]["event_id"]]
    encoded = b"".join((json.dumps(row) + "\n").encode() for row in causal)
    assert DatabasePortalExecutionBridge._verified_event_chain(SimpleNamespace(events=path), payload=encoded) == causal
    assert path.read_bytes() == raw


@pytest.mark.parametrize("field,value", [
    ("causal_parent_ids", None),
    ("causal_parent_ids", "sha256:" + "a" * 64),
    ("causal_parent_ids", ()),
    ("causal_parent_ids", [True]),
    ("causal_parent_ids", ["sha256:" + "a" * 64] * 2),
    ("causal_parent_ids", ["sha256:" + f"{i:064x}" for i in range(257)]),
    ("causal_parent_ids", ["not-an-event-id"]),
    ("coalescing_key", False),
    ("coalescing_key", "x\x00"),
    ("coalescing_key", " x "),
    ("coalescing_key", "x" * 257),
    ("coalescing_key", "é" * 129),
    ("coalescing_key", "\ud800"),
    ("coalescing_forbidden", 1),
    ("coalescing_forbidden", "false"),
])
def test_malformed_causal_envelope_is_denied_without_coercion(field, value):
    event = {"causal_parent_ids": [], "coalescing_key": "", "coalescing_forbidden": False}
    event[field] = value
    with pytest.raises(ValueError):
        strict_event_envelope_fields(event)


@pytest.mark.parametrize("missing", sorted(CAUSAL_EVENT_ENVELOPE_FIELDS))
def test_partial_causal_envelope_is_never_legacy(missing):
    event = {"causal_parent_ids": [], "coalescing_key": "", "coalescing_forbidden": False}
    del event[missing]
    with pytest.raises(ValueError, match="incomplete"):
        strict_event_envelope_fields(event)


def test_exact_causal_bounds_are_accepted():
    event = {"causal_parent_ids": ["sha256:" + f"{i:064x}" for i in range(256)],
             "coalescing_key": "é" * 128, "coalescing_forbidden": False}
    assert strict_event_envelope_fields(event) == LEGACY_EVENT_ENVELOPE_FIELDS | CAUSAL_EVENT_ENVELOPE_FIELDS


@pytest.mark.parametrize("mutation", ["stale_hash", "future_parent", "self_parent", "partial"])
def test_chain_causal_mutations_never_gain_authority(tmp_path: Path, mutation):
    path = tmp_path / "events.jsonl"
    first = append_jsonl_event(path, "source", {"value": 1})
    second = append_jsonl_event(path, "callback", {"value": 2})
    raw = path.read_bytes()
    first, second = _explicit_causal_events([first, second])
    forged = dict(second)
    if mutation == "stale_hash":
        forged["coalescing_key"] = "changed"
    elif mutation == "future_parent":
        forged["causal_parent_ids"] = ["sha256:" + "a" * 64]
        forged["event_id"] = _identity(forged)
    elif mutation == "self_parent":
        forged["causal_parent_ids"] = [forged["event_id"]]
    else:
        del forged["coalescing_key"]
        forged["event_id"] = _identity(forged)
    encoded = (json.dumps(first) + "\n" + json.dumps(forged) + "\n").encode()
    with pytest.raises(DatabasePortalBridgeError):
        DatabasePortalExecutionBridge._verified_event_chain(SimpleNamespace(events=path), payload=encoded)
    assert path.read_bytes() == raw


@pytest.mark.parametrize("event_type", ["lease_renewed", "fence_advanced", "proof_checked", "receipt_recorded"])
def test_safety_event_producer_cannot_be_claimed_coalescible(tmp_path, event_type):
    event = append_jsonl_event(tmp_path / "events.jsonl", event_type, {
        "causal_parent_ids": [], "coalescing_key": "", "coalescing_forbidden": True,
    })
    assert event["event_id"] == _identity(event)
    assert event["coalescing_forbidden"] is True and event["coalescing_key"] == ""
    strict_event_envelope_fields(event)
    with pytest.raises(ValueError, match="producer policy"):
        strict_event_envelope_fields({**event, "coalescing_forbidden": False})
    with pytest.raises(ValueError, match="producer policy"):
        strict_event_envelope_fields({**event, "coalescing_key": "nonempty"})
