"""Portable reader/relay contract; native lane integration has separate tests."""

import copy
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import attempt_custody_observation as obs


@pytest.fixture
def readers():
    identity = {**{k: k + ":fixture" for k in obs.IDENTITY_TEXT},
                **{k: 4 for k in obs.IDENTITY_NUMBERS}}
    rows = {
        "execution": {**identity, "status": "running", "committed_phase": "context",
                      "revision": 2, "body": {"private": "do not relay"}},
        "claim": {**identity, "state": "accepted", "expires_at_ms": 100,
                  "revision": 1, "body": {}},
        "coordinated": {**identity, "status": "running", "revision": 1},
        "phases": [{"phase": phase, "revision": n, "committed_at_ms": n,
                    "fencing_token": 4, "fence_epoch": 4, "body": {}}
                   for n, phase in enumerate(("claimed", "context"), 1)],
        "counts": {"provider_invocation_count": 0, "effect_claim_count": 0},
    }

    def record(name):
        return SimpleNamespace(to_dict=lambda: copy.deepcopy(rows[name]))

    daemon = SimpleNamespace(
        owner_session_id=identity["owner_session_id"],
        get_attempt=lambda _: record("execution"),
        coordinator=SimpleNamespace(get_task_claim=lambda _: record("claim"),
                                    get_task_attempt=lambda _: record("coordinated")),
        phase_history=lambda _: copy.deepcopy(rows["phases"]),
        _attempt_execution_evidence_counts=lambda _: dict(rows["counts"]),
    )
    return daemon, SimpleNamespace(**identity), rows


def test_closed_observation_redacts_bodies_and_denies_all_authority(readers):
    daemon, attempt, rows = readers
    before = copy.deepcopy(rows)
    value = obs.observe_attempt(daemon, attempt)
    assert all(value[k] is False for k in obs.DENIALS)
    assert value["callback_outcome"] == "unknown"
    assert value["receipt_counts"]["provider_invocation_count"] == 0
    assert b"do not relay" not in obs._bytes(value)
    assert rows == before
    assert obs.validate_observation(value) == value
    clone = obs.validate_observation(value)
    clone["attempt"]["attempt_id"] = "changed"
    assert value["attempt"]["attempt_id"] == attempt.attempt_id


@pytest.mark.parametrize("row", ["execution", "claim", "coordinated"])
def test_exact_fence_required_for_every_reader(readers, row):
    daemon, attempt, rows = readers
    rows[row]["fencing_token"] = 5
    with pytest.raises(obs.AttemptObservationUnavailable):
        obs.observe_attempt(daemon, attempt)


def test_full_body_change_between_reads_is_unavailable(readers):
    daemon, attempt, rows = readers
    calls = []

    def phases(_):
        calls.append(True)
        # Same identity and revision, but retained bytes changed between reads.
        rows["execution"]["body"]["private"] = "changed"
        return copy.deepcopy(rows["phases"])

    daemon.phase_history = phases
    with pytest.raises(obs.AttemptObservationUnavailable):
        obs.observe_attempt(daemon, attempt)
    assert len(calls) == 2


@pytest.mark.parametrize("bad", ["retry", "bool_count", "phase_fence", "phase_revision", "extra", "digest"])
def test_even_rehashed_invalid_relays_remain_unavailable(readers, bad):
    daemon, attempt, _ = readers
    value = obs.observe_attempt(daemon, attempt)
    if bad == "retry":
        value["retry_authorized"] = True
    elif bad == "bool_count":
        value["receipt_counts"]["effect_claim_count"] = False
    elif bad == "phase_fence":
        value["phases"][0]["fence_epoch"] = 5
    elif bad == "phase_revision":
        value["phases"][-1]["revision"] = 3
    elif bad == "extra":
        value["proof"] = "unverified"
    value["observation_sha256"] = obs._digest({k: v for k, v in value.items() if k != "observation_sha256"})
    if bad == "digest":
        value["observation_sha256"] = "0" * 64
    with pytest.raises(obs.AttemptObservationUnavailable):
        obs.validate_observation(value)


def test_reader_exception_is_redacted_and_never_replayed(readers):
    daemon, attempt, _ = readers
    calls = []

    def failed(_):
        calls.append(True)
        raise RuntimeError("sensitive transport detail")

    daemon.get_attempt = failed
    with pytest.raises(obs.AttemptObservationUnavailable) as caught:
        obs.observe_attempt(daemon, attempt)
    assert "sensitive" not in str(caught.value)
    assert len(calls) == 1


def test_missing_reader_or_foreign_session_is_unavailable(readers):
    daemon, attempt, _ = readers
    daemon.owner_session_id = "session:other"
    with pytest.raises(obs.AttemptObservationUnavailable):
        obs.observe_attempt(daemon, attempt)
