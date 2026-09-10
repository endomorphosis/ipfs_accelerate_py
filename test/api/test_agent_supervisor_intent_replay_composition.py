"""Replay keeps published repair events and producer evidence timestamps exact."""

from copy import deepcopy

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.completion_projection_repair import (
    COUNTER,
    EVENT,
    apply_projection,
    prepare_repair,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    IntentEventType,
    IntentRepositoryIntegrityError,
)
from test.api.test_agent_supervisor_completion_projection_repair import legacy_task
from test.api.test_agent_supervisor_intent_repository import _repo, _seed_graph


def _record_repair(repo, *, event_type=EVENT, invalid_body=False):
    cid, revision, body = legacy_task(repo)
    with repo._connection(write=True) as connection:
        repair = prepare_repair(
            connection,
            task_cid=cid,
            expected_revision=revision,
            expected_body_cid=content_identity(body),
        )
        apply_projection(connection, repair)
        if invalid_body:
            # Keep an admitted envelope, but violate the exact projection CAS.
            repair = deepcopy(repair)
            repair["after_body_cid"] = "foreign-body"
        event = repo._append_event(
            connection, event_type=event_type, subject_id=cid, task_cid=cid,
            body=repair,
        )
    return cid, event


def test_strict_replay_admits_existing_exact_completion_projection_repair(tmp_path):
    with _repo(tmp_path) as repo:
        cid, _ = _record_repair(repo)
        before = repo.get_task(cid)
        with repo._connection(write=True) as connection:
            receipts = connection.execute(
                "SELECT * FROM completion_receipts ORDER BY receipt_cid"
            ).fetchall()
            repo._rebuild_projections_from_events_on(connection, strict=True)
            assert connection.execute(
                "SELECT * FROM completion_receipts ORDER BY receipt_cid"
            ).fetchall() == receipts
        assert repo.get_task(cid) == before
        assert before["body"][COUNTER] == 2
        assert COUNTER not in before["body"]["completion_receipt"]


@pytest.mark.parametrize("strict", [False, True])
def test_replay_does_not_silently_skip_invalid_published_repair(tmp_path, strict):
    with _repo(tmp_path) as repo:
        cid, _ = _record_repair(repo, invalid_body=True)
        before = repo.get_task(cid)
        with pytest.raises(IntentRepositoryIntegrityError, match="repair replay CAS"):
            if strict:
                with repo._connection(write=True) as connection:
                    repo._rebuild_projections_from_events_on(connection, strict=True)
            else:
                repo.rebuild_projections_from_events()
        assert repo.get_task(cid) == before


@pytest.mark.parametrize("corruption", ["cid", "envelope", "sequence"])
def test_known_repair_still_requires_exact_admitted_event(tmp_path, corruption):
    with _repo(tmp_path) as repo:
        cid, event = _record_repair(repo)
        before = repo.get_task(cid)
        with repo._connection(write=True) as connection:
            if corruption == "cid":
                connection.execute(
                    "UPDATE domain_events SET event_id='forged' WHERE event_id=?",
                    [event.event_id],
                )
            elif corruption == "envelope":
                connection.execute(
                    "UPDATE domain_events SET recorded_at='2020-01-01T00:00:00Z' "
                    "WHERE event_id=?", [event.event_id],
                )
            else:
                connection.execute(
                    "UPDATE domain_events SET sequence=sequence+1 WHERE event_id=?",
                    [event.event_id],
                )
        message = {"cid": "content identity", "envelope": "envelope", "sequence": "sequence"}
        with pytest.raises(IntentRepositoryIntegrityError, match=message[corruption]):
            with repo._connection(write=True) as connection:
                repo._rebuild_projections_from_events_on(connection, strict=True)
        assert repo.get_task(cid) == before


def test_strict_replay_does_not_admit_unknown_repair_type(tmp_path):
    with _repo(tmp_path) as repo:
        cid, _ = _record_repair(repo, event_type=EVENT + ".unreviewed")
        before = repo.get_task(cid)
        with pytest.raises(IntentRepositoryIntegrityError, match="unsupported admitted"):
            with repo._connection(write=True) as connection:
                repo._rebuild_projections_from_events_on(connection, strict=True)
        assert repo.get_task(cid) == before


@pytest.mark.parametrize(
    "created_at",
    [None, "", "not-a-time", "2026-09-10T00:00:00", "2026-09-10T00:00:00+00:00"],
)
def test_admitted_evidence_cannot_fall_back_to_replay_or_envelope_time(tmp_path, created_at):
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        payload = dict(
            evidence_id="evidence:invalid-time", task_cid=ids["task_a"],
            evidence_kind="validation", digest="digest:invalid-time", body={},
        )
        if created_at is not None:
            payload["created_at"] = created_at
        with repo._connection(write=True) as connection:
            repo._append_event(
                connection, event_type=IntentEventType.EVIDENCE_RECORDED,
                subject_id=payload["evidence_id"], task_cid=ids["task_a"], body=payload,
            )
        before = repo.get_task(ids["task_a"])
        with pytest.raises(IntentRepositoryIntegrityError, match="evidence recorded created_at"):
            with repo._connection(write=True) as connection:
                repo._rebuild_projections_from_events_on(connection, strict=True)
        assert repo.get_task(ids["task_a"]) == before
        with repo._connection(write=False) as connection:
            assert connection.execute(
                "SELECT 1 FROM evidence_nodes WHERE evidence_id=?", [payload["evidence_id"]]
            ).fetchone() is None


def test_evidence_replay_preserves_exact_producer_time(tmp_path):
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        payload = dict(
            evidence_id="evidence:historical", task_cid=ids["task_a"],
            evidence_kind="validation", digest="digest:historical", body={},
            created_at="2020-01-01T00:00:00Z",
        )
        with repo._connection(write=True) as connection:
            receipt = repo._append_event(
                connection, event_type=IntentEventType.EVIDENCE_RECORDED,
                subject_id=payload["evidence_id"], task_cid=ids["task_a"], body=payload,
            )
            assert receipt.recorded_at != payload["created_at"]
            repo._rebuild_projections_from_events_on(connection, strict=True)
            assert connection.execute(
                "SELECT created_at FROM evidence_nodes WHERE evidence_id=?",
                [payload["evidence_id"]],
            ).fetchone()[0] == payload["created_at"]
