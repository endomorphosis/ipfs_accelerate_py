"""Complete, bounded discovery through the native owner protocol."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.derived_coordination import (
    MAX_REQUEST_BYTES, DerivedCoordinationClient, DerivedDiscoveryLimitExceeded,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import TypedStateOwnerError
from test.api.test_agent_supervisor_derived_coordination import owner  # noqa: F401


def _source(number: int, *, long: bool = False) -> dict:
    return {
        "tree_id": "tree:discovery",
        "ast_cid": "cid:ast:" + ("a" * 490 if long else "") + str(number),
        "content_hash": "sha256:" + hashlib.sha256(str(number).encode()).hexdigest(),
        "state_root": "cid:state:" + ("b" * 490 if long else "") + str(number),
    }


def test_more_than_256_references_are_discovered_by_another_native_client(owner):
    _, _, attach = owner
    producer = attach()
    api = DerivedCoordinationClient(producer, repository_id="repo:one")
    try:
        expected = [
            api.call("record_reference", **_source(i))["result"]["reference"]
            for i in range(263)
        ]
        api.call("record_reference", **_source(0))
        api.call("record_reference", **{**_source(500), "tree_id": "tree:other"})
    finally:
        producer.close()
    sessions = []

    def connect():
        session = attach()
        sessions.append(session)
        return session

    consumer = DerivedCoordinationClient(connection_factory=connect, repository_id="repo:one")
    actual = list(consumer.iter_references(tree_id="tree:discovery", limit=31))
    assert actual == sorted(expected, key=lambda row: row["reference_id"])
    assert len(sessions) == 9
    other = DerivedCoordinationClient(connection_factory=lambda: attach("repo:two"), repository_id="repo:two")
    assert list(other.iter_references(tree_id="tree:discovery")) == []


def test_large_reference_pages_stay_bounded_without_losing_rows(owner):
    _, _, attach = owner
    connection = attach()
    api = DerivedCoordinationClient(connection, repository_id="repo:one")
    try:
        for i in range(256):
            api.call("record_reference", **_source(i, long=True))
        page = api.call("list_references", tree_id="tree:discovery")
        assert len(json.dumps(page, default=dict, sort_keys=True, separators=(",", ":")).encode()) <= MAX_REQUEST_BYTES
        assert 0 < len(page["result"]["references"]) < 256
        assert page["result"]["has_more"] is True
        assert len(list(api.iter_references(tree_id="tree:discovery"))) == 256
    finally:
        connection.close()


def test_page_budget_can_resume_and_artifact_discovery_remains_unverified(owner):
    _, _, attach = owner
    connection = attach()
    api = DerivedCoordinationClient(connection, repository_id="repo:one")
    try:
        for i in range(7):
            api.call("record_reference", **_source(i))
            api.call(
                "record_artifact", tree_id="tree:discovery", artifact_kind="proof_cache",
                input_digest="sha256:" + hashlib.sha256(str(i).encode()).hexdigest(),
                producer_id="datasets-producer", producer_revision="revision:one",
                parameters_digest="sha256:" + "b" * 64, artifact_cid=f"cid:proof:{i}",
            )
        iterator = api.iter_references(tree_id="tree:discovery", limit=3, max_pages=1)
        first = [next(iterator) for _ in range(3)]
        with pytest.raises(DerivedDiscoveryLimitExceeded) as caught:
            next(iterator)
        rest = list(api.iter_references(tree_id="tree:discovery", after=caught.value.next_cursor, limit=3))
        assert len({r["reference_id"] for r in first + rest}) == 7
        artifacts = list(api.iter_artifacts(tree_id="tree:discovery", artifact_kind="proof_cache", limit=2))
        assert len({row["artifact_key"] for row in artifacts}) == 7
        assert all("verified" not in row for row in artifacts)
    finally:
        connection.close()


@pytest.mark.parametrize("parameters", [
    {"limit": True}, {"limit": 0}, {"limit": 257}, {"after": "SELECT *"}, {"after": None},
])
def test_native_owner_rejects_invalid_pagination(owner, parameters):
    _, _, attach = owner
    connection = attach()
    try:
        with pytest.raises(TypedStateOwnerError):
            DerivedCoordinationClient(connection, repository_id="repo:one").call(
                "list_references", tree_id="tree:discovery", **parameters,
            )
    finally:
        connection.close()


@pytest.mark.parametrize("corruption", [
    "cursor", "more_type", "empty_more", "foreign_repo", "foreign_tree", "digest", "authority",
])
def test_bad_page_is_rejected_before_any_row_is_yielded(owner, corruption):
    _, _, attach = owner
    connection = attach()
    api = DerivedCoordinationClient(connection, repository_id="repo:one")
    try:
        for i in range(3):
            api.call("record_reference", **_source(i))
        response = json.loads(json.dumps(
            api.call("list_references", tree_id="tree:discovery", limit=2), default=dict,
        ))
    finally:
        connection.close()
    result = response["result"]
    if corruption == "cursor":
        result["next_cursor"] = "sha256:" + "0" * 64
    elif corruption == "more_type":
        result["has_more"] = 1
    elif corruption == "empty_more":
        result["references"] = []
    elif corruption == "foreign_repo":
        result["references"][-1]["repository_id"] = "repo:two"
    elif corruption == "foreign_tree":
        result["references"][-1]["tree_id"] = "tree:other"
    elif corruption == "digest":
        result["references"][-1]["state_root"] = "cid:tampered"
    else:
        response["completion_authority"] = True
    consumer = DerivedCoordinationClient(
        SimpleNamespace(derived_coordination=lambda _: response), repository_id="repo:one",
    )
    with pytest.raises(ValueError):
        next(consumer.iter_references(tree_id="tree:discovery", limit=2))


def test_transport_failure_closes_session_without_automatic_retry():
    calls, closed = [], []

    def unavailable(_):
        calls.append("request")
        raise ConnectionError("owner restarted")

    api = DerivedCoordinationClient(
        connection_factory=lambda: SimpleNamespace(
            derived_coordination=unavailable, close=lambda: closed.append(True),
        ), repository_id="repo:one",
    )
    with pytest.raises(ConnectionError):
        list(api.iter_references(tree_id="tree:discovery"))
    assert calls == ["request"]
    assert closed == [True]
