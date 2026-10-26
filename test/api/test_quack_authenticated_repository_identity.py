"""Repository observations from real authenticated typed-owner sockets.

These private-DuckDB tests exercise a genuine gateway and TypedStateOwnerConnection.
They establish client identity compatibility only; they grant no native/Lake admit.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackClientIdentityError,
    QuackStateClient,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TYPED_STATE_OWNER_SOCKET_ENV,
    TYPED_STATE_OWNER_TOKEN_ENV,
    TypedStateOwnerConnection,
    TypedStateOwnerGateway,
)

_REPOSITORY = "repository:authenticated-private-fixture"
_MISSING = object()


@contextmanager
def _real_owner(
    tmp_path: Path, *, repository: Any = _REPOSITORY,
) -> Iterator[dict[str, Any]]:
    db = tmp_path / "control.duckdb"
    install_control_plane_schema(
        db, application_version="0.0.45", tool_version="1.5.2",
        owner_id="repository-identity-fixture",
    )
    seed = QuackStateClient(owner_id="repository-identity-seed")
    try:
        seed.attach(db, seed_generation=True)
        baseline = seed.session.store_identity
        generation = seed.load_generation()
    finally:
        seed.close()
    identity = {
        "server_id": "server:repository-identity-fixture",
        "store_id": generation.store_id,
        "database_uuid": generation.database_uuid,
        "schema_revision": generation.schema_revision,
        "schema_fingerprint": baseline.schema_fingerprint,
        "generation": generation.generation,
        "fence_epoch": generation.fence_epoch,
        "revision": generation.revision,
        "process_birth_id": generation.birth_id,
    }
    if repository is not _MISSING:
        identity["repository_id"] = repository
    connection = open_duckdb_connection(db)
    socket_path = tmp_path / "owner.sock"
    gateway = TypedStateOwnerGateway(
        connection=connection, socket_path=socket_path,
        store_id=generation.store_id, identity=identity,
    )
    state = {
        "gateway": gateway, "connection": connection, "identity": identity,
        "baseline": baseline, "socket": socket_path, "database": db,
    }
    try:
        gateway.start()
        yield state
    finally:
        try:
            state["gateway"].stop()
        finally:
            connection.close()


def _grant(state: dict[str, Any], client: QuackStateClient, monkeypatch: Any) -> None:
    token, _grant_record = state["gateway"].issue_grant(
        client_id=client.owner_id, process_birth_id=client.process_birth_id,
        allowed_operations=("whoami_metadata", "load_store_generation"),
        peer_pid=os.getpid(),
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_SOCKET_ENV, str(state["socket"]))
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)


def test_real_typed_owner_repository_is_observed_without_expectation(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    with _real_owner(tmp_path) as state:
        client = QuackStateClient(owner_id="repository-observer")
        try:
            _grant(state, client, monkeypatch)
            session = client.attach("quack:127.0.0.1:7777")
            assert type(client._adapter.raw) is TypedStateOwnerConnection
            actual = client._adapter.raw.identity
            assert actual["repository_id"] == _REPOSITORY
            assert session.server_id == actual["server_id"]
            assert session.generation == actual["generation"]
            assert session.store_identity.repository_id == actual["repository_id"]
            assert session.store_identity.database_uuid == actual["database_uuid"]
        finally:
            client.close()


@pytest.mark.parametrize("at_attach", [False, True])
def test_explicit_repository_expectation_cannot_overwrite_real_handshake(
    tmp_path: Path, monkeypatch: Any, at_attach: bool,
) -> None:
    with _real_owner(tmp_path) as state:
        expected = replace(state["baseline"], repository_id="repository:wrong")
        client = QuackStateClient(
            owner_id="repository-mismatch",
            expected_identity=None if at_attach else expected,
        )
        try:
            _grant(state, client, monkeypatch)
            with pytest.raises(QuackClientIdentityError, match="repository_id mismatch"):
                client.attach(
                    "quack:127.0.0.1:7777",
                    **({"expected_identity": expected} if at_attach else {}),
                )
            assert not client.attached
            assert client.session is None
        finally:
            client.close()


@pytest.mark.parametrize("at_attach", [False, True])
def test_matching_explicit_repository_expectation_preserves_real_identity(
    tmp_path: Path, monkeypatch: Any, at_attach: bool,
) -> None:
    with _real_owner(tmp_path) as state:
        expected = replace(state["baseline"], repository_id=_REPOSITORY)
        client = QuackStateClient(
            owner_id="repository-match",
            expected_identity=None if at_attach else expected,
        )
        try:
            _grant(state, client, monkeypatch)
            session = client.attach(
                "quack:127.0.0.1:7777",
                **({"expected_identity": expected} if at_attach else {}),
            )
            assert session.store_identity.repository_id == _REPOSITORY
        finally:
            client.close()


@pytest.mark.parametrize("repository", [_MISSING, ""], ids=["absent", "empty"])
def test_legacy_owner_without_repository_stays_explicitly_unbound(
    tmp_path: Path, monkeypatch: Any, repository: Any,
) -> None:
    with _real_owner(tmp_path, repository=repository) as state:
        client = QuackStateClient(owner_id="repository-legacy")
        try:
            _grant(state, client, monkeypatch)
            session = client.attach("quack:127.0.0.1:7777")
            assert session.store_identity.repository_id == "repository:local"
        finally:
            client.close()


@pytest.mark.parametrize("repository", [_MISSING, ""], ids=["absent", "empty"])
def test_legacy_owner_cannot_manufacture_expected_repository(
    tmp_path: Path, monkeypatch: Any, repository: Any,
) -> None:
    with _real_owner(tmp_path, repository=repository) as state:
        expected = replace(state["baseline"], repository_id=_REPOSITORY)
        client = QuackStateClient(owner_id="repository-unproved", expected_identity=expected)
        try:
            _grant(state, client, monkeypatch)
            with pytest.raises(QuackClientIdentityError, match="no repository identity"):
                client.attach("quack:127.0.0.1:7777")
            assert not client.attached
        finally:
            client.close()


@pytest.mark.parametrize("repository", [7, True, ["repository:invalid"]])
def test_real_owner_repository_must_be_a_string(
    tmp_path: Path, monkeypatch: Any, repository: Any,
) -> None:
    with _real_owner(tmp_path, repository=repository) as state:
        client = QuackStateClient(owner_id="repository-malformed")
        try:
            _grant(state, client, monkeypatch)
            with pytest.raises(QuackClientIdentityError, match="not a string"):
                client.attach("quack:127.0.0.1:7777")
            assert not client.attached
        finally:
            client.close()


def test_real_owner_reconnect_preserves_observed_repository(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    with _real_owner(tmp_path) as state:
        client = QuackStateClient(owner_id="repository-reconnect")
        try:
            _grant(state, client, monkeypatch)
            before = client.attach("quack:127.0.0.1:7777")
            _grant(state, client, monkeypatch)
            after = client.reconnect()
            assert after.session_id != before.session_id
            assert after.store_identity.repository_id == before.store_identity.repository_id == _REPOSITORY
            assert type(client._adapter.raw) is TypedStateOwnerConnection
        finally:
            client.close()


def test_real_owner_reconnect_refuses_changed_repository(
    tmp_path: Path, monkeypatch: Any,
) -> None:
    with _real_owner(tmp_path) as state:
        client = QuackStateClient(owner_id="repository-reconnect-mismatch")
        try:
            _grant(state, client, monkeypatch)
            client.attach("quack:127.0.0.1:7777")
            # Construct a second genuine owner against the same private DB.
            # Its actual admitted open reply advertises a different repository.
            state["gateway"].stop()
            second_identity = {**state["identity"], "repository_id": "repository:other-owner"}
            state["gateway"] = TypedStateOwnerGateway(
                connection=state["connection"], socket_path=state["socket"],
                store_id=second_identity["store_id"], identity=second_identity,
            )
            state["gateway"].start()
            _grant(state, client, monkeypatch)
            with pytest.raises(QuackClientIdentityError, match="repository_id mismatch"):
                client.reconnect()
            assert not client.attached
        finally:
            client.close()


def test_embedded_repository_fallback_is_unchanged(tmp_path: Path) -> None:
    db = tmp_path / "embedded.duckdb"
    install_control_plane_schema(
        db, application_version="0.0.45", tool_version="1.5.2",
        owner_id="repository-embedded-fixture",
    )
    client = QuackStateClient(owner_id="repository-embedded")
    try:
        session = client.attach(db, seed_generation=True)
        assert session.store_identity.repository_id == "repository:local"
    finally:
        client.close()


@pytest.mark.parametrize("at_attach", [False, True])
def test_embedded_explicit_repository_fallback_remains_compatible(
    tmp_path: Path, at_attach: bool,
) -> None:
    db = tmp_path / "embedded-expected.duckdb"
    install_control_plane_schema(
        db, application_version="0.0.45", tool_version="1.5.2",
        owner_id="repository-embedded-expected-fixture",
    )
    seed = QuackStateClient(owner_id="repository-embedded-expected-seed")
    try:
        seed.attach(db, seed_generation=True)
        expected = replace(seed.session.store_identity, repository_id=_REPOSITORY)
    finally:
        seed.close()
    client = QuackStateClient(
        owner_id="repository-embedded-expected",
        expected_identity=None if at_attach else expected,
    )
    try:
        session = client.attach(
            db, **({"expected_identity": expected} if at_attach else {}),
        )
        assert session.store_identity.repository_id == _REPOSITORY
    finally:
        client.close()
