"""Hermetic qualification of the sealed CASF bootstrap over typed Quack.

The test client never opens ``control.duckdb``.  The real migrated DuckDB
connection remains owned by :class:`QuackStateServer`; all client reads and
mutations cross its typed Unix-socket gateway using named operations.
"""

# Python 3.8 compatibility intentionally uses ``timezone.utc``.
# ruff: noqa: UP017

from __future__ import annotations

import base64
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.bootstrap_runtime import (
    BOOTSTRAP_PROFILE_SCHEMA,
    BootstrapAdmission,
    admit_bootstrap_federation,
    validate_bootstrap_profile,
)
from ipfs_accelerate_py.agent_supervisor.federation.contracts import (
    FederationAuthorizationDecision,
    FederationContractError,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnerLiveness,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    FakeQuackTransport,
    QuackStateServer,
    build_server,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    DEFAULT_QUACK_BETA_LIMITATIONS,
    ExtensionObservation,
    ParsedVersion,
    QuackCapabilityReport,
    QuackCapabilityStatus,
    default_compatibility_profile,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackStateClient,
    TransportMode,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TYPED_STATE_OWNER_SOCKET_ENV,
    TYPED_STATE_OWNER_TOKEN_ENV,
    build_control_plane_operation_catalog,
    catalog_fingerprint,
)

_EXTENSION_DIGEST = "sha256:" + ("ab" * 32)
_NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)


def _profile(**changes: Any) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "schema": BOOTSTRAP_PROFILE_SCHEMA,
        "tenant_id": "tenant:casf-bootstrap-test",
        "caller_did": "did:local:casf-bootstrap-test",
        "issuer_did": "did:local:casf-policy-owner-test",
        "audience": "agent-supervisor:casf-bootstrap-test",
        "policy_ref": "policy:casf-bootstrap-test",
        "policy_revision": 1,
        "requested_supervisor_profile": "profile:casf-first-tranche-test",
        "allowed_operations": [
            "federation.create",
            "supervisor.runtime.attest",
            "supervisor.transition",
            "event.wait",
            "event.delivery.record",
            "event.acknowledge",
        ],
        "allowed_effect": "effect.read",
        "risk_ceiling": "risk.low",
        "expires_at": "2031-01-01T00:00:00Z",
        "cpu_millis": 100_000,
        "processes": 2,
        "input_tokens": 20_000,
        "output_tokens": 5_000,
        "model_calls": 8,
        "maximum_supervisors": 1,
        "maximum_subagents": 1,
        "maximum_concurrent_subagents": 1,
    }
    profile.update(changes)
    return profile


def _capability() -> QuackCapabilityReport:
    profile = default_compatibility_profile()
    return QuackCapabilityReport(
        status=QuackCapabilityStatus.COMPATIBLE,
        profile=profile,
        duckdb_importable=True,
        duckdb_version="1.5.5",
        duckdb_version_parsed=ParsedVersion(1, 5, 5, raw="1.5.5"),
        platform_name="Linux",
        platform_machine="test",
        extension=ExtensionObservation(
            name="quack",
            installed=True,
            loaded=True,
            install_path="/qualified/quack.duckdb_extension",
            extension_version="test",
        ),
        extension_fingerprint=_EXTENSION_DIGEST,
        observed_functions=("quack_serve", "quack_query"),
        observed_surfaces=profile.required_surfaces,
        beta_limitations=DEFAULT_QUACK_BETA_LIMITATIONS,
    )


def _migrate(database: Path) -> Any:
    return install_control_plane_schema(
        database,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="owner:casf-bootstrap-migration-test",
    )


@contextmanager
def _typed_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[QuackStateServer, QuackStateClient, Any]]:
    """Yield the canonical repository while the server exclusively owns DuckDB."""

    server = build_server(
        database_path=tmp_path / "control.duckdb",
        state_dir=tmp_path / "owner",
        repository_id="repository:ipfs_accelerate_py",
        store_id="casf-bootstrap-test-v1",
        transport=FakeQuackTransport(),
        capability_probe=_capability,
        migrate=_migrate,
        connection_factory=open_duckdb_connection,
        owner_liveness_probe=lambda _birth: OwnerLiveness.DEAD,
    )
    identity = server.start()
    catalog = build_control_plane_operation_catalog()
    client_id = "client:casf-bootstrap-test"
    token = server.issue_typed_client_grant(
        client_id=client_id,
        process_birth_id=identity.process_birth_id,
        allowed_operations=tuple(catalog),
        allowed_command_operations=(
            "federation.create",
            "budget.reserve",
            "budget.release",
            "supervisor.register",
            "subagent.register",
            "subscription.register",
        ),
        peer_pid=os.getpid(),
    )
    monkeypatch.setenv(
        TYPED_STATE_OWNER_SOCKET_ENV,
        str(server.typed_command_socket_path()),
    )
    monkeypatch.setenv(TYPED_STATE_OWNER_TOKEN_ENV, token)
    client = QuackStateClient(
        owner_id=client_id,
        store_id=identity.store_id,
        process_birth_id=identity.process_birth_id,
        expected_identity=identity.store_identity(),
    )
    try:
        session = client.attach(identity.listen_uri, server_id=identity.server_id)
        assert session.transport_mode is TransportMode.QUACK
        assert session.session_id.startswith("session:owner:")
        repository = server.bind_federation_repository(
            client,
            require_quack_authority=True,
        )
        yield server, client, repository
    finally:
        client.close()
        server.stop()


def _admit(
    repository: Any,
    client: QuackStateClient,
    *,
    profile: Mapping[str, Any] | None = None,
) -> BootstrapAdmission:
    generation = client.load_generation()
    return admit_bootstrap_federation(
        repository,
        profile=_profile() if profile is None else profile,
        repository_id="repository:ipfs_accelerate_py",
        repository_tree_id="tree:casf-bootstrap-test-v1",
        plan_root_ref="plan-root:casf-bootstrap-test-v1",
        operation_catalog_ref=catalog_fingerprint(
            build_control_plane_operation_catalog()
        ),
        control_plane_generation=generation.generation,
        fencing_epoch=generation.fence_epoch,
        ready_task_refs=("CASF-002",),
        authentication_key=b"casf-bootstrap-test-key-material",
        now=_NOW,
    )


def _cid_digest(cid: str) -> str:
    encoded = cid.removeprefix("b").upper()
    encoded += "=" * ((8 - len(encoded) % 8) % 8)
    raw = base64.b32decode(encoded)
    assert raw.startswith(b"\x01\xa9\x02\x12\x20")
    assert len(raw) == 37
    return "sha256:" + raw[-32:].hex()


def test_real_migrated_schema_cid_has_one_owner_and_client_store_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _typed_repository(tmp_path, monkeypatch) as (server, client, repository):
        identity = server.identity
        session = client.session
        assert identity is not None
        assert session is not None

        metadata = {
            str(item["key"]): str(item["value"])
            for item in client.execute("whoami_metadata")
        }
        raw_schema_cid = metadata["schema_fingerprint"]
        assert raw_schema_cid.startswith("b")
        assert _cid_digest(raw_schema_cid) == identity.schema_fingerprint
        assert session.store_identity.schema_fingerprint == identity.schema_fingerprint
        assert repository.store_generation() == identity.generation
        assert server.status()["typed_command_gateway"]["raw_sql_permitted"] is False


def test_bootstrap_admission_replays_without_duplicate_authoritative_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _typed_repository(tmp_path, monkeypatch) as (_server, client, repository):
        first = _admit(repository, client)
        generation_after_first = client.load_generation()

        replay = _admit(repository, client)
        generation_after_replay = client.load_generation()

        assert replay == first
        assert generation_after_replay == generation_after_first
        authentication_refs = tuple(
            item
            for item in first.federation_receipt.evidence_refs
            if item.startswith("authentication:casf-local-bootstrap:")
        )
        assert authentication_refs == (
            first.federation_identity.binding.authorization_evidence_ref,
        )
        scope = {
            "tenant_id": first.supervisor.binding.tenant_id,
            "federation_id": first.federation_identity.record_id,
        }
        assert client.execute("casf_count_supervisors", scope)[0]["population"] == 1
        assert client.execute("casf_count_subagents", scope)[0]["population"] == 1
        federation_rows = client.execute(
            "casf_select_federation",
            {
                "federation_id": first.federation_identity.record_id,
                "tenant_id": first.supervisor.binding.tenant_id,
            },
        )
        assert len(federation_rows) == 1
        decision_rows = client.execute(
            "casf_select_authorization_decision",
            {
                "federation_id": first.federation_identity.record_id,
                "tenant_id": first.supervisor.binding.tenant_id,
            },
        )
        assert len(decision_rows) == 1
        decision = FederationAuthorizationDecision.from_dict(
            json.loads(str(decision_rows[0]["body_json"]))
        )
        assert decision.cid == decision_rows[0]["authorization_decision_id"]
        assert decision.cid == decision_rows[0]["content_ref"]
        assert decision.request_cid == decision_rows[0]["request_cid"]
        assert decision.authentication_evidence_cid == decision_rows[0]["evidence_ref"]
        assert decision.authentication_evidence_cid in first.federation_receipt.evidence_refs
        assert decision.cid in first.federation_receipt.evidence_refs
        assert (
            decision.authentication_evidence_cid
            != first.federation_identity.binding.authorization_evidence_ref
        )
        assert "signature" not in str(decision_rows[0]["body_json"])
        assert "key_handle" not in str(decision_rows[0]["body_json"])
        assert repository.load_subscription(
            tenant_id=first.subscription.tenant_id,
            federation_id=first.subscription.federation_id,
            subscription_id=first.subscription.subscription_id,
        ) == first.subscription
        assert first.public_dict()["registered_logical_subagents"] == 1
        assert first.public_dict()["active_subagent_processes"] == 0


@pytest.mark.parametrize(
    "bad_profile, message",
    [
        ({**_profile(), "model_policy_outcome": "allow"}, "unknown fields"),
        (_profile(expires_at="2029-12-31T23:59:59Z"), "expired"),
    ],
)
def test_unknown_or_expired_bootstrap_profile_fails_closed_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_profile: Mapping[str, Any],
    message: str,
) -> None:
    with _typed_repository(tmp_path, monkeypatch) as (_server, client, repository):
        before = client.load_generation()

        with pytest.raises(FederationContractError, match=message):
            _admit(repository, client, profile=bad_profile)

        assert client.load_generation() == before
        assert validate_bootstrap_profile(_profile())["maximum_supervisors"] == 1
