"""Tests for control-plane store, schema, identity, and authority contracts."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CONTROL_PLANE_STORE_IDENTITY_INTERFACE,
    STATE_COMMAND_INTERFACE,
    STATE_EXPORT_RECEIPT_INTERFACE,
    STATE_SNAPSHOT_INTERFACE,
    STORE_GENERATION_INTERFACE,
    CommandKind,
    ControlPlaneAliasError,
    ControlPlaneAuthorityError,
    ControlPlaneBounds,
    ControlPlaneBoundsError,
    ControlPlaneGenerationError,
    ControlPlaneIdentityError,
    ControlPlaneSecretError,
    ControlPlaneStoreIdentity,
    ExportProfile,
    FenceToken,
    IdempotencyBinding,
    RevisionToken,
    SchemaIdentity,
    SessionIdentity,
    StateAuthorityClass,
    StateCommand,
    StateExportReceipt,
    StateSnapshot,
    StoreGeneration,
    assert_generation_revision_match,
    closed_authority_classes,
    closed_command_kinds,
    closed_export_profiles,
    content_identity,
    redact_mapping,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_identity import (
    canonical_content_cid,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

_DIGEST_A = "sha256:" + ("a" * 64)
_DIGEST_B = "sha256:" + ("b" * 64)
_DIGEST_C = "sha256:" + ("c" * 64)
_DIGEST_D = "sha256:" + ("d" * 64)
_DIGEST_E = "sha256:" + ("e" * 64)
_UUID = "550e8400-e29b-41d4-a716-446655440000"
_TS = "2026-08-08T12:00:00Z"


def _schema(**changes: object) -> SchemaIdentity:
    values: dict[str, object] = {
        "schema_revision": 3,
        "schema_fingerprint": _DIGEST_A,
        "catalog_fingerprint": _DIGEST_B,
        "migration_head": 3,
        "application_version": "1.0.0",
        "tool_version": "1.5.2",
    }
    values.update(changes)
    return SchemaIdentity(**values)  # type: ignore[arg-type]


def _generation(**changes: object) -> StoreGeneration:
    values: dict[str, object] = {
        "generation": 7,
        "credential_generation": 7,
        "startup_epoch": 7,
        "fencing_epoch": 4,
    }
    values.update(changes)
    return StoreGeneration(**values)  # type: ignore[arg-type]


def _store(**changes: object) -> ControlPlaneStoreIdentity:
    values: dict[str, object] = {
        "repository_id": "repository:sha256:test-control-plane",
        "database_uuid": _UUID,
        "schema": _schema(),
        "generation": _generation(),
        "store_path_digest": _DIGEST_C,
        "extension_fingerprint": _DIGEST_D,
        "listen_uri_digest": _DIGEST_E,
        "process_birth_id": "birth:session-1",
        "display_alias": "control-primary",
    }
    values.update(changes)
    return ControlPlaneStoreIdentity(**values)  # type: ignore[arg-type]


def _fence(**changes: object) -> FenceToken:
    values: dict[str, object] = {
        "fencing_epoch": 4,
        "session_id": "session:owner-1",
        "scope": ("tasks", "leases"),
        "generation": 7,
    }
    values.update(changes)
    return FenceToken(**values)  # type: ignore[arg-type]


def _idempotency(**changes: object) -> IdempotencyBinding:
    values: dict[str, object] = {
        "key": "claim/task-1/7",
        "operation": "task.claim",
        "caller": "principal:daemon-1",
        "generation": 7,
    }
    values.update(changes)
    return IdempotencyBinding(**values)  # type: ignore[arg-type]


def _command(**changes: object) -> StateCommand:
    store = _store()
    values: dict[str, object] = {
        "command_id": "command:claim-1",
        "kind": CommandKind.MUTATION,
        "store_id": store.store_id,
        "repository_id": store.repository_id,
        "database_uuid": store.database_uuid,
        "generation": store.store_generation,
        "expected_revision": 12,
        "issued_at": _TS,
        "fence": _fence(),
        "idempotency": _idempotency(),
        "parameters": {"task_id": "task:DQP-002"},
    }
    values.update(changes)
    return StateCommand(**values)  # type: ignore[arg-type]


def _snapshot(**changes: object) -> StateSnapshot:
    store = _store()
    values: dict[str, object] = {
        "snapshot_id": "snapshot:1",
        "store": store,
        "revision": 12,
        "transaction_watermark": "tx:0000000c",
        "captured_at": _TS,
    }
    values.update(changes)
    return StateSnapshot(**values)  # type: ignore[arg-type]


def _export(**changes: object) -> StateExportReceipt:
    values: dict[str, object] = {
        "export_id": "export:taskboard-1",
        "snapshot": _snapshot(),
        "profile": ExportProfile.MARKDOWN_TASKBOARD,
        "renderer_revision": "renderer:markdown@1",
        "query_revision": "view:tasks@1",
        "parameters_digest": _DIGEST_A,
        "artifact_digest": _DIGEST_B,
        "destination_digest": _DIGEST_C,
        "exported_at": _TS,
        "omitted_fields": ("private_notes",),
    }
    values.update(changes)
    return StateExportReceipt(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Happy path / interfaces
# ---------------------------------------------------------------------------


def test_interfaces_and_closed_vocabularies() -> None:
    assert CONTROL_PLANE_STORE_IDENTITY_INTERFACE == "ControlPlaneStoreIdentity@1"
    assert STORE_GENERATION_INTERFACE == "StoreGeneration@1"
    assert STATE_COMMAND_INTERFACE == "StateCommand@1"
    assert STATE_SNAPSHOT_INTERFACE == "StateSnapshot@1"
    assert STATE_EXPORT_RECEIPT_INTERFACE == "StateExportReceipt@1"
    assert "authority" in closed_authority_classes()
    assert "export" in closed_authority_classes()
    assert "mutation" in closed_command_kinds()
    assert "markdown_taskboard" in closed_export_profiles()


def test_store_identity_is_canonical_stable_and_immutable() -> None:
    store = _store()
    restored = ControlPlaneStoreIdentity.from_dict(store.to_dict())
    assert restored == store
    assert restored.content_id == store.content_id
    assert restored.store_id == store.content_id
    assert restored.content_id == content_identity(store._identity_payload())
    # display_alias is provenance only and does not affect identity.
    other_alias = _store(display_alias="other-label")
    assert other_alias.content_id == store.content_id
    with pytest.raises(FrozenInstanceError):
        store.database_uuid = "other"  # type: ignore[misc]


def test_generation_schema_session_revision_fence_round_trip() -> None:
    generation = _generation()
    schema = _schema()
    session = SessionIdentity(
        session_id="session:owner-1",
        owner_principal="principal:daemon-1",
        generation=7,
        fencing_epoch=4,
        opened_at=_TS,
    )
    revision = RevisionToken(revision=12, generation=7)
    fence = _fence()
    assert StoreGeneration.from_dict(generation.to_dict()) == generation
    assert SchemaIdentity.from_dict(schema.to_dict()) == schema
    assert SessionIdentity.from_dict(session.to_dict()) == session
    assert RevisionToken.from_dict(revision.to_dict()) == revision
    assert FenceToken.from_dict(fence.to_dict()) == fence
    assert_generation_revision_match(generation, revision)


def test_state_command_and_snapshot_and_export_round_trip() -> None:
    command = _command()
    snapshot = _snapshot()
    export = _export()
    assert StateCommand.from_dict(command.to_dict()) == command
    assert StateSnapshot.from_dict(snapshot.to_dict()) == snapshot
    assert StateExportReceipt.from_dict(export.to_dict()) == export
    command.assert_matches_store(_store())
    command.assert_revision_consistent(snapshot.revision_token())
    assert export.authority_class is StateAuthorityClass.EXPORT
    assert export.is_authoritative is False


# ---------------------------------------------------------------------------
# Rejection cases required by DQP-002 acceptance
# ---------------------------------------------------------------------------


def test_rejects_empty_ids() -> None:
    with pytest.raises(ControlPlaneIdentityError, match="must not be empty"):
        _store(repository_id="")
    with pytest.raises(ControlPlaneIdentityError, match="must not be empty"):
        _store(database_uuid="")
    with pytest.raises(ControlPlaneIdentityError, match="must not be empty"):
        _command(command_id="")
    with pytest.raises(ControlPlaneIdentityError, match="must not be empty"):
        SchemaIdentity(
            schema_revision=1,
            schema_fingerprint="",
            catalog_fingerprint=_DIGEST_B,
        )


def test_rejects_forged_and_inconsistent_ids() -> None:
    with pytest.raises(ControlPlaneIdentityError, match="forged or inconsistent"):
        _store(content_id="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    with pytest.raises(ControlPlaneIdentityError, match="forged or inconsistent"):
        _generation(content_id=canonical_content_cid({"wrong": True}))
    with pytest.raises(ControlPlaneIdentityError, match="forged or inconsistent"):
        _command(content_id=canonical_content_cid({"forged": 1}))


def test_rejects_non_finite_bounds() -> None:
    with pytest.raises(ControlPlaneBoundsError, match="finite"):
        ControlPlaneBounds(max_record_bytes=math.inf)  # type: ignore[arg-type]
    with pytest.raises(ControlPlaneBoundsError, match="finite"):
        ControlPlaneBounds(max_depth=math.nan)  # type: ignore[arg-type]
    with pytest.raises(ControlPlaneBoundsError, match="floating-point"):
        ControlPlaneBounds(max_parameters=1.5)  # type: ignore[arg-type]
    with pytest.raises(ControlPlaneBoundsError, match="floating-point"):
        StoreGeneration(
            generation=1.0,  # type: ignore[arg-type]
            credential_generation=1,
            startup_epoch=1,
        )
    with pytest.raises(ControlPlaneBoundsError, match="outside the supported bound"):
        ControlPlaneBounds(max_depth=0)
    with pytest.raises(ControlPlaneBoundsError, match="floating-point"):
        _command(parameters={"limit": 1.25})


def test_rejects_generation_revision_mismatch() -> None:
    with pytest.raises(ControlPlaneGenerationError, match="credential_generation"):
        _generation(generation=3, credential_generation=4, startup_epoch=3)
    with pytest.raises(ControlPlaneGenerationError, match="startup_epoch"):
        _generation(generation=3, credential_generation=3, startup_epoch=4)
    with pytest.raises(ControlPlaneGenerationError, match="migration_head"):
        _schema(schema_revision=2, migration_head=3)
    revision = RevisionToken(revision=1, generation=2)
    with pytest.raises(ControlPlaneGenerationError, match="does not match"):
        revision.assert_matches_generation(_generation())
    command = _command(generation=7, expected_revision=5)
    with pytest.raises(ControlPlaneGenerationError, match="does not match"):
        command.assert_revision_consistent(
            RevisionToken(revision=5, generation=8)
        )
    with pytest.raises(ControlPlaneGenerationError, match="does not match"):
        command.assert_revision_consistent(
            RevisionToken(revision=9, generation=7)
        )
    with pytest.raises(ControlPlaneGenerationError, match="does not match store"):
        mismatched = _command(
            generation=99,
            fence=_fence(generation=99),
            idempotency=_idempotency(generation=99),
        )
        mismatched.assert_matches_store(_store())
    with pytest.raises(ControlPlaneGenerationError, match="idempotency generation"):
        _command(idempotency=_idempotency(generation=1))


def test_rejects_secrets() -> None:
    with pytest.raises(ControlPlaneSecretError, match="secret"):
        _command(parameters={"api_key": "sk-not-a-real-key-value-12345"})
    with pytest.raises(ControlPlaneSecretError, match="secret"):
        _command(parameters={"password": "hunter2"})
    with pytest.raises(ControlPlaneSecretError, match="secret"):
        _store(process_birth_id="ghp_abcdefghijklmnopqrstuvwxyz012345")
    redacted = redact_mapping(
        {"task_id": "task:1", "api_key": "secret", "nested": {"token": "x"}}
    )
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["nested"]["token"] == "[REDACTED]"
    assert redacted["task_id"] == "task:1"


def test_rejects_mutable_aliases_as_identity() -> None:
    with pytest.raises(ControlPlaneAliasError, match="mutable alias"):
        _store(repository_id="/var/lib/control.duckdb")
    with pytest.raises(ControlPlaneAliasError, match="mutable alias"):
        _store(database_uuid="localhost")
    with pytest.raises(ControlPlaneAliasError, match="mutable alias"):
        _store(process_birth_id="pid:12345")
    with pytest.raises(ControlPlaneAliasError, match="mutable alias"):
        _store(process_birth_id="12345")
    with pytest.raises(ControlPlaneAliasError, match="mutable alias"):
        SessionIdentity(
            session_id="Primary Session",
            owner_principal="principal:daemon-1",
            generation=7,
            fencing_epoch=4,
            opened_at=_TS,
        )
    # Display alias may exist but is excluded from content identity.
    left = _store(display_alias="alpha")
    right = _store(display_alias="beta")
    assert left.content_id == right.content_id


def test_rejects_export_labeled_authoritative() -> None:
    with pytest.raises(ControlPlaneAuthorityError, match="authoritative"):
        _export(is_authoritative=True)
    with pytest.raises(ControlPlaneAuthorityError, match="authority"):
        _export(authority_class=StateAuthorityClass.AUTHORITY)
    with pytest.raises(ControlPlaneAuthorityError, match="must be export"):
        _export(authority_class=StateAuthorityClass.CACHE)
    # Store identity itself must remain authority-class.
    with pytest.raises(ControlPlaneAuthorityError, match="must be authority"):
        _store(authority_class=StateAuthorityClass.EXPORT)


def test_mutation_requires_fence_and_idempotency() -> None:
    with pytest.raises(Exception, match="fence"):
        _command(fence=None)
    with pytest.raises(Exception, match="idempotency"):
        _command(idempotency=None)
    # Reads may omit both.
    read = _command(
        kind=CommandKind.READ,
        command_id="command:read-1",
        fence=None,
        idempotency=None,
        expected_revision=0,
        parameters={"limit": 10},
    )
    assert read.kind is CommandKind.READ
    assert read.fence is None


def test_bounds_record_is_finite_and_round_trips() -> None:
    bounds = ControlPlaneBounds()
    restored = ControlPlaneBounds.from_dict(bounds.to_dict())
    assert restored == bounds
    assert bounds.max_record_bytes > 0


# ---------------------------------------------------------------------------
# Cold import: no filesystem / database / network / provider / process action
# ---------------------------------------------------------------------------


def test_cold_import_performs_no_external_actions() -> None:
    """Fresh-process import must stay free of external side effects.

    Python still needs to read ``.py`` sources to load the module; the gate
    forbids *module-level* filesystem/database/network/provider/process work
    and any import of optional heavy providers.
    """

    script = r"""
import ast
import json
import socket
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

before_modules = set(sys.modules)

def forbidden(*args, **kwargs):
    raise AssertionError(
        "control_plane_contracts cold import performed an external action"
    )

# Process / network / database surfaces remain blocked for the whole import.
subprocess.Popen = forbidden
subprocess.run = forbidden
subprocess.call = forbidden
subprocess.check_call = forbidden
subprocess.check_output = forbidden
threading.Thread.start = forbidden
socket.socket = forbidden
socket.create_connection = forbidden
sqlite3.connect = forbidden
if "duckdb" in sys.modules:
    sys.modules["duckdb"].connect = forbidden

from ipfs_accelerate_py.agent_supervisor.task_sources import (
    control_plane_contracts as cpc,
)

# Source-level proof: the contracts module body never opens IO/providers.
module_path = Path(cpc.__file__).resolve()
tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
forbidden_calls = {
    "open",
    "connect",
    "urlopen",
    "Popen",
    "system",
    "check_output",
}
found = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        func = node.func
        name = ""
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name in forbidden_calls:
            found.append(name)
assert found == [], found

# In-memory construction only (no store files, no DB handles).
digest_a = "sha256:" + ("a" * 64)
digest_b = "sha256:" + ("b" * 64)
schema = cpc.SchemaIdentity(
    schema_revision=1,
    schema_fingerprint=digest_a,
    catalog_fingerprint=digest_b,
)
generation = cpc.StoreGeneration(
    generation=1,
    credential_generation=1,
    startup_epoch=1,
    fencing_epoch=1,
)
store = cpc.ControlPlaneStoreIdentity(
    repository_id="repository:sha256:cold",
    database_uuid="550e8400-e29b-41d4-a716-446655440000",
    schema=schema,
    generation=generation,
)
export = cpc.StateExportReceipt(
    export_id="export:cold",
    snapshot=cpc.StateSnapshot(
        snapshot_id="snapshot:cold",
        store=store,
        revision=0,
        transaction_watermark="tx:0",
        captured_at="2026-08-08T00:00:00Z",
    ),
    profile=cpc.ExportProfile.JSON_STATUS,
    renderer_revision="renderer:json@1",
    query_revision="view:status@1",
    parameters_digest=digest_a,
    artifact_digest=digest_b,
    destination_digest=digest_a,
    exported_at="2026-08-08T00:00:00Z",
)

after_modules = set(sys.modules)
added = sorted(after_modules - before_modules)
forbidden_prefixes = (
    "duckdb",
    "openai",
    "anthropic",
    "torch",
    "transformers",
    "httpx",
    "requests",
    "aiohttp",
    "urllib3",
    "neo4j",
    "sentence_transformers",
)
added_forbidden = [
    name
    for name in added
    if any(
        name == prefix or name.startswith(prefix + ".")
        for prefix in forbidden_prefixes
    )
]

print(
    json.dumps(
        {
            "store_id": store.store_id,
            "export_authoritative": export.is_authoritative,
            "export_authority": export.authority_class.value,
            "added_forbidden": added_forbidden,
            "interface": cpc.CONTROL_PLANE_STORE_IDENTITY_INTERFACE,
            "module_path_suffix": "/".join(module_path.parts[-4:]),
        }
    )
)
"""
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["IPFS_ACCEL_SKIP_CORE"] = "1"
    env["IPFS_ACCEL_IMPORT_EAGER"] = "0"
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("PYTEST_VERSION", None)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=30,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["export_authoritative"] is False
    assert payload["export_authority"] == "export"
    assert payload["added_forbidden"] == []
    assert payload["interface"] == "ControlPlaneStoreIdentity@1"
    assert payload["store_id"].startswith("b")
    assert payload["module_path_suffix"].endswith(
        "task_sources/control_plane_contracts.py"
    )
