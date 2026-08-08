"""Contract tests for control-plane store, schema, identity, and authority."""

from __future__ import annotations

import importlib
import json
import math
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CONTROL_PLANE_STORE_IDENTITY_INTERFACE,
    ControlPlaneBounds,
    ControlPlaneContractError,
    ControlPlaneFailure,
    ControlPlaneFailureCode,
    ControlPlaneStoreIdentity,
    EmptyIdentityError,
    ExportAuthorityError,
    ExportProfile,
    FenceIdentity,
    ForgedIdentityError,
    GenerationMismatchError,
    InconsistentIdentityError,
    MutableAliasIdentityError,
    NonFiniteBoundsError,
    RevisionMismatchError,
    SchemaIdentity,
    SecretHandle,
    SecretMaterialError,
    SessionIdentity,
    STATE_COMMAND_INTERFACE,
    STATE_EXPORT_RECEIPT_INTERFACE,
    STATE_SNAPSHOT_INTERFACE,
    STORE_GENERATION_INTERFACE,
    StateAuthorityClass,
    StateCommand,
    StateCommandKind,
    StateExportReceipt,
    StateRevision,
    StateSnapshot,
    StoreGeneration,
    content_identity,
    redact_public_text,
    validate_generation_revision_alignment,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DIGEST_A = "sha256:" + ("ab" * 32)
DIGEST_B = "sha256:" + ("cd" * 32)
DIGEST_C = "sha256:" + ("ef" * 32)
STORE_UUID = "550e8400-e29b-41d4-a716-446655440000"


def _schema_identity(**changes: Any) -> SchemaIdentity:
    values: dict[str, Any] = {
        "schema_revision": 3,
        "schema_fingerprint": DIGEST_A,
        "catalog_fingerprint": DIGEST_B,
        "migration_head": 3,
    }
    values.update(changes)
    return SchemaIdentity(**values)


def _store_generation(**changes: Any) -> StoreGeneration:
    values: dict[str, Any] = {
        "generation": 2,
        "store_uuid": STORE_UUID,
        "epoch_id": "epoch:bootstrap-1",
        "created_at_ms": 1_700_000_000_000,
        "reason": "bootstrap",
    }
    values.update(changes)
    return StoreGeneration(**values)


def _store(**changes: Any) -> ControlPlaneStoreIdentity:
    values: dict[str, Any] = {
        "store_uuid": STORE_UUID,
        "repository_id": "repository:sha256:demo",
        "generation": _store_generation(),
        "schema_identity": _schema_identity(),
        "authority": StateAuthorityClass.AUTHORITATIVE,
        "annotations": {"display_name": "primary-control"},
    }
    values.update(changes)
    return ControlPlaneStoreIdentity(**values)


def _session(**changes: Any) -> SessionIdentity:
    values: dict[str, Any] = {
        "session_id": "session:owner-1",
        "store_uuid": STORE_UUID,
        "generation": 2,
        "process_birth_id": "proc-birth:1-1700000000-abcdef",
        "role": "state-owner",
        "annotations": {"pid": 4242},
    }
    values.update(changes)
    return SessionIdentity(**values)


def _revision(**changes: Any) -> StateRevision:
    values: dict[str, Any] = {
        "revision": 7,
        "store_uuid": STORE_UUID,
        "generation": 2,
        "stream_id": "stream:tasks",
        "revision_digest": DIGEST_C,
    }
    values.update(changes)
    return StateRevision(**values)


def _fence(**changes: Any) -> FenceIdentity:
    values: dict[str, Any] = {
        "fence_epoch": 4,
        "fence_token": DIGEST_A,
        "session_id": "session:owner-1",
        "store_uuid": STORE_UUID,
        "generation": 2,
        "scope_id": "scope:task:DQP-002",
        "expected_revision": 7,
    }
    values.update(changes)
    return FenceIdentity(**values)


def _bounds(**changes: Any) -> ControlPlaneBounds:
    values: dict[str, Any] = {
        "max_items": 64,
        "max_serialized_bytes": 65_536,
        "max_text_bytes": 4_096,
        "max_paths": 32,
        "max_depth": 6,
        "timeout_ms": 5_000,
        "max_retries": 3,
    }
    values.update(changes)
    return ControlPlaneBounds(**values)


def _command(**changes: Any) -> StateCommand:
    values: dict[str, Any] = {
        "command_id": "cmd:claim-1",
        "kind": StateCommandKind.CLAIM,
        "store": _store(),
        "session": _session(),
        "fence": _fence(),
        "expected_revision": _revision(),
        "idempotency_key": "idem:claim-1",
        "bounds": _bounds(),
        "parameters": {"task_id": "DQP-002"},
        "issued_at_ms": 1_700_000_000_100,
    }
    values.update(changes)
    return StateCommand(**values)


def _snapshot(**changes: Any) -> StateSnapshot:
    values: dict[str, Any] = {
        "snapshot_id": "snapshot:1",
        "store": _store(),
        "transaction_watermark": 99,
        "snapshot_digest": DIGEST_B,
        "captured_at_ms": 1_700_000_000_200,
        "authority": StateAuthorityClass.AUTHORITATIVE,
    }
    values.update(changes)
    return StateSnapshot(**values)


def _export(**changes: Any) -> StateExportReceipt:
    values: dict[str, Any] = {
        "export_id": "export:taskboard-1",
        "snapshot": _snapshot(),
        "profile": ExportProfile.HUMAN_TASKBOARD,
        "view_revision": "view:taskboard@1",
        "renderer_revision": "renderer:markdown@1",
        "artifact_digest": DIGEST_C,
        "destination": "exports/taskboard.md",
        "parameters": {"lossy": True},
        "authority": StateAuthorityClass.EXPORT,
        "exported_at_ms": 1_700_000_000_300,
    }
    values.update(changes)
    return StateExportReceipt(**values)


def test_interface_version_constants_are_stable() -> None:
    assert CONTROL_PLANE_STORE_IDENTITY_INTERFACE == "ControlPlaneStoreIdentity@1"
    assert STORE_GENERATION_INTERFACE == "StoreGeneration@1"
    assert STATE_COMMAND_INTERFACE == "StateCommand@1"
    assert STATE_SNAPSHOT_INTERFACE == "StateSnapshot@1"
    assert STATE_EXPORT_RECEIPT_INTERFACE == "StateExportReceipt@1"


def test_happy_path_records_are_content_addressed_and_round_trip() -> None:
    store = _store()
    command = _command()
    snapshot = _snapshot()
    export = _export()

    assert store.content_id.startswith("b")
    assert command.content_id.startswith("b")
    assert snapshot.store.store_uuid == STORE_UUID
    assert export.authority is StateAuthorityClass.EXPORT
    assert not export.authority.grants_write_authority

    assert ControlPlaneStoreIdentity.from_dict(store.to_record()).content_id == (
        store.content_id
    )
    assert StateCommand.from_dict(command.to_record()).content_id == command.content_id
    assert StateSnapshot.from_dict(snapshot.to_record()).content_id == (
        snapshot.content_id
    )
    assert StateExportReceipt.from_dict(export.to_record()).content_id == (
        export.content_id
    )
    assert SecretHandle.from_dict(
        SecretHandle(
            handle_id="secret-handle:quack-auth-1", purpose="quack_auth", generation=1
        ).to_record()
    ).handle_id == "secret-handle:quack-auth-1"


def test_empty_identities_are_rejected() -> None:
    with pytest.raises(EmptyIdentityError):
        _store(repository_id="")
    with pytest.raises(EmptyIdentityError):
        _session(session_id="   ")
    with pytest.raises(EmptyIdentityError):
        _command(command_id="")
    with pytest.raises(EmptyIdentityError):
        SchemaIdentity(
            schema_revision=1,
            schema_fingerprint="",
            catalog_fingerprint=DIGEST_B,
        )


def test_forged_content_ids_are_rejected() -> None:
    store = _store()
    forged = store.to_record()
    forged["content_id"] = "b" + ("a" * 58)
    with pytest.raises(ForgedIdentityError):
        ControlPlaneStoreIdentity.from_dict(forged)

    command = _command()
    forged_cmd = command.to_record()
    forged_cmd["content_id"] = content_identity({"not": "this"})
    with pytest.raises(ForgedIdentityError):
        StateCommand.from_dict(forged_cmd)


def test_inconsistent_store_and_generation_uuid_rejected() -> None:
    with pytest.raises(InconsistentIdentityError, match="store_uuid"):
        _store(
            generation=_store_generation(
                store_uuid="11111111-1111-4111-8111-111111111111"
            )
        )


def test_generation_and_revision_mismatch_rejected() -> None:
    with pytest.raises(GenerationMismatchError, match="session.generation"):
        _command(session=_session(generation=99))

    with pytest.raises(GenerationMismatchError, match="fence.generation"):
        _command(fence=_fence(generation=99))

    with pytest.raises(GenerationMismatchError, match="expected_revision.generation"):
        _command(expected_revision=_revision(generation=99))

    with pytest.raises(RevisionMismatchError, match="fence.expected_revision"):
        _command(fence=_fence(expected_revision=1))

    with pytest.raises(GenerationMismatchError):
        validate_generation_revision_alignment(
            store_generation=2,
            command_generation=3,
            expected_revision=7,
            observed_revision=7,
        )
    with pytest.raises(RevisionMismatchError):
        validate_generation_revision_alignment(
            store_generation=2,
            command_generation=2,
            expected_revision=7,
            observed_revision=8,
        )
    validate_generation_revision_alignment(
        store_generation=2,
        command_generation=2,
        expected_revision=7,
        observed_revision=7,
    )


def test_non_finite_and_non_integer_bounds_rejected() -> None:
    with pytest.raises(NonFiniteBoundsError):
        ControlPlaneBounds(max_items=float("nan"))  # type: ignore[arg-type]
    with pytest.raises(NonFiniteBoundsError):
        ControlPlaneBounds(timeout_ms=float("inf"))  # type: ignore[arg-type]
    with pytest.raises(NonFiniteBoundsError):
        ControlPlaneBounds(max_retries=-1)
    with pytest.raises(NonFiniteBoundsError):
        ControlPlaneBounds(max_items=0)
    with pytest.raises(NonFiniteBoundsError):
        ControlPlaneBounds(max_items=True)  # type: ignore[arg-type]
    with pytest.raises(NonFiniteBoundsError):
        ControlPlaneBounds(max_paths=100, max_items=10)
    with pytest.raises(NonFiniteBoundsError):
        ControlPlaneBounds(max_items=10_001)

    # Nested non-finite values in parameters fail closed.
    with pytest.raises(NonFiniteBoundsError):
        _command(parameters={"score": math.nan})
    with pytest.raises(NonFiniteBoundsError):
        _command(parameters={"limit": 1.5})


def test_secrets_are_rejected_from_public_contracts() -> None:
    with pytest.raises(SecretMaterialError):
        _store(annotations={"api_key": "sk_live_not_real_but_long_enough_to_trip"})
    with pytest.raises(SecretMaterialError):
        ControlPlaneStoreIdentity.from_dict(
            {
                **_store().to_dict(),
                "token": "super-secret",
            }
        )
    with pytest.raises(SecretMaterialError):
        SecretHandle.from_dict(
            {
                "schema": SecretHandle(
                    handle_id="secret-handle:quack-auth-1"
                ).SCHEMA,
                "handle_id": "secret-handle:quack-auth-1",
                "purpose": "quack_auth",
                "generation": 1,
                "value": "raw-secret",
            }
        )
    with pytest.raises(SecretMaterialError):
        _session(process_birth_id="eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.signaturexx")
    with pytest.raises(SecretMaterialError):
        _command(parameters={"password": "hunter2-not-allowed"})

    redacted = redact_public_text("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.aa.bb")
    assert "eyJ" not in redacted
    assert "[redacted]" in redacted or "[redacted-field]" in redacted


def test_mutable_aliases_cannot_serve_as_identity() -> None:
    with pytest.raises(MutableAliasIdentityError):
        _store(
            repository_id="primary-control",
            annotations={"display_name": "primary-control"},
        )
    with pytest.raises(MutableAliasIdentityError):
        _session(process_birth_id="4242")
    with pytest.raises(MutableAliasIdentityError):
        _session(
            process_birth_id="same-as-pid",
            annotations={"pid": "same-as-pid"},
        )
    with pytest.raises(MutableAliasIdentityError):
        SessionIdentity(
            session_id="display_name",
            store_uuid=STORE_UUID,
            generation=1,
            process_birth_id="proc-birth:ok",
        )
    # Display annotations are allowed when they are not the identity key.
    session = _session(annotations={"display_name": "lane-0", "pid": 9})
    assert session.annotations["display_name"] == "lane-0"
    assert session.process_birth_id != "9"


def test_export_labeled_authoritative_is_rejected() -> None:
    with pytest.raises(ExportAuthorityError, match="authoritative"):
        _export(authority=StateAuthorityClass.AUTHORITATIVE)
    with pytest.raises(InconsistentIdentityError):
        _export(authority=StateAuthorityClass.CACHE)

    # Round-trip of a valid export stays non-authoritative.
    export = _export()
    assert export.authority is StateAuthorityClass.EXPORT
    assert export.authority.is_projection
    assert StateAuthorityClass.AUTHORITATIVE.grants_write_authority


def test_authority_classes_are_closed() -> None:
    assert {item.value for item in StateAuthorityClass} == {
        "authoritative",
        "static_input",
        "immutable_evidence",
        "cache",
        "export",
        "os_bootstrap",
        "emergency_diagnostic",
    }
    with pytest.raises(ControlPlaneContractError):
        _store(authority="trusted")  # type: ignore[arg-type]


def test_malformed_digests_and_uuids_fail_closed() -> None:
    with pytest.raises(ControlPlaneContractError):
        _store_generation(store_uuid="not-a-uuid")
    with pytest.raises(ControlPlaneContractError):
        _revision(revision_digest="sha256:deadbeef")
    with pytest.raises(ControlPlaneContractError):
        _fence(fence_token="token-without-digest")


def test_typed_failure_projection_is_secret_free() -> None:
    try:
        _command(session=_session(generation=9))
    except GenerationMismatchError as exc:
        failure = ControlPlaneFailure.from_exception(exc)
    else:  # pragma: no cover
        raise AssertionError("expected GenerationMismatchError")
    assert failure.code is ControlPlaneFailureCode.GENERATION_MISMATCH
    assert "generation" in failure.message
    record = failure.to_record()
    assert record["content_id"] == failure.content_id
    assert ControlPlaneFailure.from_dict(record).code is failure.code


def test_canonical_serialization_is_byte_stable() -> None:
    left = _command().to_json_bytes()
    right = _command().to_json_bytes()
    assert left == right
    payload = json.loads(left.decode("utf-8"))
    assert payload["schema"].endswith("@1")
    assert "content_id" not in payload  # identity is non-recursive


def test_cold_import_performs_no_io_or_provider_action() -> None:
    """Importing contracts must not touch FS/DB/network/providers/processes."""

    module_name = (
        "ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts"
    )
    # Drop a previously imported module so the probe is meaningful in-process.
    sys.modules.pop(module_name, None)

    opened: list[str] = []
    original_open = open

    def guarded_open(file, *args, **kwargs):  # type: ignore[no-untyped-def]
        path = str(file)
        # Allow Python import machinery (source/bytecode). Reject durable state
        # and credential artifacts that contracts must never open at import.
        lowered = path.lower().replace("\\", "/")
        basename = Path(lowered).name
        if basename.endswith(
            (".duckdb", ".db", ".sqlite", ".sqlite3", ".wal", ".pid", ".lock")
        ) or basename in {"credentials", "credentials.json", "quack.token"}:
            opened.append(path)
            raise AssertionError(f"unexpected filesystem open during import: {path}")
        return original_open(file, *args, **kwargs)

    connect_calls: list[Any] = []
    original_create_connection = socket.create_connection
    original_socket = socket.socket

    class GuardedSocket(original_socket):  # type: ignore[misc,valid-type]
        def connect(self, address):  # type: ignore[no-untyped-def]
            connect_calls.append(address)
            raise AssertionError(f"unexpected socket connect: {address}")

    def guarded_create_connection(address, *args, **kwargs):  # type: ignore[no-untyped-def]
        connect_calls.append(address)
        raise AssertionError(f"unexpected create_connection: {address}")

    popen_calls: list[Any] = []
    original_popen = subprocess.Popen

    class GuardedPopen(original_popen):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            popen_calls.append((args, kwargs))
            raise AssertionError(f"unexpected process start: {args!r}")

    import builtins

    builtins.open = guarded_open  # type: ignore[assignment]
    socket.socket = GuardedSocket  # type: ignore[assignment,misc]
    socket.create_connection = guarded_create_connection  # type: ignore[assignment]
    subprocess.Popen = GuardedPopen  # type: ignore[assignment,misc]
    before_modules = set(sys.modules)
    try:
        module = importlib.import_module(module_name)
        # Touch stable surface so the import is not optimized away.
        assert module.CONTROL_PLANE_CONTRACT_VERSION == 1
        assert module.StateAuthorityClass.EXPORT.value == "export"
        ControlPlaneBounds = module.ControlPlaneBounds
        bounds = ControlPlaneBounds()
        assert bounds.max_items > 0
    finally:
        builtins.open = original_open  # type: ignore[assignment]
        socket.socket = original_socket  # type: ignore[assignment,misc]
        socket.create_connection = original_create_connection  # type: ignore[assignment]
        subprocess.Popen = original_popen  # type: ignore[assignment,misc]

    assert opened == []
    assert connect_calls == []
    assert popen_calls == []

    added = set(sys.modules) - before_modules
    forbidden_roots = {
        "aiohttp",
        "anthropic",
        "duckdb",
        "httpx",
        "openai",
        "requests",
        "torch",
        "transformers",
        "urllib3",
        "neo4j",
        "sentence_transformers",
    }
    loaded_forbidden = sorted(
        {
            name.split(".", 1)[0]
            for name in added
            if name.split(".", 1)[0] in forbidden_roots
        }
    )
    assert loaded_forbidden == []

    # Module source itself must not eagerly import side-effecting packages.
    source = Path(module.__file__ or "").read_text(encoding="utf-8")
    for banned in (
        "import duckdb",
        "import requests",
        "import httpx",
        "import openai",
        "import socket",
        "import subprocess",
        "urllib.request",
    ):
        assert banned not in source


def test_fresh_process_cold_import_probe() -> None:
    """Subprocess probe: import contracts with no duckdb/network/provider load."""

    script = """
import json
import sys

before = set(sys.modules)
import ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts as m
after = set(sys.modules) - before
forbidden = {
    "aiohttp", "anthropic", "duckdb", "httpx", "openai", "requests",
    "torch", "transformers", "urllib3", "neo4j", "sentence_transformers",
}
loaded = sorted({
    name.split(".", 1)[0]
    for name in after
    if name.split(".", 1)[0] in forbidden
})
store = m.ControlPlaneStoreIdentity(
    store_uuid="550e8400-e29b-41d4-a716-446655440000",
    repository_id="repository:sha256:demo",
    generation=m.StoreGeneration(
        generation=1,
        store_uuid="550e8400-e29b-41d4-a716-446655440000",
        epoch_id="epoch:1",
        created_at_ms=0,
    ),
    schema_identity=m.SchemaIdentity(
        schema_revision=1,
        schema_fingerprint="sha256:" + ("11" * 32),
        catalog_fingerprint="sha256:" + ("22" * 32),
        migration_head=1,
    ),
)
print(json.dumps({
    "loaded_forbidden": loaded,
    "content_id_prefix": store.content_id[:1],
    "authority": store.authority.value,
}))
"""
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.pop("PYTEST_CURRENT_TEST", None)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["loaded_forbidden"] == []
    assert payload["content_id_prefix"] == "b"
    assert payload["authority"] == "authoritative"
