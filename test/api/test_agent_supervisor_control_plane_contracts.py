"""Contracts for DuckDB/Quack control-plane store identity and authority (DQP-002).

Covers closed records for store/generation/schema/session/command/revision/
fence/snapshot/export identities, authority classes, bounds, typed failures,
and redaction.  Cold import must remain free of filesystem, database, network,
provider, and process side effects.
"""

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
    ControlPlaneAuthorityError,
    ControlPlaneBounds,
    ControlPlaneBoundsError,
    ControlPlaneGenerationError,
    ControlPlaneIdentityError,
    ControlPlaneSecretError,
    ControlPlaneStoreIdentity,
    ExportFidelity,
    ExportProfile,
    FenceToken,
    MAX_REDACTION_MARK,
    SchemaIdentity,
    SessionIdentity,
    StateAuthorityClass,
    StateCommand,
    StateCommandKind,
    StateExportReceipt,
    StateRevision,
    StateSnapshot,
    StoreGeneration,
    assert_generation_revision_consistent,
    canonical_control_plane_json_bytes,
    classify_state_authority,
    redact_secrets,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

_REPO_ID = "sha256:" + ("ab" * 32)
_DB_UUID = "550e8400-e29b-41d4-a716-446655440000"
_DIGEST_A = "sha256:" + ("11" * 32)
_DIGEST_B = "sha256:" + ("22" * 32)
_DIGEST_C = "sha256:" + ("33" * 32)
_SESSION_ID = "sha256:" + ("44" * 32)
_COMMAND_ID = "sha256:" + ("55" * 32)


def _store(**changes: object) -> ControlPlaneStoreIdentity:
    values: dict[str, object] = {
        "repository_id": _REPO_ID,
        "database_uuid": _DB_UUID,
        "schema_revision": 3,
        "store_namespace": "control",
        "schema_fingerprint": _DIGEST_A,
    }
    values.update(changes)
    return ControlPlaneStoreIdentity(**values)


def _generation(**changes: object) -> StoreGeneration:
    values: dict[str, object] = {
        "store": _store(),
        "generation": 7,
        "schema_revision": 3,
        "opened_at_ms": 1_700_000_000_000,
        "parent_generation": 6,
        "generation_reason": "restore",
    }
    values.update(changes)
    return StoreGeneration(**values)


def _revision(**changes: object) -> StateRevision:
    values: dict[str, object] = {
        "generation": 7,
        "revision": 42,
        "stream": "default",
        "watermark": 100,
    }
    values.update(changes)
    return StateRevision(**values)


def _fence(**changes: object) -> FenceToken:
    values: dict[str, object] = {
        "lease_id": "lease:control-owner",
        "session_id": _SESSION_ID,
        "fencing_epoch": 9,
        "scope": "store",
        "expires_at_ms": 1_700_000_060_000,
    }
    values.update(changes)
    return FenceToken(**values)


def _snapshot(**changes: object) -> StateSnapshot:
    values: dict[str, object] = {
        "store_generation": _generation(),
        "revision": _revision(),
        "snapshot_digest": _DIGEST_B,
        "captured_at_ms": 1_700_000_001_000,
        "row_count": 12,
    }
    values.update(changes)
    return StateSnapshot(**values)


def _command(**changes: object) -> StateCommand:
    values: dict[str, object] = {
        "kind": StateCommandKind.MUTATE,
        "store_generation": _generation(),
        "expected_revision": _revision(),
        "command_id": _COMMAND_ID,
        "idempotency_key": "mutate/task-1/attempt-1",
        "fence": _fence(),
        "parameters": {"target": "tasks", "op": "claim"},
        "issued_at_ms": 1_700_000_002_000,
    }
    values.update(changes)
    return StateCommand(**values)


def _export(**changes: object) -> StateExportReceipt:
    values: dict[str, object] = {
        "snapshot": _snapshot(),
        "profile": ExportProfile.JSON,
        "fidelity": ExportFidelity.LOSSLESS,
        "artifact_digest": _DIGEST_C,
        "renderer_revision": "export-renderer@1",
        "view_revision": "state-view@1",
        "destination": "exports/control-plane.json",
        "parameters": {"include_events": True},
        "exported_at_ms": 1_700_000_003_000,
    }
    values.update(changes)
    return StateExportReceipt(**values)


def test_store_generation_command_snapshot_export_round_trip() -> None:
    store = _store()
    generation = _generation(store=store)
    command = _command(store_generation=generation)
    snapshot = _snapshot(store_generation=generation)
    export = _export(snapshot=snapshot)

    assert ControlPlaneStoreIdentity.from_json(store.to_json()) == store
    assert StoreGeneration.from_json(generation.to_json()) == generation
    assert StateCommand.from_json(command.to_json()) == command
    assert StateSnapshot.from_json(snapshot.to_json()) == snapshot
    assert StateExportReceipt.from_json(export.to_json()) == export

    assert store.store_id
    assert generation.generation_id.startswith("b")
    assert snapshot.snapshot_id == StateSnapshot.from_dict(snapshot.to_dict()).snapshot_id
    assert export.export_id == StateExportReceipt.from_dict(export.to_dict()).export_id
    assert export.authority_class is StateAuthorityClass.EXPORT
    assert canonical_control_plane_json_bytes(store) == store.canonical_bytes()


def test_contracts_are_deeply_immutable_and_identity_stable() -> None:
    store = _store()
    command = _command(parameters={"b": 2, "a": 1})
    same = _command(parameters={"a": 1, "b": 2})

    with pytest.raises(FrozenInstanceError):
        store.schema_revision = 99  # type: ignore[misc]
    with pytest.raises(TypeError):
        command.parameters["a"] = 3  # type: ignore[index]

    assert command.content_id == same.content_id
    assert command.to_json() == same.to_json()


def test_rejects_empty_forged_and_inconsistent_ids() -> None:
    with pytest.raises(ControlPlaneIdentityError, match="must not be empty"):
        _store(repository_id="")
    with pytest.raises(ControlPlaneIdentityError, match="UUID"):
        _store(database_uuid="not-a-uuid")
    with pytest.raises(ControlPlaneIdentityError, match="digest"):
        _store(schema_fingerprint="md5:deadbeef")

    forged = _store().to_record()
    forged["repository_id"] = "sha256:" + ("ff" * 32)
    with pytest.raises(ControlPlaneIdentityError, match="forged|inconsistent"):
        ControlPlaneStoreIdentity.from_dict(forged)

    bad_store_id = _store().to_dict()
    bad_store_id["store_id"] = "sha256:" + ("00" * 32)
    with pytest.raises(ControlPlaneIdentityError, match="store_id"):
        ControlPlaneStoreIdentity.from_dict(bad_store_id)

    bad_generation = _generation().to_dict()
    bad_generation["generation_id"] = "b" + ("a" * 50)
    with pytest.raises(ControlPlaneIdentityError, match="generation_id"):
        StoreGeneration.from_dict(bad_generation)

    bad_export = _export().to_dict()
    bad_export["export_id"] = "b" + ("c" * 50)
    with pytest.raises(ControlPlaneIdentityError, match="export_id"):
        StateExportReceipt.from_dict(bad_export)


def test_rejects_non_finite_bounds() -> None:
    with pytest.raises(ControlPlaneBoundsError, match="finite"):
        ControlPlaneBounds(max_items=math.inf)  # type: ignore[arg-type]
    with pytest.raises(ControlPlaneBoundsError, match="finite"):
        ControlPlaneBounds(max_depth=math.nan)  # type: ignore[arg-type]
    with pytest.raises(ControlPlaneBoundsError, match="finite"):
        ControlPlaneBounds(max_serialized_bytes=1.5)  # type: ignore[arg-type]
    with pytest.raises(ControlPlaneBoundsError, match="finite"):
        ControlPlaneBounds(max_items=True)  # type: ignore[arg-type]
    with pytest.raises(ControlPlaneBoundsError, match="at least"):
        ControlPlaneBounds(max_depth=0)
    with pytest.raises(ControlPlaneBoundsError, match="floating|finite"):
        _command(parameters={"limit": 1.25})
    with pytest.raises(ControlPlaneBoundsError, match="floating|finite"):
        _command(parameters={"limit": math.inf})


def test_rejects_generation_revision_mismatch() -> None:
    with pytest.raises(ControlPlaneGenerationError, match="schema_revision"):
        _generation(schema_revision=99)
    with pytest.raises(ControlPlaneGenerationError, match="parent_generation"):
        _generation(parent_generation=7)
    with pytest.raises(ControlPlaneGenerationError, match="does not match"):
        _command(expected_revision=_revision(generation=1))
    with pytest.raises(ControlPlaneGenerationError, match="does not match"):
        _snapshot(revision=_revision(generation=1))
    with pytest.raises(ControlPlaneGenerationError, match="mismatch"):
        assert_generation_revision_consistent(_generation(), _revision(generation=1))

    assert_generation_revision_consistent(_generation(), _revision())


def test_rejects_secrets_and_supports_redaction() -> None:
    with pytest.raises(ControlPlaneSecretError, match="secret"):
        _command(parameters={"api_key": "sk-this-is-not-allowed-1234567890"})
    with pytest.raises(ControlPlaneSecretError, match="secret"):
        _command(parameters={"note": "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----"})
    with pytest.raises(ControlPlaneSecretError, match="secret"):
        _export(parameters={"token": "ghp_abcdefghijklmnopqrstuvwxyz0123456789"})

    redacted = redact_secrets(
        {
            "api_key": "secret-value",
            "safe": "ok",
            "nested": {"password": "x", "count": 1},
        }
    )
    assert redacted["api_key"] == MAX_REDACTION_MARK
    assert redacted["safe"] == "ok"
    assert redacted["nested"]["password"] == MAX_REDACTION_MARK
    assert redacted["nested"]["count"] == 1


def test_rejects_mutable_aliases_as_identity() -> None:
    with pytest.raises(ControlPlaneIdentityError, match="mutable alias"):
        ControlPlaneStoreIdentity.from_dict(
            {
                **_store().to_dict(),
                "display_name": "primary-db",
            }
        )
    with pytest.raises(ControlPlaneIdentityError, match="mutable alias"):
        StoreGeneration.from_dict(
            {
                **_generation().to_dict(),
                "alias": "current",
            }
        )
    with pytest.raises(ControlPlaneIdentityError, match="PID alias"):
        SessionIdentity(
            session_id=_SESSION_ID,
            process_birth_id="12345",
            store_generation=_generation(),
            fencing_epoch=1,
        )
    with pytest.raises(ControlPlaneIdentityError, match="mutable alias"):
        StateExportReceipt.from_dict(
            {
                **_export().to_dict(),
                "label": "nightly",
            }
        )


def test_rejects_export_labeled_authoritative() -> None:
    with pytest.raises(ControlPlaneAuthorityError, match="authoritative|authority"):
        _export(authority_class=StateAuthorityClass.AUTHORITY)
    with pytest.raises(ControlPlaneAuthorityError, match="authoritative|projections"):
        StateExportReceipt.from_dict(
            {
                **_export().to_dict(),
                "authority_class": "authoritative",
            }
        )
    with pytest.raises(ControlPlaneAuthorityError, match="authoritative|projections"):
        StateExportReceipt.from_dict(
            {
                **_export().to_dict(),
                "authority_class": "authority",
            }
        )


def test_schema_session_fence_and_authority_helpers() -> None:
    schema = SchemaIdentity(
        schema_revision=3,
        migration_catalog_digest=_DIGEST_A,
    )
    session = SessionIdentity(
        session_id=_SESSION_ID,
        process_birth_id="birth:host-1:boot-9:pid-ns-4",
        store_generation=_generation(),
        fencing_epoch=9,
        principal_id="principal:state-owner",
    )
    fence = _fence()
    assert SchemaIdentity.from_json(schema.to_json()) == schema
    assert SessionIdentity.from_json(session.to_json()) == session
    assert FenceToken.from_json(fence.to_json()) == fence
    assert classify_state_authority("export") is StateAuthorityClass.EXPORT
    assert StateAuthorityClass.AUTHORITY.may_authorize_decisions is True
    assert StateAuthorityClass.EXPORT.may_authorize_decisions is False
    assert StateCommandKind.MUTATE.requires_fence is True
    assert StateCommandKind.READ.requires_fence is False

    with pytest.raises(Exception, match="fence"):
        _command(kind=StateCommandKind.MUTATE, fence=None)
    with pytest.raises(Exception, match="idempotency"):
        _command(idempotency_key="")
    read = _command(
        kind=StateCommandKind.READ,
        fence=None,
        idempotency_key="",
        parameters={"limit": 10},
    )
    assert read.kind is StateCommandKind.READ


def test_lossy_markdown_export_requires_omitted_fields() -> None:
    with pytest.raises(Exception, match="omitted fields"):
        _export(
            profile=ExportProfile.MARKDOWN,
            fidelity=ExportFidelity.INTENTIONALLY_LOSSY,
            intentionally_omitted_fields=(),
        )
    receipt = _export(
        profile=ExportProfile.MARKDOWN,
        fidelity=ExportFidelity.INTENTIONALLY_LOSSY,
        intentionally_omitted_fields=("raw_event_payload", "internal_metrics"),
    )
    assert receipt.fidelity is ExportFidelity.INTENTIONALLY_LOSSY
    assert "raw_event_payload" in receipt.intentionally_omitted_fields


def test_json_round_trip_preserves_contract_version() -> None:
    payload = json.loads(_export().to_json())
    assert payload["contract_version"] == 1
    assert payload["schema"].endswith("@1")
    assert payload["authority_class"] == "export"
    assert payload["snapshot"]["authority_class"] == "authority"


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env["IPFS_ACCEL_SKIP_CORE"] = "1"
    env["IPFS_ACCEL_IMPORT_EAGER"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("PYTEST_VERSION", None)
    return env


def test_cold_import_has_no_filesystem_database_network_provider_or_process_action() -> None:
    """Fresh-process import must stay pure: no I/O, sockets, DB, or providers."""

    script = r"""
import json
import socket
import subprocess
import sys

forbidden_roots = (
    "aiohttp",
    "anthropic",
    "duckdb",
    "httpx",
    "openai",
    "requests",
    "sentence_transformers",
    "torch",
    "transformers",
    "urllib3",
)

# Instrument before import so cold-import side effects are observed.
write_opens = []
db_opens = []
connected = []
popen_calls = []
original_open = open

def tracking_open(*args, **kwargs):
    path = args[0] if args else kwargs.get("file", "")
    mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
    text = str(path)
    mode_text = str(mode)
    # Flag database-shaped paths and any write/append regardless of import reads.
    if any(
        token in text
        for token in (".duckdb", ".sqlite", "control.duckdb")
    ):
        db_opens.append({"path": text, "mode": mode_text})
    if any(flag in mode_text for flag in ("w", "a", "x", "+")):
        # Bytecode writes are disabled via PYTHONDONTWRITEBYTECODE; any remaining
        # write during cold import is a contract violation.
        write_opens.append({"path": text, "mode": mode_text})
    return original_open(*args, **kwargs)

import builtins
builtins.open = tracking_open

def _no_connect(*args, **kwargs):
    connected.append({"args": [str(a) for a in args]})
    raise RuntimeError("network connect attempted during cold import")
socket.create_connection = _no_connect

def _no_socket_connect(self, *args, **kwargs):
    connected.append({"sock": True, "args": [str(a) for a in args]})
    raise RuntimeError("socket.connect attempted during cold import")
socket.socket.connect = _no_socket_connect

def _no_popen(*args, **kwargs):
    popen_calls.append({"args": [str(a) for a in args]})
    raise RuntimeError("subprocess attempted during cold import")
subprocess.Popen = _no_popen

before = set(sys.modules)
import ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts as mod
after = set(sys.modules)

loaded_forbidden = sorted({
    name.split(".")[0]
    for name in (after - before)
    if name.split(".")[0] in forbidden_roots
})

# Exercise pure helpers; still no side effects.
bounds = mod.ControlPlaneBounds()
store = mod.ControlPlaneStoreIdentity(
    repository_id="sha256:" + ("ab" * 32),
    database_uuid="550e8400-e29b-41d4-a716-446655440000",
    schema_revision=1,
)
_ = store.content_id
_ = mod.redact_secrets({"api_key": "x", "ok": 1})
_ = mod.classify_state_authority("export")

print(json.dumps({
    "ok": True,
    "module": mod.__name__,
    "write_opens": write_opens,
    "db_opens": db_opens,
    "connected": connected,
    "popen_calls": popen_calls,
    "loaded_forbidden": loaded_forbidden,
    "store_id_prefix": store.store_id[:1],
    "bounds_max_items": bounds.max_items,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=_subprocess_env(),
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True
    assert payload["write_opens"] == []
    assert payload["db_opens"] == []
    assert payload["connected"] == []
    assert payload["popen_calls"] == []
    assert payload["loaded_forbidden"] == []
    assert payload["store_id_prefix"] == "b"
    assert payload["bounds_max_items"] >= 1
