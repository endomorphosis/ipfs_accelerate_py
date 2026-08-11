"""Contract tests for DuckDB control-plane identities and authority (DQP-002)."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    REDACTION_MARKER,
    CommandKind,
    ControlPlaneAuthorityError,
    ControlPlaneBounds,
    ControlPlaneBoundsError,
    ControlPlaneContractError,
    ControlPlaneGenerationError,
    ControlPlaneIdentityError,
    ControlPlaneSecretError,
    ControlPlaneStoreIdentity,
    SecretHandle,
    StateAuthorityClass,
    StateCommand,
    StateExportReceipt,
    StateSnapshot,
    StoreGeneration,
    closed_authority_classes,
    closed_command_kinds,
    closed_identity_kinds,
    content_identity,
    is_secret_handle,
    redact_mapping,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_DIGEST = "sha256:" + ("ab" * 32)
_UUID = "123e4567-e89b-12d3-a456-426614174000"


def _store_identity(**changes: object) -> ControlPlaneStoreIdentity:
    values: dict[str, object] = {
        "repository_id": "repository:sha256:test-repo",
        "database_uuid": _UUID,
        "store_id": "control.duckdb",
        "schema_revision": 1,
        "generation": 1,
        "schema_fingerprint": _DIGEST,
        "authority_class": StateAuthorityClass.AUTHORITATIVE,
        "server_birth_id": "birth:server-1",
        "extension_fingerprint": _DIGEST,
        "metadata": {"domain": "control"},
    }
    values.update(changes)
    return ControlPlaneStoreIdentity(**values)  # type: ignore[arg-type]


def _generation(**changes: object) -> StoreGeneration:
    values: dict[str, object] = {
        "store_id": "control.duckdb",
        "generation": 1,
        "schema_revision": 1,
        "fence_epoch": 3,
        "revision": 7,
        "database_uuid": _UUID,
        "birth_id": "birth:server-1",
    }
    values.update(changes)
    return StoreGeneration(**values)  # type: ignore[arg-type]


def _command(**changes: object) -> StateCommand:
    values: dict[str, object] = {
        "command_id": "cmd:claim-1",
        "command_kind": CommandKind.CLAIM,
        "store_id": "control.duckdb",
        "session_id": "session:worker-1",
        "expected_generation": 1,
        "expected_revision": 7,
        "fence_epoch": 3,
        "idempotency_key": "idem:claim-1",
        "authority_class": StateAuthorityClass.AUTHORITATIVE,
        "parameters": {"task_id": "DQP-002"},
        "secret_handle": "",
    }
    values.update(changes)
    return StateCommand(**values)  # type: ignore[arg-type]


def _snapshot(**changes: object) -> StateSnapshot:
    values: dict[str, object] = {
        "snapshot_id": "snap:1",
        "store_id": "control.duckdb",
        "database_uuid": _UUID,
        "generation": 1,
        "schema_revision": 1,
        "revision": 7,
        "fence_epoch": 3,
        "event_watermark": 42,
        "snapshot_digest": _DIGEST,
        "authority_class": StateAuthorityClass.AUTHORITATIVE,
    }
    values.update(changes)
    return StateSnapshot(**values)  # type: ignore[arg-type]


def _export(**changes: object) -> StateExportReceipt:
    values: dict[str, object] = {
        "export_id": "export:taskboard-1",
        "snapshot_id": "snap:1",
        "store_id": "control.duckdb",
        "database_uuid": _UUID,
        "schema_revision": 1,
        "generation": 1,
        "revision": 7,
        "event_watermark": 42,
        "renderer_revision": "renderer:markdown@1",
        "query_revision": "view:ready_tasks@1",
        "artifact_digest": _DIGEST,
        "destination": "docs/exports/taskboard.md",
        "parameters": {"profile": "human-taskboard"},
        "authority_class": StateAuthorityClass.EXPORT,
        "intentional_loss": True,
    }
    values.update(changes)
    return StateExportReceipt(**values)  # type: ignore[arg-type]


def test_store_identity_round_trip_and_stable_cid() -> None:
    identity = _store_identity()
    payload = identity.to_record()
    restored = ControlPlaneStoreIdentity.from_dict(payload)
    assert restored.to_dict() == identity.to_dict()
    assert restored.content_id == identity.content_id
    assert payload["content_id"] == identity.content_id
    assert content_identity(identity.to_dict()) == identity.content_id


def test_empty_and_forged_ids_rejected() -> None:
    with pytest.raises(ControlPlaneIdentityError):
        _store_identity(repository_id="")
    with pytest.raises(ControlPlaneIdentityError):
        _store_identity(database_uuid="not-a-uuid")
    with pytest.raises(ControlPlaneIdentityError):
        _store_identity(schema_fingerprint="md5:deadbeef")
    with pytest.raises(ControlPlaneIdentityError):
        _generation(store_id="has spaces")

    good = _store_identity()
    forged = good.to_record()
    forged["content_id"] = "b" + ("a" * 58)
    with pytest.raises(ControlPlaneIdentityError, match="forged"):
        ControlPlaneStoreIdentity.from_dict(forged)

    forged_gen = _generation().to_record()
    forged_gen["content_id"] = "b" + ("c" * 58)
    with pytest.raises(ControlPlaneIdentityError, match="forged"):
        StoreGeneration.from_dict(forged_gen)


def test_non_finite_and_float_bounds_rejected() -> None:
    with pytest.raises(ControlPlaneBoundsError):
        ControlPlaneBounds(max_depth=0)
    with pytest.raises(ControlPlaneBoundsError):
        ControlPlaneBounds(max_record_bytes=-1)
    with pytest.raises(ControlPlaneBoundsError):
        ControlPlaneBounds.from_dict(
            {
                "schema": ControlPlaneBounds.SCHEMA,
                "max_depth": float("inf"),
            }
        )
    with pytest.raises(ControlPlaneBoundsError):
        ControlPlaneBounds.from_dict(
            {
                "schema": ControlPlaneBounds.SCHEMA,
                "max_depth": math.nan,
            }
        )
    with pytest.raises(ControlPlaneBoundsError):
        _store_identity(metadata={"score": 1.5})
    with pytest.raises(ControlPlaneBoundsError):
        _command(parameters={"limit": float("nan")})


def test_generation_revision_mismatch_rejected() -> None:
    base = _generation(generation=2, schema_revision=3, revision=10, fence_epoch=5)
    newer = _generation(generation=2, schema_revision=3, revision=11, fence_epoch=5)
    base.assert_compatible_with(newer)

    stale_revision = _generation(
        generation=2, schema_revision=3, revision=9, fence_epoch=5
    )
    with pytest.raises(ControlPlaneGenerationError):
        base.assert_compatible_with(stale_revision)

    schema_drift = _generation(
        generation=2, schema_revision=4, revision=10, fence_epoch=5
    )
    with pytest.raises(ControlPlaneGenerationError):
        base.assert_compatible_with(schema_drift)

    downgrade = _generation(
        generation=1, schema_revision=3, revision=10, fence_epoch=5
    )
    with pytest.raises(ControlPlaneGenerationError):
        base.assert_compatible_with(downgrade)

    command = _command(
        expected_generation=base.generation,
        expected_revision=base.revision,
        fence_epoch=base.fence_epoch,
    )
    command.assert_matches_generation(base)
    with pytest.raises(ControlPlaneGenerationError):
        command.assert_matches_generation(newer)


def test_secrets_rejected_without_embedding_material() -> None:
    # Reject by secret-bearing field name; values stay short / non-credential.
    with pytest.raises(ControlPlaneSecretError):
        _store_identity(metadata={"api_key": "x"})
    with pytest.raises(ControlPlaneSecretError):
        _command(parameters={"password": "x"})
    with pytest.raises(ControlPlaneSecretError):
        _export(parameters={"client_secret": "x"})
    with pytest.raises(ControlPlaneSecretError):
        _command(secret_handle="plaintext-not-a-handle")
    with pytest.raises(ControlPlaneSecretError):
        SecretHandle(handle="not-opaque")

    handle = SecretHandle(handle="env://CONTROL_PLANE_QUACK_TOKEN", generation=1)
    assert handle.authority_class is StateAuthorityClass.SECRET_HANDLE
    assert is_secret_handle(handle.handle)

    # Redaction replaces secret-bearing keys with the classification marker.
    redacted = redact_mapping({"ok": 1, "password": "x", "nested": {"token": "y"}})
    assert redacted["password"] == REDACTION_MARKER
    assert redacted["nested"]["token"] == REDACTION_MARKER
    assert redacted["ok"] == 1
    assert REDACTION_MARKER == "secret_material"


def test_mutable_aliases_cannot_be_identity() -> None:
    with pytest.raises(ControlPlaneIdentityError, match="mutable aliases"):
        _store_identity(metadata={"display_name": "pretty"})
    with pytest.raises(ControlPlaneIdentityError, match="mutable aliases"):
        _store_identity(metadata={"worktree_path": "/tmp/wt"})
    with pytest.raises(ControlPlaneIdentityError, match="mutable aliases"):
        _store_identity(metadata={"pid": "1234"})


def test_export_cannot_be_labeled_authoritative() -> None:
    with pytest.raises(ControlPlaneAuthorityError, match="authoritative"):
        _export(authority_class=StateAuthorityClass.AUTHORITATIVE)
    with pytest.raises(ControlPlaneAuthorityError):
        _export(authority_class=StateAuthorityClass.SECRET_HANDLE)
    with pytest.raises(ControlPlaneAuthorityError):
        _store_identity(authority_class=StateAuthorityClass.EXPORT)
    with pytest.raises(ControlPlaneAuthorityError):
        _command(authority_class=StateAuthorityClass.EXPORT)
    with pytest.raises(ControlPlaneAuthorityError):
        _snapshot(authority_class=StateAuthorityClass.EXPORT)

    receipt = _export()
    snap = _snapshot()
    assert receipt.binds_snapshot(snap)
    assert receipt.authority_class is StateAuthorityClass.EXPORT
    restored = StateExportReceipt.from_dict(receipt.to_record())
    assert restored.content_id == receipt.content_id


def test_command_snapshot_export_round_trips() -> None:
    command = _command(secret_handle="vault://quack/rotation-1")
    assert StateCommand.from_dict(command.to_record()).to_dict() == command.to_dict()

    snapshot = _snapshot()
    assert StateSnapshot.from_dict(snapshot.to_record()).content_id == snapshot.content_id
    assert snapshot.to_generation().store_id == snapshot.store_id

    bounds = ControlPlaneBounds()
    assert ControlPlaneBounds.from_dict(bounds.to_dict()).to_dict() == bounds.to_dict()


def test_unknown_fields_and_closed_vocabularies() -> None:
    payload = _store_identity().to_dict()
    payload["extra_field"] = "nope"
    with pytest.raises(ControlPlaneContractError, match="unsupported fields"):
        ControlPlaneStoreIdentity.from_dict(payload)

    assert "authoritative" in closed_authority_classes()
    assert "export" in closed_authority_classes()
    assert "database" in closed_identity_kinds()
    assert "claim" in closed_command_kinds()

    with pytest.raises(ControlPlaneContractError):
        _command(command_kind="not-a-command")


def test_cold_import_is_side_effect_free() -> None:
    """Fresh process: import performs no FS/DB/network/provider/process action."""

    module_path = (
        REPO_ROOT
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "task_sources"
        / "control_plane_contracts.py"
    )
    script = f"""
import importlib.util
import json
import sys

path = {str(module_path)!r}
before = set(sys.modules)
spec = importlib.util.spec_from_file_location(
    "control_plane_contracts_cold", path
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
after = set(sys.modules) - before
forbidden_roots = {{
    "duckdb",
    "httpx",
    "openai",
    "anthropic",
    "requests",
    "aiohttp",
    "urllib3",
    "torch",
    "transformers",
}}
loaded = sorted(
    {{
        name.split(".", 1)[0]
        for name in after
        if name.split(".", 1)[0] in forbidden_roots
    }}
)
assert module.ControlPlaneStoreIdentity.INTERFACE == "ControlPlaneStoreIdentity@1"
assert module.StoreGeneration.INTERFACE == "StoreGeneration@1"
assert module.StateCommand.INTERFACE == "StateCommand@1"
assert module.StateSnapshot.INTERFACE == "StateSnapshot@1"
assert module.StateExportReceipt.INTERFACE == "StateExportReceipt@1"
# Construct a minimal record to prove pure validation (no I/O).
digest = "sha256:" + ("ab" * 32)
identity = module.ControlPlaneStoreIdentity(
    repository_id="repository:sha256:cold",
    database_uuid="123e4567-e89b-12d3-a456-426614174000",
    store_id="control.duckdb",
    schema_revision=0,
    generation=1,
    schema_fingerprint=digest,
)
assert identity.content_id.startswith("b")
print(json.dumps({{"loaded_forbidden": loaded, "ok": True}}))
"""
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["IPFS_ACCEL_SKIP_CORE"] = "1"
    env["IPFS_ACCEL_IMPORT_EAGER"] = "0"
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
    assert payload["ok"] is True
    assert payload["loaded_forbidden"] == []
