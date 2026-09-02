"""Tests for the loopback Quack state-owner service (DQP-006).

Acceptance:

* No token appears in argv, logs, status, exports, or provider environment
* A second owner fails closed
* Ready requires live query plus matching store/generation/schema/server identities
* Non-loopback bind requires a separately reviewed policy unavailable by default
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server as quack_state_server_module
import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    DEFAULT_LOOPBACK_HOST,
    QUACK_STATE_SERVER_INTERFACE,
    STATE_SERVER_IDENTITY_INTERFACE,
    ExclusiveOwnerLease,
    FakeQuackTransport,
    OwnerMarker,
    QuackStateServer,
    QuackStateServerBindError,
    QuackStateServerCapabilityError,
    QuackStateServerConfig,
    QuackStateServerControlError,
    QuackStateServerOwnershipError,
    QuackStateServerReadyError,
    QuackStateServerTokenCompromisedError,
    QuackStateServerTokenError,
    RemoteBindPolicy,
    ServerLifecycle,
    StateServerIdentity,
    TokenHandoffRetirement,
    TokenVault,
    assert_bind_admitted,
    begin_token_handoff_retirement,
    build_server,
    listen_uri,
    provider_safe_environment,
    reclaim_stale_owner_marker,
    rearm_token_handoff,
    rearm_token_handoff_if_coordinator_absent,
    recover_stale_state_server,
    retire_token_handoff,
    sanitize_for_export,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    MigrationRunReport,
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    DEFAULT_QUACK_BETA_LIMITATIONS,
    ExtensionObservation,
    ParsedVersion,
    QuackCapabilityReport,
    QuackCapabilityStatus,
    default_compatibility_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OPS_SCRIPT = REPO_ROOT / "scripts" / "ops" / "agent_supervisor" / "quack_state_server.py"
_DIGEST = "sha256:" + ("ab" * 32)
_UUID = "123e4567-e89b-12d3-a456-426614174000"


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


class _Result:
    def __init__(self, row: Any = None) -> None:
        self._row = row

    def fetchone(self) -> Any:
        return self._row

    def fetchall(self) -> list[Any]:
        return [] if self._row is None else [self._row]


class FakeConnection:
    """Minimal DuckDB stand-in for hermetic state-owner tests."""

    def __init__(
        self,
        *,
        database_uuid: str = _UUID,
        schema_version: str = "1",
        schema_fingerprint: str = _DIGEST,
        max_generation: int = 0,
    ) -> None:
        self.database_uuid = database_uuid
        self.schema_version = schema_version
        self.schema_fingerprint = schema_fingerprint
        self.max_generation = max_generation
        self.statements: list[str] = []
        self.closed = False
        self._meta = {
            "database_uuid": database_uuid,
            "schema_version": schema_version,
            "schema_fingerprint": schema_fingerprint,
        }

    def execute(self, sql: str, params: Any = None) -> _Result:
        text = " ".join(str(sql).strip().split())
        self.statements.append(text)
        upper = text.upper()
        if "FROM CONTROL_PLANE_METADATA" in upper and "KEY" in upper:
            key = None
            if params:
                key = params[0] if not isinstance(params, dict) else params.get("key")
            if key is None and "KEY =" in upper:
                # Not parameterized in some paths
                pass
            return _Result((self._meta.get(str(key), ""),))
        if "FROM STORE_GENERATIONS" in upper or "MAX(GENERATION)" in upper:
            return _Result((self.max_generation,))
        if upper.startswith("SELECT 1"):
            return _Result((1,))
        if upper.startswith("CHECKPOINT"):
            return _Result()
        if upper.startswith("INSERT ") or upper.startswith("UPDATE "):
            return _Result()
        if upper.startswith("LOAD "):
            return _Result()
        return _Result()

    def close(self) -> None:
        self.closed = True


def _compatible_report(
    *,
    status: QuackCapabilityStatus = QuackCapabilityStatus.COMPATIBLE,
    fingerprint: str = _DIGEST,
) -> QuackCapabilityReport:
    profile = default_compatibility_profile()
    return QuackCapabilityReport(
        status=status,
        profile=profile,
        duckdb_importable=True,
        duckdb_version="1.5.2",
        duckdb_version_parsed=ParsedVersion(1, 5, 2, raw="1.5.2"),
        platform_name="Linux",
        platform_machine="x86_64",
        extension=ExtensionObservation(
            name="quack",
            installed=True,
            loaded=True,
            install_path="/tmp/quack.duckdb_extension",
            extension_version="0.1.0",
        ),
        extension_fingerprint=fingerprint,
        observed_functions=("quack_serve", "quack_query"),
        observed_surfaces=profile.required_surfaces,
        beta_limitations=DEFAULT_QUACK_BETA_LIMITATIONS,
    )


def _migration_report() -> MigrationRunReport:
    return MigrationRunReport(
        from_version=0,
        to_version=1,
        receipts=(),
        schema_fingerprint=_DIGEST,
        catalog_fingerprint=_DIGEST,
        changed=True,
    )


def _birth(*, pid: int = 4242, ticks: int = 999, boot: str = "boot-1") -> ProcessBirthIdentity:
    return ProcessBirthIdentity(
        pid=pid,
        start_time_ticks=ticks,
        boot_id=boot,
        parent_pid=1,
    )


def _server(
    tmp_path: Path,
    *,
    host: str = DEFAULT_LOOPBACK_HOST,
    port: int = 0,
    remote_policy: RemoteBindPolicy | None = None,
    transport: FakeQuackTransport | None = None,
    capability: QuackCapabilityReport | None = None,
    schema_version: str = "1",
    liveness: OwnerLiveness = OwnerLiveness.DEAD,
    birth: ProcessBirthIdentity | None = None,
    secret_handle: str = "",
) -> QuackStateServer:
    db = tmp_path / "control.duckdb"
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    connection = FakeConnection(schema_version=schema_version)
    live_map = {"value": liveness}

    def probe_liveness(_birth: ProcessBirthIdentity) -> OwnerLiveness:
        return live_map["value"]

    return build_server(
        database_path=db,
        state_dir=state,
        host=host,
        port=port,
        repository_id="repository:sha256:test",
        secret_handle=secret_handle,
        remote_bind_policy=remote_policy,
        transport=transport or FakeQuackTransport(),
        capability_probe=lambda **_kwargs: capability or _compatible_report(),
        migrate=lambda _path: _migration_report(),
        connection_factory=lambda _path: connection,
        process_birth_factory=lambda: birth or _birth(),
        owner_liveness_probe=probe_liveness,
    )


# ---------------------------------------------------------------------------
# Interface identity
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert QUACK_STATE_SERVER_INTERFACE == "QuackStateServer@1"
    assert STATE_SERVER_IDENTITY_INTERFACE == "StateServerIdentity@1"
    assert QuackStateServer.INTERFACE == QUACK_STATE_SERVER_INTERFACE
    assert StateServerIdentity.INTERFACE == STATE_SERVER_IDENTITY_INTERFACE


# ---------------------------------------------------------------------------
# Bind policy
# ---------------------------------------------------------------------------


def test_loopback_bind_admitted_by_default() -> None:
    assert_bind_admitted("127.0.0.1")
    assert_bind_admitted("::1")
    assert_bind_admitted("localhost")


def test_non_loopback_bind_requires_reviewed_policy() -> None:
    with pytest.raises(QuackStateServerBindError, match="separately reviewed"):
        assert_bind_admitted("0.0.0.0")
    with pytest.raises(QuackStateServerBindError, match="unavailable by default"):
        QuackStateServerConfig(
            database_path=Path("/tmp/control.duckdb"),
            state_dir=Path("/tmp/state"),
            host="0.0.0.0",
        )


def test_remote_policy_admits_listed_host_only() -> None:
    policy = RemoteBindPolicy(
        policy_id="policy:remote-1",
        reviewed_by="security-reviewer",
        review_receipt="receipt:sha256:deadbeef",
        allowed_hosts=("10.0.0.5",),
    )
    assert_bind_admitted("10.0.0.5", remote_policy=policy)
    with pytest.raises(QuackStateServerBindError, match="not admitted"):
        assert_bind_admitted("10.0.0.6", remote_policy=policy)


def test_remote_policy_unavailable_without_receipt() -> None:
    with pytest.raises(QuackStateServerBindError, match="review_receipt"):
        RemoteBindPolicy(
            policy_id="policy:x",
            reviewed_by="rev",
            review_receipt="",
            allowed_hosts=("1.2.3.4",),
        )


# ---------------------------------------------------------------------------
# Token handling
# ---------------------------------------------------------------------------


def test_token_vault_mints_handle_only_and_destroys(tmp_path: Path) -> None:
    vault = TokenVault(tmp_path)
    handle = vault.mint(secret_handle="handle:quack-token:test:g1", generation=1)
    assert handle.handle.startswith("handle:")
    token = vault.resolve()
    assert token
    assert token not in handle.handle
    status = {"secret_handle": handle.handle, "token": token}
    with pytest.raises(QuackStateServerTokenError):
        vault.assert_absent_from(status, surface_name="status")
    vault.destroy()
    with pytest.raises(QuackStateServerTokenError):
        vault.resolve()


def test_token_handoff_retirement_rollback_restores_exact_bytes(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    vault = TokenVault(state_dir)
    handle = "handle:quack-token:test:g1"
    vault.mint(secret_handle=handle, generation=1)
    token = vault.resolve()
    token_path = next(state_dir.glob("*.quack-token"))

    transaction = begin_token_handoff_retirement(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    )

    assert transaction.state == "begun"
    assert transaction.credential_sha256 == (
        "sha256:" + hashlib.sha256(token.encode("ascii")).hexdigest()
    )
    assert not token_path.exists()
    assert token not in repr(transaction)
    receipt = transaction.rollback()
    assert receipt == transaction.rollback(), "rollback must be idempotent"
    assert receipt["restored"] is True
    assert transaction.state == "rolled_back"
    assert token_path.read_bytes() == token.encode("ascii")
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
    with pytest.raises(QuackStateServerTokenError, match="after rollback"):
        transaction.commit()
    vault.destroy()


def test_abandoned_token_handoff_rolls_back_and_releases_fds_and_lock(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    vault = TokenVault(state_dir)
    handle = "handle:quack-token:test:abandoned"
    vault.mint(secret_handle=handle, generation=1)
    token = vault.resolve()
    token_path = next(state_dir.glob("*.quack-token"))
    transaction = begin_token_handoff_retirement(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    )
    directory_fd = transaction._directory_fd  # noqa: SLF001 - lifecycle seam
    lock_fd = transaction._lock_fd  # noqa: SLF001 - lifecycle seam
    retained_bytes = transaction._token_bytes  # noqa: SLF001 - wipe seam

    del transaction
    gc.collect()

    assert token_path.read_bytes() == token.encode("ascii")
    assert retained_bytes == bytearray()
    for descriptor in (directory_fd, lock_fd):
        with pytest.raises(OSError):
            os.fstat(descriptor)
    replay = begin_token_handoff_retirement(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    )
    assert replay.rollback()["restored"] is True
    vault.destroy()


def test_abandoned_token_handoff_closes_and_wipes_when_rollback_is_blocked(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    vault = TokenVault(state_dir)
    handle = "handle:quack-token:test:abandoned-blocked"
    vault.mint(secret_handle=handle, generation=1)
    token = vault.resolve()
    token_path = next(state_dir.glob("*.quack-token"))
    transaction = begin_token_handoff_retirement(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    )
    directory_fd = transaction._directory_fd  # noqa: SLF001 - lifecycle seam
    lock_fd = transaction._lock_fd  # noqa: SLF001 - lifecycle seam
    retained_bytes = transaction._token_bytes  # noqa: SLF001 - wipe seam
    substitute = b"do-not-overwrite"
    token_path.write_bytes(substitute)
    token_path.chmod(0o600)

    del transaction
    gc.collect()

    assert token_path.read_bytes() == substitute
    assert retained_bytes == bytearray()
    for descriptor in (directory_fd, lock_fd):
        with pytest.raises(OSError):
            os.fstat(descriptor)
    with pytest.raises(QuackStateServerTokenError) as caught:
        begin_token_handoff_retirement(
            state_dir=state_dir,
            secret_handle=handle,
            expected_token=token,
        )
    assert "already locked" not in str(caught.value)
    token_path.unlink()
    vault.destroy()


def test_token_handoff_context_commits_or_rolls_back_with_scope(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    vault = TokenVault(state_dir)
    handle = "handle:quack-token:test:context"
    vault.mint(secret_handle=handle, generation=1)
    token = vault.resolve()
    token_path = next(state_dir.glob("*.quack-token"))

    with pytest.raises(RuntimeError, match="launch failed"):
        with begin_token_handoff_retirement(
            state_dir=state_dir,
            secret_handle=handle,
            expected_token=token,
        ):
            raise RuntimeError("launch failed")
    assert token_path.read_bytes() == token.encode("ascii")

    with begin_token_handoff_retirement(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    ) as transaction:
        assert transaction.state == "begun"
    assert transaction.state == "committed"
    assert not token_path.exists()
    vault.destroy()


def test_token_handoff_retirement_commit_is_final_and_compatible(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    vault = TokenVault(state_dir)
    handle = "handle:quack-token:test:g2"
    vault.mint(secret_handle=handle, generation=2)
    token = vault.resolve()
    token_path = next(state_dir.glob("*.quack-token"))

    transaction = begin_token_handoff_retirement(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    )
    assert transaction.credential_sha256 == (
        "sha256:" + hashlib.sha256(token.encode("ascii")).hexdigest()
    )
    receipt = transaction.commit()

    assert receipt == transaction.commit(), "commit must be idempotent"
    assert receipt == {
        "schema": "ipfs_accelerate_py/quack-token-handoff-retirement@1",
        "retired": True,
        "already_absent": False,
        "secret_handle": handle,
    }
    assert not token_path.exists()
    with pytest.raises(QuackStateServerTokenError, match="cannot be rolled back"):
        transaction.rollback()

    # The compatibility facade is still an idempotent begin+commit when the
    # authenticated coordinator already holds the in-memory credential.
    assert retire_token_handoff(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    )["already_absent"] is True
    vault.destroy()


def test_token_handoff_expected_commit_receipt_and_active_authority_are_exact(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    vault = TokenVault(state_dir)
    handle = "handle:quack-token:test:authority-binding"
    vault.mint(secret_handle=handle, generation=2)
    token = vault.resolve()
    digest = "sha256:" + hashlib.sha256(token.encode("ascii")).hexdigest()
    transaction = begin_token_handoff_retirement(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    )
    expected_receipt = {
        "schema": "ipfs_accelerate_py/quack-token-handoff-retirement@1",
        "retired": True,
        "already_absent": False,
        "secret_handle": handle,
    }

    exposed_receipt = transaction.expected_commit_receipt
    assert exposed_receipt == expected_receipt
    exposed_receipt["retired"] = False
    assert transaction.expected_commit_receipt == expected_receipt
    binding = transaction.validate_active(
        state_dir=state_dir,
        secret_handle=handle,
        credential_sha256=digest,
    )
    assert binding == {
        "schema": (
            "ipfs_accelerate_py/"
            "quack-token-handoff-authority-binding@1"
        ),
        "state_dir": str(state_dir),
        "secret_handle": handle,
        "credential_sha256": digest,
    }
    binding["state_dir"] = "substituted"
    assert transaction.validate_active(
        state_dir=state_dir,
        secret_handle=handle,
        credential_sha256=digest,
    )["state_dir"] == str(state_dir)
    with pytest.raises(QuackStateServerTokenError, match="binding differs"):
        transaction.validate_active(
            state_dir=state_dir,
            secret_handle=handle,
            credential_sha256="sha256:" + ("00" * 32),
        )

    assert transaction.commit() == expected_receipt
    assert transaction.expected_commit_receipt == expected_receipt
    with pytest.raises(QuackStateServerTokenError, match="not active"):
        transaction.validate_active(
            state_dir=state_dir,
            secret_handle=handle,
            credential_sha256=digest,
        )
    vault.destroy()


def test_token_handoff_commit_receipt_survives_post_state_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "state"
    vault = TokenVault(state_dir)
    handle = "handle:quack-token:test:post-commit-exception"
    vault.mint(secret_handle=handle, generation=2)
    token = vault.resolve()
    transaction = begin_token_handoff_retirement(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    )
    expected_receipt = transaction.expected_commit_receipt
    directory_fd = transaction._directory_fd  # noqa: SLF001 - lifecycle seam
    lock_fd = transaction._lock_fd  # noqa: SLF001 - lifecycle seam
    retained_bytes = transaction._token_bytes  # noqa: SLF001 - wipe seam
    original_release = TokenHandoffRetirement._release_resources

    def fail_after_release(
        current: TokenHandoffRetirement,
        *,
        unlock: bool = True,
    ) -> None:
        original_release(current, unlock=unlock)
        if current is transaction:
            raise RuntimeError("injected post-state cleanup failure")

    monkeypatch.setattr(
        TokenHandoffRetirement,
        "_release_resources",
        fail_after_release,
    )

    with pytest.raises(RuntimeError, match="post-state cleanup"):
        transaction.commit()

    assert transaction.state == "committed"
    assert transaction.expected_commit_receipt == expected_receipt
    assert transaction.commit() == expected_receipt
    assert retained_bytes == bytearray()
    for descriptor in (directory_fd, lock_fd):
        with pytest.raises(OSError):
            os.fstat(descriptor)
    vault.destroy()


def test_token_handoff_close_without_rollback_is_terminal_and_idempotent(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    vault = TokenVault(state_dir)
    handle = "handle:quack-token:test:close-unproven"
    vault.mint(secret_handle=handle, generation=2)
    token = vault.resolve()
    token_path = next(state_dir.glob("*.quack-token"))
    transaction = begin_token_handoff_retirement(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    )
    directory_fd = transaction._directory_fd  # noqa: SLF001 - lifecycle seam
    lock_fd = transaction._lock_fd  # noqa: SLF001 - lifecycle seam
    retained_bytes = transaction._token_bytes  # noqa: SLF001 - wipe seam

    receipt = transaction.close_without_rollback()

    assert receipt == {
        "schema": (
            "ipfs_accelerate_py/"
            "quack-token-handoff-retirement-closed@1"
        ),
        "closed": True,
        "terminal": True,
        "reason": "child_liveness_unproven",
        "completion_authority": False,
        "task_authority": False,
        "secret_handle": handle,
        "credential_sha256": (
            "sha256:" + hashlib.sha256(token.encode("ascii")).hexdigest()
        ),
    }
    receipt["closed"] = False
    assert transaction.close_without_rollback()["closed"] is True
    assert transaction.state == "closed"
    assert retained_bytes == bytearray()
    assert not token_path.exists(), "terminal close must not republish the token"
    for descriptor in (directory_fd, lock_fd):
        with pytest.raises(OSError):
            os.fstat(descriptor)
    with pytest.raises(QuackStateServerTokenError, match="reason"):
        transaction.close_without_rollback(reason="caller_uncertain")
    with pytest.raises(QuackStateServerTokenError):
        transaction.commit()
    vault.destroy()


def test_token_handoff_retirement_constructor_requires_factory_authority(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="begin_token_handoff_retirement"):
        TokenHandoffRetirement(
            _construction_authority=object(),
            directory=tmp_path,
            directory_fd=-1,
            directory_identity=(0, 0, 0, 0),
            filename="forged.quack-token",
            lock_fd=-1,
            lock_filename="forged.retirement.lock",
            lock_identity=(0, 0, 0, 0, 0),
            secret_handle="handle:quack-token:test:forged",
            credential_sha256="sha256:" + ("00" * 32),
            token_bytes=bytearray(b"forged-token"),
            already_absent=False,
        )


def test_token_handoff_retirement_absent_rollback_does_not_invent_secret(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir(mode=0o700)
    handle = "handle:quack-token:missing:g1"
    transaction = begin_token_handoff_retirement(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token="authenticated-token",
    )

    receipt = transaction.rollback()

    assert receipt["already_absent"] is True
    assert receipt["restored"] is False
    assert list(state_dir.glob("*.quack-token")) == []


def test_token_handoff_retirement_rejects_symlinked_directory_component(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "real-state"
    vault = TokenVault(state_dir)
    handle = "handle:quack-token:test:symlink-dir"
    vault.mint(secret_handle=handle, generation=1)
    token = vault.resolve()
    token_path = next(state_dir.glob("*.quack-token"))
    alias = tmp_path / "state-alias"
    alias.symlink_to(state_dir, target_is_directory=True)

    with pytest.raises(QuackStateServerTokenError, match="symbolic link"):
        begin_token_handoff_retirement(
            state_dir=alias,
            secret_handle=handle,
            expected_token=token,
        )

    assert token_path.read_bytes() == token.encode("ascii")
    vault.destroy()


def test_token_handoff_directory_open_rejects_component_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "state-open-race"
    vault = TokenVault(state_dir)
    handle = "handle:quack-token:test:directory-open-race"
    vault.mint(secret_handle=handle, generation=1)
    token = vault.resolve()
    displaced = tmp_path / "displaced-open-race"
    original_open = quack_state_server_module.os.open
    substituted = False

    def substitute_before_open(path, flags, *args, dir_fd=None, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal substituted
        if path == state_dir.name and dir_fd is not None and not substituted:
            state_dir.rename(displaced)
            state_dir.mkdir(mode=0o700)
            substituted = True
        return original_open(path, flags, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr(
        quack_state_server_module.os,
        "open",
        substitute_before_open,
    )

    with pytest.raises(QuackStateServerTokenError, match="changed while opening"):
        begin_token_handoff_retirement(
            state_dir=state_dir,
            secret_handle=handle,
            expected_token=token,
        )

    monkeypatch.setattr(quack_state_server_module.os, "open", original_open)
    assert next(displaced.glob("*.quack-token")).read_bytes() == token.encode(
        "ascii"
    )
    state_dir.rmdir()
    displaced.rename(state_dir)
    vault.destroy()


def test_token_handoff_retirement_lock_serializes_same_handoff(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    vault = TokenVault(state_dir)
    handle = "handle:quack-token:test:locked"
    vault.mint(secret_handle=handle, generation=1)
    token = vault.resolve()
    first = begin_token_handoff_retirement(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    )

    with pytest.raises(QuackStateServerTokenError, match="already locked"):
        begin_token_handoff_retirement(
            state_dir=state_dir,
            secret_handle=handle,
            expected_token=token,
        )

    assert first.rollback()["restored"] is True
    vault.destroy()


def test_token_handoff_rollback_never_publishes_partial_canonical_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "state"
    vault = TokenVault(state_dir)
    handle = "handle:quack-token:test:atomic-rollback"
    vault.mint(secret_handle=handle, generation=1)
    token = vault.resolve()
    token_path = next(state_dir.glob("*.quack-token"))
    transaction = begin_token_handoff_retirement(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    )
    original_write = quack_state_server_module.os.write
    canonical_visibility: list[bool] = []

    def short_write(descriptor: int, value) -> int:  # type: ignore[no-untyped-def]
        canonical_visibility.append(token_path.exists())
        return original_write(descriptor, value[:3])

    monkeypatch.setattr(quack_state_server_module.os, "write", short_write)

    assert transaction.rollback()["restored"] is True
    assert canonical_visibility
    assert not any(canonical_visibility)
    assert token_path.read_bytes() == token.encode("ascii")
    assert list(
        state_dir.glob(
            quack_state_server_module.TOKEN_ROLLBACK_TEMP_PREFIX + "*"
        )
    ) == []
    vault.destroy()


def test_token_handoff_rollback_write_failure_cleans_exact_temp_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "state"
    vault = TokenVault(state_dir)
    handle = "handle:quack-token:test:atomic-rollback-failure"
    vault.mint(secret_handle=handle, generation=1)
    token = vault.resolve()
    token_path = next(state_dir.glob("*.quack-token"))
    transaction = begin_token_handoff_retirement(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    )
    original_write = quack_state_server_module.os.write
    failed = False

    def fail_after_partial_temp_write(
        descriptor: int,
        value,
    ) -> int:  # type: ignore[no-untyped-def]
        nonlocal failed
        if not failed:
            failed = True
            original_write(descriptor, value[:3])
            raise OSError("injected rollback write failure")
        return original_write(descriptor, value)

    monkeypatch.setattr(
        quack_state_server_module.os,
        "write",
        fail_after_partial_temp_write,
    )

    with pytest.raises(QuackStateServerTokenError, match="could not restore"):
        transaction.rollback()

    assert transaction.state == "begun"
    assert not token_path.exists(), "partial bytes must never use canonical name"
    assert list(
        state_dir.glob(
            quack_state_server_module.TOKEN_ROLLBACK_TEMP_PREFIX + "*"
        )
    ) == []
    monkeypatch.setattr(quack_state_server_module.os, "write", original_write)
    assert transaction.rollback()["restored"] is True
    assert token_path.read_bytes() == token.encode("ascii")
    vault.destroy()


def test_token_handoff_retirement_marks_hardlink_race_compromised_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "state"
    vault = TokenVault(state_dir)
    handle = "handle:quack-token:test:hardlink-race"
    vault.mint(secret_handle=handle, generation=1)
    token = vault.resolve()
    token_path = next(state_dir.glob("*.quack-token"))
    surviving_link = state_dir / "surviving-token-link"
    original_unlink = quack_state_server_module.os.unlink

    def hardlink_then_unlink(path, *args, dir_fd=None, **kwargs):  # type: ignore[no-untyped-def]
        if path == token_path.name and dir_fd is not None:
            os.link(token_path, surviving_link)
        return original_unlink(path, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr(
        quack_state_server_module.os,
        "unlink",
        hardlink_then_unlink,
    )

    with pytest.raises(QuackStateServerTokenCompromisedError) as caught:
        begin_token_handoff_retirement(
            state_dir=state_dir,
            secret_handle=handle,
            expected_token=token,
        )

    assert not token_path.exists()
    assert surviving_link.read_bytes() == token.encode("ascii")
    assert caught.value.state == "compromised"
    assert caught.value.transaction.state == "compromised"
    assert caught.value.transaction._directory_fd == -1  # noqa: SLF001
    assert caught.value.transaction._lock_fd == -1  # noqa: SLF001
    assert caught.value.transaction._token_bytes == bytearray()  # noqa: SLF001
    assert caught.value.receipt["compromised"] is True
    assert caught.value.receipt["terminal"] is True
    assert caught.value.receipt["reason"] == "retired_inode_has_surviving_hardlink"
    assert caught.value.receipt["marker_persisted"] is True
    assert token not in json.dumps(dict(caught.value.receipt), sort_keys=True)
    marker_path = token_path.with_name(
        token_path.name + quack_state_server_module.TOKEN_COMPROMISE_MARKER_SUFFIX
    )
    assert marker_path.is_file()
    assert stat.S_IMODE(marker_path.stat().st_mode) == 0o600

    # Every later retirement/rearm path is barred by the durable marker even
    # though the compromised transaction released its lock and descriptors.
    with pytest.raises(
        QuackStateServerTokenCompromisedError,
        match="durable compromise marker",
    ):
        begin_token_handoff_retirement(
            state_dir=state_dir,
            secret_handle=handle,
            expected_token=token,
        )
    pid_dir = tmp_path / "coordinator"
    pid_dir.mkdir(mode=0o700)
    with pytest.raises(
        QuackStateServerTokenCompromisedError,
        match="durable compromise marker",
    ):
        rearm_token_handoff_if_coordinator_absent(
            state_dir=state_dir,
            secret_handle=handle,
            expected_token=token,
            coordinator_pid_path=pid_dir / "coordinator.pid",
        )
    monkeypatch.setattr(quack_state_server_module.os, "unlink", original_unlink)
    surviving_link.unlink()
    vault.destroy()


def test_token_handoff_retirement_commit_refuses_recreated_path(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    vault = TokenVault(state_dir)
    handle = "handle:quack-token:test:commit-race"
    vault.mint(secret_handle=handle, generation=1)
    token = vault.resolve()
    token_path = next(state_dir.glob("*.quack-token"))
    transaction = begin_token_handoff_retirement(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    )
    substitute = b"commit-substitute"
    token_path.write_bytes(substitute)
    token_path.chmod(0o600)

    with pytest.raises(QuackStateServerTokenError, match="recreated"):
        transaction.commit()

    assert transaction.state == "begun"
    assert token_path.read_bytes() == substitute
    token_path.unlink()
    assert transaction.rollback()["restored"] is True
    vault.destroy()


def test_token_handoff_rollback_final_check_refuses_fsync_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "state"
    vault = TokenVault(state_dir)
    handle = "handle:quack-token:test:rollback-race"
    vault.mint(secret_handle=handle, generation=1)
    token = vault.resolve()
    token_path = next(state_dir.glob("*.quack-token"))
    transaction = begin_token_handoff_retirement(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    )
    directory_fd = transaction._directory_fd  # noqa: SLF001 - race seam
    original_fsync = quack_state_server_module.os.fsync
    substituted = False
    substitute = b"rollback-final-substitute"

    def substitute_after_directory_fsync(descriptor: int) -> None:
        nonlocal substituted
        original_fsync(descriptor)
        if descriptor == directory_fd and token_path.exists() and not substituted:
            token_path.unlink()
            token_path.write_bytes(substitute)
            token_path.chmod(0o600)
            substituted = True

    monkeypatch.setattr(
        quack_state_server_module.os,
        "fsync",
        substitute_after_directory_fsync,
    )

    with pytest.raises(QuackStateServerTokenError, match="final identity"):
        transaction.rollback()

    assert token_path.read_bytes() == substitute
    monkeypatch.setattr(quack_state_server_module.os, "fsync", original_fsync)
    token_path.unlink()
    assert transaction.rollback()["restored"] is True
    assert token_path.read_bytes() == token.encode("ascii")
    vault.destroy()


def test_token_handoff_retirement_rollback_refuses_path_substitution(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    vault = TokenVault(state_dir)
    handle = "handle:quack-token:test:g3"
    vault.mint(secret_handle=handle, generation=3)
    token = vault.resolve()
    token_path = next(state_dir.glob("*.quack-token"))
    transaction = begin_token_handoff_retirement(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    )
    substitute = b"attacker-controlled-substitute"
    token_path.write_bytes(substitute)
    token_path.chmod(0o600)

    with pytest.raises(QuackStateServerTokenError, match="recreated"):
        transaction.rollback()

    assert token_path.read_bytes() == substitute
    token_path.unlink()
    assert transaction.rollback()["restored"] is True
    assert token_path.read_bytes() == token.encode("ascii")
    vault.destroy()


def test_token_handoff_retirement_rollback_refuses_directory_substitution(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    vault = TokenVault(state_dir)
    handle = "handle:quack-token:test:g4"
    vault.mint(secret_handle=handle, generation=4)
    token = vault.resolve()
    transaction = begin_token_handoff_retirement(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    )
    displaced = tmp_path / "displaced-state"
    state_dir.rename(displaced)
    state_dir.mkdir(mode=0o700)

    with pytest.raises(QuackStateServerTokenError, match="identity"):
        transaction.rollback()

    assert list(state_dir.iterdir()) == []
    state_dir.rmdir()
    displaced.rename(state_dir)
    assert transaction.rollback()["restored"] is True
    vault.destroy()


def test_token_handoff_retirement_validation_fails_before_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "state"
    vault = TokenVault(state_dir)
    handle = "handle:quack-token:test:g5"
    vault.mint(secret_handle=handle, generation=5)
    token_path = next(state_dir.glob("*.quack-token"))
    original = token_path.read_bytes()
    wipes: list[tuple[bytes, bytes]] = []
    original_wipe = quack_state_server_module._wipe_token_bytes

    def record_wipe(value: bytearray) -> None:
        before = bytes(value)
        original_wipe(value)
        wipes.append((before, bytes(value)))

    monkeypatch.setattr(
        quack_state_server_module,
        "_wipe_token_bytes",
        record_wipe,
    )

    with pytest.raises(QuackStateServerTokenError, match="authenticated owner"):
        begin_token_handoff_retirement(
            state_dir=state_dir,
            secret_handle=handle,
            expected_token="wrong-authenticated-token",
        )

    assert token_path.read_bytes() == original
    assert (original, b"") in wipes
    vault.destroy()


def test_rearm_token_handoff_is_atomic_exact_and_idempotent(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    vault = TokenVault(state_dir)
    handle = "handle:quack-token:test:rearm"
    vault.mint(secret_handle=handle, generation=6)
    token = vault.resolve()
    token_path = next(state_dir.glob("*.quack-token"))
    begin_token_handoff_retirement(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    ).commit()
    expected = {
        "schema": "ipfs_accelerate_py/quack-token-handoff-rearm@1",
        "rearmed": True,
        "secret_handle": handle,
        "credential_sha256": (
            "sha256:" + hashlib.sha256(token.encode("ascii")).hexdigest()
        ),
    }

    assert rearm_token_handoff(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    ) == expected
    assert token_path.read_bytes() == token.encode("ascii")
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
    assert rearm_token_handoff(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    ) == expected
    with pytest.raises(QuackStateServerTokenError, match="differs"):
        rearm_token_handoff(
            state_dir=state_dir,
            secret_handle=handle,
            expected_token="different-authenticated-token",
        )
    assert token_path.read_bytes() == token.encode("ascii")
    assert list(state_dir.glob(".quack-token-rollback.*")) == []
    vault.destroy()


@pytest.mark.parametrize("crash_boundary", ["pre_link", "post_link", "disjoint"])
def test_rearm_recovers_only_exact_unique_rollback_temp_artifacts(
    tmp_path: Path,
    crash_boundary: str,
) -> None:
    state_dir = tmp_path / "state"
    vault = TokenVault(state_dir)
    handle = f"handle:quack-token:test:rearm-crash-{crash_boundary}"
    vault.mint(secret_handle=handle, generation=6)
    token = vault.resolve()
    token_path = next(state_dir.glob("*.quack-token"))
    temp_path = state_dir / (
        quack_state_server_module.TOKEN_ROLLBACK_TEMP_PREFIX + ("a" * 32)
    )
    if crash_boundary == "pre_link":
        begin_token_handoff_retirement(
            state_dir=state_dir,
            secret_handle=handle,
            expected_token=token,
        ).commit()
        temp_path.write_bytes(token.encode("ascii"))
        temp_path.chmod(0o600)
    elif crash_boundary == "post_link":
        os.link(token_path, temp_path)
        assert token_path.stat().st_nlink == 2
    else:
        temp_path.write_bytes(token.encode("ascii"))
        temp_path.chmod(0o600)
        assert temp_path.stat().st_ino != token_path.stat().st_ino

    receipt = rearm_token_handoff(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    )

    assert receipt["rearmed"] is True
    assert not temp_path.exists()
    assert token_path.read_bytes() == token.encode("ascii")
    assert token_path.stat().st_nlink == 1
    assert not list(
        state_dir.glob("*" + quack_state_server_module.TOKEN_COMPROMISE_MARKER_SUFFIX)
    )
    vault.destroy()


@pytest.mark.parametrize("pid_state", ["absent", "empty", "dead"])
def test_rearm_probe_admits_only_proven_coordinator_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pid_state: str,
) -> None:
    state_dir = tmp_path / "state"
    pid_dir = tmp_path / "coordinator"
    pid_dir.mkdir(mode=0o700)
    pid_path = pid_dir / "configured-board-master.pid"
    vault = TokenVault(state_dir)
    handle = f"handle:quack-token:test:probe-{pid_state}"
    vault.mint(secret_handle=handle, generation=7)
    token = vault.resolve()
    token_path = next(state_dir.glob("*.quack-token"))
    begin_token_handoff_retirement(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    ).commit()
    if pid_state == "empty":
        pid_path.write_bytes(b"")
        pid_path.chmod(0o600)
    elif pid_state == "dead":
        pid_path.write_bytes(b"424242\n")
        pid_path.chmod(0o600)
        monkeypatch.setattr(
            quack_state_server_module,
            "_coordinator_pid_projection_liveness",
            lambda _pid: OwnerLiveness.DEAD,
        )

    receipt = rearm_token_handoff_if_coordinator_absent(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
        coordinator_pid_path=pid_path,
    )

    assert receipt == {
        "schema": "ipfs_accelerate_py/quack-token-handoff-rearm-probe@1",
        "closed": True,
        "rearmed": True,
        "pid_quarantined": pid_state in {"empty", "dead"},
        "recovery_admitted": True,
        "completion_authority": False,
        "task_authority": False,
        "reason": f"coordinator_pid_{pid_state}",
        "secret_handle": handle,
        "credential_sha256": (
            "sha256:" + hashlib.sha256(token.encode("ascii")).hexdigest()
        ),
    }
    assert token_path.read_bytes() == token.encode("ascii")
    if pid_state in {"empty", "dead"}:
        assert not pid_path.exists()
        quarantined = list(
            (pid_dir / ".quack-coordinator-pid-quarantine").iterdir()
        )
        assert len(quarantined) == 1
        assert quarantined[0].read_bytes() == (
            b"" if pid_state == "empty" else b"424242\n"
        )
    vault.destroy()


@pytest.mark.parametrize("pid_state", ["empty", "dead"])
def test_rearm_probe_quarantines_orphan_pid_when_handoff_already_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pid_state: str,
) -> None:
    state_dir = tmp_path / "state"
    pid_dir = tmp_path / "coordinator"
    pid_dir.mkdir(mode=0o700)
    pid_path = pid_dir / "configured-board-master.pid"
    vault = TokenVault(state_dir)
    handle = f"handle:quack-token:test:present-orphan-{pid_state}"
    vault.mint(secret_handle=handle, generation=7)
    token = vault.resolve()
    token_path = next(state_dir.glob("*.quack-token"))
    payload = b"" if pid_state == "empty" else b"454545\n"
    pid_path.write_bytes(payload)
    pid_path.chmod(0o600)
    if pid_state == "dead":
        monkeypatch.setattr(
            quack_state_server_module,
            "_coordinator_pid_projection_liveness",
            lambda _pid: OwnerLiveness.DEAD,
        )

    receipt = rearm_token_handoff_if_coordinator_absent(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
        coordinator_pid_path=pid_path,
    )

    assert receipt["closed"] is True
    assert receipt["rearmed"] is False
    assert receipt["pid_quarantined"] is True
    assert receipt["recovery_admitted"] is True
    assert receipt["reason"] == f"coordinator_pid_{pid_state}"
    assert token_path.read_bytes() == token.encode("ascii")
    assert not pid_path.exists()
    quarantined = list(
        (pid_dir / ".quack-coordinator-pid-quarantine").iterdir()
    )
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == payload

    # The preserved stale evidence no longer blocks the scheduler's next
    # exact O_EXCL PID reservation at its canonical pathname.
    descriptor = os.open(
        pid_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    os.close(descriptor)
    vault.destroy()


@pytest.mark.parametrize(
    ("pid_state", "expected_reason"),
    [
        ("alive", "coordinator_pid_alive"),
        ("unknown", "coordinator_pid_unknown"),
        ("malformed", "coordinator_pid_malformed"),
        ("symlink", "coordinator_pid_unsafe"),
        ("hardlink", "coordinator_pid_unsafe"),
    ],
)
def test_rearm_probe_fails_closed_on_live_unknown_or_unsafe_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pid_state: str,
    expected_reason: str,
) -> None:
    state_dir = tmp_path / "state"
    pid_dir = tmp_path / "coordinator"
    pid_dir.mkdir(mode=0o700)
    pid_path = pid_dir / "configured-board-master.pid"
    vault = TokenVault(state_dir)
    handle = f"handle:quack-token:test:probe-noop-{pid_state}"
    vault.mint(secret_handle=handle, generation=8)
    token = vault.resolve()
    token_path = next(state_dir.glob("*.quack-token"))
    begin_token_handoff_retirement(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    ).commit()
    if pid_state in {"alive", "unknown"}:
        pid_path.write_bytes(b"434343\n")
        pid_path.chmod(0o600)
        liveness = (
            OwnerLiveness.ALIVE if pid_state == "alive" else OwnerLiveness.UNKNOWN
        )
        monkeypatch.setattr(
            quack_state_server_module,
            "_coordinator_pid_projection_liveness",
            lambda _pid: liveness,
        )
    elif pid_state == "malformed":
        pid_path.write_bytes(b"not-a-pid\n")
        pid_path.chmod(0o600)
    elif pid_state == "symlink":
        target = pid_dir / "pid-target"
        target.write_bytes(b"434343\n")
        target.chmod(0o600)
        pid_path.symlink_to(target.name)
    else:
        target = pid_dir / "pid-hardlink-target"
        target.write_bytes(b"434343\n")
        target.chmod(0o600)
        os.link(target, pid_path)

    receipt = rearm_token_handoff_if_coordinator_absent(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
        coordinator_pid_path=pid_path,
    )

    assert receipt["rearmed"] is False
    assert receipt["closed"] is True
    assert receipt["pid_quarantined"] is False
    assert receipt["recovery_admitted"] is False
    assert receipt["completion_authority"] is False
    assert receipt["task_authority"] is False
    assert receipt["reason"] == expected_reason
    assert not token_path.exists()
    assert pid_path.exists() or pid_path.is_symlink()
    vault.destroy()


def test_rearm_probe_is_nonblocking_while_retirement_lock_is_held(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    pid_dir = tmp_path / "coordinator"
    pid_dir.mkdir(mode=0o700)
    pid_path = pid_dir / "configured-board-master.pid"
    vault = TokenVault(state_dir)
    handle = "handle:quack-token:test:probe-lock"
    vault.mint(secret_handle=handle, generation=9)
    token = vault.resolve()
    token_path = next(state_dir.glob("*.quack-token"))
    retirement = begin_token_handoff_retirement(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
    )

    receipt = rearm_token_handoff_if_coordinator_absent(
        state_dir=state_dir,
        secret_handle=handle,
        expected_token=token,
        coordinator_pid_path=pid_path,
    )

    assert receipt["rearmed"] is False
    assert receipt["closed"] is True
    assert receipt["pid_quarantined"] is False
    assert receipt["recovery_admitted"] is False
    assert receipt["reason"] == "retirement_lock_held"
    assert not token_path.exists()
    assert retirement.rollback()["restored"] is True
    vault.destroy()


def test_started_server_never_leaks_token_to_surfaces(tmp_path: Path) -> None:
    server = _server(tmp_path)
    identity = server.start()
    token = server._vault.resolve()  # noqa: SLF001 — test inspects vault

    status = server.status()
    export = server.export_identity()
    ready = server.ready()
    logs = server.logs()
    argv = server.argv_safe_launch_spec()
    provider_env = server.provider_environment(
        {
            "PATH": "/usr/bin",
            "QUACK_TOKEN": token,
            "AUTH_TOKEN": token,
            "NORMAL": "ok",
        }
    )

    surfaces = [status, export, ready, list(logs), argv, provider_env, identity.to_dict()]
    for surface in surfaces:
        blob = json.dumps(surface, default=str) if not isinstance(surface, str) else surface
        if isinstance(surface, (list, tuple)):
            blob = " ".join(str(item) for item in surface)
        assert token not in blob
        assert token not in json.dumps(status)

    assert "QUACK_TOKEN" not in provider_env
    assert "AUTH_TOKEN" not in provider_env
    assert provider_env.get("NORMAL") == "ok"
    assert identity.secret_handle
    assert "token" not in identity.secret_handle or identity.secret_handle.startswith(
        "handle:"
    )
    # Status may include secret_handle but never raw token keys with values.
    assert status.get("secret_handle") == identity.secret_handle
    assert status.get("token") in (None, "secret_material")
    server.stop()


def test_sanitize_for_export_redacts_token_keys() -> None:
    payload = {"auth_token": "super-secret", "server_id": "server:1"}
    out = sanitize_for_export(payload)
    assert out["auth_token"] == "secret_material"
    assert out["server_id"] == "server:1"


def test_provider_safe_environment_strips_credential_names() -> None:
    env = provider_safe_environment(
        {
            "PATH": "/usr/bin",
            "QUACK_TOKEN": "abc",
            "MY_SECRET": "x",
            "HOME": "/tmp",
        }
    )
    assert env == {"PATH": "/usr/bin", "HOME": "/tmp"}


# ---------------------------------------------------------------------------
# Exclusive ownership / second owner / stale recovery
# ---------------------------------------------------------------------------


def test_second_owner_fails_closed(tmp_path: Path) -> None:
    first = _server(tmp_path)
    first.start()
    # Second server on the same DB must refuse while first is live.
    second = _server(tmp_path, liveness=OwnerLiveness.ALIVE, birth=_birth(pid=9999))
    with pytest.raises(QuackStateServerOwnershipError, match="second state-owner"):
        second.start()
    first.stop()


def test_stale_marker_recovery_allows_new_owner(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    marker_path = db.with_name(f".{db.name}.state-owner.json")
    lock_path = db.with_name(f".{db.name}.state-owner.lock")
    dead = _birth(pid=111, ticks=1, boot="old")
    marker = OwnerMarker(
        server_id="server:dead",
        process_birth=dead,
        database_path=str(db),
        started_at="2020-01-01T00:00:00Z",
        fence_token="fence-old",
        generation=1,
    )
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(marker.to_dict()), encoding="utf-8")

    result = reclaim_stale_owner_marker(
        marker_path=marker_path,
        lock_path=lock_path,
        liveness=lambda _b: OwnerLiveness.DEAD,
    )
    assert result["reclaimed"] is True
    assert not marker_path.exists()

    # New owner can start after reclaim.
    server = _server(tmp_path, liveness=OwnerLiveness.DEAD)
    identity = server.start()
    assert identity.server_id != "server:dead"
    server.stop()


def test_stale_marker_not_reclaimed_when_owner_alive(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    marker_path = db.with_name(f".{db.name}.state-owner.json")
    lock_path = db.with_name(f".{db.name}.state-owner.lock")
    live = _birth(pid=222, ticks=2, boot="live")
    marker = OwnerMarker(
        server_id="server:live",
        process_birth=live,
        database_path=str(db),
        started_at="2020-01-01T00:00:00Z",
        fence_token="fence-live",
        generation=1,
    )
    marker_path.write_text(json.dumps(marker.to_dict()), encoding="utf-8")
    result = reclaim_stale_owner_marker(
        marker_path=marker_path,
        lock_path=lock_path,
        liveness=lambda _b: OwnerLiveness.ALIVE,
    )
    assert result["reclaimed"] is False
    assert result["reason"] == "owner_alive"
    assert marker_path.exists()


@pytest.mark.parametrize(
    "receipt_filename",
    (
        "",
        ".",
        "..",
        "../receipt.json",
        "nested/receipt.json",
        "nested\\receipt.json",
        "/tmp/receipt.json",
        "quack-state-server.status.json",
        "quack-state-server.stop",
        "receipt\n.json",
        "receipt\x7f.json",
        "receipt\u202e.json",
        "x" * 256,
        None,
        Path("receipt.json"),
    ),
)
def test_stale_owner_recovery_rejects_unconfined_receipt_name_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_filename: object,
) -> None:
    db = tmp_path / "control.duckdb"
    state = tmp_path / "state"
    state.mkdir()
    database_bytes = b"not-a-database-but-must-remain-byte-identical"
    status_bytes = b'{"sentinel":"status-must-not-change"}\n'
    db.write_bytes(database_bytes)
    (state / "quack-state-server.status.json").write_bytes(status_bytes)
    entries_before = sorted(path.name for path in tmp_path.rglob("*"))

    def forbidden_database_open(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("invalid receipt name reached database mutation")

    monkeypatch.setattr(
        quack_state_server_module,
        "open_duckdb_connection",
        forbidden_database_open,
    )
    with pytest.raises(QuackStateServerControlError, match="confined basename"):
        recover_stale_state_server(
            database_path=db,
            state_dir=state,
            expected_store_id="control.duckdb",
            expected_generation=1,
            expected_database_uuid=_UUID,
            liveness=lambda _birth: OwnerLiveness.DEAD,
            receipt_filename=receipt_filename,  # type: ignore[arg-type]
        )

    assert db.read_bytes() == database_bytes
    assert (state / "quack-state-server.status.json").read_bytes() == status_bytes
    assert sorted(path.name for path in tmp_path.rglob("*")) == entries_before


def test_stale_owner_recovery_receipt_must_not_alias_database(
    tmp_path: Path,
) -> None:
    db = tmp_path / "control.duckdb"
    database_bytes = b"database-must-not-be-replaced-by-receipt"
    db.write_bytes(database_bytes)

    with pytest.raises(QuackStateServerControlError, match="confined basename"):
        recover_stale_state_server(
            database_path=db,
            state_dir=tmp_path,
            expected_store_id="control.duckdb",
            expected_generation=1,
            expected_database_uuid=_UUID,
            liveness=lambda _birth: OwnerLiveness.DEAD,
            receipt_filename=db.name,
        )

    assert db.read_bytes() == database_bytes


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_dead_ready_owner_recovery_settles_database_and_receipt(
    tmp_path: Path,
) -> None:
    db = tmp_path / "control.duckdb"
    state = tmp_path / "state"
    state.mkdir()
    install_control_plane_schema(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="recovery-test",
    )
    predecessor = build_server(
        database_path=db,
        state_dir=state,
        repository_id="repository:recovery-test",
        transport=FakeQuackTransport(),
        capability_probe=lambda **_kwargs: _compatible_report(),
        process_birth_factory=lambda: _birth(pid=123_455, ticks=76),
        owner_liveness_probe=lambda _birth: OwnerLiveness.DEAD,
    )
    predecessor_identity = predecessor.start()
    predecessor.stop()

    server = build_server(
        database_path=db,
        state_dir=state,
        repository_id="repository:recovery-test",
        transport=FakeQuackTransport(),
        capability_probe=lambda **_kwargs: _compatible_report(),
        process_birth_factory=lambda: _birth(pid=123_456, ticks=77),
        owner_liveness_probe=lambda _birth: OwnerLiveness.DEAD,
    )
    identity = server.start()
    marker = OwnerMarker.from_dict(
        json.loads(server.owner_marker_path().read_text(encoding="utf-8"))
    )
    assert identity.generation == predecessor_identity.generation + 1
    assert marker.generation == 1
    assert marker.generation != identity.generation

    # Rehearse process death: operating-system locks and the database handle
    # disappear, while the ready marker/status and canonical rows remain.
    assert server._connection is not None
    server._connection.close()
    server._connection = None
    assert server._owner is not None and server._owner._handle is not None
    server._owner._handle.close()
    server._owner._handle = None

    receipt = recover_stale_state_server(
        database_path=db,
        state_dir=state,
        expected_store_id=identity.store_id,
        expected_generation=identity.generation,
        expected_database_uuid=identity.database_uuid,
        liveness=lambda _birth: OwnerLiveness.DEAD,
        stopped_at="2026-08-29T07:00:00Z",
    )

    assert receipt["resulting_status"] == "stopped"
    assert receipt["owner_liveness"] == "dead"
    assert receipt["task_completion_authority"] is False
    assert str(receipt["receipt_cid"]).startswith("baguqeera")
    assert not server.owner_marker_path().exists()
    status = json.loads(server.status_path().read_text(encoding="utf-8"))
    assert status["lifecycle"] == "stopped"
    assert status["identity"]["status"] == "stopped"
    published = json.loads(
        (state / "quack-stale-owner-recovery-receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert published == receipt

    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        open_duckdb_connection,
    )

    connection = open_duckdb_connection(db, prefer_quack=False)
    try:
        rows = connection.execute(
            "SELECT status, stopped_at FROM state_servers WHERE server_id = ?",
            [identity.server_id],
        ).fetchall()
        epochs = connection.execute(
            "SELECT ended_at FROM server_epochs WHERE server_id = ?",
            [identity.server_id],
        ).fetchall()
    finally:
        connection.close()
    assert [tuple(row[index] for index in range(2)) for row in rows] == [
        ("stopped", "2026-08-29T07:00:00Z")
    ]
    assert [tuple(row[index] for index in range(1)) for row in epochs] == [
        ("2026-08-29T07:00:00Z",)
    ]


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_generation_specific_recovery_receipt_preserves_history_and_replays(
    tmp_path: Path,
) -> None:
    db = tmp_path / "control.duckdb"
    state = tmp_path / "state"
    state.mkdir()
    install_control_plane_schema(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="generation-recovery-test",
    )

    predecessor = build_server(
        database_path=db,
        state_dir=state,
        repository_id="repository:generation-recovery-test",
        transport=FakeQuackTransport(),
        capability_probe=lambda **_kwargs: _compatible_report(),
        process_birth_factory=lambda: _birth(pid=123_458, ticks=79),
        owner_liveness_probe=lambda _birth: OwnerLiveness.DEAD,
    )
    predecessor_identity = predecessor.start()
    assert predecessor._connection is not None
    predecessor._connection.close()
    predecessor._connection = None
    assert predecessor._owner is not None and predecessor._owner._handle is not None
    predecessor._owner._handle.close()
    predecessor._owner._handle = None

    historical_receipt = recover_stale_state_server(
        database_path=db,
        state_dir=state,
        expected_store_id=predecessor_identity.store_id,
        expected_generation=predecessor_identity.generation,
        expected_database_uuid=predecessor_identity.database_uuid,
        liveness=lambda _birth: OwnerLiveness.DEAD,
        stopped_at="2026-08-29T08:00:00Z",
    )
    historical_path = state / "quack-stale-owner-recovery-receipt.json"
    historical_bytes = historical_path.read_bytes()
    assert json.loads(historical_bytes) == historical_receipt

    successor = build_server(
        database_path=db,
        state_dir=state,
        repository_id="repository:generation-recovery-test",
        transport=FakeQuackTransport(),
        capability_probe=lambda **_kwargs: _compatible_report(),
        process_birth_factory=lambda: _birth(pid=123_459, ticks=80),
        owner_liveness_probe=lambda _birth: OwnerLiveness.DEAD,
    )
    successor_identity = successor.start()
    assert successor_identity.generation == predecessor_identity.generation + 1
    assert successor._connection is not None
    successor._connection.close()
    successor._connection = None
    assert successor._owner is not None and successor._owner._handle is not None
    successor._owner._handle.close()
    successor._owner._handle = None

    receipt_filename = (
        "quack-stale-owner-recovery-receipt."
        f"generation-{successor_identity.generation}.json"
    )
    receipt_path = state / receipt_filename
    receipt = recover_stale_state_server(
        database_path=db,
        state_dir=state,
        expected_store_id=successor_identity.store_id,
        expected_generation=successor_identity.generation,
        expected_database_uuid=successor_identity.database_uuid,
        liveness=lambda _birth: OwnerLiveness.DEAD,
        stopped_at="2026-08-29T08:30:00Z",
        receipt_filename=receipt_filename,
    )
    receipt_bytes = receipt_path.read_bytes()
    assert json.loads(receipt_bytes) == receipt
    assert historical_path.read_bytes() == historical_bytes

    replayed = recover_stale_state_server(
        database_path=db,
        state_dir=state,
        expected_store_id=successor_identity.store_id,
        expected_generation=successor_identity.generation,
        expected_database_uuid=successor_identity.database_uuid,
        liveness=lambda _birth: OwnerLiveness.DEAD,
        stopped_at="2026-08-29T08:30:00Z",
        receipt_filename=receipt_filename,
    )
    assert replayed == receipt
    assert receipt_path.read_bytes() == receipt_bytes
    assert historical_path.read_bytes() == historical_bytes


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
@pytest.mark.parametrize("failure_boundary", ("status", "receipt"))
def test_dead_owner_recovery_replays_after_publication_crash_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_boundary: str,
) -> None:
    db = tmp_path / "control.duckdb"
    state = tmp_path / "state"
    state.mkdir()
    install_control_plane_schema(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="recovery-replay-test",
    )
    server = build_server(
        database_path=db,
        state_dir=state,
        repository_id="repository:recovery-replay-test",
        transport=FakeQuackTransport(),
        capability_probe=lambda **_kwargs: _compatible_report(),
        process_birth_factory=lambda: _birth(pid=123_457, ticks=78),
        owner_liveness_probe=lambda _birth: OwnerLiveness.DEAD,
    )
    identity = server.start()
    assert server._connection is not None
    initial_revision = server._connection.execute(
        "SELECT revision FROM state_servers WHERE server_id = ?",
        [identity.server_id],
    ).fetchone()[0]
    server._connection.close()
    server._connection = None
    assert server._owner is not None and server._owner._handle is not None
    server._owner._handle.close()
    server._owner._handle = None

    status_path = server.status_path()
    receipt_path = state / "quack-stale-owner-recovery-receipt.json"
    original_atomic_write_json = quack_state_server_module._atomic_write_json
    failed = {"value": False}

    def fail_once(path: Path, value: object, *, mode: int = 0o600) -> None:
        target = status_path if failure_boundary == "status" else receipt_path
        if Path(path) == target and not failed["value"]:
            failed["value"] = True
            raise OSError(f"injected {failure_boundary} publication failure")
        original_atomic_write_json(Path(path), value, mode=mode)

    monkeypatch.setattr(
        quack_state_server_module,
        "_atomic_write_json",
        fail_once,
    )
    with pytest.raises(OSError, match="injected"):
        recover_stale_state_server(
            database_path=db,
            state_dir=state,
            expected_store_id=identity.store_id,
            expected_generation=identity.generation,
            expected_database_uuid=identity.database_uuid,
            liveness=lambda _birth: OwnerLiveness.DEAD,
            stopped_at="2026-08-29T07:30:00Z",
        )

    receipt = recover_stale_state_server(
        database_path=db,
        state_dir=state,
        expected_store_id=identity.store_id,
        expected_generation=identity.generation,
        expected_database_uuid=identity.database_uuid,
        liveness=lambda _birth: OwnerLiveness.DEAD,
        stopped_at="2026-08-29T07:30:00Z",
    )
    assert failed["value"] is True
    assert receipt["replay_safe"] is True
    assert not server.owner_marker_path().exists()
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt

    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        open_duckdb_connection,
    )

    connection = open_duckdb_connection(db, prefer_quack=False)
    try:
        settled = connection.execute(
            "SELECT status, stopped_at, revision FROM state_servers "
            "WHERE server_id = ?",
            [identity.server_id],
        ).fetchall()
    finally:
        connection.close()
    assert [tuple(row[index] for index in range(3)) for row in settled] == [
        ("stopped", "2026-08-29T07:30:00Z", initial_revision + 1)
    ]


def test_exclusive_owner_lease_fence_mismatch_on_release(tmp_path: Path) -> None:
    lock_path = tmp_path / "owner.lock"
    marker_path = tmp_path / "owner.json"
    lease = ExclusiveOwnerLease(
        lock_path=lock_path,
        marker_path=marker_path,
        liveness=lambda _b: OwnerLiveness.DEAD,
    )
    lease.acquire(
        server_id="server:1",
        process_birth=_birth(),
        database_path=tmp_path / "control.duckdb",
    )
    with pytest.raises(Exception, match="fence"):
        lease.release(fence_token="wrong-fence")
    lease.release()


# ---------------------------------------------------------------------------
# Ready / identity / migration / lifecycle
# ---------------------------------------------------------------------------


def test_start_ready_checkpoint_stop_lifecycle(tmp_path: Path) -> None:
    transport = FakeQuackTransport()
    server = _server(tmp_path, transport=transport, port=0)
    identity = server.start()

    assert server.lifecycle is ServerLifecycle.READY
    assert identity.listen_uri.startswith("quack:127.0.0.1:")
    assert identity.schema_fingerprint == _DIGEST
    assert identity.database_uuid == _UUID
    assert identity.generation >= 1
    assert identity.process_birth_id.startswith("birth:")
    assert transport.started is True

    ready = server.ready()
    assert ready["ready"] is True
    assert ready["store_id"] == identity.store_id
    assert ready["generation"] == identity.generation
    assert ready["schema_fingerprint"] == identity.schema_fingerprint
    assert ready["server_id"] == identity.server_id

    checkpoint = server.checkpoint()
    assert checkpoint["checkpointed"] is True

    stop = server.stop()
    assert stop["stopped"] is True
    assert server.lifecycle is ServerLifecycle.STOPPED
    assert transport.stopped is True
    assert not server.owner_marker_path().exists()


@pytest.mark.parametrize(
    ("max_generation", "expected_overrides", "mismatched_field"),
    (
        (43, {"expected_generation": 43}, "generation"),
        (
            0,
            {"expected_database_uuid": "123e4567-e89b-12d3-a456-426614174001"},
            "database_uuid",
        ),
        (0, {"expected_store_id": "other-control.duckdb"}, "store_id"),
        (
            0,
            {"expected_listen_uri": "quack:127.0.0.1:4343"},
            "listen_uri",
        ),
    ),
    ids=("unexpected-generation-44", "database-uuid", "store-id", "listen-uri"),
)
def test_expected_startup_binding_fails_before_credentials_transport_or_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_generation: int,
    expected_overrides: dict[str, Any],
    mismatched_field: str,
) -> None:
    connection = FakeConnection(max_generation=max_generation)
    transport = FakeQuackTransport()
    expected = {
        "expected_generation": max_generation + 1,
        "expected_database_uuid": _UUID,
        "expected_store_id": "control.duckdb",
        "expected_listen_uri": "quack:127.0.0.1:4242",
        **expected_overrides,
    }
    server = build_server(
        database_path=tmp_path / "control.duckdb",
        state_dir=tmp_path / "state",
        port=4242,
        transport=transport,
        capability_probe=lambda **_kwargs: _compatible_report(),
        migrate=lambda _path: _migration_report(),
        connection_factory=lambda _path: connection,
        process_birth_factory=lambda: _birth(),
        owner_liveness_probe=lambda _birth: OwnerLiveness.DEAD,
        **expected,
    )
    assert server._vault is not None  # noqa: SLF001 -- pre-publication boundary
    mint_calls: list[tuple[str, int]] = []

    def record_mint(*, secret_handle: str, generation: int) -> None:
        mint_calls.append((secret_handle, generation))

    monkeypatch.setattr(server._vault, "mint", record_mint)  # noqa: SLF001

    with pytest.raises(QuackStateServerControlError, match=mismatched_field):
        server.start()

    assert server.lifecycle is ServerLifecycle.FAILED
    assert server.identity is None
    assert mint_calls == []
    assert transport.started is False
    assert transport.start_calls == []
    assert not any(
        statement.upper().startswith(("INSERT ", "UPDATE "))
        for statement in connection.statements
    )
    assert connection.closed is True
    assert not server.owner_marker_path().exists()
    assert list((tmp_path / "state").glob("*.quack-token")) == []


def test_ready_requires_live_query(tmp_path: Path) -> None:
    transport = FakeQuackTransport(fail_live_query=True)
    server = _server(tmp_path, transport=transport)
    server.start()
    with pytest.raises(QuackStateServerReadyError, match="live query"):
        server.ready()
    assert server.is_ready() is False
    # Clear failure for clean stop path
    transport.fail_live_query = False
    server.stop()


def test_ready_requires_matching_identities(tmp_path: Path) -> None:
    class DriftTransport(FakeQuackTransport):
        def live_query(self, connection, *, identity, token):  # type: ignore[no-untyped-def]
            del connection, token
            return {
                "live": True,
                "server_id": identity.server_id,
                "store_id": "wrong-store",
                "database_uuid": identity.database_uuid,
                "schema_revision": identity.schema_revision,
                "schema_fingerprint": identity.schema_fingerprint,
                "generation": identity.generation,
                "process_birth_id": identity.process_birth_id,
            }

    server = _server(tmp_path, transport=DriftTransport())
    server.start()
    with pytest.raises(QuackStateServerReadyError, match="do not match"):
        server.ready()
    server.stop()


def test_ready_requires_complete_live_identity_fields(tmp_path: Path) -> None:
    class IncompleteTransport(FakeQuackTransport):
        def live_query(self, connection, *, identity, token):  # type: ignore[no-untyped-def]
            del connection, token
            return {
                "live": True,
                "server_id": identity.server_id,
                # store_id intentionally omitted — must not fall back silently
                "database_uuid": identity.database_uuid,
                "schema_revision": identity.schema_revision,
                "schema_fingerprint": identity.schema_fingerprint,
                "generation": identity.generation,
                "process_birth_id": identity.process_birth_id,
            }

    server = _server(tmp_path, transport=IncompleteTransport())
    server.start()
    with pytest.raises(QuackStateServerReadyError, match="missing identity fields"):
        server.ready()
    server.stop()


def test_migration_required_before_ready(tmp_path: Path) -> None:
    server = _server(tmp_path, schema_version="0")
    with pytest.raises(Exception, match="migrated before ready|schema must be migrated"):
        server.start()


def test_capability_admission_fail_closed(tmp_path: Path) -> None:
    bad = _compatible_report(status=QuackCapabilityStatus.UNAVAILABLE)
    server = _server(tmp_path, capability=bad)
    with pytest.raises(QuackStateServerCapabilityError):
        server.start()


def test_whoami_process_birth_published(tmp_path: Path) -> None:
    birth = _birth(pid=7777, ticks=12345, boot="boot-xyz")
    server = _server(tmp_path, birth=birth)
    identity = server.start()
    assert identity.process_birth.pid == 7777
    assert identity.process_birth.start_time_ticks == 12345
    assert identity.process_birth.boot_id == "boot-xyz"
    status = server.status()
    assert status["identity"]["process_birth"]["pid"] == 7777
    # whoami-style export
    export = server.export_identity()
    assert export["identity"]["process_birth_id"] == identity.process_birth_id
    server.stop()


def test_graceful_stop_uses_fence_control_path(tmp_path: Path) -> None:
    server = _server(tmp_path)
    identity = server.start()
    request = server.request_stop()
    assert request["requested"] is True
    assert Path(request["control_path"]).is_file()
    control = json.loads(Path(request["control_path"]).read_text(encoding="utf-8"))
    assert control["server_id"] == identity.server_id
    assert control["fence_token"]
    # fence is ownership fence, not quack auth token
    token = server._vault.resolve()  # noqa: SLF001
    assert control["fence_token"] != token
    result = server.stop()
    assert result["stopped"] is True


def test_listen_uri_format() -> None:
    assert listen_uri("127.0.0.1", 4242) == "quack:127.0.0.1:4242"


# ---------------------------------------------------------------------------
# Ops CLI argv policy
# ---------------------------------------------------------------------------


def test_ops_script_rejects_token_argv() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(OPS_SCRIPT),
            "--token",
            "raw-secret",
            "start",
            "--database",
            "/tmp/x.duckdb",
            "--state-dir",
            "/tmp/state",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "refusing argv credential flag" in (proc.stderr + proc.stdout)


def test_ops_script_help_is_cold() -> None:
    proc = subprocess.run(
        [sys.executable, str(OPS_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "secret-handle" in proc.stdout
    assert "Never accepts raw auth tokens" in proc.stdout
    # Help must not advertise a raw-token flag as an option.
    assert "--token " not in proc.stdout
    assert "--token\n" not in proc.stdout


def test_ops_module_import_is_cold() -> None:
    # Importing the ops facade must not open a database.
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util, sys; "
                f"spec = importlib.util.spec_from_file_location('qss', {str(OPS_SCRIPT)!r}); "
                "mod = importlib.util.module_from_spec(spec); "
                "spec.loader.exec_module(mod); "
                "print('ok')"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0
    assert "ok" in proc.stdout


# ---------------------------------------------------------------------------
# Optional integration with real DuckDB (migration path)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required for integration path")
def test_real_duckdb_migration_then_fake_transport_ready(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    state = tmp_path / "state"
    state.mkdir()
    install_control_plane_schema(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="test-owner",
    )

    # Use real connection factory via duckdb_state but fake transport/capability.
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        open_duckdb_connection,
    )

    server = build_server(
        database_path=db,
        state_dir=state,
        transport=FakeQuackTransport(),
        capability_probe=lambda **_k: _compatible_report(),
        # migrate is no-op / real install already done; still call real installer
        # which is replay-safe.
        process_birth_factory=lambda: _birth(pid=os.getpid()),
        owner_liveness_probe=lambda _b: OwnerLiveness.DEAD,
    )
    # Override connection to keep open across ready.
    # Default migrate+connection_factory use real duckdb.
    identity = server.start()
    assert identity.database_uuid
    assert identity.schema_revision >= 1
    assert identity.schema_fingerprint.startswith("sha256:")
    ready = server.ready()
    assert ready["ready"] is True
    export = server.export_identity()
    assert export["identity"]["server_id"] == identity.server_id
    server.checkpoint()
    server.stop()
    # Connection closed; marker gone.
    assert not server.owner_marker_path().exists()


def test_config_rejects_raw_token_as_secret_handle(tmp_path: Path) -> None:
    with pytest.raises(QuackStateServerTokenError):
        QuackStateServerConfig(
            database_path=tmp_path / "control.duckdb",
            state_dir=tmp_path / "state",
            secret_handle="raw-not-a-handle",
        )
