"""Focused tests for exact configured-board DuckDB extension projections."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from types import ModuleType

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_extension_projection import (
    CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV,
    CONFIGURED_BOARD_EXTENSION_PIN_SCHEMA,
    CONFIGURED_BOARD_EXTENSION_SET_PIN_ENV,
    CONFIGURED_BOARD_EXTENSION_SET_PIN_SCHEMA,
    ConfiguredBoardExtensionProjectionError,
    build_configured_board_extension_set_pin,
    configured_board_extension_set_id,
    inspect_configured_board_extension_sources,
    parse_configured_board_extension_pin,
    parse_configured_board_extension_set_pin,
    parse_configured_board_extension_set_pin_json,
    project_configured_board_extension_home,
    project_configured_board_extension_set_home,
    seal_configured_board_extension_set_home,
    verify_configured_board_extension_home,
    verify_configured_board_extension_set_home,
)

PAYLOAD = b"synthetic-quack-extension-v1\x00\x01"
INFO = b'{"extension":"quack","synthetic":true}\n'


def _write_sources(
    root: Path,
    *,
    payload: bytes = PAYLOAD,
    info: bytes = INFO,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    extension = root / "quack.duckdb_extension"
    metadata = root / "quack.duckdb_extension.info"
    extension.write_bytes(payload)
    metadata.write_bytes(info)
    return extension, metadata


def _inspect(extension: Path, metadata: Path):
    return inspect_configured_board_extension_sources(
        extension,
        metadata,
        name="quack",
        engine_version="v1.5.5",
        platform="linux_arm64",
    )


def _prepare_parent(path: Path) -> Path:
    path.mkdir(mode=0o700)
    os.chmod(path, 0o700)
    return path


def _restore_tree_permissions(root: Path) -> None:
    """Leave pytest able to remove a projection whose custody is read-only."""

    if not root.exists() or root.is_symlink():
        return
    os.chmod(root, 0o700)
    for current, directories, files in os.walk(root):
        current_path = Path(current)
        os.chmod(current_path, 0o700)
        for name in directories:
            candidate = current_path / name
            if not candidate.is_symlink():
                os.chmod(candidate, 0o700)
        for name in files:
            candidate = current_path / name
            if not candidate.is_symlink():
                os.chmod(candidate, 0o600)


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def _exact_set(tmp_path: Path):
    sources: dict[str, tuple[Path, Path]] = {}
    pins = {}
    for name, payload in (("httpfs", b"httpfs-v1"), ("quack", PAYLOAD)):
        root = tmp_path / "sources" / name
        root.mkdir(parents=True)
        extension = root / f"{name}.duckdb_extension"
        info = root / f"{name}.duckdb_extension.info"
        extension.write_bytes(payload)
        info.write_bytes(f"metadata:{name}".encode())
        sources[name] = (extension, info)
        pins[name] = inspect_configured_board_extension_sources(
            extension,
            info,
            name=name,
            engine_version="v1.5.5",
            platform="linux_arm64",
        )
    set_pin = build_configured_board_extension_set_pin(
        pins,
        versions={"httpfs": "httpfs-version", "quack": "quack-version"},
    )
    parent = _prepare_parent(tmp_path / "launch-set")
    home = project_configured_board_extension_set_home(
        pins,
        sources=sources,
        parent=parent,
    )
    return sources, pins, set_pin, home


def test_pin_is_path_independent_deterministic_and_round_trips(
    tmp_path: Path,
) -> None:
    first_extension, first_info = _write_sources(tmp_path / "first")
    second_extension, second_info = _write_sources(tmp_path / "second")

    first = _inspect(first_extension, first_info)
    second = _inspect(second_extension, second_info)

    assert first == second
    assert first.schema == CONFIGURED_BOARD_EXTENSION_PIN_SCHEMA
    assert str(tmp_path) not in first.to_json()
    assert parse_configured_board_extension_pin(first.as_dict()) == first
    assert first.to_json() == json.dumps(
        first.as_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    identity_body = first.as_dict()
    identity_body.pop("projection_id")
    expected_identity = "sha256:" + hashlib.sha256(
        json.dumps(
            identity_body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert first.projection_id == expected_identity


def test_pin_parser_rejects_unknown_missing_and_textual_duplicate_fields(
    tmp_path: Path,
) -> None:
    extension, metadata = _write_sources(tmp_path / "sources")
    pin = _inspect(extension, metadata)

    unknown = {**pin.as_dict(), "unreviewed": True}
    missing = pin.as_dict()
    missing.pop("info_sha256")
    wrong_identity = {**pin.as_dict(), "projection_id": "sha256:" + "0" * 64}
    for candidate in (unknown, missing, wrong_identity):
        with pytest.raises(ConfiguredBoardExtensionProjectionError):
            parse_configured_board_extension_pin(candidate)

    # This API accepts exact dicts only, so textual duplicate JSON keys never
    # reach a lossy JSON decoder or acquire a pin interpretation.
    duplicate_json = '{"name":"quack","name":"foreign"}'
    with pytest.raises(ConfiguredBoardExtensionProjectionError):
        parse_configured_board_extension_pin(duplicate_json)


def test_projection_has_exact_files_and_read_only_custody(tmp_path: Path) -> None:
    extension, metadata = _write_sources(tmp_path / "sources")
    pin = _inspect(extension, metadata)
    parent = _prepare_parent(tmp_path / "launch")

    home = project_configured_board_extension_home(
        pin,
        extension_path=extension,
        info_path=metadata,
        parent=parent,
    )
    try:
        assert home.is_absolute()
        assert home.name == pin.projection_id.removeprefix("sha256:")
        assert verify_configured_board_extension_home(pin, home) == home
        extension_copy = home / pin.relative_directory / pin.extension_filename
        info_copy = extension_copy.with_name(f"{extension_copy.name}.info")
        assert extension_copy.read_bytes() == PAYLOAD
        assert info_copy.read_bytes() == INFO
        assert _mode(extension_copy) == _mode(info_copy) == 0o400

        expected_directories = {
            home,
            home / ".duckdb",
            home / ".duckdb/extensions",
            home / ".duckdb/extensions" / pin.engine_version,
            home / pin.relative_directory,
            home / ".python-user-base",
            home / ".cache",
        }
        assert {
            item for item in (home, *home.rglob("*")) if item.is_dir()
        } == expected_directories
        assert all(
            _mode(item) == 0o500
            for item in expected_directories - {home / ".cache"}
        )
        assert _mode(home / ".cache") == 0o700
    finally:
        _restore_tree_permissions(home)


@pytest.mark.parametrize("mismatch", ("payload", "info"))
def test_projection_rejects_sources_that_differ_from_pin(
    tmp_path: Path,
    mismatch: str,
) -> None:
    accepted_extension, accepted_info = _write_sources(tmp_path / "accepted")
    pin = _inspect(accepted_extension, accepted_info)
    candidate_extension, candidate_info = _write_sources(
        tmp_path / "candidate",
        payload=b"wrong-extension" if mismatch == "payload" else PAYLOAD,
        info=b"wrong-info" if mismatch == "info" else INFO,
    )
    parent = _prepare_parent(tmp_path / "launch")

    with pytest.raises(
        ConfiguredBoardExtensionProjectionError,
        match="differs from its accepted pin",
    ):
        project_configured_board_extension_home(
            pin,
            extension_path=candidate_extension,
            info_path=candidate_info,
            parent=parent,
        )


def test_source_inspection_rejects_symlinks_and_nonregular_nodes(
    tmp_path: Path,
) -> None:
    extension, metadata = _write_sources(tmp_path / "real")
    linked_extension = tmp_path / "linked.duckdb_extension"
    linked_extension.symlink_to(extension)
    nonregular_info = tmp_path / "info-directory"
    nonregular_info.mkdir()

    with pytest.raises(ConfiguredBoardExtensionProjectionError):
        _inspect(linked_extension, metadata)
    with pytest.raises(ConfiguredBoardExtensionProjectionError):
        _inspect(extension, nonregular_info)


def test_projected_byte_tamper_fails_verification_and_reprojection(
    tmp_path: Path,
) -> None:
    extension, metadata = _write_sources(tmp_path / "sources")
    pin = _inspect(extension, metadata)
    parent = _prepare_parent(tmp_path / "launch")
    home = project_configured_board_extension_home(
        pin,
        extension_path=extension,
        info_path=metadata,
        parent=parent,
    )
    projected = home / pin.relative_directory / pin.extension_filename
    try:
        os.chmod(projected, 0o600)
        projected.write_bytes(b"tampered-projection")
        with pytest.raises(
            ConfiguredBoardExtensionProjectionError,
            match="bytes drifted",
        ):
            verify_configured_board_extension_home(pin, home)
        with pytest.raises(ConfiguredBoardExtensionProjectionError):
            project_configured_board_extension_home(
                pin,
                extension_path=extension,
                info_path=metadata,
                parent=parent,
            )
    finally:
        _restore_tree_permissions(home)


def test_projection_is_idempotent_without_republishing(tmp_path: Path) -> None:
    extension, metadata = _write_sources(tmp_path / "sources")
    pin = _inspect(extension, metadata)
    parent = _prepare_parent(tmp_path / "launch")
    first = project_configured_board_extension_home(
        pin,
        extension_path=extension,
        info_path=metadata,
        parent=parent,
    )
    projected = first / pin.relative_directory / pin.extension_filename
    first_home_inode = os.lstat(first).st_ino
    first_payload_inode = os.lstat(projected).st_ino
    try:
        second = project_configured_board_extension_home(
            pin,
            extension_path=extension,
            info_path=metadata,
            parent=parent,
        )
        assert second == first
        assert os.lstat(second).st_ino == first_home_inode
        assert os.lstat(projected).st_ino == first_payload_inode
        assert verify_configured_board_extension_home(pin, second) == second
        homes = parent / "qualification-homes"
        assert [item.name for item in homes.iterdir()] == [
            pin.projection_id.removeprefix("sha256:")
        ]
    finally:
        _restore_tree_permissions(first)


def test_exact_httpfs_quack_set_projection_is_deterministic(tmp_path: Path) -> None:
    sources: dict[str, tuple[Path, Path]] = {}
    pins = {}
    for name, payload in (("httpfs", b"httpfs-v1"), ("quack", PAYLOAD)):
        root = tmp_path / "sources" / name
        root.mkdir(parents=True)
        extension = root / f"{name}.duckdb_extension"
        info = root / f"{name}.duckdb_extension.info"
        extension.write_bytes(payload)
        info.write_bytes(f"metadata:{name}".encode())
        sources[name] = (extension, info)
        pins[name] = inspect_configured_board_extension_sources(
            extension,
            info,
            name=name,
            engine_version="v1.5.5",
            platform="linux_arm64",
        )
    parent = _prepare_parent(tmp_path / "launch-set")
    home = project_configured_board_extension_set_home(
        pins,
        sources=sources,
        parent=parent,
    )
    try:
        assert home.name == configured_board_extension_set_id(pins).removeprefix(
            "sha256:"
        )
        assert verify_configured_board_extension_set_home(pins, home) == home
        assert project_configured_board_extension_set_home(
            pins,
            sources=sources,
            parent=parent,
        ) == home
        directory = home / pins["quack"].relative_directory
        assert sorted(item.name for item in directory.iterdir()) == [
            "httpfs.duckdb_extension",
            "httpfs.duckdb_extension.info",
            "quack.duckdb_extension",
            "quack.duckdb_extension.info",
        ]
    finally:
        _restore_tree_permissions(home)


def test_exact_set_pin_json_is_closed_canonical_and_round_trips(
    tmp_path: Path,
) -> None:
    _sources, _pins, set_pin, home = _exact_set(tmp_path)
    try:
        assert set_pin.schema == CONFIGURED_BOARD_EXTENSION_SET_PIN_SCHEMA
        assert tuple(member.pin.name for member in set_pin.members) == (
            "httpfs",
            "quack",
        )
        assert parse_configured_board_extension_set_pin(set_pin.as_dict()) == set_pin
        assert parse_configured_board_extension_set_pin_json(set_pin.to_json()) == set_pin

        unknown = {**set_pin.as_dict(), "unreviewed": True}
        with pytest.raises(ConfiguredBoardExtensionProjectionError):
            parse_configured_board_extension_set_pin(unknown)
        duplicate = set_pin.to_json().replace(
            '"schema":',
            '"schema":"duplicate","schema":',
            1,
        )
        with pytest.raises(
            ConfiguredBoardExtensionProjectionError,
            match="duplicate keys",
        ):
            parse_configured_board_extension_set_pin_json(duplicate)
        with pytest.raises(
            ConfiguredBoardExtensionProjectionError,
            match="not canonical",
        ):
            parse_configured_board_extension_set_pin_json(
                json.dumps(set_pin.as_dict())
            )
    finally:
        _restore_tree_permissions(home)


def test_sealed_set_uses_exact_memfds_and_detects_load_path_tamper(
    tmp_path: Path,
) -> None:
    _sources, _pins, set_pin, home = _exact_set(tmp_path)
    sealed = seal_configured_board_extension_set_home(set_pin, home)
    try:
        sealed.verify()
        assert set(sealed.install_paths) == {"httpfs", "quack"}
        for path in sealed.install_paths.values():
            assert path.is_symlink()
            assert os.readlink(path).startswith("/proc/self/fd/")

        quack_info = sealed.install_paths["quack"].with_name(
            "quack.duckdb_extension.info"
        )
        with pytest.raises(
            ConfiguredBoardExtensionProjectionError,
            match="bytes drifted|custody changed",
        ):
            with sealed.load_guard():
                os.chmod(quack_info, 0o600)
                quack_info.write_bytes(b"tampered-during-load")
    finally:
        sealed.close()
        _restore_tree_permissions(home)
    assert not sealed.parent.exists()


def test_quack_client_loads_only_the_exact_locked_set_and_releases_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state

    _sources, _pins, set_pin, home = _exact_set(tmp_path)
    observed: dict[str, object] = {"loads": []}

    class Result:
        def __init__(self, rows=(), row=None) -> None:
            self._rows = list(rows)
            self._row = row

        def fetchall(self):
            return list(self._rows)

        def fetchone(self):
            return self._row

    class Connection:
        def __init__(self, config: dict[str, str]) -> None:
            self.config = config
            self.closed = False

        def execute(self, sql: str):
            if sql.startswith("LOAD "):
                observed["loads"].append(sql)
                return Result()
            if "FROM duckdb_extensions()" in sql:
                directory = Path(self.config["extension_directory"])
                install_root = directory / "v1.5.5/linux_arm64"
                return Result(
                    rows=[
                        (
                            name,
                            str(install_root / f"{name}.duckdb_extension"),
                            set_pin.versions[name],
                            True,
                            True,
                        )
                        for name in ("httpfs", "quack")
                    ]
                )
            if "current_setting('autoinstall_known_extensions')" in sql:
                return Result(row=(False, False, False, False, True))
            if sql.startswith("SET "):
                return Result()
            if sql.startswith(("ATTACH ", "USE ")):
                return Result()
            if "SELECT count(*)" in sql:
                return Result(rows=[(1,)])
            raise AssertionError(sql)

        def close(self) -> None:
            self.closed = True

    fake_duckdb = ModuleType("duckdb")

    def connect(database: str, *, config: dict[str, str]):
        assert database == ":memory:"
        observed["config"] = dict(config)
        connection = Connection(config)
        observed["connection"] = connection
        return connection

    fake_duckdb.connect = connect
    monkeypatch.setitem(sys.modules, "duckdb", fake_duckdb)
    monkeypatch.setenv(
        CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV,
        str(home / ".duckdb/extensions"),
    )
    monkeypatch.setenv(CONFIGURED_BOARD_EXTENSION_SET_PIN_ENV, set_pin.to_json())
    monkeypatch.delenv(duckdb_state.QUACK_MUTATION_BINDING_ENV, raising=False)

    try:
        wrapped = duckdb_state.open_quack_transport_connection(
            "quack:127.0.0.1:45123"
        )
        config = observed["config"]
        assert isinstance(config, dict)
        assert config["autoinstall_known_extensions"] == "false"
        assert config["autoload_known_extensions"] == "false"
        assert config["enable_external_access"] == "true"
        assert config["allow_unsigned_extensions"] == "false"
        assert "lock_configuration" not in config
        sealed_directory = Path(config["extension_directory"])
        sealed_parent = sealed_directory.parents[2]
        assert sealed_parent.name.startswith("configured-board-sealed-extensions-")
        assert observed["loads"] == ["LOAD httpfs", "LOAD quack"]
        wrapped.close()
        assert not sealed_parent.exists()
    finally:
        _restore_tree_permissions(home)


def test_quack_client_fails_closed_for_incomplete_or_mismatched_set_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state

    _sources, _pins, set_pin, home = _exact_set(tmp_path)
    calls = {"connect": 0}
    fake_duckdb = ModuleType("duckdb")

    def connect(*_args, **_kwargs):
        calls["connect"] += 1
        raise AssertionError("DuckDB must not open for an invalid set contract")

    fake_duckdb.connect = connect
    monkeypatch.setitem(sys.modules, "duckdb", fake_duckdb)
    monkeypatch.setenv(
        CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV,
        str(home / ".duckdb/extensions"),
    )
    monkeypatch.delenv(CONFIGURED_BOARD_EXTENSION_SET_PIN_ENV, raising=False)
    try:
        with pytest.raises(
            duckdb_state.DuckDBConnectionPolicyError,
            match="must be provided together",
        ):
            duckdb_state.open_quack_transport_connection(
                "quack:127.0.0.1:45123"
            )

        wrong = set_pin.as_dict()
        wrong["members"][1]["extension_version"] = "wrong-version"
        monkeypatch.setenv(
            CONFIGURED_BOARD_EXTENSION_SET_PIN_ENV,
            json.dumps(
                wrong,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        with pytest.raises(
            duckdb_state.DuckDBConnectionPolicyError,
            match="exact extension set is invalid",
        ):
            duckdb_state.open_quack_transport_connection(
                "quack:127.0.0.1:45123"
            )
        assert calls["connect"] == 0
    finally:
        _restore_tree_permissions(home)
