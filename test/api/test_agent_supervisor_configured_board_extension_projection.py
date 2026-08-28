"""Focused tests for exact configured-board DuckDB extension projections."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_extension_projection import (
    CONFIGURED_BOARD_EXTENSION_PIN_SCHEMA,
    ConfiguredBoardExtensionProjectionError,
    inspect_configured_board_extension_sources,
    parse_configured_board_extension_pin,
    project_configured_board_extension_home,
    verify_configured_board_extension_home,
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
