"""Exact, load-only DuckDB extension projection for configured boards.

DuckDB requires an extension filename ending in ``.duckdb_extension`` and
cannot load the same sealed bytes directly from an anonymous memfd path.  The
operator launcher therefore copies an externally accepted extension and its
metadata into a private, read-only DuckDB ``HOME``.  Every byte is rehashed
before and after publication.  This adapter grants no acceptance authority;
the caller must bind the pin to a protected dependency seal.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

CONFIGURED_BOARD_EXTENSION_PIN_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor."
    "configured-board-duckdb-extension-pin@1"
)
CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV: Final = (
    "IPFS_ACCELERATE_AGENT_DUCKDB_EXTENSION_DIRECTORY"
)
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}")
_PIN_FIELDS: Final = frozenset(
    {
        "schema",
        "name",
        "engine_version",
        "platform",
        "payload_sha256",
        "payload_size",
        "info_sha256",
        "info_size",
        "projection_id",
    }
)
_MAX_PAYLOAD_BYTES: Final = 64 * 1024 * 1024
_MAX_INFO_BYTES: Final = 64 * 1024


class ConfiguredBoardExtensionProjectionError(ValueError):
    """A configured-board extension pin or projection failed closed."""


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _projection_id(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("projection_id", None)
    return "sha256:" + hashlib.sha256(_canonical_json(payload)).hexdigest()


def _text(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
        or any(character in value for character in "\x00\r\n")
    ):
        raise ConfiguredBoardExtensionProjectionError(f"{field} is invalid")
    return value


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ConfiguredBoardExtensionProjectionError(f"{field} is invalid")
    return value


def _size(value: object, field: str, *, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 < value <= maximum
    ):
        raise ConfiguredBoardExtensionProjectionError(f"{field} is invalid")
    return value


@dataclass(frozen=True, slots=True)
class ConfiguredBoardExtensionPin:
    """Path-independent accepted bytes for one load-only DuckDB extension."""

    schema: str
    name: str
    engine_version: str
    platform: str
    payload_sha256: str
    payload_size: int
    info_sha256: str
    info_size: int
    projection_id: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "name": self.name,
            "engine_version": self.engine_version,
            "platform": self.platform,
            "payload_sha256": self.payload_sha256,
            "payload_size": self.payload_size,
            "info_sha256": self.info_sha256,
            "info_size": self.info_size,
            "projection_id": self.projection_id,
        }

    def to_json(self) -> str:
        return _canonical_json(self.as_dict()).decode("utf-8")

    @property
    def relative_directory(self) -> Path:
        return Path(".duckdb/extensions") / self.engine_version / self.platform

    @property
    def extension_filename(self) -> str:
        return f"{self.name}.duckdb_extension"


def parse_configured_board_extension_pin(
    value: object,
) -> ConfiguredBoardExtensionPin:
    """Strictly parse and recompute a path-independent extension pin."""

    if type(value) is not dict or set(value) != _PIN_FIELDS:
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension pin fields are noncanonical"
        )
    pin = ConfiguredBoardExtensionPin(
        schema=_text(value.get("schema"), "schema"),
        name=_text(value.get("name"), "name"),
        engine_version=_text(value.get("engine_version"), "engine_version"),
        platform=_text(value.get("platform"), "platform"),
        payload_sha256=_digest(value.get("payload_sha256"), "payload_sha256"),
        payload_size=_size(
            value.get("payload_size"),
            "payload_size",
            maximum=_MAX_PAYLOAD_BYTES,
        ),
        info_sha256=_digest(value.get("info_sha256"), "info_sha256"),
        info_size=_size(
            value.get("info_size"), "info_size", maximum=_MAX_INFO_BYTES
        ),
        projection_id=_digest(value.get("projection_id"), "projection_id"),
    )
    if (
        pin.schema != CONFIGURED_BOARD_EXTENSION_PIN_SCHEMA
        or any(
            _TOKEN.fullmatch(value) is None
            for value in (pin.name, pin.engine_version, pin.platform)
        )
        or not pin.engine_version.startswith("v")
        or pin.name != "quack"
        or pin.projection_id != _projection_id(pin.as_dict())
    ):
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension pin identity is invalid"
        )
    return pin


def _stable_regular_bytes(path: Path, *, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension source is unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= maximum
        ):
            raise ConfiguredBoardExtensionProjectionError(
                "configured-board extension source is not stable evidence"
            )
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            block = os.pread(
                descriptor, min(1024 * 1024, before.st_size - offset), offset
            )
            if not block:
                break
            chunks.append(block)
            offset += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_uid,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
    raw = b"".join(chunks)
    if len(raw) != before.st_size or identity(before) != identity(after):
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension source changed while read"
        )
    return raw


def inspect_configured_board_extension_sources(
    extension_path: Path | str,
    info_path: Path | str,
    *,
    name: str,
    engine_version: str,
    platform: str,
) -> ConfiguredBoardExtensionPin:
    """Return path-free evidence for extension bytes; grant no authority."""

    payload = _stable_regular_bytes(Path(extension_path), maximum=_MAX_PAYLOAD_BYTES)
    info = _stable_regular_bytes(Path(info_path), maximum=_MAX_INFO_BYTES)
    body: dict[str, object] = {
        "schema": CONFIGURED_BOARD_EXTENSION_PIN_SCHEMA,
        "name": name,
        "engine_version": engine_version,
        "platform": platform,
        "payload_sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "payload_size": len(payload),
        "info_sha256": "sha256:" + hashlib.sha256(info).hexdigest(),
        "info_size": len(info),
    }
    body["projection_id"] = _projection_id(body)
    return parse_configured_board_extension_pin(body)


def _write_read_only(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ConfiguredBoardExtensionProjectionError(
                    "configured-board extension projection write failed"
                )
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
    finally:
        os.close(descriptor)


def _private_directory(path: Path, *, mode: int) -> None:
    metadata = os.lstat(path)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != mode
    ):
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension projection directory is unsafe"
        )


def verify_configured_board_extension_home(
    pin: ConfiguredBoardExtensionPin,
    home: Path | str,
) -> Path:
    """Verify exact projected bytes and the closed read-only directory shape."""

    parsed = parse_configured_board_extension_pin(pin.as_dict())
    root = Path(home)
    if not root.is_absolute() or root.name != parsed.projection_id.removeprefix(
        "sha256:"
    ):
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension projection path is invalid"
        )
    directory = root / parsed.relative_directory
    extension = directory / parsed.extension_filename
    info = directory / f"{parsed.extension_filename}.info"
    expected_directories = {
        root,
        root / ".duckdb",
        root / ".duckdb/extensions",
        root / ".duckdb/extensions" / parsed.engine_version,
        directory,
        root / ".python-user-base",
        root / ".cache",
    }
    observed_directories = {root}
    observed_files: set[Path] = set()
    try:
        for entry in root.rglob("*"):
            metadata = os.lstat(entry)
            if stat.S_ISLNK(metadata.st_mode) or metadata.st_uid != os.geteuid():
                raise ConfiguredBoardExtensionProjectionError(
                    "configured-board extension projection custody changed"
                )
            if stat.S_ISDIR(metadata.st_mode):
                observed_directories.add(entry)
            elif stat.S_ISREG(metadata.st_mode):
                observed_files.add(entry)
            else:
                raise ConfiguredBoardExtensionProjectionError(
                    "configured-board extension projection has a foreign node"
                )
    except OSError as exc:
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension projection is unavailable"
        ) from exc
    if observed_directories != expected_directories or observed_files != {
        extension,
        info,
    }:
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension projection contents drifted"
        )
    for candidate in expected_directories - {root / ".cache"}:
        _private_directory(candidate, mode=0o500)
    _private_directory(root / ".cache", mode=0o700)
    for candidate, expected_digest, expected_size in (
        (extension, parsed.payload_sha256, parsed.payload_size),
        (info, parsed.info_sha256, parsed.info_size),
    ):
        metadata = os.lstat(candidate)
        raw = _stable_regular_bytes(
            candidate,
            maximum=max(_MAX_INFO_BYTES, expected_size),
        )
        if (
            stat.S_IMODE(metadata.st_mode) != 0o400
            or len(raw) != expected_size
            or "sha256:" + hashlib.sha256(raw).hexdigest() != expected_digest
        ):
            raise ConfiguredBoardExtensionProjectionError(
                "configured-board extension projection bytes drifted"
            )
    return root


def project_configured_board_extension_home(
    pin: ConfiguredBoardExtensionPin,
    *,
    extension_path: Path | str,
    info_path: Path | str,
    parent: Path | str,
) -> Path:
    """Publish an exact private DuckDB HOME below an owned launch directory."""

    parsed = parse_configured_board_extension_pin(pin.as_dict())
    projection_parent = Path(parent)
    _private_directory(projection_parent, mode=0o700)
    homes = projection_parent / "qualification-homes"
    try:
        homes.mkdir(mode=0o700)
    except FileExistsError:
        pass
    _private_directory(homes, mode=0o700)
    destination = homes / parsed.projection_id.removeprefix("sha256:")
    try:
        existing = os.lstat(destination)
    except FileNotFoundError:
        existing = None
    except OSError as exc:
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension projection destination is unavailable"
        ) from exc
    if existing is not None:
        if stat.S_ISLNK(existing.st_mode):
            raise ConfiguredBoardExtensionProjectionError(
                "configured-board extension projection destination is a symlink"
            )
        return verify_configured_board_extension_home(parsed, destination)
    payload = _stable_regular_bytes(
        Path(extension_path), maximum=_MAX_PAYLOAD_BYTES
    )
    info = _stable_regular_bytes(Path(info_path), maximum=_MAX_INFO_BYTES)
    if (
        len(payload) != parsed.payload_size
        or "sha256:" + hashlib.sha256(payload).hexdigest()
        != parsed.payload_sha256
        or len(info) != parsed.info_size
        or "sha256:" + hashlib.sha256(info).hexdigest() != parsed.info_sha256
    ):
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension source differs from its accepted pin"
        )
    staging = Path(tempfile.mkdtemp(prefix=".extension-home-", dir=homes))
    try:
        directory = staging / parsed.relative_directory
        directory.mkdir(parents=True, mode=0o700)
        (staging / ".python-user-base").mkdir(mode=0o700)
        (staging / ".cache").mkdir(mode=0o700)
        _write_read_only(directory / parsed.extension_filename, payload)
        _write_read_only(directory / f"{parsed.extension_filename}.info", info)
        for candidate in sorted(
            (
                item
                for item in staging.rglob("*")
                if item.is_dir() and item != staging / ".cache"
            ),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            os.chmod(candidate, 0o500)
        os.chmod(staging, 0o500)
        os.rename(staging, destination)
        parent_fd = os.open(homes, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        if staging.exists():
            try:
                os.chmod(staging, 0o700)
                for candidate in staging.rglob("*"):
                    if candidate.is_dir():
                        os.chmod(candidate, 0o700)
            except OSError:
                pass
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_configured_board_extension_home(parsed, destination)


__all__ = (
    "CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV",
    "CONFIGURED_BOARD_EXTENSION_PIN_SCHEMA",
    "ConfiguredBoardExtensionPin",
    "ConfiguredBoardExtensionProjectionError",
    "inspect_configured_board_extension_sources",
    "parse_configured_board_extension_pin",
    "project_configured_board_extension_home",
    "verify_configured_board_extension_home",
)
