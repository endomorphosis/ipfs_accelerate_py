"""Exact, load-only DuckDB extension projection for configured boards.

DuckDB requires an extension filename ending in ``.duckdb_extension`` and
cannot load the same sealed bytes directly from an anonymous memfd path.  The
operator launcher therefore copies an externally accepted extension and its
metadata into a private, read-only DuckDB ``HOME``.  Every byte is rehashed
before and after publication.  This adapter grants no acceptance authority;
the caller must bind the pin to a protected dependency seal.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

CONFIGURED_BOARD_EXTENSION_PIN_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor."
    "configured-board-duckdb-extension-pin@1"
)
CONFIGURED_BOARD_EXTENSION_SET_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor."
    "configured-board-duckdb-extension-set@1"
)
CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV: Final = (
    "IPFS_ACCELERATE_AGENT_DUCKDB_EXTENSION_DIRECTORY"
)
CONFIGURED_BOARD_EXTENSION_SET_PIN_ENV: Final = (
    "IPFS_ACCELERATE_AGENT_DUCKDB_EXTENSION_SET_PIN_JSON"
)
CONFIGURED_BOARD_EXTENSION_SET_PIN_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor."
    "configured-board-duckdb-extension-set-pin@1"
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
_SET_PIN_FIELDS: Final = frozenset({"schema", "members", "set_id"})
_SET_MEMBER_FIELDS: Final = frozenset({"extension_version", "pin"})


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
        or pin.name not in {"httpfs", "quack"}
        or pin.projection_id != _projection_id(pin.as_dict())
    ):
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension pin identity is invalid"
        )
    return pin


def _extension_set(
    pins: Mapping[str, ConfiguredBoardExtensionPin],
) -> tuple[ConfiguredBoardExtensionPin, ...]:
    if not isinstance(pins, Mapping) or not pins:
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension set is empty"
        )
    parsed: list[ConfiguredBoardExtensionPin] = []
    for name, value in sorted(pins.items()):
        if not isinstance(name, str) or not isinstance(
            value, ConfiguredBoardExtensionPin
        ):
            raise ConfiguredBoardExtensionProjectionError(
                "configured-board extension set is invalid"
            )
        pin = parse_configured_board_extension_pin(value.as_dict())
        if name != pin.name:
            raise ConfiguredBoardExtensionProjectionError(
                "configured-board extension set name differs from its pin"
            )
        parsed.append(pin)
    names = tuple(pin.name for pin in parsed)
    if len(names) != len(set(names)) or not set(names) <= {"httpfs", "quack"}:
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension set members are invalid"
        )
    if len({(pin.engine_version, pin.platform) for pin in parsed}) != 1:
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension set engine/platform differs"
        )
    return tuple(parsed)


def configured_board_extension_set_id(
    pins: Mapping[str, ConfiguredBoardExtensionPin],
) -> str:
    """Return the path-independent identity of an exact extension set."""

    parsed = _extension_set(pins)
    return "sha256:" + hashlib.sha256(
        _canonical_json(
            {
                "schema": CONFIGURED_BOARD_EXTENSION_SET_SCHEMA,
                "pins": [pin.as_dict() for pin in parsed],
            }
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ConfiguredBoardExtensionSetMember:
    extension_version: str
    pin: ConfiguredBoardExtensionPin

    def as_dict(self) -> dict[str, object]:
        return {
            "extension_version": self.extension_version,
            "pin": self.pin.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class ConfiguredBoardExtensionSetPin:
    """Exact path-independent load contract for httpfs and Quack."""

    schema: str
    members: tuple[ConfiguredBoardExtensionSetMember, ...]
    set_id: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "members": [member.as_dict() for member in self.members],
            "set_id": self.set_id,
        }

    def to_json(self) -> str:
        return _canonical_json(self.as_dict()).decode("utf-8")

    @property
    def pins(self) -> dict[str, ConfiguredBoardExtensionPin]:
        return {member.pin.name: member.pin for member in self.members}

    @property
    def versions(self) -> dict[str, str]:
        return {
            member.pin.name: member.extension_version
            for member in self.members
        }


def _set_pin_id(value: Mapping[str, object]) -> str:
    body = dict(value)
    body.pop("set_id", None)
    return "sha256:" + hashlib.sha256(_canonical_json(body)).hexdigest()


def build_configured_board_extension_set_pin(
    pins: Mapping[str, ConfiguredBoardExtensionPin],
    *,
    versions: Mapping[str, str],
) -> ConfiguredBoardExtensionSetPin:
    """Build a strict httpfs+Quack set pin from separately admitted pins."""

    parsed = _extension_set(pins)
    if {pin.name for pin in parsed} != {"httpfs", "quack"} or set(
        versions
    ) != {"httpfs", "quack"}:
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension load set must be exact httpfs+quack"
        )
    body: dict[str, object] = {
        "schema": CONFIGURED_BOARD_EXTENSION_SET_PIN_SCHEMA,
        "members": [
            {
                "extension_version": _text(
                    versions[pin.name],
                    f"{pin.name}.extension_version",
                ),
                "pin": pin.as_dict(),
            }
            for pin in parsed
        ],
    }
    body["set_id"] = _set_pin_id(body)
    return parse_configured_board_extension_set_pin(body)


def parse_configured_board_extension_set_pin(
    value: object,
) -> ConfiguredBoardExtensionSetPin:
    """Strictly parse an exact, closed httpfs+Quack set pin."""

    if type(value) is not dict or set(value) != _SET_PIN_FIELDS:
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension set pin fields are noncanonical"
        )
    raw_members = value.get("members")
    if not isinstance(raw_members, list) or len(raw_members) != 2:
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension set pin members are invalid"
        )
    members: list[ConfiguredBoardExtensionSetMember] = []
    for index, raw in enumerate(raw_members):
        if type(raw) is not dict or set(raw) != _SET_MEMBER_FIELDS:
            raise ConfiguredBoardExtensionProjectionError(
                "configured-board extension set member is noncanonical"
            )
        version = _text(
            raw.get("extension_version"),
            f"members[{index}].extension_version",
        )
        if re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z.+_-]{0,63}", version) is None:
            raise ConfiguredBoardExtensionProjectionError(
                "configured-board extension version is invalid"
            )
        members.append(
            ConfiguredBoardExtensionSetMember(
                extension_version=version,
                pin=parse_configured_board_extension_pin(raw.get("pin")),
            )
        )
    pin = ConfiguredBoardExtensionSetPin(
        schema=_text(value.get("schema"), "schema"),
        members=tuple(members),
        set_id=_digest(value.get("set_id"), "set_id"),
    )
    names = tuple(member.pin.name for member in pin.members)
    if (
        pin.schema != CONFIGURED_BOARD_EXTENSION_SET_PIN_SCHEMA
        or names != ("httpfs", "quack")
        or len(
            {
                (member.pin.engine_version, member.pin.platform)
                for member in pin.members
            }
        )
        != 1
        or pin.set_id != _set_pin_id(pin.as_dict())
    ):
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension set pin identity is invalid"
        )
    return pin


def parse_configured_board_extension_set_pin_json(
    value: object,
) -> ConfiguredBoardExtensionSetPin:
    """Parse canonical JSON without duplicate-key or whitespace ambiguity."""

    if not isinstance(value, str) or not 0 < len(value.encode("utf-8")) <= 16_384:
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension set pin JSON is invalid"
        )

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ConfiguredBoardExtensionProjectionError(
                    "configured-board extension set pin JSON has duplicate keys"
                )
            result[key] = item
        return result

    try:
        decoded = json.loads(value, object_pairs_hook=reject_duplicates)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension set pin JSON is invalid"
        ) from exc
    pin = parse_configured_board_extension_set_pin(decoded)
    if pin.to_json() != value:
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension set pin JSON is not canonical"
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


def _verify_extension_home(
    pins: tuple[ConfiguredBoardExtensionPin, ...],
    root: Path,
    *,
    identity: str,
) -> Path:
    if not root.is_absolute() or root.name != identity.removeprefix("sha256:"):
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension projection path is invalid"
        )
    first = pins[0]
    directory = root / first.relative_directory
    expected_directories = {
        root,
        root / ".duckdb",
        root / ".duckdb/extensions",
        root / ".duckdb/extensions" / first.engine_version,
        directory,
        root / ".python-user-base",
        root / ".cache",
    }
    expected_files: dict[Path, tuple[str, int]] = {}
    for pin in pins:
        extension = directory / pin.extension_filename
        expected_files[extension] = (pin.payload_sha256, pin.payload_size)
        expected_files[extension.with_name(f"{extension.name}.info")] = (
            pin.info_sha256,
            pin.info_size,
        )
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
    if (
        observed_directories != expected_directories
        or observed_files != set(expected_files)
    ):
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension projection contents drifted"
        )
    for candidate in expected_directories - {root / ".cache"}:
        _private_directory(candidate, mode=0o500)
    _private_directory(root / ".cache", mode=0o700)
    for candidate, (expected_digest, expected_size) in expected_files.items():
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


def _project_extension_home(
    pins: tuple[ConfiguredBoardExtensionPin, ...],
    sources: Mapping[str, tuple[Path, Path]],
    *,
    parent: Path,
    identity: str,
) -> Path:
    _private_directory(parent, mode=0o700)
    homes = parent / "qualification-homes"
    try:
        homes.mkdir(mode=0o700)
    except FileExistsError:
        pass
    _private_directory(homes, mode=0o700)
    destination = homes / identity.removeprefix("sha256:")
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
        return _verify_extension_home(pins, destination, identity=identity)

    raw_by_name: dict[str, tuple[bytes, bytes]] = {}
    if set(sources) != {pin.name for pin in pins}:
        raise ConfiguredBoardExtensionProjectionError(
            "configured-board extension projection sources are incomplete"
        )
    for pin in pins:
        source = sources.get(pin.name)
        if (
            not isinstance(source, tuple)
            or len(source) != 2
            or not all(isinstance(item, Path) for item in source)
        ):
            raise ConfiguredBoardExtensionProjectionError(
                "configured-board extension projection source is invalid"
            )
        payload = _stable_regular_bytes(source[0], maximum=_MAX_PAYLOAD_BYTES)
        info = _stable_regular_bytes(source[1], maximum=_MAX_INFO_BYTES)
        if (
            len(payload) != pin.payload_size
            or "sha256:" + hashlib.sha256(payload).hexdigest()
            != pin.payload_sha256
            or len(info) != pin.info_size
            or "sha256:" + hashlib.sha256(info).hexdigest() != pin.info_sha256
        ):
            raise ConfiguredBoardExtensionProjectionError(
                "configured-board extension source differs from its accepted pin"
            )
        raw_by_name[pin.name] = (payload, info)

    staging = Path(tempfile.mkdtemp(prefix=".extension-home-", dir=homes))
    try:
        directory = staging / pins[0].relative_directory
        directory.mkdir(parents=True, mode=0o700)
        (staging / ".python-user-base").mkdir(mode=0o700)
        (staging / ".cache").mkdir(mode=0o700)
        for pin in pins:
            payload, info = raw_by_name[pin.name]
            extension = directory / pin.extension_filename
            _write_read_only(extension, payload)
            _write_read_only(extension.with_name(f"{extension.name}.info"), info)
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
    return _verify_extension_home(pins, destination, identity=identity)


def verify_configured_board_extension_home(
    pin: ConfiguredBoardExtensionPin,
    home: Path | str,
) -> Path:
    """Verify exact projected bytes and the closed read-only directory shape."""

    parsed = parse_configured_board_extension_pin(pin.as_dict())
    return _verify_extension_home(
        (parsed,),
        Path(home),
        identity=parsed.projection_id,
    )


def verify_configured_board_extension_set_home(
    pins: Mapping[str, ConfiguredBoardExtensionPin],
    home: Path | str,
) -> Path:
    """Verify one exact, co-versioned multi-extension private projection."""

    parsed = _extension_set(pins)
    identity = configured_board_extension_set_id(
        {pin.name: pin for pin in parsed}
    )
    return _verify_extension_home(parsed, Path(home), identity=identity)


def project_configured_board_extension_home(
    pin: ConfiguredBoardExtensionPin,
    *,
    extension_path: Path | str,
    info_path: Path | str,
    parent: Path | str,
) -> Path:
    """Publish an exact private DuckDB HOME below an owned launch directory."""

    parsed = parse_configured_board_extension_pin(pin.as_dict())
    return _project_extension_home(
        (parsed,),
        {
            parsed.name: (
                Path(extension_path),
                Path(info_path),
            )
        },
        parent=Path(parent),
        identity=parsed.projection_id,
    )


def project_configured_board_extension_set_home(
    pins: Mapping[str, ConfiguredBoardExtensionPin],
    *,
    sources: Mapping[str, tuple[Path | str, Path | str]],
    parent: Path | str,
) -> Path:
    """Publish an exact co-versioned extension set in one private DuckDB HOME."""

    parsed = _extension_set(pins)
    normalized_sources = {
        name: (Path(value[0]), Path(value[1]))
        for name, value in sources.items()
        if isinstance(name, str)
        and isinstance(value, tuple)
        and len(value) == 2
    }
    identity = configured_board_extension_set_id(
        {pin.name: pin for pin in parsed}
    )
    return _project_extension_home(
        parsed,
        normalized_sources,
        parent=Path(parent),
        identity=identity,
    )


class ConfiguredBoardSealedExtensionSet:
    """Suffix-preserving private load paths backed by immutable memfds."""

    def __init__(
        self,
        pin: ConfiguredBoardExtensionSetPin,
        *,
        parent: Path,
        home: Path,
        descriptors: Mapping[str, int],
    ) -> None:
        self.pin = parse_configured_board_extension_set_pin(pin.as_dict())
        self.parent = parent
        self.home = home
        self._descriptors = dict(descriptors)
        self._closed = False

    @property
    def extension_directory(self) -> Path:
        return self.home / ".duckdb/extensions"

    @property
    def install_paths(self) -> dict[str, Path]:
        first = self.pin.members[0].pin
        directory = self.home / first.relative_directory
        return {
            name: directory / member_pin.extension_filename
            for name, member_pin in self.pin.pins.items()
        }

    def verify(self) -> None:
        if self._closed:
            raise ConfiguredBoardExtensionProjectionError(
                "sealed extension set is closed"
            )
        pins = self.pin.pins
        paths = self.install_paths
        if set(self._descriptors) != {"httpfs", "quack"}:
            raise ConfiguredBoardExtensionProjectionError(
                "sealed extension set descriptors are incomplete"
            )
        first = pins["httpfs"]
        expected_directories = {
            self.home,
            self.home / ".duckdb",
            self.home / ".duckdb/extensions",
            self.home / ".duckdb/extensions" / first.engine_version,
            self.home / first.relative_directory,
        }
        observed_directories = {self.home}
        observed_files: set[Path] = set()
        observed_links: set[Path] = set()
        try:
            for entry in self.home.rglob("*"):
                metadata = os.lstat(entry)
                if metadata.st_uid != os.geteuid():
                    raise ConfiguredBoardExtensionProjectionError(
                        "sealed extension set custody changed"
                    )
                if stat.S_ISDIR(metadata.st_mode):
                    observed_directories.add(entry)
                elif stat.S_ISREG(metadata.st_mode):
                    observed_files.add(entry)
                elif stat.S_ISLNK(metadata.st_mode):
                    observed_links.add(entry)
                else:
                    raise ConfiguredBoardExtensionProjectionError(
                        "sealed extension set contains a foreign node"
                    )
        except OSError as exc:
            raise ConfiguredBoardExtensionProjectionError(
                "sealed extension set is unavailable"
            ) from exc
        expected_links = set(paths.values())
        expected_files = {
            path.with_name(f"{path.name}.info") for path in paths.values()
        }
        if (
            observed_directories != expected_directories
            or observed_links != expected_links
            or observed_files != expected_files
        ):
            raise ConfiguredBoardExtensionProjectionError(
                "sealed extension set contents drifted"
            )
        for directory in expected_directories:
            _private_directory(directory, mode=0o500)
        required_seals = (
            fcntl.F_SEAL_SEAL
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_WRITE
        )
        for name, pin in pins.items():
            descriptor = self._descriptors[name]
            extension = paths[name]
            info = extension.with_name(f"{extension.name}.info")
            try:
                descriptor_stat = os.fstat(descriptor)
                target_stat = os.stat(extension)
                target = os.readlink(extension)
                seals = int(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS))
            except OSError as exc:
                raise ConfiguredBoardExtensionProjectionError(
                    "sealed extension set memfd is unavailable"
                ) from exc
            if (
                target != f"/proc/self/fd/{descriptor}"
                or (target_stat.st_dev, target_stat.st_ino, target_stat.st_size)
                != (
                    descriptor_stat.st_dev,
                    descriptor_stat.st_ino,
                    descriptor_stat.st_size,
                )
                or seals != required_seals
                or descriptor_stat.st_size != pin.payload_size
            ):
                raise ConfiguredBoardExtensionProjectionError(
                    "sealed extension set memfd identity drifted"
                )
            digest = hashlib.sha256()
            offset = 0
            while offset < descriptor_stat.st_size:
                block = os.pread(
                    descriptor,
                    min(1024 * 1024, descriptor_stat.st_size - offset),
                    offset,
                )
                if not block:
                    break
                digest.update(block)
                offset += len(block)
            info_metadata = os.lstat(info)
            info_raw = _stable_regular_bytes(info, maximum=_MAX_INFO_BYTES)
            if (
                offset != descriptor_stat.st_size
                or "sha256:" + digest.hexdigest() != pin.payload_sha256
                or not stat.S_ISREG(info_metadata.st_mode)
                or info_metadata.st_nlink != 1
                or stat.S_IMODE(info_metadata.st_mode) != 0o400
                or len(info_raw) != pin.info_size
                or "sha256:" + hashlib.sha256(info_raw).hexdigest()
                != pin.info_sha256
            ):
                raise ConfiguredBoardExtensionProjectionError(
                    "sealed extension set bytes drifted"
                )

    def _open_watch(self) -> int:
        libc = ctypes.CDLL(None, use_errno=True)
        init = libc.inotify_init1
        init.argtypes = [ctypes.c_int]
        init.restype = ctypes.c_int
        add = libc.inotify_add_watch
        add.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        add.restype = ctypes.c_int
        descriptor = int(init(os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)))
        if descriptor < 0:
            raise ConfiguredBoardExtensionProjectionError(
                "sealed extension set race detector is unavailable"
            )
        mask = 0x00000002 | 0x00000004 | 0x00000008 | 0x00000080
        mask |= 0x00000100 | 0x00000200 | 0x00000400 | 0x00000800
        mask |= 0x00002000 | 0x00004000
        try:
            directories = [self.parent, self.home]
            directories.extend(
                item for item in self.home.rglob("*") if item.is_dir()
            )
            for directory in directories:
                if int(add(descriptor, os.fsencode(directory), mask)) < 0:
                    raise ConfiguredBoardExtensionProjectionError(
                        "sealed extension set race detector could not bind custody"
                    )
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    @staticmethod
    def _watch_changed(descriptor: int) -> bool:
        changed = False
        while True:
            try:
                block = os.read(descriptor, 64 * 1024)
            except BlockingIOError:
                break
            if not block:
                break
            changed = True
        return changed

    @contextmanager
    def load_guard(self) -> Iterator[None]:
        """Fail if any path-custody event occurs during native LOAD."""

        watch = self._open_watch()
        try:
            self.verify()
            yield
            self.verify()
            if self._watch_changed(watch):
                raise ConfiguredBoardExtensionProjectionError(
                    "sealed extension set custody changed during LOAD"
                )
        finally:
            os.close(watch)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for descriptor in self._descriptors.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        self._descriptors = {}
        try:
            for current, directories, files in os.walk(self.parent):
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
            shutil.rmtree(self.parent)
        except OSError:
            return


def seal_configured_board_extension_set_home(
    pin: ConfiguredBoardExtensionSetPin,
    source_home: Path | str,
) -> ConfiguredBoardSealedExtensionSet:
    """Seal an exact regular projection into immutable, suffix-loadable memfds."""

    parsed = parse_configured_board_extension_set_pin(pin.as_dict())
    pins = parsed.pins
    regular_home = verify_configured_board_extension_set_home(pins, source_home)
    parent = Path(
        tempfile.mkdtemp(prefix="configured-board-sealed-extensions-", dir="/tmp")
    )
    os.chmod(parent, 0o700)
    home = parent / parsed.set_id.removeprefix("sha256:")
    directory = home / pins["httpfs"].relative_directory
    descriptors: dict[str, int] = {}
    try:
        directory.mkdir(parents=True, mode=0o700)
        required_seals = (
            fcntl.F_SEAL_SEAL
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_WRITE
        )
        for name, member_pin in sorted(pins.items()):
            source = (
                regular_home
                / member_pin.relative_directory
                / member_pin.extension_filename
            )
            source_info = source.with_name(f"{source.name}.info")
            raw = _stable_regular_bytes(source, maximum=_MAX_PAYLOAD_BYTES)
            info_raw = _stable_regular_bytes(source_info, maximum=_MAX_INFO_BYTES)
            descriptor = os.memfd_create(
                f"configured-board-{name}-extension",
                flags=getattr(os, "MFD_CLOEXEC", 0) | os.MFD_ALLOW_SEALING,
            )
            descriptors[name] = descriptor
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise ConfiguredBoardExtensionProjectionError(
                        "sealed extension set memfd write failed"
                    )
                view = view[written:]
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o500)
            fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, required_seals)
            extension = directory / member_pin.extension_filename
            extension.symlink_to(f"/proc/self/fd/{descriptor}")
            _write_read_only(
                extension.with_name(f"{extension.name}.info"),
                info_raw,
            )
        for candidate in sorted(
            (item for item in home.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            os.chmod(candidate, 0o500)
        os.chmod(home, 0o500)
        sealed = ConfiguredBoardSealedExtensionSet(
            parsed,
            parent=parent,
            home=home,
            descriptors=descriptors,
        )
        sealed.verify()
        return sealed
    except BaseException:
        for descriptor in descriptors.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            for current, directories, files in os.walk(parent):
                os.chmod(current, 0o700)
                for name in directories:
                    candidate = Path(current) / name
                    if not candidate.is_symlink():
                        os.chmod(candidate, 0o700)
                for name in files:
                    candidate = Path(current) / name
                    if not candidate.is_symlink():
                        os.chmod(candidate, 0o600)
            shutil.rmtree(parent)
        except OSError:
            pass
        raise


__all__ = (
    "CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV",
    "CONFIGURED_BOARD_EXTENSION_PIN_SCHEMA",
    "CONFIGURED_BOARD_EXTENSION_SET_PIN_ENV",
    "CONFIGURED_BOARD_EXTENSION_SET_PIN_SCHEMA",
    "CONFIGURED_BOARD_EXTENSION_SET_SCHEMA",
    "ConfiguredBoardExtensionPin",
    "ConfiguredBoardExtensionProjectionError",
    "ConfiguredBoardExtensionSetMember",
    "ConfiguredBoardExtensionSetPin",
    "ConfiguredBoardSealedExtensionSet",
    "build_configured_board_extension_set_pin",
    "configured_board_extension_set_id",
    "inspect_configured_board_extension_sources",
    "parse_configured_board_extension_pin",
    "parse_configured_board_extension_set_pin",
    "parse_configured_board_extension_set_pin_json",
    "project_configured_board_extension_home",
    "project_configured_board_extension_set_home",
    "seal_configured_board_extension_set_home",
    "verify_configured_board_extension_home",
    "verify_configured_board_extension_set_home",
)
