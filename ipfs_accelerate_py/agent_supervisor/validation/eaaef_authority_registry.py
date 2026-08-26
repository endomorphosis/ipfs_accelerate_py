"""Owner-private, checkout-independent authority storage for EAAEF.

The reviewed EAAEF authority prefix remains the logical identifier used in
signed material.  It is deliberately not a repository-relative storage
location: artifacts are mapped into a per-account platform-state registry.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import fcntl
import json
import os
import pwd
import stat
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

EAAEF_LOGICAL_AUTHORITY_PREFIX: Final = PurePosixPath(
    "data/agent_supervisor/external_agent_autonomous_execution_fabric/authority"
)
EAAEF_AUTHORITY_PRODUCT_ROOT: Final = "ipfs_accelerate_py-eaaef-authority-v1"
DEFAULT_MAX_JSON_BYTES: Final = 4 * 1024 * 1024

_DIRECTORY_MODE: Final = 0o700
_ARTIFACT_MODE: Final = 0o400
_LOCK_MODE: Final = 0o600
_LOCK_NAME: Final = ".registry.lock"
_AT_EMPTY_PATH: Final = 0x1000
_MAX_PATH_PARTS: Final = 64


class EAAEFAuthorityRegistryError(RuntimeError):
    """Base error for malformed, unsafe, or unavailable registry state."""


class EAAEFAuthorityNotFound(EAAEFAuthorityRegistryError):
    """Raised when a requested immutable authority artifact does not exist."""


class EAAEFAuthorityConflict(EAAEFAuthorityRegistryError):
    """Raised when a create-once path already contains different bytes."""


def _mode(metadata: os.stat_result) -> int:
    return stat.S_IMODE(metadata.st_mode)


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_uid),
        int(metadata.st_gid),
        int(metadata.st_mode),
    )


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_uid),
        int(metadata.st_gid),
        int(metadata.st_mode),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _open_directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _absolute_lexical_path(value: str | Path, name: str) -> Path:
    text = os.fspath(value)
    if not text or "\x00" in text or "\n" in text or "\r" in text or "\\" in text:
        raise EAAEFAuthorityRegistryError(f"{name} is not a safe absolute path")
    if not os.path.isabs(text) or os.path.normpath(text) != text or text == "/":
        raise EAAEFAuthorityRegistryError(f"{name} must be a normalized absolute non-root path")
    path = Path(text)
    if len(path.parts) > _MAX_PATH_PARTS:
        raise EAAEFAuthorityRegistryError(f"{name} has too many path components")
    return path


def _is_within(candidate: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath((os.fspath(candidate), os.fspath(parent))) == os.fspath(parent)
    except ValueError:
        return False


def _logical_suffix(value: str | Path) -> PurePosixPath:
    text = os.fspath(value)
    if (
        not text
        or "\x00" in text
        or "\n" in text
        or "\r" in text
        or "\\" in text
        or text.startswith("/")
    ):
        raise EAAEFAuthorityRegistryError("logical authority path is unsafe")
    logical = PurePosixPath(text)
    if logical.as_posix() != text or any(part in {"", ".", ".."} for part in logical.parts):
        raise EAAEFAuthorityRegistryError(
            "logical authority path must be normalized and repository-relative"
        )
    prefix = EAAEF_LOGICAL_AUTHORITY_PREFIX.parts
    if logical.parts[: len(prefix)] != prefix:
        raise EAAEFAuthorityRegistryError(
            "logical authority path is outside the reviewed EAAEF prefix"
        )
    suffix_parts = logical.parts[len(prefix) :]
    if any(part == _LOCK_NAME or part.startswith(".pending-") for part in suffix_parts):
        raise EAAEFAuthorityRegistryError("logical authority path uses a reserved name")
    return PurePosixPath(*suffix_parts) if suffix_parts else PurePosixPath(".")


@dataclass
class _DirectoryChain:
    descriptors: list[int]
    child_names: list[str]
    initial_identities: list[tuple[int, int, int, int, int]]
    private_from: int

    @property
    def leaf(self) -> int:
        return self.descriptors[-1]

    def assert_stable(self) -> None:
        effective_uid = os.geteuid()
        for index, descriptor in enumerate(self.descriptors):
            observed = os.fstat(descriptor)
            if not stat.S_ISDIR(observed.st_mode):
                raise EAAEFAuthorityRegistryError("registry directory became non-directory")
            if _directory_identity(observed) != self.initial_identities[index]:
                raise EAAEFAuthorityRegistryError("registry directory identity changed")
            if index >= self.private_from and (
                int(observed.st_uid) != effective_uid or _mode(observed) != _DIRECTORY_MODE
            ):
                raise EAAEFAuthorityRegistryError(
                    "registry directories must be owner-controlled mode 0700"
                )
            if index:
                linked = os.stat(
                    self.child_names[index - 1],
                    dir_fd=self.descriptors[index - 1],
                    follow_symlinks=False,
                )
                if _directory_identity(linked) != _directory_identity(observed):
                    raise EAAEFAuthorityRegistryError(
                        "registry directory walk is no longer anchored"
                    )

    def close(self) -> None:
        while self.descriptors:
            os.close(self.descriptors.pop())


def _open_or_create_directory(parent_fd: int, name: str, *, create: bool) -> int:
    flags = _open_directory_flags()
    created = False
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            raise
        try:
            os.mkdir(name, _DIRECTORY_MODE, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            pass
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    if created:
        os.fchmod(descriptor, _DIRECTORY_MODE)
        os.fsync(descriptor)
        os.fsync(parent_fd)
    return descriptor


def _canonical_json(payload: Mapping[str, Any], maximum_bytes: int) -> bytes:
    if not isinstance(payload, Mapping):
        raise EAAEFAuthorityRegistryError("authority payload must be a JSON object")
    try:
        encoded = (
            json.dumps(
                dict(payload),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise EAAEFAuthorityRegistryError("authority payload is not canonical JSON") from exc
    if len(encoded) > maximum_bytes:
        raise EAAEFAuthorityRegistryError("authority payload exceeds the bounded size")
    return encoded


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise EAAEFAuthorityRegistryError("authority artifact write did not progress")
        view = view[written:]


def _open_anonymous_file(parent_fd: int) -> int | None:
    temporary_flag = getattr(os, "O_TMPFILE", 0)
    if not temporary_flag:
        return None
    flags = os.O_WRONLY | temporary_flag | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(".", flags, _ARTIFACT_MODE, dir_fd=parent_fd)
    except OSError as exc:
        unsupported = {
            errno.EINVAL,
            errno.EISDIR,
            errno.ENOSYS,
            errno.EOPNOTSUPP,
        }
        if exc.errno in unsupported:
            return None
        raise
    os.fchmod(descriptor, _ARTIFACT_MODE)
    return descriptor


def _link_anonymous_file(descriptor: int, parent_fd: int, name: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    linkat = libc.linkat
    linkat.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    )
    linkat.restype = ctypes.c_int
    result = linkat(
        descriptor,
        ctypes.c_char_p(b""),
        parent_fd,
        ctypes.c_char_p(os.fsencode(name)),
        _AT_EMPTY_PATH,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), name)


class EAAEFAuthorityRegistry:
    """Resolve and securely operate on one account's EAAEF authority registry."""

    def __init__(
        self,
        *,
        repo_root: str | Path | None = None,
        authority_root: str | Path | None = None,
        max_json_bytes: int = DEFAULT_MAX_JSON_BYTES,
    ) -> None:
        if authority_root is None:
            account_home = _absolute_lexical_path(
                pwd.getpwuid(os.geteuid()).pw_dir,
                "account home",
            )
            # Plan-bound workers intentionally replace or remove HOME/XDG
            # variables.  Authority location must therefore be stable for the
            # effective account across issuer, supervisor, and child gates.
            state_home = account_home / ".local/state"
            selected_root = state_home / EAAEF_AUTHORITY_PRODUCT_ROOT
        else:
            selected_root = _absolute_lexical_path(authority_root, "authority_root")
            state_home = selected_root.parent
        if not isinstance(max_json_bytes, int) or not 0 < max_json_bytes <= 64 * 1024 * 1024:
            raise EAAEFAuthorityRegistryError("max_json_bytes is outside the safe bound")
        if repo_root is not None:
            checkout = Path(repo_root).resolve(strict=False)
            lexical_checkout = Path(os.path.abspath(os.fspath(repo_root)))
            if _is_within(selected_root, lexical_checkout) or _is_within(
                Path(os.path.realpath(selected_root)), checkout
            ):
                raise EAAEFAuthorityRegistryError(
                    "EAAEF authority root must remain outside the repository checkout"
                )
        self.root = selected_root
        self._state_home = state_home
        self.max_json_bytes = max_json_bytes
        self._mutex = threading.RLock()
        self._lock_depth = 0
        self._lock_descriptor: int | None = None
        self._root_chain: _DirectoryChain | None = None

    def physical_path(self, logical_path: str | Path) -> Path:
        """Map a reviewed logical authority identifier to platform state."""

        suffix = _logical_suffix(logical_path)
        return self.root if suffix == PurePosixPath(".") else self.root.joinpath(*suffix.parts)

    def _open_root(self, *, create: bool) -> _DirectoryChain:
        root_parts = self.root.parts[1:]
        state_parts = self._state_home.parts[1:]
        if root_parts[: len(state_parts)] != state_parts:
            raise EAAEFAuthorityRegistryError("authority root escaped its state home")
        descriptors = [os.open("/", _open_directory_flags())]
        names: list[str] = []
        identities = [_directory_identity(os.fstat(descriptors[0]))]
        private_from = len(state_parts)
        try:
            for index, part in enumerate(root_parts, start=1):
                descriptor = _open_or_create_directory(descriptors[-1], part, create=create)
                descriptors.append(descriptor)
                names.append(part)
                identities.append(_directory_identity(os.fstat(descriptor)))
                if index >= private_from:
                    observed = os.fstat(descriptor)
                    if (
                        not stat.S_ISDIR(observed.st_mode)
                        or int(observed.st_uid) != os.geteuid()
                        or _mode(observed) != _DIRECTORY_MODE
                    ):
                        raise EAAEFAuthorityRegistryError(
                            "platform state and registry roots must be owner-controlled mode 0700"
                        )
            chain = _DirectoryChain(descriptors, names, identities, private_from)
            chain.assert_stable()
            return chain
        except FileNotFoundError as exc:
            while descriptors:
                os.close(descriptors.pop())
            raise EAAEFAuthorityNotFound("authority artifact parent does not exist") from exc
        except Exception:
            while descriptors:
                os.close(descriptors.pop())
            raise

    def _open_parent(self, suffix: PurePosixPath, *, create: bool) -> _DirectoryChain:
        if self._root_chain is None:
            raise EAAEFAuthorityRegistryError("registry ceremony lock is not held")
        self._assert_lock_stable()
        root_descriptor = os.dup(self._root_chain.leaf)
        descriptors = [root_descriptor]
        names: list[str] = []
        identities = [_directory_identity(os.fstat(root_descriptor))]
        try:
            for part in suffix.parts[:-1]:
                descriptor = _open_or_create_directory(descriptors[-1], part, create=create)
                descriptors.append(descriptor)
                names.append(part)
                identities.append(_directory_identity(os.fstat(descriptor)))
                observed = os.fstat(descriptor)
                if (
                    not stat.S_ISDIR(observed.st_mode)
                    or int(observed.st_uid) != os.geteuid()
                    or _mode(observed) != _DIRECTORY_MODE
                ):
                    raise EAAEFAuthorityRegistryError(
                        "authority artifact parents must be owner-controlled mode 0700"
                    )
            chain = _DirectoryChain(descriptors, names, identities, 0)
            chain.assert_stable()
            return chain
        except FileNotFoundError as exc:
            while descriptors:
                os.close(descriptors.pop())
            raise EAAEFAuthorityNotFound("authority artifact parent does not exist") from exc
        except Exception:
            while descriptors:
                os.close(descriptors.pop())
            raise

    def _open_lock(self, root_fd: int, *, create: bool) -> int:
        flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        created = False
        if create:
            try:
                descriptor = os.open(
                    _LOCK_NAME,
                    flags | os.O_CREAT | os.O_EXCL,
                    _LOCK_MODE,
                    dir_fd=root_fd,
                )
                created = True
            except FileExistsError:
                descriptor = os.open(_LOCK_NAME, flags, dir_fd=root_fd)
        else:
            try:
                descriptor = os.open(_LOCK_NAME, flags, dir_fd=root_fd)
            except FileNotFoundError as exc:
                raise EAAEFAuthorityNotFound("authority registry is not initialized") from exc
        if created:
            os.fchmod(descriptor, _LOCK_MODE)
            os.fsync(descriptor)
            os.fsync(root_fd)
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or int(observed.st_uid) != os.geteuid()
            or int(observed.st_nlink) != 1
            or _mode(observed) != _LOCK_MODE
        ):
            os.close(descriptor)
            raise EAAEFAuthorityRegistryError(
                "registry lock must be an owned single-link mode-0600 regular file"
            )
        return descriptor

    def _assert_lock_stable(self) -> None:
        if self._root_chain is None or self._lock_descriptor is None:
            raise EAAEFAuthorityRegistryError("registry ceremony lock is not held")
        self._root_chain.assert_stable()
        opened = os.fstat(self._lock_descriptor)
        linked = os.stat(
            _LOCK_NAME,
            dir_fd=self._root_chain.leaf,
            follow_symlinks=False,
        )
        if _file_identity(opened) != _file_identity(linked):
            raise EAAEFAuthorityRegistryError("registry lock identity changed")

    @contextlib.contextmanager
    def ceremony(self) -> Iterator[EAAEFAuthorityRegistry]:
        """Hold the registry-wide exclusive lock across a multi-artifact ceremony."""

        self._mutex.acquire()
        try:
            if self._lock_depth == 0:
                chain = self._open_root(create=True)
                descriptor: int | None = None
                try:
                    descriptor = self._open_lock(chain.leaf, create=True)
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                except Exception:
                    if descriptor is not None:
                        os.close(descriptor)
                    chain.close()
                    raise
                self._root_chain = chain
                self._lock_descriptor = descriptor
                try:
                    self._assert_lock_stable()
                except Exception:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                    os.close(descriptor)
                    chain.close()
                    self._root_chain = None
                    self._lock_descriptor = None
                    raise
            self._lock_depth += 1
            try:
                yield self
            finally:
                self._lock_depth -= 1
                if self._lock_depth == 0:
                    descriptor = self._lock_descriptor
                    chain = self._root_chain
                    self._lock_descriptor = None
                    self._root_chain = None
                    if descriptor is not None:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                        os.close(descriptor)
                    if chain is not None:
                        chain.close()
        finally:
            self._mutex.release()

    @contextlib.contextmanager
    def _read_ceremony(self) -> Iterator[None]:
        self._mutex.acquire()
        try:
            # ``_lock_depth`` describes state protected by this instance's
            # RLock, not thread-local state.  Inspect it only after acquiring
            # the RLock: the owning thread may re-enter immediately, while a
            # different thread must wait for the whole outer ceremony.
            if self._lock_depth:
                self._assert_lock_stable()
                yield
                return
            try:
                chain = self._open_root(create=False)
            except FileNotFoundError as exc:
                raise EAAEFAuthorityNotFound("authority registry does not exist") from exc
            descriptor: int | None = None
            try:
                descriptor = self._open_lock(chain.leaf, create=False)
                fcntl.flock(descriptor, fcntl.LOCK_SH)
                self._root_chain = chain
                self._lock_descriptor = descriptor
                self._lock_depth = 1
                self._assert_lock_stable()
                yield
            finally:
                self._lock_depth = 0
                self._root_chain = None
                self._lock_descriptor = None
                if descriptor is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                    os.close(descriptor)
                chain.close()
        finally:
            self._mutex.release()

    def _read_bytes(self, parent: _DirectoryChain, name: str) -> bytes:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(name, flags, dir_fd=parent.leaf)
        except FileNotFoundError as exc:
            raise EAAEFAuthorityNotFound(f"authority artifact does not exist: {name}") from exc
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or int(before.st_uid) != os.geteuid()
                or int(before.st_nlink) != 1
                or _mode(before) != _ARTIFACT_MODE
                or int(before.st_size) > self.max_json_bytes
            ):
                raise EAAEFAuthorityRegistryError(
                    "authority artifact must be an owned single-link mode-0400 bounded file"
                )
            linked_before = os.stat(name, dir_fd=parent.leaf, follow_symlinks=False)
            if _file_identity(linked_before) != _file_identity(before):
                raise EAAEFAuthorityRegistryError("authority artifact path is unstable")
            chunks: list[bytes] = []
            remaining = self.max_json_bytes + 1
            while remaining:
                chunk = os.read(descriptor, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            if len(data) > self.max_json_bytes:
                raise EAAEFAuthorityRegistryError("authority artifact exceeds the bounded size")
            after = os.fstat(descriptor)
            linked_after = os.stat(name, dir_fd=parent.leaf, follow_symlinks=False)
            parent.assert_stable()
            self._assert_lock_stable()
            if (
                _file_identity(before) != _file_identity(after)
                or _file_identity(after) != _file_identity(linked_after)
                or len(data) != int(after.st_size)
            ):
                raise EAAEFAuthorityRegistryError("authority artifact changed during stable read")
            return data
        finally:
            os.close(descriptor)

    def _decode_json(self, data: bytes) -> dict[str, Any]:
        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            decoded_object: dict[str, Any] = {}
            for key, value in pairs:
                if key in decoded_object:
                    raise EAAEFAuthorityRegistryError(
                        "authority artifact contains a duplicate JSON object key"
                    )
                decoded_object[key] = value
            return decoded_object

        try:
            decoded = json.loads(
                data.decode("utf-8"),
                object_pairs_hook=unique_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EAAEFAuthorityRegistryError("authority artifact is not valid JSON") from exc
        if not isinstance(decoded, dict):
            raise EAAEFAuthorityRegistryError("authority artifact is not a JSON object")
        return decoded

    def read_json(self, logical_path: str | Path) -> dict[str, Any]:
        """Securely and stably load one immutable authority JSON object."""

        suffix = _logical_suffix(logical_path)
        if suffix == PurePosixPath(".") or not suffix.name.endswith(".json"):
            raise EAAEFAuthorityRegistryError("authority artifact path must name a JSON file")
        with self._read_ceremony():
            parent = self._open_parent(suffix, create=False)
            try:
                return self._decode_json(self._read_bytes(parent, suffix.name))
            finally:
                parent.close()

    def _existing_bytes(self, parent: _DirectoryChain, name: str) -> bytes | None:
        try:
            return self._read_bytes(parent, name)
        except EAAEFAuthorityNotFound:
            return None

    def _publish_anonymous(
        self,
        parent: _DirectoryChain,
        name: str,
        encoded: bytes,
    ) -> bool:
        descriptor = _open_anonymous_file(parent.leaf)
        if descriptor is None:
            return False
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or int(before.st_uid) != os.geteuid()
                or int(before.st_nlink) != 0
                or _mode(before) != _ARTIFACT_MODE
            ):
                raise EAAEFAuthorityRegistryError("anonymous authority staging is unsafe")
            _write_all(descriptor, encoded)
            os.fsync(descriptor)
            written = os.fstat(descriptor)
            if int(written.st_size) != len(encoded) or int(written.st_nlink) != 0:
                raise EAAEFAuthorityRegistryError("anonymous authority staging changed")
            parent.assert_stable()
            self._assert_lock_stable()
            try:
                _link_anonymous_file(descriptor, parent.leaf, name)
            except FileExistsError:
                existing = self._read_bytes(parent, name)
                if existing != encoded:
                    raise EAAEFAuthorityConflict(
                        f"immutable authority path already contains different bytes: {name}"
                    ) from None
                return True
            published = os.fstat(descriptor)
            linked = os.stat(name, dir_fd=parent.leaf, follow_symlinks=False)
            parent.assert_stable()
            self._assert_lock_stable()
            if (
                int(published.st_nlink) != 1
                or _mode(published) != _ARTIFACT_MODE
                or _file_identity(published) != _file_identity(linked)
            ):
                raise EAAEFAuthorityRegistryError("published authority artifact is unstable")
            os.fsync(parent.leaf)
            parent.assert_stable()
            self._assert_lock_stable()
            return True
        finally:
            os.close(descriptor)

    def _publish_direct(
        self,
        parent: _DirectoryChain,
        name: str,
        encoded: bytes,
    ) -> None:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(name, flags, _ARTIFACT_MODE, dir_fd=parent.leaf)
        except FileExistsError:
            existing = self._read_bytes(parent, name)
            if existing != encoded:
                raise EAAEFAuthorityConflict(
                    f"immutable authority path already contains different bytes: {name}"
                ) from None
            return
        try:
            os.fchmod(descriptor, _ARTIFACT_MODE)
            _write_all(descriptor, encoded)
            os.fsync(descriptor)
            observed = os.fstat(descriptor)
            linked = os.stat(name, dir_fd=parent.leaf, follow_symlinks=False)
            parent.assert_stable()
            self._assert_lock_stable()
            if (
                not stat.S_ISREG(observed.st_mode)
                or int(observed.st_uid) != os.geteuid()
                or int(observed.st_nlink) != 1
                or _mode(observed) != _ARTIFACT_MODE
                or int(observed.st_size) != len(encoded)
                or _file_identity(observed) != _file_identity(linked)
            ):
                raise EAAEFAuthorityRegistryError("published authority artifact is unstable")
            os.fsync(parent.leaf)
            parent.assert_stable()
            self._assert_lock_stable()
        finally:
            # Never unlink here: if a racing peer replaced the visible name,
            # conditional unlink is not portable.  A failed direct fallback is
            # therefore left fail-closed instead of risking deletion of a peer.
            os.close(descriptor)

    def publish_json(
        self,
        logical_path: str | Path,
        payload: Mapping[str, Any],
    ) -> Path:
        """Create one immutable JSON artifact, or accept exact byte replay."""

        suffix = _logical_suffix(logical_path)
        if suffix == PurePosixPath(".") or not suffix.name.endswith(".json"):
            raise EAAEFAuthorityRegistryError("authority artifact path must name a JSON file")
        encoded = _canonical_json(payload, self.max_json_bytes)
        with self.ceremony():
            parent = self._open_parent(suffix, create=True)
            try:
                existing = self._existing_bytes(parent, suffix.name)
                if existing is not None:
                    if existing != encoded:
                        raise EAAEFAuthorityConflict(
                            "immutable authority path already contains different bytes: "
                            f"{suffix.name}"
                        )
                    return self.physical_path(logical_path)
                if not self._publish_anonymous(parent, suffix.name, encoded):
                    self._publish_direct(parent, suffix.name, encoded)
                verified = self._read_bytes(parent, suffix.name)
                if verified != encoded:
                    raise EAAEFAuthorityRegistryError(
                        "published authority artifact failed exact verification"
                    )
                return self.physical_path(logical_path)
            finally:
                parent.close()


__all__ = [
    "DEFAULT_MAX_JSON_BYTES",
    "EAAEF_AUTHORITY_PRODUCT_ROOT",
    "EAAEF_LOGICAL_AUTHORITY_PREFIX",
    "EAAEFAuthorityConflict",
    "EAAEFAuthorityNotFound",
    "EAAEFAuthorityRegistry",
    "EAAEFAuthorityRegistryError",
]
