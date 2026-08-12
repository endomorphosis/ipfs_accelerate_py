"""Fail-closed, host-local authority for isolated supervisor worktrees.

The profile is signed by an Ed25519 ``did:key`` identity.  Key rotation and
revocation are additionally fenced by a private lifecycle anchor outside the
profile directory.  Consequently, restoring a copied profile directory (and
its old private key) cannot restore an older generation of authority.
"""
from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import math
import os
import pwd
import re
import secrets
import stat
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

ALLOWED_LOCAL_CAPABILITIES = frozenset(
    {"read", "edit", "test", "isolated_worktree", "write_worktree"}
)
DENIED_LOCAL_CAPABILITIES = frozenset(
    {
        "current_checkout_rewrite",
        "merge",
        "push",
        "deploy",
        "destructive_cleanup",
        "arbitrary_secrets",
        "arbitrary_network",
        "secrets",
        "network",
    }
)
PROFILE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/local-dev-profile@5"
LIFECYCLE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/local-profile-lifecycle-anchor@3"
)
PROFILE_FILENAME = "local_dev_profile.json"
SIGNATURE_FILENAME = "local_dev_profile.sig"
KEY_FILENAME = "local_dev_profile.key"
REVOKE_MARKER = "local_dev_profile.revoked"
LIFECYCLE_FILENAME = "local_dev_profile.lifecycle.jsonl"
DEFAULT_PROFILE_DIR_ENV = "AGENT_SUPERVISOR_LOCAL_PROFILE_DIR"
LIFECYCLE_DIR_ENV = "AGENT_SUPERVISOR_LOCAL_PROFILE_LIFECYCLE_DIR"
SIGNING_KEY_ENV = "AGENT_SUPERVISOR_LOCAL_PROFILE_KEY"
_BASE58_ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_ED25519_PUB_MULTICODEC = b"\xed\x01"
DEFAULT_SCOPED_ROUTE_ID = (
    "agent-supervisor-prompt-v3-grok45-terra56-high-auth-or-hard-quota-v1"
)
DEFAULT_REVIEWER_PROVIDER = "local_operator"
DEFAULT_FALLBACK_PROVIDER_ID = "codex"
DEFAULT_FALLBACK_MODEL_ID = "gpt-5.6-terra"
DEFAULT_FALLBACK_REASONING_EFFORT = "high"
_MAX_PROFILE_BYTES = 64 * 1024
_MAX_SIGNATURE_BYTES = 4 * 1024
_MAX_KEY_BYTES = 32
_MAX_ANCHOR_BYTES = 16 * 1024
_MAX_REVOKE_MARKER_BYTES = 64
_LIFECYCLE_ROOT_REGISTRY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/local-profile-root-registry@2"
)
LIFECYCLE_WITNESS_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/local-profile-lifecycle-witness@1"
)
LIFECYCLE_ROOT_KEY_FILENAME = "lifecycle_root_ed25519.key"
_MAX_LIFECYCLE_WITNESS_BYTES = 128 * 1024
_MAX_LIFECYCLE_WITNESS_AGE_MS = 10 * 60 * 1000
_LIFECYCLE_DID_STATE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/local-profile-did-state@1"
)
_LIFECYCLE_REGISTRY_ROOT_OVERRIDE: Path | None = None


class LocalProfileError(ValueError):
    pass


class LocalProfileDenied(LocalProfileError):
    pass


class LocalProfileTampered(LocalProfileDenied):
    pass


class LocalProfileRevoked(LocalProfileDenied):
    pass


class LocalProfilePermissive(LocalProfileDenied):
    pass


class LocalProfileWrongRepository(LocalProfileDenied):
    pass


class LocalProfilePromptDerived(LocalProfileDenied):
    pass


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _base58btc_encode(value: bytes) -> str:
    """Encode bytes with the base58btc alphabet (without multibase prefix)."""

    zeroes = len(value) - len(value.lstrip(b"\0"))
    integer = int.from_bytes(value, "big")
    encoded = bytearray()
    while integer:
        integer, remainder = divmod(integer, 58)
        encoded.append(_BASE58_ALPHABET[remainder])
    return (b"1" * zeroes + bytes(reversed(encoded))).decode("ascii")


def _base58btc_decode(value: str) -> bytes:
    if not value:
        raise ValueError("empty base58btc value")
    indexes = {chr(byte): index for index, byte in enumerate(_BASE58_ALPHABET)}
    integer = 0
    for character in value:
        if character not in indexes:
            raise ValueError("invalid base58btc character")
        integer = integer * 58 + indexes[character]
    zeroes = len(value) - len(value.lstrip("1"))
    body = (
        integer.to_bytes((integer.bit_length() + 7) // 8, "big")
        if integer
        else b""
    )
    return b"\0" * zeroes + body


def ed25519_did_key(public_key: Ed25519PublicKey | bytes) -> str:
    """Return the standards-compatible ``did:key`` for an Ed25519 key.

    The multibase payload is the Ed25519 public-key multicodec ``0xed01``
    followed by the 32 raw public-key bytes, encoded as base58btc.
    """

    raw = (
        public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
        if isinstance(public_key, Ed25519PublicKey)
        else bytes(public_key)
    )
    if len(raw) != 32:
        raise LocalProfileTampered("Ed25519 public key must contain 32 bytes")
    return "did:key:z" + _base58btc_encode(_ED25519_PUB_MULTICODEC + raw)


def ed25519_public_key_from_did(identity_did: str) -> Ed25519PublicKey:
    prefix = "did:key:z"
    try:
        decoded = _base58btc_decode(str(identity_did)[len(prefix) :])
    except (TypeError, ValueError) as exc:
        raise LocalProfileTampered("Ed25519 did:key is invalid") from exc
    if not str(identity_did).startswith(prefix) or not decoded.startswith(
        _ED25519_PUB_MULTICODEC
    ) or len(decoded) != 34:
        raise LocalProfileTampered("Ed25519 did:key is invalid")
    try:
        return Ed25519PublicKey.from_public_bytes(decoded[2:])
    except ValueError as exc:
        raise LocalProfileTampered("Ed25519 did:key is invalid") from exc


def verify_did_key_signature(
    *, identity_did: str, payload: Mapping[str, Any], signature: str
) -> None:
    """Verify a canonical-JSON Ed25519 signature against a ``did:key``."""

    try:
        encoded = base64.b64decode(str(signature).encode("ascii"), validate=True)
        ed25519_public_key_from_did(identity_did).verify(encoded, _canonical(payload))
    except (InvalidSignature, UnicodeError, ValueError, LocalProfileTampered) as exc:
        raise LocalProfileTampered("Ed25519 did:key signature is invalid") from exc


def _path(directory: Path | None) -> Path:
    configured = os.environ.get(DEFAULT_PROFILE_DIR_ENV, "").strip()
    return Path(directory) if directory is not None else Path(
        configured
        or Path.home()
        / ".ipfs_accelerate"
        / "agent_supervisor"
        / "local_profile"
    )


def _absolute_without_symlinks(path: Path) -> Path:
    """Return an absolute lexical path after rejecting every symlink component.

    ``Path.resolve`` is deliberately forbidden before this walk: resolving a
    symlink first destroys the evidence needed to reject a caller-selected
    authority directory.  Parent traversal is rejected for the same reason.
    """

    expanded = path.expanduser()
    candidate = expanded if expanded.is_absolute() else Path.cwd() / expanded
    cursor = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        if component in {"", "."}:
            continue
        if component == "..":
            raise LocalProfileTampered(
                "local authority path cannot contain parent traversal"
            )
        cursor /= component
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise LocalProfileTampered(
                "local authority path component is unavailable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise LocalProfileTampered(
                "local authority path cannot contain symlink components"
            )
    return Path(os.path.abspath(os.fspath(candidate)))


def _entry_exists(path: Path) -> bool:
    """Like lexists: dangling symlinks are existing, invalid entries."""

    secured = _absolute_without_symlinks(path)
    try:
        os.lstat(secured)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise LocalProfileTampered("local authority artifact is unavailable") from exc
    return True


def _regular_snapshot(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _validate_private_regular(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise LocalProfileTampered(
            "local authority artifact must be an owned mode-0600 regular file"
        )


def _read_private_file(path: Path, *, maximum_bytes: int) -> bytes:
    """Read one bounded, stable private file through one no-follow descriptor."""

    secured = _absolute_without_symlinks(path)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise LocalProfileTampered("no-follow authority reads are unavailable")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | nofollow
    )
    parent_descriptor = os.open(secured.anchor, directory_flags)
    try:
        for component in secured.parts[1:-1]:
            child = os.open(
                component,
                directory_flags,
                dir_fd=parent_descriptor,
            )
            os.close(parent_descriptor)
            parent_descriptor = child
        descriptor = os.open(
            secured.name,
            os.O_RDONLY
            | nofollow
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        os.close(parent_descriptor)
        raise LocalProfileTampered("local authority artifact is missing") from exc
    try:
        before = os.fstat(descriptor)
        _validate_private_regular(before)
        if before.st_size > maximum_bytes:
            raise LocalProfileTampered("local authority artifact is oversized")

        remaining = maximum_bytes + 1
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > maximum_bytes:
            raise LocalProfileTampered("local authority artifact is oversized")

        after = os.fstat(descriptor)
        _validate_private_regular(after)
        final_path = os.stat(
            secured.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            _regular_snapshot(before) != _regular_snapshot(after)
            or _regular_snapshot(after) != _regular_snapshot(final_path)
            or len(payload) != after.st_size
        ):
            raise LocalProfileTampered(
                "local authority artifact changed while being read"
            )
        return payload
    except OSError as exc:
        raise LocalProfileTampered(
            "local authority artifact changed while being read"
        ) from exc
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)


class _DuplicateJSONKey(ValueError):
    pass


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJSONKey(key)
        value[key] = item
    return value


def _read_private_json(path: Path, *, maximum_bytes: int) -> Any:
    try:
        text = _read_private_file(path, maximum_bytes=maximum_bytes).decode("utf-8")
        return json.loads(text, object_pairs_hook=_unique_json_object)
    except (UnicodeError, json.JSONDecodeError, _DuplicateJSONKey) as exc:
        raise LocalProfileTampered("local authority JSON is invalid") from exc


def _fsync_directory(path: Path) -> None:
    path = _absolute_without_symlinks(path)
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, data: bytes, mode: int = 0o600) -> None:
    path = _absolute_without_symlinks(path)
    temporary = path.with_name("." + path.name + ".tmp-" + secrets.token_hex(8))
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        os.chmod(path, mode)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _create_directory_chain(path: Path) -> tuple[int, int, int, int]:
    """Create/open an authority directory chain through no-follow dirfds."""

    secured = _absolute_without_symlinks(path)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise LocalProfileTampered(
            "no-follow authority directory creation is unavailable"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | nofollow
    )
    descriptor = os.open(secured.anchor, flags)
    try:
        for component in secured.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_uid,
        )
    except OSError as exc:
        raise LocalProfileTampered(
            "local authority directory cannot be created"
        ) from exc
    finally:
        os.close(descriptor)


def _owned_directory(
    path: Path,
    *,
    expected_identity: tuple[int, int, int, int] | None = None,
) -> None:
    secured = _absolute_without_symlinks(path)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise LocalProfileTampered("no-follow authority checks are unavailable")
    try:
        descriptor = os.open(
            secured,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | nofollow,
        )
    except OSError as exc:
        raise LocalProfileTampered("local authority directory is missing") from exc
    try:
        metadata = os.fstat(descriptor)
        final_path = os.lstat(secured)
        observed_identity = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_uid,
        )
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or observed_identity
            != (
                final_path.st_dev,
                final_path.st_ino,
                final_path.st_mode,
                final_path.st_uid,
            )
            or (
                expected_identity is not None
                and observed_identity != expected_identity
            )
        ):
            raise LocalProfileTampered(
                "local authority directory must be an owned mode-0700 directory"
            )
    except OSError as exc:
        raise LocalProfileTampered(
            "local authority directory changed while being checked"
        ) from exc
    finally:
        os.close(descriptor)


def _ensure_owned_directory(path: Path) -> None:
    secured = _absolute_without_symlinks(path)
    created_identity: tuple[int, int, int, int] | None = None
    if not _entry_exists(secured):
        created_identity = _create_directory_chain(secured)
    if _absolute_without_symlinks(secured) != secured:
        raise LocalProfileTampered(
            "local authority directory changed during creation"
        )
    _owned_directory(secured, expected_identity=created_identity)


def _registry_root() -> Path:
    override = _LIFECYCLE_REGISTRY_ROOT_OVERRIDE
    if override is not None:
        root = _absolute_without_symlinks(Path(override))
    else:
        try:
            real_home = Path(pwd.getpwuid(os.geteuid()).pw_dir)
        except (KeyError, OSError) as exc:
            raise LocalProfileTampered(
                "local lifecycle registry account root is unavailable"
            ) from exc
        root = _absolute_without_symlinks(
            real_home
            / ".local"
            / "state"
            / "ipfs_accelerate_py"
            / "local-profile-root-registry"
        )
    _ensure_owned_directory(root)
    # Rewalk after creation so an inserted intermediate symlink cannot be
    # hidden by the pre-mkdir lexical check.
    if _absolute_without_symlinks(root) != root:
        raise LocalProfileTampered("local lifecycle registry root changed")
    _owned_directory(root)
    return root


def _lifecycle_root_key(
    *,
    registry_root: Path | None = None,
    create: bool,
) -> Ed25519PrivateKey:
    """Load the account's create-once lifecycle root, never a profile key."""

    root = registry_root or _registry_root()
    key_path = root / LIFECYCLE_ROOT_KEY_FILENAME
    if _entry_exists(key_path):
        return _private_material(
            _read_private_file(key_path, maximum_bytes=_MAX_KEY_BYTES)
        )
    if not create:
        raise LocalProfileTampered("local lifecycle root identity is missing")
    material = Ed25519PrivateKey.generate().private_bytes(
        Encoding.Raw,
        PrivateFormat.Raw,
        NoEncryption(),
    )
    try:
        descriptor = os.open(
            key_path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError:
        descriptor = -1
    except OSError as exc:
        raise LocalProfileTampered(
            "local lifecycle root identity cannot be created"
        ) from exc
    if descriptor >= 0:
        try:
            view = memoryview(material)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise LocalProfileTampered(
                        "local lifecycle root identity write was incomplete"
                    )
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(root)
    private = _private_material(
        _read_private_file(key_path, maximum_bytes=_MAX_KEY_BYTES)
    )
    return private


def lifecycle_root_identity_did() -> str:
    """Return the stable account lifecycle root DID without creating it."""

    return _did(_lifecycle_root_key(create=False))


def initialize_local_profile_lifecycle_root() -> str:
    """Create/load the account root independently of any reviewer profile."""

    return _did(_lifecycle_root_key(create=True))


def _registered_anchor_root(
    directory: Path,
    selected_root: Path,
    *,
    register: bool,
) -> Path:
    profile_path = str(_absolute_without_symlinks(directory))
    selected = str(_absolute_without_symlinks(selected_root))
    key = hashlib.sha256(profile_path.encode("utf-8")).hexdigest()
    registry_root = _registry_root()
    root_key = _lifecycle_root_key(
        registry_root=registry_root,
        create=register,
    )
    root_did = _did(root_key)
    record_path = registry_root / f"{key}.json"
    if _entry_exists(record_path):
        value = _read_private_json(record_path, maximum_bytes=_MAX_ANCHOR_BYTES)
        expected = {
            "schema",
            "profile_path",
            "lifecycle_root",
            "root_identity_did",
            "registry_id",
        }
        if (
            not isinstance(value, dict)
            or set(value) != expected
            or value.get("schema") != _LIFECYCLE_ROOT_REGISTRY_SCHEMA
            or value.get("profile_path") != profile_path
            or value.get("lifecycle_root") != selected
            or value.get("root_identity_did") != root_did
            or value.get("registry_id")
            != "sha256:"
            + hashlib.sha256(
                _canonical(
                    {
                        key: item
                        for key, item in value.items()
                        if key != "registry_id"
                    }
                )
            ).hexdigest()
        ):
            raise LocalProfileTampered(
                "local profile lifecycle root does not match its registry"
            )
        return Path(selected)
    if not register:
        raise LocalProfileTampered(
            "local profile lifecycle root is not registered"
        )
    body = {
        "schema": _LIFECYCLE_ROOT_REGISTRY_SCHEMA,
        "profile_path": profile_path,
        "lifecycle_root": selected,
        "root_identity_did": root_did,
    }
    body["registry_id"] = "sha256:" + hashlib.sha256(
        _canonical(body)
    ).hexdigest()
    _atomic_write(record_path, _canonical(body) + b"\n")
    # Verify through the same bounded no-follow reader before granting any
    # profile authority.
    return _registered_anchor_root(
        directory,
        selected_root,
        register=False,
    )


def _anchor_root(
    directory: Path,
    lifecycle_dir: Path | None,
    *,
    register: bool = False,
) -> Path:
    configured = os.environ.get(LIFECYCLE_DIR_ENV, "").strip()
    if lifecycle_dir is not None:
        root = Path(lifecycle_dir)
    elif configured:
        root = Path(configured)
    else:
        state_home = Path(
            os.environ.get("XDG_STATE_HOME", "").strip()
            or Path.home() / ".local" / "state"
        )
        root = state_home / "ipfs_accelerate_py" / "local-profile-lifecycle"
    root = _absolute_without_symlinks(root)
    profile = _absolute_without_symlinks(directory)
    if root == profile or root.is_relative_to(profile):
        raise LocalProfileTampered(
            "local profile lifecycle anchor must be outside the profile directory"
        )
    _ensure_owned_directory(root)
    if _absolute_without_symlinks(root) != root:
        raise LocalProfileTampered("local profile lifecycle root changed")
    return _registered_anchor_root(
        directory,
        root,
        register=register,
    )


def _anchor_path(
    directory: Path,
    lifecycle_dir: Path | None,
    *,
    register: bool = False,
) -> tuple[Path, str]:
    profile_location = str(_absolute_without_symlinks(directory))
    anchor_id = hashlib.sha256(profile_location.encode("utf-8")).hexdigest()
    return (
        _anchor_root(directory, lifecycle_dir, register=register)
        / f"{anchor_id}.json",
        anchor_id,
    )


def resolve_local_profile_state_paths(
    *,
    profile_dir: Path | None = None,
    lifecycle_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Return the exact secured profile and pinned lifecycle-root paths."""

    directory = _absolute_without_symlinks(_path(profile_dir))
    _owned_directory(directory)
    anchor_path, _ = _anchor_path(directory, lifecycle_dir)
    return directory, anchor_path.parent


@contextmanager
def _anchor_lock(anchor_path: Path):
    """Serialize lifecycle transitions without making the lock authority."""

    lock_path = anchor_path.with_suffix(".lock")
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT
            | os.O_RDWR
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise LocalProfileTampered("local lifecycle lock is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        try:
            final_path = os.lstat(lock_path)
        except OSError as exc:
            raise LocalProfileTampered("local lifecycle lock is invalid") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or _regular_snapshot(metadata) != _regular_snapshot(final_path)
        ):
            raise LocalProfileTampered("local lifecycle lock is invalid")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _did_state_path(identity_did: str) -> Path:
    identity = _bound(identity_did, "identity_did")
    return _registry_root() / (
        "did-" + hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".json"
    )


def _sign_did_state(body: Mapping[str, Any]) -> dict[str, Any]:
    root_key = _lifecycle_root_key(create=False)
    signed: dict[str, Any] = {
        **{
            key: item
            for key, item in body.items()
            if key not in {"root_identity_did", "root_signature", "state_id"}
        },
        "root_identity_did": _did(root_key),
    }
    signed["root_signature"] = _signature(root_key, signed)
    signed["state_id"] = "sha256:" + hashlib.sha256(
        _canonical(signed)
    ).hexdigest()
    return signed


def _validate_did_state(
    value: Mapping[str, Any],
    *,
    expected_root_identity_did: str,
) -> dict[str, Any]:
    expected = {
        "schema",
        "identity_did",
        "status",
        "profile_path",
        "profile_id",
        "profile_content_id",
        "anchor_id",
        "generation",
        "previous_identity_did",
        "updated_at_ns",
        "root_identity_did",
        "root_signature",
        "state_id",
    }
    text_names = expected - {"generation", "updated_at_ns"}
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or any(not isinstance(value.get(name), str) for name in text_names)
        or value.get("schema") != _LIFECYCLE_DID_STATE_SCHEMA
        or value.get("status") not in {"active", "revoked"}
        or not value.get("identity_did")
        or not value.get("profile_path")
        or not value.get("profile_id")
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(value.get("profile_content_id") or ""),
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(value.get("anchor_id") or "")
        )
        is None
        or isinstance(value.get("generation"), bool)
        or not isinstance(value.get("generation"), int)
        or int(value.get("generation") or 0) < 1
        or isinstance(value.get("updated_at_ns"), bool)
        or not isinstance(value.get("updated_at_ns"), int)
        or int(value.get("updated_at_ns") or 0) <= 0
        or value.get("root_identity_did") != expected_root_identity_did
        or value.get("state_id")
        != "sha256:"
        + hashlib.sha256(
            _canonical(
                {
                    key: item
                    for key, item in value.items()
                    if key != "state_id"
                }
            )
        ).hexdigest()
    ):
        raise LocalProfileTampered("local lifecycle DID state is invalid")
    verify_did_key_signature(
        identity_did=expected_root_identity_did,
        payload={
            key: item
            for key, item in value.items()
            if key not in {"root_signature", "state_id"}
        },
        signature=str(value.get("root_signature") or ""),
    )
    return dict(value)


def _load_did_state(identity_did: str) -> dict[str, Any]:
    value = _read_private_json(
        _did_state_path(identity_did),
        maximum_bytes=_MAX_ANCHOR_BYTES,
    )
    if not isinstance(value, dict):
        raise LocalProfileTampered("local lifecycle DID state is invalid")
    state = _validate_did_state(
        value,
        expected_root_identity_did=lifecycle_root_identity_did(),
    )
    if state.get("identity_did") != identity_did:
        raise LocalProfileTampered("local lifecycle DID identity drifted")
    return state


def _write_did_state(
    profile: "SignedSupervisorProfile",
    directory: Path,
    *,
    status: str,
    previous_identity_did: str,
    require_absent: bool,
) -> dict[str, Any]:
    path = _did_state_path(profile.identity_did)
    if require_absent and _entry_exists(path):
        raise LocalProfileTampered(
            "local profile identity was already used or revoked"
        )
    body = {
        "schema": _LIFECYCLE_DID_STATE_SCHEMA,
        "identity_did": profile.identity_did,
        "status": status,
        "profile_path": str(_absolute_without_symlinks(directory)),
        "profile_id": profile.profile_id,
        "profile_content_id": profile.content_id,
        "anchor_id": profile.lifecycle_anchor_id,
        "generation": profile.lifecycle_generation,
        "previous_identity_did": previous_identity_did,
        "updated_at_ns": time.time_ns(),
    }
    state = _sign_did_state(body)
    _atomic_write(path, _canonical(state) + b"\n")
    return _load_did_state(profile.identity_did)


def _serialized_lifecycle(function: Any) -> Any:
    """Apply the per-profile external lifecycle lock to public transitions."""

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        directory = _absolute_without_symlinks(_path(kwargs.get("profile_dir")))
        lifecycle_dir = kwargs.get("lifecycle_dir")
        anchor_path, _ = _anchor_path(
            directory,
            lifecycle_dir,
            register=function.__name__ == "initialize_local_profile",
        )
        global_state = _registry_root() / "did-state-global.json"
        with _anchor_lock(global_state):
            with _anchor_lock(anchor_path):
                return function(*args, **kwargs)

    return wrapped


def _sign_lifecycle_anchor(body: Mapping[str, Any]) -> dict[str, Any]:
    root_key = _lifecycle_root_key(create=False)
    signed = {
        **{
            key: item
            for key, item in body.items()
            if key not in {"root_identity_did", "root_signature"}
        },
        "root_identity_did": _did(root_key),
    }
    signed["root_signature"] = _signature(root_key, signed)
    return signed


def _validate_lifecycle_anchor(
    value: Mapping[str, Any],
    *,
    expected_root_identity_did: str = "",
) -> dict[str, Any]:
    expected = {
        "schema",
        "anchor_id",
        "generation",
        "status",
        "repository_cid",
        "profile_id",
        "profile_content_id",
        "identity_did",
        "did_state_id",
        "did_status",
        "previous_profile_id",
        "previous_profile_content_id",
        "previous_identity_did",
        "previous_anchor_digest",
        "updated_at_ns",
        "root_identity_did",
        "root_signature",
    }
    text_names = expected - {"generation", "updated_at_ns"}
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or any(not isinstance(value.get(name), str) for name in text_names)
        or value.get("schema") != LIFECYCLE_SCHEMA
        or not isinstance(value.get("generation"), int)
        or isinstance(value.get("generation"), bool)
        or int(value.get("generation") or 0) < 1
        or isinstance(value.get("updated_at_ns"), bool)
        or not isinstance(value.get("updated_at_ns"), int)
        or int(value.get("updated_at_ns") or 0) <= 0
        or value.get("status") not in {"active", "revoked"}
        or re.fullmatch(
            r"[0-9a-f]{64}", str(value.get("anchor_id") or "")
        )
        is None
        or not str(value.get("repository_cid") or "")
        or not str(value.get("profile_id") or "")
        or not str(value.get("identity_did") or "")
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(value.get("did_state_id") or "")
        )
        is None
        or value.get("did_status") not in {"active", "revoked"}
        or value.get("did_status") != value.get("status")
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(value.get("profile_content_id") or ""),
        )
        is None
        or (
            int(value.get("generation") or 0) == 1
            and (
                any(
                    value.get(name) != ""
                    for name in (
                        "previous_profile_id",
                        "previous_profile_content_id",
                        "previous_identity_did",
                    )
                )
                or (
                    value.get("status") == "active"
                    and value.get("previous_anchor_digest") != ""
                )
                or (
                    value.get("status") == "revoked"
                    and re.fullmatch(
                        r"sha256:[0-9a-f]{64}",
                        str(value.get("previous_anchor_digest") or ""),
                    )
                    is None
                )
            )
        )
        or (
            int(value.get("generation") or 0) > 1
            and (
                not value.get("previous_profile_id")
                or re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(value.get("previous_profile_content_id") or ""),
                )
                is None
                or not value.get("previous_identity_did")
                or re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(value.get("previous_anchor_digest") or ""),
                )
                is None
            )
        )
    ):
        raise LocalProfileTampered("local profile lifecycle anchor is invalid")
    root_did = str(value.get("root_identity_did") or "")
    if expected_root_identity_did and root_did != expected_root_identity_did:
        raise LocalProfileTampered("local lifecycle root identity drifted")
    verify_did_key_signature(
        identity_did=root_did,
        payload={
            key: item
            for key, item in value.items()
            if key != "root_signature"
        },
        signature=str(value.get("root_signature") or ""),
    )
    return dict(value)


def _load_anchor(path: Path) -> dict[str, Any]:
    try:
        value = _read_private_json(path, maximum_bytes=_MAX_ANCHOR_BYTES)
    except LocalProfileTampered as exc:
        raise LocalProfileTampered(
            "local profile lifecycle anchor is invalid"
        ) from exc
    if not isinstance(value, dict):
        raise LocalProfileTampered("local profile lifecycle anchor is invalid")
    return _validate_lifecycle_anchor(
        value,
        expected_root_identity_did=lifecycle_root_identity_did(),
    )


def _private_material(value: bytes) -> Ed25519PrivateKey:
    if not isinstance(value, bytes) or len(value) != 32:
        raise LocalProfileTampered("local signing identity is invalid")
    try:
        return Ed25519PrivateKey.from_private_bytes(value)
    except ValueError as exc:
        raise LocalProfileTampered("local signing identity is invalid") from exc


def _key(
    directory: Path,
    supplied: bytes | None,
    *,
    create: bool,
    fresh: bool = False,
) -> Ed25519PrivateKey:
    path = directory / KEY_FILENAME
    configured = os.environ.get(SIGNING_KEY_ENV)
    provided = supplied if supplied is not None else (
        configured.encode("utf-8") if configured else None
    )
    if create:
        if _entry_exists(path):
            if fresh:
                raise LocalProfileTampered(
                    "fresh local signing identity path is already occupied"
                )
            existing = _read_private_file(path, maximum_bytes=_MAX_KEY_BYTES)
            if provided is not None and existing != provided:
                raise LocalProfileTampered("supplied signing identity does not match")
            return _private_material(existing)
        material = provided or Ed25519PrivateKey.generate().private_bytes(
            Encoding.Raw, PrivateFormat.Raw, NoEncryption()
        )
        private = _private_material(material)
        _atomic_write(path, material)
        return private
    material = _read_private_file(path, maximum_bytes=_MAX_KEY_BYTES)
    if provided is not None and provided != material:
        raise LocalProfileTampered("supplied signing identity does not match")
    return _private_material(material)


def _did(key: Ed25519PrivateKey) -> str:
    return ed25519_did_key(key.public_key())


def _caps(values: Sequence[str]) -> frozenset[str]:
    caps = frozenset(str(value).strip() for value in values)
    if (
        not caps
        or "" in caps
        or caps & DENIED_LOCAL_CAPABILITIES
        or caps - ALLOWED_LOCAL_CAPABILITIES
    ):
        raise LocalProfilePermissive(
            "profile capabilities are not the bounded local allowlist"
        )
    return caps


def _bound(value: str, name: str, *, allow_empty: bool = False) -> str:
    normalized = str(value or "").strip()
    if (not normalized and not allow_empty) or any(
        character in normalized for character in ("\0", "\n", "\r")
    ):
        raise LocalProfileTampered(f"{name} is invalid")
    return normalized


@dataclass(frozen=True)
class SignedSupervisorProfile:
    schema: str
    repository_cid: str
    baseline_commit: str
    capabilities: frozenset[str]
    created_at: float
    profile_id: str
    identity_did: str = ""
    revoked: bool = False
    lifecycle_generation: int = 0
    lifecycle_anchor_id: str = ""
    lifecycle_root_path: str = ""
    effect_bounds: tuple[str, ...] = field(default_factory=tuple)
    budget_cid: str = ""
    resource_cid: str = ""
    route_id: str = ""
    reviewer_identity: str = ""
    reviewer_provider: str = ""
    fallback_provider_id: str = ""
    fallback_model_id: str = ""
    fallback_reasoning_effort: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "repository_cid": self.repository_cid,
            "baseline_commit": self.baseline_commit,
            "capabilities": sorted(self.capabilities),
            "created_at": self.created_at,
            "profile_id": self.profile_id,
            "identity_did": self.identity_did,
            "revoked": self.revoked,
            "lifecycle_generation": self.lifecycle_generation,
            "lifecycle_anchor_id": self.lifecycle_anchor_id,
            "lifecycle_root_path": self.lifecycle_root_path,
            "effect_bounds": list(self.effect_bounds),
            "budget_cid": self.budget_cid,
            "resource_cid": self.resource_cid,
            "route_id": self.route_id,
            "reviewer_identity": self.reviewer_identity,
            "reviewer_provider": self.reviewer_provider,
            "fallback_provider_id": self.fallback_provider_id,
            "fallback_model_id": self.fallback_model_id,
            "fallback_reasoning_effort": self.fallback_reasoning_effort,
        }

    def allows(self, capability: str) -> bool:
        return (
            capability in self.capabilities
            and capability in self.effect_bounds
            and capability not in DENIED_LOCAL_CAPABILITIES
        )

    @property
    def content_id(self) -> str:
        return "sha256:" + hashlib.sha256(_canonical(self.to_dict())).hexdigest()


LocalDevProfile = SignedSupervisorProfile


@dataclass(frozen=True)
class SignedProfileLifecycleReceipt:
    kind: str
    profile_id: str
    identity_did: str
    repository_cid: str
    generation: int
    recorded_at_ns: int
    signature: str

    @property
    def content_id(self) -> str:
        return "sha256:" + hashlib.sha256(_canonical(self.__dict__)).hexdigest()


@dataclass(frozen=True)
class AuthLifecycleFinding:
    """Verified lifecycle observation; evidence never creates authority."""

    profile_id: str
    identity_did: str
    kind: str
    generation: int
    recorded_at_ns: int
    content_id: str


@dataclass(frozen=True)
class ProfileRotationReceipt:
    old_profile_id: str
    new_profile_id: str
    repository_cid: str
    rotated_at: float
    old_identity_did: str = ""
    new_identity_did: str = ""
    lifecycle_generation: int = 0

    @property
    def content_id(self) -> str:
        return "sha256:" + hashlib.sha256(_canonical(self.__dict__)).hexdigest()


def _reject_prompt(
    source: str | None, payload: Mapping[str, Any] | None = None
) -> None:
    if source and source.strip().casefold() in {
        "prompt",
        "prompt_text",
        "user_prompt",
        "chat",
        "message",
    }:
        raise LocalProfilePromptDerived("prompt text cannot supply local authority")
    if payload and any(
        key in payload
        for key in ("prompt", "prompt_text", "user_message", "from_prompt")
    ):
        raise LocalProfilePromptDerived("prompt-derived profile rejected")


def _signature(key: Ed25519PrivateKey, payload: Mapping[str, Any]) -> str:
    return base64.b64encode(key.sign(_canonical(payload))).decode("ascii")


def _verify(
    key: Ed25519PrivateKey, payload: Mapping[str, Any], signature: str
) -> None:
    try:
        key.public_key().verify(
            base64.b64decode(str(signature).encode("ascii"), validate=True),
            _canonical(payload),
        )
    except (InvalidSignature, UnicodeError, ValueError) as exc:
        raise LocalProfileTampered("profile signature invalid") from exc


def sign_profile_binding(
    *,
    profile_dir: Path | None,
    payload: Mapping[str, Any],
    lifecycle_dir: Path | None = None,
) -> dict[str, str]:
    """Sign an exact canonical binding with the active, anchored profile key."""

    directory = _absolute_without_symlinks(_path(profile_dir))
    try:
        raw = _read_private_json(
            directory / PROFILE_FILENAME,
            maximum_bytes=_MAX_PROFILE_BYTES,
        )
        repository_cid = raw["repository_cid"]
        if not isinstance(repository_cid, str):
            raise LocalProfileTampered(
                "signed local profile repository identity is invalid"
            )
    except (LocalProfileTampered, KeyError, TypeError) as exc:
        raise LocalProfileTampered("signed local profile is unavailable") from exc
    profile = load_local_profile(
        repository_cid=repository_cid,
        profile_dir=directory,
        lifecycle_dir=lifecycle_dir,
    )
    key = _key(directory, None, create=False)
    return {
        "identity": profile.identity_did,
        "signature": _signature(key, payload),
        "profile_id": profile.profile_id,
    }


def _write_lifecycle_receipt(
    directory: Path,
    key: Ed25519PrivateKey,
    *,
    kind: str,
    profile: SignedSupervisorProfile,
) -> SignedProfileLifecycleReceipt:
    recorded_at_ns = time.time_ns()
    body = {
        "kind": kind,
        "profile_id": profile.profile_id,
        "identity_did": profile.identity_did,
        "repository_cid": profile.repository_cid,
        "generation": profile.lifecycle_generation,
        "recorded_at_ns": recorded_at_ns,
    }
    receipt = SignedProfileLifecycleReceipt(
        kind=kind,
        profile_id=profile.profile_id,
        identity_did=profile.identity_did,
        repository_cid=profile.repository_cid,
        generation=profile.lifecycle_generation,
        recorded_at_ns=recorded_at_ns,
        signature=_signature(key, body),
    )
    # This is a current-generation audit receipt, not the revocation authority.
    # The external monotonic anchor below is authoritative across key rotation.
    _atomic_write(
        directory / LIFECYCLE_FILENAME,
        _canonical(receipt.__dict__) + b"\n",
    )
    return receipt


def _profile_from_raw(raw: Mapping[str, Any]) -> SignedSupervisorProfile:
    expected = set(SignedSupervisorProfile(
        schema="", repository_cid="", baseline_commit="",
        capabilities=frozenset(), created_at=0.0, profile_id=""
    ).to_dict())
    if set(raw) != expected or raw.get("schema") != PROFILE_SCHEMA:
        raise LocalProfileTampered("unsupported or incomplete profile schema")
    text_fields = expected - {
        "capabilities",
        "effect_bounds",
        "created_at",
        "revoked",
        "lifecycle_generation",
    }
    capabilities = raw.get("capabilities")
    effect_bounds = raw.get("effect_bounds")
    created_at = raw.get("created_at")
    generation = raw.get("lifecycle_generation")
    if (
        any(not isinstance(raw.get(name), str) for name in text_fields)
        or not isinstance(capabilities, list)
        or not isinstance(effect_bounds, list)
        or any(not isinstance(item, str) for item in capabilities)
        or any(not isinstance(item, str) for item in effect_bounds)
        or capabilities != sorted(set(capabilities))
        or effect_bounds != sorted(set(effect_bounds))
        or isinstance(created_at, bool)
        or not isinstance(created_at, (int, float))
        or not math.isfinite(float(created_at))
        or float(created_at) <= 0
        or not isinstance(raw.get("revoked"), bool)
        or isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 1
    ):
        raise LocalProfileTampered("profile fields have noncanonical types")
    try:
        profile = SignedSupervisorProfile(
            schema=PROFILE_SCHEMA,
            repository_cid=_bound(raw["repository_cid"], "repository_cid"),
            baseline_commit=_bound(raw["baseline_commit"], "baseline_commit"),
            capabilities=_caps(capabilities),
            created_at=float(raw["created_at"]),
            profile_id=_bound(raw["profile_id"], "profile_id"),
            identity_did=_bound(raw["identity_did"], "identity_did"),
            revoked=raw["revoked"],
            lifecycle_generation=generation,
            lifecycle_anchor_id=_bound(
                raw["lifecycle_anchor_id"], "lifecycle_anchor_id"
            ),
            lifecycle_root_path=_bound(
                raw["lifecycle_root_path"], "lifecycle_root_path"
            ),
            effect_bounds=tuple(sorted(_caps(effect_bounds))),
            budget_cid=_bound(raw["budget_cid"], "budget_cid"),
            resource_cid=_bound(raw["resource_cid"], "resource_cid"),
            route_id=_bound(raw["route_id"], "route_id"),
            reviewer_identity=_bound(
                raw["reviewer_identity"], "reviewer_identity", allow_empty=True
            ),
            reviewer_provider=_bound(
                raw["reviewer_provider"], "reviewer_provider", allow_empty=True
            ),
            fallback_provider_id=_bound(
                raw["fallback_provider_id"],
                "fallback_provider_id",
                allow_empty=True,
            ),
            fallback_model_id=_bound(
                raw["fallback_model_id"], "fallback_model_id", allow_empty=True
            ),
            fallback_reasoning_effort=_bound(
                raw["fallback_reasoning_effort"],
                "fallback_reasoning_effort",
                allow_empty=True,
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LocalProfileTampered("profile fields invalid") from exc
    if (
        profile.revoked
        or profile.lifecycle_generation < 1
        or not profile.effect_bounds
        or not set(profile.effect_bounds).issubset(profile.capabilities)
        or profile.reviewer_identity != profile.identity_did
        or profile.reviewer_provider.casefold()
        in {"", "codex", "openai"}
        or profile.route_id != DEFAULT_SCOPED_ROUTE_ID
        or profile.fallback_provider_id != DEFAULT_FALLBACK_PROVIDER_ID
        or profile.fallback_model_id != DEFAULT_FALLBACK_MODEL_ID
        or profile.fallback_reasoning_effort
        != DEFAULT_FALLBACK_REASONING_EFFORT
    ):
        raise LocalProfileTampered("profile authority bounds are invalid")
    return profile


def _default_bound(prefix: str, repository_cid: str, baseline_commit: str) -> str:
    return "sha256:" + hashlib.sha256(
        f"{prefix}\0{repository_cid}\0{baseline_commit}".encode("utf-8")
    ).hexdigest()


@_serialized_lifecycle
def initialize_local_profile(
    *,
    repository_cid: str,
    baseline_commit: str,
    capabilities: Sequence[str] | None = None,
    profile_dir: Path | None = None,
    lifecycle_dir: Path | None = None,
    signing_key: bytes | None = None,
    force: bool = False,
    effect_bounds: Sequence[str] | None = None,
    budget_cid: str = "",
    resource_cid: str = "",
    route_id: str = "",
    reviewer_identity: str = "",
    reviewer_provider: str = "",
    fallback_provider_id: str = "",
    fallback_model_id: str = "",
    fallback_reasoning_effort: str = "",
) -> SignedSupervisorProfile:
    repository = _bound(repository_cid, "repository_cid")
    baseline = _bound(baseline_commit, "baseline_commit")
    directory = _absolute_without_symlinks(_path(profile_dir))
    _ensure_owned_directory(directory)
    profile_file = directory / PROFILE_FILENAME
    profile_exists = _entry_exists(profile_file)
    if profile_exists and not force:
        return load_local_profile(
            repository_cid=repository,
            profile_dir=directory,
            lifecycle_dir=lifecycle_dir,
            signing_key=signing_key,
        )
    if profile_exists and force:
        return _rotate_profile(
            repository_cid=repository,
            baseline_commit=baseline,
            capabilities=capabilities,
            profile_dir=directory,
            lifecycle_dir=lifecycle_dir,
            signing_key=signing_key,
            effect_bounds=effect_bounds,
            budget_cid=budget_cid,
            resource_cid=resource_cid,
            route_id=route_id,
            reviewer_identity=reviewer_identity,
            reviewer_provider=reviewer_provider,
            fallback_provider_id=fallback_provider_id,
            fallback_model_id=fallback_model_id,
            fallback_reasoning_effort=fallback_reasoning_effort,
        ).profile

    anchor_path, anchor_id = _anchor_path(
        directory,
        lifecycle_dir,
        register=True,
    )
    if _entry_exists(anchor_path):
        # A missing/restored profile directory cannot roll an existing anchor
        # back by simply invoking the initializer again.
        _load_anchor(anchor_path)
        raise LocalProfileTampered(
            "local profile lifecycle anchor already exists for this location"
        )
    key = _key(directory, signing_key, create=True, fresh=True)
    caps = _caps(capabilities or tuple(sorted(ALLOWED_LOCAL_CAPABILITIES)))
    effects = tuple(sorted(_caps(effect_bounds or tuple(sorted(caps)))))
    now = time.time()
    identity = _did(key)
    if _entry_exists(_did_state_path(identity)):
        raise LocalProfileTampered(
            "local profile identity was already used or revoked"
        )
    profile = SignedSupervisorProfile(
        schema=PROFILE_SCHEMA,
        repository_cid=repository,
        baseline_commit=baseline,
        capabilities=caps,
        created_at=now,
        profile_id=hashlib.sha256(
            f"{repository}\0{baseline}\0{identity}\0{now}\0{secrets.token_hex(16)}".encode()
        ).hexdigest()[:32],
        identity_did=identity,
        revoked=False,
        lifecycle_generation=1,
        lifecycle_anchor_id=anchor_id,
        lifecycle_root_path=str(anchor_path.parent),
        effect_bounds=effects,
        budget_cid=_bound(
            budget_cid or _default_bound("single-attempt-budget", repository, baseline),
            "budget_cid",
        ),
        resource_cid=_bound(
            resource_cid
            or _default_bound("isolated-worktree-resource", repository, baseline),
            "resource_cid",
        ),
        route_id=_bound(route_id or DEFAULT_SCOPED_ROUTE_ID, "route_id"),
        reviewer_identity=_bound(
            reviewer_identity or identity, "reviewer_identity"
        ),
        reviewer_provider=_bound(
            reviewer_provider or DEFAULT_REVIEWER_PROVIDER,
            "reviewer_provider",
        ),
        fallback_provider_id=_bound(
            fallback_provider_id or DEFAULT_FALLBACK_PROVIDER_ID,
            "fallback_provider_id",
        ),
        fallback_model_id=_bound(
            fallback_model_id or DEFAULT_FALLBACK_MODEL_ID,
            "fallback_model_id",
        ),
        fallback_reasoning_effort=_bound(
            fallback_reasoning_effort or DEFAULT_FALLBACK_REASONING_EFFORT,
            "fallback_reasoning_effort",
        ),
    )
    # Validate cross-field bounds before persisting any authority.
    profile = _profile_from_raw(profile.to_dict())
    _atomic_write(profile_file, _canonical(profile.to_dict()) + b"\n")
    _atomic_write(
        directory / SIGNATURE_FILENAME,
        (_signature(key, profile.to_dict()) + "\n").encode("ascii"),
    )
    _write_lifecycle_receipt(directory, key, kind="initialized", profile=profile)
    did_state = _write_did_state(
        profile,
        directory,
        status="active",
        previous_identity_did="",
        require_absent=True,
    )
    anchor = _sign_lifecycle_anchor({
        "schema": LIFECYCLE_SCHEMA,
        "anchor_id": anchor_id,
        "generation": 1,
        "status": "active",
        "repository_cid": repository,
        "profile_id": profile.profile_id,
        "profile_content_id": profile.content_id,
        "identity_did": identity,
        "did_state_id": did_state["state_id"],
        "did_status": "active",
        "previous_profile_id": "",
        "previous_profile_content_id": "",
        "previous_identity_did": "",
        "previous_anchor_digest": "",
        "updated_at_ns": time.time_ns(),
    })
    _atomic_write(anchor_path, _canonical(anchor) + b"\n")
    return profile


def load_local_profile(
    *,
    repository_cid: str,
    profile_dir: Path | None = None,
    lifecycle_dir: Path | None = None,
    signing_key: bytes | None = None,
    source: str | None = None,
    prompt_payload: Mapping[str, Any] | None = None,
) -> SignedSupervisorProfile:
    _reject_prompt(source, prompt_payload)
    directory = _absolute_without_symlinks(_path(profile_dir))
    _owned_directory(directory)
    marker = directory / REVOKE_MARKER
    if _entry_exists(marker):
        _read_private_file(marker, maximum_bytes=_MAX_REVOKE_MARKER_BYTES)
        raise LocalProfileRevoked("local profile is revoked")
    try:
        raw = _read_private_json(
            directory / PROFILE_FILENAME,
            maximum_bytes=_MAX_PROFILE_BYTES,
        )
        signature = _read_private_file(
            directory / SIGNATURE_FILENAME,
            maximum_bytes=_MAX_SIGNATURE_BYTES,
        ).decode("ascii").strip()
    except (LocalProfileTampered, UnicodeError) as exc:
        raise LocalProfileTampered(
            "signed local profile is missing or unreadable"
        ) from exc
    if not isinstance(raw, dict):
        raise LocalProfileTampered("profile must be an object")
    _reject_prompt(source, raw)
    key = _key(directory, signing_key, create=False)
    _verify(key, raw, signature)
    profile = _profile_from_raw(raw)
    if profile.repository_cid != repository_cid:
        raise LocalProfileWrongRepository(
            "profile is bound to a different repository"
        )
    if profile.identity_did != _did(key):
        raise LocalProfileTampered(
            "profile identity does not match its private signing key"
        )
    # Parsing the DID through the public verifier catches non-multicodec and
    # non-base58 impostors even when the private file is present.
    ed25519_public_key_from_did(profile.identity_did)
    anchor_path, anchor_id = _anchor_path(directory, lifecycle_dir)
    anchor = _load_anchor(anchor_path)
    did_state = _load_did_state(profile.identity_did)
    if anchor.get("status") == "revoked":
        raise LocalProfileRevoked("local profile is revoked")
    if (
        anchor.get("anchor_id") != anchor_id
        or profile.lifecycle_anchor_id != anchor_id
        or profile.lifecycle_root_path != str(anchor_path.parent)
        or anchor.get("generation") != profile.lifecycle_generation
        or anchor.get("repository_cid") != profile.repository_cid
        or anchor.get("profile_id") != profile.profile_id
        or anchor.get("profile_content_id") != profile.content_id
        or anchor.get("identity_did") != profile.identity_did
        or anchor.get("did_status") != "active"
        or anchor.get("did_state_id") != did_state.get("state_id")
        or did_state.get("status") != "active"
        or did_state.get("profile_path") != str(directory)
        or did_state.get("profile_id") != profile.profile_id
        or did_state.get("profile_content_id") != profile.content_id
        or did_state.get("anchor_id") != anchor_id
        or did_state.get("generation") != profile.lifecycle_generation
    ):
        raise LocalProfileTampered(
            "local profile does not match the monotonic lifecycle anchor"
        )
    return profile


def resolve_reviewer_local_profile_state(
    *,
    repository_cid: str,
    reviewer_identity: str,
    expected_profile_id: str = "",
    expected_profile_content_id: str = "",
    expected_lifecycle_anchor_id: str = "",
    expected_lifecycle_generation: int = 0,
) -> tuple["SignedSupervisorProfile", Path, Path]:
    """Load the live profile bound to one reviewer DID.

    Multi-board hosts keep a default local-profile directory plus one or more
    board-scoped directories.  Route authorization names the reviewer DID;
    the signed DID registry is the only locator that may select that
    directory.  The default profile directory is never consulted.
    """

    identity = _bound(reviewer_identity, "reviewer_identity")
    state = _load_did_state(identity)
    if state.get("status") != "active":
        raise LocalProfileRevoked("reviewer local profile is revoked")
    profile_dir = Path(str(state["profile_path"]))
    profile_location = str(_absolute_without_symlinks(profile_dir))
    record_path = _registry_root() / (
        hashlib.sha256(profile_location.encode("utf-8")).hexdigest() + ".json"
    )
    if not _entry_exists(record_path):
        raise LocalProfileTampered(
            "reviewer local profile lifecycle root is not registered"
        )
    record = _read_private_json(record_path, maximum_bytes=_MAX_ANCHOR_BYTES)
    if not isinstance(record, dict) or not isinstance(
        record.get("lifecycle_root"), str
    ):
        raise LocalProfileTampered(
            "reviewer local profile lifecycle registry is invalid"
        )
    lifecycle_dir = _registered_anchor_root(
        profile_dir,
        Path(str(record["lifecycle_root"])),
        register=False,
    )
    profile = load_local_profile(
        repository_cid=repository_cid,
        profile_dir=profile_dir,
        lifecycle_dir=lifecycle_dir,
    )
    if (
        profile.identity_did != identity
        or profile.profile_id != state["profile_id"]
        or profile.content_id != state["profile_content_id"]
        or profile.lifecycle_anchor_id != state["anchor_id"]
        or profile.lifecycle_generation != state["generation"]
        or (
            expected_profile_id
            and profile.profile_id != expected_profile_id
        )
        or (
            expected_profile_content_id
            and profile.content_id != expected_profile_content_id
        )
        or (
            expected_lifecycle_anchor_id
            and profile.lifecycle_anchor_id != expected_lifecycle_anchor_id
        )
        or (
            expected_lifecycle_generation > 0
            and profile.lifecycle_generation != expected_lifecycle_generation
        )
    ):
        raise LocalProfileTampered(
            "reviewer local profile does not match its DID registry"
        )
    resolved_dir, resolved_lifecycle = resolve_local_profile_state_paths(
        profile_dir=profile_dir,
        lifecycle_dir=lifecycle_dir,
    )
    if (
        resolved_dir != _absolute_without_symlinks(profile_dir)
        or resolved_lifecycle != _absolute_without_symlinks(lifecycle_dir)
    ):
        raise LocalProfileTampered(
            "reviewer local profile state paths drifted"
        )
    return profile, resolved_dir, resolved_lifecycle


@dataclass(frozen=True)
class _RotationResult:
    receipt: ProfileRotationReceipt
    profile: SignedSupervisorProfile

    def __getattr__(self, name: str) -> Any:
        # Internal compatibility for the initializer's force path.
        return getattr(self.receipt, name)


def _rotate_profile(
    *,
    repository_cid: str,
    baseline_commit: str,
    capabilities: Sequence[str] | None,
    profile_dir: Path,
    lifecycle_dir: Path | None,
    signing_key: bytes | None,
    effect_bounds: Sequence[str] | None,
    budget_cid: str,
    resource_cid: str,
    route_id: str,
    reviewer_identity: str,
    reviewer_provider: str,
    fallback_provider_id: str,
    fallback_model_id: str,
    fallback_reasoning_effort: str,
) -> _RotationResult:
    old = load_local_profile(
        repository_cid=repository_cid,
        profile_dir=profile_dir,
        lifecycle_dir=lifecycle_dir,
    )
    anchor_path, anchor_id = _anchor_path(profile_dir, lifecycle_dir)
    anchor = _load_anchor(anchor_path)
    old_material = _read_private_file(
        profile_dir / KEY_FILENAME,
        maximum_bytes=_MAX_KEY_BYTES,
    )
    configured = os.environ.get(SIGNING_KEY_ENV)
    supplied = signing_key if signing_key is not None else (
        configured.encode("utf-8") if configured else None
    )
    new_material = supplied or Ed25519PrivateKey.generate().private_bytes(
        Encoding.Raw, PrivateFormat.Raw, NoEncryption()
    )
    if new_material == old_material:
        raise LocalProfileTampered("rotation requires a fresh Ed25519 identity")
    new_key = _private_material(new_material)
    if _did(new_key) == old.identity_did:
        raise LocalProfileTampered("rotation requires a fresh Ed25519 identity")
    generation = int(anchor["generation"]) + 1
    caps = _caps(capabilities or tuple(sorted(old.capabilities)))
    effects = tuple(
        sorted(_caps(effect_bounds or tuple(sorted(old.effect_bounds))))
    )
    now = time.time()
    identity = _did(new_key)
    if _entry_exists(_did_state_path(identity)):
        raise LocalProfileTampered(
            "rotation identity was already used or revoked"
        )
    profile = SignedSupervisorProfile(
        schema=PROFILE_SCHEMA,
        repository_cid=repository_cid,
        baseline_commit=baseline_commit,
        capabilities=caps,
        created_at=now,
        profile_id=hashlib.sha256(
            f"{repository_cid}\0{baseline_commit}\0{identity}\0{generation}\0{secrets.token_hex(16)}".encode()
        ).hexdigest()[:32],
        identity_did=identity,
        revoked=False,
        lifecycle_generation=generation,
        lifecycle_anchor_id=anchor_id,
        lifecycle_root_path=str(anchor_path.parent),
        effect_bounds=effects,
        budget_cid=budget_cid or old.budget_cid,
        resource_cid=resource_cid or old.resource_cid,
        route_id=route_id or old.route_id,
        # Rotation always moves reviewer authority to the fresh local key.
        # Carrying the old did:key would make the new profile unable to sign
        # its own invocation bindings and would defeat key revocation.
        reviewer_identity=(reviewer_identity or identity),
        reviewer_provider=(reviewer_provider or old.reviewer_provider),
        fallback_provider_id=(fallback_provider_id or old.fallback_provider_id),
        fallback_model_id=(fallback_model_id or old.fallback_model_id),
        fallback_reasoning_effort=(
            fallback_reasoning_effort or old.fallback_reasoning_effort
        ),
    )
    profile = _profile_from_raw(profile.to_dict())
    # Files first, anchor last: a crash at any intermediate point fails closed
    # against the old anchor.  It never makes an old generation current.
    _atomic_write(profile_dir / KEY_FILENAME, new_material)
    _atomic_write(
        profile_dir / PROFILE_FILENAME, _canonical(profile.to_dict()) + b"\n"
    )
    _atomic_write(
        profile_dir / SIGNATURE_FILENAME,
        (_signature(new_key, profile.to_dict()) + "\n").encode("ascii"),
    )
    _write_lifecycle_receipt(
        profile_dir, new_key, kind="rotated", profile=profile
    )
    old_did_state = _load_did_state(old.identity_did)
    _write_did_state(
        old,
        profile_dir,
        status="revoked",
        previous_identity_did=str(
            old_did_state.get("previous_identity_did") or ""
        ),
        require_absent=False,
    )
    new_did_state = _write_did_state(
        profile,
        profile_dir,
        status="active",
        previous_identity_did=old.identity_did,
        require_absent=True,
    )
    new_anchor = _sign_lifecycle_anchor({
        "schema": LIFECYCLE_SCHEMA,
        "anchor_id": anchor_id,
        "generation": generation,
        "status": "active",
        "repository_cid": repository_cid,
        "profile_id": profile.profile_id,
        "profile_content_id": profile.content_id,
        "identity_did": identity,
        "did_state_id": new_did_state["state_id"],
        "did_status": "active",
        "previous_profile_id": old.profile_id,
        "previous_profile_content_id": old.content_id,
        "previous_identity_did": old.identity_did,
        "previous_anchor_digest": "sha256:"
        + hashlib.sha256(_canonical(anchor)).hexdigest(),
        "updated_at_ns": time.time_ns(),
    })
    _atomic_write(anchor_path, _canonical(new_anchor) + b"\n")
    receipt = ProfileRotationReceipt(
        old_profile_id=old.profile_id,
        new_profile_id=profile.profile_id,
        repository_cid=repository_cid,
        rotated_at=time.time(),
        old_identity_did=old.identity_did,
        new_identity_did=profile.identity_did,
        lifecycle_generation=generation,
    )
    return _RotationResult(receipt=receipt, profile=profile)


@_serialized_lifecycle
def rotate_local_profile(**kwargs: Any) -> ProfileRotationReceipt:
    directory = _absolute_without_symlinks(_path(kwargs.get("profile_dir")))
    result = _rotate_profile(
        repository_cid=_bound(kwargs["repository_cid"], "repository_cid"),
        baseline_commit=_bound(kwargs["baseline_commit"], "baseline_commit"),
        capabilities=kwargs.get("capabilities"),
        profile_dir=directory,
        lifecycle_dir=kwargs.get("lifecycle_dir"),
        signing_key=kwargs.get("signing_key"),
        effect_bounds=kwargs.get("effect_bounds"),
        budget_cid=str(kwargs.get("budget_cid") or ""),
        resource_cid=str(kwargs.get("resource_cid") or ""),
        route_id=str(kwargs.get("route_id") or ""),
        reviewer_identity=str(kwargs.get("reviewer_identity") or ""),
        reviewer_provider=str(kwargs.get("reviewer_provider") or ""),
        fallback_provider_id=str(kwargs.get("fallback_provider_id") or ""),
        fallback_model_id=str(kwargs.get("fallback_model_id") or ""),
        fallback_reasoning_effort=str(
            kwargs.get("fallback_reasoning_effort") or ""
        ),
    )
    return result.receipt


@_serialized_lifecycle
def revoke_local_profile(
    *, profile_dir: Path | None = None, lifecycle_dir: Path | None = None
) -> None:
    directory = _absolute_without_symlinks(_path(profile_dir))
    try:
        raw = _read_private_json(
            directory / PROFILE_FILENAME,
            maximum_bytes=_MAX_PROFILE_BYTES,
        )
        repository_cid = raw["repository_cid"]
        if not isinstance(repository_cid, str):
            raise LocalProfileTampered(
                "signed local profile repository identity is invalid"
            )
    except (LocalProfileTampered, KeyError, TypeError) as exc:
        raise LocalProfileTampered("signed local profile is unavailable") from exc
    profile = load_local_profile(
        repository_cid=repository_cid,
        profile_dir=directory,
        lifecycle_dir=lifecycle_dir,
    )
    key = _key(directory, None, create=False)
    anchor_path, anchor_id = _anchor_path(directory, lifecycle_dir)
    anchor = _load_anchor(anchor_path)
    current_did_state = _load_did_state(profile.identity_did)
    revoked_did_state = _write_did_state(
        profile,
        directory,
        status="revoked",
        previous_identity_did=str(
            current_did_state.get("previous_identity_did") or ""
        ),
        require_absent=False,
    )
    revoked = _sign_lifecycle_anchor({
        **anchor,
        "anchor_id": anchor_id,
        "status": "revoked",
        "did_state_id": revoked_did_state["state_id"],
        "did_status": "revoked",
        "previous_anchor_digest": "sha256:"
        + hashlib.sha256(_canonical(anchor)).hexdigest(),
        "updated_at_ns": time.time_ns(),
    })
    # The external anchor is made terminal before the convenience marker. A
    # crash or marker deletion therefore cannot revive the authority.
    _atomic_write(anchor_path, _canonical(revoked) + b"\n")
    _write_lifecycle_receipt(directory, key, kind="revoked", profile=profile)
    _atomic_write(directory / REVOKE_MARKER, b"revoked\n")


def _profile_registry_record(directory: Path) -> dict[str, Any]:
    profile_path = str(_absolute_without_symlinks(directory))
    record_path = _registry_root() / (
        hashlib.sha256(profile_path.encode("utf-8")).hexdigest() + ".json"
    )
    value = _read_private_json(record_path, maximum_bytes=_MAX_ANCHOR_BYTES)
    expected = {
        "schema",
        "profile_path",
        "lifecycle_root",
        "root_identity_did",
        "registry_id",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or any(not isinstance(value.get(name), str) for name in expected)
        or value.get("schema") != _LIFECYCLE_ROOT_REGISTRY_SCHEMA
        or value.get("profile_path") != profile_path
        or value.get("root_identity_did") != lifecycle_root_identity_did()
        or value.get("registry_id")
        != "sha256:"
        + hashlib.sha256(
            _canonical(
                {
                    key: item
                    for key, item in value.items()
                    if key != "registry_id"
                }
            )
        ).hexdigest()
    ):
        raise LocalProfileTampered("local lifecycle registry record is invalid")
    return value


@_serialized_lifecycle
def export_local_profile_lifecycle_witness(
    *,
    repository_cid: str,
    board_namespace: str,
    base_head: str,
    base_tree: str,
    nonce: str,
    profile_dir: Path | None = None,
    lifecycle_dir: Path | None = None,
    observed_at_ms: int | None = None,
    expires_at_ms: int | None = None,
) -> dict[str, Any]:
    """Export a portable dual-signed witness of current reviewer authority."""

    namespace = _bound(board_namespace, "board_namespace")
    witness_nonce = _bound(nonce, "nonce")
    if (
        not isinstance(base_head, str)
        or re.fullmatch(r"[0-9a-f]{40}", base_head) is None
        or not isinstance(base_tree, str)
        or re.fullmatch(r"[0-9a-f]{40}", base_tree) is None
    ):
        raise LocalProfileTampered("lifecycle witness Git identity is invalid")
    observed = (
        int(time.time() * 1000)
        if observed_at_ms is None
        else observed_at_ms
    )
    expires = (
        observed + _MAX_LIFECYCLE_WITNESS_AGE_MS
        if expires_at_ms is None
        else expires_at_ms
    )
    if (
        isinstance(observed, bool)
        or not isinstance(observed, int)
        or observed <= 0
        or isinstance(expires, bool)
        or not isinstance(expires, int)
        or not observed < expires
        or expires - observed > _MAX_LIFECYCLE_WITNESS_AGE_MS
    ):
        raise LocalProfileTampered("lifecycle witness freshness is invalid")
    directory = _absolute_without_symlinks(_path(profile_dir))
    profile = load_local_profile(
        repository_cid=repository_cid,
        profile_dir=directory,
        lifecycle_dir=lifecycle_dir,
    )
    anchor_path, _ = _anchor_path(directory, lifecycle_dir)
    anchor = _load_anchor(anchor_path)
    registry = _profile_registry_record(directory)
    did_state = _load_did_state(profile.identity_did)
    profile_signature = _read_private_file(
        directory / SIGNATURE_FILENAME,
        maximum_bytes=_MAX_SIGNATURE_BYTES,
    ).decode("ascii").strip()
    root_key = _lifecycle_root_key(create=False)
    profile_key = _key(directory, None, create=False)
    body: dict[str, Any] = {
        "schema": LIFECYCLE_WITNESS_SCHEMA,
        "board_namespace": namespace,
        "base_head": base_head,
        "base_tree": base_tree,
        "observed_at_ms": observed,
        "expires_at_ms": expires,
        "nonce": witness_nonce,
        "profile": profile.to_dict(),
        "profile_content_id": profile.content_id,
        "profile_signature": profile_signature,
        "anchor": anchor,
        "anchor_digest": "sha256:"
        + hashlib.sha256(_canonical(anchor)).hexdigest(),
        "registry": registry,
        "did_state": did_state,
        "did_state_digest": "sha256:"
        + hashlib.sha256(_canonical(did_state)).hexdigest(),
        "root_identity_did": _did(root_key),
    }
    witness: dict[str, Any] = {
        **body,
        "active_key_signature": _signature(profile_key, body),
    }
    witness["root_signature"] = _signature(root_key, witness)
    witness["witness_id"] = "sha256:" + hashlib.sha256(
        _canonical(witness)
    ).hexdigest()
    if len(_canonical(witness)) > _MAX_LIFECYCLE_WITNESS_BYTES:
        raise LocalProfileTampered("lifecycle witness is oversized")
    verify_local_profile_lifecycle_witness(
        witness,
        expected_board_namespace=namespace,
        expected_base_head=base_head,
        expected_base_tree=base_tree,
        expected_nonce=witness_nonce,
        expected_root_identity_did=_did(root_key),
        reference_time_ms=observed,
        max_age_ms=expires - observed,
    )
    return witness


def verify_local_profile_lifecycle_witness(
    witness: Mapping[str, Any],
    *,
    expected_board_namespace: str,
    expected_base_head: str,
    expected_base_tree: str,
    expected_nonce: str,
    expected_root_identity_did: str,
    reference_time_ms: int,
    max_age_ms: int = _MAX_LIFECYCLE_WITNESS_AGE_MS,
) -> SignedSupervisorProfile:
    """Verify a portable witness at a runtime or historical commit time."""

    expected = {
        "schema",
        "board_namespace",
        "base_head",
        "base_tree",
        "observed_at_ms",
        "expires_at_ms",
        "nonce",
        "profile",
        "profile_content_id",
        "profile_signature",
        "anchor",
        "anchor_digest",
        "registry",
        "did_state",
        "did_state_digest",
        "root_identity_did",
        "active_key_signature",
        "root_signature",
        "witness_id",
    }
    if (
        not isinstance(witness, Mapping)
        or set(witness) != expected
        or len(_canonical(witness)) > _MAX_LIFECYCLE_WITNESS_BYTES
    ):
        raise LocalProfileTampered("lifecycle witness fields are invalid")
    text_names = expected - {
        "observed_at_ms",
        "expires_at_ms",
        "profile",
        "anchor",
        "registry",
        "did_state",
    }
    observed = witness.get("observed_at_ms")
    expires = witness.get("expires_at_ms")
    if (
        any(not isinstance(witness.get(name), str) for name in text_names)
        or witness.get("schema") != LIFECYCLE_WITNESS_SCHEMA
        or witness.get("board_namespace") != expected_board_namespace
        or witness.get("base_head") != expected_base_head
        or witness.get("base_tree") != expected_base_tree
        or witness.get("nonce") != expected_nonce
        or witness.get("root_identity_did") != expected_root_identity_did
        or isinstance(reference_time_ms, bool)
        or not isinstance(reference_time_ms, int)
        or reference_time_ms <= 0
        or isinstance(max_age_ms, bool)
        or not isinstance(max_age_ms, int)
        or not 0 < max_age_ms <= _MAX_LIFECYCLE_WITNESS_AGE_MS
        or isinstance(observed, bool)
        or not isinstance(observed, int)
        or observed <= 0
        or isinstance(expires, bool)
        or not isinstance(expires, int)
        or not observed <= reference_time_ms <= expires
        or not observed < expires
        or expires - observed > max_age_ms
    ):
        raise LocalProfileTampered("lifecycle witness binding is invalid")
    profile_raw = witness.get("profile")
    anchor_raw = witness.get("anchor")
    registry = witness.get("registry")
    did_state_raw = witness.get("did_state")
    if (
        not isinstance(profile_raw, dict)
        or not isinstance(anchor_raw, dict)
        or not isinstance(registry, dict)
        or not isinstance(did_state_raw, dict)
    ):
        raise LocalProfileTampered("lifecycle witness objects are invalid")
    profile = _profile_from_raw(profile_raw)
    verify_did_key_signature(
        identity_did=profile.identity_did,
        payload=profile_raw,
        signature=str(witness.get("profile_signature") or ""),
    )
    if witness.get("profile_content_id") != profile.content_id:
        raise LocalProfileTampered("lifecycle witness profile identity drifted")
    anchor = _validate_lifecycle_anchor(
        anchor_raw,
        expected_root_identity_did=expected_root_identity_did,
    )
    if witness.get("anchor_digest") != "sha256:" + hashlib.sha256(
        _canonical(anchor)
    ).hexdigest():
        raise LocalProfileTampered("lifecycle witness anchor digest drifted")
    did_state = _validate_did_state(
        did_state_raw,
        expected_root_identity_did=expected_root_identity_did,
    )
    if witness.get("did_state_digest") != "sha256:" + hashlib.sha256(
        _canonical(did_state)
    ).hexdigest():
        raise LocalProfileTampered("lifecycle witness DID state drifted")
    registry_expected = {
        "schema",
        "profile_path",
        "lifecycle_root",
        "root_identity_did",
        "registry_id",
    }
    if (
        set(registry) != registry_expected
        or any(
            not isinstance(registry.get(name), str)
            for name in registry_expected
        )
        or registry.get("schema") != _LIFECYCLE_ROOT_REGISTRY_SCHEMA
        or registry.get("root_identity_did") != expected_root_identity_did
        or registry.get("profile_path") == ""
        or registry.get("lifecycle_root") != profile.lifecycle_root_path
        or anchor.get("anchor_id")
        != hashlib.sha256(
            str(registry.get("profile_path") or "").encode("utf-8")
        ).hexdigest()
        or registry.get("registry_id")
        != "sha256:"
        + hashlib.sha256(
            _canonical(
                {
                    key: item
                    for key, item in registry.items()
                    if key != "registry_id"
                }
            )
        ).hexdigest()
        or anchor.get("status") != "active"
        or anchor.get("did_status") != "active"
        or anchor.get("did_state_id") != did_state.get("state_id")
        or did_state.get("status") != "active"
        or did_state.get("identity_did") != profile.identity_did
        or did_state.get("profile_path") != registry.get("profile_path")
        or did_state.get("profile_id") != profile.profile_id
        or did_state.get("profile_content_id") != profile.content_id
        or did_state.get("anchor_id") != profile.lifecycle_anchor_id
        or did_state.get("generation") != profile.lifecycle_generation
        or anchor.get("repository_cid") != profile.repository_cid
        or anchor.get("profile_id") != profile.profile_id
        or anchor.get("profile_content_id") != profile.content_id
        or anchor.get("identity_did") != profile.identity_did
        or anchor.get("generation") != profile.lifecycle_generation
        or anchor.get("anchor_id") != profile.lifecycle_anchor_id
    ):
        raise LocalProfileTampered("lifecycle witness cross-binding is invalid")
    body = {
        key: item
        for key, item in witness.items()
        if key not in {"active_key_signature", "root_signature", "witness_id"}
    }
    verify_did_key_signature(
        identity_did=profile.identity_did,
        payload=body,
        signature=str(witness.get("active_key_signature") or ""),
    )
    root_payload = {
        **body,
        "active_key_signature": witness.get("active_key_signature"),
    }
    verify_did_key_signature(
        identity_did=expected_root_identity_did,
        payload=root_payload,
        signature=str(witness.get("root_signature") or ""),
    )
    if witness.get("witness_id") != "sha256:" + hashlib.sha256(
        _canonical(
            {
                key: item
                for key, item in witness.items()
                if key != "witness_id"
            }
        )
    ).hexdigest():
        raise LocalProfileTampered("lifecycle witness identity is invalid")
    return profile


class SecureLocalIdentityStore:
    """Explicit façade used by effect boundaries and operator tooling."""

    initialize = staticmethod(lambda **kwargs: initialize_local_profile(**kwargs))
    load = staticmethod(lambda **kwargs: load_local_profile(**kwargs))
    revoke = staticmethod(lambda **kwargs: revoke_local_profile(**kwargs))
    rotate = staticmethod(lambda **kwargs: rotate_local_profile(**kwargs))


class LocalProfileInitializer:
    initialize = staticmethod(initialize_local_profile)
    load = staticmethod(load_local_profile)
    verify = staticmethod(load_local_profile)
    revoke = staticmethod(revoke_local_profile)
    rotate = staticmethod(rotate_local_profile)


def assert_capability_allowed(
    profile: SignedSupervisorProfile, capability: str
) -> None:
    if not profile.allows(capability):
        raise LocalProfileDenied(f"capability {capability!r} is denied by local profile")


def local_profile_authority_view(
    profile: SignedSupervisorProfile,
) -> dict[str, Any]:
    return {
        "kind": "local_dev_profile",
        "profile_id": profile.profile_id,
        "identity_did": profile.identity_did,
        "repository_cid": profile.repository_cid,
        "baseline_commit": profile.baseline_commit,
        "capabilities": sorted(profile.capabilities),
        "effect_bounds": list(profile.effect_bounds),
        "budget_cid": profile.budget_cid,
        "resource_cid": profile.resource_cid,
        "route_id": profile.route_id,
        "reviewer_identity": profile.reviewer_identity,
        "reviewer_provider": profile.reviewer_provider,
        "fallback_provider_id": profile.fallback_provider_id,
        "fallback_model_id": profile.fallback_model_id,
        "fallback_reasoning_effort": profile.fallback_reasoning_effort,
        "lifecycle_generation": profile.lifecycle_generation,
        "lifecycle_anchor_id": profile.lifecycle_anchor_id,
        "denied": sorted(DENIED_LOCAL_CAPABILITIES),
        "completion_authoritative": False,
        "proof_authoritative": False,
        "repository_write_allowed": False,
        "isolated_worktree_only": True,
        "current_checkout_rewrite": False,
    }


def inspect_local_profile(
    *,
    repository_cid: str,
    profile_dir: Path | None = None,
    lifecycle_dir: Path | None = None,
    signing_key: bytes | None = None,
) -> dict[str, Any]:
    return local_profile_authority_view(
        load_local_profile(
            repository_cid=repository_cid,
            profile_dir=profile_dir,
            lifecycle_dir=lifecycle_dir,
            signing_key=signing_key,
        )
    )
