"""Durable, cross-process, once-only provider-effect reservations.

This module intentionally owns no routing or retry policy.  It records only a
router-authorized logical attempt and performs the final compare-and-swap that
separates "safe to start the Docker effect" from "adopt existing state".
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

CAS_SCHEMA = "ipfs_accelerate_py/agent-supervisor/provider-attempt-cas@5"
EFFECT_LAUNCH_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/provider-effect-launch@2"
)
_LEGACY_EFFECT_LAUNCH_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/provider-effect-launch@1"
)
EFFECT_ADOPTION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/provider-effect-adoption@1"
)
_STATES = frozenset({"reserved", "effect_started", "quarantined", "terminal"})
_MAX_RESERVATION_BYTES = 768 * 1024
_MAX_DOCKER_INSPECTION_BYTES = 256 * 1024
_MAX_AUTHORIZATION_CONTEXT_BYTES = 512 * 1024
_DOCKER_LOCAL_HOST = "unix:///var/run/docker.sock"
_CODEX_CONTAINER_NAME_RE = re.compile(
    r"ipfs-accelerate-codex-[0-9]+-[0-9a-f]{32}"
)
_PROCESS_EFFECT_OWNER_ID = "sha256:" + hashlib.sha256(
    f"{os.getpid()}\0{time.time_ns()}\0{secrets.token_hex(32)}".encode("utf-8")
).hexdigest()


def _process_start_ticks(pid: int) -> int:
    try:
        descriptor = os.open(
            f"/proc/{pid}/stat",
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            raw = os.read(descriptor, 4097)
        finally:
            os.close(descriptor)
        if len(raw) > 4096:
            raise ValueError("process stat is oversized")
        fields = raw.decode("ascii").split()
        value = int(fields[21])
    except (OSError, UnicodeError, ValueError, IndexError) as exc:
        raise ProviderAttemptStoreError(
            "effect owner process birth identity is unavailable"
        ) from exc
    if value < 0:
        raise ProviderAttemptStoreError("effect owner process birth is invalid")
    return value


class ProviderAttemptStoreError(ValueError):
    pass


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _token(value: object, name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized or any(character in normalized for character in "\0\n\r"):
        raise ProviderAttemptStoreError(f"{name} must be nonempty")
    return normalized


def _timestamp(value: int | None, name: str) -> int:
    timestamp = int(time.time() * 1000) if value is None else value
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, int)
        or timestamp < 0
    ):
        raise ProviderAttemptStoreError(f"{name} is invalid")
    return timestamp


def _absolute_without_symlinks(path: Path) -> Path:
    """Preserve and reject symlink components before lexical normalization."""

    expanded = path.expanduser()
    candidate = expanded if expanded.is_absolute() else Path.cwd() / expanded
    cursor = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        if component in {"", "."}:
            continue
        if component == "..":
            raise ProviderAttemptStoreError(
                "attempt reservation path cannot contain parent traversal"
            )
        cursor /= component
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ProviderAttemptStoreError(
                "attempt reservation path component is unavailable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ProviderAttemptStoreError(
                "attempt reservation path cannot contain symlink components"
            )
    return Path(os.path.abspath(os.fspath(candidate)))


def _entry_exists(path: Path) -> bool:
    secured = _absolute_without_symlinks(path)
    try:
        os.lstat(secured)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ProviderAttemptStoreError(
            "attempt reservation is unavailable"
        ) from exc
    return True


def _create_directory_chain(path: Path) -> tuple[int, int, int, int]:
    """Create/open an absolute directory chain through no-follow dirfds."""

    secured = _absolute_without_symlinks(path)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ProviderAttemptStoreError(
            "no-follow attempt reservation creation is unavailable"
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
                    # A concurrent creator is acceptable only if the exact
                    # entry can now be opened as a no-follow directory.
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
        raise ProviderAttemptStoreError(
            "attempt reservation directory cannot be created"
        ) from exc
    finally:
        os.close(descriptor)


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
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise ProviderAttemptStoreError(
            "attempt reservation is not an owned private regular file"
        )


def _read_private_file(path: Path, *, maximum_bytes: int) -> bytes:
    secured = _absolute_without_symlinks(path)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ProviderAttemptStoreError(
            "no-follow attempt reservation reads are unavailable"
        )
    try:
        descriptor = os.open(
            secured,
            os.O_RDONLY
            | nofollow
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
    except OSError as exc:
        raise ProviderAttemptStoreError(
            "attempt reservation is unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        _validate_private_regular(before)
        if before.st_size > maximum_bytes:
            raise ProviderAttemptStoreError("attempt reservation is oversized")

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
            raise ProviderAttemptStoreError("attempt reservation is oversized")

        after = os.fstat(descriptor)
        _validate_private_regular(after)
        final_path = os.lstat(secured)
        if (
            _regular_snapshot(before) != _regular_snapshot(after)
            or _regular_snapshot(after) != _regular_snapshot(final_path)
            or len(payload) != after.st_size
        ):
            raise ProviderAttemptStoreError(
                "attempt reservation changed while being read"
            )
        return payload
    except OSError as exc:
        raise ProviderAttemptStoreError(
            "attempt reservation changed while being read"
        ) from exc
    finally:
        os.close(descriptor)


class _DuplicateJSONKey(ValueError):
    pass


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJSONKey(key)
        value[key] = item
    return value


def _owned_directory(
    path: Path,
    *,
    expected_identity: tuple[int, int, int, int] | None = None,
) -> str:
    secured = _absolute_without_symlinks(path)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ProviderAttemptStoreError(
            "no-follow attempt reservation checks are unavailable"
        )
    try:
        descriptor = os.open(
            secured,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | nofollow,
        )
    except OSError as exc:
        raise ProviderAttemptStoreError(
            "attempt reservation directory is unavailable"
        ) from exc
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
            raise ProviderAttemptStoreError(
                "attempt reservation directory must be owned mode 0700"
            )
    except OSError as exc:
        raise ProviderAttemptStoreError(
            "attempt reservation directory changed while being checked"
        ) from exc
    finally:
        os.close(descriptor)
    body = {
        "path": str(secured),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "uid": metadata.st_uid,
    }
    return "sha256:" + hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _fsync_directory(path: Path) -> None:
    secured = _absolute_without_symlinks(path)
    descriptor = os.open(
        secured,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class ProviderAttemptReservation:
    logical_attempt_id: str
    route_id: str
    decision_id: str
    task_id: str
    worktree_id: str
    reservation_id: str
    state: str
    created_at_ms: int
    authorization_context: Mapping[str, Any] = field(default_factory=dict)
    effect_started_at_ms: int | None = None
    effect_launch_receipt: Mapping[str, Any] = field(default_factory=dict)
    effect_adoption_generation: int = 0
    effect_adoption_receipt: Mapping[str, Any] = field(default_factory=dict)
    completion_capability_sha256: str = ""
    quarantine_at_ms: int | None = None
    quarantine_receipt: Mapping[str, Any] = field(default_factory=dict)
    quarantine_terminalization_receipt: Mapping[str, Any] = field(
        default_factory=dict
    )
    terminal_at_ms: int | None = None
    terminal_returncode: int | None = None
    terminal_outcome_id: str = ""
    terminal_outcome: Mapping[str, Any] = field(default_factory=dict)
    schema: str = CAS_SCHEMA

    @property
    def content_id(self) -> str:
        return "sha256:" + hashlib.sha256(_canonical(asdict(self))).hexdigest()

    @property
    def effect_may_start(self) -> bool:
        return self.state == "reserved"

    @property
    def effect_already_started(self) -> bool:
        return self.state in {"effect_started", "quarantined", "terminal"}

    @property
    def terminal(self) -> bool:
        return self.state == "terminal"


@dataclass(frozen=True)
class ProviderAttemptCASResult:
    reservation: ProviderAttemptReservation
    created: bool
    adopted: bool
    launch_authorized: bool = False
    adoption_authorized: bool = False
    effect_launch_receipt: Mapping[str, Any] = field(default_factory=dict)
    completion_capability: str = field(default="", repr=False, compare=False)


def _valid_effect_launch_receipt(
    reservation: ProviderAttemptReservation,
) -> bool:
    receipt = reservation.effect_launch_receipt
    common = {
        "schema",
        "logical_attempt_id",
        "reservation_id",
        "effect_owner_id",
        "effect_owner_pid",
        "effect_owner_start_ticks",
        "provider_id",
        "command_id",
        "runtime_id",
        "image_id",
        "mount_id",
        "environment_id",
        "container_name",
        "container_id",
        "claimed_at_ms",
        "receipt_id",
    }
    protected_details = {
        "cleanup_id",
        "runtime_receipt",
        "image_receipt",
        "command_receipt",
        "mount_receipt",
        "environment_receipt",
        "cleanup_receipt",
    }
    if (
        not isinstance(receipt, Mapping)
        or frozenset(receipt)
        not in {frozenset(common), frozenset(common | protected_details)}
    ):
        return False
    owner_pid = receipt.get("effect_owner_pid")
    owner_start = receipt.get("effect_owner_start_ticks")
    claimed = receipt.get("claimed_at_ms")
    return bool(
        receipt.get("schema")
        == (
            EFFECT_LAUNCH_SCHEMA
            if protected_details.issubset(receipt)
            else _LEGACY_EFFECT_LAUNCH_SCHEMA
        )
        and receipt.get("logical_attempt_id") == reservation.logical_attempt_id
        and receipt.get("reservation_id") == reservation.reservation_id
        and isinstance(receipt.get("effect_owner_id"), str)
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(receipt.get("effect_owner_id") or ""),
        )
        and isinstance(owner_pid, int)
        and not isinstance(owner_pid, bool)
        and owner_pid > 0
        and isinstance(owner_start, int)
        and not isinstance(owner_start, bool)
        and owner_start >= 0
        and isinstance(claimed, int)
        and not isinstance(claimed, bool)
        and claimed == reservation.effect_started_at_ms
        and _token(receipt.get("provider_id"), "provider_id")
        == receipt.get("provider_id")
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(receipt.get("command_id") or ""),
        )
        and all(
            re.fullmatch(r"sha256:[0-9a-f]{64}", str(receipt.get(name) or ""))
            for name in ("runtime_id", "image_id", "mount_id", "environment_id")
        )
        and (
            not protected_details.issubset(receipt)
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(receipt.get("cleanup_id") or ""),
            )
            is not None
        )
        and _token(receipt.get("container_name"), "container_name")
        == receipt.get("container_name")
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(receipt.get("container_id") or ""),
        )
        is not None
        and receipt.get("receipt_id")
        == "sha256:"
        + hashlib.sha256(
            _canonical(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "receipt_id"
                }
            )
        ).hexdigest()
        and (
            not protected_details.issubset(receipt)
            or _valid_effect_launch_details(receipt)
        )
    )


def _valid_effect_launch_details(receipt: Mapping[str, Any]) -> bool:
    """Validate the exact argv/runtime/image/mount/environment receipt."""

    runtime = receipt.get("runtime_receipt")
    image = receipt.get("image_receipt")
    command = receipt.get("command_receipt")
    mounts = receipt.get("mount_receipt")
    environment = receipt.get("environment_receipt")
    cleanup = receipt.get("cleanup_receipt")
    if (
        not isinstance(runtime, Mapping)
        or set(runtime)
        != {
            "path",
            "device",
            "inode",
            "mode",
            "uid",
            "size",
            "mtime_ns",
            "ctime_ns",
        }
        or any(
            isinstance(runtime.get(name), bool)
            or not isinstance(runtime.get(name), int)
            or int(runtime.get(name) or 0) < 0
            for name in (
                "device",
                "inode",
                "mode",
                "uid",
                "size",
                "mtime_ns",
                "ctime_ns",
            )
        )
        or not isinstance(runtime.get("path"), str)
        or not isinstance(image, Mapping)
        or set(image) != {"image_id", "image_label"}
        or image.get("image_id") != receipt.get("image_id")
        or not isinstance(image.get("image_label"), str)
        or not image.get("image_label")
        or not isinstance(command, Mapping)
        or set(command) != {"create_argv", "start_argv", "provider_argv"}
        or not isinstance(mounts, list)
        or not mounts
        or not isinstance(environment, Mapping)
        or set(environment) != {"docker_cli", "container"}
        or not isinstance(cleanup, Mapping)
        or cleanup.get("receipt_id") != receipt.get("cleanup_id")
        or receipt.get("cleanup_id")
        != _effect_identity(
            {key: value for key, value in cleanup.items() if key != "receipt_id"}
        )
    ):
        return False
    for name in ("create_argv", "start_argv", "provider_argv"):
        argv = command.get(name)
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(item, str) for item in argv)
        ):
            return False
    if any(not isinstance(item, str) or not item for item in mounts):
        return False
    for name in ("docker_cli", "container"):
        values = environment.get(name)
        if (
            not isinstance(values, Mapping)
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in values.items()
            )
        ):
            return False
    identities_valid = bool(
        receipt.get("runtime_id") == _effect_identity(runtime)
        and receipt.get("command_id") == _effect_identity(command)
        and receipt.get("mount_id") == _effect_identity(mounts)
        and receipt.get("environment_id") == _effect_identity(environment)
    )
    if not identities_valid:
        return False
    try:
        from ipfs_accelerate_py.llm_router import (
            _agent_effect_launch_details_valid,
        )

        return bool(_agent_effect_launch_details_valid(receipt))
    except (ImportError, TypeError, ValueError):
        return False


def _valid_effect_adoption_receipt(
    reservation: ProviderAttemptReservation,
) -> bool:
    receipt = reservation.effect_adoption_receipt
    if reservation.effect_adoption_generation == 0:
        return not receipt
    expected = {
        "schema",
        "logical_attempt_id",
        "reservation_id",
        "adoption_generation",
        "previous_receipt_id",
        "previous_owner_id",
        "previous_owner_pid",
        "previous_owner_start_ticks",
        "effect_owner_id",
        "effect_owner_pid",
        "effect_owner_start_ticks",
        "transition_kind",
        "inspection_status",
        "inspection_runtime_id",
        "inspection_command_id",
        "inspection_observed_at_ms",
        "provider_id",
        "command_id",
        "runtime_id",
        "image_id",
        "mount_id",
        "environment_id",
        "container_name",
        "container_id",
        "container_returncode",
        "inspected_at_ms",
        "prior_adoption_receipts",
        "receipt_id",
    }

    def valid_one(item: object, generation: int) -> bool:
        if not isinstance(item, Mapping) or set(item) != expected:
            return False
        prior = item.get("prior_adoption_receipts")
        if (
            not isinstance(prior, list)
            or generation < 1
            or generation > 8
            or len(prior) != generation - 1
            or item.get("adoption_generation") != generation
        ):
            return False
        if generation > 1:
            if not valid_one(prior[-1], generation - 1):
                return False
            prior_owner = prior[-1]
        else:
            prior_owner = reservation.effect_launch_receipt
        if generation > 1 and prior[:-1] != prior_owner.get(
            "prior_adoption_receipts"
        ):
            return False
        integer_names = (
            "adoption_generation",
            "previous_owner_pid",
            "previous_owner_start_ticks",
            "effect_owner_pid",
            "effect_owner_start_ticks",
            "inspection_observed_at_ms",
            "inspected_at_ms",
        )
        returncode = item.get("container_returncode")
        status_value = item.get("inspection_status")
        container_id = item.get("container_id")
        launch = reservation.effect_launch_receipt
        return bool(
            all(
                isinstance(item.get(name), int)
                and not isinstance(item.get(name), bool)
                and int(item.get(name) or 0) >= 0
                for name in integer_names
            )
            and item.get("schema") == EFFECT_ADOPTION_SCHEMA
            and item.get("logical_attempt_id") == reservation.logical_attempt_id
            and item.get("reservation_id") == reservation.reservation_id
            and int(item.get("effect_owner_pid") or 0) > 0
            and item.get("transition_kind")
            in {"dead_owner_adoption", "winner_reconciliation"}
            and (
                (
                    item.get("transition_kind") == "winner_reconciliation"
                    and item.get("previous_owner_id")
                    == item.get("effect_owner_id")
                    and item.get("previous_owner_pid")
                    == item.get("effect_owner_pid")
                    and item.get("previous_owner_start_ticks")
                    == item.get("effect_owner_start_ticks")
                )
                or (
                    item.get("transition_kind") == "dead_owner_adoption"
                    and item.get("previous_owner_id")
                    != item.get("effect_owner_id")
                )
            )
            and item.get("previous_receipt_id") == prior_owner.get("receipt_id")
            and item.get("previous_owner_id")
            == prior_owner.get("effect_owner_id")
            and item.get("previous_owner_pid")
            == prior_owner.get("effect_owner_pid")
            and item.get("previous_owner_start_ticks")
            == prior_owner.get("effect_owner_start_ticks")
            and status_value in {"created", "running", "exited", "absent"}
            and item.get("inspection_runtime_id") == launch.get("runtime_id")
            and re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(item.get("inspection_command_id") or ""),
            )
            is not None
            and item.get("inspection_observed_at_ms")
            == item.get("inspected_at_ms")
            and int(item.get("inspected_at_ms") or 0)
            >= int(reservation.effect_started_at_ms or 0)
            and (
                (
                    status_value == "exited"
                    and isinstance(returncode, int)
                    and not isinstance(returncode, bool)
                )
                or (status_value != "exited" and returncode is None)
            )
            and (
                (
                    status_value in {"created", "running", "exited"}
                    and re.fullmatch(
                        r"sha256:[0-9a-f]{64}", str(container_id or "")
                    )
                    is not None
                    and container_id == launch.get("container_id")
                )
                or (status_value == "absent" and container_id == "")
            )
            and item.get("provider_id") == launch.get("provider_id")
            and all(
                item.get(name) == launch.get(name)
                for name in (
                    "command_id",
                    "runtime_id",
                    "image_id",
                    "mount_id",
                    "environment_id",
                    "container_name",
                )
            )
            and item.get("receipt_id")
            == "sha256:"
            + hashlib.sha256(
                _canonical(
                    {
                        key: value
                        for key, value in item.items()
                        if key != "receipt_id"
                    }
                )
            ).hexdigest()
        )

    return valid_one(receipt, reservation.effect_adoption_generation)


def _valid_effect_quarantine_receipt(
    reservation: ProviderAttemptReservation,
) -> bool:
    receipt = reservation.quarantine_receipt
    launch = reservation.effect_launch_receipt
    expected = {
        "schema",
        "logical_attempt_id",
        "reservation_id",
        "adoption_generation",
        "reason",
        "required_operator_action",
        "provider_id",
        "runtime_id",
        "container_name",
        "container_id",
        "inspection_status",
        "inspection_command_id",
        "container_returncode",
        "quarantined_at_ms",
        "incident_id",
    }
    status_value = receipt.get("inspection_status")
    return bool(
        isinstance(receipt, Mapping)
        and set(receipt) == expected
        and receipt.get("schema")
        == "ipfs_accelerate_py/agent-supervisor/provider-effect-quarantine@1"
        and receipt.get("logical_attempt_id") == reservation.logical_attempt_id
        and receipt.get("reservation_id") == reservation.reservation_id
        and receipt.get("adoption_generation")
        == reservation.effect_adoption_generation
        and receipt.get("reason") == "adoption_transfer_limit_exhausted"
        and receipt.get("required_operator_action")
        == "inspect_exact_container_and_terminalize_without_relaunch"
        and receipt.get("provider_id") == launch.get("provider_id") == "codex"
        and receipt.get("runtime_id") == launch.get("runtime_id")
        and receipt.get("container_name") == launch.get("container_name")
        and status_value in {"created", "running", "exited", "absent"}
        and (
            (status_value == "absent" and receipt.get("container_id") == "")
            or (
                status_value != "absent"
                and receipt.get("container_id") == launch.get("container_id")
            )
        )
        and (
            (
                status_value == "exited"
                and isinstance(receipt.get("container_returncode"), int)
                and not isinstance(receipt.get("container_returncode"), bool)
            )
            or (
                status_value != "exited"
                and receipt.get("container_returncode") is None
            )
        )
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(receipt.get("inspection_command_id") or ""),
        )
        is not None
        and receipt.get("quarantined_at_ms") == reservation.quarantine_at_ms
        and receipt.get("incident_id")
        == "sha256:"
        + hashlib.sha256(
            _canonical(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "incident_id"
                }
            )
        ).hexdigest()
    )


def _valid_effect_quarantine_terminalization_receipt(
    reservation: ProviderAttemptReservation,
) -> bool:
    """Validate the sealed no-relaunch repair claim for a quarantine."""

    receipt = reservation.quarantine_terminalization_receipt
    launch = reservation.effect_launch_receipt
    quarantine = reservation.quarantine_receipt
    expected = {
        "schema",
        "logical_attempt_id",
        "reservation_id",
        "incident_id",
        "repair_generation",
        "previous_repair_receipt_id",
        "prior_repair_receipts",
        "operator_action",
        "effect_owner_id",
        "effect_owner_pid",
        "effect_owner_start_ticks",
        "provider_id",
        "command_id",
        "runtime_id",
        "image_id",
        "mount_id",
        "environment_id",
        "container_name",
        "container_id",
        "inspection_status",
        "inspection_command_id",
        "inspected_at_ms",
        "container_returncode",
        "terminal_returncode",
        "outcome_decision",
        "fallback_dispatched",
        "receipt_id",
    }
    status_value = receipt.get("inspection_status")
    repair_generation = receipt.get("repair_generation")
    prior_repairs = receipt.get("prior_repair_receipts")
    inspected_at_ms = receipt.get("inspected_at_ms")
    terminal_returncode = receipt.get("terminal_returncode")
    owner_pid = receipt.get("effect_owner_pid")
    owner_start = receipt.get("effect_owner_start_ticks")
    expected_decision = (
        "effect_not_created"
        if status_value == "absent"
        else (
            "fallback_succeeded"
            if receipt.get("container_returncode") == 0
            else "fallback_failed"
        )
    )
    return bool(
        isinstance(receipt, Mapping)
        and set(receipt) == expected
        and receipt.get("schema")
        == (
            "ipfs_accelerate_py/agent-supervisor/"
            "provider-effect-quarantine-terminalization@1"
        )
        and receipt.get("logical_attempt_id") == reservation.logical_attempt_id
        and receipt.get("reservation_id") == reservation.reservation_id
        and receipt.get("incident_id") == quarantine.get("incident_id")
        and isinstance(repair_generation, int)
        and not isinstance(repair_generation, bool)
        and repair_generation >= 1
        and isinstance(prior_repairs, list)
        and len(prior_repairs) == repair_generation - 1
        and (
            (
                repair_generation == 1
                and receipt.get("previous_repair_receipt_id") == ""
                and not prior_repairs
            )
            or (
                repair_generation > 1
                and re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(receipt.get("previous_repair_receipt_id") or ""),
                )
                is not None
                and isinstance(prior_repairs[-1], Mapping)
                and receipt.get("previous_repair_receipt_id")
                == prior_repairs[-1].get("receipt_id")
                and prior_repairs[:-1]
                == prior_repairs[-1].get("prior_repair_receipts")
                and _valid_effect_quarantine_terminalization_receipt(
                    replace(
                        reservation,
                        quarantine_terminalization_receipt=prior_repairs[-1],
                    )
                )
            )
        )
        and receipt.get("operator_action")
        == "terminalize_exact_quarantined_effect_without_relaunch"
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(receipt.get("effect_owner_id") or ""),
        )
        is not None
        and isinstance(owner_pid, int)
        and not isinstance(owner_pid, bool)
        and owner_pid > 0
        and isinstance(owner_start, int)
        and not isinstance(owner_start, bool)
        and owner_start >= 0
        and status_value in {"absent", "exited"}
        and isinstance(inspected_at_ms, int)
        and not isinstance(inspected_at_ms, bool)
        and inspected_at_ms >= int(reservation.quarantine_at_ms or 0)
        and all(
            receipt.get(name) == launch.get(name)
            for name in (
                "provider_id",
                "command_id",
                "runtime_id",
                "image_id",
                "mount_id",
                "environment_id",
                "container_name",
            )
        )
        and (
            (
                status_value == "absent"
                and receipt.get("container_id") == ""
                and receipt.get("container_returncode") is None
                and terminal_returncode == 125
                and receipt.get("fallback_dispatched") is False
            )
            or (
                status_value == "exited"
                and receipt.get("container_id") == launch.get("container_id")
                and isinstance(receipt.get("container_returncode"), int)
                and not isinstance(receipt.get("container_returncode"), bool)
                and terminal_returncode == receipt.get("container_returncode")
                and receipt.get("fallback_dispatched") is True
            )
        )
        and receipt.get("outcome_decision") == expected_decision
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(receipt.get("inspection_command_id") or ""),
        )
        is not None
        and receipt.get("receipt_id")
        == "sha256:"
        + hashlib.sha256(
            _canonical(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "receipt_id"
                }
            )
        ).hexdigest()
    )


def _effective_effect_owner(
    reservation: ProviderAttemptReservation,
) -> Mapping[str, Any]:
    return (
        reservation.effect_adoption_receipt
        if reservation.effect_adoption_generation > 0
        else reservation.effect_launch_receipt
    )


def _process_identity_alive(pid: int, start_ticks: int) -> bool:
    try:
        return _process_start_ticks(pid) == start_ticks
    except ProviderAttemptStoreError:
        return False


def _effect_identity(value: object) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _native_docker_runtime(runtime_id: object) -> str:
    """Resolve only the root-owned Docker binary named by a launch receipt."""

    if not isinstance(runtime_id, str) or re.fullmatch(
        r"sha256:[0-9a-f]{64}", runtime_id
    ) is None:
        raise ProviderAttemptStoreError("recorded Docker runtime is invalid")
    for candidate in (Path("/usr/bin/docker"), Path("/usr/local/bin/docker")):
        try:
            unresolved = candidate.lstat()
            resolved = candidate.resolve(strict=True)
            metadata = resolved.stat()
        except OSError:
            continue
        if (
            not stat.S_ISREG(unresolved.st_mode)
            or candidate != resolved
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_mode & 0o022
            or not os.access(resolved, os.X_OK)
        ):
            continue
        observed_id = _effect_identity(
            {
                "path": str(resolved),
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
                "mode": metadata.st_mode,
                "uid": metadata.st_uid,
                "size": metadata.st_size,
                "mtime_ns": metadata.st_mtime_ns,
                "ctime_ns": metadata.st_ctime_ns,
            }
        )
        if observed_id == runtime_id:
            return str(resolved)
    raise ProviderAttemptStoreError("recorded Docker runtime identity drifted")


def _docker_control_environment() -> dict[str, str]:
    return {"HOME": "/nonexistent", "PATH": "/usr/bin:/bin"}


def _bounded_docker_query(
    command: list[str],
    *,
    timeout: float = 15.0,
) -> tuple[int, bytes, bytes]:
    try:
        completed = subprocess.run(
            command,
            env=_docker_control_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProviderAttemptStoreError(
            "recorded Docker effect inspection failed"
        ) from exc
    if (
        len(completed.stdout) > _MAX_DOCKER_INSPECTION_BYTES
        or len(completed.stderr) > _MAX_DOCKER_INSPECTION_BYTES
    ):
        raise ProviderAttemptStoreError(
            "recorded Docker effect inspection was oversized"
        )
    return int(completed.returncode), completed.stdout, completed.stderr


def _reject_duplicate_json_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for key, value in pairs:
        if key in decoded:
            raise ValueError("duplicate Docker inspection key")
        decoded[key] = value
    return decoded


def _inspect_recorded_docker_effect(
    launch: Mapping[str, Any],
    observed_at_ms: int,
) -> Mapping[str, Any]:
    """Inspect the recorded effect through fixed store-owned Docker commands.

    This function is deliberately private and receives no caller-provided
    command, callback, status, or result mapping.  The durable launch receipt
    is the only selector and every returned identity is recomputed here.
    """

    container_name = launch.get("container_name")
    recorded_container_id = launch.get("container_id")
    if (
        not isinstance(container_name, str)
        or _CODEX_CONTAINER_NAME_RE.fullmatch(container_name) is None
        or not isinstance(recorded_container_id, str)
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", recorded_container_id
        ) is None
        or isinstance(observed_at_ms, bool)
        or not isinstance(observed_at_ms, int)
        or observed_at_ms <= 0
    ):
        raise ProviderAttemptStoreError(
            "recorded Docker effect identity is invalid"
        )
    runtime_id = launch.get("runtime_id")
    docker = _native_docker_runtime(runtime_id)
    semantic_inspection = {
        "runtime_id": runtime_id,
        "host": _DOCKER_LOCAL_HOST,
        "operation": "container_inspect",
        "container_name": container_name,
        "container_id": recorded_container_id,
    }
    inspection_command_id = _effect_identity(semantic_inspection)
    with tempfile.TemporaryDirectory(
        prefix="asref-codex-adoption-docker-config-"
    ) as config_root:
        inspect_command = [
            docker,
            f"--host={_DOCKER_LOCAL_HOST}",
            "--config",
            config_root,
            "container",
            "inspect",
            recorded_container_id.removeprefix("sha256:"),
        ]
        returncode, stdout, _stderr = _bounded_docker_query(inspect_command)
        if returncode != 0:
            list_command = [
                docker,
                f"--host={_DOCKER_LOCAL_HOST}",
                "--config",
                config_root,
                "container",
                "ls",
                "--all",
                "--no-trunc",
                "--filter",
                f"name=^{container_name}$",
                "--format",
                "{{.ID}}",
            ]
            list_returncode, listed, _list_stderr = _bounded_docker_query(
                list_command
            )
            if list_returncode != 0 or listed.strip():
                raise ProviderAttemptStoreError(
                    "recorded Docker container could not be inspected"
                )
            status_value = "absent"
            container_id = ""
            container_returncode: int | None = None
        else:
            try:
                decoded = json.loads(
                    stdout.decode("utf-8"),
                    object_pairs_hook=_reject_duplicate_json_pairs,
                )
            except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
                raise ProviderAttemptStoreError(
                    "recorded Docker inspection is malformed"
                ) from exc
            if (
                not isinstance(decoded, list)
                or len(decoded) != 1
                or not isinstance(decoded[0], Mapping)
            ):
                raise ProviderAttemptStoreError(
                    "recorded Docker inspection is ambiguous"
                )
            record = decoded[0]
            state = record.get("State")
            raw_container_id = record.get("Id")
            if (
                record.get("Name") != "/" + container_name
                or record.get("Image") != launch.get("image_id")
                or not isinstance(state, Mapping)
                or not isinstance(raw_container_id, str)
                or re.fullmatch(r"[0-9a-f]{64}", raw_container_id) is None
            ):
                raise ProviderAttemptStoreError(
                    "recorded Docker container identity does not match"
                )
            container_id = "sha256:" + raw_container_id
            if container_id != launch.get("container_id"):
                raise ProviderAttemptStoreError(
                    "recorded Docker container identity drifted"
                )
            running = state.get("Running")
            if running is True:
                status_value = "running"
                container_returncode = None
            elif running is False and state.get("Status") == "created":
                status_value = "created"
                container_returncode = None
            elif running is False and state.get("Status") in {"exited", "dead"}:
                exit_code = state.get("ExitCode")
                if isinstance(exit_code, bool) or not isinstance(exit_code, int):
                    raise ProviderAttemptStoreError(
                        "recorded Docker exit status is invalid"
                    )
                status_value = "exited"
                container_returncode = exit_code
            else:
                raise ProviderAttemptStoreError(
                    "recorded Docker container is not adoptable"
                )
    return {
        "status": status_value,
        "inspection_runtime_id": runtime_id,
        "inspection_command_id": inspection_command_id,
        "observed_at_ms": observed_at_ms,
        "provider_id": launch.get("provider_id"),
        "command_id": launch.get("command_id"),
        "runtime_id": runtime_id,
        "image_id": launch.get("image_id"),
        "mount_id": launch.get("mount_id"),
        "environment_id": launch.get("environment_id"),
        "container_name": container_name,
        "container_id": container_id,
        "returncode": container_returncode,
    }


class DurableProviderAttemptCAS:
    """File-lock-backed compare-and-swap for one fallback provider effect."""

    def __init__(
        self,
        directory: Path | str,
        *,
        expected_directory_identity: str = "",
    ) -> None:
        self.directory = _absolute_without_symlinks(Path(directory))
        if _entry_exists(self.directory):
            observed_identity = _owned_directory(self.directory)
        else:
            created_identity = _create_directory_chain(self.directory)
            if _absolute_without_symlinks(self.directory) != self.directory:
                raise ProviderAttemptStoreError(
                    "attempt reservation directory changed during creation"
                )
            observed_identity = _owned_directory(
                self.directory,
                expected_identity=created_identity,
            )
        expected = str(expected_directory_identity or "").strip()
        if expected and observed_identity != expected:
            raise ProviderAttemptStoreError(
                "attempt reservation directory identity drifted"
            )
        self.directory_identity = observed_identity
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            self._directory_fd = os.open(self.directory, flags)
            bound = os.fstat(self._directory_fd)
        except OSError as exc:
            raise ProviderAttemptStoreError(
                "attempt reservation directory cannot be bound"
            ) from exc
        self._directory_metadata_identity = (
            bound.st_dev,
            bound.st_ino,
            bound.st_mode,
            bound.st_uid,
        )
        if (
            not stat.S_ISDIR(bound.st_mode)
            or bound.st_uid != os.geteuid()
            or stat.S_IMODE(bound.st_mode) != 0o700
        ):
            os.close(self._directory_fd)
            self._directory_fd = -1
            raise ProviderAttemptStoreError(
                "attempt reservation directory binding is invalid"
            )

    def __del__(self) -> None:
        descriptor = getattr(self, "_directory_fd", -1)
        if isinstance(descriptor, int) and descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
            self._directory_fd = -1

    def _path(self, logical_attempt_id: str) -> Path:
        safe = hashlib.sha256(
            _token(logical_attempt_id, "logical_attempt_id").encode("utf-8")
        ).hexdigest()
        return self.directory / (safe + ".json")

    def _lock(self, logical_attempt_id: str) -> tuple[int, Path]:
        observed_identity = _owned_directory(self.directory)
        if observed_identity != self.directory_identity:
            raise ProviderAttemptStoreError(
                "attempt reservation directory identity drifted"
            )
        try:
            bound = os.fstat(self._directory_fd)
        except OSError as exc:
            raise ProviderAttemptStoreError(
                "attempt reservation directory binding is unavailable"
            ) from exc
        if (
            bound.st_dev,
            bound.st_ino,
            bound.st_mode,
            bound.st_uid,
        ) != self._directory_metadata_identity:
            raise ProviderAttemptStoreError(
                "attempt reservation directory binding drifted"
            )
        path = self._path(logical_attempt_id)
        lock_name = path.with_suffix(".lock").name
        try:
            descriptor = os.open(
                lock_name,
                os.O_CREAT
                | os.O_RDWR
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=self._directory_fd,
            )
        except OSError as exc:
            raise ProviderAttemptStoreError(
                "attempt reservation lock is unavailable"
            ) from exc
        metadata = os.fstat(descriptor)
        try:
            final_path = os.stat(
                lock_name,
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            os.close(descriptor)
            raise ProviderAttemptStoreError(
                "attempt reservation lock is invalid"
            ) from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or _regular_snapshot(metadata) != _regular_snapshot(final_path)
        ):
            os.close(descriptor)
            raise ProviderAttemptStoreError("attempt reservation lock is invalid")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return descriptor, path

    @staticmethod
    def _unlock(descriptor: int) -> None:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    @staticmethod
    def _parse(value: Mapping[str, Any]) -> ProviderAttemptReservation:
        expected = set(asdict(ProviderAttemptReservation(
            logical_attempt_id="x", route_id="x", decision_id="x",
            task_id="x", worktree_id="x", reservation_id="x",
            state="reserved", created_at_ms=1,
        )))
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ProviderAttemptStoreError("attempt reservation fields are invalid")
        try:
            reservation = ProviderAttemptReservation(**dict(value))
        except (TypeError, ValueError) as exc:
            raise ProviderAttemptStoreError(
                "attempt reservation fields are invalid"
            ) from exc
        timestamp_fields = (
            reservation.created_at_ms,
            reservation.effect_started_at_ms,
            reservation.quarantine_at_ms,
            reservation.terminal_at_ms,
        )
        if (
            not isinstance(reservation.schema, str)
            or reservation.schema != CAS_SCHEMA
            or not isinstance(reservation.state, str)
            or reservation.state not in _STATES
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item < 0
                for item in timestamp_fields
                if item is not None
            )
            or (
                reservation.terminal_returncode is not None
                and (
                    isinstance(reservation.terminal_returncode, bool)
                    or not isinstance(reservation.terminal_returncode, int)
                )
            )
            or (
                reservation.effect_started_at_ms is not None
                and reservation.effect_started_at_ms
                < reservation.created_at_ms
            )
            or (
                reservation.terminal_at_ms is not None
                and (
                    reservation.effect_started_at_ms is None
                    or reservation.terminal_at_ms
                    < reservation.effect_started_at_ms
                )
            )
            or not isinstance(reservation.terminal_outcome_id, str)
            or not isinstance(reservation.terminal_outcome, Mapping)
            or not isinstance(reservation.authorization_context, Mapping)
            or len(_canonical(reservation.authorization_context))
            > _MAX_AUTHORIZATION_CONTEXT_BYTES
            or not isinstance(reservation.effect_launch_receipt, Mapping)
            or isinstance(reservation.effect_adoption_generation, bool)
            or not isinstance(reservation.effect_adoption_generation, int)
            or reservation.effect_adoption_generation < 0
            or not isinstance(reservation.effect_adoption_receipt, Mapping)
            or not isinstance(reservation.completion_capability_sha256, str)
            or not isinstance(reservation.quarantine_receipt, Mapping)
            or not isinstance(
                reservation.quarantine_terminalization_receipt, Mapping
            )
            or any(
                not isinstance(item, str)
                for item in (
                    reservation.logical_attempt_id,
                    reservation.route_id,
                    reservation.decision_id,
                    reservation.task_id,
                    reservation.worktree_id,
                    reservation.reservation_id,
                )
            )
            or any(
                _token(item, "reservation identity") != item
                for item in (
                    reservation.logical_attempt_id,
                    reservation.route_id,
                    reservation.decision_id,
                    reservation.task_id,
                    reservation.worktree_id,
                    reservation.reservation_id,
                )
            )
            or (reservation.state == "reserved" and (
                reservation.effect_adoption_generation != 0
                or any(
                    bool(item) if isinstance(item, (str, Mapping)) else item is not None
                    for item in (
                    reservation.effect_started_at_ms,
                    reservation.effect_launch_receipt,
                    reservation.effect_adoption_receipt,
                    reservation.completion_capability_sha256,
                    reservation.quarantine_at_ms,
                    reservation.quarantine_receipt,
                    reservation.quarantine_terminalization_receipt,
                    reservation.terminal_at_ms,
                    reservation.terminal_returncode,
                    reservation.terminal_outcome_id,
                    reservation.terminal_outcome,
                    )
                )
            ))
            or (reservation.state == "effect_started" and (
                reservation.effect_started_at_ms is None
                or not _valid_effect_launch_receipt(reservation)
                or not _valid_effect_adoption_receipt(reservation)
                or re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    reservation.completion_capability_sha256,
                )
                is None
                or reservation.terminal_at_ms is not None
                or reservation.terminal_returncode is not None
                or reservation.terminal_outcome_id
                or reservation.terminal_outcome
                or reservation.quarantine_at_ms is not None
                or reservation.quarantine_receipt
                or reservation.quarantine_terminalization_receipt
            ))
            or (reservation.state == "quarantined" and (
                reservation.effect_started_at_ms is None
                or not _valid_effect_launch_receipt(reservation)
                or not _valid_effect_adoption_receipt(reservation)
                or re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    reservation.completion_capability_sha256,
                )
                is None
                or reservation.quarantine_at_ms is None
                or reservation.quarantine_at_ms
                < reservation.effect_started_at_ms
                or not _valid_effect_quarantine_receipt(reservation)
                or (
                    bool(reservation.quarantine_terminalization_receipt)
                    and not _valid_effect_quarantine_terminalization_receipt(
                        reservation
                    )
                )
                or reservation.terminal_at_ms is not None
                or reservation.terminal_returncode is not None
                or reservation.terminal_outcome_id
                or reservation.terminal_outcome
            ))
            or (reservation.state == "terminal" and (
                reservation.effect_started_at_ms is None
                or not _valid_effect_launch_receipt(reservation)
                or not _valid_effect_adoption_receipt(reservation)
                or re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    reservation.completion_capability_sha256,
                )
                is None
                or reservation.terminal_at_ms is None
                or reservation.terminal_returncode is None
                or not reservation.terminal_outcome_id
                or (
                    reservation.quarantine_at_ms is None
                    and (
                        reservation.quarantine_receipt
                        or reservation.quarantine_terminalization_receipt
                    )
                )
                or (
                    reservation.quarantine_at_ms is not None
                    and (
                        not _valid_effect_quarantine_receipt(reservation)
                        or not _valid_effect_quarantine_terminalization_receipt(
                            reservation
                        )
                        or reservation.terminal_returncode
                        != reservation.quarantine_terminalization_receipt.get(
                            "terminal_returncode"
                        )
                    )
                )
            ))
        ):
            raise ProviderAttemptStoreError("attempt reservation state is invalid")
        for name in (
            "logical_attempt_id",
            "route_id",
            "decision_id",
            "task_id",
            "worktree_id",
            "reservation_id",
        ):
            _token(getattr(reservation, name), name)
        try:
            expected_outcome_id = (
                "sha256:"
                + hashlib.sha256(_canonical(reservation.terminal_outcome)).hexdigest()
            )
        except (TypeError, ValueError) as exc:
            raise ProviderAttemptStoreError(
                "terminal outcome is invalid"
            ) from exc
        if (
            reservation.terminal_outcome_id
            and reservation.terminal_outcome_id != expected_outcome_id
        ):
            raise ProviderAttemptStoreError("terminal outcome identity is invalid")
        return reservation

    def _read(self, path: Path) -> ProviderAttemptReservation:
        try:
            descriptor = os.open(
                path.name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=self._directory_fd,
            )
            try:
                before = os.fstat(descriptor)
                _validate_private_regular(before)
                if before.st_size > _MAX_RESERVATION_BYTES:
                    raise ProviderAttemptStoreError(
                        "attempt reservation is oversized"
                    )
                remaining = _MAX_RESERVATION_BYTES + 1
                chunks: list[bytes] = []
                while remaining:
                    chunk = os.read(descriptor, min(64 * 1024, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                raw = b"".join(chunks)
                after = os.fstat(descriptor)
                final = os.stat(
                    path.name,
                    dir_fd=self._directory_fd,
                    follow_symlinks=False,
                )
                _validate_private_regular(after)
                _validate_private_regular(final)
                if (
                    len(raw) > _MAX_RESERVATION_BYTES
                    or len(raw) != after.st_size
                    or _regular_snapshot(before) != _regular_snapshot(after)
                    or _regular_snapshot(after) != _regular_snapshot(final)
                ):
                    raise ProviderAttemptStoreError(
                        "attempt reservation changed while being read"
                    )
            finally:
                os.close(descriptor)
            value = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_unique_json_object,
            )
        except (
            ProviderAttemptStoreError,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            _DuplicateJSONKey,
        ) as exc:
            raise ProviderAttemptStoreError(
                "attempt reservation is unreadable"
            ) from exc
        return self._parse(value)

    def _write(self, path: Path, reservation: ProviderAttemptReservation) -> None:
        # Parse our own serialization before replacing durable state.
        self._parse(asdict(reservation))
        encoded = _canonical(asdict(reservation)) + b"\n"
        if len(encoded) > _MAX_RESERVATION_BYTES:
            raise ProviderAttemptStoreError("attempt reservation is oversized")
        temporary_name = "." + path.name + "." + secrets.token_hex(8)
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=self._directory_fd,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fchmod(stream.fileno(), 0o600)
                os.fsync(stream.fileno())
            os.replace(
                temporary_name,
                path.name,
                src_dir_fd=self._directory_fd,
                dst_dir_fd=self._directory_fd,
            )
            os.fsync(self._directory_fd)
        finally:
            try:
                os.unlink(temporary_name, dir_fd=self._directory_fd)
            except FileNotFoundError:
                pass

    def _entry_exists(self, path: Path) -> bool:
        try:
            os.stat(
                path.name,
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise ProviderAttemptStoreError(
                "attempt reservation is unavailable"
            ) from exc
        return True

    @staticmethod
    def _values(
        *,
        logical_attempt_id: str,
        route_id: str,
        decision_id: str,
        task_id: str,
        worktree_id: str,
    ) -> dict[str, str]:
        return {
            name: _token(value, name)
            for name, value in {
                "logical_attempt_id": logical_attempt_id,
                "route_id": route_id,
                "decision_id": decision_id,
                "task_id": task_id,
                "worktree_id": worktree_id,
            }.items()
        }

    @staticmethod
    def _matches(
        reservation: ProviderAttemptReservation, values: Mapping[str, str]
    ) -> bool:
        return all(getattr(reservation, name) == value for name, value in values.items())

    def reserve_or_adopt(
        self,
        *,
        logical_attempt_id: str,
        route_id: str,
        decision_id: str,
        task_id: str,
        worktree_id: str,
        authorized: bool,
        authorization_context: Mapping[str, Any] | None = None,
        launch_context: Mapping[str, Any] | None = None,
        effect_owner_id: str = _PROCESS_EFFECT_OWNER_ID,
        now_ms: int | None = None,
    ) -> ProviderAttemptCASResult:
        """Create a pre-effect reservation or adopt exact durable state."""

        if authorized is not True:
            raise ProviderAttemptStoreError(
                "router decision did not authorize fallback"
            )
        values = self._values(
            logical_attempt_id=logical_attempt_id,
            route_id=route_id,
            decision_id=decision_id,
            task_id=task_id,
            worktree_id=worktree_id,
        )
        context = dict(authorization_context or {})
        try:
            encoded_context = _canonical(context)
        except (TypeError, ValueError) as exc:
            raise ProviderAttemptStoreError(
                "provider effect authorization context is invalid"
            ) from exc
        if len(encoded_context) > _MAX_AUTHORIZATION_CONTEXT_BYTES:
            raise ProviderAttemptStoreError(
                "provider effect authorization context is oversized"
            )
        descriptor, path = self._lock(values["logical_attempt_id"])
        try:
            if self._entry_exists(path):
                reservation = self._read(path)
                if not self._matches(reservation, values):
                    raise ProviderAttemptStoreError(
                        "existing fallback reservation does not match logical attempt"
                    )
                if launch_context is None or reservation.state != "reserved":
                    return ProviderAttemptCASResult(
                        reservation=reservation,
                        created=False,
                        adopted=True,
                        launch_authorized=False,
                    )
                if dict(reservation.authorization_context) != context:
                    raise ProviderAttemptStoreError(
                        "existing fallback authorization context changed"
                    )
                claimed = self._claim_transition(
                    reservation,
                    launch_context=launch_context,
                    effect_owner_id=effect_owner_id,
                    timestamp=_timestamp(now_ms, "effect-start timestamp"),
                )
                self._write(path, claimed.reservation)
                return claimed
            timestamp = _timestamp(now_ms, "reservation timestamp")
            reservation = ProviderAttemptReservation(
                logical_attempt_id=values["logical_attempt_id"],
                route_id=values["route_id"],
                decision_id=values["decision_id"],
                task_id=values["task_id"],
                worktree_id=values["worktree_id"],
                reservation_id="sha256:"
                + hashlib.sha256(
                    (
                        values["logical_attempt_id"]
                        + "\0"
                        + secrets.token_hex(32)
                    ).encode("utf-8")
                ).hexdigest(),
                state="reserved",
                created_at_ms=timestamp,
                authorization_context=context,
            )
            if launch_context is not None:
                claimed = self._claim_transition(
                    reservation,
                    launch_context=launch_context,
                    effect_owner_id=effect_owner_id,
                    timestamp=timestamp,
                )
                # One atomic replace publishes effect_started directly.  A
                # process crash can leave either no record or the complete
                # winner receipt, never a poisonous reserved intermediary.
                self._write(path, claimed.reservation)
                return replace(claimed, created=True)
            self._write(path, reservation)
            return ProviderAttemptCASResult(
                reservation=reservation,
                created=True,
                adopted=False,
                launch_authorized=False,
            )
        finally:
            self._unlock(descriptor)

    @staticmethod
    def _claim_transition(
        current: ProviderAttemptReservation,
        *,
        launch_context: Mapping[str, Any],
        effect_owner_id: str,
        timestamp: int,
    ) -> ProviderAttemptCASResult:
        """Construct one validated reserved→effect_started CAS value."""

        if timestamp < current.created_at_ms:
            raise ProviderAttemptStoreError(
                "effect-start timestamp predates reservation"
            )
        legacy_context = {
            "provider_id",
            "command_id",
            "runtime_id",
            "image_id",
            "mount_id",
            "environment_id",
            "container_name",
            "container_id",
        }
        protected_details = {
            "cleanup_id",
            "runtime_receipt",
            "image_receipt",
            "command_receipt",
            "mount_receipt",
            "environment_receipt",
            "cleanup_receipt",
        }
        if frozenset(launch_context) not in {
            frozenset(legacy_context),
            frozenset(legacy_context | protected_details),
        }:
            raise ProviderAttemptStoreError(
                "provider effect launch context is invalid"
            )
        owner = _token(effect_owner_id, "effect_owner_id")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", owner) is None:
            raise ProviderAttemptStoreError("effect owner identity is invalid")
        provider_id = _token(launch_context.get("provider_id"), "provider_id")
        command_id = _token(launch_context.get("command_id"), "command_id")
        container_name = _token(
            launch_context.get("container_name"), "container_name"
        )
        container_id = _token(
            launch_context.get("container_id"), "container_id"
        )
        if re.fullmatch(r"sha256:[0-9a-f]{64}", command_id) is None:
            raise ProviderAttemptStoreError("effect command identity is invalid")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", container_id) is None:
            raise ProviderAttemptStoreError("effect container identity is invalid")
        runtime_id = _token(launch_context.get("runtime_id"), "runtime_id")
        image_id = _token(launch_context.get("image_id"), "image_id")
        mount_id = _token(launch_context.get("mount_id"), "mount_id")
        environment_id = _token(
            launch_context.get("environment_id"), "environment_id"
        )
        if any(
            re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
            for value in (runtime_id, image_id, mount_id, environment_id)
        ):
            raise ProviderAttemptStoreError("effect runtime identity is invalid")
        detailed = protected_details.issubset(launch_context)
        detail_values: dict[str, Any] = {}
        if detailed:
            for name in (
                "runtime_receipt",
                "image_receipt",
                "command_receipt",
                "environment_receipt",
                "cleanup_receipt",
            ):
                value = launch_context.get(name)
                if not isinstance(value, Mapping):
                    raise ProviderAttemptStoreError(
                        "provider effect launch receipt is invalid"
                    )
                detail_values[name] = json.loads(_canonical(value).decode("ascii"))
            mount_value = launch_context.get("mount_receipt")
            if not isinstance(mount_value, list):
                raise ProviderAttemptStoreError(
                    "provider effect mount receipt is invalid"
                )
            detail_values["mount_receipt"] = list(mount_value)
            cleanup_id = _token(
                launch_context.get("cleanup_id"), "cleanup_id"
            )
            if re.fullmatch(r"sha256:[0-9a-f]{64}", cleanup_id) is None:
                raise ProviderAttemptStoreError(
                    "provider effect cleanup identity is invalid"
                )
            detail_values["cleanup_id"] = cleanup_id
        launch_receipt: dict[str, Any] = {
            "schema": (
                EFFECT_LAUNCH_SCHEMA if detailed else _LEGACY_EFFECT_LAUNCH_SCHEMA
            ),
            "logical_attempt_id": current.logical_attempt_id,
            "reservation_id": current.reservation_id,
            "effect_owner_id": owner,
            "effect_owner_pid": os.getpid(),
            "effect_owner_start_ticks": _process_start_ticks(os.getpid()),
            "provider_id": provider_id,
            "command_id": command_id,
            "runtime_id": runtime_id,
            "image_id": image_id,
            "mount_id": mount_id,
            "environment_id": environment_id,
            "container_name": container_name,
            "container_id": container_id,
            "claimed_at_ms": timestamp,
            **detail_values,
        }
        if detailed and not _valid_effect_launch_details(launch_receipt):
            raise ProviderAttemptStoreError(
                "provider effect launch details do not match their identities"
            )
        launch_receipt["receipt_id"] = "sha256:" + hashlib.sha256(
            _canonical(launch_receipt)
        ).hexdigest()
        completion_capability = secrets.token_hex(32)
        claimed = ProviderAttemptReservation(
            **{
                **asdict(current),
                "state": "effect_started",
                "effect_started_at_ms": timestamp,
                "effect_launch_receipt": launch_receipt,
                "completion_capability_sha256": "sha256:"
                + hashlib.sha256(
                    completion_capability.encode("ascii")
                ).hexdigest(),
            }
        )
        return ProviderAttemptCASResult(
            reservation=claimed,
            created=False,
            adopted=False,
            launch_authorized=True,
            effect_launch_receipt=launch_receipt,
            completion_capability=completion_capability,
        )

    def claim_effect(
        self,
        reservation: ProviderAttemptReservation,
        *,
        effect_owner_id: str = _PROCESS_EFFECT_OWNER_ID,
        launch_context: Mapping[str, Any],
        now_ms: int | None = None,
    ) -> ProviderAttemptCASResult:
        """Perform the final durable CAS immediately before the provider effect.

        Exactly one process can transition ``reserved`` to ``effect_started``
        and receives ``launch_authorized=True``.  A concurrent process or a
        restarted runner adopts ``effect_started``/``terminal`` and must not
        replay Docker.
        """

        descriptor, path = self._lock(reservation.logical_attempt_id)
        try:
            if not self._entry_exists(path):
                raise ProviderAttemptStoreError("fallback reservation is absent")
            current = self._read(path)
            identity_fields = {
                name: getattr(reservation, name)
                for name in (
                    "logical_attempt_id",
                    "route_id",
                    "decision_id",
                    "task_id",
                    "worktree_id",
                    "reservation_id",
                )
            }
            if not self._matches(current, identity_fields):
                raise ProviderAttemptStoreError("fallback reservation changed")
            if current.state in {"effect_started", "terminal"}:
                return ProviderAttemptCASResult(
                    reservation=current,
                    created=False,
                    adopted=True,
                    launch_authorized=False,
                    effect_launch_receipt=current.effect_launch_receipt,
                )
            timestamp = _timestamp(now_ms, "effect-start timestamp")
            if timestamp < current.created_at_ms:
                raise ProviderAttemptStoreError(
                    "effect-start timestamp predates reservation"
                )
            legacy_context = {
                "provider_id",
                "command_id",
                "runtime_id",
                "image_id",
                "mount_id",
                "environment_id",
                "container_name",
                "container_id",
            }
            protected_details = {
                "cleanup_id",
                "runtime_receipt",
                "image_receipt",
                "command_receipt",
                "mount_receipt",
                "environment_receipt",
                "cleanup_receipt",
            }
            protected_context = legacy_context | protected_details
            if frozenset(launch_context) not in {
                frozenset(legacy_context),
                frozenset(protected_context),
            }:
                raise ProviderAttemptStoreError(
                    "provider effect launch context is invalid"
                )
            owner = _token(effect_owner_id, "effect_owner_id")
            if re.fullmatch(r"sha256:[0-9a-f]{64}", owner) is None:
                raise ProviderAttemptStoreError("effect owner identity is invalid")
            provider_id = _token(
                launch_context.get("provider_id"), "provider_id"
            )
            command_id = _token(
                launch_context.get("command_id"), "command_id"
            )
            container_name = _token(
                launch_context.get("container_name"), "container_name"
            )
            container_id = _token(
                launch_context.get("container_id"), "container_id"
            )
            if re.fullmatch(r"sha256:[0-9a-f]{64}", command_id) is None:
                raise ProviderAttemptStoreError("effect command identity is invalid")
            if re.fullmatch(r"sha256:[0-9a-f]{64}", container_id) is None:
                raise ProviderAttemptStoreError("effect container identity is invalid")
            runtime_id = _token(launch_context.get("runtime_id"), "runtime_id")
            image_id = _token(launch_context.get("image_id"), "image_id")
            mount_id = _token(launch_context.get("mount_id"), "mount_id")
            environment_id = _token(
                launch_context.get("environment_id"), "environment_id"
            )
            if any(
                re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
                for value in (runtime_id, image_id, mount_id, environment_id)
            ):
                raise ProviderAttemptStoreError(
                    "effect runtime identity is invalid"
                )
            detailed = protected_details.issubset(launch_context)
            detail_values: dict[str, Any] = {}
            if detailed:
                for name in (
                    "runtime_receipt",
                    "image_receipt",
                    "command_receipt",
                    "environment_receipt",
                    "cleanup_receipt",
                ):
                    value = launch_context.get(name)
                    if not isinstance(value, Mapping):
                        raise ProviderAttemptStoreError(
                            "provider effect launch receipt is invalid"
                        )
                    detail_values[name] = json.loads(
                        _canonical(value).decode("ascii")
                    )
                mount_value = launch_context.get("mount_receipt")
                if not isinstance(mount_value, list):
                    raise ProviderAttemptStoreError(
                        "provider effect mount receipt is invalid"
                    )
                detail_values["mount_receipt"] = list(mount_value)
                cleanup_id = _token(
                    launch_context.get("cleanup_id"), "cleanup_id"
                )
                if re.fullmatch(r"sha256:[0-9a-f]{64}", cleanup_id) is None:
                    raise ProviderAttemptStoreError(
                        "provider effect cleanup identity is invalid"
                    )
                detail_values["cleanup_id"] = cleanup_id
            launch_receipt: dict[str, Any] = {
                "schema": (
                    EFFECT_LAUNCH_SCHEMA
                    if detailed
                    else _LEGACY_EFFECT_LAUNCH_SCHEMA
                ),
                "logical_attempt_id": current.logical_attempt_id,
                "reservation_id": current.reservation_id,
                "effect_owner_id": owner,
                "effect_owner_pid": os.getpid(),
                "effect_owner_start_ticks": _process_start_ticks(os.getpid()),
                "provider_id": provider_id,
                "command_id": command_id,
                "runtime_id": runtime_id,
                "image_id": image_id,
                "mount_id": mount_id,
                "environment_id": environment_id,
                "container_name": container_name,
                "container_id": container_id,
                "claimed_at_ms": timestamp,
                **detail_values,
            }
            if detailed and not _valid_effect_launch_details(launch_receipt):
                raise ProviderAttemptStoreError(
                    "provider effect launch details do not match their identities"
                )
            launch_receipt["receipt_id"] = "sha256:" + hashlib.sha256(
                _canonical(launch_receipt)
            ).hexdigest()
            completion_capability = secrets.token_hex(32)
            completion_capability_sha256 = "sha256:" + hashlib.sha256(
                completion_capability.encode("ascii")
            ).hexdigest()
            claimed = ProviderAttemptReservation(
                **{
                    **asdict(current),
                    "state": "effect_started",
                    "effect_started_at_ms": timestamp,
                    "effect_launch_receipt": launch_receipt,
                    "completion_capability_sha256": (
                        completion_capability_sha256
                    ),
                }
            )
            self._write(path, claimed)
            return ProviderAttemptCASResult(
                reservation=claimed,
                created=False,
                adopted=False,
                launch_authorized=True,
                effect_launch_receipt=launch_receipt,
                completion_capability=completion_capability,
            )
        finally:
            self._unlock(descriptor)

    def claim_quarantined_terminalization(
        self,
        reservation: ProviderAttemptReservation,
        *,
        completion_capability: str = "",
        effect_owner_id: str = _PROCESS_EFFECT_OWNER_ID,
        now_ms: int | None = None,
    ) -> ProviderAttemptCASResult:
        """Claim accounting-only repair of an exact quarantined effect.

        This transition can never create, start, or relaunch a provider.  It
        internally reinspects the CAS winner's exact container and grants a
        process-bound completion capability only after the effect is proven
        absent or exited.  Created/running effects remain quarantined for a
        later sealed recovery/operator pass.
        """

        descriptor, path = self._lock(reservation.logical_attempt_id)
        try:
            current = self._read(path)
            if current.state == "terminal":
                return ProviderAttemptCASResult(
                    reservation=current,
                    created=False,
                    adopted=True,
                    launch_authorized=False,
                    effect_launch_receipt=current.effect_launch_receipt,
                )
            if (
                current.state != "quarantined"
                or current.reservation_id != reservation.reservation_id
                or current.logical_attempt_id != reservation.logical_attempt_id
                or not _valid_effect_quarantine_receipt(current)
            ):
                raise ProviderAttemptStoreError(
                    "fallback quarantine is not available for terminalization"
                )
            owner = _token(effect_owner_id, "effect_owner_id")
            if re.fullmatch(r"sha256:[0-9a-f]{64}", owner) is None:
                raise ProviderAttemptStoreError(
                    "quarantine repair owner identity is invalid"
                )
            previous = current.quarantine_terminalization_receipt
            repair_generation = 1
            previous_receipt_id = ""
            prior_repairs: list[Mapping[str, Any]] = []
            if previous:
                if not _valid_effect_quarantine_terminalization_receipt(current):
                    raise ProviderAttemptStoreError(
                        "quarantine repair receipt is invalid"
                    )
                previous_pid = int(previous.get("effect_owner_pid") or 0)
                previous_start = int(
                    previous.get("effect_owner_start_ticks") or 0
                )
                previous_alive = _process_identity_alive(
                    previous_pid,
                    previous_start,
                )
                same_live_owner = bool(
                    previous_alive
                    and previous.get("effect_owner_id") == owner
                    and previous_pid == os.getpid()
                    and previous_start == _process_start_ticks(os.getpid())
                    and isinstance(completion_capability, str)
                    and re.fullmatch(
                        r"[0-9a-f]{64}", completion_capability
                    )
                    is not None
                    and "sha256:"
                    + hashlib.sha256(
                        completion_capability.encode("ascii")
                    ).hexdigest()
                    == current.completion_capability_sha256
                )
                if same_live_owner:
                    return ProviderAttemptCASResult(
                        reservation=current,
                        created=False,
                        adopted=True,
                        launch_authorized=False,
                        adoption_authorized=True,
                        effect_launch_receipt=current.effect_launch_receipt,
                        completion_capability=completion_capability,
                    )
                if previous_alive:
                    raise ProviderAttemptStoreError(
                        "quarantine repair owner is still alive"
                    )
                repair_generation = int(previous["repair_generation"]) + 1
                previous_receipt_id = str(previous["receipt_id"])
                prior_repairs = [
                    *list(previous.get("prior_repair_receipts", [])),
                    dict(previous),
                ]

            timestamp = _timestamp(now_ms, "quarantine inspection timestamp")
            if timestamp < int(current.quarantine_at_ms or 0):
                raise ProviderAttemptStoreError(
                    "quarantine inspection predates quarantine"
                )
            try:
                inspection = _inspect_recorded_docker_effect(
                    dict(current.effect_launch_receipt),
                    timestamp,
                )
            except ProviderAttemptStoreError:
                raise
            except Exception as exc:
                raise ProviderAttemptStoreError(
                    "quarantine effect inspection failed"
                ) from exc
            launch = current.effect_launch_receipt
            expected_inspection = {
                "status",
                "inspection_runtime_id",
                "inspection_command_id",
                "observed_at_ms",
                "provider_id",
                "command_id",
                "runtime_id",
                "image_id",
                "mount_id",
                "environment_id",
                "container_name",
                "container_id",
                "returncode",
            }
            status_value = (
                inspection.get("status")
                if isinstance(inspection, Mapping)
                else None
            )
            returncode = (
                inspection.get("returncode")
                if isinstance(inspection, Mapping)
                else None
            )
            if (
                not isinstance(inspection, Mapping)
                or set(inspection) != expected_inspection
                or status_value not in {"created", "running", "exited", "absent"}
                or inspection.get("inspection_runtime_id")
                != launch.get("runtime_id")
                or re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(inspection.get("inspection_command_id") or ""),
                )
                is None
                or inspection.get("observed_at_ms") != timestamp
                or any(
                    inspection.get(name) != launch.get(name)
                    for name in (
                        "provider_id",
                        "command_id",
                        "runtime_id",
                        "image_id",
                        "mount_id",
                        "environment_id",
                        "container_name",
                    )
                )
                or (
                    status_value == "exited"
                    and (
                        isinstance(returncode, bool)
                        or not isinstance(returncode, int)
                    )
                )
                or (status_value != "exited" and returncode is not None)
                or (
                    status_value in {"created", "running", "exited"}
                    and inspection.get("container_id")
                    != launch.get("container_id")
                )
                or (
                    status_value == "absent"
                    and inspection.get("container_id") != ""
                )
            ):
                raise ProviderAttemptStoreError(
                    "quarantine inspection does not match the winning effect"
                )
            if status_value in {"created", "running"}:
                return ProviderAttemptCASResult(
                    reservation=current,
                    created=False,
                    adopted=True,
                    launch_authorized=False,
                    adoption_authorized=False,
                    effect_launch_receipt=current.effect_launch_receipt,
                )

            terminal_returncode = 125 if status_value == "absent" else returncode
            outcome_decision = (
                "effect_not_created"
                if status_value == "absent"
                else (
                    "fallback_succeeded"
                    if terminal_returncode == 0
                    else "fallback_failed"
                )
            )
            repair_receipt: dict[str, Any] = {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "provider-effect-quarantine-terminalization@1"
                ),
                "logical_attempt_id": current.logical_attempt_id,
                "reservation_id": current.reservation_id,
                "incident_id": current.quarantine_receipt["incident_id"],
                "repair_generation": repair_generation,
                "previous_repair_receipt_id": previous_receipt_id,
                "prior_repair_receipts": prior_repairs,
                "operator_action": (
                    "terminalize_exact_quarantined_effect_without_relaunch"
                ),
                "effect_owner_id": owner,
                "effect_owner_pid": os.getpid(),
                "effect_owner_start_ticks": _process_start_ticks(os.getpid()),
                "provider_id": launch["provider_id"],
                "command_id": launch["command_id"],
                "runtime_id": launch["runtime_id"],
                "image_id": launch["image_id"],
                "mount_id": launch["mount_id"],
                "environment_id": launch["environment_id"],
                "container_name": launch["container_name"],
                "container_id": inspection["container_id"],
                "inspection_status": status_value,
                "inspection_command_id": inspection["inspection_command_id"],
                "inspected_at_ms": timestamp,
                "container_returncode": returncode,
                "terminal_returncode": terminal_returncode,
                "outcome_decision": outcome_decision,
                "fallback_dispatched": status_value == "exited",
            }
            repair_receipt["receipt_id"] = "sha256:" + hashlib.sha256(
                _canonical(repair_receipt)
            ).hexdigest()
            capability = secrets.token_hex(32)
            updated = ProviderAttemptReservation(
                **{
                    **asdict(current),
                    "quarantine_terminalization_receipt": repair_receipt,
                    "completion_capability_sha256": (
                        "sha256:"
                        + hashlib.sha256(capability.encode("ascii")).hexdigest()
                    ),
                }
            )
            if not _valid_effect_quarantine_terminalization_receipt(updated):
                raise ProviderAttemptStoreError(
                    "quarantine terminalization receipt is invalid"
                )
            self._write(path, updated)
            return ProviderAttemptCASResult(
                reservation=updated,
                created=False,
                adopted=True,
                launch_authorized=False,
                adoption_authorized=True,
                effect_launch_receipt=updated.effect_launch_receipt,
                completion_capability=capability,
            )
        finally:
            self._unlock(descriptor)

    def complete(
        self,
        reservation: ProviderAttemptReservation,
        *,
        returncode: int = 0,
        outcome: Mapping[str, Any] | None = None,
        completion_capability: str = "",
        effect_owner_id: str = _PROCESS_EFFECT_OWNER_ID,
        now_ms: int | None = None,
    ) -> ProviderAttemptReservation:
        """Record an immutable terminal result without touching retry counters."""

        if isinstance(returncode, bool) or not isinstance(returncode, int):
            raise ProviderAttemptStoreError("terminal returncode is invalid")
        terminal_outcome = dict(outcome or {"returncode": returncode})
        outcome_id = "sha256:" + hashlib.sha256(
            _canonical(terminal_outcome)
        ).hexdigest()
        descriptor, path = self._lock(reservation.logical_attempt_id)
        try:
            if not self._entry_exists(path):
                raise ProviderAttemptStoreError("fallback reservation is absent")
            current = self._read(path)
            if current.state == "terminal":
                if (
                    current.reservation_id == reservation.reservation_id
                    and current.terminal_returncode == returncode
                    and current.terminal_outcome_id == outcome_id
                ):
                    return current
                raise ProviderAttemptStoreError("terminal fallback result changed")
            if (
                current.state not in {"effect_started", "quarantined"}
                or current.reservation_id != reservation.reservation_id
                or current.logical_attempt_id != reservation.logical_attempt_id
                or current.route_id != reservation.route_id
                or current.decision_id != reservation.decision_id
                or current.task_id != reservation.task_id
                or current.worktree_id != reservation.worktree_id
            ):
                raise ProviderAttemptStoreError(
                    "fallback effect was not durably claimed"
                )
            completion_owner = (
                current.quarantine_terminalization_receipt
                if current.state == "quarantined"
                else _effective_effect_owner(current)
            )
            if current.state == "quarantined":
                repair = current.quarantine_terminalization_receipt
                if (
                    not _valid_effect_quarantine_terminalization_receipt(current)
                    or returncode != repair.get("terminal_returncode")
                    or terminal_outcome.get("decision")
                    != repair.get("outcome_decision")
                    or terminal_outcome.get("fallback_dispatched")
                    is not repair.get("fallback_dispatched")
                    or terminal_outcome.get("fallback_returncode") != returncode
                    or terminal_outcome.get("reservation_id")
                    != current.reservation_id
                    or terminal_outcome.get("effect_launch_receipt")
                    != current.effect_launch_receipt
                    or terminal_outcome.get("effect_adoption_receipt")
                    != current.effect_adoption_receipt
                    or terminal_outcome.get("effect_quarantine_receipt")
                    != current.quarantine_receipt
                    or terminal_outcome.get(
                        "effect_quarantine_terminalization_receipt"
                    )
                    != current.quarantine_terminalization_receipt
                ):
                    raise ProviderAttemptStoreError(
                        "quarantine terminal outcome does not match repair receipt"
                    )
            if (
                not isinstance(completion_capability, str)
                or re.fullmatch(r"[0-9a-f]{64}", completion_capability) is None
                or "sha256:"
                + hashlib.sha256(
                    completion_capability.encode("ascii")
                ).hexdigest()
                != current.completion_capability_sha256
                or completion_owner.get("effect_owner_id")
                != _token(effect_owner_id, "effect_owner_id")
                or completion_owner.get("effect_owner_pid") != os.getpid()
                or completion_owner.get("effect_owner_start_ticks")
                != _process_start_ticks(os.getpid())
            ):
                raise ProviderAttemptStoreError(
                    "terminal completion is not held by the effect winner"
                )
            timestamp = _timestamp(now_ms, "terminal timestamp")
            minimum_terminal_timestamp = max(
                int(current.effect_started_at_ms or 0),
                int(
                    current.quarantine_terminalization_receipt.get(
                        "inspected_at_ms", 0
                    )
                    if current.state == "quarantined"
                    else 0
                ),
            )
            if timestamp < minimum_terminal_timestamp:
                raise ProviderAttemptStoreError(
                    "terminal timestamp predates effect start"
                )
            terminal = ProviderAttemptReservation(
                **{
                    **asdict(current),
                    "state": "terminal",
                    "terminal_at_ms": timestamp,
                    "terminal_returncode": returncode,
                    "terminal_outcome_id": outcome_id,
                    "terminal_outcome": terminal_outcome,
                }
            )
            self._write(path, terminal)
            return terminal
        finally:
            self._unlock(descriptor)

    def adopt_effect(
        self,
        reservation: ProviderAttemptReservation,
        *,
        completion_capability: str = "",
        effect_owner_id: str = _PROCESS_EFFECT_OWNER_ID,
        now_ms: int | None = None,
    ) -> ProviderAttemptCASResult:
        """Transfer a dead owner's exact effect without ever relaunching it.

        The store inspects the immutable launch receipt itself while holding
        the CAS lock.  No public callback or caller-authored result mapping is
        accepted: otherwise any process could claim that the winning
        container was absent or exited.
        """

        descriptor, path = self._lock(reservation.logical_attempt_id)
        try:
            current = self._read(path)
            if current.state == "terminal":
                return ProviderAttemptCASResult(
                    reservation=current,
                    created=False,
                    adopted=True,
                    launch_authorized=False,
                    effect_launch_receipt=current.effect_launch_receipt,
                )
            if (
                current.state != "effect_started"
                or current.reservation_id != reservation.reservation_id
                or current.logical_attempt_id != reservation.logical_attempt_id
            ):
                raise ProviderAttemptStoreError(
                    "fallback effect is not available for adoption"
                )
            previous_owner = _effective_effect_owner(current)
            previous_pid = int(previous_owner.get("effect_owner_pid") or 0)
            previous_start = int(
                previous_owner.get("effect_owner_start_ticks") or 0
            )
            owner = _token(effect_owner_id, "effect_owner_id")
            if re.fullmatch(r"sha256:[0-9a-f]{64}", owner) is None:
                raise ProviderAttemptStoreError("effect owner identity is invalid")
            previous_alive = _process_identity_alive(
                previous_pid,
                previous_start,
            )
            winner_reconciliation = bool(
                previous_alive
                and previous_owner.get("effect_owner_id") == owner
                and previous_pid == os.getpid()
                and previous_start == _process_start_ticks(os.getpid())
                and isinstance(completion_capability, str)
                and re.fullmatch(r"[0-9a-f]{64}", completion_capability)
                is not None
                and "sha256:"
                + hashlib.sha256(
                    completion_capability.encode("ascii")
                ).hexdigest()
                == current.completion_capability_sha256
            )
            if previous_alive and not winner_reconciliation:
                raise ProviderAttemptStoreError(
                    "fallback effect owner is still alive"
                )
            timestamp = _timestamp(now_ms, "effect adoption timestamp")
            if timestamp < int(current.effect_started_at_ms or 0):
                raise ProviderAttemptStoreError(
                    "effect adoption timestamp predates effect start"
                )
            try:
                inspection = _inspect_recorded_docker_effect(
                    dict(current.effect_launch_receipt),
                    timestamp,
                )
            except ProviderAttemptStoreError:
                raise
            except Exception as exc:
                raise ProviderAttemptStoreError(
                    "effect adoption inspection failed"
                ) from exc
            expected_inspection = {
                "status",
                "inspection_runtime_id",
                "inspection_command_id",
                "observed_at_ms",
                "provider_id",
                "command_id",
                "runtime_id",
                "image_id",
                "mount_id",
                "environment_id",
                "container_name",
                "container_id",
                "returncode",
            }
            if (
                not isinstance(inspection, Mapping)
                or set(inspection) != expected_inspection
            ):
                raise ProviderAttemptStoreError(
                    "effect adoption inspection is invalid"
                )
            launch = current.effect_launch_receipt
            status_value = inspection.get("status")
            returncode = inspection.get("returncode")
            if (
                status_value not in {"created", "running", "exited", "absent"}
                or inspection.get("inspection_runtime_id")
                != launch.get("runtime_id")
                or re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(inspection.get("inspection_command_id") or ""),
                )
                is None
                or inspection.get("observed_at_ms") != timestamp
                or any(
                    inspection.get(name) != launch.get(name)
                    for name in (
                        "provider_id",
                        "command_id",
                        "runtime_id",
                        "image_id",
                        "mount_id",
                        "environment_id",
                        "container_name",
                    )
                )
                or (
                    status_value == "exited"
                    and (
                        isinstance(returncode, bool)
                        or not isinstance(returncode, int)
                    )
                )
                or (status_value != "exited" and returncode is not None)
                or (
                    status_value in {"created", "running", "exited"}
                    and re.fullmatch(
                        r"sha256:[0-9a-f]{64}",
                        str(inspection.get("container_id") or ""),
                    )
                    is None
                )
                or (
                    status_value in {"created", "running", "exited"}
                    and inspection.get("container_id")
                    != launch.get("container_id")
                )
                or (
                    status_value == "absent"
                    and inspection.get("container_id") != ""
                )
            ):
                raise ProviderAttemptStoreError(
                    "effect adoption inspection does not match the winner"
                )
            if current.effect_adoption_generation >= 8:
                quarantine_receipt: dict[str, Any] = {
                    "schema": (
                        "ipfs_accelerate_py/agent-supervisor/"
                        "provider-effect-quarantine@1"
                    ),
                    "logical_attempt_id": current.logical_attempt_id,
                    "reservation_id": current.reservation_id,
                    "adoption_generation": current.effect_adoption_generation,
                    "reason": "adoption_transfer_limit_exhausted",
                    "required_operator_action": (
                        "inspect_exact_container_and_terminalize_without_relaunch"
                    ),
                    "provider_id": launch["provider_id"],
                    "runtime_id": launch["runtime_id"],
                    "container_name": launch["container_name"],
                    "container_id": inspection["container_id"],
                    "inspection_status": status_value,
                    "inspection_command_id": inspection[
                        "inspection_command_id"
                    ],
                    "container_returncode": returncode,
                    "quarantined_at_ms": timestamp,
                }
                quarantine_receipt["incident_id"] = "sha256:" + hashlib.sha256(
                    _canonical(quarantine_receipt)
                ).hexdigest()
                quarantined = ProviderAttemptReservation(
                    **{
                        **asdict(current),
                        "state": "quarantined",
                        "quarantine_at_ms": timestamp,
                        "quarantine_receipt": quarantine_receipt,
                    }
                )
                self._write(path, quarantined)
                return ProviderAttemptCASResult(
                    reservation=quarantined,
                    created=False,
                    adopted=True,
                    launch_authorized=False,
                    adoption_authorized=False,
                    effect_launch_receipt=quarantined.effect_launch_receipt,
                )
            generation = current.effect_adoption_generation + 1
            prior_adoptions = (
                [
                    *list(
                        current.effect_adoption_receipt.get(
                            "prior_adoption_receipts", []
                        )
                    ),
                    dict(current.effect_adoption_receipt),
                ]
                if current.effect_adoption_generation
                else []
            )
            previous_receipt_id = str(
                previous_owner.get("receipt_id") or ""
            )
            adoption_receipt: dict[str, Any] = {
                "schema": EFFECT_ADOPTION_SCHEMA,
                "logical_attempt_id": current.logical_attempt_id,
                "reservation_id": current.reservation_id,
                "adoption_generation": generation,
                "previous_receipt_id": previous_receipt_id,
                "previous_owner_id": str(
                    previous_owner.get("effect_owner_id") or ""
                ),
                "previous_owner_pid": previous_pid,
                "previous_owner_start_ticks": previous_start,
                "effect_owner_id": owner,
                "effect_owner_pid": os.getpid(),
                "effect_owner_start_ticks": _process_start_ticks(os.getpid()),
                "transition_kind": (
                    "winner_reconciliation"
                    if winner_reconciliation
                    else "dead_owner_adoption"
                ),
                "inspection_status": status_value,
                "inspection_runtime_id": inspection["inspection_runtime_id"],
                "inspection_command_id": inspection["inspection_command_id"],
                "inspection_observed_at_ms": inspection["observed_at_ms"],
                "provider_id": launch["provider_id"],
                "command_id": launch["command_id"],
                "runtime_id": launch["runtime_id"],
                "image_id": launch["image_id"],
                "mount_id": launch["mount_id"],
                "environment_id": launch["environment_id"],
                "container_name": launch["container_name"],
                "container_id": inspection["container_id"],
                "container_returncode": returncode,
                "inspected_at_ms": timestamp,
                "prior_adoption_receipts": prior_adoptions,
            }
            adoption_receipt["receipt_id"] = "sha256:" + hashlib.sha256(
                _canonical(adoption_receipt)
            ).hexdigest()
            completion_capability = secrets.token_hex(32)
            capability_digest = "sha256:" + hashlib.sha256(
                completion_capability.encode("ascii")
            ).hexdigest()
            updated_values = {
                **asdict(current),
                "effect_adoption_generation": generation,
                "effect_adoption_receipt": adoption_receipt,
                "completion_capability_sha256": capability_digest,
            }
            adopted = ProviderAttemptReservation(**updated_values)
            self._write(path, adopted)
            return ProviderAttemptCASResult(
                reservation=adopted,
                created=False,
                adopted=True,
                launch_authorized=False,
                adoption_authorized=True,
                effect_launch_receipt=adopted.effect_launch_receipt,
                completion_capability=completion_capability,
            )
        finally:
            self._unlock(descriptor)

    def read(self, logical_attempt_id: str) -> ProviderAttemptReservation | None:
        """Read one durable state under its CAS lock."""

        descriptor, path = self._lock(logical_attempt_id)
        try:
            return self._read(path) if self._entry_exists(path) else None
        finally:
            self._unlock(descriptor)
