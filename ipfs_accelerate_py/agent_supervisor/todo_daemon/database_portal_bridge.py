"""Attempt-local Portal execution for database-authoritative task claims.

``DatabaseImplementationDaemon`` owns the durable claim and completion state.
``PortalImplementationDaemon`` owns the already-landed implementation pipeline
(provider routing, isolated worktrees, validation, proof gates, and merge
reconciliation).  This module joins those authorities without allowing the
Portal daemon to mutate the canonical task board: each database attempt gets a
single-task Markdown *projection* below its private state directory.

The projection is deliberately disposable and non-authoritative.  Its
immutable fields are sealed before provider execution; only its status line
may change.  A database phase may consume the result only after the projected
task has a matching durable Portal completion event.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shlex
import stat
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Final

from ..task_sources.intent_repository import (
    VALIDATION_ARGV_REPRESENTATION,
    VALIDATION_REPRESENTATION_POLICY_KEY,
    VALIDATION_SHELL_TEXT_REPRESENTATION,
)

DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE: Final[str] = "DatabasePortalExecutionBridge@1"
DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-execution-receipt@1"
)
DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-portal-attempt-binding@1"
)
DATABASE_PORTAL_ATTEMPT_RECONCILIATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "database-portal-attempt-reconciliation@1"
)
_TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {"completed", "complete", "done"}
)
_MUTABLE_PROJECTION_LINE = re.compile(r"(?mi)^-\s*status\s*:\s*.*$")
_HEADER = re.compile(r"(?m)^##\s+([^\s]+)(?:\s+.*)?$")
_VALIDATION_REPRESENTATIONS: Final[frozenset[str]] = frozenset(
    {
        VALIDATION_SHELL_TEXT_REPRESENTATION,
        VALIDATION_ARGV_REPRESENTATION,
    }
)
_ATTEMPT_BINDING_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "interface",
        "attempt_id",
        "claim_id",
        "task_cid",
        "task_alias",
        "goal_cid",
        "plan_cid",
        "task_revision",
        "fencing_token",
        "fence_epoch",
        "lease_id",
        "task_body_digest",
        "projection_seed_digest",
        "projection_immutable_digest",
        "authoritative_task_store",
        "projection_authority",
        "binding_id",
    }
)
_ATTEMPT_RECONCILIATION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "interface",
        "attempt_id",
        "claim_id",
        "task_cid",
        "task_alias",
        "attempt_number",
        "owner_session_id",
        "fencing_token",
        "fence_epoch",
        "attempt_root",
        "stage",
        "trigger",
        "reconciled_at",
        "reconciled",
        "blocked",
        "reason",
        "binding_id",
        "binding_path",
        "historical_binding",
        "nested_state",
        "provider_runner_fence",
        "provider_runner_reconciliation_authority",
        "portal_reconciliation",
        "terminal_provider_evidence",
        "terminal_provider_receipt_id",
        "error_type",
        "error",
        "intended_database_disposition",
        "database_disposition",
        "database_attempt_status",
        "database_attempt_phase",
        "database_task_status",
        "terminal_reconciliation_evidence_id",
        "retry_receipt",
        "prepared_reconciliation_receipt_id",
        "receipt_id",
    }
)
_PROVIDER_RUNNER_FENCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "applicable",
        "safe_to_restart",
        "fenced",
        "reason",
        "pid",
        "pid_reused",
        "parent_pid_before_fence",
    }
)


class DatabasePortalBridgeError(RuntimeError):
    """A database claim could not obtain trustworthy Portal evidence."""


class DatabasePortalBridgeDeferred(DatabasePortalBridgeError):
    """Portal execution made bounded progress but is not yet acceptable."""


class DatabasePortalPreEntryPublicationDeferred(DatabasePortalBridgeDeferred):
    """A transient immutable publication fault occurred before Portal entry."""


@dataclass(frozen=True)
class DatabasePortalAttemptPaths:
    """Private, non-authoritative paths for one database task attempt."""

    root: Path
    task_projection: Path
    binding: Path
    state: Path
    strategy: Path
    events: Path
    implementation_logs: Path
    reconciliation: Path


PortalDaemonFactory = Callable[[DatabasePortalAttemptPaths, str], Any]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise DatabasePortalBridgeError(
            f"could not read Portal attempt artifact {path.name!r}"
        ) from exc


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_durable_directory(path: Path) -> None:
    """Create *path* and durably publish every new directory entry."""

    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            raise DatabasePortalBridgeError(
                "database Portal attempt directory has no existing authority root"
            )
        cursor = parent
    if cursor.is_symlink() or not cursor.is_dir():
        raise DatabasePortalBridgeError(
            "database Portal attempt directory authority is not a directory"
        )
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            pass
        if directory.is_symlink() or not directory.is_dir():
            raise DatabasePortalBridgeError(
                "database Portal attempt directory changed during publication"
            )
        _fsync_directory(directory)
        _fsync_directory(directory.parent)
    # A previous interrupted publication can leave the directory visible in
    # this process without proving its directory entry durable.  Re-fsync the
    # exact attempt directory and parent on every prepared-stage resume.
    _fsync_directory(path)
    _fsync_directory(path.parent)


@contextmanager
def _immutable_publication_lock(path: Path) -> Any:
    """Serialize bounded cooperating publication/recovery in one directory."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path.parent, flags)
    except OSError as exc:
        raise DatabasePortalBridgeError(
            "database Portal immutable evidence directory is unavailable"
        ) from exc
    locked = False
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence directory authority changed"
            )
        deadline = time.monotonic() + 5.0
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise DatabasePortalBridgeError(
                        "database Portal immutable evidence publication lock timed out"
                    ) from exc
                time.sleep(0.01)
        yield
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _cleanup_immutable_publication_stages_unlocked(path: Path) -> None:
    """Remove only bounded, exact, non-authoritative crashed stage files."""

    exact_stage = re.compile(
        rf"^\.{re.escape(path.name)}\.[A-Za-z0-9_-]+\.stage$"
    )
    try:
        stages = sorted(
            child
            for child in path.parent.iterdir()
            if exact_stage.fullmatch(child.name)
        )
    except OSError as exc:
        raise DatabasePortalBridgeError(
            "database Portal immutable evidence stage set is unreadable"
        ) from exc
    if len(stages) > 8:
        raise DatabasePortalBridgeError(
            "database Portal immutable evidence has too many stages"
        )
    stage_records: list[tuple[Path, Path, os.stat_result]] = []
    for stage_path in stages:
        try:
            metadata = stage_path.lstat()
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence stage is unreadable"
            ) from exc
        if (
            stage_path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or int(metadata.st_nlink) not in {1, 2}
            or int(metadata.st_size) > 262_144
        ):
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence stage is unsafe"
            )
        ready_path = stage_path.with_name(
            stage_path.name[: -len(".stage")] + ".tmp"
        )
        if int(metadata.st_nlink) == 2:
            try:
                ready_metadata = ready_path.lstat()
            except OSError as exc:
                raise DatabasePortalBridgeError(
                    "database Portal immutable evidence linked stage is incomplete"
                ) from exc
            if (
                ready_path.is_symlink()
                or not stat.S_ISREG(ready_metadata.st_mode)
                or ready_metadata.st_uid != os.geteuid()
                or int(ready_metadata.st_nlink) != 2
                or ready_metadata.st_dev != metadata.st_dev
                or ready_metadata.st_ino != metadata.st_ino
            ):
                raise DatabasePortalBridgeError(
                    "database Portal immutable evidence linked stage is not exact"
                )
        stage_records.append((stage_path, ready_path, metadata))
    removed = False
    for stage_path, _ready_path, _metadata in stage_records:
        try:
            stage_path.unlink()
            removed = True
        except FileNotFoundError:
            # The advisory lock excludes cooperating writers.  A vanished
            # stage has no authority; re-enumeration below still rejects any
            # replacement or unresolved exact stage.
            continue
    if removed:
        _fsync_directory(path.parent)
    try:
        remaining = [
            child
            for child in path.parent.iterdir()
            if exact_stage.fullmatch(child.name)
        ]
    except OSError as exc:
        raise DatabasePortalBridgeError(
            "database Portal immutable evidence stage set changed unreadably"
        ) from exc
    if remaining:
        raise DatabasePortalBridgeError(
            "database Portal immutable evidence stage cleanup did not converge"
        )


def _atomic_write_if_absent(path: Path, payload: bytes) -> None:
    """Publish immutable evidence without replacing an existing object."""

    _ensure_durable_directory(path.parent)
    with _immutable_publication_lock(path):
        _cleanup_immutable_publication_stages_unlocked(path)
        # An earlier process can have stopped at any durable publication
        # boundary.  Repair only an exact ready temporary while no cooperating
        # writer can still be mutating it.
        _recover_immutable_link_publication_unlocked(
            path,
            expected_payload=payload,
        )
        if path.exists() or path.is_symlink():
            _durably_revalidate_immutable_final_unlocked(
                path,
                expected_payload=payload,
            )
            return

        descriptor, stage_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".stage",
            dir=path.parent,
        )
        stage_path = Path(stage_name)
        ready_path = stage_path.with_name(
            stage_path.name[: -len(".stage")] + ".tmp"
        )
        stage_removed = False
        ready_owned = False
        ready_removed = False
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(stage_path, ready_path)
            except FileExistsError as exc:
                raise DatabasePortalBridgeError(
                    "database Portal immutable evidence ready name already exists"
                ) from exc
            ready_owned = True
            _fsync_directory(path.parent)
            stage_path.unlink()
            stage_removed = True
            _fsync_directory(path.parent)
            try:
                os.link(ready_path, path)
            except FileExistsError:
                _recover_immutable_link_publication_unlocked(
                    path,
                    expected_payload=payload,
                    require_final=True,
                )
            except FileNotFoundError:
                _recover_immutable_link_publication_unlocked(
                    path,
                    expected_payload=payload,
                    require_final=True,
                )
            try:
                ready_path.unlink()
            except FileNotFoundError:
                pass
            ready_removed = True
            _fsync_directory(path.parent)
            _fsync_directory(path.parent.parent)
            _durably_revalidate_immutable_final_unlocked(
                path,
                expected_payload=payload,
            )
        finally:
            cleanup_changed = False
            if not stage_removed:
                try:
                    stage_path.unlink()
                    cleanup_changed = True
                except FileNotFoundError:
                    pass
            if ready_owned and not ready_removed:
                try:
                    ready_path.unlink()
                    cleanup_changed = True
                except FileNotFoundError:
                    pass
            if cleanup_changed:
                _fsync_directory(path.parent)


def _immutable_receipt_payload_is_exact(path: Path, payload: bytes) -> bool:
    """Return whether *payload* is the canonical self-CID receipt at *path*."""

    if len(payload) > 262_144:
        return False

    def closed_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        receipt = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=closed_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("nonfinite value")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return False
    if not isinstance(receipt, Mapping):
        return False
    receipt = dict(receipt)
    expected_id = f"sha256:{path.stem}"
    if receipt.get("receipt_id") != expected_id:
        return False
    unsigned = dict(receipt)
    unsigned.pop("receipt_id", None)
    if _sha256_bytes(_canonical_json(unsigned)) != expected_id:
        return False
    expected_storage = (
        json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    return payload == expected_storage


def _recover_immutable_link_publication_unlocked(
    path: Path,
    *,
    expected_payload: bytes | None = None,
    require_final: bool = False,
) -> None:
    """Finish one bounded, exact immutable-publication crash prefix."""

    exact_temp = re.compile(
        rf"^\.{re.escape(path.name)}\.[A-Za-z0-9_-]+\.tmp$"
    )
    def publication_candidates() -> list[Path]:
        try:
            selected = sorted(
                child
                for child in path.parent.iterdir()
                if exact_temp.fullmatch(child.name)
            )
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence publication is unreadable"
            ) from exc
        if len(selected) > 8:
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence has too many temporaries"
            )
        return selected

    def exact_regular(selected: Path) -> tuple[os.stat_result, bytes]:
        try:
            selected_stat = selected.lstat()
            selected_payload = selected.read_bytes()
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence object is unreadable"
            ) from exc
        if (
            selected.is_symlink()
            or not stat.S_ISREG(selected_stat.st_mode)
            or selected_stat.st_uid != os.geteuid()
            or int(selected_stat.st_size) > 262_144
            or (
                expected_payload is not None
                and selected_payload != expected_payload
            )
            or (
                expected_payload is None
                and not _immutable_receipt_payload_is_exact(
                    path,
                    selected_payload,
                )
            )
        ):
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence object is not exact"
            )
        return selected_stat, selected_payload

    def settled_final() -> tuple[os.stat_result, bytes] | None:
        """Return only a twice-observed exact final with no temp authority."""

        try:
            selected_stat, selected_payload = exact_regular(path)
        except DatabasePortalBridgeError:
            if not (path.exists() or path.is_symlink()):
                return None
            # The name may have appeared between the failed observation and
            # the existence check.  It still must pass the ordinary strict
            # object validation; malformed visible state never disappears
            # into a benign concurrent-writer classification.
            selected_stat, selected_payload = exact_regular(path)
        if int(selected_stat.st_nlink) != 1 or publication_candidates():
            return None
        observed_stat, observed_payload = exact_regular(path)
        if (
            int(observed_stat.st_nlink) != 1
            or observed_payload != selected_payload
            or observed_stat.st_dev != selected_stat.st_dev
            or observed_stat.st_ino != selected_stat.st_ino
            or publication_candidates()
        ):
            return None
        return observed_stat, observed_payload

    def exact_candidate(selected: Path) -> tuple[os.stat_result, bytes] | None:
        """Validate a listed temp, or accept only a fully settled takeover."""

        try:
            return exact_regular(selected)
        except DatabasePortalBridgeError:
            if not (selected.exists() or selected.is_symlink()):
                if settled_final() is not None:
                    return None
            raise

    candidates = publication_candidates()

    path_visible = path.exists() or path.is_symlink()
    if not path_visible:
        if not candidates:
            if require_final and settled_final() is None:
                raise DatabasePortalBridgeError(
                    "database Portal immutable evidence publication did not "
                    "produce a settled final object"
                )
            return
        candidate_records: list[tuple[Path, os.stat_result, bytes]] = []
        for candidate in candidates:
            candidate_record = exact_candidate(candidate)
            if candidate_record is None:
                # Another exact recovery consumed the listed temporary and
                # left the sole exact final.  Nothing from the stale listing
                # remains authoritative to promote or remove.
                return
            candidate_records.append((candidate, *candidate_record))
        first_payload = candidate_records[0][2]
        if any(
            int(candidate_stat.st_nlink) != 1
            or candidate_payload != first_payload
            for _candidate, candidate_stat, candidate_payload in candidate_records
        ):
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence temporaries are ambiguous"
            )
        try:
            try:
                os.link(candidate_records[0][0], path)
            except FileExistsError:
                # An identical concurrent writer won publication after this
                # process observed the final absent.  Continue only through
                # the same strict final/temp revalidation below.
                pass
            except FileNotFoundError:
                # A concurrent exact recovery may already have linked the
                # candidate and removed its temporary name.  That race is
                # admissible only if the strict convergence check below sees
                # one exact final and no unresolved publication temporary.
                pass
            _fsync_directory(path.parent)
            for candidate, _candidate_stat, _candidate_payload in candidate_records:
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    # A concurrent cleaner may have won this unlink.  Do not
                    # infer success from ENOENT itself; the final strict
                    # convergence check below is the only success authority.
                    continue
            _fsync_directory(path.parent)
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence promotion failed"
            ) from exc

    final_stat, final_payload = exact_regular(path)
    candidate_records = []
    for candidate in publication_candidates():
        candidate_record = exact_candidate(candidate)
        if candidate_record is None:
            return
        candidate_records.append((candidate, *candidate_record))
    final_links = int(final_stat.st_nlink)
    if final_links == 1:
        if any(
            int(candidate_stat.st_nlink) != 1
            or candidate_payload != final_payload
            for _candidate, candidate_stat, candidate_payload in candidate_records
        ):
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence has an unrelated temporary"
            )
    elif final_links == 2:
        linked_candidates = [
            (temporary_stat, temporary_payload)
            for _temporary, temporary_stat, temporary_payload in candidate_records
            if temporary_stat.st_dev == final_stat.st_dev
            and temporary_stat.st_ino == final_stat.st_ino
        ]
        if len(linked_candidates) != 1 or any(
            temporary_payload != final_payload
            or (
                int(temporary_stat.st_nlink) != 2
                if (
                    temporary_stat.st_dev == final_stat.st_dev
                    and temporary_stat.st_ino == final_stat.st_ino
                )
                else int(temporary_stat.st_nlink) != 1
            )
            for _temporary, temporary_stat, temporary_payload in candidate_records
        ):
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence temporary is not exact"
            )
    else:
        raise DatabasePortalBridgeError(
            "database Portal immutable evidence link count is ambiguous"
        )

    if candidate_records:
        try:
            for temporary, _temporary_stat, _temporary_payload in candidate_records:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    # As above, concurrent disappearance is acceptable only
                    # when the final/no-temp state converges exactly below.
                    continue
            _fsync_directory(path.parent)
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal immutable evidence temporary recovery failed"
            ) from exc
    settled = settled_final()
    if settled is None:
        raise DatabasePortalBridgeError(
            "database Portal immutable evidence recovery did not converge"
        )
    repaired_stat, repaired_payload = settled
    if (
        int(repaired_stat.st_nlink) != 1
        or repaired_payload != final_payload
        or repaired_stat.st_dev != final_stat.st_dev
        or repaired_stat.st_ino != final_stat.st_ino
    ):
        raise DatabasePortalBridgeError(
            "database Portal immutable evidence recovery changed authority"
        )


def _durably_revalidate_immutable_final_unlocked(
    path: Path,
    *,
    expected_payload: bytes | None,
) -> None:
    """Persist and reobserve the sole exact final before authority advances."""

    _recover_immutable_link_publication_unlocked(
        path,
        expected_payload=expected_payload,
        require_final=True,
    )
    _fsync_directory(path.parent)
    _fsync_directory(path.parent.parent)
    _recover_immutable_link_publication_unlocked(
        path,
        expected_payload=expected_payload,
        require_final=True,
    )


def _recover_immutable_link_publication(
    path: Path,
    *,
    expected_payload: bytes | None = None,
    require_final: bool = False,
) -> None:
    """Recover an exact immutable publication under its crash-released lock."""

    with _immutable_publication_lock(path):
        _cleanup_immutable_publication_stages_unlocked(path)
        _recover_immutable_link_publication_unlocked(
            path,
            expected_payload=expected_payload,
            require_final=require_final,
        )
        if path.exists() or path.is_symlink():
            _durably_revalidate_immutable_final_unlocked(
                path,
                expected_payload=expected_payload,
            )


def _line_value(value: Any) -> str:
    if isinstance(value, str):
        selected = value
    elif isinstance(value, Mapping):
        selected = _canonical_json(dict(value)).decode("utf-8")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        selected = ", ".join(_line_value(item) for item in value)
    else:
        selected = str(value or "")
    return " ".join(selected.replace("\x00", "").splitlines()).strip()


def _mapping_path(value: Mapping[str, Any]) -> str:
    return _line_value(
        value.get("path")
        or value.get("output")
        or value.get("artifact_id")
        or value.get("fluent_id")
        or value
    )


def _output_values(record: Any, body: Mapping[str, Any]) -> list[str]:
    raw = getattr(record, "outputs", ()) or body.get("outputs") or ()
    if isinstance(raw, (str, Mapping)):
        raw = (raw,)
    return list(
        dict.fromkeys(
            selected
            for item in raw
            if (
                selected := (
                    _mapping_path(item) if isinstance(item, Mapping) else _line_value(item)
                )
            )
        )
    )


def _validation_values(record: Any, body: Mapping[str, Any]) -> list[str]:
    raw = (
        getattr(record, "validations", ())
        or body.get("validations")
        or body.get("validation_commands")
        or body.get("validation")
        or ()
    )
    if isinstance(raw, (str, Mapping)):
        raw = (raw,)
    selected: list[str] = []
    for item in raw:
        if isinstance(item, Mapping):
            argv = item.get("argv")
            raw_policy = item.get("policy")
            if raw_policy is None:
                policy: Mapping[str, Any] = {}
            elif isinstance(raw_policy, Mapping):
                policy = raw_policy
            else:
                raise DatabasePortalBridgeError(
                    "database validation policy is malformed"
                )
            representation = str(
                policy.get(VALIDATION_REPRESENTATION_POLICY_KEY)
                or item.get(VALIDATION_REPRESENTATION_POLICY_KEY)
                or ""
            ).strip()
            if representation and representation not in _VALIDATION_REPRESENTATIONS:
                raise DatabasePortalBridgeError(
                    "database validation representation is unknown"
                )
            if isinstance(argv, Sequence) and not isinstance(
                argv, (str, bytes, bytearray, memoryview)
            ):
                parts = tuple(argv)
                if not parts or any(
                    not isinstance(part, str) or not part.strip()
                    for part in parts
                ):
                    raise DatabasePortalBridgeError(
                        "database validation argv is malformed"
                    )
                if representation == VALIDATION_SHELL_TEXT_REPRESENTATION:
                    if len(parts) != 1:
                        raise DatabasePortalBridgeError(
                            "shell_text validation must contain exactly one command"
                        )
                    # Shell-text is an already reviewed command program.  It
                    # must survive the database projection byte-for-text;
                    # shlex.join([program]) would quote the whole program as
                    # one missing executable.
                    value = _line_value(parts[0])
                else:
                    # Explicit argv, and legacy untyped rows, retain their
                    # shell-safe projection behavior.
                    value = shlex.join(parts)
            else:
                command = item.get("command") or item.get("value")
                if representation == VALIDATION_ARGV_REPRESENTATION:
                    raise DatabasePortalBridgeError(
                        "argv validation must provide an argv sequence"
                    )
                if (
                    representation == VALIDATION_SHELL_TEXT_REPRESENTATION
                    and not isinstance(
                    command, str
                    )
                ):
                    raise DatabasePortalBridgeError(
                        "shell_text validation must provide command text"
                    )
                value = _line_value(command or item)
        else:
            value = _line_value(item)
        if value and value not in selected:
            selected.append(value)
    return selected


def _acceptance_value(record: Any, body: Mapping[str, Any]) -> str:
    raw = (
        getattr(record, "acceptance", ())
        or body.get("acceptance")
        or body.get("completion_contract")
        or body.get("completion rule")
        or body.get("completion_rule")
        or ()
    )
    if isinstance(raw, (str, Mapping)):
        raw = (raw,)
    values: list[str] = []
    for item in raw:
        if isinstance(item, Mapping):
            value = _line_value(
                item.get("criterion") or item.get("statement") or item.get("value") or item
            )
        else:
            value = _line_value(item)
        if value:
            values.append(value)
    return " ; ".join(values)


def _projection_immutable_digest(text: str) -> str:
    normalized = _MUTABLE_PROJECTION_LINE.sub("- Status: <mutable>", text)
    return _sha256_bytes(normalized.encode("utf-8"))


def _projection_status(text: str) -> str:
    match = re.search(r"(?mi)^-\s*status\s*:\s*([^\r\n]+)$", text)
    return str(match.group(1) if match else "").strip().lower().replace("-", "_")


def _bounded_portal_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep control evidence while excluding raw provider/model payloads."""

    summary: dict[str, Any] = {}
    for key in (
        "task_count",
        "completed_count",
        "ready_count",
        "blocked_count",
        "active_task_id",
        "selection_idle_reason",
        "unchanged",
        "write_count",
        "blocked",
        "reason",
    ):
        if key in result:
            summary[key] = result[key]
    implementation = result.get("implementation_result")
    if isinstance(implementation, Mapping):
        summary["implementation"] = {
            key: implementation[key]
            for key in (
                "task_id",
                "attempt",
                "returncode",
                "reason",
                "deferred",
                "skipped",
                "implementation_commit",
                "branch",
                "merge_queued",
            )
            if key in implementation
        }
    reconciliation = result.get("merge_reconciliation")
    if isinstance(reconciliation, Sequence) and not isinstance(
        reconciliation, (str, bytes, bytearray, memoryview)
    ):
        summary["merge_reconciliation"] = [
            {
                key: item[key]
                for key in (
                    "task_id",
                    "returncode",
                    "reason",
                    "status",
                    "implementation_commit",
                    "merge_commit",
                    "resolved",
                )
                if key in item
            }
            for item in reconciliation[-8:]
            if isinstance(item, Mapping)
        ]
    return summary


class DatabasePortalExecutionBridge:
    """Run one database claim through a private Portal execution projection."""

    INTERFACE = DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE
    RECEIPT_SCHEMA = DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA

    def __init__(
        self,
        *,
        task_source: Any,
        attempt_root: Path | str,
        portal_factory: PortalDaemonFactory,
        task_header_prefix: str = "## ",
        max_passes: int = 4,
    ) -> None:
        if not callable(portal_factory):
            raise TypeError("portal_factory must be callable")
        if isinstance(max_passes, bool) or not isinstance(max_passes, int) or max_passes < 1:
            raise ValueError("max_passes must be a positive integer")
        self.task_source = task_source
        self.attempt_root = Path(attempt_root).absolute()
        self.portal_factory = portal_factory
        self.task_header_prefix = str(task_header_prefix or "## ")
        self.max_passes = max_passes
        self._binding_recorder: (
            Callable[[Any, Mapping[str, Any], str], Any] | None
        ) = None
        self._reconciliation_binding_recorder: (
            Callable[[Any, Mapping[str, Any], str], Any] | None
        ) = None
        self._binding_lookup: Callable[[Any], Mapping[str, Any] | None] | None = None

    def bind_attempt_binding_authority(
        self,
        *,
        recorder: Callable[[Any, Mapping[str, Any], str], Any],
        reconciliation_recorder: Callable[
            [Any, Mapping[str, Any], str], Any
        ],
        lookup: Callable[[Any], Mapping[str, Any] | None],
    ) -> None:
        if (
            not callable(recorder)
            or not callable(reconciliation_recorder)
            or not callable(lookup)
        ):
            raise TypeError("database Portal binding authority must be callable")
        if (
            self._binding_recorder is not None
            or self._reconciliation_binding_recorder is not None
            or self._binding_lookup is not None
        ):
            raise DatabasePortalBridgeError(
                "database Portal binding authority is already bound"
            )
        self._binding_recorder = recorder
        self._reconciliation_binding_recorder = reconciliation_recorder
        self._binding_lookup = lookup

    def _paths(self, attempt: Any) -> DatabasePortalAttemptPaths:
        attempt_key = hashlib.sha256(str(attempt.attempt_id).encode("utf-8")).hexdigest()[:24]
        root = self.attempt_root / attempt_key
        return DatabasePortalAttemptPaths(
            root=root,
            task_projection=root / "task-projection.md",
            binding=root / "database-attempt-binding.json",
            state=root / "portal-task-state.json",
            strategy=root / "portal-strategy.json",
            events=root / "portal-events.jsonl",
            implementation_logs=root / "implementation-logs",
            reconciliation=root / "database-attempt-reconciliations",
        )

    @staticmethod
    def _binding_without_identity(binding: Mapping[str, Any]) -> dict[str, Any]:
        unsigned = dict(binding)
        unsigned.pop("binding_id", None)
        return unsigned

    @classmethod
    def _verify_binding_identity(cls, binding: Mapping[str, Any]) -> None:
        """Verify the closed binding before it can nominate nested state."""

        if frozenset(str(key) for key in binding) != _ATTEMPT_BINDING_FIELDS:
            raise DatabasePortalBridgeError(
                "database Portal attempt binding has unknown or missing fields"
            )
        if (
            binding.get("schema") != DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA
            or binding.get("interface") != DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE
            or binding.get("authoritative_task_store") != "duckdb"
            or binding.get("projection_authority") is not False
        ):
            raise DatabasePortalBridgeError(
                "database Portal attempt binding has an unknown authority"
            )
        for field in (
            "attempt_id",
            "claim_id",
            "task_cid",
            "task_alias",
            "binding_id",
        ):
            if not isinstance(binding.get(field), str) or not str(
                binding.get(field) or ""
            ).strip():
                raise DatabasePortalBridgeError(
                    f"database Portal attempt binding lacks {field}"
                )
        for field in ("fencing_token", "fence_epoch", "task_revision"):
            value = binding.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DatabasePortalBridgeError(
                    f"database Portal attempt binding has malformed {field}"
                )
        expected_id = _sha256_bytes(
            _canonical_json(cls._binding_without_identity(binding))
        )
        if str(binding.get("binding_id") or "") != expected_id:
            raise DatabasePortalBridgeError(
                "database Portal attempt binding identity does not verify"
            )

    @staticmethod
    def _strict_state_record(
        path: Path,
    ) -> tuple[Mapping[str, Any] | None, str]:
        if not path.exists():
            return None, ""
        if path.is_symlink() or not path.is_file():
            raise DatabasePortalBridgeError(
                "database Portal nested task state is not a regular file"
            )

        def closed_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise DatabasePortalBridgeError(
                        "database Portal nested task state contains duplicate keys"
                    )
                result[key] = value
            return result

        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
            payload = json.loads(
                text,
                object_pairs_hook=closed_object,
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    DatabasePortalBridgeError(
                        "database Portal nested task state contains a nonfinite value"
                    )
                ),
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatabasePortalBridgeError(
                "database Portal nested task state is unreadable"
            ) from exc
        if not isinstance(payload, Mapping):
            raise DatabasePortalBridgeError(
                "database Portal nested task state is malformed"
            )
        from .implementation_daemon import PortalTaskState

        allowed_fields = {item.name for item in fields(PortalTaskState)}
        unknown = set(payload) - allowed_fields
        if unknown:
            raise DatabasePortalBridgeError(
                "database Portal nested task state has unknown fields"
            )
        defaults = PortalTaskState()
        nullable_integer_fields = {
            "last_implementation_returncode",
            "last_merge_returncode",
        }
        for state_field in fields(PortalTaskState):
            name = state_field.name
            if name not in payload:
                continue
            observed = payload[name]
            expected = getattr(defaults, name)
            valid = False
            if name in nullable_integer_fields:
                valid = observed is None or (
                    isinstance(observed, int) and not isinstance(observed, bool)
                )
            elif isinstance(expected, bool):
                valid = isinstance(observed, bool)
            elif isinstance(expected, int):
                valid = isinstance(observed, int) and not isinstance(
                    observed, bool
                )
            elif isinstance(expected, str):
                valid = isinstance(observed, str)
            elif isinstance(expected, list):
                valid = isinstance(observed, list) and all(
                    isinstance(item, str) for item in observed
                )
            elif isinstance(expected, dict):
                valid = isinstance(observed, dict)
            if not valid:
                raise DatabasePortalBridgeError(
                    "database Portal nested task state has malformed " + name
                )
        return payload, _sha256_bytes(raw)

    @staticmethod
    def _strict_state_payload(path: Path) -> Mapping[str, Any] | None:
        payload, _digest = DatabasePortalExecutionBridge._strict_state_record(
            path
        )
        return payload

    def _verify_nested_state_identity(
        self,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        identity: Mapping[str, str],
        *,
        payload: Mapping[str, Any] | None,
        state_digest: str,
    ) -> dict[str, Any]:
        """Bind a nonempty nested Portal state to its exact DB attempt."""

        if payload is None:
            return {
                "present": False,
                "state_path": str(paths.state),
                "state_digest": "",
                "active": False,
            }
        alias = str(binding.get("task_alias") or "")
        active_task_id = str(payload.get("active_task_id") or "")
        active_task_key = str(payload.get("active_task_key") or "")
        active_task_cid = str(payload.get("active_task_cid") or "")
        if active_task_id and active_task_id != alias:
            raise DatabasePortalBridgeError(
                "database Portal nested state names a different task"
            )
        if active_task_key and active_task_key != identity["canonical_task_key"]:
            raise DatabasePortalBridgeError(
                "database Portal nested state task key changed"
            )
        if active_task_cid and active_task_cid != identity["canonical_task_cid"]:
            raise DatabasePortalBridgeError(
                "database Portal nested state task CID changed"
            )
        task_identities = payload.get("task_identities") or {}
        if not isinstance(task_identities, Mapping):
            raise DatabasePortalBridgeError(
                "database Portal nested state task identities are malformed"
            )
        nested_identity = task_identities.get(alias)
        if nested_identity is not None:
            if not isinstance(nested_identity, Mapping) or any(
                str(nested_identity.get(field) or "") != expected
                for field, expected in identity.items()
                if field != "task_id"
            ):
                raise DatabasePortalBridgeError(
                    "database Portal nested state identity is not exact"
                )
        named_task_ids: set[str] = set()
        for field in (
            "completed_task_ids",
            "ready_task_ids",
            "selectable_ready_task_ids",
            "eligible_ready_task_ids",
            "waiting_task_ids",
            "blocked_task_ids",
        ):
            values = payload.get(field) or []
            if not isinstance(values, list):
                raise DatabasePortalBridgeError(
                    "database Portal nested task population is malformed"
                )
            named_task_ids.update(str(item) for item in values if str(item))
        statuses = payload.get("task_statuses") or {}
        if not isinstance(statuses, Mapping):
            raise DatabasePortalBridgeError(
                "database Portal nested task statuses are malformed"
            )
        named_task_ids.update(str(item) for item in statuses)
        named_task_ids.update(str(item) for item in task_identities)
        if named_task_ids - {alias}:
            raise DatabasePortalBridgeError(
                "database Portal nested state contains an unrelated task"
            )
        raw_attempt = payload.get("active_attempt", 0)
        if isinstance(raw_attempt, bool):
            raise DatabasePortalBridgeError(
                "database Portal nested active attempt is malformed"
            )
        try:
            active_attempt = int(raw_attempt or 0)
        except (TypeError, ValueError) as exc:
            raise DatabasePortalBridgeError(
                "database Portal nested active attempt is malformed"
            ) from exc
        if active_attempt < 0:
            raise DatabasePortalBridgeError(
                "database Portal nested active attempt is malformed"
            )
        active = bool(
            payload.get("implementation_in_progress")
            or active_task_id
            or active_attempt
            or payload.get("active_worktree_path")
            or payload.get("active_branch")
            or payload.get("active_provider_runner")
        )
        if active and (
            active_task_id != alias
            or active_task_key != identity["canonical_task_key"]
            or active_task_cid != identity["canonical_task_cid"]
            or not isinstance(nested_identity, Mapping)
        ):
            raise DatabasePortalBridgeError(
                "database Portal active nested state lacks its exact task identity"
            )
        return {
            "present": True,
            "state_path": str(paths.state),
            "state_digest": state_digest,
            "active": active,
            "active_task_id": active_task_id,
            "active_attempt": active_attempt,
            "active_phase": str(payload.get("active_phase") or ""),
            "active_phase_detail": str(
                payload.get("active_phase_detail") or ""
            ),
            "active_worktree_path": str(
                payload.get("active_worktree_path") or ""
            ),
            "active_branch": str(payload.get("active_branch") or ""),
        }

    @staticmethod
    def _validated_provider_runner_fence(
        raw: Any,
    ) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise DatabasePortalBridgeError(
                "database Portal provider fence returned a non-object"
            )
        result = dict(raw)
        if set(result) - _PROVIDER_RUNNER_FENCE_FIELDS:
            raise DatabasePortalBridgeError(
                "database Portal provider fence returned unknown fields"
            )
        for field in ("applicable", "safe_to_restart", "fenced"):
            if not isinstance(result.get(field), bool):
                raise DatabasePortalBridgeError(
                    "database Portal provider fence has malformed " + field
                )
        reason = result.get("reason")
        if (
            not isinstance(reason, str)
            or not reason
            or len(reason) > 256
            or any(character in reason for character in "\0\n\r")
        ):
            raise DatabasePortalBridgeError(
                "database Portal provider fence has malformed reason"
            )
        for field in ("pid", "parent_pid_before_fence"):
            if field not in result:
                continue
            value = result[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DatabasePortalBridgeError(
                    "database Portal provider fence has malformed " + field
                )
        if "pid_reused" in result and not isinstance(
            result["pid_reused"], bool
        ):
            raise DatabasePortalBridgeError(
                "database Portal provider fence has malformed pid_reused"
            )
        if result["fenced"] is True and (
            result["applicable"] is not True
            or result["safe_to_restart"] is not True
        ):
            raise DatabasePortalBridgeError(
                "database Portal provider fence contradicts its authority"
            )
        return result

    def validate_active_attempt_roots(
        self,
        attempts: Sequence[Any],
    ) -> dict[str, str]:
        """Validate only DB-nominated roots for the exact running attempts.

        The attempt-id hash is the one path-selection authority.  Historical,
        sibling, and merely similar directories are never scanned and cannot
        nominate a process, lifecycle record, or database mutation.
        """

        if self.attempt_root.is_symlink() or (
            self.attempt_root.exists() and not self.attempt_root.is_dir()
        ):
            raise DatabasePortalBridgeError(
                "database Portal attempt authority is not a regular directory"
            )
        selected: dict[str, str] = {}
        for attempt in attempts:
            attempt_id = str(attempt.attempt_id)
            paths = self._paths(attempt)
            has_any = any(
                path.exists()
                for path in (paths.binding, paths.task_projection, paths.state)
            )
            if not has_any:
                continue
            if not paths.binding.exists():
                raise DatabasePortalBridgeError(
                    "database Portal active attempt lacks its exact binding"
                )
            payload = self._read_binding(paths.binding)
            self._verify_binding_identity(payload)
            if str(payload.get("attempt_id") or "") != attempt_id:
                raise DatabasePortalBridgeError(
                    "database Portal active attempt binding changed identity"
                )
            selected[attempt_id] = str(paths.binding)
        return selected

    @staticmethod
    def _record_for_attempt(task_source: Any, attempt: Any) -> Any:
        getter = getattr(task_source, "get_task", None) or getattr(task_source, "get", None)
        if not callable(getter):
            raise DatabasePortalBridgeError("database task source does not expose get_task()")
        record = getter(str(attempt.task_cid))
        if record is None:
            raise DatabasePortalBridgeError(
                f"claimed database task {attempt.task_cid!r} disappeared"
            )
        if str(getattr(record, "task_cid", "")) != str(attempt.task_cid):
            raise DatabasePortalBridgeError("database task identity changed")
        attempt_alias = str(getattr(attempt, "task_alias", "") or "")
        record_alias = str(getattr(record, "task_alias", "") or "")
        if attempt_alias and record_alias and attempt_alias != record_alias:
            raise DatabasePortalBridgeError("database task alias changed")
        return record

    def _binding(self, attempt: Any, record: Any, seed: str) -> dict[str, Any]:
        body = dict(getattr(record, "body", {}) or {})
        payload = {
            "schema": DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA,
            "interface": self.INTERFACE,
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "task_cid": str(attempt.task_cid),
            "task_alias": str(
                getattr(record, "task_alias", "")
                or getattr(attempt, "task_alias", "")
                or attempt.task_cid
            ),
            "goal_cid": str(getattr(record, "goal_cid", "") or ""),
            "plan_cid": str(getattr(record, "plan_cid", "") or ""),
            "task_revision": int(getattr(record, "revision", 0) or 0),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "lease_id": str(getattr(attempt, "lease_id", "") or ""),
            "task_body_digest": _sha256_bytes(_canonical_json(body)),
            "projection_seed_digest": _sha256_bytes(seed.encode("utf-8")),
            "projection_immutable_digest": _projection_immutable_digest(seed),
            "authoritative_task_store": "duckdb",
            "projection_authority": False,
        }
        payload["binding_id"] = _sha256_bytes(_canonical_json(payload))
        return payload

    def _render_projection(self, attempt: Any, record: Any) -> str:
        body = dict(getattr(record, "body", {}) or {})
        alias = _line_value(
            getattr(record, "task_alias", "")
            or getattr(attempt, "task_alias", "")
            or attempt.task_cid
        )
        if not alias or any(character.isspace() for character in alias):
            raise DatabasePortalBridgeError("database task alias is not projection-safe")
        title = _line_value(
            body.get("objective") or body.get("title") or body.get("description") or alias
        )
        outputs = _output_values(record, body)
        validations = _validation_values(record, body)
        acceptance = _acceptance_value(record, body)
        priority = _line_value(
            getattr(record, "priority", "") or body.get("priority") or "P2"
        )
        reserved = {
            "status",
            "completion",
            "priority",
            "track",
            "depends on",
            "depends_on",
            "outputs",
            "validation",
            "validations",
            "validation_commands",
            "acceptance",
        }
        lines = [
            "# Database attempt projection (non-authoritative)",
            "",
            f"## {alias} {title}",
            "",
            "- Status: ready",
            f"- Completion: {_line_value(body.get('completion') or 'auto')}",
            f"- Priority: {priority}",
            f"- Track: {_line_value(body.get('track') or 'implementation')}",
            "- Depends on:",
            f"- Outputs: {', '.join(outputs)}",
            f"- Validation: {' ; '.join(validations)}",
            f"- Acceptance: {acceptance}",
            f"- Database task CID: {_line_value(attempt.task_cid)}",
            f"- Database attempt ID: {_line_value(attempt.attempt_id)}",
            f"- Database claim ID: {_line_value(attempt.claim_id)}",
            f"- Database dependency CIDs: {_line_value(getattr(record, 'dependencies', ()))}",
            "- Projection authority: false",
        ]
        for key in sorted(body):
            normalized = str(key).strip().lower().replace("_", " ")
            if not normalized or normalized in reserved:
                continue
            if "credential" in normalized or "secret" in normalized:
                continue
            value = _line_value(body[key])
            if value:
                label = " ".join(word.capitalize() for word in normalized.split())
                lines.append(f"- {label}: {value}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _read_binding(path: Path) -> Mapping[str, Any]:
        def closed_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise DatabasePortalBridgeError(
                        "database Portal attempt binding contains duplicate keys"
                    )
                result[key] = value
            return result

        try:
            value = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=closed_object,
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    DatabasePortalBridgeError(
                        "database Portal attempt binding contains a nonfinite value"
                    )
                ),
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise DatabasePortalBridgeError(
                "database Portal attempt binding is unreadable"
            ) from exc
        if not isinstance(value, Mapping):
            raise DatabasePortalBridgeError("database Portal attempt binding is malformed")
        return value

    def _ensure_attempt_projection(
        self,
        attempt: Any,
        record: Any,
        *,
        admit_before_publish: bool = False,
    ) -> tuple[DatabasePortalAttemptPaths, Mapping[str, Any]]:
        paths = self._paths(attempt)
        seed = self._render_projection(attempt, record)
        expected = self._binding(attempt, record, seed)
        durable_binding = (
            self._binding_lookup(attempt)
            if admit_before_publish and self._binding_lookup is not None
            else None
        )
        durable_stage = str((durable_binding or {}).get("stage") or "")
        if isinstance(durable_binding, Mapping):
            if (
                durable_binding.get("binding_id") != expected.get("binding_id")
                or durable_binding.get("projection_immutable_digest")
                != expected.get("projection_immutable_digest")
                or durable_stage not in {"prepared", "published", "portal_entered"}
            ):
                raise DatabasePortalBridgeError(
                    "database Portal durable binding changed across resume"
                )
        binding_present = paths.binding.exists()
        projection_present = paths.task_projection.exists()
        if paths.root.exists() and (
            paths.root.is_symlink() or not paths.root.is_dir()
        ):
            raise DatabasePortalBridgeError(
                "database Portal attempt directory is not an exact directory"
            )
        if paths.binding.is_symlink() or paths.task_projection.is_symlink():
            raise DatabasePortalBridgeError(
                "database Portal attempt projection contains a symlink"
            )
        if binding_present:
            observed = self._read_binding(paths.binding)
            if observed != expected:
                raise DatabasePortalBridgeError(
                    "database Portal attempt binding changed across resume"
                )
        if projection_present:
            self._verify_projection(paths, expected)
        if isinstance(durable_binding, Mapping) and durable_stage in {
            "published",
            "portal_entered",
        }:
            if not binding_present or not projection_present:
                raise DatabasePortalBridgeError(
                    "database Portal published artifacts disappeared before resume"
                )
            return paths, expected
        if isinstance(durable_binding, Mapping) and durable_stage == "prepared":
            if projection_present and not binding_present:
                raise DatabasePortalBridgeError(
                    "database Portal prepared projection lacks its binding"
                )
        elif binding_present or projection_present:
            if not binding_present or not projection_present:
                raise DatabasePortalBridgeError(
                    "database Portal unadmitted projection is partial"
                )
            if admit_before_publish and self._binding_recorder is not None:
                # A released predecessor may already have entered Portal.  A
                # complete exact legacy pair is therefore admitted directly
                # as entered, never retrospectively called pre-Portal.
                self._binding_recorder(attempt, expected, "portal_entered")
            return paths, expected
        if (
            admit_before_publish
            and self._binding_recorder is not None
            and not isinstance(durable_binding, Mapping)
        ):
            # The exact fenced database admission is durable before either
            # filesystem artifact is published.  A crash can therefore never
            # leave an unadmitted self-hashed projection as historical cleanup
            # authority, and Portal construction remains strictly later.
            self._binding_recorder(attempt, expected, "prepared")
        _ensure_durable_directory(paths.root)
        if not paths.binding.exists():
            _atomic_write(
                paths.binding,
                json.dumps(expected, indent=2, sort_keys=True).encode("utf-8") + b"\n",
            )
        if not paths.task_projection.exists():
            _atomic_write(paths.task_projection, seed.encode("utf-8"))
        self._verify_projection(paths, expected)
        if (
            admit_before_publish
            and self._binding_recorder is not None
            and durable_stage != "published"
        ):
            self._binding_recorder(attempt, expected, "published")
        return paths, expected

    @staticmethod
    def _verify_projection(paths: DatabasePortalAttemptPaths, binding: Mapping[str, Any]) -> str:
        try:
            text = paths.task_projection.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise DatabasePortalBridgeError("Portal task projection is unreadable") from exc
        if _projection_immutable_digest(text) != str(
            binding.get("projection_immutable_digest") or ""
        ):
            raise DatabasePortalBridgeError(
                "Portal task projection changed outside its mutable status field"
            )
        headers = _HEADER.findall(text)
        if headers != [str(binding.get("task_alias") or "")]:
            raise DatabasePortalBridgeError(
                "Portal task projection no longer contains exactly the claimed task"
            )
        return text

    def _projection_task_identity(
        self,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        projection_text: str | None = None,
    ) -> dict[str, str]:
        """Return the current Portal authority's identity for the sealed task.

        Importing the parser locally avoids a module-import cycle: the Portal
        daemon imports this bridge only while constructing a database-backed
        execution route.  Parsing the attempt-local projection is important;
        reproducing the identity here would create a second task-identity
        authority and could drift from the completion event producer.
        """

        from .implementation_daemon import parse_task_text

        text = (
            projection_text
            if projection_text is not None
            else self._verify_projection(paths, binding)
        )
        alias = str(binding.get("task_alias") or "")
        try:
            tasks = parse_task_text(
                text,
                path=paths.task_projection,
                # The database claim already chose one exact alias.  Parsing
                # by that alias is stricter than a board-wide family prefix
                # and also works for direct bridge users whose outer parser
                # prefix is unrelated to the claimed task family.
                task_header_prefix=f"## {alias}",
            )
        except (TypeError, ValueError) as exc:
            raise DatabasePortalBridgeError(
                "Portal task projection identity is malformed"
            ) from exc
        if len(tasks) != 1 or tasks[0].task_id != alias:
            raise DatabasePortalBridgeError(
                "Portal task projection identity does not match the claimed task"
            )
        task = tasks[0]
        identity = {
            "task_id": task.task_id,
            "canonical_task_key": str(task.canonical_task_key or ""),
            "canonical_task_cid": str(task.canonical_task_cid or ""),
            "board_namespace": str(task.board_namespace or ""),
        }
        if any(not value for value in identity.values()):
            raise DatabasePortalBridgeError(
                "Portal task projection lacks a complete canonical identity"
            )
        return identity

    @staticmethod
    def _has_completion_event(
        paths: DatabasePortalAttemptPaths,
        identity: Mapping[str, str],
    ) -> bool:
        if not paths.events.is_file():
            return False
        try:
            lines = paths.events.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            return False
        for line in reversed(lines[-4096:]):
            try:
                event = json.loads(line)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if (
                isinstance(event, Mapping)
                and event.get("type") == "task_completed"
                and all(
                    str(event.get(field) or "") == expected
                    for field, expected in identity.items()
                )
            ):
                return True
        return False

    @staticmethod
    def _terminal_failure(result: Mapping[str, Any]) -> str:
        if result.get("blocked") is True:
            return str(result.get("reason") or "portal_execution_blocked")
        implementation = result.get("implementation_result")
        if not isinstance(implementation, Mapping):
            return ""
        if implementation.get("deferred") is True:
            return str(implementation.get("reason") or "portal_execution_deferred")
        returncode = implementation.get("returncode")
        if isinstance(returncode, int) and not isinstance(returncode, bool) and returncode != 0:
            return str(implementation.get("reason") or "portal_provider_failed")
        if implementation.get("skipped") is True:
            return str(implementation.get("reason") or "portal_execution_skipped")
        return ""

    def _acceptance_receipt(
        self,
        *,
        attempt: Any,
        paths: DatabasePortalAttemptPaths,
        binding: Mapping[str, Any],
        summaries: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        alias = str(binding.get("task_alias") or "")
        projection_text = self._verify_projection(paths, binding)
        identity = self._projection_task_identity(
            paths,
            binding,
            projection_text,
        )
        if _projection_status(projection_text) not in _TERMINAL_STATUSES:
            raise DatabasePortalBridgeDeferred("Portal task projection is not complete")
        if not self._has_completion_event(paths, identity):
            raise DatabasePortalBridgeError(
                "Portal completion lacks an exact canonical task_completed event"
            )
        evidence = {
            "binding_id": str(binding.get("binding_id") or ""),
            "task_cid": str(attempt.task_cid),
            "task_alias": alias,
            "canonical_task_key": identity["canonical_task_key"],
            "canonical_task_cid": identity["canonical_task_cid"],
            "board_namespace": identity["board_namespace"],
            "attempt_id": str(attempt.attempt_id),
            "projection_digest": _sha256_bytes(projection_text.encode("utf-8")),
            "projection_immutable_digest": str(binding.get("projection_immutable_digest") or ""),
            "state_digest": _sha256_file(paths.state) if paths.state.is_file() else "",
            "events_digest": _sha256_file(paths.events),
            "portal_passes": [dict(item) for item in summaries],
        }
        evidence_digest = _sha256_bytes(_canonical_json(evidence))
        receipt = {
            "schema": self.RECEIPT_SCHEMA,
            "interface": self.INTERFACE,
            "status": "succeeded",
            "provider": "PortalImplementationDaemon",
            "execution_mode": "database-authoritative-portal-bridge",
            "accepted": True,
            "completion_authority": "DatabaseImplementationDaemon",
            "task_cid": str(attempt.task_cid),
            "task_alias": alias,
            "canonical_task_key": identity["canonical_task_key"],
            "canonical_task_cid": identity["canonical_task_cid"],
            "board_namespace": identity["board_namespace"],
            "attempt_id": str(attempt.attempt_id),
            "binding_id": str(binding.get("binding_id") or ""),
            "evidence_digest": evidence_digest,
            "portal_evidence": evidence,
        }
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        return receipt

    def run_provider(self, attempt: Any) -> Mapping[str, Any]:
        """Run bounded real Portal passes and return only accepted evidence."""

        record = self._record_for_attempt(self.task_source, attempt)
        try:
            paths, binding = self._ensure_attempt_projection(
                attempt,
                record,
                admit_before_publish=True,
            )
        except DatabasePortalPreEntryPublicationDeferred:
            raise
        except (OSError, TimeoutError, ConnectionError) as exc:
            durable = (
                self._binding_lookup(attempt)
                if self._binding_lookup is not None
                else None
            )
            stage = str((durable or {}).get("stage") or "")
            if stage in {"prepared", "published"}:
                raise DatabasePortalPreEntryPublicationDeferred(
                    "database_portal_preentry_" + stage
                ) from exc
            raise
        if self._binding_recorder is not None:
            # This durable monotonic transition precedes Portal construction;
            # any later absence or partial artifact set is therefore unknown,
            # never evidence of a safe pre-provider retry.
            try:
                self._binding_recorder(attempt, binding, "portal_entered")
            except DatabasePortalPreEntryPublicationDeferred:
                raise
            except (OSError, TimeoutError, ConnectionError) as exc:
                durable = (
                    self._binding_lookup(attempt)
                    if self._binding_lookup is not None
                    else None
                )
                if str((durable or {}).get("stage") or "") == "published":
                    raise DatabasePortalPreEntryPublicationDeferred(
                        "database_portal_preentry_published"
                    ) from exc
                raise
        summaries: list[Mapping[str, Any]] = []
        daemon = self.portal_factory(
            paths,
            str(binding.get("task_alias") or attempt.task_cid),
        )
        if daemon is None or not callable(getattr(daemon, "run_once", None)):
            raise DatabasePortalBridgeError(
                "portal_factory did not return a Portal-compatible daemon"
            )
        try:
            for _pass_index in range(self.max_passes):
                projection = self._verify_projection(paths, binding)
                identity = self._projection_task_identity(
                    paths,
                    binding,
                    projection,
                )
                if _projection_status(
                    projection
                ) in _TERMINAL_STATUSES and self._has_completion_event(
                    paths, identity
                ):
                    return self._acceptance_receipt(
                        attempt=attempt,
                        paths=paths,
                        binding=binding,
                        summaries=summaries,
                    )
                raw_result = daemon.run_once()
                if not isinstance(raw_result, Mapping):
                    raise DatabasePortalBridgeError("Portal daemon returned a non-object result")
                summary = _bounded_portal_result(raw_result)
                summaries.append(summary)
                self._verify_projection(paths, binding)
                failure = self._terminal_failure(raw_result)
                if failure:
                    if (
                        "deferred" in failure
                        or "backoff" in failure
                        or "capacity" in failure
                        or "resource_claim" in failure
                        or failure
                        in {
                            "inflight_process",
                            "inflight_process_missing",
                            "worktree_lifecycle_claim_exists",
                        }
                    ):
                        raise DatabasePortalBridgeDeferred(failure)
                    raise DatabasePortalBridgeError(failure)
            return self._acceptance_receipt(
                attempt=attempt,
                paths=paths,
                binding=binding,
                summaries=summaries,
            )
        finally:
            close = getattr(daemon, "close_event_runtime", None) or getattr(daemon, "close", None)
            if callable(close):
                close()

    def recover_provider_result(self, attempt: Any) -> Mapping[str, Any] | None:
        """Recover an exact terminal Portal receipt without dispatching a model.

        The outer database daemon calls this only after it finds a durable
        provider-dispatch marker with no matching provider phase.  Recovery is
        deliberately read-only: absent or incomplete Portal evidence returns
        ``None`` and the outer attempt fails closed rather than reimplementing
        work whose outcome is unknown.
        """

        record = self._record_for_attempt(self.task_source, attempt)
        paths = self._paths(attempt)
        if not paths.binding.is_file() or not paths.task_projection.is_file():
            return None
        seed = self._render_projection(attempt, record)
        expected = self._binding(attempt, record, seed)
        observed = self._read_binding(paths.binding)
        if observed != expected:
            raise DatabasePortalBridgeError(
                "database Portal attempt binding changed before recovery"
            )
        projection = self._verify_projection(paths, expected)
        identity = self._projection_task_identity(paths, expected, projection)
        if (
            _projection_status(projection) not in _TERMINAL_STATUSES
            or not self._has_completion_event(paths, identity)
        ):
            return None
        return self._acceptance_receipt(
            attempt=attempt,
            paths=paths,
            binding=expected,
            summaries=(),
        )

    def reconcile_quiesced_attempt(self, attempt: Any) -> dict[str, Any]:
        """Reconcile the exact nested Portal state for one active DB attempt.

        The database attempt id selects one content-stable directory.  A
        sibling directory, task alias, or merely similar Portal state can
        never nominate work for cleanup.  Missing artifacts are an explicit
        pre-provider state; partial or mismatched artifacts fail closed.
        """

        paths = self._paths(attempt)
        try:
            confined_parent = paths.root.parent.resolve()
            authority_root = self.attempt_root.resolve()
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal attempt root is unavailable"
            ) from exc
        if (
            confined_parent != authority_root
            or self.attempt_root.is_symlink()
            or (
                self.attempt_root.exists()
                and not self.attempt_root.is_dir()
            )
            or paths.root.is_symlink()
            or any(
                path.is_symlink()
                for path in (
                    paths.binding,
                    paths.task_projection,
                    paths.state,
                )
            )
        ):
            raise DatabasePortalBridgeError(
                "database Portal attempt artifacts escape their exact root"
            )
        present = {
            "binding": paths.binding.exists(),
            "projection": paths.task_projection.exists(),
            "state": paths.state.exists(),
        }
        durable_binding = (
            self._binding_lookup(attempt)
            if self._binding_lookup is not None
            else None
        )
        record: Any | None = None
        seed = ""
        expected: Mapping[str, Any] | None = None
        try:
            record = self._record_for_attempt(self.task_source, attempt)
            seed = self._render_projection(attempt, record)
            expected = self._binding(attempt, record, seed)
        except DatabasePortalBridgeError:
            # A deleted/replaced canonical projection cannot authorize new
            # files, but a prior exact DB admission may still prove that the
            # provider boundary had not been crossed.
            record = None
        admitted_current = bool(
            isinstance(durable_binding, Mapping)
            and isinstance(expected, Mapping)
            and durable_binding.get("binding_id") == expected.get("binding_id")
            and durable_binding.get("projection_immutable_digest")
            == expected.get("projection_immutable_digest")
        )
        durable_stage = str((durable_binding or {}).get("stage") or "")

        def preportal_result(*, historical: bool) -> dict[str, Any]:
            return {
                "reconciled": True,
                "blocked": False,
                "reason": "admitted_preportal_artifacts_absent",
                "attempt_id": str(attempt.attempt_id),
                "claim_id": str(attempt.claim_id),
                "task_cid": str(attempt.task_cid),
                "task_alias": str(getattr(attempt, "task_alias", "") or ""),
                "attempt_number": int(attempt.attempt_number),
                "owner_session_id": str(attempt.owner_session_id),
                "attempt_root": str(paths.root),
                "binding_id": str(
                    (durable_binding or {}).get("binding_id") or ""
                ),
                "historical_binding": bool(historical),
                "nested_state": {
                    "present": False,
                    "state_path": str(paths.state),
                    "state_digest": "",
                    "active": False,
                },
                "provider_runner_fence": {
                    "applicable": False,
                    "safe_to_restart": True,
                    "fenced": False,
                    "reason": "admitted_preportal_boundary",
                },
                "provider_runner_reconciliation_authority": "not_applicable",
                "terminal_provider_evidence": False,
                "terminal_provider_receipt_id": "",
            }

        known_preportal_names = {
            paths.binding.name,
            paths.task_projection.name,
        }
        recoverable_temps: list[Path] = []
        unexpected_preportal_children: list[Path] = []
        invalid_reconciliation_store = False
        if paths.root.is_dir():
            temp_name = re.compile(
                r"^\.(?:database-attempt-binding\.json|task-projection\.md)\."
                r"[A-Za-z0-9_-]+\.tmp$"
            )
            for child in paths.root.iterdir():
                if child == paths.reconciliation:
                    if child.is_symlink() or not child.is_dir():
                        unexpected_preportal_children.append(child)
                        invalid_reconciliation_store = True
                        continue
                    try:
                        receipt_paths = sorted(child.iterdir())
                    except OSError:
                        unexpected_preportal_children.append(child)
                        invalid_reconciliation_store = True
                        continue
                    if len(receipt_paths) > 256:
                        unexpected_preportal_children.append(child)
                        invalid_reconciliation_store = True
                        continue
                    try:
                        recovery_targets: set[Path] = set()
                        for receipt_path in receipt_paths:
                            if re.fullmatch(
                                r"[0-9a-f]{64}\.json",
                                receipt_path.name,
                            ):
                                recovery_targets.add(receipt_path)
                                continue
                            temporary_match = re.fullmatch(
                                r"\.([0-9a-f]{64}\.json)\."
                                r"[A-Za-z0-9_-]+\.(?:tmp|stage)",
                                receipt_path.name,
                            )
                            if temporary_match is not None:
                                # A SIGKILL may leave either a non-authoritative
                                # stage or a fully fsynced ready temp without a
                                # final pathname.  Derive only the closed
                                # content-addressed final name; locked recovery
                                # discards safe stages and strictly validates
                                # ready bytes before promotion.
                                recovery_targets.add(
                                    child / temporary_match.group(1)
                                )
                        for receipt_path in sorted(recovery_targets):
                            _recover_immutable_link_publication(receipt_path)
                        receipt_paths = sorted(child.iterdir())
                    except (OSError, DatabasePortalBridgeError):
                        unexpected_preportal_children.append(child)
                        invalid_reconciliation_store = True
                        continue
                    if len(receipt_paths) > 128:
                        unexpected_preportal_children.append(child)
                        invalid_reconciliation_store = True
                        continue
                    valid_receipts = True
                    for receipt_path in receipt_paths:
                        match = re.fullmatch(r"([0-9a-f]{64})\.json", receipt_path.name)
                        try:
                            receipt_stat = receipt_path.lstat()
                        except OSError:
                            valid_receipts = False
                            break
                        if (
                            match is None
                            or receipt_path.is_symlink()
                            or not receipt_path.is_file()
                            or int(receipt_stat.st_nlink) != 1
                            or int(receipt_stat.st_size) > 1024 * 1024
                        ):
                            valid_receipts = False
                            break
                        try:
                            self.load_reconciliation_receipt(
                                attempt,
                                "sha256:" + match.group(1),
                            )
                        except DatabasePortalBridgeError:
                            valid_receipts = False
                            break
                    if not valid_receipts:
                        unexpected_preportal_children.append(child)
                        invalid_reconciliation_store = True
                    continue
                if child.name in known_preportal_names:
                    continue
                try:
                    stat = child.lstat()
                except OSError:
                    unexpected_preportal_children.append(child)
                    continue
                if (
                    temp_name.fullmatch(child.name)
                    and child.is_file()
                    and not child.is_symlink()
                    and int(stat.st_nlink) == 1
                    and int(stat.st_size) <= 1024 * 1024
                    and len(recoverable_temps) < 8
                ):
                    recoverable_temps.append(child)
                else:
                    unexpected_preportal_children.append(child)
        if invalid_reconciliation_store:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation evidence store is not exact"
            )
        clean_preportal_root = bool(
            not paths.root.exists()
            or (
                paths.root.is_dir()
                and not unexpected_preportal_children
            )
        )
        if (
            isinstance(durable_binding, Mapping)
            and durable_stage == "prepared"
            and clean_preportal_root
        ):
            for temporary in recoverable_temps:
                try:
                    temporary.unlink()
                except OSError as exc:
                    raise DatabasePortalBridgeError(
                        "admitted database Portal temporary artifact could not "
                        "be quarantined"
                    ) from exc
            if recoverable_temps:
                directory_fd = os.open(paths.root, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        if not any(present.values()):
            if isinstance(durable_binding, Mapping):
                if durable_stage != "prepared" or not clean_preportal_root:
                    raise DatabasePortalBridgeError(
                        "database Portal artifacts disappeared after its "
                        "published or entered boundary"
                    )
                return preportal_result(historical=not admitted_current)
            else:
                return {
                    "reconciled": True,
                    "blocked": False,
                    "reason": "portal_attempt_artifacts_absent",
                    "attempt_id": str(attempt.attempt_id),
                    "claim_id": str(attempt.claim_id),
                    "task_cid": str(attempt.task_cid),
                    "task_alias": str(getattr(attempt, "task_alias", "") or ""),
                    "attempt_root": str(paths.root),
                    "binding_id": "",
                    "nested_state": {
                        "present": False,
                        "state_path": str(paths.state),
                        "state_digest": "",
                        "active": False,
                    },
                    "terminal_provider_evidence": False,
                }
        if present["binding"] and not present["projection"]:
            if (
                not isinstance(durable_binding, Mapping)
                or durable_stage != "prepared"
                or not clean_preportal_root
            ):
                raise DatabasePortalBridgeError(
                    "database Portal active attempt has partial binding artifacts"
                )
            observed_preportal = self._read_binding(paths.binding)
            self._verify_binding_identity(observed_preportal)
            if (
                durable_binding.get("binding_id")
                != observed_preportal.get("binding_id")
                or durable_binding.get("projection_immutable_digest")
                != observed_preportal.get("projection_immutable_digest")
            ):
                raise DatabasePortalBridgeError(
                    "database Portal partial binding lacks exact DB admission"
                )
            if admitted_current and observed_preportal == expected:
                # The durable prepared admission proves Portal construction
                # has not begun.  Recreate only the missing immutable
                # projection from the still-current canonical task; this is
                # preparation repair, not provider execution.
                _atomic_write(paths.task_projection, seed.encode("utf-8"))
                self._verify_projection(paths, expected)
                return preportal_result(historical=False)
            else:
                control_claim = dict(
                    getattr(attempt, "body", {}).get("control_claim") or {}
                )
                historical_expected = {
                    "attempt_id": str(attempt.attempt_id),
                    "claim_id": str(attempt.claim_id),
                    "task_cid": str(attempt.task_cid),
                    "task_alias": str(getattr(attempt, "task_alias", "") or ""),
                    "task_revision": int(control_claim.get("revision") or 0),
                    "fencing_token": int(attempt.fencing_token),
                    "fence_epoch": int(attempt.fence_epoch),
                    "lease_id": str(attempt.lease_id),
                }
                if any(
                    observed_preportal.get(field) != value
                    for field, value in historical_expected.items()
                ):
                    raise DatabasePortalBridgeError(
                        "database Portal partial historical binding changed "
                        "attempt authority"
                    )
                return preportal_result(historical=True)
        if present["projection"] and not present["binding"]:
            raise DatabasePortalBridgeError(
                "database Portal active attempt has partial binding artifacts"
            )
        if not present["binding"] or not present["projection"]:
            raise DatabasePortalBridgeError(
                "database Portal active attempt has partial binding artifacts"
            )
        observed = self._read_binding(paths.binding)
        self._verify_binding_identity(observed)
        expected_root = self.attempt_root / hashlib.sha256(
            str(observed.get("attempt_id") or "").encode("utf-8")
        ).hexdigest()[:24]
        if paths.root != expected_root:
            raise DatabasePortalBridgeError(
                "database Portal attempt binding is stored under the wrong root"
            )
        if isinstance(durable_binding, Mapping) and durable_stage in {
            "prepared",
            "published",
        }:
            if not clean_preportal_root or present["state"]:
                raise DatabasePortalBridgeError(
                    "database Portal pre-entry binding has unexpected active "
                    "artifacts"
                )
            if (
                durable_binding.get("binding_id")
                != observed.get("binding_id")
                or durable_binding.get("projection_immutable_digest")
                != observed.get("projection_immutable_digest")
            ):
                raise DatabasePortalBridgeError(
                    "database Portal pre-entry binding changed DB admission"
                )
            return preportal_result(historical=observed != expected)
        if (
            not isinstance(durable_binding, Mapping)
            and self._reconciliation_binding_recorder is not None
        ):
            # One-way migration for the released predecessor: full historical
            # artifacts for the still-current task may already have crossed
            # the Portal boundary.  Admit them directly as entered; never
            # reinterpret them as prelaunch.  A replaced/deleted task cannot
            # validate a pre-admission filesystem pair and therefore remains
            # fail closed.
            if expected is None or observed != expected:
                raise DatabasePortalBridgeError(
                    "database Portal predecessor binding lacks durable admission"
                )
            self._reconciliation_binding_recorder(
                attempt,
                observed,
                "portal_entered",
            )
            durable_binding = self._binding_lookup(attempt)
            durable_stage = str((durable_binding or {}).get("stage") or "")
        if isinstance(durable_binding, Mapping) and durable_stage != "portal_entered":
            raise DatabasePortalBridgeError(
                "database Portal binding has an unknown execution stage"
            )
        historical_binding = expected is None or observed != expected
        if historical_binding:
            # A legitimate canonical task replacement cannot reproduce the
            # old body/projection digests.  Validate the sealed historical
            # projection against the running attempt's immutable control
            # claim instead; the database daemon later decides whether that
            # exact old attempt is superseded.  This never makes the old
            # projection authoritative for the replacement task.
            control_claim = dict(
                getattr(attempt, "body", {}).get("control_claim") or {}
            )
            historical_expected = {
                "attempt_id": str(attempt.attempt_id),
                "claim_id": str(attempt.claim_id),
                "task_cid": str(attempt.task_cid),
                "task_alias": str(getattr(attempt, "task_alias", "") or ""),
                "task_revision": int(control_claim.get("revision") or 0),
                "fencing_token": int(attempt.fencing_token),
                "fence_epoch": int(attempt.fence_epoch),
                "lease_id": str(attempt.lease_id),
            }
            durable_binding = (
                self._binding_lookup(attempt)
                if self._binding_lookup is not None
                else None
            )
            if (
                not control_claim
                or any(
                    observed.get(field) != value
                    for field, value in historical_expected.items()
                )
                or not str(control_claim.get("execution_spec_cid") or "")
                or not str(control_claim.get("validation_spec_cid") or "")
                or not isinstance(durable_binding, Mapping)
                or durable_binding.get("binding_id")
                != observed.get("binding_id")
                or durable_binding.get("projection_immutable_digest")
                != observed.get("projection_immutable_digest")
            ):
                raise DatabasePortalBridgeError(
                    "database Portal historical binding does not match the "
                    "attempt control claim"
                )
            expected = observed
        projection = self._verify_projection(paths, expected)
        identity = self._projection_task_identity(paths, expected, projection)
        strict_state, strict_state_digest = self._strict_state_record(
            paths.state
        )
        nested_state = self._verify_nested_state_identity(
            paths,
            expected,
            identity,
            payload=strict_state,
            state_digest=strict_state_digest,
        )
        _state_before_fence, state_digest_before_fence = (
            self._strict_state_record(paths.state)
        )
        if state_digest_before_fence != strict_state_digest:
            return {
                "reconciled": False,
                "blocked": True,
                "reason": "nested_state_changed_before_provider_fence",
                "attempt_id": str(attempt.attempt_id),
                "claim_id": str(attempt.claim_id),
                "task_cid": str(attempt.task_cid),
                "task_alias": str(expected.get("task_alias") or ""),
                "attempt_root": str(paths.root),
                "binding_id": str(expected.get("binding_id") or ""),
                "nested_state": nested_state,
                "provider_runner_fence": {
                    "applicable": bool(nested_state.get("active")),
                    "safe_to_restart": False,
                    "fenced": False,
                    "reason": "nested_state_changed_before_provider_fence",
                },
                "terminal_provider_evidence": False,
            }
        from .supervisor import fence_ordinary_provider_runner

        provider_runner_fence = self._validated_provider_runner_fence(
            fence_ordinary_provider_runner(
                strict_state or {},
                grace_seconds=1.0,
            )
        )
        provider_runner_reconciliation_authority = (
            # The sealed runner has a distinct descriptor/latch authority.
            # There is no public standalone validator for a dead sealed
            # receipt, so preserve that predecessor boundary and let the
            # existing Portal reconciliation path decide it.  This bridge
            # never signals a process from the sealed schema alone.
            "delegated_to_portal_sealed_authority"
            if provider_runner_fence.get("reason")
            == "sealed_provider_runner_receipt_not_applicable"
            else (
                "ordinary_provider_runner_fence"
                if provider_runner_fence.get("applicable") is True
                else "not_applicable"
            )
        )
        if (
            provider_runner_fence.get("safe_to_restart") is not True
            or (
                nested_state.get("active_phase") == "implementing"
                and provider_runner_fence.get("applicable") is not True
                and provider_runner_fence.get("reason")
                != "sealed_provider_runner_receipt_not_applicable"
            )
        ):
            fence_reason = (
                "nested_active_provider_runner_fence_missing"
                if nested_state.get("active_phase") == "implementing"
                and provider_runner_fence.get("applicable") is not True
                else "nested_provider_runner_fence_unproven"
            )
            return {
                "reconciled": False,
                "blocked": True,
                "reason": fence_reason,
                "attempt_id": str(attempt.attempt_id),
                "claim_id": str(attempt.claim_id),
                "task_cid": str(attempt.task_cid),
                "task_alias": str(expected.get("task_alias") or ""),
                "attempt_root": str(paths.root),
                "binding_id": str(expected.get("binding_id") or ""),
                "nested_state": nested_state,
                "provider_runner_fence": dict(provider_runner_fence),
                "provider_runner_reconciliation_authority": (
                    provider_runner_reconciliation_authority
                ),
                "terminal_provider_evidence": False,
            }
        _state_after_fence, state_digest_after_fence = (
            self._strict_state_record(paths.state)
        )
        if state_digest_after_fence != nested_state["state_digest"]:
            return {
                "reconciled": False,
                "blocked": True,
                "reason": "nested_state_changed_during_provider_fence",
                "attempt_id": str(attempt.attempt_id),
                "claim_id": str(attempt.claim_id),
                "task_cid": str(attempt.task_cid),
                "task_alias": str(expected.get("task_alias") or ""),
                "attempt_root": str(paths.root),
                "binding_id": str(expected.get("binding_id") or ""),
                "nested_state": nested_state,
                "provider_runner_fence": dict(provider_runner_fence),
                "provider_runner_reconciliation_authority": (
                    provider_runner_reconciliation_authority
                ),
                "terminal_provider_evidence": False,
            }
        daemon = self.portal_factory(
            paths,
            str(expected.get("task_alias") or attempt.task_cid),
        )
        reconcile = getattr(daemon, "reconcile_quiesced_active_attempt", None)
        if not callable(reconcile):
            raise DatabasePortalBridgeError(
                "portal_factory does not expose quiesced-attempt reconciliation"
            )
        interrupted_validation_evidence = (
            self._interrupted_validation_recovery_evidence(
                attempt,
                expected,
            )
        )
        reconcile_interrupted_validation = getattr(
            daemon,
            "reconcile_interrupted_database_validation_attempt",
            None,
        )
        try:
            if (
                interrupted_validation_evidence is not None
                and callable(reconcile_interrupted_validation)
            ):
                raw_reconciliation = reconcile_interrupted_validation(
                    interrupted_validation_evidence
                )
            else:
                raw_reconciliation = reconcile()
            if not isinstance(raw_reconciliation, Mapping):
                raise DatabasePortalBridgeError(
                    "Portal nested reconciliation returned a non-object"
                )
            reconciliation = dict(raw_reconciliation)
        finally:
            close = getattr(daemon, "close_event_runtime", None) or getattr(
                daemon, "close", None
            )
            if callable(close):
                close()
        if reconciliation.get("reconciled") is not True or reconciliation.get(
            "blocked"
        ) is True:
            return {
                "reconciled": False,
                "blocked": True,
                "reason": "nested_portal_attempt_reconciliation_blocked",
                "attempt_id": str(attempt.attempt_id),
                "claim_id": str(attempt.claim_id),
                "task_cid": str(attempt.task_cid),
                "task_alias": str(expected.get("task_alias") or ""),
                "attempt_root": str(paths.root),
                "binding_id": str(expected.get("binding_id") or ""),
                "nested_state": nested_state,
                "provider_runner_fence": dict(provider_runner_fence),
                "provider_runner_reconciliation_authority": (
                    provider_runner_reconciliation_authority
                ),
                "portal_reconciliation": reconciliation,
                "terminal_provider_evidence": False,
            }
        terminal_provider_evidence = (
            None if historical_binding else self.recover_provider_result(attempt)
        )
        return {
            "reconciled": True,
            "blocked": False,
            "reason": "nested_portal_attempt_reconciled",
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "task_cid": str(attempt.task_cid),
            "task_alias": str(expected.get("task_alias") or ""),
            "attempt_number": int(attempt.attempt_number),
            "owner_session_id": str(attempt.owner_session_id),
            "attempt_root": str(paths.root),
            "binding_id": str(expected.get("binding_id") or ""),
            "historical_binding": historical_binding,
            "nested_state": nested_state,
            "provider_runner_fence": dict(provider_runner_fence),
            "provider_runner_reconciliation_authority": (
                provider_runner_reconciliation_authority
            ),
            "portal_reconciliation": reconciliation,
            "terminal_provider_evidence": bool(terminal_provider_evidence),
            "terminal_provider_receipt_id": str(
                (terminal_provider_evidence or {}).get("receipt_id") or ""
            ),
        }

    def persist_reconciliation_receipt(
        self,
        attempt: Any,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist bounded task/attempt-linked terminal reconciliation evidence."""

        paths = self._paths(attempt)
        authoritative = {
            "schema": DATABASE_PORTAL_ATTEMPT_RECONCILIATION_SCHEMA,
            "interface": self.INTERFACE,
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "task_cid": str(attempt.task_cid),
            "task_alias": str(getattr(attempt, "task_alias", "") or ""),
            "attempt_number": int(attempt.attempt_number),
            "owner_session_id": str(attempt.owner_session_id),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "attempt_root": str(paths.root),
        }
        supplied = dict(payload)
        supplied.pop("receipt_id", None)
        unknown = set(supplied) - _ATTEMPT_RECONCILIATION_FIELDS
        if unknown:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation payload has unknown fields: "
                + ", ".join(sorted(str(field) for field in unknown))
            )
        conflicts = [
            field
            for field, expected in authoritative.items()
            if field in supplied and supplied[field] != expected
        ]
        if conflicts:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation payload overrides authority: "
                + ", ".join(sorted(conflicts))
            )
        receipt = {**supplied, **authoritative}
        stage = str(receipt.get("stage") or "")
        if stage not in {"prepared", "commit_barrier", "terminal", "blocked"}:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation stage is not closed"
            )
        if not isinstance(receipt.get("trigger"), str) or not str(
            receipt.get("trigger") or ""
        ).strip():
            raise DatabasePortalBridgeError(
                "database Portal reconciliation trigger is missing"
            )
        if not isinstance(receipt.get("reconciled_at"), str) or not str(
            receipt.get("reconciled_at") or ""
        ).strip():
            raise DatabasePortalBridgeError(
                "database Portal reconciliation timestamp is missing"
            )
        try:
            json.dumps(receipt, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation payload is not strict JSON"
            ) from exc
        if any(receipt[field] != expected for field, expected in authoritative.items()):
            raise DatabasePortalBridgeError(
                "database Portal reconciliation authority changed"
            )
        receipt["receipt_id"] = _sha256_bytes(_canonical_json(receipt))
        try:
            confined_parent = paths.root.parent.resolve()
            authority_root = self.attempt_root.resolve()
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation root is unavailable"
            ) from exc
        if (
            confined_parent != authority_root
            or self.attempt_root.is_symlink()
            or paths.root.is_symlink()
            or paths.reconciliation.is_symlink()
            or (
                paths.reconciliation.exists()
                and not paths.reconciliation.is_dir()
            )
        ):
            raise DatabasePortalBridgeError(
                "database Portal reconciliation store escapes its exact root"
            )
        receipt_path = paths.reconciliation / (
            receipt["receipt_id"].removeprefix("sha256:") + ".json"
        )
        encoded = (
            json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8")
            + b"\n"
        )
        _atomic_write_if_absent(receipt_path, encoded)
        receipt["receipt_path"] = str(receipt_path)
        return receipt

    def load_reconciliation_receipt(
        self,
        attempt: Any,
        receipt_id: str,
        *,
        required_stage: str = "",
    ) -> dict[str, Any]:
        """Load one exact immutable reconciliation object fail closed.

        This is the replay authority for the narrow crash window after the
        canonical task CAS but before claim release/local attempt
        terminalization.  The receipt id selects one file below the exact
        attempt-id-derived root; no directory enumeration or task re-render is
        involved.
        """

        normalized_id = str(receipt_id or "").strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", normalized_id):
            raise DatabasePortalBridgeError(
                "database Portal reconciliation receipt id is malformed"
            )
        paths = self._paths(attempt)
        try:
            authority_root = self.attempt_root.resolve()
            confined_parent = paths.root.parent.resolve()
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation root is unavailable"
            ) from exc
        if (
            confined_parent != authority_root
            or self.attempt_root.is_symlink()
            or paths.root.is_symlink()
            or paths.reconciliation.is_symlink()
            or not paths.reconciliation.is_dir()
        ):
            raise DatabasePortalBridgeError(
                "database Portal reconciliation store escapes its exact root"
            )
        receipt_path = paths.reconciliation / (
            normalized_id.removeprefix("sha256:") + ".json"
        )
        _recover_immutable_link_publication(receipt_path)
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise DatabasePortalBridgeError(
                "database Portal reconciliation receipt is not a regular file"
            )

        def closed_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise DatabasePortalBridgeError(
                        "database Portal reconciliation receipt contains "
                        "duplicate keys"
                    )
                result[key] = value
            return result

        try:
            raw = receipt_path.read_bytes()
            if len(raw) > 262_144:
                raise DatabasePortalBridgeError(
                    "database Portal reconciliation receipt is oversized"
                )
            receipt = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=closed_object,
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    DatabasePortalBridgeError(
                        "database Portal reconciliation receipt contains a "
                        "nonfinite value"
                    )
                ),
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation receipt is unreadable"
            ) from exc
        if not isinstance(receipt, Mapping):
            raise DatabasePortalBridgeError(
                "database Portal reconciliation receipt is malformed"
            )
        receipt = dict(receipt)
        unknown = set(receipt) - _ATTEMPT_RECONCILIATION_FIELDS
        if unknown:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation receipt has unknown fields"
            )
        authoritative = {
            "schema": DATABASE_PORTAL_ATTEMPT_RECONCILIATION_SCHEMA,
            "interface": self.INTERFACE,
            "attempt_id": str(attempt.attempt_id),
            "claim_id": str(attempt.claim_id),
            "task_cid": str(attempt.task_cid),
            "task_alias": str(getattr(attempt, "task_alias", "") or ""),
            "attempt_number": int(attempt.attempt_number),
            "owner_session_id": str(attempt.owner_session_id),
            "fencing_token": int(attempt.fencing_token),
            "fence_epoch": int(attempt.fence_epoch),
            "attempt_root": str(paths.root),
            "receipt_id": normalized_id,
        }
        mismatched = [
            field
            for field, expected in authoritative.items()
            if receipt.get(field) != expected
        ]
        if mismatched:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation receipt changed authority: "
                + ", ".join(sorted(mismatched))
            )
        stage = str(receipt.get("stage") or "")
        if stage not in {
            "prepared",
            "commit_barrier",
            "terminal",
            "blocked",
        } or (
            required_stage and stage != str(required_stage)
        ):
            raise DatabasePortalBridgeError(
                "database Portal reconciliation receipt has the wrong stage"
            )
        unsigned = dict(receipt)
        unsigned.pop("receipt_id", None)
        if _sha256_bytes(_canonical_json(unsigned)) != normalized_id:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation receipt identity does not verify"
            )
        return receipt

    def _interrupted_validation_recovery_evidence(
        self,
        attempt: Any,
        binding: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Return one exact durable post-provider/pre-validation crash proof.

        The nested Portal state is mutable and may already have been cleared by
        an earlier crash-reconciliation pass.  Recovery may therefore use only
        an immutable database-attempt reconciliation receipt that captured the
        exact ``validating`` state while fencing the exact ordinary provider
        birth.  Repeated receipts for the same exact recovery identity are
        coalesced; distinct qualifying identities remain ambiguous.
        """

        paths = self._paths(attempt)
        if not paths.reconciliation.exists():
            return None
        if (
            paths.reconciliation.is_symlink()
            or not paths.reconciliation.is_dir()
        ):
            raise DatabasePortalBridgeError(
                "database Portal reconciliation evidence store is not exact"
            )
        try:
            receipt_paths = sorted(paths.reconciliation.iterdir())
        except OSError as exc:
            raise DatabasePortalBridgeError(
                "database Portal reconciliation evidence store is unreadable"
            ) from exc
        task_alias = str(binding.get("task_alias") or "")
        binding_id = str(binding.get("binding_id") or "")
        matches: list[dict[str, Any]] = []
        for receipt_path in receipt_paths:
            match = re.fullmatch(r"([0-9a-f]{64})\.json", receipt_path.name)
            if match is None:
                raise DatabasePortalBridgeError(
                    "database Portal reconciliation evidence name is malformed"
                )
            receipt = self.load_reconciliation_receipt(
                attempt,
                "sha256:" + match.group(1),
            )
            if receipt.get("stage") != "blocked":
                continue
            nested = receipt.get("nested_state")
            fence = receipt.get("provider_runner_fence")
            portal = receipt.get("portal_reconciliation")
            if not (
                receipt.get("blocked") is True
                and receipt.get("reconciled") is False
                and receipt.get("reason")
                == "nested_portal_attempt_reconciliation_blocked"
                and receipt.get("binding_id") == binding_id
                and receipt.get("task_alias") == task_alias
                and receipt.get("terminal_provider_evidence") is False
                and receipt.get("provider_runner_reconciliation_authority")
                == "ordinary_provider_runner_fence"
                and isinstance(nested, Mapping)
                and nested.get("active") is True
                and nested.get("active_phase") == "validating"
                and nested.get("active_task_id") == task_alias
                and isinstance(nested.get("active_attempt"), int)
                and not isinstance(nested.get("active_attempt"), bool)
                and int(nested.get("active_attempt") or 0) > 0
                and isinstance(nested.get("active_worktree_path"), str)
                and bool(str(nested.get("active_worktree_path") or "").strip())
                and isinstance(nested.get("active_branch"), str)
                and bool(str(nested.get("active_branch") or "").strip())
                and nested.get("state_path") == str(paths.state)
                and isinstance(fence, Mapping)
                and fence.get("applicable") is True
                and fence.get("fenced") is True
                and fence.get("safe_to_restart") is True
                and fence.get("reason")
                == "ordinary_provider_runner_exact_birth_fenced"
                and isinstance(portal, Mapping)
                and portal.get("blocked") is True
                and portal.get("reconciled") is False
                and portal.get("reason")
                == "task_claim_reconciliation_blocked"
            ):
                continue

            protected = portal.get("protected_path_reconciliation")
            lifecycle = portal.get("worktree_lifecycle_reconciliation")
            claim = portal.get("task_claim_reconciliation")
            attempt_recovery = portal.get("attempt_recovery")
            active_attempt = int(nested["active_attempt"])
            workspace = str(nested["active_worktree_path"])
            if not (
                isinstance(protected, Mapping)
                and protected.get("blocked") is False
                and protected.get("reason") == "crash_reconciliation_unchanged"
                and protected.get("task_id") == task_alias
                and protected.get("workspace_path") == workspace
                and isinstance(lifecycle, Mapping)
                and lifecycle.get("blocked") is False
                and lifecycle.get("reconciled") is True
                and lifecycle.get("state") == "terminal"
                and lifecycle.get("task_id") == task_alias
                and lifecycle.get("workspace_path") == workspace
                and lifecycle.get("attempt") == active_attempt
                and isinstance(claim, Mapping)
                and claim.get("blocked") is True
                and claim.get("reconciled") is False
                and claim.get("reason") == "canonical_task_not_terminal"
                and claim.get("task_id") == task_alias
                and isinstance(attempt_recovery, Mapping)
                and attempt_recovery.get("consumed") is False
                and attempt_recovery.get("attempt") == active_attempt
                and attempt_recovery.get("task_id") == task_alias
            ):
                continue
            matches.append(receipt)

        if not matches:
            return None
        recovery_groups: dict[str, list[dict[str, Any]]] = {}
        for receipt in matches:
            nested = dict(receipt["nested_state"])
            portal = dict(receipt["portal_reconciliation"])
            lifecycle = dict(portal["worktree_lifecycle_reconciliation"])
            claim = dict(portal["task_claim_reconciliation"])
            recovery_identity = _sha256_bytes(
                _canonical_json(
                    {
                        "binding_id": binding_id,
                        "nested_state_digest": nested.get("state_digest"),
                        "active_attempt": nested.get("active_attempt"),
                        "active_task_id": nested.get("active_task_id"),
                        "active_worktree_path": nested.get(
                            "active_worktree_path"
                        ),
                        "active_branch": nested.get("active_branch"),
                        "lifecycle_record_id": lifecycle.get("record_id"),
                        "lifecycle_fence": lifecycle.get("fence"),
                        "canonical_task_cid": claim.get(
                            "canonical_task_cid"
                        ),
                    }
                )
            )
            recovery_groups.setdefault(recovery_identity, []).append(receipt)
        if len(recovery_groups) != 1:
            raise DatabasePortalBridgeError(
                "database Portal interrupted validation evidence is ambiguous"
            )
        recovery_identity, equivalent_receipts = next(
            iter(recovery_groups.items())
        )
        selected = min(
            equivalent_receipts,
            key=lambda item: str(item.get("receipt_id") or ""),
        )
        evidence = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "database-portal-interrupted-validation-recovery@1"
            ),
            "binding": dict(binding),
            "recovery_identity": recovery_identity,
            "equivalent_receipt_ids": sorted(
                str(item.get("receipt_id") or "")
                for item in equivalent_receipts
            ),
            "reconciliation_receipt": selected,
        }
        evidence["evidence_id"] = _sha256_bytes(_canonical_json(evidence))
        return evidence

    @staticmethod
    def _require_accepted_provider(attempt: Any, provider_result: Mapping[str, Any]) -> str:
        if (
            provider_result.get("schema") != DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA
            or provider_result.get("interface") != DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE
            or provider_result.get("accepted") is not True
            or provider_result.get("status") != "succeeded"
            or provider_result.get("provider") != "PortalImplementationDaemon"
            or str(provider_result.get("task_cid") or "") != str(attempt.task_cid)
            or str(provider_result.get("attempt_id") or "") != str(attempt.attempt_id)
        ):
            raise DatabasePortalBridgeError(
                "database effect rejected unaccepted Portal provider evidence"
            )
        digest = str(provider_result.get("evidence_digest") or "")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise DatabasePortalBridgeError(
                "database effect rejected malformed Portal evidence identity"
            )
        return digest

    def apply_effect(self, attempt: Any, provider_result: Mapping[str, Any]) -> Mapping[str, Any]:
        """Bind the already-applied Portal effect to the database phase."""

        digest = self._require_accepted_provider(attempt, provider_result)
        return {
            "status": "applied",
            "effect": "portal-supervised-accepted-effect",
            "effect_key": f"portal:{attempt.task_cid}:{attempt.attempt_id}",
            "task_cid": str(attempt.task_cid),
            "attempt_id": str(attempt.attempt_id),
            "portal_receipt_id": str(provider_result.get("receipt_id") or ""),
            "evidence_digest": digest,
        }

    def validate_effect(self, attempt: Any, effect_result: Mapping[str, Any]) -> Mapping[str, Any]:
        """Admit only an exact effect derived from accepted Portal evidence."""

        digest = str(effect_result.get("evidence_digest") or "")
        if (
            effect_result.get("status") != "applied"
            or effect_result.get("effect") != "portal-supervised-accepted-effect"
            or str(effect_result.get("task_cid") or "") != str(attempt.task_cid)
            or str(effect_result.get("attempt_id") or "") != str(attempt.attempt_id)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
        ):
            raise DatabasePortalBridgeError(
                "database validation rejected unbound Portal effect evidence"
            )
        return {
            "outcome": "passed",
            "evidence_digest": digest,
            "argv": ["portal-supervisor-gates"],
            "validator": self.INTERFACE,
            "task_cid": str(attempt.task_cid),
            "attempt_id": str(attempt.attempt_id),
            "portal_receipt_id": str(effect_result.get("portal_receipt_id") or ""),
        }


__all__ = (
    "DATABASE_PORTAL_ATTEMPT_BINDING_SCHEMA",
    "DATABASE_PORTAL_ATTEMPT_RECONCILIATION_SCHEMA",
    "DATABASE_PORTAL_EXECUTION_BRIDGE_INTERFACE",
    "DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA",
    "DatabasePortalAttemptPaths",
    "DatabasePortalBridgeDeferred",
    "DatabasePortalBridgeError",
    "DatabasePortalPreEntryPublicationDeferred",
    "DatabasePortalExecutionBridge",
    "PortalDaemonFactory",
)
