"""Fail-closed repair for a sealed Markdown projection of database task state.

This module exists for configured boards whose mutable task authority is a
database served through Quack.  Their tracked Markdown board is an immutable
bootstrap projection, not a place where runtime guardrail tasks may acquire a
second authority.

Repair is deliberately narrow.  It can restore the exact board bytes named by
the bootstrap receipt only when the current file is that byte sequence plus a
suffix of supervisor-generated guardrails.  The canonical task identities are
read from Quack before admission or mutation.  Any unknown edit, canonical
task in the suffix, unavailable authority, untrusted Git provenance, or active
checkout mutation lease makes the operation abstain.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final

from ..merge.checkout_lock import (
    GENERATED_PROTECTED_BOARD_COMMIT_MARKER,
    acquire_checkout_mutation_lease,
    checkout_lock_metadata,
    checkout_mutation_lock_path,
    generated_protected_board_commit_subject,
    release_checkout_mutation_lease,
)

REPAIR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "authoritative-board-projection-repair@1"
)
POLICY_KEY: Final = "authoritative_board_projection_repair"
ALLOWED_MODE: Final = "sealed_bootstrap_projection"
ALLOWED_DRIFT: Final = "supervisor_generated_guardrail_suffix_only"
COMMIT_SUBJECT: Final = generated_protected_board_commit_subject(
    "Agent: restore sealed authoritative board projection"
)
HEX40_RE: Final = re.compile(r"^[0-9a-f]{40}$")
QUACK_ENDPOINT_RE: Final = re.compile(
    r"^quack:(?://)?(?:127(?:\.\d{1,3}){3}|localhost):\d{1,5}$",
    re.IGNORECASE,
)
GENERATED_STATUSES: Final = frozenset({"todo", "blocked", "completed"})
GENERATED_ACCEPTANCE_MARKERS: Final = (
    "Dependency guardrail filed this",
    "Retry-budget guardrail filed this",
    "Reconciliation guardrail filed this",
)


class BoardProjectionRepairError(RuntimeError):
    """The projection cannot be repaired without weakening an invariant."""


def _identity(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise BoardProjectionRepairError(f"{label} is not a regular file")
        value = json.loads(path.read_text(encoding="utf-8"))
    except BoardProjectionRepairError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BoardProjectionRepairError(f"{label} is unavailable or malformed") from exc
    if not isinstance(value, dict):
        raise BoardProjectionRepairError(f"{label} root must be an object")
    return value


def _safe_path(
    repo_root: Path,
    value: Any,
    *,
    label: str,
    require_file: bool = False,
) -> Path:
    text = str(value or "").strip()
    candidate = PurePosixPath(text)
    if not text or candidate.is_absolute() or ".." in candidate.parts:
        raise BoardProjectionRepairError(
            f"{label} must be a safe repository-relative path"
        )
    path = (repo_root / Path(*candidate.parts)).resolve(strict=False)
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise BoardProjectionRepairError(f"{label} escapes repository") from exc
    if require_file:
        try:
            metadata = os.lstat(path)
        except OSError as exc:
            raise BoardProjectionRepairError(f"{label} is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise BoardProjectionRepairError(f"{label} is not a regular file")
    return path


def _git(
    repo_root: Path,
    arguments: Sequence[str],
    *,
    binary: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[Any]:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        capture_output=True,
        text=not binary,
        check=False,
    )
    if check and result.returncode != 0:
        raise BoardProjectionRepairError(
            f"git {arguments[0] if arguments else 'operation'} failed"
        )
    return result


def _atomic_bytes(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.repair.{os.getpid()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        mode,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_bytes(path, body, mode=0o600)


def _board_task_ids(payload: bytes, *, task_prefix: str) -> tuple[str, ...]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BoardProjectionRepairError("sealed board is not UTF-8") from exc
    prefix = str(task_prefix or "").removeprefix("## ").strip()
    if not prefix:
        raise BoardProjectionRepairError("task prefix is absent")
    pattern = re.compile(
        rf"^## ({re.escape(prefix)}[0-9]+)(?:\s+.+)?$",
        re.MULTILINE,
    )
    task_ids = tuple(match.group(1) for match in pattern.finditer(text))
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise BoardProjectionRepairError(
            "sealed board task identities are empty or duplicated"
        )
    return task_ids


def _metadata_fields(block: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("- ") or ":" not in stripped:
            raise BoardProjectionRepairError(
                "generated reconciliation suffix contains non-metadata content"
            )
        key, value = stripped[2:].split(":", 1)
        normalized = " ".join(key.lower().replace("_", " ").split())
        if normalized in fields:
            raise BoardProjectionRepairError(
                "generated reconciliation suffix contains duplicate metadata"
            )
        fields[normalized] = value.strip()
    return fields


def _generated_suffix_task_ids(
    suffix: bytes,
    *,
    task_prefix: str,
    canonical_task_ids: Sequence[str],
    taskboard_relative: str,
) -> tuple[str, ...]:
    try:
        text = suffix.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BoardProjectionRepairError(
            "board drift suffix is not UTF-8"
        ) from exc
    prefix = str(task_prefix or "").removeprefix("## ").strip()
    heading = re.compile(
        rf"^## ({re.escape(prefix)}[0-9]+) (.+)$",
        re.MULTILINE,
    )
    matches = list(heading.finditer(text))
    if not matches or text[: matches[0].start()].strip():
        raise BoardProjectionRepairError(
            "board drift is not a generated guardrail-only suffix"
        )
    canonical = set(canonical_task_ids)
    generated: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        task_id = match.group(1)
        if task_id in canonical:
            raise BoardProjectionRepairError(
                "refusing to remove a canonical task from the board projection"
            )
        fields = _metadata_fields(text[match.end() : end])
        acceptance = fields.get("acceptance", "")
        if (
            fields.get("status", "").lower() not in GENERATED_STATUSES
            or fields.get("completion", "").lower() != "manual"
            or fields.get("track", "").lower() != "ops"
            or not any(marker in acceptance for marker in GENERATED_ACCEPTANCE_MARKERS)
        ):
            raise BoardProjectionRepairError(
                f"{task_id} is not a closed generated guardrail projection"
            )
        reconciliation = "Reconciliation guardrail filed this" in acceptance
        if reconciliation and (
            fields.get("is schedulable", "").lower() != "false"
            or fields.get("review only", "").lower() != "true"
            or fields.get("blocked reason", "").lower()
            != "operator_reconciliation_required"
            or not fields.get("dedupe key", "").startswith(
                "reconciliation_guardrail:"
            )
            or taskboard_relative not in fields.get("outputs", "")
        ):
            raise BoardProjectionRepairError(
                f"{task_id} is not a closed reconciliation guardrail projection"
            )
        generated.append(task_id)
    if len(generated) != len(set(generated)):
        raise BoardProjectionRepairError(
            "generated reconciliation suffix contains duplicate task identities"
        )
    return tuple(generated)


def classify_projection_drift(
    *,
    sealed_board: bytes,
    current_board: bytes,
    canonical_task_ids: Sequence[str],
    task_prefix: str,
    taskboard_relative: str,
) -> dict[str, Any]:
    """Classify one board without mutating it.

    The canonical task set must exactly equal the sealed task set.  This keeps
    projection repair from masking a database migration or a genuine blocker
    introduced under a canonical task identity.
    """

    sealed_task_ids = _board_task_ids(sealed_board, task_prefix=task_prefix)
    canonical = tuple(str(item) for item in canonical_task_ids)
    if canonical != sealed_task_ids:
        raise BoardProjectionRepairError(
            "canonical task identities differ from the sealed board"
        )
    if current_board == sealed_board:
        return {
            "drift": False,
            "sealed_task_ids": list(sealed_task_ids),
            "generated_task_ids": [],
        }
    if not current_board.startswith(sealed_board):
        raise BoardProjectionRepairError(
            "board drift modifies the sealed bootstrap bytes"
        )
    generated = _generated_suffix_task_ids(
        current_board[len(sealed_board) :],
        task_prefix=task_prefix,
        canonical_task_ids=canonical,
        taskboard_relative=taskboard_relative,
    )
    return {
        "drift": True,
        "sealed_task_ids": list(sealed_task_ids),
        "generated_task_ids": list(generated),
    }


def _private_token(path: Path) -> str:
    try:
        metadata = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise BoardProjectionRepairError(
                "Quack token vault file is not a private regular file"
            )
        token = path.read_text(encoding="utf-8").strip()
    except BoardProjectionRepairError:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        raise BoardProjectionRepairError("Quack token vault is unavailable") from exc
    if re.fullmatch(r"[A-Za-z0-9_-]{8,}", token) is None:
        raise BoardProjectionRepairError("Quack token vault material is malformed")
    return token


def _canonical_snapshot_from_quack(
    *,
    repo_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    from ..task_sources.duckdb_state import open_quack_transport_connection

    database = config.get("database_program")
    runtime_paths = config.get("runtime_paths")
    if not isinstance(database, Mapping) or not isinstance(runtime_paths, Mapping):
        raise BoardProjectionRepairError("database or runtime configuration is absent")
    if (
        database.get("authority_mode") != "quack"
        or database.get("task_source_kind") != "duckdb"
    ):
        raise BoardProjectionRepairError(
            "board projection repair requires DuckDB authority over Quack"
        )
    endpoint = str(database.get("quack_endpoint") or "")
    if QUACK_ENDPOINT_RE.fullmatch(endpoint) is None:
        raise BoardProjectionRepairError("canonical Quack endpoint is not bounded loopback")
    secret_handle = str(database.get("endpoint_secret_handle") or "").strip()
    if not secret_handle:
        raise BoardProjectionRepairError("canonical Quack secret handle is absent")
    owner_dir = _safe_path(
        repo_root,
        runtime_paths.get("quack_owner"),
        label="runtime_paths.quack_owner",
    )
    safe_handle = secret_handle.replace(":", "_").replace("/", "_")
    token = _private_token(owner_dir / f"{safe_handle}.quack-token")
    connection = None
    try:
        connection = open_quack_transport_connection(endpoint, token=token)
        task_rows = connection.execute(
            "SELECT task_cid, task_alias, ordinal, status "
            "FROM tasks ORDER BY ordinal, task_alias"
        ).fetchall()
        block_rows = connection.execute(
            "SELECT task_cid FROM task_blocks WHERE state = 'active'"
        ).fetchall()
    except Exception as exc:
        raise BoardProjectionRepairError(
            "canonical Quack task authority is unavailable"
        ) from exc
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
    active_block_cids = {str(row[0]) for row in block_rows}
    blocked = [
        str(row[1])
        for row in task_rows
        if str(row[3]).lower() == "blocked" or str(row[0]) in active_block_cids
    ]
    return {
        "task_ids": [str(row[1]) for row in task_rows],
        "status_by_task_id": {str(row[1]): str(row[3]) for row in task_rows},
        "blocked_task_ids": blocked,
    }


def _trusted_board_history(
    *,
    repo_root: Path,
    source_head: str,
    taskboard_relative: str,
    sealed_board: bytes,
    current_board: bytes,
) -> bool:
    ancestor = _git(
        repo_root,
        ["merge-base", "--is-ancestor", source_head, "HEAD"],
        check=False,
    )
    if ancestor.returncode != 0:
        return False
    if not current_board.startswith(sealed_board) or not current_board[
        len(sealed_board) :
    ].startswith(b"\n"):
        return False
    first_suffix_line = sealed_board.count(b"\n") + 1
    last_line = current_board.count(b"\n") + (
        0 if current_board.endswith(b"\n") else 1
    )
    if last_line < first_suffix_line:
        return False
    blame = _git(
        repo_root,
        [
            "blame",
            "--line-porcelain",
            "-L",
            f"{first_suffix_line},{last_line}",
            "HEAD",
            "--",
            taskboard_relative,
        ],
    ).stdout
    subjects = [
        line.removeprefix("summary ").strip()
        for line in str(blame).splitlines()
        if line.startswith("summary ")
    ]
    return bool(subjects) and all(
        GENERATED_PROTECTED_BOARD_COMMIT_MARKER in subject for subject in subjects
    )


def _commit_repair(
    *,
    repo_root: Path,
    taskboard: Path,
    taskboard_relative: str,
    sealed_board: bytes,
    expected_current: bytes,
    expected_head: str,
    branch: str,
) -> str:
    from ..merge.checkout_lock import checkout_mutation_lease_state

    metadata = checkout_lock_metadata(
        kind="authoritative-board-projection-repair",
        repo_root=repo_root,
        branch=branch,
        owner_script=Path(__file__).name,
        extra={"operation": "restore_sealed_board_projection"},
    )
    lease, reason, _owner, _wait = acquire_checkout_mutation_lease(
        checkout_mutation_lock_path(repo_root),
        metadata,
        owner_active=lambda _metadata: True,
        timeout_seconds=0.0,
    )
    if lease is None:
        raise BoardProjectionRepairError(
            f"checkout mutation lease is unavailable: {reason}"
        )
    try:
        if checkout_mutation_lease_state(lease) != "current":
            raise BoardProjectionRepairError("checkout mutation lease was replaced")
        observed_head = str(
            _git(repo_root, ["rev-parse", "HEAD"]).stdout
        ).strip()
        if observed_head != expected_head or taskboard.read_bytes() != expected_current:
            raise BoardProjectionRepairError(
                "board or repository changed before fenced repair"
            )
        working_diff = _git(
            repo_root,
            ["diff", "--quiet", "--", taskboard_relative],
            check=False,
        )
        staged_diff = _git(
            repo_root,
            ["diff", "--cached", "--quiet", "--", taskboard_relative],
            check=False,
        )
        if working_diff.returncode != 0 or staged_diff.returncode != 0:
            raise BoardProjectionRepairError(
                "uncommitted board edits cannot be repaired automatically"
            )
        tracked = _git(
            repo_root,
            ["show", f"HEAD:{taskboard_relative}"],
            binary=True,
        ).stdout
        if tracked != expected_current:
            raise BoardProjectionRepairError(
                "uncommitted board edits cannot be repaired automatically"
            )
        _atomic_bytes(taskboard, sealed_board)
        commit = _git(
            repo_root,
            [
                "-c",
                "user.name=Agent Supervisor Control Plane",
                "-c",
                "user.email=agent-supervisor@example.invalid",
                "commit",
                "--only",
                "-m",
                COMMIT_SUBJECT,
                "--",
                taskboard_relative,
            ],
            check=False,
        )
        if commit.returncode != 0:
            _atomic_bytes(taskboard, expected_current)
            _git(
                repo_root,
                ["restore", "--staged", "--", taskboard_relative],
                check=False,
            )
            raise BoardProjectionRepairError("sealed board repair commit failed")
        repair_commit = str(_git(repo_root, ["rev-parse", "HEAD"]).stdout).strip()
        if (
            not HEX40_RE.fullmatch(repair_commit)
            or taskboard.read_bytes() != sealed_board
            or _git(
                repo_root,
                ["show", f"HEAD:{taskboard_relative}"],
                binary=True,
            ).stdout
            != sealed_board
        ):
            raise BoardProjectionRepairError(
                "sealed board repair postcondition failed"
            )
        return repair_commit
    finally:
        released = release_checkout_mutation_lease(lease)
        if not released:
            raise BoardProjectionRepairError(
                "checkout mutation lease release failed"
            )


def repair_authoritative_board_projection(
    config_path: Path,
    *,
    repo_root: Path,
    canonical_snapshot_provider: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Repair one opted-in configured board before a real scheduler launch."""

    root = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config_file = config_file.resolve(strict=False)
    try:
        config_file.relative_to(root)
    except ValueError as exc:
        raise BoardProjectionRepairError("scheduler config escapes repository") from exc
    config = _read_object(config_file, label="scheduler config")
    policy = config.get(POLICY_KEY)
    if policy is None:
        return {
            "schema": REPAIR_SCHEMA,
            "enabled": False,
            "repaired": False,
            "reason_code": "policy_absent",
        }
    if not isinstance(policy, Mapping):
        raise BoardProjectionRepairError("board projection repair policy is malformed")
    if policy.get("automatic_repair_before_launch") is not True:
        return {
            "schema": REPAIR_SCHEMA,
            "enabled": False,
            "repaired": False,
            "reason_code": "automatic_repair_disabled",
        }
    if (
        policy.get("mode") != ALLOWED_MODE
        or policy.get("allowed_drift") != ALLOWED_DRIFT
        or policy.get("canonical_block_mutation_permitted") is not False
        or policy.get("markdown_task_mutation_permitted") is not False
    ):
        raise BoardProjectionRepairError(
            "board projection repair policy weakens the authority boundary"
        )
    control_plane = config.get("operational_control_plane")
    if (
        not isinstance(control_plane, Mapping)
        or control_plane.get("markdown_is_authority") is not False
    ):
        raise BoardProjectionRepairError(
            "board projection repair policy weakens the authority boundary"
        )
    taskboard_relative = str(config.get("taskboard_path") or "")
    taskboard = _safe_path(
        root,
        taskboard_relative,
        label="taskboard_path",
        require_file=True,
    )
    receipt = _read_object(
        _safe_path(
            root,
            policy.get("bootstrap_receipt_path"),
            label=f"{POLICY_KEY}.bootstrap_receipt_path",
            require_file=True,
        ),
        label="bootstrap receipt",
    )
    source_head = str(receipt.get("source_head") or "")
    source_identities = receipt.get("source_identities")
    source_identities = (
        source_identities if isinstance(source_identities, Mapping) else {}
    )
    expected_identity = str(source_identities.get("taskboard") or "")
    if not HEX40_RE.fullmatch(source_head) or not expected_identity.startswith(
        "sha256:"
    ):
        raise BoardProjectionRepairError(
            "bootstrap receipt lacks a sealed board source identity"
        )
    sealed_board = _git(
        root,
        ["show", f"{source_head}:{taskboard_relative}"],
        binary=True,
    ).stdout
    if not isinstance(sealed_board, bytes) or _identity(sealed_board) != expected_identity:
        raise BoardProjectionRepairError("sealed board bytes fail receipt identity")
    current_board = taskboard.read_bytes()
    provider = canonical_snapshot_provider or (
        lambda: _canonical_snapshot_from_quack(repo_root=root, config=config)
    )
    snapshot = dict(provider())
    canonical_task_ids = snapshot.get("task_ids")
    if (
        not isinstance(canonical_task_ids, Sequence)
        or isinstance(canonical_task_ids, (str, bytes, bytearray))
    ):
        raise BoardProjectionRepairError("canonical task snapshot is malformed")
    classification = classify_projection_drift(
        sealed_board=sealed_board,
        current_board=current_board,
        canonical_task_ids=[str(item) for item in canonical_task_ids],
        task_prefix=str(config.get("task_prefix") or ""),
        taskboard_relative=taskboard_relative,
    )
    blocked_task_ids = [
        str(item) for item in snapshot.get("blocked_task_ids", [])
    ]
    if not classification["drift"]:
        report: dict[str, Any] = {
            "schema": REPAIR_SCHEMA,
            "enabled": True,
            "repaired": False,
            "reason_code": "projection_current",
            "source_head": source_head,
            "taskboard_identity": expected_identity,
            "canonical_task_count": len(canonical_task_ids),
            "canonical_blocked_task_ids": blocked_task_ids,
            "canonical_blocks_mutated": False,
        }
        report["observed_at_epoch"] = time.time()
        report["repair_receipt_id"] = _identity(
            json.dumps(report, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        _atomic_json(
            _safe_path(
                root,
                policy.get("repair_receipt_path"),
                label=f"{POLICY_KEY}.repair_receipt_path",
            ),
            report,
        )
        return report
    if not _trusted_board_history(
        repo_root=root,
        source_head=source_head,
        taskboard_relative=taskboard_relative,
        sealed_board=sealed_board,
        current_board=current_board,
    ):
        raise BoardProjectionRepairError(
            "board drift was not introduced exclusively by trusted supervisor commits"
        )
    branch = str(_git(root, ["branch", "--show-current"]).stdout).strip()
    required_branch = str(config.get("merge_target_branch") or "").strip()
    if not branch or (required_branch and branch != required_branch):
        raise BoardProjectionRepairError("board repair checkout branch is not admitted")
    expected_head = str(_git(root, ["rev-parse", "HEAD"]).stdout).strip()
    repair_commit = _commit_repair(
        repo_root=root,
        taskboard=taskboard,
        taskboard_relative=taskboard_relative,
        sealed_board=sealed_board,
        expected_current=current_board,
        expected_head=expected_head,
        branch=branch,
    )
    report: dict[str, Any] = {
        "schema": REPAIR_SCHEMA,
        "enabled": True,
        "repaired": True,
        "reason_code": "generated_projection_drift_repaired",
        "source_head": source_head,
        "starting_head": expected_head,
        "repair_commit": repair_commit,
        "taskboard_identity_before": _identity(current_board),
        "taskboard_identity_after": expected_identity,
        "removed_generated_task_ids": classification["generated_task_ids"],
        "canonical_task_count": len(canonical_task_ids),
        "canonical_blocked_task_ids": blocked_task_ids,
        "canonical_blocks_mutated": False,
    }
    receipt_path = _safe_path(
        root,
        policy.get("repair_receipt_path"),
        label=f"{POLICY_KEY}.repair_receipt_path",
    )
    report["observed_at_epoch"] = time.time()
    report["repair_receipt_id"] = _identity(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    _atomic_json(receipt_path, report)
    return report


__all__ = (
    "ALLOWED_DRIFT",
    "ALLOWED_MODE",
    "BoardProjectionRepairError",
    "POLICY_KEY",
    "REPAIR_SCHEMA",
    "classify_projection_drift",
    "repair_authoritative_board_projection",
)
