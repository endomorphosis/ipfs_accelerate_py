#!/usr/bin/env python3
# ruff: noqa: F821
"""Fail-closed supervisor launcher for the Adversarial Assurance Engine.

This is the reviewed AAE binding of the existing multi-lane implementation
supervisor.  It adds no scheduler, task authority, provider, or execution
profile; the implementation remains in the shared supervisor runtime.
"""

from __future__ import annotations

import importlib
import re
import stat
from pathlib import Path

_AAE_ADMISSION_ENV = "IPFS_ACCELERATE_AAE_LAUNCH_ADMISSION_PATH"
_AAE_ADMISSION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "adversarial-assurance-launch-admission@1"
)
_AAE_LEDGER_ENTRY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "adversarial-assurance-launch-ledger-entry@1"
)
_AAE_PREREQUISITE_REL = "config/adversarial_assurance_prerequisites.json"
_AAE_LEDGER_MAX_BYTES = 16 * 1024 * 1024
_AAE_LEDGER_NAMESPACE = (
    "agent-supervisor",
    "adversarial-assurance-engine-v1",
)


def _aae_validator(board):
    module = importlib.import_module(
        "scripts.validate_adversarial_assurance_engine_board"
    )
    module_path = Path(str(module.__file__ or "")).resolve()
    if board.repo_root not in module_path.parents:
        raise AAESchedulerError("AAE validator was imported from another tree")
    return module


def _aae_cid(board, value, *, noun: str) -> str:
    errors: list[str] = []
    identity = str(
        _aae_validator(board)._canonical_cid(
            board.repo_root,
            value,
            noun=noun,
            errors=errors,
        )
    )
    if errors or not identity:
        raise AAESchedulerError(
            f"{noun} canonical identity could not be established"
        )
    return identity


def _aae_verify_signature(
    board,
    *,
    identity_did: str,
    payload,
    signature: str,
) -> None:
    module = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.control.profile_authority"
    )
    module_path = Path(str(module.__file__ or "")).resolve()
    if board.repo_root not in module_path.parents:
        raise AAESchedulerError(
            "AAE signature verifier was imported from another tree"
        )
    try:
        module.verify_did_key_signature(
            identity_did=identity_did,
            payload=payload,
            signature=signature,
        )
    except Exception as exc:
        raise AAESchedulerError(
            "AAE launch admission signature verification failed"
        ) from exc


def _aae_prerequisite_binding(board):
    receipt = _load_json_object(board.path(_AAE_PREREQUISITE_REL))
    status_value = str(receipt.get("status") or "")
    if status_value == "blocked":
        return {"required": False, "status": "blocked"}
    if status_value != "completed":
        raise AAESchedulerError(
            "AAE prerequisite status must be blocked or completed"
        )
    release_report = _aae_validator(board).validate_prerequisites(
        board.repo_root,
        check_repository=True,
    )
    if release_report.get("valid") is not True:
        raise AAESchedulerError(
            "AAE completed prerequisite failed release validation"
        )
    source = board.payload.get("source_binding")
    if not isinstance(source, Mapping):
        raise AAESchedulerError("AAE completed gate has no source binding")
    if source.get("pin_state") != "operator_released":
        raise AAESchedulerError("AAE completed gate pins are not operator-released")
    pin_generation = source.get("pin_generation")
    if type(pin_generation) is not int or pin_generation <= 0:
        raise AAESchedulerError("AAE completed gate pin generation is invalid")
    unsigned = dict(receipt)
    unsigned.pop("canonical_identity", None)
    unsigned.pop("authorization", None)
    recomputed = _aae_cid(board, unsigned, noun="AAE prerequisite receipt")
    receipt_cid = str(receipt.get("canonical_identity") or "")
    if receipt_cid != recomputed:
        raise AAESchedulerError("AAE prerequisite receipt CID differs")
    return {
        "required": True,
        "status": "completed",
        "prerequisite_receipt_cid": receipt_cid,
        "pin_generation": pin_generation,
    }


def _aae_exact_gitlinks(board) -> dict[str, str]:
    source = board.payload.get("source_binding")
    if not isinstance(source, Mapping):
        raise AAESchedulerError("AAE source binding is absent")
    specifications = (
        ("ipfs_datasets_py", "ipfs_datasets_planning_revision"),
        ("ipfs_kit_py", "ipfs_kit_planning_revision"),
        ("ipfs_accelerate_py/mcplusplus", "mcp_plus_plus_planning_revision"),
    )
    result: dict[str, str] = {}
    for relative, field in specifications:
        expected = str(source.get(field) or "").lower()
        observed = _git(board, "rev-parse", f"HEAD:{relative}")
        actual = observed.stdout.strip().lower()
        if (
            observed.returncode != 0
            or len(expected) != 40
            or any(character not in "0123456789abcdef" for character in expected)
            or actual != expected
        ):
            raise AAESchedulerError(
                f"AAE launch admission gitlink differs: {relative}"
            )
        result[relative] = expected
    return result


def _aae_git_common_dir(board) -> Path:
    completed = _git(
        board,
        "rev-parse",
        "--path-format=absolute",
        "--git-common-dir",
    )
    raw = completed.stdout.strip()
    if completed.returncode != 0 or not raw:
        raise AAESchedulerError("cannot resolve the AAE Git common directory")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = board.repo_root / candidate
    try:
        common = candidate.resolve(strict=True)
    except OSError as exc:
        raise AAESchedulerError("AAE Git common directory is unavailable") from exc
    if not common.is_dir() or common.is_symlink():
        raise AAESchedulerError("AAE Git common directory is not a directory")
    return common


def _aae_ledger_paths(board, *, create: bool) -> tuple[Path, Path]:
    cursor = _aae_git_common_dir(board)
    for component in _AAE_LEDGER_NAMESPACE:
        cursor = cursor / component
        if create:
            try:
                cursor.mkdir(mode=0o700)
            except FileExistsError:
                pass
        if cursor.exists() and (not cursor.is_dir() or cursor.is_symlink()):
            raise AAESchedulerError("AAE launch ledger namespace is unsafe")
    return cursor / "launch-admissions.jsonl", cursor / "launch-admissions.lock"


def _aae_read_ledger(board, ledger_path: Path) -> list[dict[str, object]]:
    fd = -1
    try:
        fd = os.open(
            ledger_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise AAESchedulerError("AAE launch ledger is not a regular file")
        if metadata.st_size > _AAE_LEDGER_MAX_BYTES:
            raise AAESchedulerError("AAE launch ledger is too large")
        chunks: list[bytes] = []
        remaining = _AAE_LEDGER_MAX_BYTES + 1
        while remaining:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise AAESchedulerError("cannot read the AAE launch ledger") from exc
    finally:
        if fd >= 0:
            os.close(fd)
    if len(raw) > _AAE_LEDGER_MAX_BYTES:
        raise AAESchedulerError("AAE launch ledger is too large")
    if raw and not raw.endswith(b"\n"):
        raise AAESchedulerError("AAE launch ledger has a partial final entry")
    entries: list[dict[str, object]] = []
    previous_cid: str | None = None
    for index, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise AAESchedulerError("AAE launch ledger contains an empty entry")
        try:
            entry = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AAESchedulerError("AAE launch ledger contains invalid JSON") from exc
        if not isinstance(entry, dict):
            raise AAESchedulerError("AAE launch ledger entry is not an object")
        expected_fields = {
            "schema",
            "board_namespace",
            "launch_generation",
            "previous_ledger_cid",
            "admission_cid",
            "admission",
            "source_head",
            "prerequisite_receipt_cid",
            "pin_generation",
            "gitlinks",
            "consumed_at",
            "entry_cid",
        }
        if set(entry) != expected_fields:
            raise AAESchedulerError("AAE launch ledger entry fields differ")
        if (
            entry.get("schema") != _AAE_LEDGER_ENTRY_SCHEMA
            or entry.get("board_namespace")
            != "adversarial-assurance-engine-v1"
            or entry.get("launch_generation") != index
            or entry.get("previous_ledger_cid") != previous_cid
        ):
            raise AAESchedulerError("AAE launch ledger chain is malformed")
        gitlinks = entry.get("gitlinks")
        admission = entry.get("admission")
        gate = board.payload.get("prerequisite_gate")
        expected_authority = (
            gate.get("operator_authority_did") if isinstance(gate, Mapping) else None
        )
        if (
            re.fullmatch(r"b[a-z2-7]{20,}", str(entry.get("admission_cid") or ""))
            is None
            or re.fullmatch(r"[0-9a-f]{40}", str(entry.get("source_head") or ""))
            is None
            or re.fullmatch(
                r"b[a-z2-7]{20,}",
                str(entry.get("prerequisite_receipt_cid") or ""),
            )
            is None
            or type(entry.get("pin_generation")) is not int
            or int(entry["pin_generation"]) <= 0
            or not isinstance(gitlinks, Mapping)
            or set(gitlinks)
            != {
                "ipfs_datasets_py",
                "ipfs_kit_py",
                "ipfs_accelerate_py/mcplusplus",
            }
            or any(
                re.fullmatch(r"[0-9a-f]{40}", str(value or "")) is None
                for value in gitlinks.values()
            )
            or _parse_time(entry.get("consumed_at")) is None
            or not isinstance(admission, Mapping)
        ):
            raise AAESchedulerError("AAE launch ledger entry binding is malformed")
        admission_unsigned = dict(admission)
        admission_signature = str(admission_unsigned.pop("signature", ""))
        if (
            admission_unsigned.get("schema") != _AAE_ADMISSION_SCHEMA
            or admission_unsigned.get("identity_did") != expected_authority
            or admission_unsigned.get("audience")
            != entry.get("board_namespace")
            or admission_unsigned.get("action")
            != "launch:adversarial-assurance-engine-v1"
            or admission_unsigned.get("launch_generation")
            != entry.get("launch_generation")
            or admission_unsigned.get("previous_ledger_cid")
            != entry.get("previous_ledger_cid")
            or admission_unsigned.get("source_head") != entry.get("source_head")
            or admission_unsigned.get("prerequisite_receipt_cid")
            != entry.get("prerequisite_receipt_cid")
            or admission_unsigned.get("pin_generation")
            != entry.get("pin_generation")
            or admission_unsigned.get("gitlinks") != gitlinks
            or _aae_cid(board, admission_unsigned, noun="AAE launch admission")
            != entry.get("admission_cid")
        ):
            raise AAESchedulerError("AAE historical launch admission differs")
        _aae_verify_signature(
            board,
            identity_did=str(admission_unsigned["identity_did"]),
            payload=admission_unsigned,
            signature=admission_signature,
        )
        unsigned = dict(entry)
        entry_cid = str(unsigned.pop("entry_cid", ""))
        if (
            re.fullmatch(r"b[a-z2-7]{20,}", entry_cid) is None
            or entry_cid
            != _aae_cid(board, unsigned, noun="AAE launch ledger entry")
        ):
            raise AAESchedulerError("AAE launch ledger entry CID differs")
        previous_cid = entry_cid
        entries.append(entry)
    return entries


def _aae_external_admission_path(board) -> Path:
    raw = str(os.environ.get(_AAE_ADMISSION_ENV, "")).strip()
    if not raw:
        raise AAESchedulerError(
            f"completed AAE gate requires {_AAE_ADMISSION_ENV}"
        )
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise AAESchedulerError("AAE launch admission path must be absolute")
    cursor = Path(candidate.anchor)
    try:
        for component in candidate.parts[1:]:
            cursor /= component
            if cursor.is_symlink():
                raise AAESchedulerError(
                    "AAE launch admission path must not contain symlinks"
                )
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise AAESchedulerError("AAE launch admission is unavailable") from exc
    if not resolved.is_file() or not stat.S_ISREG(resolved.stat().st_mode):
        raise AAESchedulerError("AAE launch admission is not a regular file")
    try:
        resolved.relative_to(board.repo_root)
    except ValueError:
        pass
    else:
        raise AAESchedulerError(
            "AAE launch admission must be external to the repository"
        )
    common_dir = _aae_git_common_dir(board)
    try:
        resolved.relative_to(common_dir)
    except ValueError:
        pass
    else:
        raise AAESchedulerError(
            "AAE launch admission must not be self-referential Git state"
        )
    tracked_root = subprocess.run(
        ["git", "-C", str(resolved.parent), "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if tracked_root.returncode == 0 and tracked_root.stdout.strip():
        top = Path(tracked_root.stdout.strip()).resolve(strict=True)
        try:
            relative = resolved.relative_to(top).as_posix()
        except ValueError:
            relative = ""
        if relative:
            tracked = subprocess.run(
                ["git", "-C", str(top), "ls-files", "--error-unmatch", "--", relative],
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            if tracked.returncode == 0:
                raise AAESchedulerError(
                    "AAE launch admission must not be a committed file"
                )
    return resolved


def _aae_load_admission(board) -> dict[str, object]:
    path = _aae_external_admission_path(board)
    parent_fd = -1
    fd = -1
    try:
        parent_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        fd = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise AAESchedulerError("AAE launch admission is not a regular file")
        if metadata.st_size > MAX_JSON_BYTES:
            raise AAESchedulerError("AAE launch admission is too large")
        chunks: list[bytes] = []
        remaining = MAX_JSON_BYTES + 1
        while remaining:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    except OSError as exc:
        raise AAESchedulerError("cannot read the AAE launch admission") from exc
    finally:
        if fd >= 0:
            os.close(fd)
        if parent_fd >= 0:
            os.close(parent_fd)
    if len(raw) > MAX_JSON_BYTES:
        raise AAESchedulerError("AAE launch admission is too large")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AAESchedulerError("AAE launch admission is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise AAESchedulerError("AAE launch admission is not an object")
    return payload


def _aae_validate_launch_admission(
    board,
    plan=None,
    *,
    ledger_path: Path | None = None,
) -> dict[str, object]:
    prerequisite = _aae_prerequisite_binding(board)
    if prerequisite["required"] is not True:
        return {"required": False, "valid": True, "status": "blocked"}
    admission = _aae_load_admission(board)
    expected_fields = {
        "schema",
        "identity_did",
        "audience",
        "action",
        "source_head",
        "prerequisite_receipt_cid",
        "pin_generation",
        "gitlinks",
        "launch_generation",
        "previous_ledger_cid",
        "signature",
    }
    if set(admission) != expected_fields:
        raise AAESchedulerError("AAE launch admission fields differ")
    source_head = _source_head(board)
    if plan is not None and plan.get("source_head") != source_head:
        raise AAESchedulerError("AAE launch plan HEAD differs from admission state")
    gitlinks = _aae_exact_gitlinks(board)
    gate = board.payload.get("prerequisite_gate")
    identity_did = gate.get("operator_authority_did") if isinstance(gate, Mapping) else None
    unsigned = dict(admission)
    signature = str(unsigned.pop("signature", ""))
    expected = {
        "schema": _AAE_ADMISSION_SCHEMA,
        "identity_did": identity_did,
        "audience": "adversarial-assurance-engine-v1",
        "action": "launch:adversarial-assurance-engine-v1",
        "source_head": source_head,
        "prerequisite_receipt_cid": prerequisite[
            "prerequisite_receipt_cid"
        ],
        "pin_generation": prerequisite["pin_generation"],
        "gitlinks": gitlinks,
    }
    for field, value in expected.items():
        if unsigned.get(field) != value:
            raise AAESchedulerError(f"AAE launch admission {field} differs")
    if type(unsigned.get("launch_generation")) is not int or int(
        unsigned["launch_generation"]
    ) <= 0:
        raise AAESchedulerError("AAE launch generation is invalid")
    if unsigned.get("previous_ledger_cid") is not None and not isinstance(
        unsigned.get("previous_ledger_cid"), str
    ):
        raise AAESchedulerError("AAE previous ledger CID is invalid")
    if not signature:
        raise AAESchedulerError("AAE launch admission signature is absent")
    _aae_verify_signature(
        board,
        identity_did=str(identity_did or ""),
        payload=unsigned,
        signature=signature,
    )
    if ledger_path is None:
        ledger_path, _lock_path = _aae_ledger_paths(board, create=False)
    entries = _aae_read_ledger(board, ledger_path)
    previous_cid = str(entries[-1]["entry_cid"]) if entries else None
    expected_generation = len(entries) + 1
    if (
        unsigned["launch_generation"] != expected_generation
        or unsigned.get("previous_ledger_cid") != previous_cid
    ):
        raise AAESchedulerError(
            "AAE launch admission generation or previous ledger CID differs"
        )
    if entries:
        previous = entries[-1]
        previous_pin_generation = int(previous["pin_generation"])
        current_pin_generation = int(prerequisite["pin_generation"])
        if current_pin_generation < previous_pin_generation:
            raise AAESchedulerError("AAE pin generation regressed")
        gate_or_pins_changed = (
            prerequisite["prerequisite_receipt_cid"]
            != previous["prerequisite_receipt_cid"]
            or gitlinks != previous["gitlinks"]
        )
        if gate_or_pins_changed and current_pin_generation <= previous_pin_generation:
            raise AAESchedulerError(
                "AAE changed gate or gitlinks require a higher pin generation"
            )
    admission_cid = _aae_cid(board, unsigned, noun="AAE launch admission")
    return {
        "required": True,
        "valid": True,
        "status": "completed",
        "launch_generation": expected_generation,
        "previous_ledger_cid": previous_cid,
        "admission_cid": admission_cid,
        "admission": admission,
        "source_head": source_head,
        "prerequisite_receipt_cid": prerequisite[
            "prerequisite_receipt_cid"
        ],
        "pin_generation": prerequisite["pin_generation"],
        "gitlinks": gitlinks,
    }


def _aae_preflight_admission(board) -> dict[str, object]:
    return _aae_validate_launch_admission(board)


def _aae_consume_launch_admission(board, plan) -> dict[str, object]:
    prerequisite = _aae_prerequisite_binding(board)
    if prerequisite["required"] is not True:
        return {"required": False, "consumed": False, "status": "blocked"}
    ledger_path, lock_path = _aae_ledger_paths(board, create=True)
    lock_fd = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        admitted = _aae_validate_launch_admission(
            board,
            plan,
            ledger_path=ledger_path,
        )
        entry = {
            "schema": _AAE_LEDGER_ENTRY_SCHEMA,
            "board_namespace": "adversarial-assurance-engine-v1",
            "launch_generation": admitted["launch_generation"],
            "previous_ledger_cid": admitted["previous_ledger_cid"],
            "admission_cid": admitted["admission_cid"],
            "admission": admitted["admission"],
            "source_head": admitted["source_head"],
            "prerequisite_receipt_cid": admitted[
                "prerequisite_receipt_cid"
            ],
            "pin_generation": admitted["pin_generation"],
            "gitlinks": admitted["gitlinks"],
            "consumed_at": _utc_now(),
        }
        entry["entry_cid"] = _aae_cid(
            board, entry, noun="AAE launch ledger entry"
        )
        encoded = _canonical_json(entry) + b"\n"
        try:
            current_size = ledger_path.stat().st_size
        except FileNotFoundError:
            current_size = 0
        except OSError as exc:
            raise AAESchedulerError("cannot inspect the AAE launch ledger") from exc
        if current_size + len(encoded) > _AAE_LEDGER_MAX_BYTES:
            raise AAESchedulerError("AAE launch ledger is too large")
        fd = os.open(
            ledger_path,
            os.O_CREAT
            | os.O_WRONLY
            | os.O_APPEND
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise AAESchedulerError("AAE launch ledger is not a regular file")
            written = os.write(fd, encoded)
            if written != len(encoded):
                raise AAESchedulerError("AAE launch ledger append was incomplete")
            os.fsync(fd)
        finally:
            os.close(fd)
        directory_fd = os.open(ledger_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return {
            **admitted,
            "consumed": True,
            "entry_cid": entry["entry_cid"],
        }
    finally:
        os.close(lock_fd)


_SOURCE = Path(__file__).with_name("incremental_verification_planner_scheduler.py")
_source = _SOURCE.read_text(encoding="utf-8")
for _old, _new in (
    ("incremental-verification planner", "adversarial-assurance engine"),
    ("incremental-verification-planner", "adversarial-assurance-engine"),
    ("incremental_verification_planner", "adversarial_assurance_engine"),
    ("IVP", "AAE"),
    ("ivp", "aae"),
    (
        "config/agent_supervisor_adversarial_assurance_engine_scheduler.json",
        "config/adversarial_assurance_engine_scheduler.json",
    ),
):
    _source = _source.replace(_old, _new)

# The inherited IVP controller historically bound the common runner arguments
# to the cardinality of one particular profile.  AAE's protected controls may
# legitimately grow before its operator-owned profile is sealed.  Replace that
# brittle length check with equality against the exact arguments rendered from
# the admitted board.  This remains fail-closed and also detects value, order,
# duplication, or omission drift rather than checking only the final count.
_old_guard = (
    "    if len(tracks) != board.max_lanes or len(common) != 59:\n"
)
_new_guard = (
    "    expected_common = common_supervisor_args(board, implement=True)\n"
    "    if (\n"
    "        len(tracks) != board.max_lanes\n"
    "        or tuple(common) != expected_common\n"
    "    ):\n"
)
if _source.count(_old_guard) != 1:
    raise RuntimeError("inherited runner cardinality guard changed unexpectedly")
_source = _source.replace(_old_guard, _new_guard)

# IVP's original profile owns two planning gitlinks.  AAE also owns the scoped
# MCP++ schema/vector gitlink, so its launch preflight must bind all three
# declared repository authorities.  Keep the inherited clean/gitlink/HEAD
# checks and add only the third reviewed source-binding pair.
_old_gitlinks = (
    "    gitlink_specs = (\n"
    "        (\n"
    "            str(source.get(\"ipfs_kit_submodule_path\") or \"\"),\n"
    "            str(source.get(\"ipfs_kit_planning_revision\") or \"\"),\n"
    "        ),\n"
    "        (\n"
    "            str(source.get(\"ipfs_datasets_submodule_path\") or \"\"),\n"
    "            str(source.get(\"ipfs_datasets_planning_revision\") or \"\"),\n"
    "        ),\n"
    "    )\n"
)
_new_gitlinks = (
    "    gitlink_specs = (\n"
    "        (\n"
    "            str(source.get(\"ipfs_kit_submodule_path\") or \"\"),\n"
    "            str(source.get(\"ipfs_kit_planning_revision\") or \"\"),\n"
    "        ),\n"
    "        (\n"
    "            str(source.get(\"ipfs_datasets_submodule_path\") or \"\"),\n"
    "            str(source.get(\"ipfs_datasets_planning_revision\") or \"\"),\n"
    "        ),\n"
    "        (\n"
    "            str(source.get(\"mcp_plus_plus_submodule_path\") or \"\"),\n"
    "            str(source.get(\"mcp_plus_plus_planning_revision\") or \"\"),\n"
    "        ),\n"
    "    )\n"
)
if _source.count(_old_gitlinks) != 1:
    raise RuntimeError("inherited source gitlink checks changed unexpectedly")
_source = _source.replace(_old_gitlinks, _new_gitlinks)

# AAE has one intentional, operator-owned release gate.  The ordinary IVP
# status projection treats any blocked task as a campaign-wide failure, which
# would report a false stall while independent pre-runtime work is progressing.
# Preserve the blocked count and expose the exact expected gate, but suppress
# only that one blocker from lifecycle failure.  Terminal detection remains
# unchanged and still requires all tasks completed with zero blocked work.
_old_blocker = (
    '    blockers: list[str] = []\n'
    '    if (counts["blocked_count"] or 0) > 0:\n'
    '        blockers.append(f"blocked_tasks_present:{counts[\'blocked_count\']}")\n'
)
_new_blocker = (
    '    blockers: list[str] = []\n'
    '    blocked_task_ids = {\n'
    '        str(value) for value in (task_payload.get("blocked_task_ids") or ())\n'
    '    }\n'
    '    expected_operator_gate_blocked = (\n'
    '        counts["blocked_count"] == 1\n'
    '        and blocked_task_ids == {"AAE-006"}\n'
    '    )\n'
    '    if (counts["blocked_count"] or 0) > 0 and not expected_operator_gate_blocked:\n'
    '        blockers.append(f"blocked_tasks_present:{counts[\'blocked_count\']}")\n'
)
if _source.count(_old_blocker) != 1:
    raise RuntimeError("inherited lane blocker projection changed unexpectedly")
_source = _source.replace(_old_blocker, _new_blocker)
_return_anchor = '        "blockers": blockers,\n        **counts,\n'
_return_replacement = (
    '        "blockers": blockers,\n'
    '        "expected_operator_gate_blocked": expected_operator_gate_blocked,\n'
    '        **counts,\n'
)
if _source.count(_return_anchor) != 1:
    raise RuntimeError("inherited lane status result changed unexpectedly")
_source = _source.replace(_return_anchor, _return_replacement)

# A blocked bootstrap gate deliberately needs no post-gate launch signature.
# Once the operator completes AAE-006, however, every launch is a distinct
# signed act.  Extend preflight with a non-consuming admission proof and consume
# that proof under the inherited lifecycle lock immediately before process
# birth.  The durable generation chain itself lives in the Git common-dir and
# therefore cannot be erased by replacement of the lifecycle projection.
_preflight_anchor = (
    "    except (AAESchedulerError, OSError, subprocess.SubprocessError, ValueError) as exc:\n"
    "        checks[\"provider_route_ready\"] = False\n"
    "        errors.append(str(exc))\n\n"
    "    return {\n"
)
_preflight_replacement = (
    "    except (AAESchedulerError, OSError, subprocess.SubprocessError, ValueError) as exc:\n"
    "        checks[\"provider_route_ready\"] = False\n"
    "        errors.append(str(exc))\n\n"
    "    launch_admission: dict[str, Any] = {}\n"
    "    try:\n"
    "        launch_admission = _aae_preflight_admission(board)\n"
    "        checks[\"launch_admission\"] = True\n"
    "    except (AAESchedulerError, OSError, ValueError) as exc:\n"
    "        checks[\"launch_admission\"] = False\n"
    "        errors.append(str(exc))\n\n"
    "    return {\n"
)
if _source.count(_preflight_anchor) != 1:
    raise RuntimeError("inherited preflight provider boundary changed unexpectedly")
_source = _source.replace(_preflight_anchor, _preflight_replacement)
_preflight_result_anchor = '        "provider": provider_report,\n        "collision": collision,\n'
_preflight_result_replacement = (
    '        "provider": provider_report,\n'
    '        "launch_admission": launch_admission,\n'
    '        "collision": collision,\n'
)
if _source.count(_preflight_result_anchor) != 1:
    raise RuntimeError("inherited preflight result changed unexpectedly")
_source = _source.replace(
    _preflight_result_anchor,
    _preflight_result_replacement,
)

_consume_anchor = (
    '        if plan.get("source_head") != _source_head(board):\n'
    '            raise AAESchedulerError("source HEAD changed immediately before launch")\n'
    '        identity = process_adapter.launch(profile, fencing_epoch=0)\n'
)
_consume_replacement = (
    '        if plan.get("source_head") != _source_head(board):\n'
    '            raise AAESchedulerError("source HEAD changed immediately before launch")\n'
    '        _aae_consume_launch_admission(board, plan)\n'
    '        identity = process_adapter.launch(profile, fencing_epoch=0)\n'
)
if _source.count(_consume_anchor) != 1:
    raise RuntimeError("inherited pre-spawn lifecycle boundary changed unexpectedly")
_source = _source.replace(_consume_anchor, _consume_replacement)

# Execute a source-specialized copy so all constants, type names, lifecycle
# files, task prefix, and status schemas are AAE-bound before any command runs.
exec(compile(_source, str(Path(__file__)), "exec"), globals(), globals())
