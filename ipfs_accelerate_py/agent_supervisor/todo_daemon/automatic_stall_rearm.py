"""Bounded automatic rearm for operator-class implementation stalls.

The live boards sat idle after reboot and after a repair burned its one
per-revision retry: leftover ``cooldown_until`` values, attempt counters
restored from checkpoints, identical-failure diagnostic backoff, and
ephemeral protected-path incidents whose worktrees were already gone.

This policy grants **one** rearm per (canonical task, stall class, daemon
process generation). A new process (reboot or recycle) may rearm again. The
same process will not loop every pass. It never marks tasks complete, never
appends board work, and never touches plan/objectives files.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

AUTOMATIC_STALL_REARM_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/automatic-stall-rearm@1"
)
AUTOMATIC_STALL_REARM_POLICY = "automatic-stall-rearm-v1"
STALL_MAX_TASK_ATTEMPTS = "max_task_attempts"
STALL_IDENTICAL_FAILURE = "identical_implementation_failure"
STALL_QUEUE_COOLDOWN = "queue_cooldown"
STALL_DISPOSED_EPHEMERAL_INCIDENT = "disposed_ephemeral_incident"
STALL_REPAIR_ROUND_BUDGET = "repair_round_budget"
STALL_FRESH_COMPILE_RESCUE = "fresh_compile_rescue"
_RECEIPT_NAME = "automatic_stall_rearm.json"
_ACTIVE_INCIDENT_NAME = "implementation-protected-path-active.json"


def process_generation(
    *,
    pid: int,
    started_at: str,
    state_dir: str,
    generation_seq: int = 1,
) -> str:
    payload = {
        "schema": AUTOMATIC_STALL_REARM_SCHEMA,
        "pid": int(pid),
        "started_at": str(started_at),
        "state_dir": str(state_dir),
        "generation_seq": int(generation_seq),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return "sha256:" + digest


def rearm_fingerprint(
    *,
    process_generation_id: str,
    canonical_task_cid: str,
    stall_class: str,
) -> str:
    payload = {
        "schema": AUTOMATIC_STALL_REARM_SCHEMA,
        "policy": AUTOMATIC_STALL_REARM_POLICY,
        "process_generation": str(process_generation_id),
        "canonical_task_cid": str(canonical_task_cid),
        "stall_class": str(stall_class),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return "sha256:" + digest


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def load_or_create_process_generation(
    receipt_path: Path,
    *,
    pid: int,
    now: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """Reuse a generation while this pid owns the receipt; otherwise start one."""

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now or time.time()))
    existing = _load_json(receipt_path) if receipt_path.exists() else None
    if (
        isinstance(existing, dict)
        and int(existing.get("pid") or 0) == int(pid)
        and str(existing.get("process_generation") or "").startswith("sha256:")
    ):
        try:
            os.kill(int(pid), 0)
        except OSError:
            pass
        else:
            return str(existing["process_generation"]), existing

    previous_seq = 0
    previous_receipts: dict[str, Any] = {}
    if isinstance(existing, dict):
        try:
            previous_seq = int(existing.get("generation_seq") or 0)
        except (TypeError, ValueError):
            previous_seq = 0
        if isinstance(existing.get("receipts"), dict):
            previous_receipts = dict(existing.get("receipts") or {})
    generation_seq = previous_seq + 1
    generation = process_generation(
        pid=int(pid),
        started_at=started_at,
        state_dir=str(receipt_path.parent),
        generation_seq=generation_seq,
    )
    payload = {
        "schema": AUTOMATIC_STALL_REARM_SCHEMA,
        "policy": AUTOMATIC_STALL_REARM_POLICY,
        "pid": int(pid),
        "started_at": started_at,
        "generation_seq": generation_seq,
        "process_generation": generation,
        "receipts": previous_receipts,
    }
    _write_json(receipt_path, payload)
    return generation, payload


def workspace_has_live_runner(workspace: str) -> bool:
    """Return whether a grok/implement process still owns ``workspace``."""

    needle = str(workspace or "").strip()
    if not needle:
        return False
    proc = Path("/proc")
    if not proc.is_dir():
        return False
    markers = ("grok_cli_runner", "implementation_daemon", "/opt/ipfs-accelerate/grok")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\x00", b" ")
        except OSError:
            continue
        decoded = cmd.decode("utf-8", "replace")
        if needle in decoded and any(marker in decoded for marker in markers):
            return True
    return False


def diagnostic_stall_class(
    payload: Mapping[str, Any] | None,
    *,
    now: float,
    max_repair_rounds: int,
) -> str:
    if not isinstance(payload, Mapping):
        return ""
    try:
        repeats = int(payload.get("repeat_count") or 0)
    except (TypeError, ValueError):
        repeats = 0
    try:
        not_before = float(payload.get("not_before") or 0.0)
    except (TypeError, ValueError):
        not_before = 0.0
    if repeats >= max(1, int(max_repair_rounds)) or not_before > now:
        return STALL_IDENTICAL_FAILURE
    return ""


def reset_diagnostic_state(path: Path) -> bool:
    payload = _load_json(path)
    if payload is None:
        return False
    changed = False
    if int(payload.get("repeat_count") or 0) != 1:
        payload["repeat_count"] = 1
        changed = True
    if float(payload.get("not_before") or 0.0) != 0.0:
        payload["not_before"] = 0.0
        changed = True
    if changed:
        _write_json(path, payload)
    return changed


def archive_disposed_ephemeral_incident(active_path: Path, *, stamp: str) -> bool:
    payload = _load_json(active_path)
    if payload is None:
        return False
    if payload.get("ephemeral_worktree") is not True:
        return False
    workspace = str(payload.get("workspace_path") or "").strip()
    if not workspace:
        return False
    workspace_path = Path(workspace)
    if workspace_path.exists():
        return False
    if workspace_has_live_runner(workspace):
        return False
    dest = active_path.with_name(
        f"{active_path.name}.automatic-stall-rearm-{stamp}"
    )
    active_path.replace(dest)
    return True


def queue_entry_cooled_down(entry: Mapping[str, Any] | None, *, now: float) -> bool:
    if not isinstance(entry, Mapping):
        return False
    try:
        return float(entry.get("cooldown_until") or 0.0) > now
    except (TypeError, ValueError):
        return False


def collect_stall_classes(
    *,
    attempt_count: int,
    max_task_attempts: int,
    queue_cooled_down: bool,
    diagnostic_class: str,
    disposed_ephemeral_incident: bool,
    max_repair_rounds: int = 0,
) -> tuple[str, ...]:
    stalls: list[str] = []
    if max_task_attempts > 0 and attempt_count >= max_task_attempts:
        stalls.append(STALL_MAX_TASK_ATTEMPTS)
    if max_repair_rounds > 0 and attempt_count > max_repair_rounds:
        # Next dispatch is attempt_count + 1, so repair_round == attempt_count.
        stalls.append(STALL_REPAIR_ROUND_BUDGET)
    if diagnostic_class:
        stalls.append(diagnostic_class)
    if queue_cooled_down:
        stalls.append(STALL_QUEUE_COOLDOWN)
    if disposed_ephemeral_incident:
        stalls.append(STALL_DISPOSED_EPHEMERAL_INCIDENT)
    return tuple(stalls)


def claim_rearm_receipt(
    receipt_path: Path,
    *,
    pid: int,
    canonical_task_cid: str,
    stall_class: str,
    now: float | None = None,
) -> tuple[bool, str]:
    """Grant one receipt for ``stall_class``. Return ``(claimed_now, generation)``."""

    generation, ledger = load_or_create_process_generation(
        receipt_path,
        pid=int(pid),
        now=now,
    )
    receipts = {
        str(key): [str(item) for item in value if str(item).strip()]
        for key, value in (ledger.get("receipts") or {}).items()
        if isinstance(value, list)
    }
    fingerprint = rearm_fingerprint(
        process_generation_id=generation,
        canonical_task_cid=str(canonical_task_cid),
        stall_class=str(stall_class),
    )
    if receipts_already_granted(
        receipts,
        canonical_task_cid=str(canonical_task_cid),
        fingerprints=(fingerprint,),
    ):
        return False, generation
    prior = receipts.setdefault(str(canonical_task_cid), [])
    if fingerprint not in prior:
        prior.append(fingerprint)
    ledger["receipts"] = receipts
    ledger["last_rearm_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(now or time.time())
    )
    _write_json(receipt_path, ledger)
    return True, generation


def should_continue_after_repair_budget(
    *,
    claimed: bool,
    other_unrelated_ready_count: int,
) -> bool:
    """Keep a lone ready task implementing after the one-shot rescue is spent.

    Multi-task boards still fail-close so an exhausted task cannot starve
    peers. A retry-budget repair of the same source is not a peer.
    """

    if claimed:
        return True
    return int(other_unrelated_ready_count or 0) <= 0


def recover_retry_compile(
    *,
    compile_retry: Any,
    compile_fresh: Any,
    budget_errors: tuple[type[BaseException], ...],
) -> tuple[str, Any]:
    """Use a retry-delta compile, or one fresh compile if the delta cannot fit."""

    try:
        return "retry", compile_retry()
    except budget_errors:
        return "fresh", compile_fresh()


def safe_run_once(run_once: Any, *, logger: Any | None = None) -> dict[str, Any]:
    """Keep a long-lived daemon process alive after a single pass crash."""

    try:
        result = run_once()
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if logger is not None:
            logger.exception(
                "implementation daemon pass crashed; continuing after bounded stall"
            )
        return {
            "crashed": True,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "selection_idle_reason": "daemon_pass_crashed_recovered",
        }
    if isinstance(result, dict):
        return result
    return {"result": result}


def receipts_already_granted(
    receipts: Mapping[str, Sequence[str]],
    *,
    canonical_task_cid: str,
    fingerprints: Sequence[str],
) -> bool:
    prior = [
        str(item)
        for item in (receipts.get(canonical_task_cid) or [])
        if str(item).strip()
    ]
    return bool(fingerprints) and all(item in prior for item in fingerprints)
