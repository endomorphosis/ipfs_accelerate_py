"""Receipt-bound reconciliation of a completed restart with delayed health.

This observation-only API never repeats an action, alters a terminal receipt,
refunds an ordinal, or declares recovery. The application verifier has the same
trusted boundary as the campaign observer: it must inspect actual owner and
work evidence, not accept caller-supplied success flags.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

from ..prompt.prompt_workflow import (
    ProgrammaticRecoveryExhaustionReceipt, prompt_workflow_cid, _validate_cid,
)
from .supervisor_recovery import (
    PROGRAMMATIC_RECOVERY_CATALOG_CID, ProgrammaticRecoveryPolicy,
)

_PROOF_FIELDS = {"schema", "scope", "incident_key", "terminal_cid", "recovery_ordinal",
    "action_receipt_cid", "restart_owner_token", "restart_started_at_ms",
    "progress_observed_at_ms", "progress_task_id", "progress_task_cid", "progress_attempt",
    "current_owner_token", "evidence_cids", "effects_verified", "later_work_verified"}


def _bytes(path, *, limit=524288):
    path = Path(path)
    if path.resolve(strict=True) != path.absolute():
        raise ValueError("campaign_reconciliation_file_alias")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_size > limit):
            raise ValueError("campaign_reconciliation_file_unowned")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
        if (len(raw) != before.st_size or any(getattr(before, field) != getattr(after, field)
                for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))):
            raise ValueError("campaign_reconciliation_file_changed")
        return raw


def _active(value):
    return (value["status"] == "active" and value["healthy"] and value["progress_verified"]
        and value["owner_binding_verified"] and not value["manual_hold"] and not value["hold_reason"]
        and not value["completion_verified"] and value["active_task_id"] in value["pending_task_ids"]
        and bool(value["active_task_cid"]) and value["attempt"] > 0)


def _terminal(watchdog, entry):
    directory = watchdog.state_dir / "programmatic-recovery"
    info = directory.lstat()
    if (directory.resolve(strict=True) != directory or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()):
        raise ValueError("campaign_reconciliation_terminal_directory_unowned")
    paths = list(directory.iterdir())
    if len(paths) > watchdog.policy.max_incidents * watchdog.policy.max_recovery_attempts:
        raise ValueError("campaign_reconciliation_terminal_bound")
    policy = ProgrammaticRecoveryPolicy(max_attempts_per_action=1, max_total_attempts=1,
        max_actions=2, cooldown_ms=watchdog.policy.cooldown_ms, deadline_ms=watchdog.policy.deadline_ms)
    matches = []
    for path in paths:
        if not re.fullmatch(r"[0-9a-f]{64}\.json", path.name):
            continue
        raw = _bytes(path, limit=policy.max_receipt_bytes)
        value = json.loads(raw)
        terminal = value.get("terminal") if isinstance(value, dict) else None
        if not isinstance(terminal, dict) or terminal.get("content_id") != entry["terminal_cid"]:
            continue
        if (set(value) != {"schema", "incident_cid", "recovery_catalog_cid", "recovery_policy_cid", "terminal_kind", "terminal"}
                or value["schema"] != "programmatic-recovery-result@1" or value["terminal_kind"] != "exhaustion"
                or value["recovery_policy_cid"] != policy.policy_cid
                or value["recovery_catalog_cid"] != PROGRAMMATIC_RECOVERY_CATALOG_CID):
            raise ValueError("campaign_reconciliation_terminal_binding_invalid")
        expected_name = hashlib.sha256((value["incident_cid"] + ":" + policy.policy_cid
            + ":" + PROGRAMMATIC_RECOVERY_CATALOG_CID).encode()).hexdigest() + ".json"
        receipt = ProgrammaticRecoveryExhaustionReceipt.from_dict(terminal)
        if (expected_name != path.name or receipt.incident_cid != value["incident_cid"]
                or receipt.receipt_cid != entry["terminal_cid"]
                or any(getattr(receipt, key) != member for key, member in watchdog.scope.items())
                or receipt.exhaustion_reason != "compensation_unavailable"
                or len(receipt.attempts) != 1):
            raise ValueError("campaign_reconciliation_terminal_binding_invalid")
        attempt = receipt.attempts[0]
        if (attempt.operation.value != "restart_lane" or attempt.outcome.value != "failed"
                or attempt.attempt != 1 or attempt.target_id != watchdog.scope["run_cid"]
                or attempt.started_at_ms < entry["last_attempt_ms"]
                or attempt.finished_at_ms >= entry["last_attempt_ms"] + watchdog.policy.deadline_ms):
            raise ValueError("campaign_reconciliation_action_not_health_pending")
        matches.append((receipt, attempt, path, raw))
    if len(matches) != 1:
        raise ValueError("campaign_reconciliation_terminal_missing_or_ambiguous")
    return matches[0]


def _action_payload(value, receipt, attempt):
    expected = {"schema": "programmatic-recovery-action-observation@1", "incident_cid": receipt.incident_cid,
        "operation": "restart_lane", "target_id": attempt.target_id, "attempt": 1, "outcome": "failed",
        "observation": {"succeeded": False, "observed_effects": ["old_lane_fenced", "lane_restarted"],
            "post_action_health": {"healthy": False}, "reason": "application_recovery_observed", "partial": True}}
    if not isinstance(value, Mapping):
        raise ValueError("campaign_reconciliation_action_payload_invalid")
    value = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    if value.get("observation", {}).get("partial") is False:
        expected["observation"]["partial"] = False
    if value != expected or prompt_workflow_cid(value) != attempt.receipt_cid:
        raise ValueError("campaign_reconciliation_action_payload_invalid")
    return value


def _proof(watchdog, value, context, observation, now):
    if not isinstance(value, Mapping) or set(value) != _PROOF_FIELDS:
        raise ValueError("campaign_reconciliation_proof_schema_invalid")
    value = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    if (value["schema"] != "campaign-known-restart-effects-proof@1"
            or any(value[key] != context[key] for key in ("scope", "incident_key", "terminal_cid", "recovery_ordinal", "action_receipt_cid"))
            or value["current_owner_token"] != observation["owner_token"]
            or value["effects_verified"] is not True or value["later_work_verified"] is not True
            or not re.fullmatch(r"[0-9a-f]{64}", str(value["restart_owner_token"]))):
        raise ValueError("campaign_reconciliation_proof_binding_invalid")
    for key in ("restart_started_at_ms", "progress_observed_at_ms", "progress_attempt"):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError("campaign_reconciliation_proof_time_invalid")
    if (not context["action_started_at_ms"] <= value["restart_started_at_ms"] <= context["action_finished_at_ms"]
            or not context["action_finished_at_ms"] < value["progress_observed_at_ms"] <= now
            or value["progress_attempt"] <= 0
            or type(value["progress_task_id"]) is not str or not value["progress_task_id"]):
        raise ValueError("campaign_reconciliation_proof_time_invalid")
    _validate_cid(value["progress_task_cid"], "progress_task_cid")
    cids = value["evidence_cids"]
    if not isinstance(cids, list) or not 3 <= len(cids) <= 16 or len(set(cids)) != len(cids):
        raise ValueError("campaign_reconciliation_proof_evidence_invalid")
    for cid in cids:
        _validate_cid(cid, "evidence_cid")
    return value


def reconcile_health_pending_restart(watchdog, *, incident_key, expected_terminal_cid,
                                    action_observation, verify_effects):
    """Append a receipt and resolve the narrow known-effects misclassification."""
    try:
        if not callable(verify_effects) or not re.fullmatch(r"[0-9a-f]{64}", str(incident_key)):
            raise ValueError("campaign_reconciliation_request_invalid")
        _validate_cid(expected_terminal_cid, "terminal_cid")
        before, now = watchdog._observe(), watchdog._now()
        if not _active(before):
            return watchdog._result("refused", "campaign_reconciliation_fresh_owned_work_required")
        watchdog._directory()
        lock = watchdog.state_dir / ".campaign-progress-watchdog.lock"
        fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        with os.fdopen(fd, "r+b") as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600):
                raise ValueError("campaign_reconciliation_lock_unowned")
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return watchdog._result("refused", "watchdog_owner_busy")
            state = watchdog._load()
            entry = state["incidents"].get(incident_key)
            if (not isinstance(entry, dict) or entry.get("terminal_cid") != expected_terminal_cid
                    or entry.get("phase") not in {"uncertain", "finished"}
                    or type(entry.get("attempts")) is not int or not 0 < entry["attempts"] < watchdog.policy.max_recovery_attempts):
                raise ValueError("campaign_reconciliation_incident_invalid_or_exhausted")
            receipt, attempt, terminal_path, terminal_raw = _terminal(watchdog, entry)
            _action_payload(action_observation, receipt, attempt)
            context = {"schema": "campaign-recovery-reconciliation-context@1", "scope": dict(watchdog.scope),
                "incident_key": incident_key, "terminal_cid": expected_terminal_cid, "recovery_ordinal": entry["attempts"],
                "action_receipt_cid": attempt.receipt_cid, "action_started_at_ms": attempt.started_at_ms,
                "action_finished_at_ms": attempt.finished_at_ms}
            proof = _proof(watchdog, verify_effects(deepcopy(context)), context, before, now)
            if watchdog._observe() != before or _proof(watchdog, verify_effects(deepcopy(context)), context, before, watchdog._now()) != proof:
                raise ValueError("campaign_reconciliation_observation_changed")
            if _bytes(terminal_path) != terminal_raw or watchdog._load() != state:
                raise ValueError("campaign_reconciliation_custody_changed")
            if entry["phase"] == "finished":
                cid = entry.get("reconciliation_cid")
                if not cid:
                    raise ValueError("campaign_reconciliation_incident_not_uncertain")
                path = watchdog.state_dir / "reconciliations" / (incident_key + ".json")
                saved = json.loads(_bytes(path))
                expected = saved.pop("content_id", None)
                if (expected != cid or prompt_workflow_cid(saved) != cid or saved.get("terminal_cid") != expected_terminal_cid
                        or saved.get("scope") != watchdog.scope or saved.get("recovery_ordinal") != entry["attempts"]
                        or saved.get("proof") != proof):
                    raise ValueError("campaign_reconciliation_saved_receipt_invalid")
                return watchdog._result("reconciled", "known_restart_effects_already_reconciled",
                    reconciliation_cid=cid, recovery_attempts=entry["attempts"])
            resolution = {"schema": "campaign-recovery-effects-reconciliation@1", "scope": dict(watchdog.scope),
                "incident_key": incident_key, "terminal_cid": expected_terminal_cid, "recovery_ordinal": entry["attempts"],
                "action_receipt_cid": attempt.receipt_cid, "previous_phase": "uncertain", "proof": proof,
                "completion_authority": False, "attempt_refunded": False, "provider_calls": 0}
            cid = prompt_workflow_cid(resolution)
            resolution["content_id"] = cid
            directory = watchdog.state_dir / "reconciliations"
            directory.mkdir(mode=0o700, exist_ok=True)
            info = directory.lstat()
            if (directory.resolve(strict=True) != directory or not stat.S_ISDIR(info.st_mode)
                    or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700):
                raise ValueError("campaign_reconciliation_directory_unowned")
            path = directory / (incident_key + ".json")
            encoded = (json.dumps(resolution, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
            if path.exists() or os.path.lexists(path):
                if _bytes(path) != encoded:
                    raise ValueError("campaign_reconciliation_receipt_conflict")
            else:
                fd, temporary = tempfile.mkstemp(prefix=".reconciliation-", dir=directory)
                try:
                    with os.fdopen(fd, "wb") as output:
                        output.write(encoded); output.flush(); os.fsync(output.fileno())
                    os.link(temporary, path)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            # Preserve the ordinal, terminal, cooldown, policy and old evidence.
            entry.update(phase="finished", confirmations=0, last_confirmation_ms=0, reconciliation_cid=cid)
            watchdog._store(state)
            return watchdog._result("reconciled", "known_restart_effects_verified", reconciliation_cid=cid,
                recovery_attempts=entry["attempts"], recovery_verified=False)
    except Exception as exc:
        reason = str(exc) if isinstance(exc, ValueError) and str(exc).startswith("campaign_") else type(exc).__name__
        return watchdog._result("refused", reason)
