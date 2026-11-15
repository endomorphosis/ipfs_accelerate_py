"""Census owner-private coding children against their final router receipts.

This is an observation after worker cleanup. Event files never authorize a
launch, retry, settlement or completion. Missing evidence keeps usage coverage
unknown while original provider receipts remain available to their consumers.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

from ipfs_accelerate_py.agent_supervisor.runtime.source_formalization_context import _protected_directory
from .terminal_output_observation import _identity, _open_directory, _read

SCHEMA = "terminal-owner-worker-attempt-observation@1"
EVENT_SCHEMA = "owner-worker-attempt-event@1"
MAX_ATTEMPTS = 128
MAX_EVENT_BYTES = 16384
MAX_DIRECTORY_ENTRIES = 1024
MAX_RECEIPTS = 512
MAX_RECEIPT_BYTES = 8 * 1024 * 1024
_MAX_INTEGER = (1 << 63) - 1
_NAME = re.compile(r"worker-attempt-([0-9a-f]{32})-(prepared|spawned|retired)\.json")
_FLAGS = dict(observation_only=True, execution_authority=False,
    completion_authority=False, retry_authority=False)
_COMMON = {"schema", "event", "attempt_id", "purpose", "native_prompt_sha256",
    "workspace_identity_sha256", "boundary_manifest_sha256", "absolute_deadline_ns",
    "owner_pid", "owner_start_ticks", *_FLAGS}
_EXTRAS = {"prepared": set(),
    "spawned": {"prepared_sha256", "child_pid", "child_start_ticks"},
    "retired": {"prepared_sha256", "spawned_sha256", "child_pid", "child_start_ticks",
        "waited", "child_returncode", "cleanup_succeeded", "brokers_closed"}}


def _need(condition):
    if not condition:
        raise ValueError("unavailable owner worker attempt evidence")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hex(value, size=64):
    _need(type(value) is str and re.fullmatch("[0-9a-f]{" + str(size) + "}", value) is not None)
    return value


def _integer(value, low=1, high=_MAX_INTEGER):
    return type(value) is int and low <= value <= high


def _base(status="unavailable", reason="evidence_unavailable", *, selected=True):
    return dict(schema=SCHEMA, status=status, reason=reason,
        all_attempts_receipted=False if selected else None,
        attempts=[], counts=None, issues=[], **_FLAGS,
        settlement_authority=False, proof_authority=False,
        provider_tokens=None, total_token_savings_measured=False,
        provider_usage_complete_established=False, raw_prompts_exported=False,
        raw_sources_exported=False, raw_paths_exported=False,
        scope="owner coding-child census; not provider billing or native token completeness")


def _parse(raw, attempt_id, event):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            _need(key not in value)
            value[key] = item
        return value
    def nonfinite(_):
        raise ValueError("nonfinite worker event")
    row = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    _need(type(row) is dict and _canonical(row) == raw
        and set(row) == _COMMON | _EXTRAS[event]
        and row["schema"] == EVENT_SCHEMA and row["event"] == event
        and row["attempt_id"] == attempt_id and row["purpose"] == "coding"
        and all(type(row[key]) is type(value) and row[key] == value for key, value in _FLAGS.items())
        and all(_integer(row[key]) for key in ("absolute_deadline_ns", "owner_pid", "owner_start_ticks")))
    _hex(row["attempt_id"], 32)
    for key in ("native_prompt_sha256", "workspace_identity_sha256", "boundary_manifest_sha256"):
        _hex(row[key])
    if event != "prepared":
        _hex(row["prepared_sha256"])
    if event == "spawned":
        _need(_integer(row["child_pid"]) and _integer(row["child_start_ticks"])
            and row["child_pid"] != row["owner_pid"])
    if event == "retired":
        _need(all(type(row[key]) is bool for key in ("waited", "cleanup_succeeded", "brokers_closed")))
        if row["spawned_sha256"] is None:
            _need(row["child_pid"] is row["child_start_ticks"] is row["child_returncode"] is None
                and row["waited"] is False)
        else:
            _hex(row["spawned_sha256"])
            _need(_integer(row["child_pid"]) and _integer(row["child_start_ticks"])
                and row["child_pid"] != row["owner_pid"])
            _need(_integer(row["child_returncode"], -255, 255) if row["waited"]
                else row["child_returncode"] is None)
    return row


def _names(fd):
    """Bound enumeration before retaining directory names or opening files."""
    names = []
    with os.scandir(fd) as entries:
        for count, entry in enumerate(entries, 1):
            _need(count <= MAX_DIRECTORY_ENTRIES)
            if entry.name.startswith("worker-attempt-"):
                _need(_NAME.fullmatch(entry.name) is not None)
                names.append(entry.name)
                _need(len(names) <= 3 * MAX_ATTEMPTS)
    return tuple(sorted(names))


def _events(parent):
    owner = os.geteuid()
    _need(owner > 0 and owner == os.getuid())
    parent = Path(parent).absolute()
    _protected_directory(parent, owner=owner, group=owner)
    fd = _open_directory(parent)
    try:
        directory = os.fstat(fd)
        _need(_identity(directory) == _identity(parent.lstat()))
        names = _names(fd)
        groups, identities, issues = {}, {}, []
        for name in names:
            attempt_id, event = _NAME.fullmatch(name).groups()
            _need(attempt_id in groups or len(groups) < MAX_ATTEMPTS)
            group = groups.setdefault(attempt_id, {})
            try:
                before = os.stat(name, dir_fd=fd, follow_symlinks=False)
                _need(before.st_gid == directory.st_gid and stat.S_IMODE(before.st_mode) == 0o400)
                raw = _read(fd, name, limit=MAX_EVENT_BYTES, private=True)
                after = os.stat(name, dir_fd=fd, follow_symlinks=False)
                _need(_identity(before) == _identity(after) and after.st_gid == directory.st_gid)
                identities[name] = after
                group[event] = (_parse(raw, attempt_id, event), _sha(raw))
            except (OSError, ValueError, TypeError, KeyError, RecursionError, OverflowError, UnicodeError):
                issues.append("invalid_event")
        _need(_names(fd) == names)
        for name, before in identities.items():
            after = os.stat(name, dir_fd=fd, follow_symlinks=False)
            _need(_identity(before) == _identity(after) and before.st_gid == after.st_gid)
        _protected_directory(parent, owner=owner, group=owner)
        _need(_identity(directory) == _identity(os.fstat(fd)) == _identity(parent.lstat()))
        return groups, issues
    finally:
        os.close(fd)


def _receipts(report):
    rows = report.get("provider_invocations")
    conflicts = report.get("provider_invocation_conflicts", [])
    _need(type(rows) is list and type(conflicts) is list and len(rows) + len(conflicts) <= MAX_RECEIPTS)
    seen, coding, issues, total = {}, {}, [], 0
    for row in (*rows, *conflicts):
        _need(type(row) is dict)
        raw = _canonical(row)
        total += len(raw)
        _need(total <= MAX_RECEIPT_BYTES)
        identity = row.get("invocation_id")
        _need(type(identity) is str and 0 < len(identity.encode("utf-8")) <= 512)
        if identity in seen:
            if seen[identity] != raw:
                issues.append("conflicting_invocation_receipts")
            else:
                continue
        else:
            seen[identity] = raw
        if row.get("purpose") == "planning" and row.get("phase") == "planning":
            continue
        if (row.get("schema") != "router-implementation-invocation@1"
                or row.get("purpose") != "coding" or row.get("phase") != "coding"
                or row.get("status") not in {"failed", "provider_returned"}):
            issues.append("invalid_coding_receipt")
            continue
        attempt = row.get("owner_worker_attempt_id")
        if type(attempt) is not str or re.fullmatch("[0-9a-f]{32}", attempt) is None:
            issues.append("unbound_coding_receipt")
            continue
        coding.setdefault(attempt, []).append(row)
    return coding, issues


def _attempt(attempt_id, events, receipts):
    issues = []
    prepared = events.get("prepared")
    spawned = events.get("spawned")
    retired = events.get("retired")
    item = dict(attempt_id_sha256=_sha(attempt_id.encode()),
        prepared_sha256=None if prepared is None else prepared[1],
        spawned_sha256=None if spawned is None else spawned[1],
        retired_sha256=None if retired is None else retired[1],
        spawned=spawned is not None, explicit_no_spawn=False,
        retired=False, router_receipted=False, invocation_id_sha256=None,
        native_prompt_sha256=None, workspace_identity_sha256=None)
    if prepared is None:
        issues.append("missing_prepared_event")
    else:
        base, pin = prepared
        item.update(native_prompt_sha256=base["native_prompt_sha256"],
            workspace_identity_sha256=base["workspace_identity_sha256"])
        for value in (spawned, retired):
            if value is not None and (any(type(value[0][key]) is not type(base[key])
                    or value[0][key] != base[key] for key in _COMMON - {"event"})
                    or value[0]["prepared_sha256"] != pin):
                issues.append("event_binding_mismatch")
    if retired is None:
        issues.append("missing_retired_event")
    else:
        final = retired[0]
        if spawned is None:
            if final["spawned_sha256"] is None:
                item["explicit_no_spawn"] = True
            else:
                issues.append("missing_spawned_event")
        elif (final["spawned_sha256"] != spawned[1]
                or any(type(final[key]) is not type(spawned[0][key])
                    or final[key] != spawned[0][key] for key in ("child_pid", "child_start_ticks"))):
            issues.append("child_binding_mismatch")
        if (not final["cleanup_succeeded"] or not final["brokers_closed"]
                or spawned is not None and not final["waited"]):
            issues.append("child_retirement_unverified")
        else:
            item["retired"] = True
    if spawned is None and not item["explicit_no_spawn"]:
        issues.append("child_birth_unknown")
    if item["explicit_no_spawn"]:
        if receipts:
            issues.append("receipt_without_child_birth")
    elif len(receipts) != 1:
        issues.append("missing_router_receipt" if not receipts else "ambiguous_router_receipts")
    elif prepared is not None:
        row = receipts[0]
        workspace = row.get("workspace")
        if (row.get("native_prompt_sha256") != prepared[0]["native_prompt_sha256"]
                or type(workspace) is not str or not 0 < len(workspace.encode("utf-8")) <= 4096
                or _sha(workspace.encode("utf-8")) != prepared[0]["workspace_identity_sha256"]):
            issues.append("router_scope_mismatch")
        else:
            item["router_receipted"] = True
            item["invocation_id_sha256"] = _sha(row["invocation_id"].encode("utf-8"))
    item["all_attempts_receipted"] = not issues
    item["issues"] = sorted(set(issues))
    return item


def collect(*, parent, report):
    """Observe every prepared/started coding child, including failed launches.

    Only an explicit expected census selects directory reads. An explicit
    no-spawn retirement contributes no provider call. Counts describe observed
    worker events; they never establish native usage completeness.
    """
    if type(report) is not dict:
        return _base(reason="invalid_expected_census")
    if "owner_worker_attempt_census_expected" not in report or report["owner_worker_attempt_census_expected"] is False:
        return _base("not_selected", "census_not_selected", selected=False)
    if report["owner_worker_attempt_census_expected"] is not True:
        return _base(reason="invalid_expected_census")
    try:
        groups, issues = _events(parent)
        receipts, receipt_issues = _receipts(report)
        issues.extend(receipt_issues)
        if not groups:
            issues.append("expected_census_missing")
        if any(attempt not in groups for attempt in receipts):
            issues.append("router_without_owner_events")
        items = [_attempt(attempt, events, receipts.get(attempt, []))
            for attempt, events in sorted(groups.items())]
        for item in items:
            issues.extend(item["issues"])
        previous = report.get("unreceipted_provider_attempt")
        if previous is not None and previous is not False:
            issues.append("previous_unreceipted_attempt")
        complete = not issues
        result = _base("observed" if complete else "incomplete",
            "matched_all_owner_attempts" if complete else "attempt_evidence_incomplete")
        result.update(all_attempts_receipted=complete, attempts=items, issues=sorted(set(issues)),
            counts=dict(observed_attempts=len(groups),
                prepared_attempts=sum("prepared" in events for events in groups.values()),
                spawned_attempts=sum(item["spawned"] for item in items),
                retired_attempts=sum(item["retired"] for item in items),
                explicit_no_spawn_attempts=sum(item["explicit_no_spawn"] for item in items),
                matched_router_receipts=sum(item["router_receipted"] for item in items),
                unresolved_attempts=sum(not item["all_attempts_receipted"] for item in items)))
        return result
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, RecursionError, OverflowError, UnicodeError):
        return _base()


__all__ = ["collect"]
