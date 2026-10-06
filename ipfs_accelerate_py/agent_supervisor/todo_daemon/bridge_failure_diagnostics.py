"""Closed observations of native bridge failures; never settlement authority."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path

SCHEMA = "database-bridge-failure-diagnostic@2"
LEGACY_SCHEMA = "database-bridge-failure-diagnostic@1"
EVENT = "portal_bridge_failure_diagnostic"
LOG_PREFIX = "Native bridge failure diagnostic: "
OBSERVATION_SCHEMA = "database-bridge-failure-observation@1"
OBSERVATION_FILENAME = "bridge-failure-observation.json"
OBSERVATION_MAX_BYTES = 32_768
_TYPES = frozenset({"DatabasePortalBridgeError", "DatabasePortalBridgeDeferred",
    "DatabaseImplementationAuthorityError", "OwnershipError", "FenceMismatchError",
    "WorktreeLifecycleError", "ValueError", "TypeError", "RuntimeError", "TimeoutError",
    "TimeoutExpired", "CalledProcessError", "FileNotFoundError", "PermissionError",
    "OSError", "ImportError", "ModuleNotFoundError", "KeyError", "other"})
_REASONS = frozenset({"portal_provider_failed", "implementation_protected_path_mutated",
    "portal_validation_failed", "portal_result_unavailable", "other"})
_CALLBACK_SCHEMAS = frozenset({"database-portal-callback-intent@1",
    "database-native-provider-failure@1", "database-source384-no-dispatch-retry@1", "other"})
_STATES = frozenset({"started_outcome_unknown", "failed_outcome_settled",
    "not_dispatched", "completed", "other"})
_EFFECTS = frozenset({"unknown_may_have_started", "failed_provider_exited",
    "not_started", "started", "completed", "other"})
_EXIT_BOOLS = ("reaped", "process_group_absent", "subreaper_children_absent", "lifecycle_finalized")
_ROOT = Path(__file__).resolve().parents[1]
_FRAME_PATHS = {str((_ROOT / relative).resolve()): Path(relative).name for relative in (
    "todo_daemon/implementation_daemon.py", "todo_daemon/database_portal_bridge.py",
    "todo_daemon/supervisor_runtime.py", "merge/worktree_lifecycle.py",
    "runtime/router_implementation_runner.py", "runtime/source384_repository_context.py",
    "runtime/local_completion_bridge.py", "entrypoints/admitted_benchmark_runtime.py")}
_FRAME_NAMES = frozenset(_FRAME_PATHS.values())


def _enum(value, allowed):
    return value if type(value) is str and value in allowed else None if value is None else "other"


def _boolean(value):
    return value if type(value) is bool else None


def _returncode(value):
    return value if type(value) is int and -255 <= value <= 255 else None


def missing_child_report(status="missing"):
    return {"status": status, "diagnostic": None, "observation_only": True,
            "scope": "bounded_native_log_tail"}


def validate_child_report(value):
    if (type(value) is not dict or set(value) != {"status", "diagnostic", "observation_only", "scope"}
            or value["observation_only"] is not True
            or value["scope"] != "bounded_native_log_tail" or type(value["status"]) is not str
            or value["status"] not in {"observed", "missing", "ambiguous", "invalid", "unavailable"}):
        return None
    if value["status"] != "observed":
        return dict(value) if value["diagnostic"] is None else None
    # Reuse the runner's closed parser, loaded only for an observed row.
    from ..runtime.router_implementation_runner import validate_runner_error_envelope
    try:
        diagnostic = validate_runner_error_envelope(value["diagnostic"])
    except (ValueError, TypeError, RecursionError):
        return None
    return {**missing_child_report("observed"), "diagnostic": diagnostic}


def child_report_from_log(path, writer):
    """Read at most 64 KiB from the same regular file as the native writer.

    A matching row remains child-reported metadata: a provider could print
    that JSON itself. It grants no evidence of dispatch, exit, or settlement.
    """
    from ..runtime.router_implementation_runner import validate_runner_error_envelope
    descriptor = None
    try:
        writer.flush()
        expected = os.fstat(writer.fileno())
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        actual = os.fstat(descriptor)
        if (not stat.S_ISREG(actual.st_mode) or not stat.S_ISREG(expected.st_mode)
                or (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino)):
            return missing_child_report("unavailable")
        offset = max(0, actual.st_size - 65_536)
        os.lseek(descriptor, offset, os.SEEK_SET)
        raw = os.read(descriptor, 65_536)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
                actual.st_dev, actual.st_ino, actual.st_size, actual.st_mtime_ns):
            return missing_child_report("unavailable")
        rows = raw.splitlines(keepends=True)
        if offset and rows:
            rows = rows[1:]  # Never parse a partial leading row.
        candidates, malformed = [], False
        for row in rows:
            if not row.endswith(b"\n") or len(row) > 16_384:
                malformed = malformed or b"router-implementation-error@2" in row
                continue
            try:
                value = json.loads(row, object_pairs_hook=_unique_keys)
            except (ValueError, RecursionError, UnicodeError):
                malformed = malformed or b"router-implementation-error@2" in row
                continue
            if type(value) is dict and value.get("schema") == "router-implementation-error@2":
                candidates.append(value)
        if len(candidates) > 1 or (malformed and candidates):
            return missing_child_report("ambiguous")
        if malformed:
            return missing_child_report("invalid")
        if not candidates:
            return missing_child_report()
        try:
            projected = validate_runner_error_envelope(candidates[0])
        except (ValueError, TypeError, RecursionError):
            return missing_child_report("invalid")
        return {**missing_child_report("observed"), "diagnostic": projected}
    except (OSError, ValueError, TypeError):
        return missing_child_report("unavailable")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _unique_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate diagnostic key")
        value[key] = item
    return value


def observe_bridge_failure(failure, callback, *, phase):
    if phase not in {"unknown_callback", "terminal_failure"}:
        raise ValueError("closed bridge failure phase required")
    exceptions, seen, remaining = [], set(), 256
    current = failure
    truncated = False
    while isinstance(current, BaseException) and len(exceptions) < 4 and id(current) not in seen:
        seen.add(id(current))
        frames, trace = [], current.__traceback__
        while trace is not None and remaining:
            remaining -= 1
            code = trace.tb_frame.f_code
            filename = _FRAME_PATHS.get(code.co_filename)
            if filename is not None and type(trace.tb_lineno) is int and 0 < trace.tb_lineno <= 10_000_000:
                frames.append({"file": filename, "line": trace.tb_lineno})
                frames = frames[-8:]
            trace = trace.tb_next
        if trace is not None:
            truncated = True
        exceptions.append({"exception_type": _enum(type(current).__name__, _TYPES), "frames": frames})
        current = current.__cause__ or current.__context__
    truncated = truncated or isinstance(current, BaseException)
    callback = callback if type(callback) is dict else {}
    native = callback.get("native_exit")
    native = native if type(native) is dict and native.get("schema") == "database-native-provider-exit@1" else None
    result = getattr(failure, "result", None)
    implementation = result.get("implementation") if type(result) is dict else None
    child = implementation.get("child_reported_router_failure") if type(implementation) is dict else None
    child = validate_child_report(child) if child is not None else None
    from .native_provider_custody_observation import validate_native_provider_custody_observation
    custody = result.get("native_provider_custody_observation") if type(result) is dict else None
    custody = validate_native_provider_custody_observation(custody)
    # Do not execute an exception's custom __str__ or scan an unbounded model
    # message merely to recognize one short, source-owned reason literal.
    arguments = BaseException.args.__get__(failure, BaseException)
    reason = (arguments[0] if len(arguments) == 1 and type(arguments[0]) is str
              and len(arguments[0]) <= 256 else "other")
    return {"schema": SCHEMA, "phase": phase,
        "reason_code": _enum(reason, _REASONS),
        "exceptions": exceptions, "chain_truncated": truncated,
        "callback": {"schema": _enum(callback.get("schema"), _CALLBACK_SCHEMAS),
            "state": _enum(callback.get("callback_state"), _STATES),
            "provider_effect_state": _enum(callback.get("provider_effect_state"), _EFFECTS),
            "native_exit": {"present": native is not None,
                "returncode": _returncode(native.get("returncode")) if native else None,
                **{key: _boolean(native.get(key)) if native else None for key in _EXIT_BOOLS}}},
        "child_reported_router_failure": child or missing_child_report(),
        "native_provider_custody_observation": custody,
        "completion_authority": False, "retry_authority": False, "settlement_authority": False}


def validate_bridge_failure_diagnostic(value):
    """Validate an untrusted event projection; never confer custody authority."""
    try:
        fields = {"schema", "phase", "reason_code", "exceptions",
                "chain_truncated", "callback", "child_reported_router_failure", "completion_authority",
                "retry_authority", "settlement_authority"}
        if type(value) is not dict or value.get("schema") not in {SCHEMA, LEGACY_SCHEMA}:
            return None
        if value["schema"] == SCHEMA:
            fields.add("native_provider_custody_observation")
        if (set(value) != fields or value["phase"] not in {"unknown_callback", "terminal_failure"}
                or value["reason_code"] not in _REASONS or type(value["chain_truncated"]) is not bool
                or any(value[key] is not False for key in ("completion_authority", "retry_authority", "settlement_authority"))):
            return None
        exceptions = value["exceptions"]
        if type(exceptions) is not list or not 1 <= len(exceptions) <= 4:
            return None
        for exception in exceptions:
            if (type(exception) is not dict or set(exception) != {"exception_type", "frames"}
                    or exception["exception_type"] not in _TYPES or type(exception["frames"]) is not list
                    or len(exception["frames"]) > 8):
                return None
            for frame in exception["frames"]:
                if (type(frame) is not dict or set(frame) != {"file", "line"} or frame["file"] not in _FRAME_NAMES
                        or type(frame["line"]) is not int or not 0 < frame["line"] <= 10_000_000):
                    return None
        callback = value["callback"]
        if type(callback) is not dict or set(callback) != {"schema", "state", "provider_effect_state", "native_exit"}:
            return None
        for key, allowed in (("schema", _CALLBACK_SCHEMAS), ("state", _STATES), ("provider_effect_state", _EFFECTS)):
            if callback[key] is not None and callback[key] not in allowed:
                return None
        native = callback["native_exit"]
        if (type(native) is not dict or set(native) != {"present", "returncode", *_EXIT_BOOLS}
                or type(native["present"]) is not bool
                or (native["returncode"] is not None and _returncode(native["returncode"]) is None)
                or any(native[key] is not None and type(native[key]) is not bool for key in _EXIT_BOOLS)
                or (native["present"] is False and any(native[key] is not None for key in ("returncode", *_EXIT_BOOLS)))):
            return None
        child = validate_child_report(value["child_reported_router_failure"])
        if child is None:
            return None
        if value["schema"] == SCHEMA and value["native_provider_custody_observation"] is not None:
            from .native_provider_custody_observation import validate_native_provider_custody_observation
            if validate_native_provider_custody_observation(value["native_provider_custody_observation"]) is None:
                return None
        return json.loads(json.dumps({**value, "child_reported_router_failure": child}, allow_nan=False))
    except (ValueError, TypeError, RecursionError):
        return None


def validate_bridge_failure_observation(value):
    """Validate the optional sidecar independently of database projections."""
    if (type(value) is not dict or set(value) != {"schema", "task_cid_sha256",
            "attempt_id_sha256", "diagnostic", "observation_only"}
            or value["schema"] != OBSERVATION_SCHEMA or value["observation_only"] is not True):
        return None
    for key in ("task_cid_sha256", "attempt_id_sha256"):
        if type(value[key]) is not str or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None:
            return None
    diagnostic = validate_bridge_failure_diagnostic(value["diagnostic"])
    if diagnostic is None:
        return None
    return {**value, "diagnostic": diagnostic}


def write_bridge_failure_observation(state_dir, *, task_cid, attempt_id, diagnostic):
    """Atomically retain a bounded observation, never a state/authority input.

    Explicit native state directories remain usable with every optional JSON
    queue/event projection disabled. No fallback may write into the task tree.
    """
    if state_dir is None:
        return
    if any(type(value) is not str or not value or len(value) > 4096
           for value in (task_cid, attempt_id)):
        raise ValueError("bounded exact attempt and task identities required")
    value = validate_bridge_failure_observation({"schema": OBSERVATION_SCHEMA,
        "task_cid_sha256": hashlib.sha256(task_cid.encode("utf-8")).hexdigest(),
        "attempt_id_sha256": hashlib.sha256(attempt_id.encode("utf-8")).hexdigest(),
        "diagnostic": diagnostic, "observation_only": True})
    if value is None:
        raise ValueError("closed bridge failure observation required")
    payload = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if len(payload) > OBSERVATION_MAX_BYTES:
        raise ValueError("bridge failure observation exceeds its byte bound")
    directory = Path(state_dir)
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".bridge-failure-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / OBSERVATION_FILENAME)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
