"""Closed shutdown diagnostics; no process, completion or retry authority."""
from __future__ import annotations

from pathlib import Path

SCHEMA = "terminal-shutdown-failure-observation@1"
PHASES = frozenset({"stop_request", "stop_response_serialization",
    "post_stop_process_observation", "post_stop_context_refresh", "runtime_close"})
TYPES = frozenset({"RuntimeError", "ValueError", "TypeError", "OSError",
    "PermissionError", "FileNotFoundError", "ProcessLookupError", "InterruptedError",
    "TimeoutError", "TimeoutExpired", "ProcessIdentityMismatch", "ProcessTreeNotFenced",
    "TransactionConflictError", "StaleLeaseError", "StaleTreeError", "PartialMutationError",
    "LifecycleOrchestrationError", "KeyboardInterrupt", "SystemExit", "CancelledError", "other"})
REASONS = frozenset({"live_launched_children", "recorded_tree_nonempty", "timeout",
    "cancellation", "other", "observation_unavailable"})
_ROOT = Path(__file__).resolve().parents[3]
_RELATIVE_FILES = (
    "benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py",
    "ipfs_accelerate_py/agent_supervisor/entrypoints/isolated_benchmark_runtime.py",
    "ipfs_accelerate_py/agent_supervisor/entrypoints/admitted_benchmark_runtime.py",
    "ipfs_accelerate_py/agent_supervisor/control/lifecycle_orchestrator.py",
    "ipfs_accelerate_py/agent_supervisor/control/control_plane.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/process_tree_fencing.py",
)
_FRAME_PATHS = {str(_ROOT/name):Path(name).name for name in _RELATIVE_FILES}
FRAME_NAMES = frozenset(_FRAME_PATHS.values())
_AUTHORITY_FIELDS = ("completion_authority", "retry_authority", "settlement_authority")


def unavailable(phase):
    if phase not in PHASES:
        raise ValueError("closed shutdown phase required")
    return {"schema":SCHEMA, "status":"unavailable", "phase":phase,
        "reason_code":"observation_unavailable", "exceptions":[], "chain_truncated":False,
        "observation_only":True, **{key:False for key in _AUTHORITY_FIELDS}}


def observe(error, *, phase):
    """Inspect exception types and exact installed source frames, never messages."""
    result = unavailable(phase)
    exceptions, seen, remaining = [], set(), 256
    current = error
    truncated = False
    while isinstance(current, BaseException) and id(current) not in seen and len(exceptions) < 4:
        seen.add(id(current))
        name = type(current).__name__
        kind = name if name in TYPES else "other"
        trace, frames = current.__traceback__, []
        while trace is not None and remaining:
            remaining -= 1
            filename = _FRAME_PATHS.get(trace.tb_frame.f_code.co_filename)
            if filename is not None and type(trace.tb_lineno) is int and 0 < trace.tb_lineno <= 10_000_000:
                frames.append({"file":filename,"line":trace.tb_lineno})
                if len(frames) > 8:
                    frames.pop(0)
                    truncated = True
            trace = trace.tb_next
        truncated = truncated or trace is not None
        exceptions.append({"exception_type":kind,"frames":frames})
        current = current.__cause__ if current.__cause__ is not None else (
            None if current.__suppress_context__ else current.__context__)
    truncated = truncated or current is not None
    if not exceptions:
        return result
    reason = "other"
    if exceptions[0]["exception_type"] in {"KeyboardInterrupt","SystemExit","CancelledError"}:
        reason = "cancellation"
    elif exceptions[0]["exception_type"] in {"TimeoutError","TimeoutExpired"}:
        reason = "timeout"
    elif (type(error) is RuntimeError and type(error.args) is tuple and len(error.args) == 1
          and type(error.args[0]) is str and any(frame["file"] in {
              "isolated_benchmark_runtime.py", "admitted_benchmark_runtime.py"}
              for frame in exceptions[0]["frames"])):
        reason = {
            "stop every live launched child before releasing runtime custody":"live_launched_children",
            "stop the exact supervisor tree before closing its coordinator":"recorded_tree_nonempty",
        }.get(error.args[0], "other")
    result.update(status="observed",reason_code=reason,exceptions=exceptions,chain_truncated=truncated)
    return result


def validate(value):
    """Reject extra/unbounded values before any closed report or artifact export."""
    fields = {"schema","status","phase","reason_code","exceptions","chain_truncated",
        "observation_only",*_AUTHORITY_FIELDS}
    if (type(value) is not dict or set(value) != fields or value["schema"] != SCHEMA
            or type(value["status"]) is not str or value["status"] not in {"observed","unavailable"}
            or type(value["phase"]) is not str or value["phase"] not in PHASES
            or type(value["reason_code"]) is not str or value["reason_code"] not in REASONS
            or type(value["chain_truncated"]) is not bool or value["observation_only"] is not True
            or any(value[key] is not False for key in _AUTHORITY_FIELDS)
            or type(value["exceptions"]) is not list or len(value["exceptions"]) > 4):
        return None
    if value["status"] == "unavailable":
        return unavailable(value["phase"]) if (not value["exceptions"] and not value["chain_truncated"]
            and value["reason_code"] == "observation_unavailable") else None
    if not value["exceptions"] or value["reason_code"] == "observation_unavailable":
        return None
    exceptions = []
    for row in value["exceptions"]:
        if (type(row) is not dict or set(row) != {"exception_type","frames"}
                or type(row["exception_type"]) is not str or row["exception_type"] not in TYPES
                or type(row["frames"]) is not list or len(row["frames"]) > 8):
            return None
        frames = []
        for frame in row["frames"]:
            if (type(frame) is not dict or set(frame) != {"file","line"}
                    or type(frame["file"]) is not str or frame["file"] not in FRAME_NAMES
                    or type(frame["line"]) is not int or not 0 < frame["line"] <= 10_000_000):
                return None
            frames.append(dict(frame))
        exceptions.append({"exception_type":row["exception_type"],"frames":frames})
    reason, first = value["reason_code"], exceptions[0]
    if ((reason == "cancellation" and first["exception_type"] not in {"KeyboardInterrupt","SystemExit","CancelledError"})
            or (reason == "timeout" and first["exception_type"] not in {"TimeoutError","TimeoutExpired"})
            or (reason in {"live_launched_children","recorded_tree_nonempty"} and (
                first["exception_type"] != "RuntimeError" or not any(frame["file"] in {
                    "isolated_benchmark_runtime.py","admitted_benchmark_runtime.py"}
                    for frame in first["frames"])) )):
        return None
    return {**value,"exceptions":exceptions}


def record(report, error, *, phase):
    """Optional diagnostics cannot replace a failure or prevent native cleanup."""
    try:
        value = validate(observe(error, phase=phase))
    except Exception:
        value = None
    if value is None:
        value = unavailable(phase)
    slot = "runtime_close" if phase == "runtime_close" else "stop"
    report.setdefault("shutdown_failures", {})[slot] = value
