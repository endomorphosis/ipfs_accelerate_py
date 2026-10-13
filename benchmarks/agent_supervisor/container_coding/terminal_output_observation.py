"""Observe declared public output bytes after native settlement and worker exit.

The owner task projection selects one private attempt directory. Its native
binding and finished event select the allocation; a router receipt alone cannot
select a path. No source bodies, Git commands, discovery, or verifier reads are
used. Digests describe current bytes, not who wrote them, publication, or reward.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys

from .terminal_context_audit import _open_directory, _json
from .terminal_native_progress import COMPLETION_RECEIPT_OPERATIONS

SCHEMA = "terminal-declared-output-observation@1"
MAX_OUTPUTS = 32
MAX_FILE_BYTES = 1_000_000
MAX_EVENTS_BYTES = 4_000_000
MAX_REQUEST_BYTES = 65_536
MAX_RESULT_BYTES = 65_536


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _base(reason="evidence_unavailable"):
    return dict(schema=SCHEMA, status="unavailable", reason=reason, outputs=[],
        observation_only=True, completion_authority=False, settlement_authority=False,
        retry_authority=False, task_correctness_established=False,
        patch_application_established=False, writer_identity_established=False,
        raw_source_exported=False, provider_calls=0,
        observation_boundary="after_native_settlement_stop_and_worker_cleanup_before_container_removal")


def _text(value):
    if type(value) is not str or not 0 < len(value.encode()) <= 1024:
        raise ValueError("invalid identity")
    return value


def _digest(value):
    if type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("invalid digest")
    return value


def _relative(value):
    from ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission import _path
    value = _path(_text(value))
    if value == "." or len(PurePosixPath(value).parts) > 16:
        raise ValueError("invalid output path")
    return value


def capture_scope(verified, task):
    """Copy ONLY the independently admitted output scope, before execution.

    Callers must not re-admit the checkout after publication: legitimate source
    changes can then differ from the original manifest.
    """
    manifest = verified["manifest"]
    rows = []
    for output in task.outputs:
        path = _relative(output.path)
        if output.effect not in {"create", "modify", "delete"}:
            raise ValueError("unsupported output effect")
        source = manifest["sources"].get(path)
        if source is None and output.effect != "create":
            raise ValueError("missing output baseline")
        rows.append(dict(path=path, effect=output.effect,
            baseline_sha256=None if source is None else _digest(source["sha256"])))
    scope = dict(task_cid=_text(task.task_cid), task_alias=_text(task.task_key),
        baseline_commit=manifest["baseline_commit"], outputs=rows)
    _scope(scope)
    return scope


def _scope(scope):
    if type(scope) is not dict or set(scope) != {"task_cid", "task_alias", "baseline_commit", "outputs"}:
        raise ValueError("missing admitted scope")
    _text(scope["task_cid"]); _text(scope["task_alias"])
    if type(scope["baseline_commit"]) is not str or re.fullmatch("[0-9a-f]{40}", scope["baseline_commit"]) is None:
        raise ValueError("invalid baseline")
    rows = scope["outputs"]
    if type(rows) is not list or not 0 < len(rows) <= MAX_OUTPUTS:
        raise ValueError("output count exceeds bound")
    paths = set()
    for row in rows:
        if type(row) is not dict or set(row) != {"path", "effect", "baseline_sha256"}:
            raise ValueError("invalid output scope")
        path = _relative(row["path"])
        if path in paths or row["effect"] not in {"create", "modify", "delete"}:
            raise ValueError("duplicate or unsupported output")
        paths.add(path)
        if row["baseline_sha256"] is None:
            if row["effect"] != "create":
                raise ValueError("missing baseline bytes")
        else:
            _digest(row["baseline_sha256"])


def _identity(info):
    return tuple(getattr(info, name) for name in (
        "st_dev", "st_ino", "st_mode", "st_uid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns"))


def _read(root_fd, relative, *, limit, private=False, owner_controlled=False):
    """No-follow, single-link read with both FD and named-path identity checks."""
    _relative(relative)
    with ExitStack() as stack:
        parent = os.dup(root_fd); stack.callback(os.close, parent)
        parts = PurePosixPath(relative).parts
        links = []
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            stack.callback(os.close, child)
            links.append((parent, part, os.fstat(child)))
            parent = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        stack.callback(os.close, fd)
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or not 0 <= before.st_size <= limit
                or (private or owner_controlled) and before.st_uid != os.geteuid()
                or private and before.st_mode & 0o077
                or owner_controlled and before.st_mode & 0o022):
            raise ValueError("unsafe file")
        raw = bytearray()
        while len(raw) <= limit:
            block = os.read(fd, min(65536, limit + 1 - len(raw)))
            if not block:
                break
            raw.extend(block)
        after = os.fstat(fd)
        try:
            current = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            raise ValueError("file removed during observation") from None
        if len(raw) != before.st_size or _identity(before) != _identity(after) or _identity(after) != _identity(current):
            raise ValueError("file changed during observation")
        for directory, name, opened in links:
            try:
                current = os.stat(name, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                raise ValueError("directory removed during observation") from None
            if (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_uid) != (
                    current.st_dev, current.st_ino, current.st_mode, current.st_uid):
                raise ValueError("directory changed during observation")
        return bytes(raw)


def _file(root_fd, path, baseline):
    try:
        raw = _read(root_fd, path, limit=MAX_FILE_BYTES)
        digest = _sha(raw)
        return dict(status="observed", exists=True, size_bytes=len(raw), sha256=digest,
            equals_admitted_baseline=False if baseline is None else digest == baseline)
    except FileNotFoundError:
        return dict(status="observed", exists=False, size_bytes=None, sha256=None,
            equals_admitted_baseline=baseline is None)
    except Exception:
        return dict(status="unavailable", exists=None, size_bytes=None, sha256=None,
            equals_admitted_baseline=None)


def _private_directory(path, stack):
    fd = _open_directory(path); stack.callback(os.close, fd)
    info = os.fstat(fd)
    if info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise ValueError("owner-private directory required")
    return fd


def observe(*, native_state, workspace_root, repository, scope, report):
    """Observe a terminal owner-bound allocation; never settle or mutate it."""
    result = _base()
    stage = "admitted_scope"
    try:
        _scope(scope)
        result.update(task_cid_sha256=_sha(scope["task_cid"].encode()),
            baseline_commit=scope["baseline_commit"])
        stage = "settlement_and_cleanup"
        task = report["native_progress"]["latest"]["task"]
        state = report["task_state"]
        completion = task["completion_receipt"]
        operation = completion.get("operation")
        if (task.get("availability") != "observed"
                or any(task.get(key) != state.get(key) for key in ("task_cid", "revision", "status"))
                or state.get("task_cid") != scope["task_cid"]
                or state.get("status") not in {"blocked", "failed", "completed"}
                or type(operation) is not str
                or operation not in COMPLETION_RECEIPT_OPERATIONS | {"database_task_claim_failure"}
                or operation in COMPLETION_RECEIPT_OPERATIONS and state.get("status") != "completed"
                or operation == "database_task_claim_failure" and (
                    state.get("status") not in {"blocked", "failed"}
                    or completion.get("failure_kind") != "terminal_portal_bridge_error"
                    or completion.get("automatic_retry_admitted") is not False
                    or _digest(completion.get("settlement_id_sha256")) is None)
                or report.get("unavailable_settlement")
                or report.get("stop", {}).get("status") != "succeeded"
                or type(report.get("remaining_processes")) is not int or report["remaining_processes"] != 0
                or report.get("runtime_close", {}).get("succeeded") is not True
                or type(report.get("worker_cleanup_returncode")) is not int or report["worker_cleanup_returncode"] != 0):
            return _base("settlement_or_cleanup_unavailable")
        attempt_hash = _digest(completion["attempt_id_sha256"])
        result["attempt_id_sha256"] = attempt_hash
        with ExitStack() as stack:
            stage = "native_directory"
            _private_directory(Path(native_state), stack)
            attempt_path = Path(native_state) / "run/admitted_database_portal_attempts" / attempt_hash[:24]
            stage = "attempt_directory"
            attempt_fd = _private_directory(attempt_path, stack)
            # Reuse the native closed binding verifier through the pinned FD.
            # The extra read enforces privacy, duplicate-key and byte bounds.
            stage = "attempt_binding"
            raw = _read(attempt_fd, "database-attempt-binding.json", limit=65536, private=True)
            binding = _json(raw)
            from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge
            if DatabasePortalExecutionBridge._strict_binding(Path(f"/proc/self/fd/{attempt_fd}/database-attempt-binding.json")) != binding:
                raise ValueError("binding changed")
            if (binding["task_cid"] != scope["task_cid"] or binding["task_alias"] != scope["task_alias"]
                    or any(_sha(_text(binding[key]).encode()) != _digest(completion[key + "_sha256"])
                           for key in ("attempt_id", "claim_id", "lease_id"))
                    or any(type(completion.get(key)) is not int or completion[key] != binding[key]
                           for key in ("fencing_token", "fence_epoch"))):
                raise ValueError("binding differs from native settlement")
            stage = "owner_event_log"
            # append_jsonl_event uses the native process umask, commonly
            # producing a 0644 log. Its verified 0700 owner directory supplies
            # privacy; require owner custody and no other write permission on
            # the file, while retaining the stricter 0600 binding contract.
            events_raw = _read(attempt_fd, "portal-events.jsonl", limit=MAX_EVENTS_BYTES,
                owner_controlled=True)
            lines = events_raw.splitlines()
            if len(lines) > 4096:
                raise ValueError("event count exceeds bound")
            events = [_json(line) for line in lines if line.strip()]
            if any(type(event) is not dict for event in events):
                raise ValueError("invalid event")
            stage = "allocation_event"
            finished = [event for event in events if event.get("type") == "implementation_finished"]
            if len(finished) != 1:
                raise ValueError("ambiguous allocation")
            event = finished[0]
            if (event.get("canonical_task_cid") != scope["task_cid"]
                    or event.get("task_cid") != scope["task_cid"]
                    or event.get("task_id") != scope["task_alias"]
                    or event.get("baseline_ref") != scope["baseline_commit"]
                    or type(event.get("attempt")) is not int or not 1 <= event["attempt"] < 65536
                    or event.get("provider_dispatched") is not True or event.get("attempt_consumed") is not True
                    or event.get("lifecycle_finalize", {}).get("finalized") is not True
                    or type(event.get("returncode")) is not int):
                raise ValueError("allocation event differs")
            workspace = Path(_text(event["worktree_path"]))
            stage = "allocation_root"
            root = Path(workspace_root)
            if not root.is_absolute() or workspace.parent != root or str(workspace) != event["worktree_path"]:
                raise ValueError("allocation outside owned root")
            root_fd = _open_directory(root); stack.callback(os.close, root_fd)
            info = os.fstat(root_fd)
            if info.st_uid != os.geteuid() or info.st_mode & 0o022:
                raise ValueError("worker-writable allocation parent")
            stage = "invocation_join"
            matches = [row for row in report.get("provider_invocations", []) if type(row) is dict
                and row.get("schema") == "router-implementation-invocation@1"
                and row.get("phase") == row.get("purpose") == "coding" and row.get("workspace") == str(workspace)]
            if len(matches) != 1 or matches[0].get("public_instruction", {}).get("task_cid") != scope["task_cid"]:
                raise ValueError("invocation join unavailable")
            result.update(binding_id=binding["binding_id"], owner_events_sha256=_sha(events_raw),
                portal_attempt=event["attempt"], allocation_path_sha256=_sha(str(workspace).encode()),
                invocation_id_sha256=_sha(_text(matches[0]["invocation_id"]).encode()),
                invocation_join="router_receipt_workspace_and_task_match_owner_finished_event",
                allocation_binding="owner_private_native_attempt_binding_and_finished_event",
                original_allocation_inode_verified=False)
            stage = "declared_output_files"
            roots = {}
            for label, path in (("candidate", workspace), ("canonical", Path(repository))):
                try:
                    fd = _open_directory(path); stack.callback(os.close, fd)
                    roots[label] = fd
                except OSError:
                    roots[label] = None
            for row in scope["outputs"]:
                observed = {"path": row["path"], "effect": row["effect"], "baseline_sha256": row["baseline_sha256"]}
                for label, fd in roots.items():
                    observed[label] = (_file(fd, row["path"], row["baseline_sha256"]) if fd is not None else
                        dict(status="unavailable", exists=None, size_bytes=None, sha256=None, equals_admitted_baseline=None))
                result["outputs"].append(observed)
            # Pinning a descriptor is insufficient if the named allocation or
            # canonical root is replaced. Downgrade all facts from that root.
            for label, path in (("candidate", workspace), ("canonical", Path(repository))):
                fd = roots[label]
                if fd is None:
                    continue
                try:
                    current_fd = _open_directory(path); stack.callback(os.close, current_fd)
                    current, opened = os.fstat(current_fd), os.fstat(fd)
                    unchanged = (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino)
                except OSError:
                    unchanged = False
                if not unchanged:
                    for row in result["outputs"]:
                        row[label] = dict(status="unavailable", exists=None, size_bytes=None,
                            sha256=None, equals_admitted_baseline=None)
            complete = all(row[label]["status"] == "observed" for row in result["outputs"] for label in roots)
            result.update(status="observed" if complete else "partial", reason="current_declared_output_bytes_only")
        return result
    except Exception:
        # Never export exception text: it can contain source or private paths.
        result["reason"] = "binding_or_file_evidence_unavailable"
        result["evidence_stage"] = stage
        return result


def collect(*, native_state, workspace_root, repository, scope, report, timeout_seconds=3.0):
    """Bound even hostile filesystems/parsers in a disposable diagnostic child."""
    if native_state is None or scope is None:
        return _base("admitted_scope_or_native_runtime_unavailable")
    if timeout_seconds < .1:
        return _base("observation_budget_unavailable")
    try:
        # Only closed lifecycle projections and the narrow router join cross
        # this pipe; provider text, prompts, and usage bodies never do.
        selected = {key: report.get(key) for key in ("task_state", "native_progress", "stop", "remaining_processes",
            "runtime_close", "worker_cleanup_returncode", "unavailable_settlement")}
        selected["native_progress"] = {"latest": {"task": report.get("native_progress", {}).get("latest", {}).get("task")}}
        selected["stop"] = {"status": report.get("stop", {}).get("status")}
        selected["provider_invocations"] = [{key: row.get(key) for key in
            ("schema", "phase", "purpose", "invocation_id", "workspace")} | {
                "public_instruction": {"task_cid": (row.get("public_instruction") or {}).get("task_cid")}}
            for row in report.get("provider_invocations", []) if type(row) is dict and row.get("phase") == "coding"]
        request = json.dumps(dict(scope=scope, report=selected), allow_nan=False).encode()
        if len(request) > MAX_REQUEST_BYTES:
            return _base("observation_request_exceeds_bound")
        process = subprocess.run([sys.executable, "-B", "-P", "-m", __name__,
            "--native-state", str(native_state), "--workspace-root", str(workspace_root),
            "--repository", str(repository)], input=request, capture_output=True, check=False,
            timeout=min(5., timeout_seconds), env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        if process.returncode or len(process.stdout) > MAX_RESULT_BYTES:
            return _base("observation_process_unavailable")
        result = _json(process.stdout)
        if type(result) is not dict or result.get("schema") != SCHEMA:
            return _base("observation_result_invalid")
        return result
    except subprocess.TimeoutExpired:
        return _base("observation_timeout")
    except Exception:
        return _base("observation_unavailable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-state", "workspace-root", "repository"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("request too large")
        request = _json(raw)
        result = observe(**vars(args), **request)
        encoded = json.dumps(result, allow_nan=False)
        if len(encoded.encode()) > MAX_RESULT_BYTES:
            raise ValueError("result too large")
    except Exception:
        encoded = json.dumps(_base())
    print(encoded)


if __name__ == "__main__":
    main()
