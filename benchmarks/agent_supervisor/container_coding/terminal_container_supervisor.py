"""Execute one admitted Terminal-Bench task inside the deployed container.

Called as the nonroot supervisor owner. All model and candidate-code execution
goes through separately deployed worker-identity launchers. The original Harbor
verifier runs afterwards and remains outside this process's indexed context.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shlex
import signal
import stat
import subprocess
import time
import uuid

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as preparation
from benchmarks.agent_supervisor.container_coding.benchmark_resource_profile import (
    PROFILES, admission_environment, execution_budget, native_start_timeout_ms, planner_timeout_seconds,
    coding_timeout_seconds, implementation_watchdog_seconds,
)
from benchmarks.agent_supervisor.container_coding.terminal_native_progress import NativeProgress
from benchmarks.agent_supervisor.container_coding.terminal_shutdown_observation import record as _record_shutdown_failure
from ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission import verify_local_benchmark_admission

ROOT = Path("/opt/ipfs-supervisor")
ROUTER = ROOT / "bin/router-worker"
VALIDATOR = ROOT / "bin/validation-worker"
WORKTREES = ROOT / "worktrees"


def _write(path: Path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _bounded_repair_bytes(path: Path, limit: int) -> bytes:
    """Read one canonical regular local receipt without following links."""
    if path.resolve() != path.absolute():
        raise ValueError("canonical local repair evidence required")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
            raise ValueError("bounded regular repair evidence required")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    identity = lambda value: (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    current = path.lstat()
    if (not stat.S_ISREG(current.st_mode) or len(raw) > limit
            or identity(before) != identity(after) or identity(after) != identity(current)):
        raise ValueError("repair evidence changed while reading")
    return raw


def _repair_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate repair evidence key")
            value[key] = item
        return value

    def constant(_value):
        raise ValueError("nonfinite repair evidence")

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    if type(value) is not dict:
        raise ValueError("repair evidence object required")
    return value


def _empty_start_cleanup_observation(reason):
    return dict(schema="terminal-start-cleanup-observation@1", status="unavailable", reason=reason,
        proof_observation="unavailable", control_observation="unavailable", lifecycle_phase=None,
        control_phase=None, marker_bound_process_tree_absent=None, start_succeeded=None,
        absence_scope="recorded_marker_bound_tree", completion_authority=False,
        retry_authority=False, execution_authority=False)


def _start_cleanup_observation(runtime, start):
    """Observe exact START repair receipts; never authorize cleanup or work."""
    result = _empty_start_cleanup_observation("evidence_unavailable")
    try:
        from ipfs_accelerate_py.agent_supervisor.control.control_contracts import Operation, OperationRequest
        from ipfs_accelerate_py.agent_supervisor.control.control_plane import MutationTransactionState, MutationTransactionPhase
        from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import LifecycleAction, LifecycleSagaPhase, _SagaState
        if type(start) is not dict or start.get("status") not in {"failed", "conflict", "denied"}:
            return _empty_start_cleanup_observation("no_failed_start")
        request = runtime._requests.get(start.get("request_id"))
        if (type(request) is not OperationRequest or request.operation is not Operation.START
                or request.dry_run or request.authorization is None):
            return _empty_start_cleanup_observation("original_start_unavailable")
        expected = MutationTransactionState.prepare(request, now_ms=0)
        try:
            proof = _repair_json(_bounded_repair_bytes(runtime.state / "start-cleanup-process-proof-receipt.json", 65536))
            fields = {"schema", "transition_id", "request_id", "transaction_id", "phase",
                      "process_tree_absent", "start_succeeded", "completion_authority"}
            if (set(proof) != fields or proof["schema"] != "interrupted-start-cleanup-repair@1"
                    or proof["request_id"] != request.request_id or proof["transaction_id"] != expected.transaction_id
                    or proof["phase"] != "failed" or proof["process_tree_absent"] is not True
                    or proof["start_succeeded"] is not False or proof["completion_authority"] is not False):
                raise ValueError("repair proof differs from original failed START")
            # Bind the proof to the native journal's exact START transition,
            # including its original permit. Never use the latest STOP row.
            lines = _bounded_repair_bytes(runtime.orchestrator.store.path, 4 * 1024 * 1024).splitlines()
            if len(lines) > 512 or any(len(line) > 262144 for line in lines):
                raise ValueError("bounded native lifecycle journal required")
            states = [_SagaState.from_dict(_repair_json(line)) for line in lines]
            matching = [row for row in states if row.intent.request_id == request.request_id]
            if not matching:
                raise ValueError("original START transition unavailable")
            state = max(matching, key=lambda row: row.revision)
            intent = state.intent
            if (any(row != state for row in matching if row.revision == state.revision)
                    or state.phase is not LifecycleSagaPhase.FAILED or state.receipt is not None
                    or state.new_tree is None or state.old_tree is not None or state.old_tree_fenced
                    or state.failure_code != "interrupted_start_cleanup_fenced"
                    or intent.action is not LifecycleAction.START or intent.transition_id != proof["transition_id"]
                    or intent.authorization_decision_id != request.authorization.decision_id
                    or intent.idempotency_key != request.idempotency.key
                    or intent.expected_effect_ids != tuple(effect.effect_id for effect in request.expected_effects)
                    or any(getattr(intent, key) != getattr(request, key) for key in (
                        "repository_root", "state_root", "repository_id", "tree_id", "objective_id",
                        "objective_revision", "policy_id", "policy_revision", "caller", "lease_id", "fencing_epoch"))
                    or any(getattr(intent, key) != getattr(runtime.profile, key) for key in (
                        "target_id", "profile_id", "run_root", "run_id", "configuration_root"))):
                raise ValueError("repair proof differs from native START transition")
            result.update(proof_observation="observed", lifecycle_phase="failed",
                          marker_bound_process_tree_absent=True, start_succeeded=False)
        except FileNotFoundError:
            result["proof_observation"] = "missing"
        except Exception:
            result["proof_observation"] = "invalid"
        try:
            raw = _repair_json(_bounded_repair_bytes(runtime.state / "start-cleanup-repair-receipt.json", 65536))
            if set(raw) != set(expected.to_dict()) or type(raw.get("contract_version")) is not int:
                raise ValueError("exact typed control repair receipt required")
            control = MutationTransactionState.from_dict(raw)
            if (control.phase is not MutationTransactionPhase.REPAIRED
                    or control.transaction_id != expected.transaction_id or control.request_id != request.request_id
                    or (control.result is not None and control.result.succeeded)):
                raise ValueError("control repair differs from original failed START")
            result.update(control_observation="observed", control_phase="repaired")
        except FileNotFoundError:
            result["control_observation"] = "missing"
        except Exception:
            result["control_observation"] = "invalid"
        observed = sum(result[key] == "observed" for key in ("proof_observation", "control_observation"))
        result.update(status="available" if observed == 2 else "partial" if observed else "unavailable",
                      reason="bound_receipts" if observed == 2 else "partial_evidence" if observed else "evidence_unavailable")
        return result
    except Exception:
        return _empty_start_cleanup_observation("collection_unavailable")


def _published_retrieval_options(*, repository, bundle, task, model_snapshot):
    """Select a successor lane from the authenticated retrieval population."""
    if bundle is None:
        return dict(published_retrieval_policy=None, published_learned_artifacts=None)
    from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import load_task_context_nomination
    from ipfs_accelerate_py.agent_supervisor.runtime.published_task_context import (
        _previous_retrieval, EMPTY_RETRIEVAL_POLICY,
    )
    metadata = load_task_context_nomination(repository=repository, artifact=bundle["artifact"],
        expected_sha256=bundle["sha256"], task_cid=task.task_cid, task_id=task.task_key)
    previous = _previous_retrieval(repository, metadata, task.task_key)
    if previous is None or previous[0]["status"] != "current":
        raise ValueError("publication retrieval policy requires authenticated current initial retrieval")
    if previous[1] is None:
        return dict(published_retrieval_policy=EMPTY_RETRIEVAL_POLICY, published_learned_artifacts=None)
    if model_snapshot is None:
        return dict(published_retrieval_policy="lexical-tfidf-symbols@1", published_learned_artifacts=None)
    return dict(published_retrieval_policy="local-safetensors-symbols@1", published_learned_artifacts={
        "result": ".runtime/terminal-vectors/result.json",
        "manifest": ".runtime/terminal-vectors/model-manifest.json",
        "model_snapshot": str(model_snapshot.resolve(strict=True)),
    })


def _empty_retrieval_activity(value):
    """Recognize the native absence observation separately from learned receipts."""
    fields = {"schema", "status", "source_population_cid", "previous_source_population_cid",
        "embedding_calls", "model_loading_calls", "execution_authority", "completion_authority"}
    if (type(value) is not dict or set(value) != fields
            or value["schema"] != "supervisor-published-empty-retrieval-observation@1"
            or type(value["status"]) is not str or value["status"] not in {"current", "unavailable"}
            or any(type(value[key]) is not int or value[key] != 0
                for key in ("embedding_calls", "model_loading_calls"))
            or any(value[key] is not False for key in ("execution_authority", "completion_authority"))
            or type(value["previous_source_population_cid"]) is not str
            or re.fullmatch(r"baguqeera[a-z2-7]{52}", value["previous_source_population_cid"]) is None
            or (value["status"] == "unavailable" and value["source_population_cid"] is not None)
            or (value["status"] == "current" and (type(value["source_population_cid"]) is not str
                or re.fullmatch(r"baguqeera[a-z2-7]{52}", value["source_population_cid"]) is None))):
        return None
    return dict(local_embedding_calls=0, local_embedding_texts=0,
        remote_embedding_calls=0, text_generation_calls=0)


def _arm_cleanup_deadline(deadline: float) -> None:
    """Replace the work alarm with the bounded cleanup window.

    STOP can begin just before the work deadline or while unwinding that
    deadline's exception. In either case the work alarm must not cut short
    the separately reserved shutdown budget. Leave two seconds for the final
    result before the caller's total deadline.
    """
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic() - 2))


BOOTSTRAP_FAILURE_PHASES = frozenset({
    "validation", "peer", "request", "process_tree", "duplicate_birth", "lease", "grant", "response"})
BOOTSTRAP_FAILURE_REASONS = frozenset({
    "validation_failed", "peer_mismatch", "request_mismatch", "process_tree_unavailable",
    "process_root_missing", "process_root_ambiguous", "process_child_missing",
    "process_child_ambiguous", "process_child_is_root", "process_parent_mismatch",
    "process_scope_mismatch", "duplicate_birth", "lease_unavailable", "lease_mismatch",
    "lease_expired", "grant_unavailable", "grant_lifetime_exceeded", "response_failed", "unknown"})


def _project_native_startup(value):
    """Keep startup timing diagnostic-only and exclude runtime/source bodies."""
    fields = {"schema", "start_timeout_ms", "stop_timeout_ms", "bootstrap_wait_seconds",
        "observations", "observations_truncated", "bootstrap_receipt_count", "bootstrap_error_count"}
    if (type(value) is not dict or set(value) not in (fields, fields | {"bootstrap_failure_counts"})
            or type(value["schema"]) is not str or value["schema"] != "admitted-native-startup-observation@1"):
        raise ValueError("closed native startup observation required")
    if (type(value["start_timeout_ms"]) is not int or not 2000 <= value["start_timeout_ms"] <= 120000
            or type(value["stop_timeout_ms"]) is not int or not 2000 <= value["stop_timeout_ms"] <= 30000
            or type(value["bootstrap_wait_seconds"]) not in (int, float)
            or not math.isfinite(value["bootstrap_wait_seconds"]) or not 2 <= value["bootstrap_wait_seconds"] <= 120
            or type(value["observations_truncated"]) is not bool
            or any(type(value[key]) is not int or not 0 <= value[key] <= 65535
                for key in ("bootstrap_receipt_count", "bootstrap_error_count"))
            or type(value["observations"]) is not list or len(value["observations"]) > 16):
        raise ValueError("bounded native startup observation required")
    observations = []
    for row in value["observations"]:
        if (type(row) is not dict or set(row) != {"phase", "status", "seconds"}
                or type(row["phase"]) is not str or row["phase"] not in {
                    "control_validation", "launch_validation", "bootstrap_validation"}
                or type(row["status"]) is not str or row["status"] not in {"running", "completed", "failed"}
                or type(row["seconds"]) not in (int, float) or not math.isfinite(row["seconds"])
                or not 0 <= row["seconds"] <= 900):
            raise ValueError("bounded startup phase observation required")
        observations.append(dict(row))
    result = {**value, "observations": observations}
    if "bootstrap_failure_counts" in value:
        failures = value["bootstrap_failure_counts"]
        if (type(failures) is not list
                or len(failures) > len(BOOTSTRAP_FAILURE_PHASES) * len(BOOTSTRAP_FAILURE_REASONS)
                or any(type(row) is not dict or set(row) != {"phase", "reason", "count"}
                       or type(row["phase"]) is not str or row["phase"] not in BOOTSTRAP_FAILURE_PHASES
                       or type(row["reason"]) is not str or row["reason"] not in BOOTSTRAP_FAILURE_REASONS
                       or type(row["count"]) is not int or not 1 <= row["count"] <= 65535
                       for row in failures)):
            raise ValueError("closed bounded native bootstrap failures required")
        keys = [(row["phase"], row["reason"]) for row in failures]
        if keys != sorted(set(keys)):
            raise ValueError("native bootstrap failures must be sorted unique buckets")
        result["bootstrap_failure_counts"] = [dict(row) for row in failures]
    return result


def _router_reply(stdout: str) -> tuple[str, dict]:
    lines = stdout.splitlines()
    matches = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and value.get("schema") == "router-implementation-invocation@1":
            matches.append((index, value))
    if len(matches) != 1:
        raise ValueError("worker did not return exactly one router invocation receipt")
    index, receipt = matches[0]
    if not receipt.get("external_container_boundary"):
        raise ValueError("worker receipt lacks the qualified container identity")
    return "\n".join(lines[index + 1:]).strip(), receipt


def _native_diagnostics(state: Path) -> dict:
    """Retain bounded startup evidence, without prompts or owner credentials."""
    result = {"schema": "native-supervisor-diagnostics@1"}
    logs = []
    paths = [state / "supervisor-process.log", *sorted(
        (state / "run").glob("admitted_managed_daemon*.log"))[-8:]]
    for path in paths:
        if not path.is_file() or path.is_symlink():
            continue
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 65536))
            raw = stream.read(65536)
        text = raw.decode(errors="replace")
        logs.append(dict(path=path.relative_to(state).as_posix(), log_bytes=path.stat().st_size,
                      tail_sha256=hashlib.sha256(raw).hexdigest(),
                      git_dubious_ownership="detected dubious ownership" in text,
                      exception_types=re.findall(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)):.*$", text, re.M)[-8:],
                      traceback_frames=[{"file": name, "line": int(line), "function": function}
                          for name, line, function in re.findall(
                              r'^\s*File "([^"\n]{1,512})", line ([0-9]{1,8}), in ([A-Za-z0-9_<>.]{1,100})', text, re.M)[-12:]]))
    if logs:
        result.update(logs=logs, git_dubious_ownership=any(row["git_dubious_ownership"] for row in logs),
                      exception_types=list(dict.fromkeys(kind for row in logs for kind in row["exception_types"]))[-8:],
                      traceback_frames=[frame for row in logs for frame in row["traceback_frames"]][-12:])
    for name in ("admitted_supervisor_status.json", "admitted_database_daemon_pass_heartbeat.json",
                 "admitted_native_owner_heartbeat.json"):
        path = state / "run" / name
        if path.is_file() and not path.is_symlink() and path.stat().st_size <= 65536:
            try:
                value = json.loads(path.read_text())
                result[name] = {key: value[key] for key in (
                    "schema", "status", "healthy", "pid", "daemon_pid", "supervisor_pid",
                    "selection_idle_reason", "error_type", "phase",
                ) if key in value and type(value[key]) in (str, int, bool, type(None))}
            except (OSError, ValueError, TypeError):
                result[name] = {"readable_json": False}
    return result


def _failure_diagnostics(error: Exception, *, phase: str) -> dict:
    """Observe a failure without exporting source, locals or exception chains.

    These observations grant no authority and never change admission. Resource
    values are a new sample at error handling, not the earlier lease decision.
    A bounded traceback walk avoids source/linecache reads during unwinding.
    """
    result = {"error_phase": phase}
    try:
        frames = deque(maxlen=20)
        current = error.__traceback__
        walked = 0
        while current is not None and walked < 256:
            code = current.tb_frame.f_code
            frames.append({"file": code.co_filename[:512],
                           "function": code.co_name[:128], "line": current.tb_lineno})
            current = current.tb_next
            walked += 1
        result["error_traceback"] = {"frames": list(frames),
            "frames_walked": walked, "frames_omitted": walked - len(frames),
            "walk_truncated": current is not None}
    except Exception as diagnostic_error:
        result["failure_traceback_error"] = type(diagnostic_error).__name__[:128]
    try:
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import collect_proof_host_resources
        from benchmarks.agent_supervisor.container_coding.terminal_resource_diagnostics import project_failure_resources
        result["failure_resources"] = project_failure_resources(collect_proof_host_resources())
    except Exception as diagnostic_error:
        result["failure_resource_error"] = type(diagnostic_error).__name__[:128]
    try:
        from benchmarks.agent_supervisor.container_coding.terminal_resource_diagnostics import collect_failure_scheduler
        result["failure_scheduler"] = collect_failure_scheduler()
    except Exception:
        result["failure_scheduler_error"] = "collection_unavailable"
    try:
        from benchmarks.agent_supervisor.container_coding.terminal_resource_diagnostics import collect_failure_admission
        primary = error
        seen = set()
        for _ in range(8):
            if not isinstance(primary, BaseException) or id(primary) in seen:
                break
            seen.add(id(primary))
            observation = collect_failure_admission(primary)
            result["failure_admission"] = observation
            if observation.get("reason") != "no_native_admission_error":
                break
            primary = primary.__cause__
    except Exception:
        result["failure_admission_error"] = "collection_unavailable"
    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.header_intent_applicability import project_header_checker_failure
        diagnostic = project_header_checker_failure(error)
        if diagnostic is not None:
            result["failure_header_checker"] = diagnostic
    except Exception:
        result["failure_header_checker_error"] = "collection_unavailable"
    return result


def _final_context_audit(report: dict, *, state: Path, deadline: float) -> None:
    """Observe completed dispatch inputs after shutdown within remaining time."""
    if report.get("arm") != "full":
        return
    try:
        from benchmarks.agent_supervisor.container_coding.terminal_context_audit import collect_terminal_context_audit
        report["context_input_audit"] = collect_terminal_context_audit(
            state=state, receipts=report.get("provider_invocations", []), workspace_root=WORKTREES,
            timeout_seconds=max(0, min(3.0, deadline - time.monotonic() - 1.0)))
    except Exception as error:
        # Observation failure must never suppress provider accounting or the
        # final report. STOP and worker cleanup have already been attempted.
        report["context_input_audit"] = {"schema": "terminal-final-context-audit@1",
            "status": "unknown", "reason": "audit_finalization_unavailable",
            "error_type": type(error).__name__, "provider_calls": 0,
            "raw_prompts_exported": False, "completion_authority": False}


class _PostStopRefreshExpired(BaseException):
    """A deadline must escape optional-component unavailability handlers."""


def _mark_initial_autoencoder_historical(report: dict) -> None:
    """Initial learned nominations do not become current after publication.

    Preserve historical training provenance even when the successor context
    has no remaining rebuild budget. Revalidating or retraining against the
    published source requires its own receipt; semantic refresh alone does not
    establish freshness of this separate learned index.
    """
    learner = report.get("initial_context", {}).get("codebase_autoencoder")
    if learner is None:
        return
    report["post_publication_autoencoder"] = {
        "schema": "terminal-published-code-autoencoder-observation@1",
        "status": "historical", "reason": "initial_checkpoint_not_revalidated_after_publication",
        "checkpoint_sha256": learner["checkpoint_sha256"],
        "receipt_sha256": learner["receipt_sha256"],
        "initial_source_hashes": dict(learner["source_hashes"]),
        "freshness_checked": False, "current_source_reuse_authority": False,
        "new_autoencoder_training_steps": 0, "provider_calls": 0,
        "formalization_authority": False, "proof_authority": False,
        "completion_authority": False,
    }


def _refresh_completed_context(runtime, report: dict, *, deadline: float,
                               work_deadline: float | None = None) -> None:
    """Spend only remaining post-STOP work time; retain finalization reserve."""
    if (report.get("arm") != "full" or report.get("task_state", {}).get("status") != "completed"
            or report.get("stop", {}).get("status") != "succeeded"
            or report.get("remaining_processes") != 0):
        return
    _mark_initial_autoencoder_historical(report)
    frozen = report.get("initial_context", {}).get("security_autoencoder_advice")
    if frozen is not None:
        report["post_publication_security_advice"] = {
            "status": "deferred", "reason": "published_source_not_revalidated",
            "checkpoint_sha256": frozen["checkpoint"]["checkpoint_sha256"],
            "initial_source_hashes": frozen["source_hashes"], "current_source_reuse_authority": False,
            "training_steps": 0, "provider_calls": 0, "download_calls": 0,
            "proof_authority": False, "completion_authority": False}
    refresh_deadline = deadline - 15.
    if work_deadline is not None:
        refresh_deadline = min(refresh_deadline, work_deadline)
    budget = max(0., refresh_deadline - time.monotonic())
    observation = {"status": "deferred", "budget_seconds": budget, "refresh_seconds": None,
        "completion_authority": False, "embedding_calls": None}
    report["post_publication_context"] = observation
    if budget < 5:
        observation["reason"] = "post_stop_refresh_budget_unavailable"
        return
    started = time.monotonic()
    previous_handler = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise _PostStopRefreshExpired("post-STOP refresh budget expired")
    try:
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, budget)
        frozen = report.get("initial_context", {}).get("security_autoencoder_advice")
        if frozen is not None:
            from ipfs_accelerate_py.agent_supervisor.runtime.security_autoencoder_advisor import refresh_security_advice
            report["post_publication_security_advice"] = refresh_security_advice(
                repository=Path(frozen["repository"]), previous=frozen,
                output=Path(frozen["output"]).parent / "published-security-advice")
        doctor = report.get("doctor_dispatch", {})
        workflow = doctor.get("contract_workflow")
        if workflow is not None:
            from ipfs_accelerate_py.agent_supervisor.runtime.doctor_contract_refresh import refresh_doctor_contract_index
            report["post_publication_proof_index"] = refresh_doctor_contract_index(
                repository=Path(workflow["repository"]), workflow=workflow,
                state=Path(doctor["result_artifact"]).parent / "published-contract-index")
        results = runtime.refresh_after_stop()
        observation["results"] = results
        counters = ("local_embedding_calls", "local_embedding_texts",
                    "remote_embedding_calls", "text_generation_calls")
        receipts = [row.get("embedding_receipt") for row in results]
        valid = [receipt for receipt in receipts if isinstance(receipt, dict)
            and receipt.get("schema") == "supervisor-published-learned-embedding-receipt@1"
            and all(type(receipt.get(key)) is int and receipt[key] >= 0 for key in counters)]
        valid.extend(activity for row in results if row.get("embedding_receipt") is None
            and (activity := _empty_retrieval_activity(row.get("empty_retrieval_observation"))) is not None)
        complete = bool(receipts) and len(valid) == len(receipts)
        observation["embedding_accounting"] = {
            "scope": "retrieval_refresh",
            "all_refreshes_receipted": complete,
            "totals": {key: sum(row[key] for row in valid) if complete else None for key in counters},
            "known_subtotals": {key: sum(row[key] for row in valid) if valid else None for key in counters}}
        if complete:
            observation["embedding_calls"] = sum(
                row["local_embedding_calls"] + row["remote_embedding_calls"] for row in valid)
        observation["status"] = ("refreshed" if results and all(
            row.get("status") == "refreshed" and row.get("retrieval_status") == "current"
            for row in results) else "incomplete")
    except (_PostStopRefreshExpired, Exception) as error:
        observation.update(status="unavailable", error_type=type(error).__name__)
    finally:
        observation["refresh_seconds"] = time.monotonic() - started
        report["phases"]["post_publication_refresh_seconds"] = observation["refresh_seconds"]
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        _arm_cleanup_deadline(deadline)


def _security_learning_inputs(*, security_initializer: Path | None,
                              canonical_cve_export: Path | None,
                              canonical_cve_manifest_sha256: str | None) -> dict:
    """Load independently pinned portable bindings within the agent budget."""
    if bool(canonical_cve_export) != bool(canonical_cve_manifest_sha256):
        raise ValueError("canonical CVE export and manifest SHA256 must be supplied together")
    weight_transfer = None
    if security_initializer is not None:
        from ipfs_datasets_py.logic.formalization.autoencoder.security.codebase_autoencoder_transfer import _read, validate_legal_shared_weight_fork

        weight_transfer = json.loads(_read(Path(security_initializer).absolute(), 32_000))
        validate_legal_shared_weight_fork(expected_receipt=weight_transfer, replay_source=False)
    canonical_cve_training = None
    if canonical_cve_export is not None:
        from ipfs_datasets_py.logic.formalization.autoencoder.security.security_cve_canonical_export import load_canonical_cve_training

        root = Path(canonical_cve_export).absolute()
        load_canonical_cve_training(root, expected_manifest_sha256=canonical_cve_manifest_sha256)
        canonical_cve_training = {"output": str(root), "manifest_sha256": canonical_cve_manifest_sha256}
    return {"weight_transfer": weight_transfer, "canonical_cve_training": canonical_cve_training}


def _security_runtime_inputs(*, security_checkpoint, security_checkpoint_manifest_sha256,
                             security_initializer, canonical_cve_export, canonical_cve_manifest_sha256,
                             security_checkpoint_hub_descriptor=None, formula_decoder_descriptor=None,
                             header_protocol_descriptor=None):
    frozen = security_checkpoint is not None or security_checkpoint_manifest_sha256 is not None
    if frozen:
        if (not security_checkpoint or not security_checkpoint_manifest_sha256
                or any(value is not None for value in (security_initializer, canonical_cve_export, canonical_cve_manifest_sha256))):
            raise ValueError("frozen checkpoint requires its manifest pin and excludes local training")
        from ipfs_datasets_py.logic.formalization.autoencoder.security.security_autoencoder_checkpoint import load_security_checkpoint
        loaded = load_security_checkpoint(Path(security_checkpoint).absolute(),
            expected_manifest_sha256=security_checkpoint_manifest_sha256)
        selected = {"train_autoencoder": False, "security_checkpoint": loaded["descriptor"]}
        if formula_decoder_descriptor is not None:
            from ipfs_accelerate_py.agent_supervisor.runtime.security_autoencoder_advisor import _read
            from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formula_decoder import load_security_formula_decoder
            decoder = json.loads(_read(Path(formula_decoder_descriptor).absolute(), 32_000))
            load_security_formula_decoder(decoder)
            selected["formula_decoder"] = decoder
            if header_protocol_descriptor is not None:
                protocol = json.loads(_read(Path(header_protocol_descriptor).absolute(), 32_000))
                from ipfs_datasets_py.logic.security_ir.doctor_header_contracts import WsgiHeaderProtocolContract
                if type(protocol) is not dict or set(protocol) != {"review_ref", "callback_parameter"}:
                    raise ValueError("explicit closed reviewed header protocol required")
                WsgiHeaderProtocolContract(**protocol)
                selected["header_protocol"] = protocol
        elif header_protocol_descriptor is not None:
            raise ValueError("header protocol requires a selected formula decoder")
        if security_checkpoint_hub_descriptor is not None:
            from ipfs_accelerate_py.agent_supervisor.runtime.security_autoencoder_hub import validate_hub_descriptor
            from ipfs_accelerate_py.agent_supervisor.runtime.security_autoencoder_advisor import _read
            hub = validate_hub_descriptor(json.loads(_read(Path(security_checkpoint_hub_descriptor).absolute(), 32_000)))
            if hub["manifest_sha256"] != security_checkpoint_manifest_sha256:
                raise ValueError("selected security Hub manifest differs")
            selected["security_checkpoint_hub"] = hub
        return selected
    if any(item is not None for item in (security_checkpoint_hub_descriptor, formula_decoder_descriptor, header_protocol_descriptor)):
        raise ValueError("security Hub and formula models require an explicit frozen checkpoint")
    return {"train_autoencoder": True, **_security_learning_inputs(
        security_initializer=security_initializer, canonical_cve_export=canonical_cve_export,
        canonical_cve_manifest_sha256=canonical_cve_manifest_sha256)}


def run(*, instruction: Path, state: Path, arm: str, timeout_seconds=None,
        provider_profile: str | None = None,
        semantic_transport_schema: str = "supervisor-semantic-router-input@1",
        coding_reply_mode: str = "legacy",
        resource_profile: str | None = None,
        model_snapshot: Path | None = None, model_revision="",
        security_initializer: Path | None = None, canonical_cve_export: Path | None = None,
        canonical_cve_manifest_sha256: str | None = None,
        security_checkpoint: Path | None = None, security_checkpoint_manifest_sha256: str | None = None,
        security_checkpoint_hub_descriptor: Path | None = None,
        formula_decoder_descriptor: Path | None = None, header_protocol_descriptor: Path | None = None,
        intent_checkpoint_descriptor: Path | None = None,
        intent_action_384_config: Path | None = None,
        source384_config: Path | None = None,
        intent_projection_request: Path | None = None,
        intent_projection_request_sha256: str | None = None,
        disable_intent_autoencoder: bool = False,
        intent_requirement_contract: Path | None = None,
        task_profile: Path | None = None) -> dict:
    from .benchmark_provider_profile import resolve_provider_profile
    from .terminal_semantic_transport_policy import validate_semantic_transport_schema, semantic_transport_selection
    validate_semantic_transport_schema(semantic_transport_schema, arm=arm)
    provider_selection = resolve_provider_profile(provider_profile)
    from .terminal_coding_reply_policy import validate_coding_reply_mode, coding_reply_selection
    validate_coding_reply_mode(coding_reply_mode, arm=arm, provider=provider_selection["provider"])
    budget = execution_budget(resource_profile)
    if timeout_seconds is None:
        timeout_seconds = budget["driver_seconds"]
    if (arm not in {"full", "no-index"} or type(timeout_seconds) is not int
            or not 90 <= timeout_seconds <= max(300, budget["driver_seconds"])):
        raise ValueError("explicit bounded arm required")
    reserved_cleanup_seconds = budget["cleanup_seconds"]
    selected_admission = admission_environment(resource_profile)
    if (any(os.environ.get(key) != value for key, value in selected_admission.items())
            or (not selected_admission and os.environ.get("IPFS_DATASETS_PROOF_RESOURCE_PROFILE", ""))):
        raise ValueError("declared benchmark admission profile differs from the selected environment")
    if arm != "full" and any(value is not None for value in (
            security_initializer, canonical_cve_export, canonical_cve_manifest_sha256,
            security_checkpoint, security_checkpoint_manifest_sha256, security_checkpoint_hub_descriptor,
            formula_decoder_descriptor, header_protocol_descriptor, source384_config)):
        raise ValueError("security training assets require the full indexed arm")
    if source384_config is not None and any(value is not None for value in (
            security_initializer, canonical_cve_export, canonical_cve_manifest_sha256,
            security_checkpoint, security_checkpoint_manifest_sha256, security_checkpoint_hub_descriptor,
            formula_decoder_descriptor, header_protocol_descriptor)):
        raise ValueError("Source384 pinned-parent and legacy security profiles are mutually exclusive")
    generic_profile = None
    if task_profile is not None:
        from .terminal_task_profile import validate_task_profile
        if (task_profile.is_symlink() or not task_profile.is_file()
                or task_profile.resolve(strict=True) != task_profile or task_profile.stat().st_size > 65536):
            raise ValueError("bounded canonical public task profile required")
        generic_profile = validate_task_profile(json.loads(task_profile.read_text()),
            instruction=instruction.read_text())
        if any(value is not None for value in (
                security_initializer, canonical_cve_export, canonical_cve_manifest_sha256)):
            raise ValueError("generic benchmark tasks do not authorize training on task inputs")
        if security_checkpoint is None and any(value is not None for value in (
                security_checkpoint_manifest_sha256, security_checkpoint_hub_descriptor,
                formula_decoder_descriptor, header_protocol_descriptor)):
            raise ValueError("generic legacy decoder assets require an explicit frozen checkpoint")
    if os.geteuid() != 1000 or not state.is_relative_to(ROOT / "state"):
        raise ValueError("container task must run as the deployed private supervisor owner")
    # Canonical files remain read-only to the model identity. The trusted
    # launcher makes only its allocated candidate worktree group-writable.
    os.umask(0o022)
    started = time.monotonic()
    deadline = started + timeout_seconds
    work_deadline = deadline - reserved_cleanup_seconds
    report = {"schema": "terminal-admitted-supervisor-run@1", "arm": arm,
              "task_completed": False, "official_reward": None,
              "start_cleanup": _empty_start_cleanup_observation("runtime_not_created"),
              "provider_profile": provider_selection,
              **semantic_transport_selection(semantic_transport_schema),
              **coding_reply_selection(coding_reply_mode),
              "max_total_agent_seconds": timeout_seconds, "provider_invocations": [],
              "reserved_cleanup_seconds": reserved_cleanup_seconds,
              "work_cutoff_seconds": timeout_seconds - reserved_cleanup_seconds,
              "resource_profile": resource_profile,
              "proof_resource_profile": selected_admission.get("IPFS_DATASETS_PROOF_RESOURCE_PROFILE"),
              "source384_timeout_seconds": budget["source384_seconds"],
              "native_start_timeout_seconds": budget["native_start_seconds"],
              "production_activation": False, "benchmark_advantage_claimed": False,
              "phases": {}, "remaining_processes": None}
    native_state = None
    progress = NativeProgress(started=started)
    report["native_progress"] = progress.report
    state.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    report_path = state.parent / (state.name + "-result.json")

    def budget_expired(signum, frame):
        # Reserve time for native STOP and durable accounting before Harbor's
        # independent outer timeout destroys the disposable container.
        signal.setitimer(signal.ITIMER_REAL, 0)
        raise TimeoutError("the total benchmark agent budget is exhausted")

    previous_alarm = signal.signal(signal.SIGALRM, budget_expired)
    previous_term = signal.signal(signal.SIGTERM, budget_expired)
    signal.setitimer(signal.ITIMER_REAL, max(1, timeout_seconds - reserved_cleanup_seconds))

    def remaining(reserve=0):
        value = int(work_deadline - time.monotonic() - reserve)
        if value < 1:
            raise TimeoutError("the total benchmark agent budget is exhausted")
        return value

    def isolated_planner(prompt, *, repository, provider, model, reasoning_effort,
                         timeout, max_new_tokens, trace_path):
        if (provider, model, reasoning_effort) != tuple(provider_selection[key] for key in
                ("provider", "model", "reasoning_effort")):
            raise ValueError("benchmark planner route differs from selected provider profile")
        planner_tree = WORKTREES / ("planner-" + uuid.uuid4().hex)
        subprocess.run(["git", "-C", str(repository), "-c", "core.hooksPath=/dev/null",
                        "worktree", "add", "--detach", str(planner_tree), "HEAD"],
                       check=True, capture_output=True, timeout=min(30, remaining(15)))
        argv = [str(ROUTER), "--model", model, "--reasoning-effort", reasoning_effort,
                "--purpose", "planning",
                "--timeout", str(min(timeout, remaining(20))),
                "--max-output-tokens", str(max_new_tokens)]
        if provider_selection["provider"] != "codex_cli":
            argv += ["--provider", provider_selection["provider"]]
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, cwd=planner_tree)
        try:
            stdout, stderr = process.communicate(prompt, timeout=min(timeout + 10, remaining(10)))
        except BaseException:
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            trace_path.write_text(stdout)
            trace_path.with_suffix(".stderr").write_text(stderr)
            try:
                _, receipt = _router_reply(stdout)
                report["provider_invocations"].append({"phase": "planning", **receipt})
            except ValueError:
                report["unreceipted_provider_attempt"] = {"phase": "planning", "usage": None,
                                                           "reason": "interrupted_without_receipt"}
            raise
        call = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
        trace_path.write_text(call.stdout)
        trace_path.with_suffix(".stderr").write_text(call.stderr)
        text, receipt = _router_reply(call.stdout)
        report["provider_invocations"].append({"phase": "planning", **receipt})
        if call.returncode or receipt.get("status") != "provider_returned":
            raise RuntimeError("isolated planning router did not complete successfully")
        return {"text": text, "observation": receipt.get("usage", {}), "execution_receipt": receipt}

    phase = "prepare"
    replay_scope = ExitStack()
    try:
        if selected_admission.get("IPFS_DATASETS_PROOF_RESOURCE_PROFILE") == "local-benchmark@1":
            from ipfs_accelerate_py.agent_supervisor.runtime.header_intent_applicability import local_benchmark_applicability_budget
            replay_scope.enter_context(local_benchmark_applicability_budget(deadline_monotonic=work_deadline))
        before = time.monotonic()
        try:
            prepared = preparation.prepare(repository=Path("/app"), instruction=instruction, state=state,
                intent_checkpoint_descriptor=intent_checkpoint_descriptor,
                intent_action_384_config=intent_action_384_config,
                intent_projection_request=intent_projection_request,
                intent_projection_request_sha256=intent_projection_request_sha256,
                disable_intent_autoencoder=disable_intent_autoencoder,
                intent_requirement_contract=intent_requirement_contract,
                **({"provider_profile": provider_profile} if provider_profile is not None else {}),
                **({"task_profile": generic_profile} if generic_profile is not None else {}),
                **({"resource_profile": resource_profile} if resource_profile is not None else {}))
            report["intent_preplanning"] = prepared["intent_preplanning"]
        finally:
            report["phases"]["prepare_seconds"] = time.monotonic() - before
        if arm == "full":
            phase = "initial_context"
            before = time.monotonic()
            try:
                report["initial_context"] = preparation.initial_context(state=state,
                    model_snapshot=model_snapshot, model_revision=model_revision,
                    **({"source384_config": source384_config, "train_autoencoder": False,
                        "source384_timeout_seconds": min(budget["source384_seconds"], remaining())}
                       if source384_config is not None else {"train_autoencoder": False}
                       if generic_profile is not None and security_checkpoint is None
                       else _security_runtime_inputs(security_checkpoint=security_checkpoint,
                        security_checkpoint_manifest_sha256=security_checkpoint_manifest_sha256,
                        security_checkpoint_hub_descriptor=security_checkpoint_hub_descriptor,
                        formula_decoder_descriptor=formula_decoder_descriptor,
                        header_protocol_descriptor=header_protocol_descriptor,
                        security_initializer=security_initializer,
                        canonical_cve_export=canonical_cve_export,
                        canonical_cve_manifest_sha256=canonical_cve_manifest_sha256)))
            finally:
                report["phases"]["initial_context_seconds"] = time.monotonic() - before
        phase = "planning"
        before = time.monotonic()
        try:
            planned = preparation.plan(state=state, provider_callable=isolated_planner,
                                       timeout_seconds=min(planner_timeout_seconds(resource_profile), remaining(30)))
        finally:
            report["phases"]["planning_seconds"] = time.monotonic() - before
        report["planning"] = planned
        if not planned.get("qualified"):
            raise RuntimeError("planning did not produce a qualified, independently admitted plan")
        bundle = None
        if arm == "full":
            phase = "context"
            before = time.monotonic()
            try:
                context = preparation.context(state=state, model_snapshot=model_snapshot,
                                              model_revision=model_revision)
            finally:
                report["phases"]["context_seconds"] = time.monotonic() - before
            report["context"] = context
            bundle = context["context_bundle"]
        phase = "admission"
        admission = json.loads((state / "admission.json").read_text())
        verified = verify_local_benchmark_admission(admission, initial=True)
        task = verified["graph"].tasks[0]
        doctor = None
        if arm == "full":
            phase = "doctor"
            before = time.monotonic()
            try:
                # Keep execution-only imports out of the numerical indexing
                # lifetime. Their loading remains inside the work deadline and
                # the phase that needs them, including on import failure.
                from benchmarks.agent_supervisor.container_coding.terminal_doctor_dispatch import prepare_terminal_doctor_dispatch
                doctor = prepare_terminal_doctor_dispatch(repository=Path("/app"), state=state,
                    admission=admission, task_cid=task.task_cid,
                    contract_profile="wsgi-header-controls@1" if generic_profile is None else None)
            finally:
                report["phases"]["doctor_seconds"] = time.monotonic() - before
            report["doctor_dispatch"] = doctor
        phase = "implementation_setup"
        from benchmarks.agent_supervisor.container_coding.terminal_doctor_dispatch import implementation_argv
        report["implementation_route"] = doctor["route"] if doctor is not None else "model_router"
        # Candidate routes carry no provider timeout. Check the same work
        # deadline without charging them the model route's unused reserve.
        provider_free_candidate = report["implementation_route"] in {"doctor_candidate", "doctor_contract_candidate"}
        implementation_budget = (remaining() if provider_free_candidate else
                                 min(coding_timeout_seconds(resource_profile), remaining(25)))
        report["provider_coding_timeout_cap_seconds"] = (
            None if provider_free_candidate else coding_timeout_seconds(resource_profile))
        report["provider_coding_timeout_seconds"] = None if provider_free_candidate else implementation_budget
        implementation = implementation_argv(router=ROUTER, model=provider_selection["model"],
            reasoning=provider_selection["reasoning_effort"],
            timeout=implementation_budget,
            semantic_repository=Path("/app") if bundle is not None else None, doctor=doctor,
            semantic_transport_schema=semantic_transport_schema,
            coding_reply_mode=coding_reply_mode)
        if not provider_free_candidate and provider_selection["provider"] != "codex_cli":
            implementation += ["--provider", provider_selection["provider"]]
        if report["implementation_route"] == "model_router":
            from ipfs_accelerate_py.agent_supervisor.runtime.router_public_instruction import prepare_public_instruction_context
            instruction_context = prepare_public_instruction_context(repository=Path("/app"),
                admission=admission, task_cid=task.task_cid, source_path=preparation.INSTRUCTION,
                expected_source_sha256=verified["manifest"]["sources"][preparation.INSTRUCTION]["sha256"])
            report["public_instruction"] = instruction_context
            implementation += ["--public-instruction-artifact", instruction_context["artifact"],
                "--public-instruction-sha256", instruction_context["sha256"],
                "--public-instruction-task-cid", task.task_cid]
        command = shlex.join(implementation)
        phase = "native_execution"
        from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
        from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
        from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
        start_timeout = native_start_timeout_ms(resource_profile, remaining_work_seconds=remaining())
        report["native_start_timeout_ms"] = 20_000 if start_timeout is None else start_timeout
        report["native_stop_timeout_ms"] = 20_000
        # Bind the daemon's implementation watchdog explicitly. Its ordinary
        # 1800s default otherwise outlives this benchmark's work window. The
        # outer work alarm remains the absolute deadline even during START.
        implementation_timeout = (remaining() if provider_free_candidate
                                  else min(implementation_watchdog_seconds(resource_profile), remaining(25)))
        report["implementation_timeout_seconds"] = implementation_timeout
        with open_existing_native_owner(
            database=state / "intent.duckdb", checkout=Path("/app"), state_dir=state / "owner",
            repository_id=verified["manifest"]["repository_cid"],
            execution_routes={task.task_key: GROK_CODEX_EXECUTION_MODE},
        ) as owner:
            runtime = AdmittedBenchmarkRuntime.create(
                state / "launch", admission=admission, server=owner.server, source=owner.source,
                implement=True, implementation_command=command, context_bundle=bundle,
                max_task_attempts=1, timeout_ms=20_000,
                implementation_timeout_seconds=implementation_timeout,
                **({"start_timeout_ms": start_timeout} if start_timeout is not None else {}),
                lifetime_seconds=min(900, max(120, remaining() + max(60, reserved_cleanup_seconds))),
                worker_worktree_root=WORKTREES, candidate_runner_argv=(str(VALIDATOR),),
                refresh_context_on_completion=bundle is not None,
                refresh_source384_on_completion=(source384_config is not None
                    and selected_admission.get("IPFS_DATASETS_PROOF_RESOURCE_PROFILE") == "local-benchmark@1"),
                **_published_retrieval_options(repository=Path("/app"), bundle=bundle, task=task,
                    model_snapshot=model_snapshot),
            )
            native_state = runtime.state
            try:
                report["coding_dispatch_possible"] = True
                report["start"] = runtime.start().to_dict()
                if report["start"]["status"] != "succeeded":
                    raise RuntimeError("native supervisor START failed")
                try:
                    startup = _project_native_startup(runtime.startup_diagnostics())
                except Exception:
                    startup = None
                progress.begin_post_start(startup=startup)
                while time.monotonic() < work_deadline - 1:
                    task_state = owner.source.get_task(task.task_cid)
                    report["task_state"] = {"task_cid": task.task_cid, "status": task_state.status,
                                             "revision": task_state.revision}
                    try:
                        startup = _project_native_startup(runtime.startup_diagnostics())
                    except Exception:
                        startup = None
                    try:
                        progress_stop = progress.sample(task=task_state, task_cid=task.task_cid,
                            state=runtime.state, now=time.monotonic(), startup=startup)
                    except Exception:
                        # Diagnostics cannot suppress native cleanup or turn a
                        # reader failure into permission to settle/retry work.
                        progress_stop = None
                        report["native_progress_error"] = "observation_unavailable"
                    if task_state.status in {"completed", "failed", "blocked", "cancelled"}:
                        break
                    if progress_stop:
                        if progress_stop == "expired_attempt_settlement_unavailable":
                            report["unavailable_settlement"] = True
                        break
                    time.sleep(.5)
                report["observation"] = runtime.observe()
            finally:
                _arm_cleanup_deadline(deadline)
                try:
                    report["native_startup"] = _project_native_startup(runtime.startup_diagnostics())
                except Exception:
                    report["native_startup_error"] = "collection_unavailable"
                shutdown_phase = "stop_request"
                try:
                    stop_response = runtime.stop()
                    shutdown_phase = "stop_response_serialization"
                    report["stop"] = stop_response.to_dict()
                    shutdown_phase = "post_stop_process_observation"
                    report["remaining_processes"] = len(runtime.process.snapshot(runtime.profile).members)
                    shutdown_phase = "post_stop_context_refresh"
                    _refresh_completed_context(runtime, report, deadline=deadline, work_deadline=work_deadline)
                except BaseException as error:
                    _record_shutdown_failure(report, error, phase=shutdown_phase)
                    raise
                finally:
                    try:
                        report["start_cleanup"] = _start_cleanup_observation(runtime, report.get("start"))
                        report["native_diagnostics"] = _native_diagnostics(runtime.state)
                    except Exception as error:
                        report["native_diagnostics_error"] = type(error).__name__
                    finally:
                        # STOP's tracked tree can be empty while the runtime
                        # still holds a live launched child. Keep the separate
                        # custody-close observation and preserve its failure.
                        report["runtime_close"] = {"attempted": True, "succeeded": False}
                        try:
                            runtime.close()
                        except BaseException as error:
                            _record_shutdown_failure(report, error, phase="runtime_close")
                            kind = type(error).__name__
                            report["runtime_close"]["error_type"] = kind if kind in {
                                "RuntimeError", "TimeoutError", "ValueError", "OSError",
                                "PermissionError", "ProcessLookupError", "InterruptedError",
                                "KeyboardInterrupt", "SystemExit",
                            } else "other"
                            raise
                        else:
                            report["runtime_close"]["succeeded"] = True
        report["task_completed"] = (
            report.get("task_state", {}).get("status") == "completed"
            and report["stop"]["status"] == "succeeded" and report["remaining_processes"] == 0
        )
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)[:2048]}
        report["error_phase"] = phase
        try:
            report.update(_failure_diagnostics(error, phase=phase))
        except Exception as diagnostic_error:
            # Optional observation must never suppress the primary failure or
            # prevent the existing cleanup and durable result publication.
            report["failure_diagnostics_error"] = type(diagnostic_error).__name__[:128]
    finally:
        replay_scope.close()
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        signal.signal(signal.SIGTERM, previous_term)
        # This deployment owns exactly one worker UID in one disposable
        # container. Reap detached provider children even after failed startup.
        try:
            cleanup = subprocess.run(["sudo", "-n", "-u", "benchmarkworker", "--",
                                      str(ROOT / "bin/worker-entry"), "--cleanup"],
                                     cwd="/", capture_output=True,
                                     timeout=max(.1, min(15, deadline - time.monotonic() - 1)))
            report["worker_cleanup_returncode"] = cleanup.returncode
        except Exception as error:
            report["worker_cleanup_error"] = type(error).__name__
        if report.get("worker_cleanup_returncode") != 0:
            report["task_completed"] = False
        seen = {item["invocation_id"] for item in report["provider_invocations"]}
        for path in state.rglob("*.log"):
            if path.stat().st_size > 8_000_000:
                continue
            for line in path.read_text(errors="replace").splitlines():
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if (isinstance(item, dict) and item.get("schema") == "router-implementation-invocation@1"
                        and item.get("invocation_id") not in seen):
                    report["provider_invocations"].append({"phase": "coding", **item})
                    seen.add(item["invocation_id"])
                elif (isinstance(item, dict) and item.get("schema") in {
                        "native-doctor-candidate-materialization@1", "native-doctor-contract-candidate-materialization@1"}
                      and item.get("task_cid") == report.get("doctor_dispatch", {}).get("task_cid")):
                    if item not in report.setdefault("doctor_invocations", []):
                        report["doctor_invocations"].append(item)
        if (report.get("coding_dispatch_possible") and report.get("implementation_route") == "model_router"
                and not any(
            row.get("phase") == "coding" for row in report["provider_invocations"]
        )):
            report["unreceipted_provider_attempt"] = {
                "phase": "coding", "usage": None,
                "reason": "implementation_may_have_dispatched_without_a_final_router_receipt",
            }
        _final_context_audit(report, state=state, deadline=deadline)
        try:
            progress.finish(report)
        except Exception:
            report["native_progress_error"] = "final_observation_unavailable"
        try:
            from .terminal_failure_observation import collect
            report["native_failure_observations"] = collect(state, report, native_state=native_state)
        except Exception:
            # A diagnostic cannot replace the primary outcome or prevent
            # durable result publication after native/worker cleanup.
            report["native_failure_observations"] = {
                "schema": "terminal-native-failure-observations@1", "status": "unavailable",
                "observation_only": True, "completion_authority": False,
                "retry_authority": False, "settlement_authority": False,
            }
        report["seconds"] = time.monotonic() - started
        _write(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--arm", choices=["full", "no-index"], required=True)
    parser.add_argument("--timeout-seconds", type=int)
    from .benchmark_provider_profile import PROVIDER_PROFILES
    parser.add_argument("--provider-profile", choices=PROVIDER_PROFILES)
    from .terminal_semantic_transport_policy import DEFAULT_SEMANTIC_TRANSPORT_SCHEMA, SEMANTIC_TRANSPORT_SCHEMAS
    parser.add_argument("--semantic-transport-schema", choices=SEMANTIC_TRANSPORT_SCHEMAS,
        default=DEFAULT_SEMANTIC_TRANSPORT_SCHEMA)
    from .terminal_coding_reply_policy import DEFAULT_CODING_REPLY_MODE, CODING_REPLY_MODES
    parser.add_argument("--coding-reply-mode", choices=CODING_REPLY_MODES,
        default=DEFAULT_CODING_REPLY_MODE)
    parser.add_argument("--resource-profile", choices=PROFILES)
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--security-initializer", type=Path)
    parser.add_argument("--canonical-cve-export", type=Path)
    parser.add_argument("--canonical-cve-manifest-sha256")
    parser.add_argument("--security-checkpoint", type=Path)
    parser.add_argument("--security-checkpoint-manifest-sha256")
    parser.add_argument("--security-checkpoint-hub-descriptor", type=Path)
    parser.add_argument("--formula-decoder-descriptor", type=Path)
    parser.add_argument("--header-protocol-descriptor", type=Path)
    parser.add_argument("--intent-checkpoint-descriptor", type=Path)
    parser.add_argument("--intent-action-384-config", type=Path)
    parser.add_argument("--source384-config", type=Path)
    parser.add_argument("--intent-projection-request", type=Path)
    parser.add_argument("--intent-projection-request-sha256")
    parser.add_argument("--disable-intent-autoencoder", action="store_true")
    parser.add_argument("--intent-requirement-contract", type=Path)
    parser.add_argument("--task-profile", type=Path)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ("task_completed", "arm", "seconds", "provider_invocations")}))
    return 0 if result["task_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
