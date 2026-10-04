"""Execute one admitted Terminal-Bench task inside the deployed container.

Called as the nonroot supervisor owner. All model and candidate-code execution
goes through separately deployed worker-identity launchers. The original Harbor
verifier runs afterwards and remains outside this process's indexed context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import signal
import subprocess
import time
import uuid

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as preparation
from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
from benchmarks.agent_supervisor.container_coding.terminal_doctor_dispatch import (
    implementation_argv, prepare_terminal_doctor_dispatch,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission import verify_local_benchmark_admission
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE

ROOT = Path("/opt/ipfs-supervisor")
ROUTER = ROOT / "bin/router-worker"
VALIDATOR = ROOT / "bin/validation-worker"
WORKTREES = ROOT / "worktrees"


def _write(path: Path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _arm_cleanup_deadline(deadline: float) -> None:
    """Replace the work alarm with the bounded cleanup window.

    STOP can begin just before the work deadline or while unwinding that
    deadline's exception. In either case the work alarm must not cut short
    the separately reserved shutdown budget. Leave two seconds for the final
    result before the caller's total deadline.
    """
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic() - 2))


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
        result["failure_admission"] = collect_failure_admission(error)
    except Exception:
        result["failure_admission_error"] = "collection_unavailable"
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


def _refresh_completed_context(runtime, report: dict, *, deadline: float) -> None:
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
    budget = max(0., deadline - time.monotonic() - 15.)
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
        complete = bool(receipts) and len(valid) == len(receipts)
        observation["embedding_accounting"] = {
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


def run(*, instruction: Path, state: Path, arm: str, timeout_seconds=285,
        model_snapshot: Path | None = None, model_revision="",
        security_initializer: Path | None = None, canonical_cve_export: Path | None = None,
        canonical_cve_manifest_sha256: str | None = None,
        security_checkpoint: Path | None = None, security_checkpoint_manifest_sha256: str | None = None,
        security_checkpoint_hub_descriptor: Path | None = None,
        formula_decoder_descriptor: Path | None = None, header_protocol_descriptor: Path | None = None,
        intent_checkpoint_descriptor: Path | None = None,
        intent_projection_request: Path | None = None,
        intent_projection_request_sha256: str | None = None,
        disable_intent_autoencoder: bool = False,
        intent_requirement_contract: Path | None = None) -> dict:
    if arm not in {"full", "no-index"} or not 90 <= timeout_seconds <= 300:
        raise ValueError("explicit bounded arm required")
    if arm != "full" and any(value is not None for value in (
            security_initializer, canonical_cve_export, canonical_cve_manifest_sha256,
            security_checkpoint, security_checkpoint_manifest_sha256, security_checkpoint_hub_descriptor,
            formula_decoder_descriptor, header_protocol_descriptor)):
        raise ValueError("security training assets require the full indexed arm")
    if os.geteuid() != 1000 or not state.is_relative_to(ROOT / "state"):
        raise ValueError("container task must run as the deployed private supervisor owner")
    # Canonical files remain read-only to the model identity. The trusted
    # launcher makes only its allocated candidate worktree group-writable.
    os.umask(0o022)
    started = time.monotonic()
    deadline = started + timeout_seconds
    work_deadline = deadline - 40
    report = {"schema": "terminal-admitted-supervisor-run@1", "arm": arm,
              "task_completed": False, "official_reward": None,
              "max_total_agent_seconds": timeout_seconds, "provider_invocations": [],
              "reserved_cleanup_seconds": 40, "work_cutoff_seconds": timeout_seconds - 40,
              "production_activation": False, "benchmark_advantage_claimed": False,
              "phases": {}, "remaining_processes": None}
    state.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    report_path = state.parent / (state.name + "-result.json")

    def budget_expired(signum, frame):
        # Reserve time for native STOP and durable accounting before Harbor's
        # independent outer timeout destroys the disposable container.
        signal.setitimer(signal.ITIMER_REAL, 0)
        raise TimeoutError("the total benchmark agent budget is exhausted")

    previous_alarm = signal.signal(signal.SIGALRM, budget_expired)
    previous_term = signal.signal(signal.SIGTERM, budget_expired)
    signal.setitimer(signal.ITIMER_REAL, max(1, timeout_seconds - 40))

    def remaining(reserve=0):
        value = int(work_deadline - time.monotonic() - reserve)
        if value < 1:
            raise TimeoutError("the total benchmark agent budget is exhausted")
        return value

    def isolated_planner(prompt, *, repository, provider, model, reasoning_effort,
                         timeout, max_new_tokens, trace_path):
        if provider != "codex_cli":
            raise ValueError("benchmark planner route differs from the native baseline")
        planner_tree = WORKTREES / ("planner-" + uuid.uuid4().hex)
        subprocess.run(["git", "-C", str(repository), "-c", "core.hooksPath=/dev/null",
                        "worktree", "add", "--detach", str(planner_tree), "HEAD"],
                       check=True, capture_output=True, timeout=min(30, remaining(15)))
        argv = [str(ROUTER), "--model", model, "--reasoning-effort", reasoning_effort,
                "--purpose", "planning",
                "--timeout", str(min(timeout, remaining(20))),
                "--max-output-tokens", str(max_new_tokens)]
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

    try:
        before = time.monotonic()
        try:
            prepared = preparation.prepare(repository=Path("/app"), instruction=instruction, state=state,
                intent_checkpoint_descriptor=intent_checkpoint_descriptor,
                intent_projection_request=intent_projection_request,
                intent_projection_request_sha256=intent_projection_request_sha256,
                disable_intent_autoencoder=disable_intent_autoencoder,
                intent_requirement_contract=intent_requirement_contract)
            report["intent_preplanning"] = prepared["intent_preplanning"]
        finally:
            report["phases"]["prepare_seconds"] = time.monotonic() - before
        if arm == "full":
            before = time.monotonic()
            try:
                report["initial_context"] = preparation.initial_context(state=state,
                    model_snapshot=model_snapshot, model_revision=model_revision,
                    **_security_runtime_inputs(security_checkpoint=security_checkpoint,
                        security_checkpoint_manifest_sha256=security_checkpoint_manifest_sha256,
                        security_checkpoint_hub_descriptor=security_checkpoint_hub_descriptor,
                        formula_decoder_descriptor=formula_decoder_descriptor,
                        header_protocol_descriptor=header_protocol_descriptor,
                        security_initializer=security_initializer,
                        canonical_cve_export=canonical_cve_export,
                        canonical_cve_manifest_sha256=canonical_cve_manifest_sha256))
            finally:
                report["phases"]["initial_context_seconds"] = time.monotonic() - before
        before = time.monotonic()
        try:
            planned = preparation.plan(state=state, provider_callable=isolated_planner,
                                       timeout_seconds=min(90, remaining(30)))
        finally:
            report["phases"]["planning_seconds"] = time.monotonic() - before
        report["planning"] = planned
        if not planned.get("qualified"):
            raise RuntimeError("the model proposal did not pass independent admission")
        bundle = None
        if arm == "full":
            before = time.monotonic()
            try:
                context = preparation.context(state=state, model_snapshot=model_snapshot,
                                              model_revision=model_revision)
            finally:
                report["phases"]["context_seconds"] = time.monotonic() - before
            report["context"] = context
            bundle = context["context_bundle"]
        admission = json.loads((state / "admission.json").read_text())
        verified = verify_local_benchmark_admission(admission, initial=True)
        task = verified["graph"].tasks[0]
        doctor = None
        if arm == "full":
            before = time.monotonic()
            try:
                doctor = prepare_terminal_doctor_dispatch(repository=Path("/app"), state=state,
                    admission=admission, task_cid=task.task_cid, contract_profile="wsgi-header-controls@1")
            finally:
                report["phases"]["doctor_seconds"] = time.monotonic() - before
            report["doctor_dispatch"] = doctor
        report["implementation_route"] = doctor["route"] if doctor is not None else "model_router"
        implementation = implementation_argv(router=ROUTER, model=preparation.MODEL,
            reasoning=preparation.REASONING, timeout=remaining(25),
            semantic_repository=Path("/app") if bundle is not None else None, doctor=doctor)
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
        with open_existing_native_owner(
            database=state / "intent.duckdb", checkout=Path("/app"), state_dir=state / "owner",
            repository_id=verified["manifest"]["repository_cid"],
            execution_routes={task.task_key: GROK_CODEX_EXECUTION_MODE},
        ) as owner:
            runtime = AdmittedBenchmarkRuntime.create(
                state / "launch", admission=admission, server=owner.server, source=owner.source,
                implement=True, implementation_command=command, context_bundle=bundle,
                max_task_attempts=1, timeout_ms=20_000, lifetime_seconds=min(600, max(120, remaining() + 60)),
                worker_worktree_root=WORKTREES, candidate_runner_argv=(str(VALIDATOR),),
                refresh_context_on_completion=bundle is not None,
                published_retrieval_policy=("local-safetensors-symbols@1" if model_snapshot is not None
                    else "lexical-tfidf-symbols@1") if bundle is not None else None,
                published_learned_artifacts=({
                    "result": ".runtime/terminal-vectors/result.json",
                    "manifest": ".runtime/terminal-vectors/model-manifest.json",
                    "model_snapshot": str(model_snapshot.resolve(strict=True)),
                } if bundle is not None and model_snapshot is not None else None),
            )
            try:
                report["coding_dispatch_possible"] = True
                report["start"] = runtime.start().to_dict()
                if report["start"]["status"] != "succeeded":
                    raise RuntimeError("native supervisor START failed")
                while time.monotonic() < work_deadline - 1:
                    task_state = owner.source.get_task(task.task_cid)
                    report["task_state"] = {"task_cid": task.task_cid, "status": task_state.status,
                                             "revision": task_state.revision}
                    if task_state.status in {"completed", "failed", "blocked", "cancelled"}:
                        break
                    heartbeat = runtime.state / "run/admitted_database_daemon_pass_heartbeat.json"
                    try:
                        reason = json.loads(heartbeat.read_text()).get("selection_idle_reason")
                    except (OSError, ValueError):
                        reason = None
                    if reason == "expired_attempt_settlement_unavailable":
                        report["unavailable_settlement"] = True
                        break
                    time.sleep(.5)
                report["observation"] = runtime.observe()
            finally:
                _arm_cleanup_deadline(deadline)
                try:
                    report["stop"] = runtime.stop().to_dict()
                    report["remaining_processes"] = len(runtime.process.snapshot(runtime.profile).members)
                    _refresh_completed_context(runtime, report, deadline=deadline)
                finally:
                    try:
                        report["native_diagnostics"] = _native_diagnostics(runtime.state)
                    except Exception as error:
                        report["native_diagnostics_error"] = type(error).__name__
                    finally:
                        runtime.close()
        report["task_completed"] = (
            report.get("task_state", {}).get("status") == "completed"
            and report["stop"]["status"] == "succeeded" and report["remaining_processes"] == 0
        )
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)[:2048]}
    finally:
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
        report["seconds"] = time.monotonic() - started
        _write(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--arm", choices=["full", "no-index"], required=True)
    parser.add_argument("--timeout-seconds", type=int, default=285)
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
    parser.add_argument("--intent-projection-request", type=Path)
    parser.add_argument("--intent-projection-request-sha256")
    parser.add_argument("--disable-intent-autoencoder", action="store_true")
    parser.add_argument("--intent-requirement-contract", type=Path)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ("task_completed", "arm", "seconds", "provider_invocations")}))
    return 0 if result["task_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
