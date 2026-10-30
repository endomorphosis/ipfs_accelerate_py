"""Finite REAL SDK regression; every provider/task fact is explicitly synthetic.

No accepting CLI exists. The actual admitted outer owner calls these functions
in one interpreter, with real SDK objects and frozen requests. This exercises
live native validation -> sealed closeout -> one-use handle -> separate native
orchestration -> isolated FIXTURE board completion. It cannot target a real
canonical LIR board or fabricate an actual provider outcome.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from contextlib import contextmanager

import controller_coordinator as coordinator
import factual_candidate as facts
import native_acceptance as acceptance

PREFIX = "legal-ir-native-acceptance-fixture-"
SCOPE = "synthetic-native-acceptance-regression-only"
LIMITS = {"validation_timeout_seconds": 900, "validation_storage_bytes": 32 * 1024 * 1024,
          "validation_memory_mb": 512, "validation_cpu_slots": 1, "validation_child_process_slots": 4,
          "controller_timeout_seconds": 900, "controller_storage_bytes": 8 * 1024 * 1024,
          "controller_memory_mb": 256, "controller_cpu_slots": 1, "controller_child_process_slots": 1}


def require(condition, reason):
    acceptance.require(condition, reason)


def write_new(path, raw):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def preparation_guard(owner):
    require(type(owner).__module__ == coordinator.PARENTED_MODULE
            and type(owner).__name__ == "ParentedPhaseResourceReservation", "actual_preparation_owner_required")
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.parented_phase_resources import ParentedPhaseResourceReservation
    require(type(owner) is ParentedPhaseResourceReservation, "actual_preparation_owner_required")
    require(owner.heartbeat(), "preparation_native_heartbeat_failed")
    owner.check_usage(owner.directory, full=True)
    lease = owner.native_lease.to_dict()
    require(lease["lane"] == "orchestration" and lease["owner_pid"] == os.getpid(), "preparation_orchestration_identity_required")


def prepare_isolated_fixture(*, preparation_owner, immutable_fixture_root, namespace):
    """Actual precharged fixture writes; never copies a real provider report."""
    preparation_guard(preparation_owner)
    require(type(namespace) is str and namespace.startswith(PREFIX) and len(namespace) <= 128,
            "explicit_synthetic_fixture_namespace_required")
    base = preparation_owner.directory / "synthetic-factual-acceptance-fixture"
    require(not base.exists(), "synthetic_fixture_must_be_fresh")
    base.mkdir(mode=0o700)
    source_root = base / "bootstrap"
    source_root.mkdir(mode=0o700)
    documents, frozen = facts.load_source_bundle(Path(immutable_fixture_root))
    names = dict(frozen)
    for role, name in facts.SOURCE_PATHS.items():
        names[name] = documents[role]
    for name, raw in names.items():
        path = source_root / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if name in {"scheduler_config.json", "tasks.execution.json"}:
            value = facts.strict_json(raw)
            value["board_namespace"] = namespace
            raw = facts.canonical(value)
        write_new(path, raw)
    documents, frozen = facts.load_source_bundle(source_root)
    report_directory, observations = base / "synthetic-worker-output", base / "synthetic-host-observations"
    report_directory.mkdir(mode=0o700)
    observations.mkdir(mode=0o700)
    report = report_directory / "candidate.json"
    write_new(report, facts.canonical(facts.candidate_template(documents, frozen)))
    receipt = observations / "synthetic-provider-outcome.json"
    observation = observations / "synthetic-controller-observation.json"
    write_new(receipt, facts.canonical({"scope": SCOPE, "provider_calls_performed": False,
                                      "synthetic_provider_id": "grok_cli", "synthetic_model_id": "grok-4.7"}))
    write_new(observation, facts.canonical({"scope": SCOPE, "real_LIR_task_observed_or_accepted": False,
                                          "namespace": namespace, "fixture_source_sha256": facts.sha(documents["authoring_tasks"])}))
    context = acceptance.NativeCandidateContext(
        task_cid="fixture:LIR-9001:" + namespace, task_revision=facts.sha(documents["execution_board"]),
        native_attempt_id="fixture-attempt:" + namespace, candidate_commit="f" * 40,
        orchestration_namespace=namespace, source_root=str(source_root), report_path=str(report), report_sha256=facts.sha(report.read_bytes()),
        provider_id="grok_cli", model_id="grok-4.7", reasoning_effort="", provider_trigger="primary",
        provider_execution_id="fixture-no-provider-execution:" + namespace,
        # A distinct synthetic worker identity exercises the separation policy;
        # this is deliberately not evidence that this PID executed any provider.
        provider_worker_pid=os.getpid() + 1,
        provider_receipt_path=str(receipt), provider_receipt_sha256=facts.sha(receipt.read_bytes()),
        controller_observation_path=str(observation), controller_observation_sha256=facts.sha(observation.read_bytes()),
    )
    preparation_guard(preparation_owner)
    return context


def pinned_input_paths(context):
    documents, frozen = facts.load_source_bundle(Path(context.source_root))
    return list(dict.fromkeys([
        *(Path(context.source_root) / name for name in facts.SOURCE_PATHS.values()),
        *(Path(context.source_root) / name for name in frozen),
        Path(context.report_path), Path(context.provider_receipt_path), Path(context.controller_observation_path),
        Path(facts.__file__).absolute(), Path(acceptance.__file__).absolute(), Path(coordinator.__file__).absolute(),
        Path(__file__).absolute(), Path(__file__).with_name("startup_barrier.py").absolute(),
    ]))


def module_pins(*modules):
    return {str(Path(module.__file__).absolute()): facts.sha(facts.read_source_file(Path(module.__file__).absolute()))
            for module in modules}


def validation_command(context, evidence_path):
    require(context.orchestration_namespace.startswith(PREFIX) and context.task_cid.startswith("fixture:"), "synthetic_fixture_identity_required")
    return " ".join(shlex.quote(value) for value in [
        "python3", "-B", str(Path(facts.__file__).absolute()), "--source-root", context.source_root,
        "--check-facts", context.report_path,
    ]) + " > " + shlex.quote(str(evidence_path))


def barrier_argv(runtime, attempt_directory, python_executable):
    from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import hermetic_validation_command
    return [str(python_executable), "-I", "-B", str(Path(__file__).with_name("startup_barrier.py").absolute()),
            str(Path(attempt_directory) / "tmp" / "native-attachment.go"), "--bounds-json",
            facts.canonical(runtime.resource_bounds.to_dict()).decode("utf-8"), "--", *hermetic_validation_command(runtime)]


def stop_actual_child(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


@contextmanager
def opened_validation(owner):
    """At most three native open tries ONLY before any claim or lease exists."""
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_daemon_resources import DaemonResourceError
    for index in range(3):
        try:
            owner.open()
            break
        except DaemonResourceError as error:
            public = owner.to_dict()
            retryable = (str(error) == "disk ledger lock unavailable" and public.get("status") == "not_entered"
                         and public.get("record") is None and public.get("resource_lease") is None)
            if not retryable or index == 2:
                raise
    try:
        yield owner
    except BaseException:
        owner.__exit__(*sys.exc_info())
        raise
    else:
        owner.__exit__(None, None, None)


def exercise_actual_native_roundtrip(*, preparation_owner, validation_owner, runtime, context,
                                    orchestration_attempt_directory, orchestration_ledger_path,
                                    orchestration_roots, handles_only=False):
    """Run a frozen real native fixture attempt and optionally publish fixture status.

The admitted caller prepares the two initially empty sibling attempt directories
and exact SDK validation request. runtime must have been constructed using the
supported build_hermetic_validation_runtime API, the command returned above,
workspace equal to validation_owner.directory, scrubbed environment and bounds.
All input/native/accelerator module pins and this regression source belong in
the exact request. Its argv is barrier_argv(...); selected_checks is [CHECK].
Do not put validation or orchestration attempts beneath preparation_owner's
outstanding attempt, because genuine claim-overlap checks correctly reject it.
    """
    preparation_guard(preparation_owner)
    require(context.orchestration_namespace.startswith(PREFIX) and context.task_cid.startswith("fixture:")
            and context.native_attempt_id.startswith("fixture-attempt:")
            and context.provider_execution_id.startswith("fixture-no-provider-execution:"), "synthetic_fixture_identity_required")
    base = Path(context.source_root)
    require(preparation_owner.directory in base.parents and base.name == "bootstrap", "fixture_source_outside_owned_preparation")
    for path in (Path(context.provider_receipt_path), Path(context.controller_observation_path)):
        require(facts.strict_json(facts.read_source_file(path))["scope"] == SCOPE, "synthetic_fixture_observation_scope_required")
    require(type(validation_owner).__module__ == acceptance.NATIVE_MODULE
            and type(validation_owner).__name__ == "ValidationResourceReservation", "actual_validation_fixture_owner_required")
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.validation_run_resources import ValidationResourceReservation
    from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import HermeticValidationRuntime
    require(type(validation_owner) is ValidationResourceReservation and type(runtime) is HermeticValidationRuntime,
            "actual_validation_runtime_and_owner_required")
    request = validation_owner.request
    require(request["timeout_seconds"] <= LIMITS["validation_timeout_seconds"]
            and request["storage_bytes"] <= LIMITS["validation_storage_bytes"]
            and request["memory_mb"] <= LIMITS["validation_memory_mb"]
            and request["cpu_slots"] == 1 and 1 <= request["child_process_slots"] <= 4,
            "finite_validation_fixture_bounds_required")
    evidence = validation_owner.directory / "exports" / "factual-correspondence.json"
    result = validation_owner.directory / "exports" / "validation-result.json"
    require(runtime.workspace_path == str(validation_owner.directory)
            and runtime.command == validation_command(context, evidence)
            and runtime.timeout_seconds <= 120, "frozen_hermetic_fixture_runtime_differs")
    expected = barrier_argv(runtime, validation_owner.directory, request["argv"][0])
    facts.same(request["argv"], expected, "frozen_fixture_barrier_argv_differs")
    facts.same(request["selected_checks"], [acceptance.CHECK], "frozen_fixture_check_differs")
    for path in pinned_input_paths(context):
        require(request["source_sha256"].get(str(path)) == facts.sha(facts.read_source_file(path)), "fixture_source_not_frozen:" + str(path))
    process = None
    with opened_validation(validation_owner):
        environment = validation_owner.environment(dict(runtime.environment))
        facts.same(environment, dict(runtime.environment), "actual_owned_environment_not_frozen_in_runtime_identity")
        log = validation_owner.directory / "logs" / "actual-hermetic-child.log"
        descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            process = subprocess.Popen(request["argv"], cwd=request["cwd"], env=environment,
                                       stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                validation_owner.attach_child(process.pid)
                require(validation_owner.heartbeat(), "native_fixture_heartbeat_failed_before_barrier")
                write_new(validation_owner.directory / "tmp" / "native-attachment.go", b"G")
                deadline = time.monotonic() + min(runtime.timeout_seconds, 120)
                next_heartbeat = 0
                while process.poll() is None:
                    validation_owner.check_limits(full=False)
                    now = time.monotonic()
                    require(now < deadline, "actual_native_fixture_deadline_exceeded")
                    if now >= next_heartbeat:
                        require(validation_owner.heartbeat(), "actual_native_fixture_heartbeat_lost")
                        next_heartbeat = now + 0.5
                    time.sleep(0.05)
                process.wait(timeout=1)
                require(process.returncode == 0, "actual_hermetic_fixture_child_failed_no_fallback")
                write_new(result, facts.canonical({
                    "schema": "owned-validation-result/v1", "request_sha256": validation_owner.request_sha256,
                    "executed_checks": [acceptance.CHECK], "returncode": 0, "completed": True,
                    "evidence_paths": [str(evidence)],
                }))
                handle = acceptance.issue_native_acceptance(reservation=validation_owner, process=process,
                    context=context, evidence_path=evidence, result_path=result)
            except BaseException:
                stop_actual_child(process)
                raise
    if handles_only:
        sources, frozen = facts.load_source_bundle(base)
        proposed, summary = acceptance.consume_for_manual_completion(handle, current_context=context,
            board_raw=sources["execution_board"], report_raw=facts.read_source_file(Path(context.report_path)),
            source_documents=sources, frozen_files=frozen)
        require(proposed.count(b"- Status: completed\n") == 1 and proposed.count(b"- Status: blocked\n") == 63,
                "fixture_completion_projection_differs")
        try:
            acceptance._entry(handle)
        except acceptance.AcceptanceRefusal:
            pass
        else:
            raise acceptance.AcceptanceRefusal("fixture_handle_was_replayable")
        return {"scope": SCOPE, "actual_native_validation_finalized": True, "actual_handle_consumed_once": True,
                "fixture_board_published": False, "real_LIR_task_accepted": False, "provider_calls_performed": False,
                "native_validation_summary": summary}
    # New real orchestration admission occurs only AFTER native validation
    # finalized. Root/phase ResourceLease objects are never forged or borrowed.
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler import open_existing_scheduler
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.parented_phase_resources import ParentedPhaseResourceReservation
    scheduler = open_existing_scheduler(request["scheduler_state_path"], request["expected_scheduler_policy_sha256"])
    root_request, phase_request = context.orchestration_namespace + ":controller_preparation:root", context.orchestration_namespace + ":controller_preparation:phase"
    root = scheduler.acquire("orchestration", cpu_slots=1, memory_mb=512, child_process_slots=2,
                             requires_gpu=False, timeout=30, request_id=root_request)
    phase = None
    try:
        phase = root.acquire_child(cpu_slots=1, memory_mb=384, child_process_slots=2,
                                   requires_gpu=False, timeout=30, request_id=phase_request)
        owner = ParentedPhaseResourceReservation(orchestration_ledger_path, roots=orchestration_roots,
            attempt_directory=orchestration_attempt_directory, parent_lease=phase,
            storage_bytes=LIMITS["controller_storage_bytes"], memory_mb=LIMITS["controller_memory_mb"],
            cpu_slots=1, child_process_slots=1, scheduler_state_path=request["scheduler_state_path"],
            expected_scheduler_policy_sha256=request["expected_scheduler_policy_sha256"], timeout_seconds=900, ledger_lock_timeout_seconds=None)
        with owner:
            preparation_guard(preparation_owner)
            from ipfs_accelerate_py.agent_supervisor.task_sources import taskboard_store
            from ipfs_accelerate_py.agent_supervisor.runtime import event_log
            pins = dict(request["source_sha256"])
            pins.update(module_pins(taskboard_store, event_log))
            controller_request = {
                "schema": coordinator.SCHEMA, "purpose": "controller_preparation", "task_id": facts.TASK,
                **{name: getattr(context, name) for name in ("task_cid", "task_revision", "native_attempt_id", "candidate_commit")},
                "board_path": str(base / facts.SOURCE_PATHS["execution_board"]),
                "expected_board_sha256": facts.sha(facts.read_source_file(base / facts.SOURCE_PATHS["execution_board"])),
                "event_path": str(owner.directory / "factual-completion.jsonl"), "source_sha256": pins,
                "scheduler_state_path": request["scheduler_state_path"],
                "expected_scheduler_policy_sha256": request["expected_scheduler_policy_sha256"],
                "root_lease_id": root.lease_id, "phase_lease_id": phase.lease_id, "disk_reservation_id": owner.reservation_id,
                "root_request_id": root_request, "phase_request_id": phase_request,
                "external_board_charge_bytes": 2 * 1024 * 1024,
            }
            controller_request_path = preparation_owner.directory / "synthetic-controller-completion-request.json"
            raw = facts.canonical(controller_request)
            write_new(controller_request_path, raw)
            publication = coordinator.complete_accepted_legacy_task(handle=handle, current_context=context,
                orchestration_owner=owner, root_lease=root, phase_lease=phase,
                request_path=controller_request_path, frozen_request_sha256=facts.sha(raw))
            preparation_guard(preparation_owner)
            native_closeout = owner.finalize(owner.directory, artifacts_durable=True)
            require(native_closeout["status"] == "released", "actual_fixture_controller_closeout_failed")
        return {"scope": SCOPE, "actual_native_validation_finalized": True,
                "actual_separate_orchestration_finalized": True, "actual_handle_consumed_once": True,
                "fixture_board_published": True, "real_LIR_task_accepted": False, "provider_calls_performed": False,
                "fixture_publication": publication}
    finally:
        if phase is not None:
            phase.release()
        root.release()
