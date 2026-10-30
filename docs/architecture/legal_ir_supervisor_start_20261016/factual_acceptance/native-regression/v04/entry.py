"""Standalone SAME-PROCESS genuine native acceptance fixture qualification.

One frozen engineering request selects a fresh isolated fixture run. Actual
native scheduler/ledger policy is opened, never initialized or reconfigured.
No provider, credential, real task, model/corpus, image acquisition or foreign
receipt is used. This owner creates and uses all actual SDK objects here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import re

HERE = Path(__file__).absolute().parent
ADAPTER = HERE.parents[1] / "adapter-v02"
COMMON = HERE.parent / "v03"
DATASETS = Path("/home/barberb/lift_coding/artifacts/legal-conditions-composition-diagnostic-20261015-01/datasets")
ACCELERATOR = Path("/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller")
ROOT = Path("/home/barberb/lift_coding/external/ipfs_datasets")
ROOTS = (ROOT / "workspace/test-logs", ROOT / "workspace/todo-queues", ROOT / "docs/implementation/reports/evidence", Path("/tmp/pytest-of-barberb"))
STATE = Path("/tmp/ipfs-datasets-resource-scheduler-1000.json")
POLICY = "f07cb14289fcb6275e4497dcc36be0dcf8bfc3dc87531ff7a9537eb85fdcc661"
LEDGER = ROOT / "workspace/test-logs/federal-corpus-audits/owned-daemon-resource-control/disk-reservations.json"
FIELDS = {"schema", "name", "purpose", "scope", "run_root", "mode", "scheduler_state_path",
          "expected_scheduler_policy_sha256", "ledger_path", "roots", "source_sha256",
          "adapter_manifest_sha256", "regression_manifest_sha256", "expected_docker_daemon_id"}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def raw_file(path, maximum=4 * 1024 * 1024):
    import stat
    path = Path(path)
    need(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), "unaliased absolute source path required")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= maximum, "bounded regular source required")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    witness = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    need(witness(before) == witness(after) == witness(path.lstat()) and len(raw) == before.st_size, "source changed while reading")
    return raw


def read_request(path, expected_sha):
    raw = raw_file(path, 1024 * 1024)
    need(hashlib.sha256(raw).hexdigest() == expected_sha, "frozen entry request SHA differs")
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate entry request key")
            result[key] = value
        return result
    request = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite request")))
    need(type(request) is dict and set(request) == FIELDS and request["schema"] == "legal-ir-native-docker-fixture-entry@4"
         and request["scope"] == "synthetic-native-acceptance-regression-only"
         and request["purpose"] == "controller_preparation", "entry request scope differs")
    need(type(request["name"]) is str and re.fullmatch(r"legal-ir-native-acceptance-fixture-[a-z0-9][a-z0-9-]{0,80}", request["name"])
         is not None, "fresh fixture name required")
    need(re.fullmatch(r"[0-9a-f-]{36}", request["expected_docker_daemon_id"]) is not None, "exact actual Docker daemon ID required")
    need(request["mode"] in {"handles_only", "isolated_fixture_publication"}, "finite fixture mode required")
    need(request["scheduler_state_path"] == str(STATE) and request["expected_scheduler_policy_sha256"] == POLICY
         and request["ledger_path"] == str(LEDGER) and request["roots"] == [str(p) for p in ROOTS], "existing native authority binding differs")
    run_root = Path(request["run_root"])
    need(type(request["run_root"]) is str and ".." not in run_root.parts
         and run_root.as_posix() == request["run_root"] and os.path.abspath(run_root) == request["run_root"],
         "canonical fixture run root required")
    need(run_root.is_absolute() and run_root.name == request["name"] and any(root in run_root.parents for root in ROOTS)
         and not any(p.is_symlink() for p in (run_root, *run_root.parents)), "isolated fixture run root required")
    need(Path("/tmp") not in run_root.parents, "owned non-/tmp fixture source root required")
    need(type(request["source_sha256"]) is dict and 0 < len(request["source_sha256"]) <= 256, "finite source pin closure required")
    for spelling, digest in request["source_sha256"].items():
        need(type(spelling) is str and type(digest) is str and len(digest) == 64
             and hashlib.sha256(raw_file(spelling)).hexdigest() == digest, "frozen entry source pin differs")
    for path, field in ((ADAPTER / "SOURCE_MANIFEST.json", "adapter_manifest_sha256"), (HERE / "SOURCE_MANIFEST.json", "regression_manifest_sha256")):
        need(hashlib.sha256(raw_file(path)).hexdigest() == request[field], "frozen adapter/regression manifest differs")
        manifest = json.loads(raw_file(path))
        for row in manifest["files"]:
            source = path.parent / row["path"]
            content = raw_file(source)
            need(len(content) == row["bytes"] and hashlib.sha256(content).hexdigest() == row["sha256"]
                 and request["source_sha256"].get(str(source)) == row["sha256"], "complete source manifest pin required")
    return request, raw


def main():
    need(len(sys.argv) == 3, "usage: entry.py ABSOLUTE_FROZEN_REQUEST EXPECTED_REQUEST_SHA256")
    request_path = Path(sys.argv[1])
    request, raw = read_request(request_path, sys.argv[2])
    run_root = Path(request["run_root"])
    need(not run_root.exists(), "fixture run root already exists; never replay")
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", IPFS_DATASETS_PY_MINIMAL_IMPORTS="1", IPFS_ACCEL_SKIP_CORE="1",
                      IPFS_AUTO_INSTALL="false", IPFS_DATASETS_AUTO_INSTALL="false", OMP_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    sys.path[:0] = [str(ADAPTER), str(HERE), str(COMMON), str(DATASETS), str(ACCELERATOR)]
    import factual_candidate as facts
    import native_acceptance as acceptance
    import controller_coordinator as coordinator
    import native_fixture_roundtrip as regression
    import docker_fixture_roundtrip as backend
    need(acceptance.HANDLE_LIFETIME_SECONDS == 900 and Path(acceptance.__file__).absolute().parent == ADAPTER,
         "reviewed adapter V02 required")
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import existing_resource_scheduler, resource_scheduler
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import parented_phase_resources, validation_run_resources, autoencoder_daemon_resources
    from ipfs_accelerate_py.agent_supervisor.task_sources import taskboard_store
    from ipfs_accelerate_py.agent_supervisor.runtime import event_log
    closure = (facts, acceptance, coordinator, regression, existing_resource_scheduler, resource_scheduler,
               parented_phase_resources, validation_run_resources, autoencoder_daemon_resources,
               backend, taskboard_store, event_log)
    for module in closure:
        path = Path(module.__file__).absolute()
        need(request["source_sha256"].get(str(path)) == facts.sha(raw_file(path)), "actual imported native/source closure not pinned")
    need(Path(resource_scheduler.__file__).absolute().is_relative_to(DATASETS), "native producer tree differs")
    scheduler = existing_resource_scheduler.open_existing_scheduler(STATE, POLICY)
    def interrupted(_number, _frame):
        raise KeyboardInterrupt("finite owned fixture regression interrupted")
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, interrupted)
    # Empty owned directory bootstrap is needed by the real disk owner API;
    # no productive file bytes are written until actual disk admission.
    run_root.mkdir(mode=0o700)
    preparation = run_root / "preparation"
    preparation.mkdir(mode=0o700)
    root_request = request["name"] + ":fixture-preparation-root"
    with scheduler.acquire("orchestration", cpu_slots=1, memory_mb=1024, child_process_slots=4,
                           requires_gpu=False, timeout=900, request_id=root_request) as root:
        with root.acquire_child(cpu_slots=1, memory_mb=1024, child_process_slots=4, requires_gpu=False,
                                timeout=900, request_id=request["name"] + ":fixture-preparation-phase") as phase:
            with parented_phase_resources.ParentedPhaseResourceReservation(LEDGER, roots=ROOTS,
                    attempt_directory=preparation, parent_lease=phase, storage_bytes=32 * 1024 * 1024,
                    memory_mb=1024, cpu_slots=1, child_process_slots=4, scheduler_state_path=STATE,
                    expected_scheduler_policy_sha256=POLICY, timeout_seconds=2400,
                    ledger_lock_timeout_seconds=None) as preparation_owner:
                read_request(request_path, sys.argv[2])
                temporary = preparation / "tmp"
                temporary.mkdir(mode=0o700)
                os.environ.update(TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary), XDG_CACHE_HOME=str(temporary))
                context = regression.prepare_isolated_fixture(preparation_owner=preparation_owner,
                    immutable_fixture_root=ADAPTER / "tests/fixtures/bootstrap", namespace=request["name"])
                preparation_owner.account_external_bytes("synthetic_fixture_sibling_directory_metadata", 4096)
                validation_attempt, controller_attempt = run_root / "validation", run_root / "controller"
                validation_attempt.mkdir(mode=0o700)
                controller_attempt.mkdir(mode=0o700)
                runner_path = ACCELERATOR / "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py"
                fixture = backend.prepare_docker_fixture(preparation_owner=preparation_owner, context=context,
                    validation_attempt=validation_attempt, expected_daemon_id=request["expected_docker_daemon_id"],
                    runner_path=runner_path, runner_sha256=request["source_sha256"][str(runner_path)])
                try:
                    source_pins = dict(request["source_sha256"])
                    source_pins.update({str(path): facts.sha(facts.read_source_file(path)) for path in regression.pinned_input_paths(context)})
                    for path in (fixture.runtime_path, fixture.control / "inner-request.json"):
                        source_pins[str(path)] = facts.sha(facts.read_source_file(path))
                    native_request = {
                        "schema": "owned-validation-request/v1", "argv": backend.start_argv(fixture, validation_attempt),
                        "cwd": str(validation_attempt), "selected_checks": [acceptance.CHECK], "source_sha256": source_pins,
                        "timeout_seconds": 900, "storage_bytes": 32 * 1024 * 1024, "memory_mb": 512,
                        "cpu_slots": 1, "child_process_slots": 4, "attempt_directory": str(validation_attempt),
                        "outputs": {name: str(validation_attempt / name) for name in ("pytest", "tmp", "cache", "db", "exports", "logs")},
                        "declared_external_writable_paths": [], "process_scope": "isolated_process_group",
                        "scheduler_state_path": str(STATE), "expected_scheduler_policy_sha256": POLICY,
                        "roots": [str(path) for path in ROOTS],
                    }
                    validation_request_path = preparation / "frozen-native-docker-validation-request.json"
                    encoded = facts.canonical(native_request)
                    regression.write_new(validation_request_path, encoded)
                    read_request(request_path, sys.argv[2])
                    regression.preparation_guard(preparation_owner)
                    validation_owner = validation_run_resources.ValidationResourceReservation(LEDGER, roots=ROOTS,
                        attempt_directory=validation_attempt, storage_bytes=32 * 1024 * 1024, memory_mb=512,
                        cpu_slots=1, child_process_slots=4, scheduler_state_path=STATE,
                        expected_scheduler_policy_sha256=POLICY, validation_manifest_path=validation_request_path,
                        frozen_validation_manifest_sha256=facts.sha(encoded), timeout_seconds=900,
                        ledger_lock_timeout_seconds=60, full_check_interval_seconds=300)
                    handle = backend.execute_actual_validation(preparation_owner=preparation_owner,
                        validation_owner=validation_owner, context=context, fixture=fixture)
                    outcome = backend.consume_or_publish_fixture(preparation_owner=preparation_owner,
                        validation_owner=validation_owner, context=context, handle=handle,
                        orchestration_attempt_directory=controller_attempt, orchestration_ledger_path=LEDGER,
                        orchestration_roots=ROOTS, handles_only=request["mode"] == "handles_only")
                    outcome.update(actual_kernel_backend="source-bound-ordinary-A91-Docker",
                        actual_container_exit_and_cleanup_verified=fixture.exit_verified and fixture.cleanup_verified)
                finally:
                    # Also covers request construction/VRR.open failure while
                    # the separately preparation-owned container is still inert.
                    fixture.cleanup(start_client_finished=True)
                need(outcome["actual_native_validation_finalized"] and not outcome["provider_calls_performed"]
                     and not outcome["real_LIR_task_accepted"], "real native fixture outcome differs")
                regression.preparation_guard(preparation_owner)
                regression.write_new(preparation / "fixture-qualification-before-preparation-closeout.json", facts.canonical({
                    "schema": "legal-ir-native-docker-fixture-qualification@4", "request_sha256": facts.sha(raw),
                    "outcome": outcome, "preparation_native_closeout_pending": True,
                    "real_LIR_task_accepted": False, "provider_calls_performed": False,
                }))
                final = preparation_owner.finalize(preparation, artifacts_durable=True)
                need(final["status"] == "released" and final["resource_lease"]["released"] is True,
                     "real preparation closeout failed")
    print(json.dumps({"schema": "legal-ir-native-docker-fixture-entry-terminal@4", "scope": regression.SCOPE,
        "request_sha256": facts.sha(raw), "actual_preparation_finalized": True, "outcome": outcome,
        "real_LIR_task_accepted": False, "provider_calls_performed": False}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({"schema": "legal-ir-native-docker-fixture-entry-terminal@4", "passed": False,
                          "error_type": type(error).__name__, "real_LIR_task_accepted": False,
                          "provider_calls_performed": False}), flush=True)
        raise
