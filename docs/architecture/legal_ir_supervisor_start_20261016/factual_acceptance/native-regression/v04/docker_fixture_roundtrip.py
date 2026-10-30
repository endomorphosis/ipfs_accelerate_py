"""Strict A91 Docker factual fixture backend, with actual daemon cleanup fences.

Docker administration/EOF lease belongs to the genuine preparation owner.
Checker outputs and direct start-client process belong to the separate genuine
VALIDATION owner. No protected builder, provider/auth route, image pull or model
call is used. Docker daemon metadata is an explicitly authorized administrative
allocation outside the SDK named-root accounting; no daemon disk quota is claimed.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

import factual_candidate as facts
import native_acceptance as acceptance
import controller_coordinator as coordinator
import native_fixture_roundtrip as common

HERE = Path(__file__).absolute().parent
HOST = "unix:///var/run/docker.sock"
IMAGE = "sha256:a91a42c918514203bb6cfc7c38b320dfb68e9dfc0571bf01976c0db993e24e5e"
IMAGE_LABEL = "2026-10-16-ordinary-v1"
PROFILE = "legal-ir-factual-validation-docker-v04"
PYTHON = Path("/usr/bin/python3.12")
CONTAINER_PYTHON = "/opt/ipfs-task-tools/bin/python"
CONTAINER_MEMORY = 256 * 1024 * 1024
CONTAINER_PIDS = 4
CONTAINER_ENV = ["PATH=/usr/bin:/bin", "HOME=/workspace", "TMPDIR=/workspace/tmp",
                 "TMP=/workspace/tmp", "TEMP=/workspace/tmp", "XDG_CACHE_HOME=/workspace/cache",
                 "LANG=C.UTF-8", "LC_ALL=C.UTF-8", "PYTHONDONTWRITEBYTECODE=1"]


def need(condition, reason):
    acceptance.require(condition, reason)


def save(path, value):
    common.write_new(path, facts.canonical(value))


def docker(arguments, *, config, timeout=30):
    result = subprocess.run(["/usr/bin/docker", "--host=" + HOST, "--config", str(config), *arguments],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout,
        env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "LANG": "C", "DOCKER_CONFIG": str(config)})
    need(len(result.stdout.encode()) + len(result.stderr.encode()) <= 1024 * 1024, "bounded_docker_observation_required")
    return result


def daemon(expected, config):
    result = docker(["info", "--format", "{{.DockerRootDir}}|{{.OSType}}|{{.Architecture}}|{{.ID}}"], config=config)
    need(result.returncode == 0 and result.stdout.strip() == "/var/lib/docker|linux|aarch64|" + expected,
         "actual_source_bound_system_daemon_identity_differs")


def inspect(identifier, config):
    result = docker(["container", "inspect", identifier], config=config)
    need(result.returncode == 0, "actual_selected_container_observation_unavailable")
    rows = facts.strict_json(result.stdout.encode())
    need(type(rows) is list and len(rows) == 1 and type(rows[0]) is dict, "single_actual_container_required")
    return rows[0]


def absent(identifier, config):
    result = docker(["container", "inspect", "--format", "{{.Id}}", identifier], config=config)
    return result.returncode != 0 and "No such container:" in result.stderr


def verify_container(row, expected, *, running=None):
    """Pure strict Docker correspondence; only the parent obtains observations."""
    need(row.get("Id") == expected["container_id"] and row.get("Image") == IMAGE
         and row.get("Name") == "/" + expected["container_name"], "actual_container_identity_differs")
    config, host = row.get("Config", {}), row.get("HostConfig", {})
    for key in ("User", "WorkingDir", "Entrypoint", "Cmd", "Env", "Labels"):
        facts.same(config.get(key), expected["config"][key], "actual_container_config_differs:" + key)
    required = {"NetworkMode": "none", "ReadonlyRootfs": True, "Privileged": False,
                "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"], "PidsLimit": CONTAINER_PIDS,
                "Memory": CONTAINER_MEMORY, "MemorySwap": CONTAINER_MEMORY, "NanoCpus": 1_000_000_000,
                "Runtime": "runc", "PidMode": "", "IpcMode": "private", "AutoRemove": False,
                "PublishAllPorts": False, "CgroupnsMode": "private"}
    for key, value in required.items():
        facts.same(host.get(key), value, "actual_container_host_isolation_differs:" + key)
    for key in ("CapAdd", "Devices", "DeviceRequests", "DeviceCgroupRules", "VolumesFrom", "Links", "GroupAdd"):
        need(host.get(key) in (None, []), "unexpected_container_host_authority:" + key)
    need(host.get("PortBindings") in (None, {}) and host.get("LogConfig", {}).get("Type") == "none"
         and host.get("RestartPolicy", {}).get("Name") == "no", "container_ports_logs_or_restart_broadened")
    need(host.get("Tmpfs") in (None, {}), "unaccounted_container_tmpfs_forbidden")
    actual_ulimits = host.get("Ulimits")
    need(type(actual_ulimits) is list, "actual_container_ulimits_missing")
    facts.same(sorted(actual_ulimits, key=lambda item: item["Name"]), [
        {"Name": "cpu", "Soft": 120, "Hard": 120},
        {"Name": "fsize", "Soft": 2 * 1024 * 1024, "Hard": 2 * 1024 * 1024},
        {"Name": "nofile", "Soft": 64, "Hard": 64},
    ], "actual_container_ulimits_differ")
    mounts = row.get("Mounts")
    need(type(mounts) is list and len(mounts) == len(expected["mounts"]), "actual_container_mount_population_differs")
    actual = sorted((m.get("Type"), m.get("Source"), m.get("Destination"), m.get("RW")) for m in mounts)
    wanted = sorted(("bind", m["source"], m["destination"], not m["readonly"]) for m in expected["mounts"])
    facts.same(actual, wanted, "actual_container_mount_boundary_differs")
    need(all(m.get("Propagation") == "rprivate" for m in mounts), "container_mount_propagation_broadened")
    state = row.get("State", {})
    if running is not None:
        need(state.get("Running") is running, "actual_container_running_state_differs")
    return state


class DockerFixture:
    """Actual legacy EOF lease plus independently observed immutable binding."""
    def __init__(self, *, runner, lease, expected, control, runtime_path, preparation_owner):
        self.runner, self.lease, self.expected = runner, lease, expected
        self.control, self.runtime_path, self.preparation_owner = control, runtime_path, preparation_owner
        self.close_requested = False
        self.cleanup_verified = False
        self.exit_verified = False
        self.creation_phase = "not_started"
        self.init_birth = None
        # Metadata-only construction after an actual lease has been allocated.
        # Every fallible birth observation belongs inside protected preparation.
        self.watchdog_birth = None

    def observe(self, *, running=None):
        daemon(self.expected["daemon_id"], self.lease.docker_config)
        return verify_container(inspect(self.expected["container_id"], self.lease.docker_config), self.expected, running=running)

    def prove_exit(self):
        state = self.observe(running=False)
        need(type(state.get("ExitCode")) is int and state["ExitCode"] == 0 and state.get("OOMKilled") is False
             and state.get("Error") == "", "actual_checker_container_did_not_exit_successfully")
        self.exit_verified = True
        save(self.control / "actual-container-exit.json", {"schema": "legal-ir-factual-docker-exit@1",
            "daemon_id": self.expected["daemon_id"], "container_id": self.expected["container_id"],
            "container_name": self.expected["container_name"], "image_id": IMAGE,
            "observed_exit_code": 0, "observed_oom_killed": False, "actual_init_birth": self.init_birth,
            "provider_calls_performed": False, "real_LIR_task_accepted": False})

    def cleanup(self, *, start_client_finished):
        """No context may exit until actual daemon/reaper/lease absence is proved.

        Invoke the genuine lease close at most once. Unknown cleanup retains
        custody observation-only for actual recovery; it never fabricates
        success, blindly recloses, or releases native validation in finally.
        """
        if self.cleanup_verified:
            return
        next_notice = 0
        while self.creation_phase == "started_result_unknown":
            # A timed-out client does not prove that the daemon has completed
            # create. Do not close the reaper and erase binding inputs before
            # an actual exact-name/config/image observation resolves its CID.
            try:
                daemon(self.expected["daemon_id"], self.lease.docker_config)
                observed = inspect(self.expected["container_name"], self.lease.docker_config)
                identifier = observed.get("Id")
                need(type(identifier) is str and re.fullmatch(r"[0-9a-f]{64}", identifier), "actual_created_cid_unresolved")
                resolved = dict(self.expected, container_id=identifier)
                verify_container(observed, resolved, running=False)
                self.expected = resolved
                self.creation_phase = "created_known"
                save(self.control / "actual-late-inert-create-resolution.json", {
                    "schema": "legal-ir-factual-inert-create-resolution@1", "container_id": identifier,
                    "container_name": self.expected["container_name"], "daemon_id": self.expected["daemon_id"],
                    "actual_full_inert_config_verified": True})
                break
            except BaseException:
                pass
            if time.monotonic() >= next_notice:
                print(facts.canonical({"status": "preparation_custody_held_inert_create_outcome_unknown", "scope": common.SCOPE,
                    "container_name": self.expected["container_name"], "daemon_id": self.expected["daemon_id"],
                    "no_checker_execution_or_native_release": True}).decode(), flush=True)
                next_notice = time.monotonic() + 30
            try:
                time.sleep(1)
            except BaseException:
                pass
        while not self.close_requested:
            try:
                daemon(self.expected["daemon_id"], self.lease.docker_config)
                self.close_requested = True
                self.lease.close(docker_run_finished=start_client_finished)
            except BaseException:
                # If close began, its EOF/actual reaper is already responsible.
                # Do not call close twice; observe actual state/recovery only.
                if not self.close_requested:
                    try:
                        time.sleep(1)
                    except BaseException:
                        pass
        while True:
            try:
                daemon(self.expected["daemon_id"], Path("/nonexistent"))
                no_container = absent(self.expected["container_name"], Path("/nonexistent")) and (
                    self.expected["container_id"] is None or absent(self.expected["container_id"], Path("/nonexistent")))
                watchdog = self.runner.read_process_birth(self.lease._watchdog.pid)
                watchdog_gone = watchdog is None or (self.watchdog_birth is not None and watchdog.start_time_ticks != self.watchdog_birth.start_time_ticks)
                # poll() reaps only our actual Popen child, never a receipt PID.
                watchdog_code = self.lease._watchdog.poll()
                no_paths = not os.path.lexists(self.lease.lease_root) and not os.path.lexists(self.lease.provider_home) and not os.path.lexists(self.lease.prompt_path)
                if no_container and watchdog_gone and watchdog_code is not None and no_paths:
                    save(self.control / "actual-docker-cleanup.json", {"schema": "legal-ir-factual-docker-cleanup@1",
                        "daemon_id": self.expected["daemon_id"], "container_id": self.expected["container_id"],
                        "container_name": self.expected["container_name"], "container_absent_by_id_and_name": True,
                        "watchdog_birth_gone": True, "watchdog_returncode": watchdog_code,
                        "lease_root_home_prompt_absent": True, "preserve_for_recovery_reported": self.lease.preserve_for_recovery,
                        "actual_independent_absence_verified": True})
                    self.cleanup_verified = True
                    return
            except BaseException:
                pass
            now = time.monotonic()
            if now >= next_notice:
                print(facts.canonical({"status": "native_custody_held_docker_cleanup_unknown", "scope": common.SCOPE,
                    "container_id": self.expected["container_id"], "container_name": self.expected["container_name"],
                    "daemon_id": self.expected["daemon_id"], "no_native_release_or_acceptance": True}).decode(), flush=True)
                next_notice = now + 30
            try:
                time.sleep(1)
            except BaseException:
                pass


def prepare_docker_fixture(*, preparation_owner, context, validation_attempt, expected_daemon_id, runner_path, runner_sha256):
    common.preparation_guard(preparation_owner)
    need(re.fullmatch(r"[0-9a-f-]{36}", expected_daemon_id) is not None, "source_bound_actual_daemon_id_required")
    need(context.task_cid.startswith("fixture:") and context.orchestration_namespace.startswith(common.PREFIX), "isolated_fixture_only")
    need(validation_attempt.is_dir() and not any(validation_attempt.iterdir()), "initially_empty_validation_attempt_required")
    control = preparation_owner.directory / "docker-factual-control"
    control.mkdir(mode=0o700)
    gate = control / "inner-gate"
    gate.mkdir(mode=0o700)
    temporary = control / "temporary"
    temporary.mkdir(mode=0o700)
    os.environ.update(TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary))
    tempfile.tempdir = str(temporary)
    need(facts.sha(facts.read_source_file(runner_path)) == runner_sha256, "reviewed_legacy_runner_source_pin_differs")
    spec = importlib.util.spec_from_file_location("legal_ir_actual_factual_docker_legacy", runner_path)
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    need(runner._ordinary_task_toolchain_image_pin() == IMAGE, "ordinary_image_source_authority_differs")
    need(runner._host_codex_task_toolchain_python() == PYTHON, "qualified_host_python_source_differs")
    daemon(expected_daemon_id, Path("/nonexistent"))
    home = Path(tempfile.mkdtemp(prefix="asref-grok-home-"))
    descriptor, prompt = tempfile.mkstemp(prefix="asref-grok-prompt-")
    os.close(descriptor)
    save(control / "actual-docker-intent.json", {"schema": "legal-ir-factual-docker-intent@1", "profile": PROFILE,
        "scope": common.SCOPE, "daemon_id": expected_daemon_id, "endpoint": HOST, "image_id": IMAGE,
        "native_preparation": preparation_owner.to_dict()["resource_lease"], "runner_sha256": runner_sha256,
        "provider_calls_performed": False, "image_acquisition": False,
        "native_named_roots_cover_daemon_storage": False, "hard_daemon_disk_quota": False})
    lease = runner._LegacyDockerContainerLease.create("/usr/bin/docker", provider="grok", provider_home=home, prompt_path=Path(prompt))
    fixture = DockerFixture(runner=runner, lease=lease, expected={"container_id": None,
        "container_name": lease.container_name, "daemon_id": expected_daemon_id}, control=control,
        runtime_path=control / "frozen-docker-runtime.json", preparation_owner=preparation_owner)
    try:
        fixture.watchdog_birth = runner.read_process_birth(lease._watchdog.pid)
        need(fixture.watchdog_birth is not None, "actual_watchdog_birth_missing")
        need(runner._docker_ordinary_task_toolchain_image_id("/usr/bin/docker", docker_config=lease.docker_config) == IMAGE,
             "actual_ordinary_image_id_label_linuxarm64_not_qualified")
        image = docker(["image", "inspect", IMAGE], config=lease.docker_config)
        need(image.returncode == 0, "actual_pinned_image_unavailable")
        image = facts.strict_json(image.stdout.encode())[0]
        need(image["Id"] == IMAGE and image["Os"] == "linux" and image["Architecture"] == "arm64"
             and image["Config"]["Labels"]["org.ipfs-accelerate.ordinary-worker"] == IMAGE_LABEL
             and not image["Config"].get("Volumes"), "actual_pinned_image_profile_differs")
        image_env = image["Config"].get("Env") or []
        need(all(type(value) is str and value.startswith("PATH=") for value in image_env), "unexpected_pinned_image_environment")
        inner_request = control / "inner-request.json"
        save(inner_request, {"schema": "legal-ir-docker-inner-factual-request@1", "scope": common.SCOPE,
            "source_root": context.source_root, "report_path": context.report_path, "report_sha256": context.report_sha256,
            "evidence_path": "/workspace/exports/factual-correspondence.json", "gate_path": "/opt/legal-ir/admission/ready"})
        mounts = [
            {"source": "/usr", "destination": "/usr", "readonly": True},
            {"source": str(PYTHON), "destination": CONTAINER_PYTHON, "readonly": True},
            {"source": str(Path(facts.__file__).absolute()), "destination": "/opt/legal-ir/factual_candidate.py", "readonly": True},
            {"source": str(HERE / "inner_checker.py"), "destination": "/opt/legal-ir/inner_checker.py", "readonly": True},
            {"source": str(inner_request), "destination": "/opt/legal-ir/request.json", "readonly": True},
            {"source": str(gate), "destination": "/opt/legal-ir/admission", "readonly": True},
            {"source": context.source_root, "destination": context.source_root, "readonly": True},
            {"source": context.report_path, "destination": context.report_path, "readonly": True},
            {"source": str(validation_attempt), "destination": "/workspace", "readonly": False},
        ]
        cmd = ["-i", *CONTAINER_ENV, CONTAINER_PYTHON, "-I", "-S", "-B", "/opt/legal-ir/inner_checker.py", "/opt/legal-ir/request.json"]
        labels = dict(image["Config"]["Labels"], **{"org.justicedao.factual-validation": PROFILE, "org.justicedao.fixture-namespace": context.orchestration_namespace})
        args = ["container", "create", "--pull=never", "--name", lease.container_name, "--cidfile", str(lease.cidfile),
            "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--runtime=runc", "--cgroupns=private",
            "--pids-limit=4", "--cpus=1", "--memory=268435456", "--memory-swap=268435456", "--restart=no",
            "--log-driver=none", "--user", f"{os.getuid()}:{os.getgid()}", "--workdir=/workspace", "--entrypoint=/usr/bin/env",
            "--ulimit=nofile=64:64", "--ulimit=fsize=2097152:2097152", "--ulimit=cpu=120:120"]
        for name in ("org.justicedao.factual-validation", "org.justicedao.fixture-namespace"):
            args.extend(["--label", name + "=" + labels[name]])
        for mount in mounts:
            source_path = Path(mount["source"])
            need(source_path.is_absolute() and ".." not in source_path.parts and "," not in mount["source"]
                 and os.path.abspath(source_path) == mount["source"]
                 and not any(path.is_symlink() for path in (source_path, *source_path.parents)), "canonical_unaliased_mount_source_required")
            args.extend(["--mount", "type=bind,src=" + mount["source"] + ",dst=" + mount["destination"] + (",readonly" if mount["readonly"] else "")])
        args.extend([IMAGE, *cmd])
        watchdog = runner.read_process_birth(lease._watchdog.pid)
        need(watchdog is not None, "actual_watchdog_birth_missing")
        save(control / "actual-lease-before-create.json", {"schema": "legal-ir-factual-docker-lease@1", "daemon_id": expected_daemon_id,
            "container_name": lease.container_name, "lease_root": str(lease.lease_root), "cidfile": str(lease.cidfile),
            "watchdog_birth": watchdog.to_dict(), "native_preparation": preparation_owner.to_dict()["resource_lease"],
            "create_is_inert_not_checker_execution": True})
        expected = {"container_id": None, "container_name": lease.container_name, "daemon_id": expected_daemon_id, "image_id": IMAGE,
            "mounts": mounts, "config": {"User": f"{os.getuid()}:{os.getgid()}", "WorkingDir": "/workspace", "Entrypoint": ["/usr/bin/env"],
            "Cmd": cmd, "Env": image_env, "Labels": labels}}
        fixture.expected = expected
        fixture.creation_phase = "started_result_unknown"
        created = docker(args, config=lease.docker_config, timeout=60)
        need(created.returncode == 0 and re.fullmatch(r"[0-9a-f]{64}\s*", created.stdout), "actual_inert_container_create_failed")
        cid = created.stdout.strip()
        need(lease.cidfile.read_text().strip() == cid, "actual_cidfile_and_daemon_return_differs")
        expected = dict(expected, container_id=cid)
        fixture.expected = expected
        verify_container(inspect(cid, lease.docker_config), expected, running=False)
        fixture.creation_phase = "created_known"
        runtime_path = control / "frozen-docker-runtime.json"
        save(runtime_path, {"schema": "legal-ir-docker-factual-validation-runtime@1", "profile": PROFILE, "scope": common.SCOPE,
            "expected": expected, "host_python": str(PYTHON), "host_python_sha256": facts.sha(PYTHON.read_bytes()),
            "outer_barrier_seconds": 180, "inner_barrier_seconds": 180, "productive_wall_seconds": 120,
            "validation_lifecycle_seconds": 900, "image_acquisition": False, "provider_calls_performed": False,
            "network_mode": "none", "host_readonly_usr": True, "no_socket_or_credentials_mount": True,
            "docker_daemon_storage_outside_native_named_root_accounting": True,
            "host_process_and_container_memory_enforcement": "SDK samples the attached host process group RSS; Docker independently enforces container cgroup256MiB; no combined RSS sampling claim"})
        fixture.runtime_path = runtime_path
        return fixture
    except BaseException:
        fixture.cleanup(start_client_finished=True)
        raise


def start_argv(fixture, validation_attempt):
    return [str(Path(sys.executable).resolve()), "-I", "-B", str(HERE / "outer_start_barrier.py"),
        str(validation_attempt / "tmp/native-docker-attachment.go"), "--", "/usr/bin/docker", "--host=" + HOST,
        "--config", str(fixture.lease.docker_config), "container", "start", "--attach", fixture.expected["container_id"]]


def execute_actual_validation(*, preparation_owner, validation_owner, context, fixture):
    need(type(validation_owner).__module__ == acceptance.NATIVE_MODULE and type(validation_owner).__name__ == "ValidationResourceReservation",
         "genuine_native_validation_owner_required")
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.validation_run_resources import ValidationResourceReservation
    need(type(validation_owner) is ValidationResourceReservation, "genuine_native_validation_owner_required")
    facts.same(validation_owner.request["argv"], start_argv(fixture, validation_owner.directory), "actual_docker_request_argv_differs")
    facts.same(validation_owner.request["selected_checks"], [acceptance.CHECK], "actual_factual_check_differs")
    process = None
    with common.opened_validation(validation_owner):
        try:
            public = validation_owner.to_dict()
            save(fixture.control / "actual-validation-admission.json", {"schema": "legal-ir-factual-actual-native-admission@1",
                "reservation_id": public["reservation_id"], "native_lease": public["resource_lease"], "lane": public["scheduler_lane"],
                "workload": public["workload"], "request_sha256": validation_owner.request_sha256,
                "container_id": fixture.expected["container_id"], "daemon_id": fixture.expected["daemon_id"],
                "owner_pid": os.getpid(), "native_grant_verified_by_actual_object": True,
                "receipt_is_inert_not_acceptance_authority": True})
            descriptor = os.open(validation_owner.directory / "logs/actual-docker-start.log",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, "wb") as output:
                environment = {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "LANG": "C",
                    "TMPDIR": str(validation_owner.directory / "tmp"), "TMP": str(validation_owner.directory / "tmp"),
                    "TEMP": str(validation_owner.directory / "tmp"), "PYTHONDONTWRITEBYTECODE": "1"}
                process = subprocess.Popen(validation_owner.request["argv"], cwd=validation_owner.request["cwd"], env=environment,
                    stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                success = False
                validation_owner.attach_child(process.pid)
                need(validation_owner.heartbeat(), "actual_validation_heartbeat_lost_before_docker_start")
                runtime = facts.strict_json(facts.read_source_file(fixture.runtime_path))
                need(facts.sha(PYTHON.read_bytes()) == runtime["host_python_sha256"], "actual_readonly_host_python_content_changed")
                save(fixture.control / "actual-attached-docker-client.json", {"schema": "legal-ir-actual-attached-docker-client@1",
                    "child_pid": process.pid, "native_validation": validation_owner.to_dict()["resource_lease"],
                    "request_sha256": validation_owner.request_sha256, "container_id": fixture.expected["container_id"]})
                common.write_new(validation_owner.directory / "tmp/native-docker-attachment.go", b"G")
                start_deadline = time.monotonic() + 30
                while True:
                    state = fixture.observe()
                    if state.get("Running") is True:
                        birth = fixture.runner.read_process_birth(state["Pid"])
                        need(birth is not None, "actual_owned_container_init_birth_required")
                        fixture.init_birth = birth.to_dict()
                        break
                    need(process.poll() is None and time.monotonic() < start_deadline, "actual_docker_start_or_inner_barrier_failed")
                    time.sleep(0.1)
                save(fixture.control / "actual-container-confinement-before-checker.json", {
                    "schema": "legal-ir-actual-docker-confinement@1", "daemon_id": fixture.expected["daemon_id"],
                    "container_id": fixture.expected["container_id"], "init_birth": fixture.init_birth,
                    "full_config_mounts_isolation_verified": True, "checker_gate_still_closed": True})
                common.write_new(fixture.control / "inner-gate/ready", b"G")
                deadline = time.monotonic() + 120
                next_heartbeat = 0
                while process.poll() is None:
                    validation_owner.check_limits(full=False)
                    need(time.monotonic() < deadline, "actual_docker_checker_productive_deadline_exceeded")
                    if time.monotonic() >= next_heartbeat:
                        need(validation_owner.heartbeat(), "actual_validation_heartbeat_lost")
                        fixture.observe()
                        next_heartbeat = time.monotonic() + 0.5
                    time.sleep(0.05)
                process.wait(timeout=1)
                need(process.returncode == 0, "actual_docker_attach_child_failed")
                fixture.prove_exit()
                success = True
        except BaseException:
            if process is not None and process.poll() is None:
                common.stop_actual_child(process)
            raise
        finally:
            # Covers admission/log/Popen failures as well as productive work.
            # Any unknown cleanup keeps this frame inside native custody.
            fixture.cleanup(start_client_finished=process is None or process.poll() is not None)
        need(success and fixture.cleanup_verified and fixture.exit_verified, "actual_docker_exit_and_cleanup_not_verified")
        evidence = validation_owner.directory / "exports/factual-correspondence.json"
        result = validation_owner.directory / "exports/validation-result.json"
        common.write_new(result, facts.canonical({"schema": "owned-validation-result/v1", "request_sha256": validation_owner.request_sha256,
            "executed_checks": [acceptance.CHECK], "returncode": 0, "completed": True, "evidence_paths": [str(evidence)]}))
        return acceptance.issue_native_acceptance(reservation=validation_owner, process=process, context=context,
            evidence_path=evidence, result_path=result)


def consume_or_publish_fixture(*, preparation_owner, validation_owner, context, handle,
        orchestration_attempt_directory, orchestration_ledger_path, orchestration_roots, handles_only):
    """Supported genuine controller path, retaining the same actual private handle."""
    base = Path(context.source_root)
    request = validation_owner.request
    if handles_only:
        sources, frozen = facts.load_source_bundle(base)
        proposed, summary = acceptance.consume_for_manual_completion(handle, current_context=context,
            board_raw=sources["execution_board"], report_raw=facts.read_source_file(Path(context.report_path)),
            source_documents=sources, frozen_files=frozen)
        need(proposed.count(b"- Status: completed\n") == 1 and proposed.count(b"- Status: blocked\n") == 63,
                "fixture_completion_projection_differs")
        try:
            acceptance._entry(handle)
        except acceptance.AcceptanceRefusal:
            pass
        else:
            raise acceptance.AcceptanceRefusal("fixture_handle_was_replayable")
        return {"scope": common.SCOPE, "actual_native_validation_finalized": True, "actual_handle_consumed_once": True,
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
            storage_bytes=common.LIMITS["controller_storage_bytes"], memory_mb=common.LIMITS["controller_memory_mb"],
            cpu_slots=1, child_process_slots=1, scheduler_state_path=request["scheduler_state_path"],
            expected_scheduler_policy_sha256=request["expected_scheduler_policy_sha256"], timeout_seconds=900, ledger_lock_timeout_seconds=None)
        with owner:
            common.preparation_guard(preparation_owner)
            from ipfs_accelerate_py.agent_supervisor.task_sources import taskboard_store
            from ipfs_accelerate_py.agent_supervisor.runtime import event_log
            pins = dict(request["source_sha256"])
            pins.update(common.module_pins(taskboard_store, event_log))
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
            common.write_new(controller_request_path, raw)
            publication = coordinator.complete_accepted_legacy_task(handle=handle, current_context=context,
                orchestration_owner=owner, root_lease=root, phase_lease=phase,
                request_path=controller_request_path, frozen_request_sha256=facts.sha(raw))
            common.preparation_guard(preparation_owner)
            native_closeout = owner.finalize(owner.directory, artifacts_durable=True)
            need(native_closeout["status"] == "released", "actual_fixture_controller_closeout_failed")
        return {"scope": common.SCOPE, "actual_native_validation_finalized": True,
                "actual_separate_orchestration_finalized": True, "actual_handle_consumed_once": True,
                "fixture_board_published": True, "real_LIR_task_accepted": False, "provider_calls_performed": False,
                "fixture_publication": publication}
    finally:
        if phase is not None:
            phase.release()
        root.release()
