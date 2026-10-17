"""Actual non-model A91 dual-pipe mechanics; never a Grok/provider outcome.

Root alone executes after reviewing a real prepared capsule/request. Actual
source startup races are preserved. No wrapper/finish/success callback, model,
auth route, native acceptance, canonical task or production deployment is used.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid

HERE = Path(__file__).absolute().parent
NEW = HERE.parent
RUN = Path("/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs/legal-ir-callable-dual-pipe-native-20261017-r02")
CAPSULE = Path("/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs/legal-ir-callable-handoff-20261017-r02/preparation/capsule")
CUSTODY_SOURCE = CAPSULE / "ipfs_accelerate_py/agent_supervisor/runtime/ordinary_custody_hold.py"
RUNNER_MANIFEST = NEW / "runner-v03/SOURCE_MANIFEST.json"
HANDOFF_MANIFEST = NEW / "handoff-v03/SOURCE_MANIFEST.json"
ADMIN_AUDIT = NEW / "dual-pipe-admin-source-audit-v01.json"
ADMIN_AUDIT_SHA = "ab24b59c2dffba4560d4298c91eafc96cbbcf34b466078614f229b5d1f47faea"
RUNNER_MANIFEST_SHA = "1a2d28a952f0bac6a7af951b4754085dcb1110f86fb46ebf319014f2e66062e2"
HANDOFF_MANIFEST_SHA = "6ca1a9ca25f1bb477841a006e2b77ba34c0c647d5930b3eaabf7537e6dbf6879"
PREPARATION_OBSERVATION = NEW / "capsule-preparation-root-observation-r02.json"
PREPARER_MANIFEST = NEW / "capsule-preparation-v02/SOURCE_MANIFEST.json"
PREPARER_MANIFEST_SHA = "df9caa66dbb9484614bc4c1829da0a21d53260b909c72786ae55da58995c0c56"
PREPARER_REQUEST_SHA = "64f5e53aadbe05d6d3a045a3c849668011e00043090dea0e421b4f178fdf4da5"
BASE_CAPSULE_SHA = "b2a78de35845425f533f7a361b3998f41628d472fafbbccb3aa346d7b50e0925"
DATASETS = Path("/home/barberb/lift_coding/artifacts/legal-conditions-composition-diagnostic-20261015-01/datasets")
ROOT = Path("/home/barberb/lift_coding/external/ipfs_datasets")
ROOTS = (ROOT / "workspace/test-logs", ROOT / "workspace/todo-queues",
         ROOT / "docs/implementation/reports/evidence", Path("/tmp/pytest-of-barberb"))
LEDGER = ROOT / "workspace/test-logs/federal-corpus-audits/owned-daemon-resource-control/disk-reservations.json"
STATE = Path("/tmp/ipfs-datasets-resource-scheduler-1000.json")
POLICY = "f07cb14289fcb6275e4497dcc36be0dcf8bfc3dc87531ff7a9537eb85fdcc661"
IMAGE = "sha256:a91a42c918514203bb6cfc7c38b320dfb68e9dfc0571bf01976c0db993e24e5e"
IMAGE_LABEL = "2026-10-16-ordinary-v1"
PYTHON = Path("/usr/bin/python3.12")
INNER_PYTHON = "/opt/ipfs-task-tools/bin/python"
PROFILE = "legal-ir-callable-dual-pipe-nonmodel-v04"
CAPSULE_BUDGET = TEMP_BUDGET = 128 * 1024 * 1024
CASES = ("within_cap", "over_cap")
OUTPUT_LIMITS = {"within_cap": 2 * 1024 * 1024, "over_cap": 64 * 1024}
EXPECTED_STDOUT = b"O" * (1024 * 1024) + b"A" * (32 * 4096)
EXPECTED_STDERR = b"E" * (32 * 1024) + b"S" * (32 * 1024)
NATIVE_NAMES = ("parented_phase_resources", "autoencoder_daemon_resources", "existing_resource_scheduler", "resource_scheduler",
                "runtime_telemetry", "proof_resource_safety", "validation_run_resources", "autoencoder_resource_recovery")
CUSTODY_NATIVE_NAMES = NATIVE_NAMES[:4]


def need(value, reason):
    if not value:
        raise RuntimeError(reason)


def read_stable(path, maximum=8 * 1024 * 1024, *, allow_empty=False):
    path = Path(path)
    need(path.is_absolute() and os.path.abspath(path) == str(path)
         and not any(item.is_symlink() for item in (path, *path.parents)), "canonical_unaliased_source_required")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_uid in (0, os.getuid()) and before.st_nlink == 1
             and before.st_mode & 0o002 == 0 and (0 <= before.st_size if allow_empty else 0 < before.st_size)
             and before.st_size <= maximum, "bounded_owned_source_required")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    witness = lambda m: (m.st_dev, m.st_ino, m.st_size, m.st_mode, m.st_uid, m.st_gid,
                         m.st_nlink, m.st_mtime_ns, m.st_ctime_ns)
    need(len(raw) == before.st_size and witness(before) == witness(after) == witness(path.lstat()),
         "source_changed_during_read")
    return raw


def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate_json_key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))


def save(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    need(len(raw) <= 2 * 1024 * 1024, "bounded_private_output_required")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(Path(path).parent)


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def progress(case, stage):
    print(json.dumps({"schema": "legal-ir-native-gate-progress@1", "fixture_case": case,
        "stage": stage, "observation_only": True, "models_called": 0, "canonical_tasks_accepted": 0}), flush=True)


def compact_case(result):
    fields = ("fixture_case", "passed", "failure_type", "failure_reason", "mechanical_helper_expected_refusal",
        "actual_returncode", "actual_observed_output_bytes", "actual_captured_output_bytes", "actual_stdout",
        "actual_stderr", "actual_pipe_readers_closed", "actual_borrowed_output_fds_open_after_helper",
        "host_globals_unchanged_by_helper", "payload_summary_verified", "cleanup", "actual_lifecycle_events",
        "actual_pre_effect_capacity_hold", "actual_post_cleanup_capacity_hold", "actual_capacity_acknowledged_after_cleanup")
    value = {field: result[field] for field in fields if field in result}
    path = RUN / "native" / result["fixture_case"] / "primary/mechanical-case.json"
    raw = read_stable(path, 2 * 1024 * 1024)
    value["private_case_file"] = {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                                  "written_before_owner_finalization": True}
    value["actual_owner_finalization"] = {}
    for label in ("primary", "watchdog"):
        public = result["actual_" + label + "_finalization"]
        native = public["resource_lease"]
        value["actual_owner_finalization"][label] = {"reservation_id": public["reservation_id"], "status": public["status"],
            "native_lease_id": native["lease_id"], "native_lease_released": native["released"]}
    return value


def verify_sources(request):
    pins = request["source_sha256"]
    need(type(pins) is dict and 16 <= len(pins) <= 128, "finite_source_pins_required")
    need(pins.get(str(ADMIN_AUDIT)) == ADMIN_AUDIT_SHA, "exact_reviewed_admin_capacity_source_audit_required")
    native_paths = {str(DATASETS / "ipfs_datasets_py/optimizers/logic_theorem_optimizer" / (name + ".py")) for name in NATIVE_NAMES}
    entry_paths = {str(DATASETS / name) for name in (
        "ipfs_datasets_py/__init__.py", "ipfs_datasets_py/router_deps.py",
        "ipfs_datasets_py/optimizers/__init__.py", "ipfs_datasets_py/optimizers/logic_theorem_optimizer/__init__.py")}
    custody_native = {str(DATASETS / "ipfs_datasets_py/optimizers/logic_theorem_optimizer" / (name + ".py")) for name in CUSTODY_NATIVE_NAMES}
    required = {str(Path(__file__).absolute()), str(HERE / "fixture_payload.py"), str(CUSTODY_SOURCE),
        str(RUNNER_MANIFEST), str(HANDOFF_MANIFEST), str(PREPARER_MANIFEST), str(CAPSULE / "SOURCE_MANIFEST.json"),
        str(CAPSULE.parent / "capsule-preparation.json"),
        request["preparation_root_observation_path"], "/usr/bin/docker", str(PYTHON)} | native_paths | entry_paths
    need(required <= set(pins), "complete_preimport_executed_source_closure_required")
    need(set(request.get("custody_source_sha256", {})) == custody_native | {str(CUSTODY_SOURCE)},
         "exact_native_custody_pins_required")
    for field in ("hook_source_sha256", "custody_source_sha256"):
        subset = request.get(field)
        need(type(subset) is dict and subset and all(pins.get(path) == digest for path, digest in subset.items()),
             "actual_hook_custody_source_closure_join_required")
    raw = read_stable(HERE / "SOURCE_MANIFEST.json", 1024 * 1024)
    need(hashlib.sha256(raw).hexdigest() == request["qualifier_manifest_sha256"], "qualifier_manifest_changed")
    manifest = strict(raw)
    need(manifest.get("schema") == "legal-ir-callable-dual-pipe-mechanical-source@1", "qualifier_source_schema_differs")
    for row in manifest["files"]:
        content = read_stable(HERE / row["path"])
        need(len(content) == row["bytes"] and hashlib.sha256(content).hexdigest() == row["sha256"]
             and pins.get(str(HERE / row["path"])) == row["sha256"], "complete_qualifier_source_closure_required")
    for spelling, expected in pins.items():
        need(type(spelling) is str and type(expected) is str and re.fullmatch("[0-9a-f]{64}", expected),
             "exact_source_pin_required")
        path = Path(spelling)
        need(path.is_absolute() and path.resolve(strict=True) == path
             and not any(p.is_symlink() for p in (path, *path.parents)), "unaliased_source_pin_required")
        before = path.stat()
        need(stat.S_ISREG(before.st_mode) and before.st_uid in (0, os.getuid()) and before.st_nlink == 1
             and before.st_mode & 0o002 == 0 and 0 < before.st_size <= 256 * 1024 * 1024,
             "bounded_trusted_source_pin_required")
        digest, count = hashlib.sha256(), 0
        witness = lambda m: (m.st_dev, m.st_ino, m.st_size, m.st_mode, m.st_uid, m.st_gid, m.st_nlink, m.st_mtime_ns, m.st_ctime_ns)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            need(witness(opened) == witness(before), "opened_source_pin_identity_changed")
            while chunk := stream.read(1024 * 1024):
                count += len(chunk)
                need(count <= opened.st_size and count <= 256 * 1024 * 1024, "source_pin_grew_beyond_bound")
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        need(count == opened.st_size and witness(before) == witness(opened) == witness(after) == witness(path.lstat())
             and digest.hexdigest() == expected, "source_pin_changed")


def verify_capsule(request):
    raw = read_stable(CAPSULE / "SOURCE_MANIFEST.json", 2 * 1024 * 1024)
    need(hashlib.sha256(raw).hexdigest() == request["capsule_manifest_sha256"], "actual_prepared_capsule_manifest_differs")
    # Root-selected exact readiness observation is provenance, never admission.
    readiness = read_stable(Path(request["preparation_root_observation_path"]), 2 * 1024 * 1024)
    need(hashlib.sha256(readiness).hexdigest() == request["preparation_root_observation_sha256"],
         "actual_preparation_root_observation_changed")
    root_observation = strict(readiness)
    need(root_observation.get("schema") == "legal-ir-callable-capsule-root-observation@1"
         and root_observation.get("actual_exec_exit_code") == 0
         and root_observation.get("original_native_preparation_finalized") is True
         and root_observation.get("source_manifest_sha256") == PREPARER_MANIFEST_SHA
         and root_observation.get("source_request_sha256") == PREPARER_REQUEST_SHA,
         "actual_original_preparation_closeout_correspondence_required")
    terminal = root_observation.get("actual_native_terminal", {})
    need(terminal.get("status") == "completed" and terminal.get("native_finalized") is True
         and terminal.get("child_returncode") == 0 and terminal.get("failure_type") is None,
         "actual_original_preparation_terminal_required")
    preparation_result = read_stable(CAPSULE.parent / "capsule-preparation.json", 2 * 1024 * 1024)
    need(root_observation.get("preparation_result_sha256") == hashlib.sha256(preparation_result).hexdigest()
         and strict(preparation_result).get("capsule_manifest_sha256") == request["capsule_manifest_sha256"]
         == root_observation.get("capsule_manifest_sha256"), "actual_prepared_capsule_result_join_required")
    need(hashlib.sha256(read_stable(PREPARER_MANIFEST, 1024 * 1024)).hexdigest() == PREPARER_MANIFEST_SHA,
         "reviewed_preparation_source_manifest_changed")
    manifest = strict(raw)
    need(manifest.get("schema") == "legal-ir-callable-handoff-capsule@1"
         and manifest.get("capsule_root") == str(CAPSULE)
         and manifest.get("runner_source_manifest_sha256") == RUNNER_MANIFEST_SHA
         and manifest.get("handoff_source_manifest_sha256") == HANDOFF_MANIFEST_SHA
         and manifest.get("base_capsule_manifest_sha256") == BASE_CAPSULE_SHA
         and manifest.get("payload_file_count") == 2807, "actual_capsule_provenance_differs")
    overrides = {}
    for path, expected in ((RUNNER_MANIFEST, RUNNER_MANIFEST_SHA), (HANDOFF_MANIFEST, HANDOFF_MANIFEST_SHA)):
        frozen_raw = read_stable(path, 1024 * 1024)
        need(hashlib.sha256(frozen_raw).hexdigest() == expected, "frozen_successor_source_manifest_changed")
        for row in strict(frozen_raw)["files"]:
            if row["path"].startswith("proposed/") and row["path"].endswith(".py"):
                overrides[row["path"][len("proposed/"):]] = row
    need(set(overrides) == {"ipfs_accelerate_py/agent_supervisor/runtime/" + name for name in
         ("grok_cli_runner.py", "ordinary_success_handoff.py", "ordinary_custody_hold.py")},
         "exact_three_callable_source_overrides_required")
    rows, seen, total = manifest.get("files"), set(), 0
    need(type(rows) is list and len(rows) == 2807, "complete_callable_python_capsule_required")
    for row in rows:
        need(type(row) is dict and set(row) == {"path", "bytes", "sha256"}, "closed_capsule_row_required")
        spelling = row["path"]
        need(type(spelling) is str, "capsule_relative_path_required")
        relative = PurePosixPath(spelling)
        need(not relative.is_absolute() and ".." not in relative.parts and relative.as_posix() == spelling
             and spelling.endswith(".py") and spelling not in seen, "unique_capsule_python_path_required")
        need(type(row["bytes"]) is int and row["bytes"] >= 0, "capsule_declared_size_required")
        content = read_stable(CAPSULE / spelling, allow_empty=row["bytes"] == 0)
        need(stat.S_IMODE((CAPSULE / spelling).lstat().st_mode) == 0o444,
             "actual_readonly_capsule_file_required")
        need(len(content) == row["bytes"] and hashlib.sha256(content).hexdigest() == row["sha256"], "capsule_source_changed")
        if spelling in overrides:
            need(row["sha256"] == overrides[spelling]["sha256"] and row["bytes"] == overrides[spelling]["bytes"],
                 "capsule_frozen_callable_bytes_differ")
        total += len(content)
        need(total <= CAPSULE_BUDGET, "capsule_source_budget_exceeded")
        seen.add(spelling)
    population = set()
    for index, path in enumerate(CAPSULE.rglob("*")):
        need(index < 8192 and not path.is_symlink(), "bounded_unaliased_capsule_population_required")
        if not path.is_dir(): population.add(path.relative_to(CAPSULE).as_posix())
        else: need(stat.S_IMODE(path.lstat().st_mode) == 0o500, "actual_readonly_capsule_directory_required")
    need(total == manifest.get("payload_bytes") and population == seen | {"SOURCE_MANIFEST.json"}
         and stat.S_IMODE(CAPSULE.lstat().st_mode) == 0o500, "capsule_population_or_size_changed")
    return manifest


def check_exact_environment(actual, expected):
    """Pure exact unique bindings; raw order is preserved in private snapshots.

    This permits only ordering differences. Extras, missing keys, duplicates
    (including identical duplicates), malformed entries and changed values
    refuse before fixture startup. A mapping grants no execution authority.
    """
    def parse(entries):
        need(type(entries) is list and len(entries) <= 32, "actual_fixture_environment_list_required")
        result = {}
        for entry in entries:
            need(type(entry) is str and 0 < len(entry) <= 4096 and "=" in entry,
                 "actual_fixture_environment_entry_malformed")
            key, value = entry.split("=", 1)
            need(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is not None and key not in result,
                 "actual_fixture_environment_duplicate_or_malformed_key")
            result[key] = value
        return result
    need(parse(actual) == parse(expected), "actual_fixture_config_differs:Env")


def process_group_dead(original_pid):
    """Independent bounded actual /proc census, never native grant/receipt."""
    need(type(original_pid) is int and original_pid > 1, "actual_owned_group_pid_required")
    entries = list(Path("/proc").iterdir())
    need(len(entries) <= 65536, "bounded_process_census_required")
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            descriptor = os.open(entry / "stat", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as stream:
                raw = stream.read(4097)
        except FileNotFoundError:
            continue
        need(len(raw) <= 4096 and b") " in raw, "actual_process_census_unavailable")
        fields = raw.rsplit(b") ", 1)[1].split()
        need(len(fields) >= 20, "actual_process_stat_unavailable")
        if int(fields[2]) == original_pid and fields[0] != b"Z":
            return False
    return True


class Fixture:
    def __init__(self, G, H, C, hold, owner, watchdog_owner, request, case):
        self.G, self.H, self.owner, self.watchdog_owner = G, H, owner, watchdog_owner
        self.C, self.hold = C, hold
        self.live_capacity_observations = []
        self.request, self.case = request, case
        self.temp = RUN / ("native/temporary-" + case)
        self.workspace = RUN / ("native/workspace-" + case)
        self.lease = self.process = self.fence = self.lifecycle = None
        self.close_requested = False
        self.created_known = False
        self.create_started = False
        self.cid = None
        self.daemon_id = request["expected_daemon_id"]
        self.runtime = None
        self.expected = None
        self.home = self.prompt = None
        self.home_identity = self.prompt_identity = None
        self.home_owner = self.invocation = self.output = None

    def query(self, arguments, maximum=512 * 1024):
        code, raw, error = self.G._bounded_docker_query(
            ["/usr/bin/docker", "--host=unix:///var/run/docker.sock", "--config", "/nonexistent", *arguments], timeout=5)
        need(code == 0 and len(raw) + len(error) <= maximum, "actual_bounded_docker_query_unavailable")
        return raw

    def daemon(self):
        need(self.query(["info", "--format", "{{.DockerRootDir}}|{{.OSType}}|{{.Architecture}}|{{.ID}}"], 4096).strip()
             == ("/var/lib/docker|linux|aarch64|" + self.daemon_id).encode(), "actual_system_daemon_changed")

    def guard(self):
        for owner in (self.owner, self.watchdog_owner):
            need(owner.heartbeat(), "actual_owner_heartbeat_failed")
            owner.check_usage(owner.directory)
        if self.lease is not None and not self.close_requested:
            need(self.lease._watchdog.poll() is None, "actual_watchdog_died_during_fixture")
        if self.process is not None and self.process.poll() is None and self.fence is not None:
            observed = self.C.observe_ordinary_custody_hold(self.hold)
            need(observed["capacity_records_retained"], "actual_live_fixture_capacity_hold_missing")
            if len(self.live_capacity_observations) < 4:
                self.live_capacity_observations.append(observed)

    def inspect(self, identifier):
        self.daemon()
        rows = strict(self.query(["container", "inspect", identifier]))
        need(type(rows) is list and len(rows) == 1 and type(rows[0]) is dict, "actual_single_fixture_container_required")
        return rows[0]

    def check_container(self, row):
        expected = self.expected
        need(row.get("Id") == self.cid and row.get("Image") == IMAGE
             and row.get("Name") == "/" + self.lease.container_name, "actual_fixture_container_identity_differs")
        config, host = row.get("Config", {}), row.get("HostConfig", {})
        for key, value in expected["config"].items():
            if key == "Env":
                check_exact_environment(config.get(key), value)
            else:
                need(config.get(key) == value, "actual_fixture_config_differs:" + key)
        required = {"NetworkMode": "none", "ReadonlyRootfs": True, "Privileged": False,
            "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"], "PidsLimit": 4,
            "Memory": 268435456, "MemorySwap": 268435456, "NanoCpus": 1000000000,
            "Runtime": "runc", "PidMode": "", "IpcMode": "none", "CgroupnsMode": "private",
            "AutoRemove": False, "PublishAllPorts": False}
        for key, value in required.items():
            need(host.get(key) == value, "actual_fixture_host_differs:" + key)
        for key in ("CapAdd", "Devices", "DeviceRequests", "DeviceCgroupRules", "VolumesFrom", "Links", "GroupAdd"):
            need(host.get(key) in (None, []), "unexpected_fixture_authority:" + key)
        need(host.get("PortBindings") in (None, {}) and host.get("Tmpfs") in (None, {})
             and host.get("LogConfig", {}).get("Type") == "none"
             and host.get("RestartPolicy", {}).get("Name") == "no", "unexpected_fixture_ports_storage_restart")
        need(sorted(host.get("Ulimits", []), key=lambda row: row["Name"]) == [
            {"Name": "cpu", "Soft": 120, "Hard": 120},
            {"Name": "fsize", "Soft": 2097152, "Hard": 2097152},
            {"Name": "nofile", "Soft": 64, "Hard": 64}], "actual_fixture_ulimits_differ")
        mounts = row.get("Mounts")
        need(type(mounts) is list and len(mounts) == len(expected["mounts"]), "actual_fixture_mount_count_differs")
        need(sorted((m.get("Type"), m.get("Source"), m.get("Destination"), m.get("RW")) for m in mounts)
             == sorted(("bind", source, target, not readonly) for source, target, readonly in expected["mounts"])
             and all(m.get("Propagation") == "rprivate" for m in mounts), "actual_fixture_mounts_differ")

    def setup(self):
        self.runtime = self.G._docker_runtime_receipt("/usr/bin/docker")
        self.daemon()
        image = strict(self.query(["image", "inspect", IMAGE]))[0]
        need(image.get("Id") == IMAGE and image.get("Os") == "linux" and image.get("Architecture") == "arm64"
             and image.get("Config", {}).get("Labels", {}).get("org.ipfs-accelerate.ordinary-worker") == IMAGE_LABEL
             and not image.get("Config", {}).get("Volumes")
             and all(type(value) is str and value.startswith("PATH=") for value in image["Config"].get("Env", [])),
             "actual_pinned_fixture_image_differs")
        self.home_owner = tempfile.TemporaryDirectory(prefix="asref-grok-home-", dir=self.temp)
        home = Path(self.home_owner.name)
        descriptor, spelling = tempfile.mkstemp(prefix="asref-grok-prompt-", dir=self.temp)
        os.close(descriptor)
        self.home, self.prompt = home, Path(spelling)
        self.home_identity = (home.stat().st_dev, home.stat().st_ino)
        self.prompt_identity = (self.prompt.stat().st_dev, self.prompt.stat().st_ino)
        self.H.ordinary_private_allocations_created(self.home_owner, self.prompt)
        self.lease = self.G._LegacyDockerContainerLease.create("/usr/bin/docker", provider="grok",
            provider_home=home, prompt_path=Path(spelling), temporary_root=self.temp,
            runner_entry=Path(self.G.__file__).resolve())
        if self.lifecycle is None:
            self.watchdog_owner.check_usage(self.watchdog_owner.directory, child_pid=self.lease._watchdog.pid, full=True)
        labels = dict(image["Config"].get("Labels", {}), **{
            "ipfs_accelerate.grok_isolation": "true", "org.justicedao.nonmodel-fixture": PROFILE,
            "org.justicedao.fixture-case": self.case})
        # grok_isolation is the actual legacy confinement-profile label only.
        # This command never mounts or invokes Grok, auth or provider code.
        env = ["PATH=/usr/bin:/bin", "HOME=/workspace", "TMPDIR=/workspace", "LANG=C.UTF-8",
               "LC_ALL=C.UTF-8", "PYTHONDONTWRITEBYTECODE=1"]
        mounts = [("/usr", "/usr", True), (str(PYTHON), INNER_PYTHON, True),
                  (str(HERE / "fixture_payload.py"), "/opt/legal-ir/fixture_payload.py", True),
                  (str(self.workspace), "/workspace", False)]
        command = ["/usr/bin/docker", "--host=unix:///var/run/docker.sock", "--config", str(self.lease.docker_config),
            "create", "--pull=never", "--interactive", "--name", self.lease.container_name,
            "--cidfile", str(self.lease.cidfile), "--network=none", "--ipc=none", "--read-only", "--cap-drop=ALL",
            "--security-opt=no-new-privileges", "--runtime=runc", "--cgroupns=private", "--pids-limit=4", "--cpus=1",
            "--memory=268435456", "--memory-swap=268435456", "--restart=no", "--log-driver=none",
            "--user", f"{os.getuid()}:{os.getgid()}", "--workdir=/workspace", "--entrypoint=" + INNER_PYTHON,
            "--ulimit=nofile=64:64", "--ulimit=fsize=2097152:2097152", "--ulimit=cpu=120:120"]
        for name, value in labels.items():
            command.extend(["--label", name + "=" + value])
        for value in env:
            command.extend(["--env", value])
        for source, target, readonly in mounts:
            need(Path(source).resolve(strict=True) == Path(source)
                 and not any(p.is_symlink() for p in (Path(source), *Path(source).parents)), "unaliased_fixture_mount_required")
            command.extend(["--mount", "type=bind,src=" + source + ",dst=" + target + (",readonly" if readonly else "")])
        command.extend([IMAGE, "-I", "-S", "-B", "/opt/legal-ir/fixture_payload.py", self.case])
        command = self.H.prepare_ordinary_gated_create(command, image=IMAGE,
            original_entrypoint=INNER_PYTHON, python_entrypoint=INNER_PYTHON) if self.lifecycle is not None else self.H.gated_create_command(
            command, image=IMAGE, original_entrypoint=INNER_PYTHON, python_entrypoint=INNER_PYTHON)
        self.expected = {"config": {"User": f"{os.getuid()}:{os.getgid()}", "WorkingDir": "/workspace",
            "Entrypoint": [INNER_PYTHON], "Cmd": command[command.index(IMAGE) + 1:],
            "Env": env, "Labels": labels}, "mounts": mounts}
        self.guard()
        self.create_started = True
        start = self.G._create_grok_container_and_build_start_command(command, workspace=self.workspace,
            docker_environment=self.G._docker_control_env(), docker_lease=self.lease)
        self.cid = self.lease.cidfile.read_text(encoding="ascii").strip()
        need(re.fullmatch("[0-9a-f]{64}", self.cid), "actual_cidfile_required")
        self.created_known = True
        row = self.inspect(self.cid)
        # Retain the bounded actual observation even if profile validation
        # refuses. This is private diagnostic evidence, not a passed profile.
        save(self.owner.directory / "created-container.json", row)
        self.check_container(row)
        need(row.get("State", {}).get("Status") == "created" and row["State"].get("Pid") == 0,
             "actual_created_inert_fixture_required")
        progress(self.case, "actual_created_inert_profile_checked")
        return start

    def cleanup(self):
        if self.lifecycle is not None:
            if self.lease is None and self.lifecycle.allocated_leases:
                self.lease = self.lifecycle.allocated_leases[0]
            if self.process is None and self.lifecycle.processes:
                self.process = self.lifecycle.processes[0]
            if self.fence is None:
                self.fence = self.lifecycle.fence
        if self.lease is None:
            # A factory can refuse before returning/recording a lease. Only
            # these actual freshly allocated EMPTY private paths may be removed;
            # None is never a blanket stopped/cleanup proof.
            need(not self.create_started and self.process is None, "no_lease_but_possible_fixture_effect")
            for path, identity, directory in ((self.home, self.home_identity, True),
                                               (self.prompt, self.prompt_identity, False)):
                if path is None or not os.path.lexists(path):
                    continue
                meta = path.lstat()
                need(path.parent == self.temp and (meta.st_dev, meta.st_ino) == identity
                     and meta.st_uid == os.getuid() and not stat.S_ISLNK(meta.st_mode)
                     and meta.st_mode & 0o077 == 0, "original_unstarted_private_path_identity_changed")
                if directory:
                    need(stat.S_ISDIR(meta.st_mode) and not any(path.iterdir()), "unstarted_fixture_home_not_empty")
                    path.rmdir()
                else:
                    need(stat.S_ISREG(meta.st_mode) and meta.st_nlink == 1 and meta.st_size == 0,
                         "unstarted_fixture_prompt_not_empty")
                    path.unlink()
            need(all(path is None or not os.path.lexists(path) for path in (self.home, self.prompt)),
                 "unstarted_fixture_private_paths_remain")
            if self.home_owner is not None:
                self.home_owner.cleanup()
            return {"actual_allocation_observed": False, "cleanup_known": True, "known_unstarted_home_prompt_absent": True}
        next_notice = 0
        while True:
            try:
                self.daemon()
                # Unknown create must first resolve its actual full inert
                # profile. Do not erase the real CID/config before resolution.
                if self.create_started and not self.created_known:
                    row = self.inspect(self.lease.container_name)
                    self.cid = row.get("Id")
                    need(type(self.cid) is str and re.fullmatch("[0-9a-f]{64}", self.cid), "late_actual_create_cid_unknown")
                    self.check_container(row)
                    need(row.get("State", {}).get("Running") is False, "late_create_unexpected_execution")
                    self.created_known = True
                valid_fence = (type(self.fence) is dict and type(self.fence.get("init_pid")) is int
                               and self.fence["init_pid"] > 1 and bool(self.fence.get("kernel_scope")))
                if self.process is not None and not valid_fence and not self.close_requested:
                    # A failed immediate H.started observation remains a
                    # failed positive case. Capture the still-blocked real
                    # init now only for stopped cleanup, never retry startup.
                    self.fence = self.G._attest_exact_docker_execution(docker_bin="/usr/bin/docker",
                        docker_config=self.lease.docker_config, provider="grok", container_name=self.lease.container_name,
                        container_id=self.cid, image_id=IMAGE, timeout=2)
                    need(self.fence.get("init_pid", 0) > 1 and self.fence.get("kernel_scope"),
                         "actual_failed_start_cleanup_scope_not_observed")
                if not self.close_requested:
                    self.close_requested = True
                    # Actual factory may already have closed after observer
                    # refusal. Never reclose that same allocation blindly.
                    if not self.lease._closed:
                        self.lease.close(docker_run_finished=self.process is not None and self.process.poll() is not None)
                if self.process is not None:
                    try:
                        self.process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        raise RuntimeError("actual_start_transport_still_live")
                absent_name = not self.query(["container", "ls", "--all", "--no-trunc", "--filter",
                    "name=^/" + self.lease.container_name + "$", "--format", "{{.Names}}"], 4096).strip()
                absent_id = self.cid is None or not self.query(["container", "ls", "--all", "--no-trunc", "--filter",
                    "id=" + self.cid, "--format", "{{.ID}}"], 4096).strip()
                scope_quiet = self.fence is None or self.G._docker_termination_scope_quiescent(self.fence)
                # A release marker was never written without a captured fence.
                if self.lifecycle is not None and "provider_start_released" in self.lifecycle.events:
                    need(self.fence is not None, "released_fixture_scope_missing")
                watchdog_dead = self.lease._watchdog.poll() is not None and process_group_dead(self.lease._watchdog.pid)
                transport_dead = self.process is None or self.process.poll() is not None and process_group_dead(self.process.pid)
                paths_gone = all(not os.path.lexists(path) for path in
                    (self.lease.lease_root, self.lease.provider_home, self.lease.prompt_path))
                need(absent_name and absent_id and scope_quiet and watchdog_dead and transport_dead and paths_gone
                     and self.G._docker_runtime_receipt("/usr/bin/docker") == self.runtime,
                     "actual_cleanup_not_yet_independently_quiescent")
                self.daemon()
                if self.home_owner is not None:
                    self.home_owner.cleanup()
                return {"actual_allocation_observed": True, "cleanup_known": True,
                    "container_absent_by_id_and_name": True, "kernel_scope_quiescent": True,
                    "actual_kernel_scope_captured": self.fence is not None and self.fence.get("init_pid", 0) > 1 and bool(self.fence.get("kernel_scope")),
                    "actual_watchdog_group_dead": True, "actual_transport_group_dead": True,
                    "lease_home_prompt_absent": True, "preserve_for_recovery_reported": self.lease.preserve_for_recovery}
            except BaseException as error:
                # Observation only after the one actual close. All original
                # native contexts and actual object references stay on stack.
                # No receipt/adoption/second close/PID guess/native grant.
                if time.monotonic() >= next_notice:
                    print(json.dumps({"status": "actual_fixture_native_custody_held_cleanup_unknown",
                        "fixture_case": self.case, "reason": type(error).__name__, "container_id": self.cid,
                        "container_name": self.lease.container_name, "models_called": 0,
                        "canonical_tasks_accepted": 0, "native_release_permitted": False}), flush=True)
                    next_notice = time.monotonic() + 30
                try:
                    time.sleep(1)
                except BaseException:
                    pass


def host_globals():
    # Never export ambient environment values, only an equality digest.
    return {"cwd": os.getcwd(), "environment_sha256": hashlib.sha256(
        json.dumps(dict(os.environ), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "tempfile_tempdir": str(tempfile.tempdir),
        "stdin_identity": id(sys.stdin), "stdout_identity": id(sys.stdout), "stderr_identity": id(sys.stderr)}


def run_case(G, H, C, phase, owner_type, request, case):
    progress(case, "case_begin")
    base = RUN / ("native/" + case)
    base.mkdir(mode=0o700)
    primary_dir, watchdog_dir = base / "primary", base / "watchdog"
    primary_dir.mkdir(mode=0o700)
    watchdog_dir.mkdir(mode=0o700)
    fsync_directory(base)
    fsync_directory(base.parent)
    result = {"fixture_case": case, "fixture_only": True, "scope": "synthetic_nonmodel_dual_pipe_mechanics_only",
        "models_called": 0, "provider_calls_performed": False, "canonical_tasks_accepted": 0,
        "acceptance_authority": False, "public_wrapper_exercised": False,
        "full_runner_exercised": False, "finish_or_success_callback_called": False,
        "actual_provider_execution_qualified": False, "immediate_start_race_preserved": True,
        "output_limit_bytes": OUTPUT_LIMITS[case]}
    with owner_type(LEDGER, roots=ROOTS, attempt_directory=primary_dir, parent_lease=phase,
            storage_bytes=512 * 1024 * 1024, memory_mb=1024, cpu_slots=1, child_process_slots=8,
            scheduler_state_path=STATE, expected_scheduler_policy_sha256=POLICY,
            timeout_seconds=request["duration_seconds"], ledger_lock_timeout_seconds=None) as primary:
        with owner_type(LEDGER, roots=ROOTS, attempt_directory=watchdog_dir, parent_lease=phase,
                storage_bytes=128 * 1024 * 1024, memory_mb=512, cpu_slots=1, child_process_slots=4,
                scheduler_state_path=STATE, expected_scheduler_policy_sha256=POLICY,
                timeout_seconds=request["duration_seconds"], ledger_lock_timeout_seconds=None) as watchdog:
            declarations = [{"path": str(CAPSULE), "bytes": CAPSULE_BUDGET},
                            {"path": str(RUN / ("native/temporary-" + case)), "bytes": TEMP_BUDGET},
                            {"path": str(RUN / ("native/workspace-" + case)), "bytes": TEMP_BUDGET}]
            for declaration in declarations:
                primary.account_external_bytes(hashlib.sha256(declaration["path"].encode()).hexdigest(), declaration["bytes"])
            H._guard_owner(primary)
            H._guard_owner(watchdog, watchdog=True)
            H.check_owner_pair_projection(primary.to_dict(), watchdog.to_dict())
            H.check_external_storage_declarations(primary.to_dict(), declarations)
            H._verify_sources(primary, request["hook_source_sha256"], Path(G.__file__).absolute())
            consumers, hold = [], None
            try:
                for owner in (primary, watchdog):
                    consumers.append(owner.native_lease.acquire_child(cpu_slots=1, memory_mb=1,
                        child_process_slots=0, requires_gpu=False, timeout=30,
                        request_id=PROFILE + ":" + case + ":custody:" + str(len(consumers))))
                hold = C.bind_ordinary_custody_hold(primary_owner=primary, watchdog_owner=watchdog,
                    primary_consumer=consumers[0], watchdog_consumer=consumers[1],
                    protected_source_sha256=request["custody_source_sha256"])
            except BaseException:
                for consumer in reversed(consumers): consumer.release()
                raise
            fixture = Fixture(G, H, C, hold, primary, watchdog, request, case)
            # Strong original references survive unknown cleanup/caller unwind.
            C.retain_ordinary_custody_references(hold, (fixture,))
            result["actual_pre_effect_capacity_hold"] = C.observe_ordinary_custody_hold(hold)
            need(result["actual_pre_effect_capacity_hold"]["capacity_records_retained"], "actual_pre_effect_hold_missing")
            progress(case, "actual_pre_effect_capacity_hold_bound")
            descriptors = []
            token = None
            try:
                fixture.temp.mkdir(mode=0o700)
                fixture.workspace.mkdir(mode=0o700)
                stdout_path, stderr_path = primary.directory / "transport-stdout.bin", primary.directory / "transport-stderr.bin"
                for path in (stdout_path, stderr_path):
                    descriptors.append(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600))
                environment = {"PATH": "/usr/bin:/bin", "HOME": str(fixture.temp),
                    "TMPDIR": str(fixture.temp), "TMP": str(fixture.temp), "TEMP": str(fixture.temp)}
                invocation = G.OrdinaryCallableInvocation(prompt="synthetic nonmodel mechanics; never dispatched to a model",
                    cwd=fixture.workspace, environment=environment, temporary_root=fixture.temp,
                    stdout_path=stdout_path, stderr_path=stderr_path,
                    stdout_fd=descriptors[0], stderr_fd=descriptors[1], output_limit_bytes=OUTPUT_LIMITS[case])
                fixture.invocation, fixture.output = invocation, G._CallableOutput(invocation)
                lifecycle = fixture.lifecycle = H._Lifecycle(primary, watchdog, dict(request["hook_source_sha256"]), G,
                    uuid.uuid4().hex, declarations)
                lifecycle.invocation, lifecycle.custody_hold, lifecycle.custody_module = invocation, hold, C
                # Original retained fixture already strongly owns these exact attributes.
                token = H._ACTIVE.set(lifecycle)
                # Existing lifecycle vocabulary selects only the legacy gate
                # mechanics. No actual Grok/ProviderSuccess is made or emitted.
                H.select_ordinary_route(None, "grok-4.7")
                H.begin_ordinary_task()
                command = fixture.setup()
                before = host_globals()
                try:
                    code, retained, total, envelope_overflow = G._run_grok_with_bounded_stderr(command,
                        env=G._docker_control_env(), invocation=invocation, output=fixture.output)
                except G.OrdinaryCallableRefusal as error:
                    need(case == "over_cap" and str(error) == "callable_received_output_limit_exceeded",
                         "actual_helper_refused_unexpected_reason:" + str(error))
                    result["mechanical_helper_expected_refusal"] = str(error)
                    code, retained, total, envelope_overflow = lifecycle.returncode, b"", len(EXPECTED_STDERR), False
                else:
                    need(case == "within_cap", "actual_over_cap_helper_did_not_refuse")
                    need(code == 0 and not envelope_overflow and retained == EXPECTED_STDERR and total == len(EXPECTED_STDERR),
                         "actual_within_cap_stderr_differs")
                    # Mirror only the actual ordinary helper caller's bounded
                    # retained-stderr write; no full _run/finish/callback path.
                    fixture.output.write("stderr", retained)
                after = host_globals()
                fixture.process, fixture.fence = lifecycle.processes[0], lifecycle.fence
                need(before == after, "actual_helper_mutated_host_globals")
                need(code == 0 and fixture.process.poll() == 0 and fixture.process.returncode == 0,
                     "actual_nonmodel_transport_exit_differs")
                need(fixture.process.stdout.closed and fixture.process.stderr.closed,
                     "actual_dual_pipe_readers_not_closed")
                for descriptor in descriptors: os.fstat(descriptor)
                result.update(actual_returncode=code, actual_pipe_readers_closed=True,
                    actual_borrowed_output_fds_open_after_helper=True, host_globals_unchanged_by_helper=True,
                    actual_observed_output_bytes=fixture.output.received,
                    actual_captured_output_bytes=fixture.output.written)
                need(fixture.output.received == len(EXPECTED_STDOUT) + len(EXPECTED_STDERR),
                     "actual_aggregate_output_count_differs")
                for descriptor in descriptors: os.fsync(descriptor)
                captured_stdout = read_stable(stdout_path, 2 * 1024 * 1024, allow_empty=True)
                captured_stderr = read_stable(stderr_path, 2 * 1024 * 1024, allow_empty=True)
                if case == "within_cap":
                    need(captured_stdout == EXPECTED_STDOUT and captured_stderr == EXPECTED_STDERR
                         and fixture.output.written == fixture.output.received, "actual_within_cap_capture_differs")
                else:
                    need(len(captured_stdout) + len(captured_stderr) <= OUTPUT_LIMITS[case]
                         and captured_stdout == EXPECTED_STDOUT[:len(captured_stdout)] and captured_stderr == b"",
                         "actual_over_cap_capture_exceeds_bound_or_differs")
                result["actual_stdout"] = {"path": str(stdout_path), "bytes": len(captured_stdout),
                    "sha256": hashlib.sha256(captured_stdout).hexdigest()}
                result["actual_stderr"] = {"path": str(stderr_path), "bytes": len(captured_stderr),
                    "sha256": hashlib.sha256(captured_stderr).hexdigest()}
                payload = strict(read_stable(fixture.workspace / "payload-summary.json", 4096))
                expected_payload = {"schema": "legal-ir-nonmodel-dual-pipe-payload@1", "fixture_case": case,
                    "stdout_bytes": len(EXPECTED_STDOUT), "stdout_sha256": hashlib.sha256(EXPECTED_STDOUT).hexdigest(),
                    "stderr_bytes": len(EXPECTED_STDERR), "stderr_sha256": hashlib.sha256(EXPECTED_STDERR).hexdigest(),
                    "models_called": 0, "provider_calls_performed": False, "canonical_tasks_accepted": 0}
                need(payload == expected_payload, "actual_nonmodel_payload_summary_differs")
                save(primary.directory / "payload-summary.json", payload)
                result["payload_summary_verified"] = True
                progress(case, "actual_dual_pipe_helper_returned")
                row = fixture.inspect(fixture.cid)
                save(primary.directory / "actual-exited-container.json", row)
                fixture.check_container(row)
                need(row.get("State", {}).get("Running") is False and row["State"].get("ExitCode") == 0
                     and row["State"].get("OOMKilled") is False and row["State"].get("Error") == "",
                     "actual_fixture_container_exit_differs")
                result["passed"] = True
            except BaseException as error:
                result.update(passed=False, failure_type=type(error).__name__, failure_reason=str(error)[:2048])
            finally:
                # Original direct native contexts and C hold remain live until
                # independent same-daemon/container/kernel/group/path proof.
                result["cleanup"] = fixture.cleanup()
                progress(case, "actual_cleanup_independently_observed")
                if fixture.lifecycle is not None:
                    result["actual_lifecycle_events"] = list(fixture.lifecycle.events)
                    result["actual_creation_intent_count"] = len(fixture.lifecycle.creation_intents)
                    result["actual_known_created_count"] = len(fixture.lifecycle.known_created)
                if token is not None: H._ACTIVE.reset(token)
                for descriptor in descriptors:
                    os.fsync(descriptor)
                    os.close(descriptor)
            result["actual_live_component_capacity_observations"] = fixture.live_capacity_observations
            if result["passed"]:
                expected_events = ["invocation_started", "route_selected", "task_started", "watchdog_native_attached", "lease_created",
                    "transport_started", "native_child_attached", "container_scope_captured", "provider_start_released", "provider_reaped", "lease_closed"]
                if result["actual_lifecycle_events"] != expected_events:
                    result.update(passed=False, failure_reason="actual_dual_pipe_lifecycle_prefix_differs")
            result["actual_post_cleanup_capacity_hold"] = C.observe_ordinary_custody_hold(hold)
            need(result["actual_post_cleanup_capacity_hold"]["capacity_records_retained"], "actual_capacity_released_before_cleanup")
            for consumer in C.held_native_consumers(hold): consumer.release()
            result["actual_capacity_acknowledged_after_cleanup"] = C.forget_acknowledged_custody_hold(hold)
            result["native_owner_inventory_before_finalization"] = {"primary": primary.to_dict(), "watchdog": watchdog.to_dict()}
            verify_sources(request)
            verify_capsule(request)
            need(primary.check_usage(primary.directory, full=True)["group_rss"]["live_processes"] == 0
                 and watchdog.check_usage(watchdog.directory, full=True)["group_rss"]["live_processes"] == 0,
                 "actual_native_groups_not_quiescent_before_finalization")
            save(primary.directory / "mechanical-case.json", result)
            fsync_directory(primary.directory)
            fsync_directory(watchdog.directory)
            result["actual_watchdog_finalization"] = watchdog.finalize(watchdog.directory, artifacts_durable=True)
            result["actual_primary_finalization"] = primary.finalize(primary.directory, artifacts_durable=True)
            need(result["actual_watchdog_finalization"]["status"] == "released"
                 and result["actual_primary_finalization"]["status"] == "released", "actual_disk_finalization_incomplete")
            progress(case, "actual_case_owners_finalized")
    result["both_native_owner_contexts_exited_after_actual_cleanup"] = True
    return result


def main():
    need(len(sys.argv) == 3 and re.fullmatch("[0-9a-f]{64}", sys.argv[2]), "request_path_and_exact_sha_required")
    request_raw = read_stable(Path(sys.argv[1]).absolute(), 2 * 1024 * 1024)
    need(hashlib.sha256(request_raw).hexdigest() == sys.argv[2], "reviewed_request_changed")
    request = strict(request_raw)
    need(request.get("schema") == "legal-ir-callable-dual-pipe-mechanical-request@1"
         and request.get("capsule_root") == str(CAPSULE)
         and request.get("runner_source_manifest_sha256") == RUNNER_MANIFEST_SHA
         and request.get("handoff_source_manifest_sha256") == HANDOFF_MANIFEST_SHA
         and type(request.get("duration_seconds")) is int and 60 <= request["duration_seconds"] <= 3600
         and request.get("cases") == list(CASES)
         and request.get("resource_envelope", {}).get("root_phase_process_slots") == 13
         and request["resource_envelope"].get("actual_owner_process_slots") == [8, 4]
         and request["resource_envelope"].get("trusted_controller_admin_process_slots") == 1
         and request.get("admin_source_audit_sha256") == ADMIN_AUDIT_SHA
         and request.get("admin_native_child_attachment_qualified") is False
         and request.get("hard_admin_quota") is False
         and type(request.get("preparation_root_observation_path")) is str
         and request["preparation_root_observation_path"] == str(PREPARATION_OBSERVATION)
         and all(type(request.get(name)) is str and re.fullmatch("[0-9a-f]{64}", request[name]) for name in
             ("capsule_manifest_sha256", "preparation_root_observation_sha256", "qualifier_manifest_sha256"))
         and type(request.get("expected_daemon_id")) is str
         and re.fullmatch("[0-9a-f-]{36}", request["expected_daemon_id"]), "bounded_isolated_actual_request_required")
    verify_sources(request)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", IPFS_DATASETS_PY_MINIMAL_IMPORTS="1", IPFS_ACCEL_SKIP_CORE="1", IPFS_AUTO_INSTALL="false",
        IPFS_DATASETS_AUTO_INSTALL="false", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(DATASETS))
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler import open_existing_scheduler
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceLane
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.parented_phase_resources import ParentedPhaseResourceReservation
    cancelled = threading.Event()
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, lambda _number, _frame: cancelled.set())
    scheduler = open_existing_scheduler(STATE, POLICY)
    need(all(getattr(scheduler.config, name) is None for name in (
        "max_memory_percent", "max_swap_percent", "max_gpu_memory_percent")),
        "existing_CPU_policy_must_skip_external_process_telemetry")
    need(CAPSULE.is_dir() and not os.path.lexists(RUN) and RUN.parent.is_dir()
         and RUN.parent.resolve(strict=True) == RUN.parent
         and not any(path.is_symlink() for path in (RUN.parent, *RUN.parent.parents)),
         "real_prepared_capsule_and_fresh_native_run_required")
    with scheduler.acquire(ResourceLane.ORCHESTRATION, cpu_slots=3, memory_mb=4096, child_process_slots=13,
            timeout=1800, cancel_event=cancelled, request_id=PROFILE) as root:
        with root.acquire_child(lane=ResourceLane.ORCHESTRATION, cpu_slots=3, memory_mb=4096,
                child_process_slots=13, timeout=1800, cancel_event=cancelled, request_id=PROFILE + ":phase") as phase:
            need(root.owner_pid == phase.owner_pid == os.getpid()
                 and phase.parent_lease_id == root.lease_id
                 and root.child_process_slots == phase.child_process_slots == 13
                 and root.cpu_slots == phase.cpu_slots == 3
                 and root.memory_mb == phase.memory_mb == 4096
                 and not root.released and not phase.released
                 and not root.cancelled and not phase.cancelled,
                 "actual_same_process_13slot_phase_capacity_required")
            verify_sources(request)
            verify_capsule(request)
            need(not any(name == "ipfs_accelerate_py" or name.startswith("ipfs_accelerate_py.") for name in sys.modules),
                 "fresh_capsule_import_required")
            sys.path.insert(0, str(CAPSULE))
            prefix = "ipfs_accelerate_py.agent_supervisor.runtime."
            G, H, C = (importlib.import_module(prefix + name) for name in
                       ("grok_cli_runner", "ordinary_success_handoff", "ordinary_custody_hold"))
            for module, name in ((G, "grok_cli_runner.py"), (H, "ordinary_success_handoff.py"), (C, "ordinary_custody_hold.py")):
                need(Path(module.__file__).absolute() == CAPSULE / "ipfs_accelerate_py/agent_supervisor/runtime" / name,
                     "actual_frozen_capsule_modules_required")
            RUN.mkdir(mode=0o700)
            fsync_directory(RUN.parent)
            (RUN / "native").mkdir(mode=0o700)
            fsync_directory(RUN)
            results = [run_case(G, H, C, phase, ParentedPhaseResourceReservation, request, case) for case in CASES]
            result = {"schema": "legal-ir-callable-dual-pipe-native-mechanics@1", "fixture_only": True,
                "scope": "synthetic_nonmodel_dual_pipe_engineering_fixture_only", "profile": PROFILE,
                "qualifier_source_revision": "dual-pipe-native-v04",
                "declared_process_slot_budget": {"primary": 8, "watchdog": 4, "trusted_controller_admin": 1, "root_phase": 13},
                "admin_source_bound_main_thread_top_level_clients": 1,
                "admin_capacity_scope": "actual_same_process_phase_cooperative_trusted_controller",
                "admin_native_child_attachment_qualified": False, "hard_admin_quota": False,
                "admin_source_audit_sha256": ADMIN_AUDIT_SHA,
                "request_sha256": sys.argv[2], "qualifier_manifest_sha256": request["qualifier_manifest_sha256"],
                "capsule_manifest_sha256": request["capsule_manifest_sha256"],
                "runner_source_manifest_sha256": RUNNER_MANIFEST_SHA, "handoff_source_manifest_sha256": HANDOFF_MANIFEST_SHA,
                "cases": [compact_case(row) for row in results], "passed": all(row["passed"] for row in results),
                "models_called": 0, "provider_calls_performed": False, "public_wrapper_exercised": False,
                "full_runner_exercised": False, "finish_or_success_callback_called": False,
                "actual_provider_execution_qualified": False, "production_hook_deployed": False,
                "canonical_tasks_accepted": 0, "acceptance_authority": False, "completion_authority": False,
                "daemon_storage_outside_named_roots": True, "hard_daemon_disk_quota_claimed": False,
                "all_case_owner_contexts_exited_after_independent_cleanup": True}
    result["actual_root_closeout"] = {"lease_id": root.lease_id, "released": root.released}
    result["actual_phase_closeout"] = {"lease_id": phase.lease_id, "released": phase.released}
    need(root.released and phase.released, "actual_root_phase_release_not_observed")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
