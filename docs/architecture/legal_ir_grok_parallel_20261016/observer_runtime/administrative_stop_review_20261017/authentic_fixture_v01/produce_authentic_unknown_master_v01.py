"""Future parent-admitted native UNKNOWN/master fixture; never production work.

This producer starts no provider executable/model/Portal daemon. Its genuine
DatabasePortalExecutionBridge uses a pinned factory that raises before Portal
construction, after the actual daemon has committed its native UNKNOWN intent.
The fixture does not promote that controlled refusal into nonreachability or
callback-closure evidence. Only the genuine empty-track runner emits its real
ACTIVE/terminal; only the separate genuine finite owner emits an owner receipt.

This helper acquires no scheduler/disk ledger root. The caller must separately
admit its exact source, request, Python and output tree through the actual owner.
Importing this module performs no filesystem, database or process operations.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import signal
import stat
import sys
import time
from types import ModuleType

SCHEMA = "authentic-native-unknown-master-producer-request@1"
RESULT_SCHEMA = "authentic-native-unknown-master-producer-result@1"
KEYS = {"schema", "fixture_root", "repo_root", "controller_head", "board_namespace", "profile",
        "master_duration_seconds", "product_sources", "producer_pin", "python_pin", "source_loader_pin",
        "published_loader_pin", "duckdb_installation_pin"}
MAX_JSON = 2 * 1024**2
MAX_SOURCE = 8 * 1024**2
MAX_FILE = 64 * 1024**2
PREFIX = "crash-copy"
BASENAME = "crash-copy.execution.shard-0001-of-0003.duckdb"
TASK = "authentic-fixture-native-unknown-task"
ALIAS = "FIXTURE-T"
PROVIDER_COLUMNS = ("invocation_id", "attempt_id", "task_cid", "idempotency_key",
                    "owner_session_id", "recorded_at_ms", "result_json")


def need(value, message):
    if not value:
        raise ValueError(message)


def decode(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate producer JSON key")
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite producer JSON")))


def regular(path, maximum=MAX_SOURCE, *, private=False, nonempty=True):
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path, "producer file alias")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        need(stat.S_ISREG(before.st_mode) and before.st_uid in (0, os.geteuid())
             and before.st_nlink == 1 and not before.st_mode & 0o002
             and int(nonempty) <= before.st_size <= maximum,
             "producer file custody/bound differs")
        need(not private or (before.st_uid == os.geteuid() and stat.S_IMODE(before.st_mode) == 0o600),
             "producer private file mode differs")
        chunks, count = [], 0
        while block := os.read(descriptor, min(65536, maximum - count + 1)):
            count += len(block)
            need(count <= maximum, "producer read bound exceeded")
            chunks.append(block)
        fields = ("st_dev", "st_ino", "st_size", "st_mode", "st_uid", "st_gid", "st_nlink", "st_mtime_ns", "st_ctime_ns")
        witness = lambda item: tuple(getattr(item, key) for key in fields)
        need(count == before.st_size and witness(before) == witness(os.fstat(descriptor)) == witness(path.lstat()),
             "producer input changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def read_pin(value, maximum=MAX_SOURCE, *, private=False):
    need(type(value) is dict and set(value) == {"path", "bytes", "sha256"}, "producer pin fields differ")
    data = regular(value["path"], maximum, private=private, nonempty=False)
    need(type(value["bytes"]) is int and value["bytes"] == len(data)
         and type(value["sha256"]) is str and hashlib.sha256(data).hexdigest() == value["sha256"],
         "producer pin bytes differ")
    return data


def pin(path, maximum=MAX_FILE):
    data = regular(path, maximum, nonempty=False)
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def identity(path):
    info = Path(path).stat()
    return dict(device=info.st_dev, inode=info.st_ino, size=info.st_size, mtime_ns=info.st_mtime_ns)


def native_metadata(connection):
    rows = connection.execute("SELECT key,value FROM daemon_execution_metadata").fetchall()
    return {(row["key"] if isinstance(row, Mapping) else row[0]):
            (row["value"] if isinstance(row, Mapping) else row[1]) for row in rows}


def write_new(path, value=None, *, raw=None):
    if raw is None:
        raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False) + "\n").encode("utf-8")
    need(len(raw) <= MAX_FILE, "producer output record exceeds bound")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        position = 0
        while position < len(raw):
            count = os.write(descriptor, raw[position:])
            need(count > 0, "producer short write")
            position += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return pin(path)


def ticks(pid):
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def local_git_head(repo):
    """Read this worktree's actual Git HEAD; never start a git subprocess."""
    dot = repo / ".git"
    if dot.is_file():
        text = regular(dot, 4096).decode().strip()
        need(text.startswith("gitdir: "), "worktree Git directory pointer differs")
        git_dir = Path(text[8:])
        if not git_dir.is_absolute():
            git_dir = repo / git_dir
        git_dir = git_dir.resolve(strict=True)
    else:
        git_dir = dot.resolve(strict=True)
    head = regular(git_dir / "HEAD", 4096).decode().strip()
    if not head.startswith("ref: "):
        need(re.fullmatch(r"[0-9a-f]{40}", head), "detached Git HEAD differs")
        return head
    ref = head[5:]
    need(ref.startswith("refs/") and ".." not in Path(ref).parts, "Git symbolic ref differs")
    common = git_dir
    if (git_dir / "commondir").is_file():
        common = (git_dir / regular(git_dir / "commondir", 4096).decode().strip()).resolve(strict=True)
    for directory in (git_dir, common):
        candidate = directory / ref
        if candidate.is_file():
            value = regular(candidate, 4096).decode().strip()
            need(re.fullmatch(r"[0-9a-f]{40}", value), "Git loose ref differs")
            return value
    packed = regular(common / "packed-refs", MAX_JSON).decode().splitlines()
    found = [line.split()[0] for line in packed if line and not line.startswith(("#", "^"))
             and len(line.split()) == 2 and line.split()[1] == ref]
    need(len(found) == 1 and re.fullmatch(r"[0-9a-f]{40}", found[0]), "Git packed ref differs")
    return found[0]


def forbid_external(event, args):
    if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.exec", "socket.connect", "socket.getaddrinfo"}:
        raise RuntimeError("authentic fixture forbids provider/process/network dispatch")
    if event == "socket.__new__" and len(args) > 1 and args[1] != 1:
        raise RuntimeError("authentic fixture forbids nonlocal sockets")


class InertFactoryRefusal(RuntimeError):
    """Fixture configuration: actual bridge reached factory, Portal uncreated."""


def prepare_imports(request, root):
    # Persist only below the new, charged fixture tree. Never repurpose HOME.
    # Original v05 retains its own environment unchanged. This child must not
    # forward inherited model/account/loader credentials into product imports.
    os.environ.clear()
    for key, value in {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC",
        "IPFS_ACCEL_SKIP_CORE": "1", "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1",
        "IPFS_AUTO_INSTALL": "false", "IPFS_DATASETS_AUTO_INSTALL": "false",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "IPFS_ACCELERATE_AGENT_QUACK_PREFER": "0", "TMPDIR": str(root), "TMP": str(root), "TEMP": str(root),
        "XDG_CONFIG_HOME": str(root / "private-config"), "XDG_CACHE_HOME": str(root / "private-cache"),
        "HF_HOME": str(root / "private-hf"), "HF_TOKEN_PATH": str(root / "absent-hf-token"),
        "IPFS_ACCELERATE_LLM_ALLOCATION_DB": str(root / "fixture-allocation.duckdb"),
        "IPFS_ACCELERATE_META_INDEX_DUCKDB": str(root / "fixture-meta-index.duckdb")}.items():
        os.environ[key] = value
    sys.pycache_prefix = str(root / "unwritten-source-pycache")
    loader = ModuleType("_pinned_authentic_fixture_source_loader")
    loader.__file__ = request["source_loader_pin"]["path"]
    exec(compile(read_pin(request["source_loader_pin"]), loader.__file__, "exec", dont_inherit=True), loader.__dict__)
    repo = Path(request["repo_root"])
    pins = request["product_sources"]
    need(type(pins) is dict and 1 <= len(pins) <= 4096, "actual product Python population differs")
    actual = {str(path) for path in (repo / "ipfs_accelerate_py").rglob("*.py")}
    need(set(pins) == actual, "actual product Python closure differs")
    total = 0
    for name, value in pins.items():
        need(name == value["path"] and Path(name).is_relative_to(repo / "ipfs_accelerate_py"),
             "product source pin crosses actual package")
        total += len(read_pin(value))
        need(total <= 256 * 1024**2, "actual product source aggregate exceeds bound")
    publication = request["published_loader_pin"]
    need(publication["path"] == str(repo / "test/api/native_execution_crash_wal_fixture.py"),
         "published backend loader source crosses product root")
    published = loader._published(Path(publication["path"]), publication["sha256"])
    installation = read_pin(request["duckdb_installation_pin"], MAX_JSON, private=True)
    write_new(root / "duckdb-installation.json", raw=installation)
    duckdb, installed = published._load_exact_installed_duckdb(root, hashlib.sha256(installation).hexdigest())
    sys.path.insert(0, str(repo))
    sys.meta_path.insert(0, loader._PinnedProductFinder(repo, pins))
    return loader, duckdb, installed


def produce(request_path, expected_sha):
    need(sys.flags.isolated == 1 and sys.flags.no_site == 1 and sys.dont_write_bytecode is True,
         "authentic producer requires exact Python -I -S -B startup")
    resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE, MAX_FILE))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    signal.setitimer(signal.ITIMER_REAL, 240)
    raw = regular(request_path, MAX_JSON, private=True)
    need(hashlib.sha256(raw).hexdigest() == expected_sha, "producer request SHA differs")
    request = decode(raw)
    need(type(request) is dict and set(request) == KEYS and request["schema"] == SCHEMA,
         "producer request closed schema differs")
    root, repo = Path(request["fixture_root"]), Path(request["repo_root"])
    need(root.is_absolute() and root.parent.resolve(strict=True) == root.parent and not root.exists()
         and not root.is_symlink() and root.name not in {".", ".."}, "fixture root is not NEW absolute private child")
    need(repo.is_absolute() and repo.resolve(strict=True) == repo
         and type(request["controller_head"]) is str and local_git_head(repo) == request["controller_head"],
         "actual qualified controller HEAD differs")
    need(request["producer_pin"]["path"] == str(Path(__file__).resolve(strict=True))
         and request["python_pin"]["path"] == str(Path(sys.executable).resolve(strict=True)),
         "exact producer/Python source identity differs")
    read_pin(request["producer_pin"])
    read_pin(request["python_pin"], 128 * 1024**2)
    need(request["profile"] in {"one_attempt", "two_attempt"}
         and type(request["master_duration_seconds"]) is int and 1 <= request["master_duration_seconds"] <= 5
         and type(request["board_namespace"]) is str
         and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{1,100}", request["board_namespace"]),
         "authentic finite fixture profile differs")
    parent_pid = os.getppid()
    need(parent_pid > 1, "actual native owner parent absent")
    parent_process = {"pid": parent_pid, "start_time_ticks": ticks(parent_pid),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    root.mkdir(mode=0o700)
    write_new(root / "producer-request.json", raw=raw)
    # From this point all attempted child/network dispatch is a qualification
    # failure; native runner empty tracks must not silently start a supervisor.
    sys.addaudithook(forbid_external)
    loader, duckdb, installed = prepare_imports(request, root)
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import DatabaseCoordinator, TASK_CLAIM_FAILURE_REARM_OPERATION
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import planning_unknown_abandonment as policy
    from ipfs_accelerate_py.agent_supervisor.runtime import multi_supervisor_runner
    loader._loaded_product_sources(request["product_sources"])
    source = DatabaseTaskSource(intent=IntentRepository(root / "canonical.duckdb"))
    repository = source.intent
    repository.upsert_objective(objective_id="authentic-fixture-objective", objective_alias="FIXTURE-O", title="DISPOSABLE FIXTURE ONLY")
    repository.upsert_goal(goal_cid="authentic-fixture-goal", goal_alias="FIXTURE-G", title="DISPOSABLE FIXTURE ONLY")
    repository.upsert_plan(plan_cid="authentic-fixture-plan", goal_cid="authentic-fixture-goal", plan_alias="FIXTURE-P")
    for number in range(64):
        repository.upsert_task(task_cid=TASK if number == 0 else f"authentic-fixture-unrelated-{number:02d}",
            task_alias=ALIAS if number == 0 else f"FIXTURE-U{number:02d}", goal_cid="authentic-fixture-goal",
            objective_id="authentic-fixture-objective", plan_cid="authentic-fixture-plan", ordinal=number,
            status="ready" if number == 0 else "blocked", body={"board_namespace": request["board_namespace"],
                "objective": "Disposable authentic native fixture; never model work", "fixture_number": number},
            outputs=[{"path": f"data/fixture-{number}.txt", "effect": "write"}],
            acceptance=[f"Disposable fixture criterion {number}"], validations=[["/usr/bin/true"]])
    coordinator = DatabaseCoordinator(root / "coordination.duckdb")
    coordinator.open()
    refusals = []

    def inert_factory(paths, alias):
        need(alias == ALIAS and paths.root.is_relative_to(root / "native-portal-projections"),
             "actual bridge reached a foreign fixture factory scope")
        refusals.append({"schema": "authentic-fixture-inert-factory-observation@1", "task_alias": alias,
                         "attempt_projection_root": str(paths.root), "portal_created": False, "model_called": False})
        raise InertFactoryRefusal("DISPOSABLE FIXTURE inert Portal factory refusal before construction")

    def daemon_for(lane):
        lane.mkdir(mode=0o700)
        daemon = DatabaseImplementationDaemon(database_path=root / "canonical.duckdb",
            coordination_path=root / "coordination.duckdb", execution_path=lane / BASENAME,
            state_dir=lane, state_prefix=PREFIX, task_shard_count=3, task_shard_index=1, strict_task_sharding=True,
            authority_mode="embedded_exclusive", task_source_kind="duckdb", task_source=source,
            coordinator=coordinator, require_real_execution=True, lease_ms=3_600_000)
        daemon.open()
        return daemon

    def native_unknown(daemon, ordinal):
        attempt = daemon.claim_next()
        need(attempt is not None and attempt.task_cid == TASK and attempt.attempt_number == ordinal,
             "actual native claim did not create exact fixture ordinal")
        attempt = daemon.commit_phase(attempt, "context", body={"fixture_only": True})
        bridge = DatabasePortalExecutionBridge(task_source=source, attempt_root=root / "native-portal-projections",
            portal_factory=inert_factory, repo_root=repo, board_namespace=request["board_namespace"], max_passes=1)
        before = len(refusals)
        observed_refusal = None
        try:
            daemon.run_provider(attempt, provider_fn=bridge.run_provider)
        except InertFactoryRefusal as error:
            observed_refusal = error
            need(len(refusals) == before + 1, "exact native bridge factory was not reached once")
        else:
            raise ValueError("fixture unexpectedly accepted a provider result")
        attempt = daemon.get_attempt(attempt.attempt_id)
        need(attempt.status == "running" and attempt.committed_phase == "context" and attempt.revision == 2,
             "native factory refusal changed execution phase")
        cursor = daemon._require_connection().execute("SELECT " + ",".join(PROVIDER_COLUMNS)
            + " FROM provider_invocations WHERE attempt_id=? ORDER BY invocation_id LIMIT 2", [attempt.attempt_id])
        rows = cursor.fetchall()
        need(len(rows) == 1, "genuine native callback UNKNOWN was not committed exactly once")
        row = dict(rows[0]) if isinstance(rows[0], Mapping) else dict(zip(PROVIDER_COLUMNS, rows[0]))
        policy.unknown_row(row, task_cid=TASK, identity=attempt.to_dict())
        need(daemon._require_connection().execute("SELECT COUNT(*) FROM effect_claims WHERE attempt_id=?",
             [attempt.attempt_id]).fetchone()[0] == 0, "unexpected native fixture effects")
        need(bridge._issued_provider_exit_failure is None, "native bridge issued incompatible failure authority")
        return attempt, row, observed_refusal

    historical = []
    if request["profile"] == "two_attempt":
        previous = daemon_for(root / "historical-lane")
        first, old_row, first_refusal = native_unknown(previous, 1)
        failure = previous._build_portal_failure_settlement_receipt(first, failure=first_refusal)
        first = previous._commit_portal_failure_phase(first, failure)
        previous._fail_coordination_claim(coordinator.get_task_claim(first.claim_id), failure)
        previous._settle_portal_failure_control_task(first, failure)
        old_attempt = policy.plain(first.to_dict())
        previous_path = previous.execution_path
        previous.close()
        need(not Path(str(previous_path) + ".wal").exists(), "historical native sidecar did not really close")
        historical = [{"row": old_row, "attempt": old_attempt, "store_pins": [pin(previous_path)]}]
        rearmed = source.rearm_blocked_task(TASK,
            receipt={"operation": TASK_CLAIM_FAILURE_REARM_OPERATION, "settlement_id": failure["settlement_id"]})
        coordinator.rearm_failed_task(failure_receipt=failure, control_task_observation=rearmed.task.to_dict())
    current = daemon_for(root / "current-lane")
    attempt, row, _current_refusal = native_unknown(current, 2 if historical else 1)
    phases = policy.plain(current.phase_history(attempt.attempt_id))
    coordination = {"claim": policy.plain(coordinator.get_task_claim(attempt.claim_id).to_dict()),
        "lease": policy.plain(coordinator.get_lease(attempt.lease_id).to_dict()),
        "attempt": policy.plain(coordinator.get_task_attempt(attempt.attempt_id).to_dict()),
        "previous_attempts": [policy.plain(coordinator.get_task_attempt(history["attempt"]["attempt_id"]).to_dict())
                              for history in historical]}
    snapshot = {"schema": policy.SNAPSHOT_SCHEMA, "task": policy.plain(source.get_task(TASK).to_dict()),
        "intent_task": policy.plain(repository.get_task(TASK)), "canonical_population": policy.canonical_population(source),
        "attempt": policy.plain(attempt.to_dict()), "phases": phases, "providers": [row], "effects": [],
        "coordination": coordination, "historical_unknowns": historical}
    snapshot_pin = write_new(root / "native-preimage.json", snapshot)
    metadata = native_metadata(current._require_connection())
    # Canonical and coordinator close before real master finalization. The
    # actual current execution connection+writer lock deliberately stay open.
    source.close()
    coordinator.close()
    closed_pins = {"canonical": pin(root / "canonical.duckdb"), "coordination": pin(root / "coordination.duckdb")}
    need(all(not Path(value["path"] + ".wal").exists() for value in closed_pins.values()),
         "canonical/coordinator did not really close")
    master = root / "current-lane/master.pid"
    output_lines = []
    active_pin = None

    def observe_native_runner_output(line):
        nonlocal active_pin
        text = str(line)
        need(len(output_lines) < 256 and sum(len(item.encode()) for item in output_lines) + len(text.encode()) <= 65536,
             "native master diagnostic output exceeds fixture bound")
        output_lines.append(text)
        if active_pin is None:
            # Native runner has already published ACTIVE before its first
            # output. Preserve those exact real bytes before terminal replaces
            # the fixed path; this producer creates no substitute ACTIVE.
            active_raw = regular(Path(str(master) + ".terminal.json"), MAX_JSON)
            active = decode(active_raw)
            need(active.get("schema") == "ipfs_accelerate_py.agent_supervisor.multi-supervisor-active-binding@1"
                 and active.get("master_pid") == os.getpid() and active.get("active") is True
                 and active.get("label") == request["board_namespace"]
                 and active.get("active_binding_cid") == policy.content_identity({key: value for key, value in active.items()
                                                                               if key != "active_binding_cid"}),
                 "genuine native ACTIVE observation differs")
            active_pin = write_new(root / "master-active-binding-observed.json", raw=active_raw)

    summary = multi_supervisor_runner.run_supervisor_tracks([], repo_root=repo, common_args=(),
        duration_seconds=request["master_duration_seconds"], heartbeat_interval_seconds=0.1,
        stop_grace_seconds=1.0, python_executable=sys.executable, master_pid_path=master,
        label=request["board_namespace"], exit_when_all_tracks_terminal=True,
        output=observe_native_runner_output)
    terminal_path = Path(str(master) + ".terminal.json")
    terminal = decode(regular(terminal_path, MAX_JSON))
    need(terminal.get("schema") == "ipfs_accelerate_py.agent_supervisor.multi-supervisor-terminal@1"
         and terminal.get("master_pid") == os.getpid() and terminal.get("label") == request["board_namespace"]
         and terminal.get("all_trees_fenced") is True and terminal.get("task_completion_authority") is False,
         "genuine empty-track master did not emit exact terminal fence")
    need(terminal.get("terminal_receipt_cid") == policy.content_identity({key: value for key, value in terminal.items()
                                                                       if key != "terminal_receipt_cid"}),
         "genuine native master terminal content identity differs")
    need(active_pin is not None and decode(read_pin(active_pin))["active_binding_cid"] == terminal.get("active_binding_cid"),
         "genuine native ACTIVE and terminal generations differ")
    connection = current._require_connection()
    # Require the actual native claim/context/callback-intent WAL. An absent
    # WAL is a failed gate, never repaired by synthetic row/metadata writes.
    wal = Path(str(current.execution_path) + ".wal")
    need(wal.is_file() and wal.stat().st_size > 0, "genuine native committed WAL absent at controlled exit")
    descriptor = os.open(wal, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    need(native_metadata(connection) == metadata,
         "fixture metadata changed across native master")
    need(os.getppid() == parent_process["pid"] and ticks(parent_process["pid"]) == parent_process["start_time_ticks"],
         "actual original native owner generation changed before producer exit")
    for value in closed_pins.values():
        read_pin(value, MAX_FILE)
    loader._loaded_product_sources(request["product_sources"])
    need(local_git_head(repo) == request["controller_head"], "qualified controller HEAD drifted")
    read_pin(request["producer_pin"])
    need(hashlib.sha256(regular(request_path, MAX_JSON, private=True)).hexdigest() == expected_sha,
         "producer request changed")
    result = {"schema": RESULT_SCHEMA, "fixture_only": True, "profile": request["profile"],
        "production_store_opened": False, "administrative_decision_performed": False,
        "completion_authority": False, "automatic_retry_admitted": False, "callback_closed": False,
        "remote_cancellation_proved": False, "nonreachability_asserted": False,
        "provider_outcome": "unknown", "possible_external_effects": "unverified",
        "actual_native_callback_intent_path": True, "provider_row_sql_synthesis": False,
        "portal_or_model_constructed": False, "factory_refusals": refusals,
        "pid": os.getpid(), "start_time_ticks": ticks(os.getpid()),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "process_group": os.getpgrp(),
        "parent_process": parent_process,
        "controller_head": request["controller_head"], "board_namespace": request["board_namespace"],
        "request_pin": pin(request_path, MAX_JSON), "producer_pin": request["producer_pin"],
        "stores": {"canonical": str(root / "canonical.duckdb"), "coordination": str(root / "coordination.duckdb"),
                   "execution": str(current.execution_path)}, "closed_store_pins": closed_pins,
        "execution_file_identity": identity(current.execution_path), "wal_file_identity": identity(wal),
        "execution_pin": pin(current.execution_path), "wal_pin": pin(wal), "metadata": metadata,
        "snapshot_pin": snapshot_pin, "snapshot": snapshot, "master_pid_path": str(master),
        "master_active_observation_pin": active_pin,
        "master_terminal_pin": pin(terminal_path, MAX_JSON), "master_summary": policy.plain(summary),
        "master_output_lines": output_lines, "duckdb_installation": installed,
        "native_owner_receipt_created_by_this_producer": False,
        "native_owner_finalization_verified_by_this_producer": False}
    write_new(root / "producer-result.json", result)
    # No current.close(), atexit handler, metadata rewrite or old-owner
    # constructor override. Actual kernel exit closes current native custody.
    os._exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    args = parser.parse_args()
    try:
        produce(args.request, args.request_sha256)
    except Exception as error:
        sys.stderr.write(json.dumps({"fixture_failed": True, "error_type": type(error).__name__,
            "error_message": str(error).encode("utf-8", "replace")[:4096].decode("utf-8", "ignore")}) + "\n")
        sys.stderr.flush()
        os._exit(2)


if __name__ == "__main__":
    main()
