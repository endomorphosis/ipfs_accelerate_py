"""Read-only trusted host candidate correspondence; never acceptance authority.

The public observer uses a genuine current ORCHESTRATION parented owner and
actual, attached, finite Git children. It cannot create a commit, dispatch a
provider, construct NativeCandidateContext, issue a handle or change task state.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import sys
import time

ACCELERATOR = Path("/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller")
DATASETS = Path("/home/barberb/lift_coding/artifacts/legal-conditions-composition-diagnostic-20261015-01/datasets")
DAEMON = ACCELERATOR / "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py"
IDENTITY = ACCELERATOR / "ipfs_accelerate_py/agent_supervisor/task_sources/task_identity.py"
NATIVE = "ipfs_datasets_py.optimizers.logic_theorem_optimizer."
NATIVE_PATHS = tuple(DATASETS / (NATIVE.replace(".", "/") + name + ".py") for name in (
    "autoencoder_daemon_resources", "resource_scheduler", "existing_resource_scheduler", "parented_phase_resources"))
PYTHON, GIT = Path("/usr/bin/python3.12"), Path("/usr/bin/git")
REPORT = "data/legal_ir_span_graph_lean_20261016/LIR-9001/handoff-reconciliation.json"
SCHEMA = "legal-ir-real-candidate-binding-observation@3"
MAX_DOCUMENT = 4 * 1024 * 1024
ORIGINAL_BOARD = Path('/home/barberb/lift_coding/artifacts/legal-ir-completion-recovery-20261017-01/acceptance-adapter/adapter-v02/tests/fixtures/bootstrap/tasks.todo.md')
ORIGINAL_AUTHORING = Path('/home/barberb/lift_coding/artifacts/legal-ir-completion-recovery-20261017-01/acceptance-adapter/adapter-v02/tests/fixtures/bootstrap/authoring/tasks.json')
ORIGINAL_BOARD_SHA256 = '5244288f69562663f44b00d27add1ee6ef338f977fed1c0e1c8855725ff372c7'
ORIGINAL_AUTHORING_SHA256 = '66ebc6a95adde4fc717dd19ffaf5f03387f909662d1a47eccbb5c2a51857e9e6'
ORIGINAL_TASK_IDS = ('LIR-9001', 'LIR-9002', 'LIR-9003', 'LIR-9004', 'LIR-0101', 'LIR-0102', 'LIR-0201', 'LIR-0202', 'LIR-0301', 'LIR-0302', 'LIR-0401', 'LIR-0402', 'LIR-0501', 'LIR-0502', 'LIR-0601', 'LIR-0602', 'LIR-0701', 'LIR-0702', 'LIR-0801', 'LIR-0802', 'LIR-0901', 'LIR-0902', 'LIR-1001', 'LIR-1002', 'LIR-1101', 'LIR-1102', 'LIR-1201', 'LIR-1202', 'LIR-1301', 'LIR-1302', 'LIR-1401', 'LIR-1402', 'LIR-1501', 'LIR-1502', 'LIR-1601', 'LIR-1701', 'LIR-1702', 'LIR-1703', 'LIR-1801', 'LIR-1802', 'LIR-1901', 'LIR-1902', 'LIR-1903', 'LIR-1904', 'LIR-1905', 'LIR-2001', 'LIR-2002', 'LIR-2101', 'LIR-2102', 'LIR-2201', 'LIR-2202', 'LIR-2301', 'LIR-2302', 'LIR-2401', 'LIR-2402', 'LIR-2403', 'LIR-9101', 'LIR-9102', 'LIR-9103', 'LIR-9104', 'LIR-9105', 'LIR-9106', 'LIR-9107', 'LIR-9108')
NO_AUTHORITY = {name: False for name in (
    "provider_outcome_observed", "native_validation_qualified", "acceptance_authority",
    "completion_authority", "dispatch_authority", "mutation_authority", "training_authority", "proof_authority")}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_OID = re.compile(r"[0-9a-f]{40}\Z")
_GIT_BARRIER = """import json,os,resource,select,sys
if not select.select([0],[],[],180)[0] or os.read(0,2)!=b'G': raise SystemExit(125)
argv=json.loads(sys.argv[1])
if type(argv) is not list or not argv or argv[0]!='/usr/bin/git': raise SystemExit(125)
for k,v in ((resource.RLIMIT_CPU,15),(resource.RLIMIT_AS,268435456),(resource.RLIMIT_FSIZE,0),(resource.RLIMIT_NOFILE,64)):
 resource.setrlimit(k,(v,v))
os.execv(argv[0],argv)
"""


class BindingRefusal(ValueError):
    pass


def need(condition, reason):
    if not condition:
        raise BindingRefusal(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical_path(path):
    path = Path(path)
    need(path.is_absolute() and os.path.abspath(path) == str(path) and ".." not in path.parts
         and not any(p.is_symlink() for p in (path, *path.parents)), "canonical_unaliased_path_required")
    return path


def check_regular_metadata(*, mode, nlink, uid, owner_uid, trusted_source):
    """Pure reader policy; no path/content observation or acceptance authority.

    Private candidate inputs cannot be group/world writable. Existing trusted
    implementation sources may be group writable, but only exact SHA-pinned
    stable regular files owned by root/current user and never world writable.
    """
    forbidden = 0o002 if trusted_source else 0o022
    need(type(trusted_source) is bool, "reader_policy_must_be_explicit")
    need(stat.S_ISREG(mode) and nlink == 1 and uid in {0, owner_uid}
         and mode & forbidden == 0,
         "bounded_sha_pinned_trusted_source_required" if trusted_source else "bounded_owned_regular_input_required")


def _read_regular(path, maximum, *, trusted_source):
    path = canonical_path(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        check_regular_metadata(mode=before.st_mode, nlink=before.st_nlink,
            uid=before.st_uid, owner_uid=os.getuid(), trusted_source=trusted_source)
        need(0 < before.st_size <= maximum,
             "bounded_sha_pinned_trusted_source_required" if trusted_source else "bounded_owned_regular_input_required")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns,
                          s.st_mode, s.st_uid, s.st_gid, s.st_nlink)
    need(len(raw) == before.st_size and identity(before) == identity(after) == identity(path.lstat()),
         "input_changed_during_read")
    return raw


def read_regular(path, maximum=MAX_DOCUMENT):
    """Strict private candidate input reader; caller cannot weaken its policy."""
    return _read_regular(path, maximum, trusted_source=False)


def read_pinned_source(path, expected_sha256, maximum=32 * 1024 * 1024):
    """Read a stable trusted implementation only with its exact frozen SHA.

    This reader permits the existing owner-controlled0664 source files, while
    strict candidate input reads retain the original no-group-write policy.
    """
    need(type(expected_sha256) is str and _SHA.fullmatch(expected_sha256),
         "implementation_digest_invalid")
    raw = _read_regular(path, maximum, trusted_source=True)
    need(sha(raw) == expected_sha256, "observer_implementation_source_changed")
    return raw


def required_source_paths():
    return (Path(__file__).absolute(), DAEMON, IDENTITY, *NATIVE_PATHS, PYTHON, GIT, ORIGINAL_BOARD, ORIGINAL_AUTHORING)


def verify_original_inventory():
    """Independently join exact original authored IDs and execution headings."""
    board = read_pinned_source(ORIGINAL_BOARD, ORIGINAL_BOARD_SHA256)
    authored = json.loads(read_pinned_source(ORIGINAL_AUTHORING, ORIGINAL_AUTHORING_SHA256))
    need(type(authored) is dict and type(authored.get("tasks")) is list,
         "original_task_inventory_invalid")
    rows = authored["tasks"]
    need(len(rows) == 64 and all(type(row) is dict and type(row.get("task_id")) is str for row in rows),
         "original_task_inventory_invalid")
    identifiers = tuple(row["task_id"] for row in rows)
    headings = tuple(re.findall(r"(?m)^## (LIR-[0-9]{4}) [^\n]+$", board.decode("utf-8")))
    need(len(set(identifiers)) == 64 and identifiers == headings == ORIGINAL_TASK_IDS,
         "original_task_inventory_correspondence_failed")
    return identifiers


def verify_source_pins(pins):
    need(type(pins) is dict and set(pins) == {str(p) for p in required_source_paths()},
         "exact_observer_implementation_pins_required")
    for spelling, digest in pins.items():
        need(type(digest) is str and _SHA.fullmatch(digest), "implementation_digest_invalid")
        read_pinned_source(Path(spelling), digest, 32 * 1024 * 1024)
    need(pins[str(ORIGINAL_BOARD)] == ORIGINAL_BOARD_SHA256
         and pins[str(ORIGINAL_AUTHORING)] == ORIGINAL_AUTHORING_SHA256, "original_task_inventory_pins_changed")
    verify_original_inventory()


def parse_commit(raw, expected_parent):
    need(type(raw) is bytes and 0 < len(raw) <= 65536 and b"\n\n" in raw, "bounded_commit_object_required")
    lines = raw.split(b"\n\n", 1)[0].split(b"\n")
    trees = [line[5:] for line in lines if line.startswith(b"tree ")]
    parents = [line[7:] for line in lines if line.startswith(b"parent ")]
    need(type(expected_parent) is str and _OID.fullmatch(expected_parent)
         and len(trees) == 1 and re.fullmatch(rb"[0-9a-f]{40}", trees[0])
         and parents == [expected_parent.encode("ascii")],
         "candidate_must_have_exact_single_baseline_parent")
    return {"tree": trees[0].decode("ascii"), "parent": parents[0].decode("ascii")}


def parse_tree_entry(raw, relative):
    need(type(raw) is bytes and raw.endswith(b"\0") and raw.count(b"\0") == 1, "single_regular_candidate_blob_required")
    header, separator, name = raw[:-1].partition(b"\t")
    fields = header.split(b" ")
    need(separator and name == relative.encode("utf-8") and len(fields) == 3
         and fields[0] == b"100644" and fields[1] == b"blob"
         and re.fullmatch(rb"[0-9a-f]{40}", fields[2]), "single_regular_candidate_blob_required")
    return fields[2].decode("ascii")


def parse_report_diff(raw, relative):
    need(type(raw) is bytes and raw.endswith(b"\0") and raw.count(b"\0") == 2, "report_only_candidate_required")
    header, name, trailing = raw.split(b"\0")
    fields = header.split(b" ")
    need(not trailing and name == relative.encode("utf-8") and len(fields) == 5
         and fields[0] in {b":000000", b":100644"} and fields[1] == b"100644"
         and all(re.fullmatch(rb"[0-9a-f]{40}", value) for value in fields[2:4])
         and ((fields[0] == b":000000" and fields[4] == b"A" and fields[2] == b"0" * 40)
              or (fields[0] == b":100644" and fields[4] == b"M" and fields[2] != fields[3])),
         "report_only_candidate_required")
    return fields[3].decode("ascii")


def reject_board_shortcuts(raw):
    need(type(raw) is bytes and 0 < len(raw) <= MAX_DOCUMENT and b"\r" not in raw, "bounded_utf8_legacy_board_required")
    text = raw.decode("utf-8")
    seen = set()
    for line in text.splitlines():
        if line.startswith("## "):
            seen = set()
        stripped = line.strip()
        if stripped.startswith("- ") and ":" in stripped:
            key = " ".join(stripped[2:].split(":", 1)[0].lower().replace("_", " ").split())
            need(key not in seen, "duplicate_legacy_task_metadata")
            seen.add(key)
            need(key not in {"canonical task key", "canonical task cid", "dedupe key", "deduplication key"},
                 "caller_supplied_task_identity_shortcut")


def derive_task_binding(raw, board_path, parsed_tasks, identity_function, output_function):
    """Pure correspondence on parsed objects; caller objects are not observations."""
    reject_board_shortcuts(raw)
    need(type(parsed_tasks) is list and [task.task_id for task in parsed_tasks]
         == list(ORIGINAL_TASK_IDS), "exact_legacy_task_population_required")
    need(all(task.status == ("todo" if task.task_id == "LIR-9001" else "blocked") for task in parsed_tasks)
         and parsed_tasks[0].completion == "manual", "real_task_frontier_or_completion_changed")
    task = parsed_tasks[0]
    need(not any(" ".join(str(key).lower().replace("_", " ").split()) == "allowed paths" and str(value).strip()
                 for key, value in task.metadata.items()), "additional_path_authority_revision_not_supported")
    identity = identity_function({"task_id": task.task_id, "title": task.title,
        "outputs": list(output_function(task)), "acceptance": task.acceptance, "metadata": dict(task.metadata)},
        board_namespace=task.board_namespace, source_path=board_path)
    need(identity.canonical_task_cid == task.canonical_task_cid
         and identity.canonical_task_key == task.canonical_task_key, "actual_parser_identity_correspondence_failed")
    need(REPORT in output_function(task), "declared_report_output_required")
    return {"task_id": task.task_id, "canonical_task_cid": identity.canonical_task_cid,
        "canonical_task_key": identity.canonical_task_key, "task_revision": identity.canonical_task_cid,
        "task_revision_convention": "ordinary-legacy-daemon-canonical-task-cid",
        "board_namespace": identity.board_namespace, "board_sha256": sha(raw),
        "expected_board_sha256": sha(raw), "all_other_63_tasks_remain_blocked": True}


def _guard_owner(owner):
    need(type(owner).__module__ == NATIVE + "parented_phase_resources"
         and type(owner).__name__ == "ParentedPhaseResourceReservation", "genuine_parented_preparation_owner_required")
    module = importlib.import_module(NATIVE + "parented_phase_resources")
    need(type(owner) is module.ParentedPhaseResourceReservation
         and Path(module.__file__).absolute() == NATIVE_PATHS[-1], "actual_pinned_native_owner_required")
    need(owner.heartbeat(), "native_owner_heartbeat_failed")
    owner.check_usage(owner.directory, full=True)
    lease, public = owner.native_lease.to_dict(), owner.to_dict()
    need(lease.get("lane") == "orchestration" and lease.get("owner_pid") == os.getpid()
         and lease.get("released") is False and lease.get("cancelled") is False
         and lease.get("cpu_slots", 0) >= 1 and lease.get("memory_mb", 0) >= 512
         and lease.get("child_process_slots", 0) >= 1 and public.get("status") == "active",
         "actual_finite_orchestration_owner_required")
    return public


def _git(owner, repository, arguments, maximum=65536):
    argv = [str(GIT), "--no-optional-locks", "--no-replace-objects", "-c", "core.hooksPath=/dev/null",
        "-c", "core.fsmonitor=false", "-c", "core.untrackedCache=false", *arguments]
    environment = {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0", "GIT_NO_REPLACE_OBJECTS": "1", "GIT_NO_LAZY_FETCH": "1", "GIT_LITERAL_PATHSPECS": "1"}
    process = subprocess.Popen([str(PYTHON), "-I", "-S", "-B", "-c", _GIT_BARRIER, json.dumps(argv)],
        cwd=repository, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True)
    selector, buffers = selectors.DefaultSelector(), {"stdout": bytearray(), "stderr": bytearray()}
    attachment_attempted = False
    try:
        attachment_attempted = True
        owner.check_usage(owner.directory, child_pid=process.pid)
        need(owner.heartbeat(), "native_git_attachment_heartbeat_failed")
        process.stdin.write(b"G")
        process.stdin.close()
        for label in buffers:
            stream = getattr(process, label)
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, label)
        deadline, next_sample = time.monotonic() + 15, 0
        while selector.get_map() or process.poll() is None:
            need(time.monotonic() < deadline, "native_git_observation_deadline_exceeded")
            if time.monotonic() >= next_sample:
                owner.check_usage(owner.directory)
                need(owner.heartbeat(), "native_git_observation_heartbeat_failed")
                next_sample = time.monotonic() + 0.5
            for key, _ in selector.select(0.1):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                buffers[key.data].extend(chunk)
                need(len(buffers[key.data]) <= (maximum if key.data == "stdout" else 16384), "bounded_git_output_exceeded")
        need(process.wait(timeout=1) == 0, "actual_git_observation_failed")
        need(owner.check_usage(owner.directory)["group_rss"]["live_processes"] == 0, "git_child_group_not_quiescent")
        return bytes(buffers["stdout"])
    finally:
        try:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        finally:
            selector.close()
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()
            if attachment_attempted:
                need(owner.check_usage(owner.directory)["group_rss"]["live_processes"] == 0,
                     "owned_git_group_closeout_requires_retained_custody")


def observe_real_candidate_binding(*, preparation_owner, repository_root, board_path,
        expected_baseline_commit, expected_candidate_commit, expected_board_sha256,
        expected_report_sha256, protected_source_sha256):
    """Return actual inert correspondence, never a NativeCandidateContext or handle.

    Requires a fresh genuine parented owner: ten Git children use its public
check_usage attachment/history. Native checks can refuse at any point; caller
retains responsibility for genuine owner closeout and output precharge.
"""
    need(sys.dont_write_bytecode is True, "no_bytecode_native_host_required")
    verify_source_pins(protected_source_sha256)
    public = _guard_owner(preparation_owner)
    record = public.get("record") or {}
    need(record.get("child") is None and not record.get("prior_children"), "fresh_native_observer_owner_required")
    repository, board = canonical_path(repository_root), canonical_path(board_path)
    need(repository.is_dir() and repository in board.parents
         and any(Path(row["path"]) in repository.parents for row in public["roots"]), "owned_repository_and_board_required")
    for value in (expected_baseline_commit, expected_candidate_commit):
        need(type(value) is str and _OID.fullmatch(value), "exact_candidate_commit_ids_required")
    need(expected_baseline_commit != expected_candidate_commit, "candidate_cannot_be_guessed_baseline_head")
    for value in (expected_board_sha256, expected_report_sha256):
        need(type(value) is str and _SHA.fullmatch(value), "exact_candidate_content_pins_required")
    report = canonical_path(repository / REPORT)
    raw_board, raw_report = read_regular(board), read_regular(report)
    need(sha(raw_board) == expected_board_sha256 and sha(raw_report) == expected_report_sha256, "actual_board_or_report_pin_differs")
    reject_board_shortcuts(raw_board)
    daemon = importlib.import_module("ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon")
    identity = importlib.import_module("ipfs_accelerate_py.agent_supervisor.task_sources.task_identity")
    need(Path(daemon.__file__).absolute() == DAEMON and Path(identity.__file__).absolute() == IDENTITY
         and Path(daemon.parse_task_text.__code__.co_filename).absolute() == DAEMON
         and daemon.parse_task_text.__globals__["canonical_task_identity"] is identity.canonical_task_identity,
         "actual_pinned_public_task_parser_required")
    tasks = daemon.parse_task_text(raw_board.decode("utf-8"), path=board, task_header_prefix="## LIR-")
    binding = derive_task_binding(raw_board, board, tasks, identity.canonical_task_identity, daemon.task_declared_output_paths)
    git = lambda args, maximum=65536: _git(preparation_owner, repository, args, maximum)
    need(git(["rev-parse", "--show-toplevel"]).decode().strip() == str(repository), "actual_git_repository_root_differs")
    need(git(["rev-parse", "--verify", "HEAD^{commit}"]).decode().strip() == expected_candidate_commit, "candidate_is_not_current_head")
    commit = parse_commit(git(["cat-file", "commit", expected_candidate_commit]), expected_baseline_commit)
    report_oid = parse_tree_entry(git(["ls-tree", "-z", expected_candidate_commit, "--", REPORT]), REPORT)
    need(git(["cat-file", "blob", report_oid], MAX_DOCUMENT) == raw_report, "candidate_report_blob_differs")
    board_relative = board.relative_to(repository).as_posix()
    board_oid = parse_tree_entry(git(["ls-tree", "-z", expected_candidate_commit, "--", board_relative]), board_relative)
    need(git(["cat-file", "blob", board_oid], MAX_DOCUMENT) == raw_board, "candidate_board_blob_differs")
    diff_oid = parse_report_diff(git(["diff-tree", "--no-commit-id", "--raw", "--abbrev=40", "--no-renames",
        "--no-ext-diff", "-r", "-z", expected_baseline_commit, expected_candidate_commit, "--"]), REPORT)
    need(diff_oid == report_oid, "candidate_diff_report_oid_differs")
    need(git(["status", "--porcelain=v1", "-z", "--untracked-files=all"]) == b"", "candidate_worktree_not_clean")
    need(git(["rev-parse", "--verify", "HEAD^{commit}"]).decode().strip() == expected_candidate_commit, "candidate_head_changed_during_observation")
    need(read_regular(board) == raw_board and read_regular(report) == raw_report, "candidate_inputs_changed_during_observation")
    verify_source_pins(protected_source_sha256)
    final = _guard_owner(preparation_owner)
    return {"schema": SCHEMA, "observation_only": True, **NO_AUTHORITY, **binding,
        "repository_root": str(repository), "board_path": str(board), "report_path": str(report),
        "report_sha256": sha(raw_report), "report_blob_oid": report_oid, "board_blob_oid": board_oid,
        "candidate_commit": expected_candidate_commit, "candidate_tree": commit["tree"],
        "candidate_parent": commit["parent"], "actual_current_head": expected_candidate_commit,
        "report_only_diff_verified": True, "worktree_clean_observed": True,
        "git_child_count": 10, "protected_source_sha256": dict(protected_source_sha256),
        "trusted_source_reader_policy": "stable_exact_sha256_owned_regular_not_world_writable",
        "private_candidate_reader_policy": "stable_owned_regular_not_group_or_world_writable",
        "original_task_inventory": {"board_sha256": ORIGINAL_BOARD_SHA256,
            "authoring_sha256": ORIGINAL_AUTHORING_SHA256, "ordered_task_ids": list(ORIGINAL_TASK_IDS)},
        "native_observation_owner": {"owner_pid": os.getpid(), "disk_reservation_id": final["reservation_id"],
            "native_lease_id": final["resource_lease"]["lease_id"]},
        "missing_real_provider_outcome_observer": True, "genuine_fixture_qualification_still_required": True}
