"""Opt-in, same-callback report freezer; no validation or completion authority.

Private captures are references to actual dispatch objects, not portable receipts.
The public daemon method supplies its own decision runtime, original checkout
lock, and original shared commit tail. This module never creates a worktree,
repairs an index, merges, validates, refunds an attempt, or writes task state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import sys
import time
from types import MappingProxyType

REPORT = "data/legal_ir_span_graph_lean_20261016/LIR-9001/handoff-reconciliation.json"
BOARD_SHA256 = "5244288f69562663f44b00d27add1ee6ef338f977fed1c0e1c8855725ff372c7"
TASK_ID = "LIR-9001"
SCHEMA = "legal-ir-report-only-frozen-candidate@3"
MAX_DOCUMENT = 4 * 1024 * 1024
MAX_INDEX = 64 * 1024 * 1024
MAX_GIT_CHILDREN = 40
MIN_ROOT_CHARGE = 128 * 1024 * 1024
MAX_ROOT_CHARGE = 256 * 1024 * 1024
MAX_EXTERNAL_TOTAL = 384 * 1024 * 1024
NATIVE = "ipfs_datasets_py.optimizers.logic_theorem_optimizer."
RUNTIME = "ipfs_accelerate_py.agent_supervisor.runtime."
MERGE = "ipfs_accelerate_py.agent_supervisor.merge."
GIT, PYTHON = Path("/usr/bin/git"), Path("/usr/bin/python3.12")
SHA = re.compile(r"[0-9a-f]{64}\Z")
OID = re.compile(r"[0-9a-f]{40}\Z")
BARRIER = """import json,os,resource,select,sys
if not select.select([0],[],[],180)[0] or os.read(0,2)!=b'G': raise SystemExit(125)
a=json.loads(sys.argv[1])
if type(a) is not list or not a or a[0]!='/usr/bin/git': raise SystemExit(125)
for k,v in ((resource.RLIMIT_CPU,30),(resource.RLIMIT_AS,268435456),(resource.RLIMIT_FSIZE,16777216),(resource.RLIMIT_NOFILE,64)):
 resource.setrlimit(k,(v,v))
os.execv(a[0],a)
"""


class ReportFreezerRefusal(RuntimeError):
    """Preserve the original worktree/index/attempt on every refusal."""


def need(condition, reason):
    if not condition:
        raise ReportFreezerRefusal(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(path):
    path = Path(path)
    need(path.is_absolute() and str(path) == os.path.abspath(path)
         and not any(p.is_symlink() for p in (path, *path.parents)), "unaliased_absolute_path_required")
    return path


def witness(metadata):
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_uid,
            metadata.st_gid, metadata.st_nlink, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns)


def check_regular_policy(*, mode, nlink, uid, owner_uid, role):
    """Pure necessary role policy; metadata dictionaries grant no authority."""
    need(role in {"report", "git_metadata", "pinned_source"}, "explicit_regular_reader_role_required")
    need(stat.S_ISREG(mode) and nlink == 1 and uid in {0, owner_uid}
         and mode & 0o002 == 0, "bounded_stable_owned_regular_file_required")
    if role == "report":
        need(stat.S_IMODE(mode) == 0o644, "private_input_mode_0644_required")
    elif role == "git_metadata":
        need(stat.S_IMODE(mode) in {0o644, 0o664}, "original_git_metadata_mode_required")


def _read_git_metadata(path, maximum):
    """Actual original Git inputs only; never used for the report body."""
    path = canonical(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        check_regular_policy(mode=before.st_mode, nlink=before.st_nlink, uid=before.st_uid,
                             owner_uid=os.getuid(), role="git_metadata")
        need(0 < before.st_size <= maximum, "bounded_original_git_metadata_required")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    need(len(raw) == before.st_size and witness(before) == witness(after) == witness(path.lstat()),
         "original_git_metadata_changed_during_read")
    return raw


def read_regular(path, maximum=MAX_DOCUMENT, *, source_sha=None, allow_empty=False):
    path = canonical(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        check_regular_policy(mode=before.st_mode, nlink=before.st_nlink, uid=before.st_uid,
            owner_uid=os.getuid(), role="pinned_source" if source_sha is not None else "report")
        need((0 if allow_empty else 1) <= before.st_size <= maximum, "bounded_stable_owned_regular_file_required")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    need(len(raw) == before.st_size and witness(before) == witness(after) == witness(path.lstat()),
         "regular_file_changed_during_read")
    if source_sha is not None:
        need(type(source_sha) is str and SHA.fullmatch(source_sha)
             and sha(raw) == source_sha, "protected_implementation_source_changed")
    return raw


def strict_report(raw):
    """Syntax only: JSON validity never implies legal/Lean/native validity."""
    need(type(raw) is bytes and 0 < len(raw) <= MAX_DOCUMENT, "bounded_report_required")
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate_json_key")
            result[key] = value
        return result
    def constant(_value):
        raise ReportFreezerRefusal("nonfinite_json_number")
    def floating(text):
        value = float(text)
        need(math.isfinite(value), "nonfinite_json_number")
        return value
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant, parse_float=floating)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ReportFreezerRefusal("report_json_invalid") from exc
    need(type(value) is dict, "report_json_object_required")
    return value


def parse_tree(raw, relative):
    need(type(raw) is bytes and raw.endswith(b"\0") and raw.count(b"\0") == 1,
         "one_regular_tree_blob_required")
    header, tab, name = raw[:-1].partition(b"\t")
    parts = header.split(b" ")
    need(tab and name == relative.encode() and len(parts) == 3 and parts[:2] == [b"100644", b"blob"]
         and re.fullmatch(rb"[0-9a-f]{40}", parts[2]), "one_regular_tree_blob_required")
    return parts[2].decode()


def parse_diff(raw):
    need(type(raw) is bytes and raw.count(b"\0") == 2 and raw.endswith(b"\0"), "report_only_diff_required")
    header, name, tail = raw.split(b"\0")
    parts = header.split(b" ")
    need(not tail and name == REPORT.encode() and len(parts) == 5 and parts[1] == b"100644"
         and all(re.fullmatch(rb"[0-9a-f]{40}", p) for p in parts[2:4])
         and ((parts[0] == b":000000" and parts[2] == b"0" * 40 and parts[4] == b"A")
              or (parts[0] == b":100644" and parts[2] != parts[3] and parts[4] == b"M")),
         "report_only_diff_required")
    return parts[3].decode()


def parse_commit(raw, baseline):
    need(type(baseline) is str and OID.fullmatch(baseline) and type(raw) is bytes
         and 0 < len(raw) <= 65536 and b"\n\n" in raw, "bounded_commit_required")
    header = raw.split(b"\n\n", 1)[0].split(b"\n")
    parents = [line[7:] for line in header if line.startswith(b"parent ")]
    trees = [line[5:] for line in header if line.startswith(b"tree ")]
    need(parents == [baseline.encode()] and len(trees) == 1
         and re.fullmatch(rb"[0-9a-f]{40}", trees[0]), "exact_single_baseline_parent_required")
    return trees[0].decode()


def allowed_dirty_paths(raw):
    """Parse exact Git name-only -z output without whitespace/path tricks."""
    need(type(raw) is bytes and (not raw or raw.endswith(b"\0")), "nul_terminated_git_paths_required")
    names = raw[:-1].split(b"\0") if raw else []
    need(len(names) <= 1 and all(name == REPORT.encode() for name in names), "extra_dirty_path_refused")
    return names


def budget_projection(*, attempt, cap, task_id, task_cid, before_by_id, before_by_cid,
                      current_by_id, current_by_cid):
    """Pure necessary counters: preserves both historical attempts and all peers."""
    need(type(attempt) is int and attempt == 3 and type(cap) is int and cap == 3
         and task_id == TASK_ID and type(task_cid) is str and bool(task_cid), "exact_third_attempt_cap_required")
    for row in (before_by_id, before_by_cid, current_by_id, current_by_cid):
        need(type(row) is dict and all(type(k) is str and type(v) is int and v >= 0 for k, v in row.items()),
             "actual_attempt_counter_maps_required")
    need(before_by_id.get(task_id) == 2 and before_by_cid.get(task_cid) == 2,
         "two_historical_consumed_attempts_required")
    wanted_id, wanted_cid = dict(before_by_id), dict(before_by_cid)
    wanted_id[task_id] = wanted_cid[task_cid] = 3
    need(current_by_id == wanted_id and current_by_cid == wanted_cid,
         "historical_or_peer_attempt_counter_changed")


def _task_projection(task):
    return (task.task_id, task.title, task.status, task.completion, task.priority, task.track,
            tuple(task.depends_on), tuple(task.outputs), tuple(task.validation), task.acceptance,
            tuple(sorted(task.metadata.items())), task.canonical_task_key, task.canonical_task_cid,
            task.board_namespace)


@dataclass(frozen=True)
class _BeforeAttempt:
    daemon: object = field(repr=False)
    state: object = field(repr=False)
    task: object = field(repr=False)
    task_projection: tuple = field(repr=False)
    task_cid: str
    task_key: str
    attempt: int
    before_by_id: object = field(repr=False)
    before_by_cid: object = field(repr=False)
    preflight: object = field(repr=False)

    def __reduce__(self):
        raise ReportFreezerRefusal("dispatch_capture_not_portable")


@dataclass(frozen=True)
class _Dispatch:
    before: _BeforeAttempt = field(repr=False)
    invocation: object = field(repr=False)
    workspace: Path
    baseline: str
    board: Path
    board_bytes: bytes = field(repr=False)
    branch: str
    lifecycle: object = field(repr=False)
    control_plane_connection: object = field(repr=False)
    preflight: object = field(repr=False)

    def __reduce__(self):
        raise ReportFreezerRefusal("dispatch_capture_not_portable")


@dataclass(frozen=True)
class _Preflight:
    daemon: object = field(repr=False)
    state: object = field(repr=False)
    task: object = field(repr=False)
    task_projection: tuple = field(repr=False)
    task_cid: str
    task_key: str
    before_by_id: object = field(repr=False)
    before_by_cid: object = field(repr=False)
    repo: Path
    authoritative_board: Path
    board_relative: str
    workspace: Path
    workspace_board: Path
    board_bytes: bytes = field(repr=False)
    baseline: str
    board_oid: str
    branch: str
    lifecycle: object = field(repr=False)
    gitdir: Path
    common: Path
    repo_gitdir: Path
    directory_identities: object = field(repr=False)
    control_plane_connection: object = field(repr=False)
    index_snapshot: object = field(repr=False)
    head_snapshot: object = field(repr=False)
    actual_git_session: object = field(repr=False)

    def __reduce__(self):
        raise ReportFreezerRefusal("dispatch_capture_not_portable")


def board_relative_path(repo, board):
    """Pure necessary lexical relation; the actual caller canonicalizes both."""
    repo, board = Path(repo), Path(board)
    need(repo.is_absolute() and board.is_absolute()
         and str(repo) == os.path.abspath(repo) and str(board) == os.path.abspath(board),
         "canonical_original_board_layout_required")
    try:
        relative = board.relative_to(repo)
    except ValueError as error:
        raise ReportFreezerRefusal("authoritative_board_outside_original_repository") from error
    need(relative.parts and not any(p in {".", "..", ".git"} for p in relative.parts)
         and all(re.fullmatch(r"[A-Za-z0-9_.-]+", p) for p in relative.parts),
         "supported_authoritative_board_relative_path_required")
    return relative.as_posix()


def check_supported_index(flags, stage):
    """Pure parser only; accepting bytes cannot authenticate a checkout."""
    need(type(flags) is bytes and flags.endswith(b"\0") and type(stage) is bytes
         and stage.endswith(b"\0"), "bounded_nul_terminated_index_inventory_required")
    names, staged = [], []
    for raw in flags[:-1].split(b"\0"):
        need(raw.startswith(b"H "), "hidden_index_or_submodule_flag_refused")
        names.append(raw[2:])
    for raw in stage[:-1].split(b"\0"):
        header, tab, name = raw.partition(b"\t")
        need(tab and re.fullmatch(rb"100(?:644|755) [0-9a-f]{40} 0", header),
             "submodule_symlink_or_unmerged_index_refused")
        staged.append(name)
    need(0 < len(names) <= 16384 and len(names) == len(set(names))
         and len(staged) == len(set(staged)) and set(names) == set(staged),
         "exact_unique_tracked_index_inventory_required")
    for name in names:
        need(name and not name.startswith(b"/") and not any(p in {b"", b".", b"..", b".git"}
             for p in name.split(b"/")) and not any(c in name for c in (b"\n", b"\r", b"\t")),
             "unaliased_tracked_index_paths_required")


def _original_connection(daemon, connection=None):
    control = importlib.import_module("ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane")
    require = getattr(control, "require_borrowed_task_state_connection", None)
    need(callable(require), "original_borrowed_state_api_required")
    actual = require(daemon.state_path, connection=connection)
    need(connection is None or actual is connection, "original_borrowed_state_connection_changed")
    return actual


def _original_idle_binding(daemon, state, task, connection):
    need(daemon.report_only_candidate_freezing is True and daemon.preserve_task_attempts is True
         and state.implementation_in_progress is False and not daemon._implementation_cancel_requested(),
         "original_idle_preserved_attempt_required")
    identity = daemon._identity_for_task(task)
    before_id, before_cid = dict(state.implementation_attempts), dict(state.implementation_attempts_by_cid)
    after_id, after_cid = dict(before_id), dict(before_cid)
    after_id[TASK_ID] = after_cid[identity.canonical_task_cid] = 3
    budget_projection(attempt=3, cap=daemon.max_task_attempts, task_id=task.task_id,
        task_cid=identity.canonical_task_cid, before_by_id=before_id, before_by_cid=before_cid,
        current_by_id=after_id, current_by_cid=after_cid)
    need(task.status == "todo" and task.completion == "manual", "manual_report_frontier_and_preserved_attempts_required")
    row = _read_bound_control_plane(daemon, _original_connection(daemon, connection))
    expected = {"implementation_in_progress": False, "implementation_attempts": before_id,
                "implementation_attempts_by_cid": before_cid}
    need(all(type(row.get(k)) is type(v) and row[k] == v for k, v in expected.items()),
         "authoritative_pre_attempt_control_plane_changed")
    return identity, before_id, before_cid


def _board_layout(daemon, state, task, connection):
    identity, before_id, before_cid = _original_idle_binding(daemon, state, task, connection)
    repo, authoritative = canonical(daemon.repo_root), canonical(daemon.todo_path)
    relative = board_relative_path(repo, authoritative)
    need(repo.is_dir(), "actual_original_repository_required")
    lifecycle = daemon._active_worktree_lifecycle
    module = importlib.import_module(MERGE + "worktree_lifecycle")
    need(type(lifecycle) is module.WorkspaceLifecycleRecord
         and type(daemon.worktree_lifecycle) is module.WorktreeLifecycleStore,
         "actual_fenced_dispatch_lifecycle_required")
    workspace = canonical(lifecycle.workspace_path)
    need(workspace.is_dir() and workspace != repo and workspace not in repo.parents
         and workspace not in authoritative.parents, "separate_original_managed_worktree_required")
    need(canonical(lifecycle.repo_root) == repo and canonical(daemon.worktree_lifecycle.repo_root) == repo
         and daemon.worktree_lifecycle.load_workspace(str(workspace)) == lifecycle
         and lifecycle.task_id == TASK_ID and lifecycle.canonical_task_cid == identity.canonical_task_cid
         and lifecycle.attempt == daemon._worktree_lifecycle_attempt(task, 3)
         and lifecycle.state is module.WorkspaceLifecycleState.ACTIVE and type(lifecycle.fence) is int
         and lifecycle.fence > 0 and math.isfinite(lifecycle.expires_at) and lifecycle.expires_at > time.time()
         and lifecycle.owner == module.read_process_birth(os.getpid()),
         "original_repository_and_live_lifecycle_association_required")
    workspace_board = canonical(workspace / relative)
    board_bytes = read_regular(authoritative, source_sha=BOARD_SHA256)
    need(read_regular(workspace_board, source_sha=BOARD_SHA256) == board_bytes,
         "authoritative_and_workspace_board_bytes_differ")
    actual_module = importlib.import_module(type(daemon).__module__)
    parsed = actual_module.parse_task_text(board_bytes.decode(), path=authoritative, task_header_prefix="## LIR-")
    need(len(parsed) == 64 and parsed[0].task_id == TASK_ID
         and _task_projection(parsed[0]) == _task_projection(task)
         and all(item.status == "blocked" for item in parsed[1:])
         and actual_module.task_declared_output_paths(task) == (REPORT,),
         "original_report_only_board_frontier_required")
    return repo, authoritative, workspace, workspace_board, relative, lifecycle, board_bytes, identity, before_id, before_cid


def _git_layout_policy(gitdir, common):
    for config in {common / "config", gitdir / "config.worktree"}:
        if os.path.lexists(config):
            data = _read_git_metadata(config, 65536)
            need(not re.search(rb"(?im)^\s*\[\s*alias(?:\s|\.|\])", data),
                 "git_config_aliases_refused_before_attempt")
    for base in {gitdir, common}:
        need(not any(os.path.lexists(base / relative) for relative in
             ("objects/info/alternates", "info/grafts", "shallow", "index.lock")),
             "unsupported_git_storage_indirection_or_lock")


def _metadata_storage(daemon, layout):
    repo, _authoritative, workspace = layout[:3]
    gitdir, common = _discover_git_storage(workspace)
    repo_gitdir, repo_common = _discover_git_storage(repo)
    need(common == repo_common and gitdir != common and gitdir != repo_gitdir,
         "actual_worktree_common_repository_association_required")
    _git_layout_policy(gitdir, common)
    _git_layout_policy(repo_gitdir, common)
    return gitdir, common, repo_gitdir


def report_only_preflight_storage_roots(daemon, state, *, task):
    """Read-only metadata derivation before native precharge; no Git/grant."""
    connection = _original_connection(daemon)
    layout = _board_layout(daemon, state, task, connection)
    gitdir, common, _repo_gitdir = _metadata_storage(daemon, layout)
    return tuple(_storage_roots(layout[2], gitdir, common))


def _directory_identity(path):
    metadata = path.lstat()
    need(stat.S_ISDIR(metadata.st_mode) and metadata.st_uid in {0, os.getuid()}
         and metadata.st_mode & 0o002 == 0, "owned_unaliased_repository_directory_required")
    return metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_uid, metadata.st_gid


def check_symbolic_head(raw, branch):
    """Pure supported-files backend shape, never baseline authority."""
    need(type(raw) is bytes and type(branch) is str and branch
         and all(re.fullmatch(r"[A-Za-z0-9_.-]+", p) and p not in {".", ".."} for p in branch.split("/"))
         and raw == ("ref: refs/heads/" + branch + "\n").encode(),
         "supported_original_symbolic_branch_head_required")


def _head_ref_snapshot(gitdir, common, branch, baseline):
    """Stable readonly refs guard closes the preflight-to-mark drift window."""
    need(type(baseline) is str and OID.fullmatch(baseline), "original_preflight_baseline_required")
    head = gitdir / "HEAD"
    data = _read_git_metadata(head, 8192)
    check_symbolic_head(data, branch)
    rows = [(str(head), witness(head.lstat()), data)]
    loose, packed = common / "refs/heads" / branch, common / "packed-refs"
    loose_raw = None
    packed_raw = None
    for path in (loose, packed):
        if os.path.lexists(path):
            data = _read_git_metadata(path, MAX_INDEX if path == packed else 8192)
            rows.append((str(path), witness(path.lstat()), data))
            if path == loose:
                loose_raw = data
            else:
                packed_raw = data
        else:
            rows.append((str(path), None, None))
    if loose_raw is not None:
        need(loose_raw == (baseline + "\n").encode(), "original_loose_branch_ref_changed")
    else:
        need(packed_raw is not None, "supported_original_branch_ref_storage_required")
        matches = [line.split(b" ", 1)[0] for line in packed_raw.splitlines()
                   if line.endswith((" refs/heads/" + branch).encode())]
        need(matches == [baseline.encode()], "original_packed_branch_ref_changed")
    return tuple(rows)


def _guard_preflight_owner(owner, parent_phase):
    native = importlib.import_module(NATIVE + "parented_phase_resources")
    resource = importlib.import_module(NATIVE + "resource_scheduler")
    need(type(owner) is native.ParentedPhaseResourceReservation and type(parent_phase) is resource.ResourceLease,
         "genuine_same_process_preflight_owner_and_phase_required")
    phase = parent_phase.to_dict()
    need(parent_phase.renew() and phase.get("lane") == "orchestration" and phase.get("owner_pid") == os.getpid()
         and phase.get("released") is False and phase.get("cancelled") is False
         and phase.get("cpu_slots") == 3 and phase.get("memory_mb") == 4096
         and phase.get("child_process_slots") == 16 and phase.get("parent_lease_id"),
         "actual_original_finite_preflight_parent_required")
    lease, public = owner.native_lease.to_dict(), owner.to_dict()
    need(lease.get("parent_lease_id") == phase.get("lease_id") and lease.get("owner_pid") == os.getpid()
         and lease.get("cpu_slots") == 1 and lease.get("child_process_slots") == 4
         and lease.get("memory_mb", 0) >= 512 and public.get("status") == "active"
         and lease.get("released") is False and lease.get("cancelled") is False
         and (public.get("record") or {}).get("child") is None
         and (public.get("record") or {}).get("prior_children", []) == [],
         "fresh_exact_sibling_preflight_owner_required")
    need(owner.heartbeat(), "actual_preflight_owner_heartbeat_failed")
    owner.check_usage(owner.directory, full=True)


def preflight_report_only_candidate(daemon, state, *, task, preflight_owner, parent_phase, control_plane_connection,
              protected_source_sha256, declared_external_storage):
    """Actual readonly native Git preflight BEFORE original ordinal mutation.

    A retained in-memory layout is created only after the original anchored
    session has stopped. The caller alone owns explicit owner finalization.
    """
    need(sys.dont_write_bytecode is True and getattr(daemon, "_report_only_preflight", None) is None,
         "fresh_no_bytecode_original_preflight_required")
    _verify_sources(daemon, protected_source_sha256)
    layout = _board_layout(daemon, state, task, _original_connection(daemon, control_plane_connection))
    repo, authoritative, workspace, workspace_board, relative, lifecycle, board_bytes, identity, before_id, before_cid = layout
    gitdir, common, repo_gitdir = _metadata_storage(daemon, layout)
    _guard_preflight_owner(preflight_owner, parent_phase)
    _check_storage(preflight_owner, workspace, gitdir, common, declared_external_storage)
    index_before = _index_snapshot(gitdir)
    with _Git(preflight_owner, workspace) as git:
        for directory, expected_git in ((workspace, gitdir), (repo, repo_gitdir)):
            prefix = [] if directory == workspace else ["-C", str(repo)]
            need(git.run([*prefix, "rev-parse", "--show-toplevel"]).decode().strip() == str(directory)
                 and git.run([*prefix, "rev-parse", "--absolute-git-dir"]).decode().strip() == str(expected_git)
                 and git.run([*prefix, "rev-parse", "--path-format=absolute", "--git-common-dir"]).decode().strip() == str(common),
                 "actual_original_repository_or_git_storage_changed")
        baseline = git.run(["rev-parse", "--verify", "HEAD^{commit}"]).decode().strip()
        need(OID.fullmatch(baseline), "locally_discovered_full_baseline_commit_required")
        branch = git.run(["symbolic-ref", "--short", "HEAD"]).decode().strip()
        need(branch == lifecycle.branch and branch and not any(c in branch for c in "\0\n\r"),
             "original_preflight_branch_changed")
        check_supported_index(git.run(["ls-files", "-v", "-z"], MAX_INDEX),
                              git.run(["ls-files", "--stage", "-z"], MAX_INDEX))
        need(git.run(["diff", "--cached", "--name-only", "-z", baseline, "--"], MAX_INDEX) == b"",
             "preexisting_staged_change_refused")
        for arguments in (["diff", "--name-only", "-z", "--no-ext-diff", "--"],
                          ["ls-files", "--others", "--exclude-standard", "-z"],
                          ["ls-files", "--others", "--ignored", "--exclude-standard", "-z"]):
            allowed_dirty_paths(git.run(arguments, MAX_INDEX))
        board_oid = parse_tree(git.run(["ls-tree", "-z", baseline, "--", relative]), relative)
        need(git.run(["cat-file", "blob", board_oid], MAX_DOCUMENT) == board_bytes,
             "original_workspace_baseline_board_differs")
        need(git.run(["rev-parse", "--verify", "HEAD^{commit}"]).decode().strip() == baseline
             and _index_snapshot(gitdir) == index_before, "readonly_preflight_head_or_index_changed")
        need(_board_layout(daemon, state, task, control_plane_connection) == layout
             and _metadata_storage(daemon, layout) == (gitdir, common, repo_gitdir),
             "original_layout_changed_during_preflight")
        head_snapshot = _head_ref_snapshot(gitdir, common, branch, baseline)
    need(git._closed is True and git._known_stopped() and (git.consumer is None or git.consumer.released),
         "original_preflight_git_custody_not_stopped")
    _verify_sources(daemon, protected_source_sha256)
    capture = _Preflight(daemon, state, task, _task_projection(task), identity.canonical_task_cid,
        identity.canonical_task_key, MappingProxyType(before_id), MappingProxyType(before_cid), repo,
        authoritative, relative, workspace, workspace_board, board_bytes, baseline, board_oid, branch,
        lifecycle, gitdir, common, repo_gitdir,
        MappingProxyType({str(p): _directory_identity(p) for p in {repo, workspace, gitdir, common, repo_gitdir}}),
        control_plane_connection, index_before, head_snapshot, git)
    daemon._report_only_preflight = capture
    return {"schema": "legal-ir-report-only-layout-preflight@1", "supported": True,
        "repository_root": str(repo), "authoritative_board_path": str(authoritative),
        "workspace": str(workspace), "workspace_board_path": str(workspace_board),
        "board_relative_path": relative, "board_sha256": BOARD_SHA256,
        "baseline_commit": baseline, "baseline_board_blob_oid": board_oid,
        "actual_preflight_git_commands": git.count, "actual_preflight_git_group_stopped": True,
        "preflight_owner_finalization_belongs_to_caller": True, "attempt_consumed": False,
        "canonical_tasks_accepted": 0, "acceptance_authority": False}


def _require_preflight(daemon, state, task, *, before_consumption):
    capture = getattr(daemon, "_report_only_preflight", None)
    need(type(capture) is _Preflight and capture.daemon is daemon and capture.state is state
         and capture.task is task and _task_projection(task) == capture.task_projection,
         "actual_original_preflight_capture_required")
    _original_connection(daemon, capture.control_plane_connection)
    need(canonical(daemon.repo_root) == capture.repo and canonical(daemon.todo_path) == capture.authoritative_board
         and board_relative_path(capture.repo, capture.authoritative_board) == capture.board_relative
         and canonical(capture.workspace / capture.board_relative) == capture.workspace_board,
         "original_dual_board_layout_changed")
    need(read_regular(capture.authoritative_board, source_sha=BOARD_SHA256) == capture.board_bytes
         and read_regular(capture.workspace_board, source_sha=BOARD_SHA256) == capture.board_bytes,
         "authoritative_or_workspace_board_changed")
    for spelling, identity in capture.directory_identities.items():
        need(_directory_identity(canonical(spelling)) == identity, "original_repository_directory_replaced")
    need(_discover_git_storage(capture.workspace) == (capture.gitdir, capture.common)
         and _discover_git_storage(capture.repo) == (capture.repo_gitdir, capture.common),
         "original_repository_git_storage_changed")
    _git_layout_policy(capture.gitdir, capture.common)
    _git_layout_policy(capture.repo_gitdir, capture.common)
    lifecycle = daemon._active_worktree_lifecycle
    module = importlib.import_module(MERGE + "worktree_lifecycle")
    need(lifecycle is capture.lifecycle and daemon.worktree_lifecycle.load_workspace(str(capture.workspace)) == lifecycle
         and canonical(lifecycle.repo_root) == capture.repo and canonical(daemon.worktree_lifecycle.repo_root) == capture.repo
         and lifecycle.state is module.WorkspaceLifecycleState.ACTIVE and lifecycle.fence > 0
         and lifecycle.expires_at > time.time() and lifecycle.owner == module.read_process_birth(os.getpid()),
         "original_actual_fenced_lifecycle_changed")
    need(capture.actual_git_session._closed and capture.actual_git_session._known_stopped()
         and (capture.actual_git_session.consumer is None or capture.actual_git_session.consumer.released),
         "original_preflight_git_custody_not_stopped")
    if before_consumption:
        need(_index_snapshot(capture.gitdir) == capture.index_snapshot
             and _head_ref_snapshot(capture.gitdir, capture.common, capture.branch, capture.baseline) == capture.head_snapshot,
             "original_preflight_head_or_index_changed_before_consumption")
        identity, before_id, before_cid = _original_idle_binding(daemon, state, task, capture.control_plane_connection)
        need(identity.canonical_task_cid == capture.task_cid and identity.canonical_task_key == capture.task_key
             and before_id == dict(capture.before_by_id) and before_cid == dict(capture.before_by_cid),
             "original_preflight_attempt_history_changed")
    return capture


def report_only_preflight_workspace(daemon, state, *, task):
    """Return only the actual captured workspace; this is no portable grant."""
    return _require_preflight(daemon, state, task, before_consumption=True).workspace


def capture_before_attempt(daemon, state, task, attempt, *, consume_attempt, worktree_path, branch_name):
    """Original mark tail requires preflight BEFORE changing active/counters."""
    preflight = _require_preflight(daemon, state, task, before_consumption=True)
    need(consume_attempt is True and type(attempt) is int and attempt == 3
         and canonical(worktree_path) == preflight.workspace and branch_name == preflight.branch,
         "preflight_original_mark_binding_required")
    return _BeforeAttempt(daemon, state, task, preflight.task_projection, preflight.task_cid,
        preflight.task_key, attempt, preflight.before_by_id, preflight.before_by_cid, preflight)


def capture_dispatch(daemon, state, task, attempt, invocation, control_plane_connection):
    """No caller board/workspace/baseline; use the original stopped preflight."""
    before = getattr(daemon, "_report_only_before_attempt", None)
    need(type(before) is _BeforeAttempt and before.daemon is daemon and before.state is state
         and before.task is task and before.attempt == attempt, "actual_original_attempt_capture_required")
    preflight = _require_preflight(daemon, state, task, before_consumption=False)
    need(before.preflight is preflight and control_plane_connection is preflight.control_plane_connection,
         "original_preflight_connection_or_capture_changed")
    need(_index_snapshot(preflight.gitdir) == preflight.index_snapshot
         and _head_ref_snapshot(preflight.gitdir, preflight.common, preflight.branch, preflight.baseline) == preflight.head_snapshot,
         "original_preflight_head_or_index_changed_before_provider")
    runner = importlib.import_module(RUNTIME + "grok_cli_runner")
    need(type(invocation) is runner.OrdinaryCallableInvocation and invocation.cwd == preflight.workspace,
         "actual_callable_invocation_capture_required")
    capture = _Dispatch(before, invocation, preflight.workspace, preflight.baseline,
        preflight.workspace_board, preflight.board_bytes, preflight.branch, preflight.lifecycle,
        control_plane_connection, preflight)
    _check_live_binding(daemon, capture)
    return capture


def _check_live_binding(daemon, capture):
    before = capture.before
    state, task = before.state, before.task
    need(_require_preflight(daemon, state, task, before_consumption=False) is capture.preflight
         and before.preflight is capture.preflight, "original_preflight_dispatch_capture_changed")
    runtime = importlib.import_module("ipfs_accelerate_py.agent_supervisor.context.decision_runtime")
    need(type(daemon.decision_runtime) is runtime.DecisionRuntime
         and daemon.decision_runtime.config.mode is runtime.DecisionRuntimeMode.ENFORCE,
         "actual_enforcing_original_decision_runtime_required")
    need(daemon.report_only_candidate_freezing is True and before.daemon is daemon
         and daemon.decision_runtime is not None and daemon.preserve_task_attempts is True,
         "actual_opt_in_decision_runtime_required")
    need(getattr(daemon, "_report_only_before_attempt", None) is before
         and _task_projection(task) == before.task_projection, "original_task_capture_changed")
    identity = daemon._identity_for_task(task)
    need(identity.canonical_task_cid == before.task_cid and identity.canonical_task_key == before.task_key
         and state.implementation_in_progress is True and state.active_task_id == TASK_ID
         and state.active_task_cid == before.task_cid and state.active_task_key == before.task_key
         and type(state.active_attempt) is int and state.active_attempt == before.attempt
         and state.active_worktree_path == str(capture.workspace) and state.active_branch == capture.branch,
         "actual_active_attempt_binding_changed")
    budget_projection(attempt=before.attempt, cap=daemon.max_task_attempts, task_id=TASK_ID,
        task_cid=before.task_cid, before_by_id=dict(before.before_by_id), before_by_cid=dict(before.before_by_cid),
        current_by_id=dict(state.implementation_attempts), current_by_cid=dict(state.implementation_attempts_by_cid))
    need(read_regular(capture.board, source_sha=BOARD_SHA256) == capture.board_bytes and sha(capture.board_bytes) == BOARD_SHA256,
         "original_board_changed")
    lifecycle = daemon._active_worktree_lifecycle
    need(lifecycle is capture.lifecycle and daemon.worktree_lifecycle.load_workspace(str(capture.workspace)) == lifecycle
         and lifecycle.task_id == TASK_ID and lifecycle.canonical_task_cid == before.task_cid
         and lifecycle.workspace_path == str(capture.workspace) and lifecycle.branch == capture.branch
         and lifecycle.state.value == "active" and lifecycle.fence > 0
         and lifecycle.expires_at > time.time(), "original_actual_fenced_lifecycle_changed")
    module = importlib.import_module(MERGE + "worktree_lifecycle")
    need(lifecycle.owner == module.read_process_birth(os.getpid()), "actual_dispatch_process_birth_changed")
    need(not daemon._implementation_cancel_requested(), "actual_attempt_cancelled")
    row = _read_bound_control_plane(daemon, capture.control_plane_connection)
    expected = {"implementation_in_progress": True, "active_task_id": TASK_ID,
                "active_task_cid": before.task_cid, "active_task_key": before.task_key,
                "active_attempt": before.attempt, "active_worktree_path": str(capture.workspace),
                "active_branch": capture.branch,
                "implementation_attempts": dict(state.implementation_attempts),
                "implementation_attempts_by_cid": dict(state.implementation_attempts_by_cid)}
    need(all(type(row.get(key)) is type(value) and row[key] == value for key, value in expected.items()),
         "authoritative_control_plane_binding_changed")


def _read_bound_control_plane(daemon, connection):
    """SELECT-only through an already-held actual existing local connection.

    No constructor/open/connect/reconnect/load/save/migration occurs here.
    PI-P04 owns acquisition and lifetime. Unsupported transports refuse.
    """
    state_module = importlib.import_module("ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state")
    control = importlib.import_module("ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane")
    need(type(connection) is state_module.DuckDBConnection, "actual_retained_control_plane_connection_required")
    _original_connection(daemon, connection)
    target, _ = control.resolve_control_plane(daemon.state_path)
    need(type(target) is str and not state_module.is_quack_transport_target(target)
         and canonical(Path(target)) == canonical(connection.path), "original_local_control_plane_target_required")
    # The existing public execute() may reconnect a poisoned owner. Holding its
    # actual condition lock and refusing poison/transactions prevents that path.
    with connection._execution_condition:
        need(connection._closed is False and connection._poisoned is False
             and connection._owner_binding_retired is False and connection._transaction_active is False
             and not connection._quack_uri and connection._default_catalog is None, "actual_usable_nontransactional_local_connection_required")
        lane_key = daemon.state_path.stem
        if control._is_board_control_database(target):
            result = connection.execute(
                "SELECT payload_json FROM artefact_generic WHERE board_namespace = ? AND kind = ? AND artefact_id = ?",
                [control._BOARD_NAMESPACE, control._BOARD_KIND, lane_key]).fetchall()
        else:
            need(control.TABLE == "portal_task_state", "original_control_plane_table_changed")
            result = connection.execute("SELECT body_json FROM portal_task_state WHERE lane_key = ?", [lane_key]).fetchall()
    need(len(result) == 1 and len(result[0]) == 1 and type(result[0][0]) is str
         and 0 < len(result[0][0].encode()) <= MAX_DOCUMENT, "one_actual_control_plane_state_row_required")
    return strict_report(result[0][0].encode())


def _source_paths(daemon):
    modules = [sys.modules[__name__], importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon")]
    names = [RUNTIME + "ordinary_success_handoff", RUNTIME + "grok_cli_runner", RUNTIME + "process_security",
             MERGE + "git_checkout_transaction", MERGE + "worktree_lifecycle",
             "ipfs_accelerate_py.agent_supervisor.context.decision_runtime",
             "ipfs_accelerate_py.agent_supervisor.task_sources.task_identity",
             "ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state",
             "ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane"]
    names += [NATIVE + name for name in ("parented_phase_resources", "autoencoder_daemon_resources",
        "resource_scheduler", "existing_resource_scheduler", "autoencoder_resource_recovery",
        "proof_resource_safety", "runtime_telemetry", "validation_run_resources")]
    modules += [importlib.import_module(name) for name in names]
    paths = {str(canonical(Path(module.__file__).absolute())) for module in modules}
    paths.update((str(PYTHON), str(GIT)))
    return paths


def _read_pinned_source(path, expected):
    path = canonical(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid in {0, os.getuid()}
             and before.st_mode & 0o002 == 0 and 0 <= before.st_size <= 256 * 1024 * 1024,
             "bounded_protected_source_required")
        digest, count = hashlib.sha256(), 0
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            count += len(chunk)
            need(count <= before.st_size, "protected_source_grew")
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    need(count == before.st_size and witness(before) == witness(after) == witness(path.lstat())
         and digest.hexdigest() == expected, "protected_implementation_source_changed")


def _verify_sources(daemon, pins):
    need(type(pins) is dict and 0 < len(pins) <= 256
         and all(type(p) is str and type(v) is str and SHA.fullmatch(v) for p, v in pins.items()),
         "exact_protected_source_closure_required")
    for path, digest in pins.items():
        _read_pinned_source(path, digest)
    need(_source_paths(daemon) <= set(pins), "required_freezer_source_closure_missing")
    daemon_module = importlib.import_module("ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon")
    need(type(daemon).__module__ == daemon_module.__name__ and type(daemon) is daemon_module.PortalImplementationDaemon
         and Path(inspect.getsourcefile(type(daemon))).absolute() == Path(daemon_module.__file__).absolute(),
         "actual_source_pinned_daemon_required")


def _guard_owner(owner, provider_owner, *, fresh=False):
    native = importlib.import_module(NATIVE + "parented_phase_resources")
    need(type(owner) is native.ParentedPhaseResourceReservation and owner is not provider_owner,
         "genuine_distinct_freezer_owner_required")
    need(owner.heartbeat(), "actual_freezer_owner_heartbeat_failed")
    owner.check_usage(owner.directory, full=True)
    public, lease = owner.to_dict(), owner.native_lease.to_dict()
    parent = provider_owner.native_lease.to_dict()
    need(public.get("status") == "active" and lease.get("lane") == "orchestration"
         and lease.get("owner_pid") == os.getpid() and lease.get("released") is False
         and lease.get("cancelled") is False and lease.get("cpu_slots", 0) >= 1
         and lease.get("memory_mb", 0) >= 512 and lease.get("child_process_slots", 0) >= 4
         and lease.get("lease_id") != parent.get("lease_id")
         and lease.get("parent_lease_id") and lease["parent_lease_id"] == parent.get("parent_lease_id"),
         "actual_same_phase_finite_freezer_owner_required")
    need(lease.get("cpu_slots") == 1 and lease.get("child_process_slots") == 4,
         "exact_one_cpu_four_slot_sequential_freezer_profile_required")
    if fresh:
        row = public.get("record") or {}
        need(row.get("child") is None and row.get("prior_children", []) == [], "fresh_freezer_owner_required")
    return public


def storage_charge_key(path):
    return "report-freezer-" + sha(str(canonical(path)).encode())


def _discover_git_storage(workspace):
    """Read the original worktree pointer before starting any Git child."""
    marker = canonical(workspace / ".git")
    if marker.is_dir():
        gitdir = marker
    else:
        text = _read_git_metadata(marker, 8192).decode("utf-8")
        need(text.startswith("gitdir: ") and text.endswith("\n") and text.count("\n") == 1,
             "original_regular_gitdir_pointer_required")
        gitdir = canonical(Path(os.path.abspath(workspace / text[8:-1])))
    need(gitdir.is_dir(), "original_git_directory_required")
    common_file = gitdir / "commondir"
    if os.path.lexists(common_file):
        text = _read_git_metadata(common_file, 8192).decode("utf-8")
        need(text.endswith("\n") and text.count("\n") == 1 and text[:-1], "regular_common_git_pointer_required")
        common = canonical(Path(os.path.abspath(gitdir / text[:-1])))
    else:
        common = gitdir
    need(common.is_dir(), "original_common_git_directory_required")
    for config in {common / "config", gitdir / "config.worktree"}:
        if os.path.lexists(config):
            raw = _read_git_metadata(config, 65536)
            need(not re.search(rb"(?im)^\s*\[\s*(include(?:if)?|filter)(?:\s|\.|\])", raw),
                 "git_config_include_or_external_filter_refused")
    # This check is repeated under the lock and immediately before staging.
    return gitdir, common


def _storage_roots(workspace, gitdir, common):
    paths = {canonical(path) for path in (workspace, gitdir, common)}
    return sorted((p for p in paths if not any(q in p.parents for q in paths)), key=str)


def _check_storage(owner, workspace, gitdir, common, declaration):
    roots = _storage_roots(workspace, gitdir, common)
    need(type(declaration) is dict and set(declaration) == {str(p) for p in roots}
         and all(type(v) is int and MIN_ROOT_CHARGE <= v <= MAX_ROOT_CHARGE for v in declaration.values())
         and sum(declaration.values()) <= MAX_EXTERNAL_TOTAL, "bounded_exact_external_storage_declaration_required")
    public = owner.to_dict()
    charges = (public.get("record") or {}).get("external_charges") or {}
    need(all(charges.get(storage_charge_key(path)) == declaration[str(path)] for path in roots),
         "actual_private_output_precharge_required")
    admitted = [Path(row["path"]) for row in public.get("roots", [])]
    need(all(any(root == p or p in root.parents for p in admitted) for root in roots),
         "actual_admitted_storage_roots_required")
    count = 0
    for root in roots:
        need(root.is_dir(), "owned_storage_directory_required")
        total = 0
        for directory, subdirs, files in os.walk(root, followlinks=False):
            for name in [*subdirs, *files]:
                path = Path(directory) / name
                metadata = path.lstat()
                count += 1
                need(count <= 16384 and not stat.S_ISLNK(metadata.st_mode)
                     and metadata.st_uid in {0, os.getuid()} and metadata.st_mode & 0o002 == 0
                     and (stat.S_ISDIR(metadata.st_mode) or (stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1)),
                     "bounded_unaliased_private_storage_inventory_required")
                if stat.S_ISREG(metadata.st_mode):
                    total += metadata.st_size
                    need(total + 16 * 1024 * 1024 <= declaration[str(root)], "private_output_growth_exceeds_precharge")
    owner.check_usage(owner.directory, full=True)


# This body replaces only _Git's process topology. All commit/staging/source
# guards remain in the original source-pinned public delegate.
ANCHOR = """import os,resource,select
for k,v in ((resource.RLIMIT_CPU,30),(resource.RLIMIT_AS,268435456),(resource.RLIMIT_FSIZE,16777216),(resource.RLIMIT_NOFILE,64)):
 resource.setrlimit(k,(v,v))
if not select.select([0],[],[],3600)[0]: raise SystemExit(125)
if os.read(0,1)!=b'': raise SystemExit(125)
"""
_RETAINED_GIT_CUSTODY = {}


class GitCustodyPending(ReportFreezerRefusal):
    """Original same-process refs, never a portable cleanup/admission token."""
    def __init__(self, session, cause):
        super().__init__("actual_git_group_cleanup_pending")
        self.actual_session = session
        self.actual_owner = session.owner
        self.actual_anchor = session.anchor
        self.actual_processes = tuple(session.processes)
        self.actual_consumer = session.consumer
        self.actual_cause = cause


def _git_process(pid):
    try:
        text = (Path('/proc') / str(pid) / 'stat').read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = text.rsplit(')', 1)[1].split()
    need(len(fields) >= 22, "bounded_actual_git_process_identity_required")
    return dict(pid=pid, state=fields[0], parent_pid=int(fields[1]), group_pid=int(fields[2]),
                session_pid=int(fields[3]), birth=fields[19])


def check_git_anchor_projection(row, *, pid, birth, parent_pid, session_pid):
    """Pure necessary shape checks; only actual internal observations use it."""
    need(type(row) is dict and row.get('pid') == pid and row.get('birth') == birth
         and row.get('state') != 'Z' and row.get('parent_pid') == parent_pid
         and row.get('group_pid') == pid and row.get('session_pid') == session_pid,
         "actual_original_git_anchor_identity_required")


def _git_group_members(group_pid):
    entries = list(Path('/proc').iterdir())
    need(len(entries) <= 100000, "bounded_actual_git_group_census_required")
    rows = []
    for entry in entries:
        if entry.name.isdecimal():
            row = _git_process(int(entry.name))
            if row is not None and row['group_pid'] == group_pid and row['state'] != 'Z':
                rows.append(row)
    return rows


class _Git:
    """One genuine attached PGID anchor; finite actual joined Git children.

    The anchor is a direct child and isolated process-group leader in the
    caller's session (process_group=0). This intentionally differs from the
    SDK doc's start_new_session=True example. Actual SDK checks are unchanged:
    direct PPID, PGID==anchorPID and retained birth, with public group sampling.
    """
    def __init__(self, owner, workspace):
        self.owner, self.workspace, self.count = owner, workspace, 0
        self.anchor, self.anchor_birth, self.consumer = None, None, None
        self.processes, self._closed, self._failed, self._pending = [], False, False, None
        self._anchor_start_attempted = False
        self._host_pid, self._host_session = os.getpid(), os.getsid(0)

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None and not self._closed:
            self._stop_failed(error)
        else:
            self.close()
        return False

    def _native_owner(self):
        native = importlib.import_module(NATIVE + 'parented_phase_resources')
        need(type(self.owner) is native.ParentedPhaseResourceReservation,
             "actual_git_parented_native_owner_required")
        public = self.owner.to_dict()
        lease = self.owner.native_lease.to_dict()
        need(public.get('status') == 'active' and lease.get('owner_pid') == os.getpid()
             and lease.get('lane') == 'orchestration' and lease.get('cpu_slots') == 1
             and lease.get('child_process_slots') == 4 and lease.get('memory_mb', 0) >= 512
             and lease.get('released') is False and lease.get('cancelled') is False,
             "actual_one_cpu_four_slot_git_owner_required")
        need(self.owner.heartbeat(), "actual_git_owner_heartbeat_failed")
        if self.consumer is not None:
            resource = importlib.import_module(NATIVE + 'resource_scheduler')
            need(type(self.consumer) is resource.ResourceLease and self.consumer.owner_pid == self._host_pid
                 and self.consumer.parent_lease_id == self.owner.native_lease.lease_id
                 and self.consumer.cpu_slots == 1 and self.consumer.memory_mb == 1
                 and self.consumer.child_process_slots == 0 and not self.consumer.released
                 and not self.consumer.cancelled, 'actual_original_git_consumer_required')
            need(self.consumer.renew() and not self.consumer.cancelled,
                 'original_git_consumer_not_productively_live')

    @staticmethod
    def _environment():
        return {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TERMINAL_PROMPT": "0",
                "GIT_OPTIONAL_LOCKS": "0", "GIT_NO_REPLACE_OBJECTS": "1", "GIT_NO_LAZY_FETCH": "1",
                "GIT_LITERAL_PATHSPECS": "1"}

    def _anchor_identity(self, *, input_open=True):
        need(type(self.anchor) is subprocess.Popen and self.anchor.poll() is None
             and self.anchor_birth is not None and self.anchor.stdin is not None
             and (not input_open or not self.anchor.stdin.closed)
             and os.getpid() == self._host_pid and os.getsid(0) == self._host_session,
             "live_original_git_anchor_object_required")
        row = _git_process(self.anchor.pid)
        check_git_anchor_projection(row, pid=self.anchor.pid, birth=self.anchor_birth,
            parent_pid=self._host_pid, session_pid=self._host_session)
        need(os.getpgrp() != self.anchor.pid, "git_anchor_must_not_be_host_group")
        return row

    def _anchor_only(self):
        self._native_owner()
        self._anchor_identity()
        usage = self.owner.check_usage(self.owner.directory)
        need(usage['group_rss']['live_processes'] == 1,
             "exact_anchor_only_group_required_between_git_commands")
        members = _git_group_members(self.anchor.pid)
        need(len(members) == 1 and members[0]['pid'] == self.anchor.pid
             and members[0]['birth'] == self.anchor_birth,
             "actual_original_anchor_only_census_required")
        need(all(type(p.poll()) is int for p in self.processes), "prior_actual_git_not_reaped")

    def _start(self):
        need(not self._closed and not self._failed and self._pending is None,
             "git_group_session_cannot_restart_after_close_or_failure")
        if self.anchor is not None:
            self._anchor_only()
            return
        self._native_owner()
        need(self.owner.check_usage(self.owner.directory, full=True)['group_rss']['live_processes'] == 0,
             "prior_git_group_not_quiescent_before_anchor")
        need(len(_RETAINED_GIT_CUSTODY) < 16, "bounded_retained_git_custody_inventory_required")
        # This actual child consumer holds original ancestry even if Popen or
        # native attachment has an unknown outcome. It is never a new root.
        self.consumer = self.owner.native_lease.acquire_child(lane='orchestration', cpu_slots=1,
            memory_mb=1, child_process_slots=0, timeout=30, request_id='report-freezer-git-group-hold')
        self._anchor_start_attempted = True
        self.anchor = subprocess.Popen([str(PYTHON), '-I', '-S', '-B', '-c', ANCHOR],
            cwd=self.workspace, env=self._environment(), stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, process_group=0)
        actual = _git_process(self.anchor.pid)
        need(actual is not None, "actual_git_anchor_birth_unavailable")
        self.anchor_birth = actual['birth']
        self._anchor_identity()
        attached = self.owner.check_usage(self.owner.directory, child_pid=self.anchor.pid, full=True)
        need(attached['group_rss']['live_processes'] == 1, "native_anchor_attachment_not_exactly_one")
        child = self.owner.to_dict()['record'].get('child')
        need(child == {'pid': self.anchor.pid, 'birth': self.anchor_birth},
             "actual_native_anchor_attachment_identity_changed")
        self._anchor_only()

    def _known_stopped(self):
        if self.anchor is None:
            return not self._anchor_start_attempted and not self.processes
        need(type(self.anchor) is subprocess.Popen, "retained_actual_anchor_required")
        if self.anchor.poll() is None or any(p.poll() is None for p in self.processes):
            return False
        leader = _git_process(self.anchor.pid)
        need(leader is None or leader['birth'] == self.anchor_birth, "original_git_anchor_pid_reused")
        return not _git_group_members(self.anchor.pid)

    def _release_consumer(self):
        need(self._known_stopped(), "actual_original_git_group_not_stopped")
        if self.consumer is not None:
            self.consumer.release()
            need(self.consumer.released, "actual_git_consumer_release_not_observed")

    def _retain(self, cause):
        self._failed = True
        if self._pending is None:
            self._pending = GitCustodyPending(self, cause)
            _RETAINED_GIT_CUSTODY[id(self)] = self
        raise self._pending from cause

    def _stop_failed(self, cause):
        if self._pending is not None:
            raise self._pending
        self._failed = True
        try:
            if self.anchor is None:
                need(not self._anchor_start_attempted, "actual_anchor_creation_outcome_unknown")
            elif self.anchor.poll() is None:
                # The only group signal uses the still-live original actual
                # anchor Popen/birth and exact isolated group, never a receipt.
                self._anchor_identity(input_open=False)
                os.killpg(self.anchor.pid, signal.SIGKILL)
            for process in [*self.processes, *([self.anchor] if self.anchor is not None else [])]:
                process.wait(timeout=5)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None and not stream.closed:
                        stream.close()
            need(self._known_stopped(), "actual_git_group_cleanup_unknown")
            self._release_consumer()
            self._closed = True
        except BaseException as cleanup_error:
            self._retain(cleanup_error)

    def close(self):
        if self._closed:
            return
        if self._pending is not None:
            raise self._pending
        try:
            if self.anchor is not None:
                self._anchor_only()
                self.anchor.stdin.close()
                need(self.anchor.wait(timeout=5) == 0, "actual_git_anchor_close_failed")
                need(self.owner.check_usage(self.owner.directory)['group_rss']['live_processes'] == 0,
                     "actual_git_group_not_quiescent_at_close")
            self._release_consumer()
            self._closed = True
        except BaseException as error:
            self._stop_failed(error)
            raise

    def run(self, arguments, maximum=65536):
        process, selector = None, None
        buffers = {'stdout': bytearray(), 'stderr': bytearray()}
        try:
            self.count += 1
            need(self.count <= MAX_GIT_CHILDREN, 'finite_git_command_bound_exceeded')
            selector = selectors.DefaultSelector()
            argv = [str(GIT), "--no-optional-locks", "--no-replace-objects", "-c", "core.hooksPath=/dev/null",
                    "-c", "core.fsmonitor=false", "-c", "core.untrackedCache=false", "-c", "commit.gpgsign=false",
                    "-c", "tag.gpgsign=false", "-c", "credential.helper=", "-c", "gc.auto=0",
                    "-c", "maintenance.auto=false", "-c", "diff.external=", *arguments]
            self._start()
            self._anchor_only()
            process = subprocess.Popen([str(PYTHON), '-I', '-S', '-B', '-c', BARRIER, json.dumps(argv)],
                cwd=self.workspace, env=self._environment(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, process_group=self.anchor.pid)
            self.processes.append(process)
            actual = _git_process(process.pid)
            need(actual is not None and actual['parent_pid'] == self._host_pid
                 and actual['group_pid'] == self.anchor.pid and actual['session_pid'] == self._host_session
                 and actual['state'] != 'Z', "actual_gated_git_must_join_original_anchor_group")
            self._anchor_identity()
            usage = self.owner.check_usage(self.owner.directory)
            need(usage['group_rss']['live_processes'] == 2, "exact_anchor_plus_gated_git_required_before_input")
            members = _git_group_members(self.anchor.pid)
            need({row['pid'] for row in members} == {self.anchor.pid, process.pid},
                 "actual_preinput_git_group_members_changed")
            process.stdin.write(b'G')
            process.stdin.close()
            for label in buffers:
                stream = getattr(process, label)
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, label)
            deadline, sample = time.monotonic() + 45, 0.0
            while selector.get_map() or process.poll() is None:
                need(time.monotonic() < deadline, "actual_git_deadline_exceeded")
                if time.monotonic() >= sample:
                    self._native_owner()
                    self._anchor_identity()
                    usage = self.owner.check_usage(self.owner.directory)
                    need(1 <= usage['group_rss']['live_processes'] <= 4, "actual_git_group_exceeded_four_slots")
                    sample = time.monotonic() + .5
                for key, _ in selector.select(.1):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    buffers[key.data].extend(chunk)
                    need(len(buffers[key.data]) <= (maximum if key.data == 'stdout' else 16384),
                         "actual_git_output_bound_exceeded")
            need(process.wait(timeout=1) == 0, "actual_git_command_failed")
            self._anchor_only()
            return bytes(buffers['stdout'])
        except BaseException as error:
            self._stop_failed(error)
            raise
        finally:
            if selector is not None:
                selector.close()
            if process is not None and process.poll() is not None:
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None and not stream.closed:
                        stream.close()

    def commit_runner(self, arguments, *, cwd):
        need(Path(cwd) == self.workspace, "shared_commit_tail_workspace_changed")
        # This adapter is internal and accepts only the original shared tail.
        need(arguments == ["rev-parse", "HEAD"] or (
             len(arguments) == 9 and arguments[:5] == ["-c", "user.name=Implementation Daemon", "-c",
                 "user.email=implementation-daemon@example.invalid", "commit"]
             and arguments[5] == "-m" and arguments[7] == "-m"), "unexpected_shared_commit_tail_command")
        return subprocess.CompletedProcess(arguments, 0, stdout=self.run(arguments).decode(), stderr="")


def acknowledge_stopped_git_custody(pending):
    """Observe only original retained objects; never signal/retry/acquire."""
    need(type(pending) is GitCustodyPending and type(pending.actual_session) is _Git,
         "actual_same_process_git_custody_required")
    session = pending.actual_session
    need(_RETAINED_GIT_CUSTODY.get(id(session)) is session and session._pending is pending
         and pending.actual_owner is session.owner and pending.actual_anchor is session.anchor
         and pending.actual_consumer is session.consumer,
         "actual_original_retained_git_custody_identity_required")
    if not session._known_stopped():
        return False
    session._release_consumer()
    session._closed = True
    session._pending = None
    del _RETAINED_GIT_CUSTODY[id(session)]
    return True


def _index_snapshot(gitdir):
    index = gitdir / "index"
    if not os.path.lexists(index):
        return None
    metadata = index.lstat()
    raw = _read_git_metadata(index, MAX_INDEX)
    return witness(metadata), raw


def _require_callback(daemon, capture, outcome, pins):
    handoff = importlib.import_module(RUNTIME + "ordinary_success_handoff")
    verifier = getattr(handoff, "require_active_ordinary_success", None)
    need(callable(verifier), "genuine_active_callback_api_not_integrated")
    verifier(outcome)
    need(getattr(outcome, "callable_invocation", None) is capture.invocation,
         "original_dispatch_invocation_identity_required")
    _verify_sources(daemon, pins)
    _check_live_binding(daemon, capture)


def freeze(daemon, *, outcome, freezer_owner, protected_source_sha256, declared_external_storage):
    """Concrete public daemon delegate. Requires actual callback + actual capture.

    The original decision runtime supplies current durable mutation permission;
    the exact retained live state/counters/lifecycle additionally fence this
    original attempt. No historical JSON state or provider dataclass is trusted.
    """
    need(sys.dont_write_bytecode is True, "no_bytecode_trusted_host_required")
    capture = getattr(daemon, "_report_only_dispatch", None)
    need(type(capture) is _Dispatch, "actual_callable_dispatch_bridge_capture_absent")
    _verify_sources(daemon, protected_source_sha256)
    _require_callback(daemon, capture, outcome, protected_source_sha256)
    _guard_owner(freezer_owner, outcome.owner, fresh=True)
    gitdir, common = _discover_git_storage(capture.workspace)
    _check_storage(freezer_owner, capture.workspace, gitdir, common, declared_external_storage)
    with _Git(freezer_owner, capture.workspace) as git:
        need(git.run(["rev-parse", "--show-toplevel"]).decode().strip() == str(capture.workspace)
             and git.run(["rev-parse", "--absolute-git-dir"]).decode().strip() == str(gitdir)
             and git.run(["rev-parse", "--path-format=absolute", "--git-common-dir"]).decode().strip() == str(common),
             "actual_original_repository_or_git_storage_changed")
        lock_module = importlib.import_module(MERGE + "git_checkout_transaction")
        lock = getattr(lock_module, "checkout_git_directory_mutation", None)
        need(callable(lock), "native_observed_original_checkout_lock_api_absent")
        def dispatch():
            with lock(gitdir):
                return _freeze_locked(daemon, capture, outcome, freezer_owner, protected_source_sha256,
                                      declared_external_storage, git, gitdir, common)
        return daemon._decision_runtime_mutation("commit", {
            "operation": "freeze_report_only_candidate", "task_id": TASK_ID, "attempt": capture.before.attempt,
            "worktree_path": str(capture.workspace)}, dispatch)

def _freeze_locked(daemon, capture, outcome, owner, pins, storage, git, gitdir, common):
    # Every candidate/index/baseline assertion and the original commit tail is
    # inside one original checkout lock and one decision mutation callback.
    _require_callback(daemon, capture, outcome, pins)
    _guard_owner(owner, outcome.owner)
    _check_storage(owner, capture.workspace, gitdir, common, storage)
    need(_discover_git_storage(capture.workspace) == (gitdir, common), "git_storage_configuration_changed")
    need(git.run(["rev-parse", "--absolute-git-dir"]).decode().strip() == str(gitdir), "checkout_git_directory_changed")
    need(git.run(["rev-parse", "--verify", "HEAD^{commit}"]).decode().strip() == capture.baseline,
         "provider_created_commit_or_baseline_drift_refused")
    need(git.run(["symbolic-ref", "--short", "HEAD"]).decode().strip() == capture.branch, "original_branch_changed")
    # Skip-worktree / assume-unchanged flags hide source dirt from diff/status.
    flags = git.run(["ls-files", "-v", "-z"], MAX_INDEX)
    need(all(row.startswith(b"H ") for row in flags.split(b"\0") if row), "hidden_index_or_submodule_flag_refused")
    stage = git.run(["ls-files", "--stage", "-z"], MAX_INDEX)
    need(all(row.startswith(b"100644 ") or row.startswith(b"100755 ") for row in stage.split(b"\0") if row),
         "submodule_symlink_or_unmerged_index_refused")
    need(all(row.split(b"\t", 1)[0].endswith(b" 0") for row in stage.split(b"\0") if row), "unmerged_index_refused")
    need(git.run(["diff", "--cached", "--name-only", "-z", capture.baseline, "--"], MAX_INDEX) == b"",
         "preexisting_staged_change_refused")
    for args in (["diff", "--name-only", "-z", "--no-ext-diff", "--"],
                 ["ls-files", "--others", "--exclude-standard", "-z"],
                 ["ls-files", "--others", "--ignored", "--exclude-standard", "-z"]):
        allowed_dirty_paths(git.run(args, MAX_INDEX))
    board_relative = capture.preflight.board_relative
    board_oid = parse_tree(git.run(["ls-tree", "-z", capture.baseline, "--", board_relative]), board_relative)
    need(board_oid == capture.preflight.board_oid
         and git.run(["cat-file", "blob", board_oid], MAX_DOCUMENT) == capture.board_bytes, "baseline_board_differs")
    report = canonical(capture.workspace / REPORT)
    raw_report = read_regular(report)
    strict_report(raw_report)
    index_before = _index_snapshot(gitdir)
    need(git.run(["rev-parse", "--verify", "HEAD^{commit}"]).decode().strip() == capture.baseline,
         "baseline_changed_before_stage")
    _require_callback(daemon, capture, outcome, pins)
    need(read_regular(report) == raw_report and _index_snapshot(gitdir) == index_before,
         "report_or_staged_preimage_changed")
    need(_discover_git_storage(capture.workspace) == (gitdir, common), "git_storage_configuration_changed")
    git.run(["add", "--sparse", "-f", "--", REPORT])
    staged_oid = parse_diff(git.run(["diff", "--cached", "--raw", "--abbrev=40", "--no-renames",
        "--no-ext-diff", "-z", capture.baseline, "--"]))
    need(git.run(["cat-file", "blob", staged_oid], MAX_DOCUMENT) == raw_report, "staged_report_blob_differs")
    staged_tree = git.run(["write-tree"]).decode().strip()
    need(OID.fullmatch(staged_tree), "exact_staged_tree_required")
    staged_index = _index_snapshot(gitdir)
    _require_callback(daemon, capture, outcome, pins)
    _guard_owner(owner, outcome.owner)
    _check_storage(owner, capture.workspace, gitdir, common, storage)
    need(read_regular(report) == raw_report and _index_snapshot(gitdir) == staged_index
         and git.run(["rev-parse", "--verify", "HEAD^{commit}"]).decode().strip() == capture.baseline,
         "staged_commit_preimage_changed")
    result = daemon._commit_staged_worktree_changes(capture.workspace, capture.before.task, capture.before.attempt,
        status="report-only candidate", git_runner=git.commit_runner)
    candidate = result.get("commit")
    need(result.get("committed") is True and type(candidate) is str and OID.fullmatch(candidate)
         and candidate != capture.baseline, "actual_shared_tail_candidate_required")
    tree = parse_commit(git.run(["cat-file", "commit", candidate]), capture.baseline)
    need(tree == staged_tree, "actual_committed_tree_changed")
    need(parse_diff(git.run(["diff-tree", "--no-commit-id", "--raw", "--abbrev=40", "--no-renames",
        "--no-ext-diff", "-r", "-z", capture.baseline, candidate, "--"])) == staged_oid,
        "actual_commit_changed_extra_path")
    need(parse_tree(git.run(["ls-tree", "-z", candidate, "--", REPORT]), REPORT) == staged_oid
         and parse_tree(git.run(["ls-tree", "-z", candidate, "--", board_relative]), board_relative) == board_oid,
         "actual_candidate_report_or_board_blob_changed")
    need(git.run(["status", "--porcelain=v1", "-z", "--untracked-files=all"]) == b"",
         "actual_candidate_worktree_not_clean")
    allowed_dirty_paths(git.run(["ls-files", "--others", "--ignored", "--exclude-standard", "-z"], MAX_INDEX))
    need(git.run(["rev-parse", "--verify", "HEAD^{commit}"]).decode().strip() == candidate
         and read_regular(report) == raw_report, "actual_candidate_changed_before_return")
    _require_callback(daemon, capture, outcome, pins)
    _check_storage(owner, capture.workspace, gitdir, common, storage)
    return {"schema": SCHEMA, "committed": True, "candidate_commit": candidate, "candidate_tree": tree,
        "baseline_commit": capture.baseline, "candidate_parent": capture.baseline,
        "repository_root": str(capture.workspace), "board_path": str(capture.board), "board_sha256": BOARD_SHA256,
        "authoritative_repository_root": str(capture.preflight.repo),
        "authoritative_board_path": str(capture.preflight.authoritative_board),
        "authoritative_board_sha256": BOARD_SHA256, "board_relative_path": capture.preflight.board_relative,
        "baseline_board_blob_oid": capture.preflight.board_oid,
        "report_path": str(report), "report_sha256": sha(raw_report), "report_blob_oid": staged_oid,
        "task_id": TASK_ID, "canonical_task_cid": capture.before.task_cid,
        "canonical_task_key": capture.before.task_key, "attempt": 3, "max_task_attempts": 3,
        "historical_consumed_attempts": 2, "all_other_63_tasks_remain_blocked": True,
        "provider_execution_id": outcome.execution_id, "actual_git_children": git.count,
        "independent_candidate_observation_required": True, "native_validation_qualified": False,
        "validation_passed": False, "acceptance_authority": False, "completion_authority": False,
        "board_written": False, "task_state_written": False, "canonical_tasks_accepted": 0}
