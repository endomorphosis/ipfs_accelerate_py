"""Finite, non-model repository preparation beneath the reviewed generic owner.

This worker creates no native authority. The original run_image_owner_v05.py
must admit and attach its actual process group before running it. All Git
children inherit that group; the original owner retains group cleanup custody.
Portable specifications, output JSON and repository paths grant no authority.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import stat
import subprocess
import sys
import time

GIT = "/usr/bin/git"
SOURCE = Path("/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-ordinary-worker-20261016-01/controller")
HEAD = "89dc25dbb32569bd668940af74c611221aa31148"
TREE = "d43a28fb17aa19976a6886c76576e2c8bef66160"
BRANCH = "agent/legal-ir-ordinary-worker-20261016"
BOARD = "docs/architecture/legal_ir_supervisor_start_20261016/tasks.todo.md"
MAX_FILES = 100000
MAX_OUTPUT = 16 * 1024 * 1024
RESERVE_MARGIN = 16 * 1024 * 1024


def need(value, reason):
    if not value:
        raise RuntimeError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(path):
    path = Path(path)
    need(path.is_absolute() and str(path) == os.path.abspath(path)
         and not any(p.is_symlink() for p in (path, *path.parents)), "canonical_unaliased_path_required")
    return path


def read(path, maximum=64 * 1024 * 1024):
    path = canonical(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
             and before.st_uid in {0, os.getuid()} and not before.st_mode & 0o002
             and 0 < before.st_size <= maximum, "stable_owned_bounded_regular_input_required")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    witness = lambda m: (m.st_dev, m.st_ino, m.st_size, m.st_mode, m.st_uid,
                         m.st_gid, m.st_nlink, m.st_mtime_ns, m.st_ctime_ns)
    need(len(raw) == before.st_size and witness(before) == witness(after) == witness(path.lstat()),
         "source_changed_during_read")
    return raw


def wire(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


class Worker:
    def __init__(self, spec):
        self.spec = spec
        self.attempt = canonical(spec["attempt_directory"])
        need(self.attempt.is_dir() and os.environ.get("TMPDIR") == str(self.attempt),
             "original_generic_owner_attempt_required")
        os.umask(0o022)
        self.repo = self.attempt / "repository"
        need(not os.path.lexists(self.repo), "fresh_repository_required")
        self.mode = spec["mode"]
        need(self.mode in {"full-clone", "selected-source-fixture"}, "explicit_fixture_mode_required")
        self.limit = spec["storage_bytes"]
        need(type(self.limit) is int and self.limit == (3 * 1024**3 if self.mode == "full-clone" else 256 * 1024**2),
             "exact_reviewed_storage_budget_required")
        self.deadline = time.monotonic() + 1700
        self.commands = []
        self.result = {"schema": "legal-ir-repository-preparation-result@1", "status": "started",
                       "mode": self.mode, "original_head": HEAD, "original_tree": TREE,
                       "repository": str(self.repo), "completion_authority": False,
                       "runtime_deployed": False, "providers_called": False,
                       "canonical_attempts_changed": False, "github_qualification_performed": False,
                       "requested_full_original_committed_snapshot": self.mode == "full-clone",
                       "full_original_committed_snapshot": False}
        self.environment = dict(PATH="/usr/bin:/bin", LANG="C", LC_ALL="C", HOME=str(self.attempt),
            XDG_CONFIG_HOME=str(self.attempt / "no-xdg-config"), TMPDIR=str(self.attempt),
            GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null", GIT_OPTIONAL_LOCKS="0",
            GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="/bin/false", GIT_LFS_SKIP_SMUDGE="1",
            GIT_ATTR_NOSYSTEM="1", GIT_NO_REPLACE_OBJECTS="1", GIT_PROTOCOL_FROM_USER="0",
            OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        # Git propagates command-scope config to upload-pack/pack-objects. The
        # local transport reads original objects; no source hooks or filters run.
        self.prefix = [GIT, "-c", "core.hooksPath=/nonexistent", "-c", "gc.auto=0",
            "-c", "maintenance.auto=false", "-c", "pack.threads=1", "-c", "pack.windowMemory=64m",
            "-c", "pack.deltaCacheSize=32m", "-c", "core.autocrlf=false",
            "-c", "filter.lfs.required=false", "-c", "filter.lfs.smudge=",
            "-c", "filter.lfs.clean=", "-c", "filter.lfs.process=",
            "-c", "protocol.allow=never",
            "-c", "protocol.file.allow=always"]

    def write(self, name, raw):
        need(type(raw) is bytes and len(raw) <= MAX_OUTPUT, "bounded_output_required")
        target = self.attempt / name
        need(target.parent == self.attempt and not target.is_symlink(), "private_output_only")
        temporary = self.attempt / (name + ".tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, target)
        self.fsync_dir(self.attempt)

    @staticmethod
    def fsync_dir(path):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try: os.fsync(fd)
        finally: os.close(fd)

    def persist(self):
        self.result["commands"] = self.commands
        self.write("repository-result.json", wire(self.result))

    def usage(self):
        size = 0; entries = 0
        for base, directories, files in os.walk(self.attempt, followlinks=False):
            for name in [*directories, *files]:
                metadata = (Path(base) / name).lstat()
                entries += 1; size += metadata.st_size
                need(entries <= MAX_FILES and size + RESERVE_MARGIN <= self.limit,
                     "finite_attempt_inventory_or_storage_limit_reached")
        need(time.monotonic() < self.deadline, "finite_worker_deadline_reached")
        self.result["last_attempt_apparent_bytes"] = size
        self.result["last_attempt_entries"] = entries
        return size

    def run(self, arguments, *, cwd=None, maximum=MAX_OUTPUT):
        need(len(self.commands) < 48, "finite_git_command_count_required")
        command = self.prefix + list(arguments)
        self.usage()
        child = subprocess.Popen(command, cwd=cwd or self.attempt, env=self.environment,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # No new session/group: all descendants stay in the original admitted
        # worker group. On failure the outer owner kills/reaps that actual group.
        row = {"arguments": list(arguments), "pid": child.pid, "status": "running"}
        self.commands.append(row); self.persist()
        output = {"stdout": bytearray(), "stderr": bytearray()}
        with selectors.DefaultSelector() as selector:
            for name in output:
                stream = getattr(child, name); os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            next_check = 0.0
            while selector.get_map() or child.poll() is None:
                for key, _ in selector.select(0.1):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if not block:
                        selector.unregister(key.fileobj); key.fileobj.close(); continue
                    output[key.data].extend(block)
                    need(len(output[key.data]) <= maximum, "bounded_git_output_exceeded")
                if time.monotonic() >= next_check:
                    self.usage(); next_check = time.monotonic() + 1.0
        code = child.wait()
        row.update(status="stopped", returncode=code,
            stdout_bytes=len(output["stdout"]), stderr_bytes=len(output["stderr"]),
            stdout_sha256=sha(output["stdout"]), stderr_sha256=sha(output["stderr"]))
        self.persist()
        need(code == 0, "genuine_git_command_failed:" + arguments[0])
        return bytes(output["stdout"])

    def original_guard(self):
        need(read(SOURCE / ".git/HEAD") == ("ref: refs/heads/" + BRANCH + "\n").encode(),
             "original_symbolic_head_changed")
        need(read(SOURCE / ".git/refs/heads" / BRANCH) == (HEAD + "\n").encode(),
             "original_owned_branch_changed")
        result = {}
        for relative in (".git/HEAD", ".git/config", ".git/index", ".git/packed-refs",
                         ".git/objects/info/alternates", ".git/refs/heads/" + BRANCH):
            result[relative] = sha(read(SOURCE / relative))
        for relative, digest in self.spec["selected_source_sha256"].items():
            need(sha(read(SOURCE / relative)) == digest, "original_selected_source_changed")
            result[relative] = digest
        return result

    def prepare(self):
        original = self.original_guard()
        need(self.run(["-C", str(SOURCE), "rev-parse", "--verify", HEAD + "^{tree}"]).strip().decode() == TREE,
             "original_exact_committed_tree_changed")
        committed = {}
        for relative, digest in self.spec["selected_source_sha256"].items():
            raw = self.run(["-C", str(SOURCE), "cat-file", "blob", HEAD + ":" + relative])
            need(sha(raw) == digest, "selected_source_not_exact_owned_committed_blob")
            committed[relative] = raw
        if self.mode == "full-clone":
            self.run(["clone", "--no-local", "--no-hardlinks", "--no-checkout", "--no-tags",
                      "--single-branch", "--branch", BRANCH, str(SOURCE), str(self.repo)])
            self.run(["-C", str(self.repo), "checkout", "--force", "--no-guess", BRANCH])
            need(read(self.repo / ".git/HEAD") == ("ref: refs/heads/" + BRANCH + "\n").encode(),
                 "clone_original_symbolic_branch_association_changed")
            need(self.run(["-C", str(self.repo), "rev-parse", "--verify", "HEAD"]).strip().decode() == HEAD,
                 "clone_head_does_not_equal_original_committed_head")
            need(self.run(["-C", str(self.repo), "rev-parse", "--verify", "HEAD^{tree}"]).strip().decode() == TREE,
                 "clone_tree_does_not_equal_original_committed_tree")
        else:
            self.repo.mkdir(mode=0o700)
            self.run(["-C", str(self.repo), "init", "-b", "fixture-legal-ir-selected-sources"])
            for relative, raw in committed.items():
                target = self.repo / relative; target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                with target.open("xb") as stream:
                    os.fchmod(stream.fileno(), 0o644); stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            self.run(["-C", str(self.repo), "add", "--", *committed])
            self.run(["-C", str(self.repo), "-c", "user.name=Legal IR Engineering Fixture",
                "-c", "user.email=legal-ir-fixture@example.invalid", "commit", "--no-gpg-sign",
                "-m", "Private selected-source engineering fixture; no canonical task work"])
            self.result["scope_limitation"] = "Fresh selected-source fixture only; not the original complete tree or runtime qualification."
        gitdir = self.repo / ".git"
        need(gitdir.is_dir() and not gitdir.is_symlink(), "standalone_regular_git_directory_required")
        for relative in ("objects/info/alternates", "shallow", "info/grafts", "index.lock"):
            need(not os.path.lexists(gitdir / relative), "repository_has_forbidden_storage_indirection")
        need(not list((gitdir / "objects/pack").glob("*.promisor")), "partial_clone_promisor_forbidden")
        for base, directories, files in os.walk(gitdir, followlinks=False):
            for name in [*directories, *files]:
                metadata = (Path(base) / name).lstat()
                need(metadata.st_uid == os.getuid() and
                     (stat.S_ISDIR(metadata.st_mode) or
                      (stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1)),
                     "private_git_metadata_contains_symlink_hardlink_or_special_file")
        self.run(["-C", str(self.repo), "fsck", "--full", "--strict", "--no-reflogs"])
        need(self.run(["-C", str(self.repo), "status", "--porcelain=v1", "-z", "--untracked-files=all"]) == b"",
             "prepared_checkout_not_clean")
        prepared_head = self.run(["-C", str(self.repo), "rev-parse", "--verify", "HEAD"]).strip().decode()
        prepared_tree = self.run(["-C", str(self.repo), "rev-parse", "--verify", "HEAD^{tree}"]).strip().decode()
        raw = self.run(["-C", str(self.repo), "ls-tree", "-r", "-z", "--full-tree", "HEAD"])
        self.write("tracked-tree.nul", raw)
        rows = []; unsupported = []; modes = {}
        need(raw.endswith(b"\0"), "nul_terminated_committed_tree_required")
        for entry in raw[:-1].split(b"\0"):
            header, tab, relative_raw = entry.partition(b"\t")
            mode, kind, oid = header.decode("ascii").split(" ")
            need(tab and re.fullmatch("[0-9a-f]{40}", oid), "exact_tree_blob_identity_required")
            relative = relative_raw.decode("utf-8")
            need(relative and not relative.startswith("/") and not any(p in {"", ".", "..", ".git"} for p in relative.split("/")),
                 "safe_committed_relative_path_required")
            path = self.repo / relative; metadata = path.lstat()
            modes[mode] = modes.get(mode, 0) + 1
            if mode in {"100644", "100755"}:
                need(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1 and metadata.st_uid == os.getuid()
                     and bool(metadata.st_mode & 0o111) == (mode == "100755"), "fresh_regular_blob_checkout_mode_changed")
                digest = hashlib.sha1(("blob " + str(metadata.st_size) + "\0").encode())
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
                need(digest.hexdigest() == oid, "checked_out_regular_blob_differs_from_committed_tree")
            elif mode == "120000":
                need(stat.S_ISLNK(metadata.st_mode), "committed_symlink_checkout_changed")
                target = os.readlink(os.fsencode(path))
                need(hashlib.sha1(("blob " + str(len(target)) + "\0").encode() + target).hexdigest() == oid,
                     "committed_symlink_blob_changed")
                unsupported.append({"path": relative, "mode": mode, "reason": "tracked_symlink_refused_by_current_freezer_profile"})
            elif mode == "160000":
                unsupported.append({"path": relative, "mode": mode, "reason": "tracked_submodule_refused_by_current_freezer_profile"})
            else: raise RuntimeError("unexpected_committed_tree_mode")
            rows.append(relative)
            need(len(rows) <= 65536, "bounded_committed_tree_inventory_required")
            if len(rows) % 256 == 0: self.usage()
        flags = self.run(["-C", str(self.repo), "ls-files", "-v", "-z"])
        need(flags.endswith(b"\0"), "bounded_nul_terminated_index_flags_required")
        flags_rows = flags[:-1].split(b"\0")
        need(len(flags_rows) == len(rows) and all(len(row) >= 3 and row[1:2] == b" " for row in flags_rows)
             and {row[2:].decode("utf-8") for row in flags_rows} == set(rows), "index_flags_do_not_match_exact_tree_paths")
        hidden_flags_absent = all(row.startswith(b"H ") for row in flags_rows)
        if not hidden_flags_absent:
            unsupported.append({"reason": "prepared_index_flags_refused_by_current_freezer_profile"})
        for relative, digest in self.spec["selected_source_sha256"].items():
            need(sha(read(self.repo / relative)) == digest, "prepared_selected_source_not_exact_committed_blob")
        need(self.original_guard() == original, "original_selected_source_or_git_metadata_mutated")
        if len(rows) > 16384:
            unsupported.append({"reason": "current_freezer_tracked_inventory_limit_exceeded", "tracked_count": len(rows)})
        self.result.update(status="materialized", tracked_paths=len(rows), tracked_modes=modes,
            prepared_head=prepared_head, prepared_tree=prepared_tree,
            full_original_committed_snapshot=self.mode == "full-clone",
            tracked_tree_sha256=sha(raw), self_contained_objects_verified=True,
            exact_checked_out_blob_identity_verified=True, hidden_index_flags_absent=hidden_flags_absent,
            original_selected_source_and_git_metadata_unchanged=True,
            current_freezer_repository_profile_supported=not unsupported,
            unsupported_profile=unsupported, selected_source_sha256=self.spec["selected_source_sha256"])
        self.usage(); self.persist()


def main():
    spec = json.loads(read(canonical(sys.argv[1]), 1024 * 1024))
    need(spec["schema"] == "legal-ir-repository-preparation-spec@1", "exact_spec_schema_required")
    from parent_attachment_gate import wait_for_original_generic_parent
    request_path = Path(__file__).absolute().parent / ("full-clone-request.json" if spec["mode"] == "full-clone"
                                                       else "selected-source-request.json")
    attachment = wait_for_original_generic_parent(request_path,
        attempt_directory=Path(spec["attempt_directory"]),
        expected_request_name=Path(spec["attempt_directory"]).name,
        storage_bytes=spec["storage_bytes"],
        worker_command=["/usr/bin/python3.12", "-B", str(Path(__file__).absolute()), str(canonical(sys.argv[1]))])
    # No constructor or Git command runs before the actual original-parent
    # attachment-order witness. Actual SDK capabilities stay in that parent.
    worker = Worker(spec)
    worker.result["parent_attachment_order_observation"] = attachment
    try: worker.prepare()
    except BaseException as error:
        worker.result.update(status="failed", error_type=type(error).__name__, error=str(error)[:512],
                             outer_owner_must_reap_original_group=True)
        worker.persist()
        raise


if __name__ == "__main__":
    main()
