"""Detached canonical source custody for one authenticated linked allocation.

The original parent/prelaunch ordinary-Git custody remains unchanged. This
additive private profile permits one registered linked checkout and lossless
ref packing at the coding-child boundary. It grants no facts, proofs, process
origin, source semantics, completion or execution authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

from ipfs_datasets_py.logic.software_contracts.content import canonical_dag_json_bytes, cid_for_structured
from ..planning import finite_integer_source_custody as original

SCHEMA = "finite-proof-query-linked-worker-source-custody@1"
PROFILE = "one-owned-linked-allocation-canonical-source@1"
_BASE_SCHEMA = "finite-proof-query-worker-source-baseline@1"
_SEAL = object()
_FALSE = {name: False for name in ("source_semantics_verified", "runtime_behavior_verified",
    "behavior_authority", "proof_authority", "execution_authority", "completion_authority",
    "mutation_authority", "atomicity_attested", "process_origin_attested", "production_admitted")}


class WorkerSourceCustodyError(original.SourceCustodyError):
    """Canonical source or its one native allocation lost a binding."""


def _need(condition, message):
    if not condition:
        raise WorkerSourceCustodyError(message)


def _wire(value):
    # Native allocation records can contain timestamps. Only their native
    # allocation CID, not their float representation, enters this material.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def _text(raw, role):
    try:
        text = raw.decode("ascii", "strict")
    except UnicodeError as error:
        raise WorkerSourceCustodyError(role + " must be ASCII") from error
    _need(text.endswith("\n") and not text.endswith("\n\n") and "\r" not in text
          and "\x00" not in text, role + " is not one canonical Git control line")
    return text[:-1]


def _oid(value, width):
    _need(type(value) is str and re.fullmatch("[0-9a-f]{" + str(width) + "}", value) is not None,
          "linked custody Git object identity is malformed")
    return value


def _ref(value):
    _need(type(value) is str and value.startswith("refs/")
          and original._safe_raw_path(value.encode("ascii", "strict")) == value
          and not any(part.startswith(".") or part.endswith((".", ".lock")) for part in value.split("/"))
          and not any(part in value for part in ("..", "@{", "//", " ", "~", "^", ":", "?", "*", "[", "\\")),
          "linked custody Git ref name is malformed")
    return value


def _references(files, commit):
    """Decode loose/packed refs to the same complete semantic map."""
    width, packed, peeled, previous = len(commit), {}, {}, None
    for line in files.get("packed-refs", b"").decode("ascii", "strict").splitlines():
        if line.startswith("#"):
            _need(line.startswith("# pack-refs with:"), "unknown packed Git ref header")
            continue
        if line.startswith("^"):
            _need(previous is not None and previous.startswith("refs/tags/") and previous not in peeled,
                  "packed Git peeled identity lacks its unique tag")
            peeled[previous] = _oid(line[1:], width)
            continue
        fields = line.split(" ")
        _need(len(fields) == 2, "packed Git ref line is malformed")
        name, value = _ref(fields[1]), _oid(fields[0], width)
        _need(name not in packed, "duplicate packed Git ref")
        packed[name], previous = value, name
    result = {name: {"oid": value, "peeled": peeled.get(name)} for name, value in packed.items()}
    for name, raw in files.items():
        if not name.startswith("refs/"):
            continue
        name, value = _ref(name), _oid(_text(raw, name), width)
        _need(name not in result or result[name]["oid"] == value,
              "loose and packed Git references disagree")
        result[name] = {"oid": value, "peeled": peeled.get(name)}
    return result


def _all_git_files(root, checkpoint):
    files, raw = {}, {}
    def read_control(name, path):
        row, value = original._read(path, role="git:" + name,
            bound=original.LIMITS["git_file_bytes"], checkpoint=checkpoint)
        info = path.lstat()
        _need(info.st_uid == os.geteuid() and info.st_nlink == 1
              and not stat.S_IMODE(info.st_mode) & 0o7113,
              "canonical Git identity control owner, link count or mode is unsafe")
        files[name], raw[name] = row, value
    for name in ("HEAD", "index", "config", "packed-refs", "shallow", "config.worktree"):
        path = root / name
        if os.path.lexists(path):
            read_control(name, path)
    for parent in ("refs", "info"):
        path = root / parent
        if not os.path.lexists(path):
            continue
        found, _directories = original._walk(path, exclusions=(),
            file_limit=original.LIMITS["git_files"], directory_limit=original.LIMITS["git_directories"],
            scan_limit=original.LIMITS["git_scan_entries"], checkpoint=checkpoint)
        for relative, path in found:
            name = parent + "/" + relative
            read_control(name, path)
    _need(len(files) <= original.LIMITS["git_files"], "linked custody Git control count exceeds bound")
    return files, raw


def _scope_fields(scope):
    return (scope, scope._owner, scope._custody, scope._server, scope._source,
        scope._runtime, scope._owner.repository, scope._owner.index,
        scope._owner.expected_head, scope._custody._material,
        canonical_dag_json_bytes(scope.to_dict()))


def _assert_scope(scope, fields):
    from .finite_repository_execution import FrozenFiniteRepositoryExecutionScope
    from .finite_proof_query_execution import FrozenFiniteProofQueryExecutionClosure
    _need(type(scope) is FrozenFiniteRepositoryExecutionScope
          and type(scope._proof_query_closure) is FrozenFiniteProofQueryExecutionClosure
          and scope._advisory_closure is None and scope._runtime is not None,
          "linked worker custody requires its exact native proof-query scope")
    current = _scope_fields(scope)
    names = ("scope", "owner", "source_custody", "server", "task_source", "runtime",
        "canonical_repository", "codebase_index", "selected_head", "source_material", "signed_scope_material")
    _need(len(current) == len(fields) == len(names), "linked worker source scope field population changed")
    for number, (left, right) in enumerate(zip(current, fields)):
        if number == 6:
            # The genuine source producer revalidates its frozen owner by
            # calling __post_init__, which normalizes an equal new Path.
            # Path value/type and all physical source/Git/CAS witnesses stay
            # bound; owners, catalogs and runtime capabilities keep identity.
            same = type(left) is type(right) and left == right
        else:
            same = left is right if number < 8 else left == right
        _need(same, "linked worker source scope was rebound: " + names[number])
    scope._active()
    scope._proof_query_closure._assert_scope(scope)


def _canonical_files(baseline, checkpoint):
    custody, owner = baseline._scope._custody, baseline._scope._owner
    working = original._working(owner, custody._exclusions, checkpoint)
    _need(working == custody._working_inventory, "canonical worker source inventory changed")
    for expected in custody._files:
        if expected.role.startswith("git:"):
            continue
        bound = original.LIMITS["native_object_bytes"] if expected.role.startswith("cas:") else original.LIMITS["working_file_bytes"]
        actual, _raw = original._read(expected.path, role=expected.role, bound=bound, checkpoint=checkpoint)
        _need(actual == expected, "canonical worker source or selected CAS physical witness changed")
    # The execution closure validates the same native head and complete SQL
    # inventory before this last physical fence. Do not invoke a native
    # observer after reading the selected source/CAS witnesses here.
    return working


def _assert_baseline(baseline):
    _need(type(baseline) is FrozenFiniteProofQueryWorkerSourceBaseline
          and baseline._seal is _SEAL, "exact captured worker source baseline required")
    _assert_scope(baseline._scope, baseline._fields)
    body = json.loads(baseline._material)
    _need(body.pop("baseline_cid") == cid_for_structured(body), "worker source baseline material changed")
    custody = baseline._scope._custody
    expected = {row.role.removeprefix("git:"): row for row in custody._files if row.role.startswith("git:")}
    files, raw = dict(baseline._git_files), dict(baseline._git_raw)
    _need(len(files) == len(baseline._git_files) and len(raw) == len(baseline._git_raw)
          and set(files) == set(raw) == set(expected)
          and baseline._commit == custody._manifest.snapshot.git_commit
          and baseline._git_directory == dict(custody._git_inventory[1])[""],
          "cached worker baseline Git identity changed")
    for name, row in files.items():
        _need(type(raw[name]) is bytes and (len(raw[name]), hashlib.sha256(raw[name]).hexdigest()) == (row.size, row.sha),
              "cached worker baseline control bytes changed")
        if name == "index":
            _need(row.witness[2] == expected[name].witness[2]
                  and canonical_dag_json_bytes(original._index_identity(raw[name], baseline._commit)) == custody._git_index,
                  "cached worker baseline staged identity changed")
        else:
            _need(row == expected[name], "cached worker baseline Git witness changed")
    _need(_references(raw, baseline._commit) == json.loads(baseline._refs)
          and body["original_ref_meaning"] == json.loads(baseline._refs)
          and body["git_directory_identity"] == list(baseline._git_directory)
          and body["original_git_controls"] == [row.material() for name, row in sorted(files.items())]
          and body["original_custody_cid"] == custody.material_binding["custody_cid"]
          and body["repository"] == str(baseline._scope._owner.repository)
          and body["baseline_source_commit"] == baseline._commit,
          "cached worker baseline differs from its closed material")


def _canonical_git(baseline, branch, checkpoint):
    custody, root = baseline._scope._custody, baseline._scope._owner.repository / ".git"
    _need(original._directory(root) == baseline._git_directory, "canonical Git directory inode or mode changed")
    for name in ("commondir", "gitdir", "modules", "index.lock", "HEAD.lock", "config.lock",
                 "packed-refs.lock", "shallow.lock", "rebase-apply", "rebase-merge"):
        _need(not os.path.lexists(root / name), "unsupported canonical linked Git layout or lock: " + name)
    files, raw = _all_git_files(root, checkpoint)
    baseline_raw, baseline_files = dict(baseline._git_raw), dict(baseline._git_files)
    expected_names = {name for name in baseline_raw if not name.startswith("refs/") and name != "packed-refs"}
    actual_names = {name for name in raw if not name.startswith("refs/") and name != "packed-refs"}
    _need(actual_names - {"info/refs", "info/packs"} == expected_names - {"info/refs", "info/packs"},
          "canonical Git identity control inventory changed")
    for name in expected_names - {"index", "info/refs", "info/packs"}:
        _need(files[name] == baseline_files[name], "canonical Git identity control bytes/mode/inode changed: " + name)
    _need(files["index"].witness[2] == baseline_files["index"].witness[2]
          and canonical_dag_json_bytes(original._index_identity(raw["index"], baseline._commit)) == custody._git_index,
          "canonical staged Git identity or index mode changed")
    refs = _references(raw, baseline._commit)
    expected = json.loads(baseline._refs)
    _need(branch not in expected, "native linked allocation reused an original source ref")
    expected[branch] = {"oid": baseline._commit, "peeled": None}
    _need(refs == expected, "canonical Git ref meaning changed beyond its one allocated branch")
    original._git_commit(raw, baseline._commit)
    if "info/refs" in raw:
        advertised = {}
        for line in raw["info/refs"].decode("ascii", "strict").splitlines():
            fields = line.split("\t")
            _need(len(fields) == 2 and fields[1] not in advertised, "Git advertised refs are malformed or duplicate")
            advertised[fields[1]] = _oid(fields[0], len(baseline._commit))
        def advertised_meaning(values):
            result = {name: value["oid"] for name, value in values.items()}
            result.update({name + "^{}": value["peeled"] for name, value in values.items() if value["peeled"] is not None})
            return result
        # Native GC can update this non-authoritative advertisement before
        # worktree.add creates its one branch. Actual refs must still equal the
        # complete current map above; only these two exact cache states pass.
        _need(advertised in (advertised_meaning(json.loads(baseline._refs)), advertised_meaning(expected)),
              "Git advertised refs differ from frozen original or current ref meaning")
    if "info/packs" in raw:
        for line in raw["info/packs"].decode("ascii", "strict").splitlines():
            if not line:
                continue
            _need(re.fullmatch(r"P pack-[0-9a-f]{40,64}\.pack", line) is not None,
                  "Git pack advertisement is malformed")
            path = root / "objects/pack" / line[2:]
            _need(path.is_file() and not path.is_symlink() and path.resolve() == path,
                  "Git pack advertisement resolves outside ordinary object storage")
    return files, raw, refs


def _reflog_rows(raw, commit):
    """Exact ordered native rows, tolerating only an empty-message tab."""
    _need(type(raw) is bytes and 0 < len(raw) <= original.LIMITS["git_file_bytes"]
          and raw.endswith(b"\n") and b"\x00" not in raw and b"\r" not in raw,
          "linked Git reflog bytes are unsupported or changing")
    lines = raw[:-1].split(b"\n")
    _need(0 < len(lines) <= 256 and all(lines), "linked Git reflog row inventory is unsupported")
    width = str(len(commit)).encode("ascii")
    header_pattern = (rb"[0-9a-f]{" + width + rb"} [0-9a-f]{" + width
        + rb"} [^\x00\r\n\t<>]+ <[^\x00\r\n\t<>]*> -?[0-9]{1,20} [+-][0-9]{4}")
    rows = []
    for line in lines:
        header, separator, message = line.partition(b"\t")
        _need(re.fullmatch(header_pattern, header) is not None,
              "linked Git reflog header is unsupported or changing")
        timezone = header[-4:]
        _need(int(timezone[:2]) <= 23 and int(timezone[2:]) <= 59,
              "linked Git reflog timezone is unsupported")
        # Git's GC writer adds the separator to an initial no-message row.
        # Every header byte, message byte, row and its order remain exact.
        rows.append((header, message if separator else b""))
    return tuple(rows)


def _linked_git(baseline, material, checkpoint):
    repository = baseline._scope._owner.repository
    root, worktree = repository / ".git", Path(material["worktree_path"])
    _need(worktree.is_absolute() and worktree.resolve(strict=True) == worktree
          and worktree != repository and not worktree.is_relative_to(repository),
          "linked worker checkout aliases canonical source")
    marker, marker_raw = original._read(worktree / ".git", role="allocated:gitfile",
        bound=original.LIMITS["git_file_bytes"], checkpoint=checkpoint)
    marker_info = marker.path.lstat()
    _need(marker_info.st_uid == os.geteuid() and marker_info.st_nlink == 1
          and stat.S_IMODE(marker_info.st_mode) in {0o400, 0o440, 0o600, 0o640, 0o644, 0o660, 0o664},
          "allocated Git marker is not the owner single-link file")
    marker_text = _text(marker_raw, "allocated Git marker")
    _need(marker_text.startswith("gitdir: "), "allocated Git marker lacks its canonical gitdir")
    admin = Path(marker_text[8:])
    _need(admin.is_absolute() and admin.resolve(strict=True) == admin and admin.parent == root / "worktrees",
          "allocated Git marker names an unrelated Git administration path")
    administration_parent = original._directory(admin.parent)
    original._directory(admin)
    for directory in (admin.parent, admin):
        info = directory.lstat()
        _need(info.st_uid == os.geteuid() and not stat.S_IMODE(info.st_mode) & 0o7002,
              "linked Git administration directory owner or mode is unsafe")
    children = []
    with os.scandir(admin.parent) as entries:
        for entry in entries:
            checkpoint()
            children.append(entry.name)
            _need(len(children) <= 1, "canonical source has an additional linked worktree allocation")
    children.sort()
    _need(children == [admin.name], "canonical source has an additional linked worktree allocation")
    files, directories = original._walk(admin, exclusions=(), file_limit=16,
        directory_limit=4, scan_limit=32, checkpoint=checkpoint)
    names = {name for name, _path in files}
    _need({"HEAD", "index", "commondir", "gitdir"} <= names
          and names <= {"HEAD", "index", "commondir", "gitdir", "ORIG_HEAD", "logs/HEAD"},
          "linked Git administration layout is unsupported or changing")
    raw, witnesses = {}, {}
    for name, path in files:
        witnesses[name], raw[name] = original._read(path, role="allocated:" + name,
            bound=original.LIMITS["git_file_bytes"], checkpoint=checkpoint)
        info = path.lstat()
        _need(info.st_uid == os.geteuid() and info.st_nlink == 1
              and not stat.S_IMODE(info.st_mode) & 0o7113,
              "linked Git control owner, link count or mode is unsafe")
    _need(_text(raw["gitdir"], "linked gitdir back-reference") == str(worktree / ".git"),
          "linked Git back-reference names another checkout")
    common = _text(raw["commondir"], "linked common directory")
    _need(common == "../.." and (admin / common).resolve(strict=True) == root,
          "linked Git common directory differs from canonical source")
    branch = _ref("refs/heads/" + material["branch"].removeprefix("refs/heads/"))
    _need(_text(raw["HEAD"], "allocated HEAD") == "ref: " + branch,
          "allocated Git HEAD does not name its registered branch")
    _need(canonical_dag_json_bytes(original._index_identity(raw["index"], baseline._commit)) == baseline._scope._custody._git_index,
          "allocated staged Git identity differs from canonical source")
    if "ORIG_HEAD" in raw:
        _need(_text(raw["ORIG_HEAD"], "allocated ORIG_HEAD") == baseline._commit,
              "allocated original HEAD differs from source baseline")
    # Deployment can give the isolated UID group writes to allocated files.
    # Git's tracked executable meaning and all source bytes remain unchanged.
    expected_working = {row.role.removeprefix("working:"): row for row in baseline._scope._custody._files if row.role.startswith("working:")}
    entries = json.loads(baseline._scope._custody._git_index)["entries"]
    for entry in entries:
        name = entry["path"]
        _need(name in expected_working, "allocated tracked path lacks canonical bounded source custody")
        expected = expected_working[name]
        actual, _raw = original._read(worktree / name, role="allocated-working:" + name,
            bound=original.LIMITS["working_file_bytes"], checkpoint=checkpoint)
        _need((actual.size, actual.sha, bool(actual.witness[2] & 0o111)) == (expected.size, expected.sha, entry["mode"] == 0o100755),
              "allocated tracked source bytes or executable meaning changed")
    return marker, (("..", administration_parent), *tuple(directories)), witnesses, raw, branch, admin


@dataclass(frozen=True, slots=True)
class FrozenFiniteProofQueryWorkerSourceBaseline:
    _seal: Any
    _scope: Any
    _fields: tuple
    _commit: str
    _git_directory: tuple
    _git_files: tuple
    _git_raw: tuple
    _refs: bytes
    _material: bytes

    @property
    def material_binding(self):
        _need(self._seal is _SEAL, "worker source baseline must come from native capture")
        return json.loads(self._material)

    def bind_allocation(self, *, allocation, checkpoint):
        from .finite_proof_query_worktree_allocation import FrozenFiniteProofQueryWorktreeAllocation
        _need(self._seal is _SEAL and callable(checkpoint)
              and type(allocation) is FrozenFiniteProofQueryWorktreeAllocation
              and allocation._scope is self._scope,
              "exact sealed native worktree allocation and worker source baseline required")
        _assert_baseline(self)
        material = FrozenFiniteProofQueryWorktreeAllocation.require_current(allocation, checkpoint=checkpoint)
        _need(material == allocation.material_binding, "native worktree allocation current binding changed")
        _need(material["repository_root"] == str(self._scope._owner.repository)
              and material["baseline_source_commit"] == self._commit,
              "native allocation differs from canonical repository or admitted baseline")
        _canonical_files(self, checkpoint)
        marker, dirs, files, raw, branch, admin = _linked_git(self, material, checkpoint)
        _canonical_git(self, branch, checkpoint)
        reflog = raw.get("logs/HEAD")
        if reflog is not None:
            _reflog_rows(reflog, self._commit)
        body = {"schema": SCHEMA, "profile": PROFILE, "original_custody_cid": self._scope._custody.material_binding["custody_cid"],
            "source_head": self._scope._owner.expected_head.to_dict(), "baseline_source_commit": self._commit,
            "allocation_cid": material["allocation_cid"], "worktree_path": str(Path(material["worktree_path"])),
            "branch": branch, "git_administration_path": str(admin),
            "git_marker": marker.material(), "linked_directories": [[name, list(witness)] for name, witness in dirs],
            "linked_controls": [row.material() for name, row in sorted(files.items())],
            "original_ref_meaning": json.loads(self._refs),
            "scope": "Canonical source/CAS/staged identity plus exactly one authentic linked allocation; lossless ref packing and complete unchanged reflog row meaning permitted; no transitive Git object or environment attestation.",
            "model_off": True, "training_steps": 0, **_FALSE}
        body["custody_cid"] = cid_for_structured(body)
        result = FrozenFiniteProofQueryWorkerSourceCustody(_SEAL, self, allocation,
            _wire(material), marker, dirs, tuple(sorted(files.items())), branch, admin, reflog, canonical_dag_json_bytes(body))
        result.require_current(checkpoint=checkpoint)
        return result


@dataclass(frozen=True, slots=True)
class FrozenFiniteProofQueryWorkerSourceCustody:
    _seal: Any
    _baseline: FrozenFiniteProofQueryWorkerSourceBaseline
    _allocation: Any
    _allocation_material: bytes
    _marker: Any
    _directories: tuple
    _files: tuple
    _branch: str
    _admin: Path
    _reflog: bytes | None
    _material: bytes

    @property
    def _scope(self):
        return self._baseline._scope

    @property
    def material_binding(self):
        _need(self._seal is _SEAL, "worker source custody requires its native producer")
        return json.loads(self._material)

    def require_current(self, checkpoint):
        _need(type(self) is FrozenFiniteProofQueryWorkerSourceCustody and self._seal is _SEAL
              and self._baseline._seal is _SEAL and callable(checkpoint), "exact live worker source custody required")
        baseline = self._baseline
        _assert_baseline(baseline)
        from .finite_proof_query_worktree_allocation import FrozenFiniteProofQueryWorktreeAllocation
        _need(type(self._allocation) is FrozenFiniteProofQueryWorktreeAllocation
              and self._allocation._scope is baseline._scope,
              "worker source allocation was rebound")
        allocation = FrozenFiniteProofQueryWorktreeAllocation.require_current(self._allocation, checkpoint=checkpoint)
        _need(_wire(allocation) == self._allocation_material and allocation == self._allocation.material_binding,
              "native worktree allocation material changed after custody binding")
        working = _canonical_files(baseline, checkpoint)
        marker, dirs, files, raw, branch, admin = _linked_git(baseline, allocation, checkpoint)
        # The owner wrapper changes only allocated Git marker mode 0644→0440
        # while giving its isolated UID read access. The inode and bytes remain
        # bound; no canonical repository control is changed by that delegation.
        allowed_marker_modes = {stat.S_IFREG | 0o440, self._marker.witness[2]}
        _need(marker.path == self._marker.path and marker.sha == self._marker.sha
              and marker.size == self._marker.size and marker.witness[:2] == self._marker.witness[:2]
              and marker.witness[2] in allowed_marker_modes,
              "allocated Git marker bytes, inode or delegated mode changed")
        _need(dirs == self._directories and branch == self._branch and admin == self._admin,
              "linked Git directories or registered branch were replaced")
        expected_files = dict(self._files)
        original_reflog = expected_files.get("logs/HEAD")
        _need((original_reflog is None) == (self._reflog is None),
              "cached linked Git reflog raw inventory changed")
        original_rows = None
        if original_reflog is not None:
            _need(type(self._reflog) is bytes and 0 < len(self._reflog) <= original.LIMITS["git_file_bytes"]
                  and len(self._reflog) == original_reflog.size
                  and hashlib.sha256(self._reflog).hexdigest() == original_reflog.sha,
                  "cached original linked Git reflog raw bytes changed")
            original_rows = _reflog_rows(self._reflog, baseline._commit)
        body = json.loads(self._material)
        _need(body.pop("custody_cid") == cid_for_structured(body)
              and body["allocation_cid"] == allocation["allocation_cid"]
              and body["original_custody_cid"] == baseline._scope._custody.material_binding["custody_cid"]
              and body["original_ref_meaning"] == json.loads(baseline._refs)
              and body["git_marker"] == self._marker.material()
              and body["linked_directories"] == [[name, list(witness)] for name, witness in self._directories]
              and body["linked_controls"] == [row.material() for name, row in sorted(expected_files.items())]
              and body["branch"] == self._branch and body["git_administration_path"] == str(self._admin)
              and body["worktree_path"] == allocation["worktree_path"]
              and body["source_head"] == baseline._scope._owner.expected_head.to_dict(),
              "cached linked source custody differs from its closed material")
        _need(set(files) == set(expected_files), "linked Git control inventory changed")
        for name in files:
            if name == "index":
                _need(files[name].witness[2] == expected_files[name].witness[2]
                      and canonical_dag_json_bytes(original._index_identity(raw[name], baseline._commit)) == baseline._scope._custody._git_index,
                      "linked staged Git identity or index mode changed")
            elif name == "logs/HEAD":
                # Native GC rewrites this non-authoritative log inode and
                # emits a tab for an empty message. Its complete original
                # row meaning, path, mode, owner and single-link check remain.
                _need(files[name].path == expected_files[name].path
                      and files[name].witness[2] == expected_files[name].witness[2]
                      and _reflog_rows(raw[name], baseline._commit) == original_rows,
                      "linked Git reflog row meaning, path or mode changed")
            else:
                _need(files[name] == expected_files[name], "linked Git control bytes, inode or mode changed: " + name)
        _canonical_git(baseline, branch, checkpoint)
        _need(original._working(baseline._scope._owner, baseline._scope._custody._exclusions, checkpoint) == working,
              "canonical source inventory changed during linked physical checks")
        _canonical_files(baseline, checkpoint)
        _canonical_git(baseline, branch, checkpoint)
        checkpoint()
        return self.material_binding


def capture_worker_source_custody_baseline(*, scope, checkpoint):
    """Private native broker constructor seed, captured before allocation."""
    _need(callable(checkpoint), "native source custody checkpoint required")
    fields = _scope_fields(scope)
    _assert_scope(scope, fields)
    scope._physical_source()
    custody, repository = scope._custody, scope._owner.repository
    root, commit = repository / ".git", custody._manifest.snapshot.git_commit
    files, raw = _all_git_files(root, checkpoint)
    original._git_commit(raw, commit)
    expected_git = {row.role.removeprefix("git:"): row for row in custody._files if row.role.startswith("git:")}
    _need(set(files) == set(expected_git), "original Git source controls changed before worker baseline")
    for name, row in files.items():
        if name == "index":
            _need(row.witness[2] == expected_git[name].witness[2]
                  and canonical_dag_json_bytes(original._index_identity(raw[name], commit)) == custody._git_index,
                  "original source staged identity changed before linked allocation")
        else:
            _need(row == expected_git[name], "original Git control witness changed before linked allocation")
    refs = _references(raw, commit)
    body = {"schema": _BASE_SCHEMA, "profile": PROFILE,
        "original_custody_cid": custody.material_binding["custody_cid"], "repository": str(repository),
        "baseline_source_commit": commit, "original_ref_meaning": refs,
        "git_directory_identity": list(original._directory(root)),
        "original_git_controls": [row.material() for name, row in sorted(files.items())], "model_off": True,
        "training_steps": 0, **_FALSE}
    body["baseline_cid"] = cid_for_structured(body)
    result = FrozenFiniteProofQueryWorkerSourceBaseline(_SEAL, scope, fields, commit,
        original._directory(root), tuple(sorted(files.items())), tuple(sorted(raw.items())),
        canonical_dag_json_bytes(refs), canonical_dag_json_bytes(body))
    _canonical_files(result, checkpoint)
    scope._physical_source()
    return result


FrozenWorkerSourceCustody = FrozenFiniteProofQueryWorkerSourceCustody


__all__ = ["SCHEMA", "PROFILE", "WorkerSourceCustodyError", "FrozenWorkerSourceCustody", "FrozenFiniteProofQueryWorkerSourceBaseline",
    "FrozenFiniteProofQueryWorkerSourceCustody", "capture_worker_source_custody_baseline"]
