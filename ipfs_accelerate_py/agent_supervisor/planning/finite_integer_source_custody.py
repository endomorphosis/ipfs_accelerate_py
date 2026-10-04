"""Detached source custody after trusted application observation callbacks.

This small profile seals regular working files, selected Git identity controls,
the admitted manifest and every selected source/AST CAS object. It neither calls
an application observer nor claims source semantics, process origin, complete
environment attestation, or atomicity against arbitrary concurrent OS writes.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
from typing import Any

from ipfs_datasets_py.logic.software_contracts.content import (
    canonical_dag_json_bytes, cid_for_bytes, cid_for_structured,
)
from ipfs_datasets_py.logic.software_contracts.codebase_ir import StaleCodebaseError
from ipfs_datasets_py.logic.software_contracts.semantic_index.snapshot import (
    _exclusion_raw, _ignored_raw, _safe_raw_path,
)

from .repository_plan_preview import RepositoryPlanPreviewOwner

SCHEMA = "finite-integer-detached-source-custody@1"
LIMITS = {"working_files": 256, "working_directories": 256, "working_scan_entries": 1024,
    "working_file_bytes": 64 * 1024, "git_files": 128, "git_directories": 128,
    "git_scan_entries": 512, "git_file_bytes": 64 * 1024,
    "native_objects": 513, "native_object_bytes": 16 * 1024**2,
    "native_total_bytes": 32 * 1024**2}
_FALSE = {name: False for name in ("source_semantics_verified", "runtime_behavior_verified",
    "behavior_authority", "proof_authority", "execution_authority", "completion_authority",
    "mutation_authority", "production_admitted", "atomicity_attested", "process_origin_attested")}
_SEAL = object()


class SourceCustodyError(ValueError):
    """Current source, catalog or selected immutable evidence changed."""


def _need(condition, message):
    if not condition:
        raise SourceCustodyError(message)


def _plain_wire(value):
    # Native SQL rows contain creation-time floats. Only their byte digest and
    # size enter DAG-JSON material identity; no native float is coerced to int.
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode()


def _identity(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


@dataclass(frozen=True, slots=True)
class _File:
    path: Path
    role: str
    size: int
    sha: str
    witness: tuple[int, ...]

    def material(self):
        return {"path": str(self.path), "role": self.role, "size_bytes": self.size,
                "sha256": self.sha, "initial_read_identity": list(self.witness),
                "validation": ("decoded_staged_identity_not_stat_or_TREE_cache"
                    if self.role == "git:index" else "exact_bytes_mode_path")}


def _read(path, *, role, bound, checkpoint):
    checkpoint()
    path = Path(path)
    _need(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(),
          "source custody requires canonical non-symlink file paths")
    chunks, digest, size = [], hashlib.sha256(), 0
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        _need(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= bound,
              "source custody regular-file byte bound exceeded")
        for block in iter(lambda: stream.read(64 * 1024), b""):
            checkpoint()
            size += len(block)
            _need(size <= bound, "source custody file grew beyond its bound")
            chunks.append(block)
            digest.update(block)
        after = os.fstat(stream.fileno())
        current = path.stat(follow_symlinks=False)
    _need(_identity(before) == _identity(after) == _identity(current)
          and size == before.st_size and path.resolve(strict=True) == path
          and not path.is_symlink(), "source custody descriptor/path changed while reading")
    checkpoint()
    return _File(path, role, size, digest.hexdigest(), _identity(after)), b"".join(chunks)


def _directory(path):
    info = path.stat(follow_symlinks=False)
    _need(stat.S_ISDIR(info.st_mode) and not path.is_symlink()
          and path.resolve(strict=True) == path, "source custody directory alias or type changed")
    # Directory mtime changes on harmless file replacement. Exact inventories
    # and per-file identities provide the stricter, explicit replacement fence.
    return (info.st_dev, info.st_ino, info.st_mode)


def _walk(root, *, exclusions, file_limit, directory_limit, scan_limit, checkpoint):
    files, directories, pending, scanned = [], [("", _directory(root))], [root], 0
    while pending:
        checkpoint()
        directory = pending.pop()
        names = []
        with os.scandir(directory) as children:
            for child in children:
                checkpoint()
                scanned += 1
                _need(scanned <= scan_limit, "source custody inventory scan bound exceeded")
                raw = os.fsencode(str(Path(child.path).relative_to(root)))
                if _ignored_raw(raw, exclusions):
                    continue
                name = _safe_raw_path(raw)
                _need(name is not None, "source custody rejects malformed repository paths")
                names.append((name, Path(child.path)))
        for name, path in sorted(names):
            info = path.stat(follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                directories.append((name, _directory(path)))
                _need(len(directories) <= directory_limit, "source custody directory bound exceeded")
                pending.append(path)
            else:
                _need(stat.S_ISREG(info.st_mode) and not path.is_symlink(),
                      "source custody rejects symlinks and nonregular inventory entries")
                files.append((name, path))
                _need(len(files) <= file_limit, "source custody working-file bound exceeded")
    return tuple(sorted(files)), tuple(sorted(directories))


def _working(owner, exclusions, checkpoint):
    return _walk(owner.repository, exclusions=exclusions, file_limit=LIMITS["working_files"],
        directory_limit=LIMITS["working_directories"], scan_limit=LIMITS["working_scan_entries"],
        checkpoint=checkpoint)


def _git(owner, checkpoint):
    root = owner.repository / ".git"
    _need(root.is_dir() and not root.is_symlink() and root.resolve(strict=True) == root,
          "source custody supports ordinary committed .git directories only; linked gitfiles are unsupported")
    _directory(root)
    for name in ("commondir", "gitdir", "worktrees", "modules", "index.lock", "HEAD.lock",
                 "config.lock", "packed-refs.lock", "shallow.lock", "rebase-apply", "rebase-merge"):
        _need(not os.path.lexists(root / name), "unsupported or changing Git metadata layout: " + name)
    controls = ("HEAD", "index", "config", "packed-refs", "shallow", "config.worktree")
    files, directories = [], [("", _directory(root))]
    absent = []
    for name in controls:
        checkpoint()
        path = root / name
        if os.path.lexists(path):
            info = path.stat(follow_symlinks=False)
            _need(stat.S_ISREG(info.st_mode) and not path.is_symlink(),
                  "source custody Git control is not a regular file")
            files.append((name, path))
        else:
            _need(name not in {"HEAD", "index", "config"}, "required Git identity control is missing")
            absent.append(name)
    for name in ("refs", "info"):
        checkpoint()
        path = root / name
        if not os.path.lexists(path):
            absent.append(name)
            continue
        found, dirs = _walk(path, exclusions=(), file_limit=LIMITS["git_files"],
            directory_limit=LIMITS["git_directories"], scan_limit=LIMITS["git_scan_entries"],
            checkpoint=checkpoint)
        files.extend((name + "/" + relative, value) for relative, value in found)
        directories.extend((name + ("/" + relative if relative else ""), witness)
                           for relative, witness in dirs)
    _need(len(files) <= LIMITS["git_files"] and len(directories) <= LIMITS["git_directories"],
          "source custody Git metadata inventory bound exceeded")
    return tuple(sorted(files)), tuple(sorted(directories)), tuple(sorted(absent))


def _git_commit(files, expected):
    text = files["HEAD"].decode("ascii", "strict").strip()
    if text.startswith("ref: "):
        ref = text[5:]
        _need(ref.startswith("refs/") and _safe_raw_path(ref.encode()) == ref,
              "source custody Git symbolic HEAD is malformed")
        if ref in files:
            text = files[ref].decode("ascii", "strict").strip()
        else:
            matches = []
            for line in files.get("packed-refs", b"").decode("ascii", "strict").splitlines():
                if not line or line.startswith(("#", "^")):
                    continue
                parts = line.split(" ")
                _need(len(parts) == 2, "source custody packed refs are malformed")
                if parts[1] == ref:
                    matches.append(parts[0])
            _need(len(matches) == 1, "source custody symbolic HEAD cannot resolve uniquely")
            text = matches[0]
    _need(text == expected, "source custody Git HEAD differs from admitted source")


def _index_identity(raw, commit):
    """Decode the bounded ordinary index subset in gitformat-index(5).

    Primary format: installed Git 2.43.0 manual
    /usr/share/man/man5/gitformat-index.5.gz. V2/V3, regular stage-zero
    entries without extended flags, SHA-1/SHA-256 and optional TREE only.
    Stat cache fields and TREE contents are not staged source meaning. Split,
    sparse, prefix-compressed V4 and other extensions explicitly refuse.
    """
    width = 20 if len(commit) == 40 else 32
    _need(len(commit) in {40, 64} and len(raw) >= 12 + width and raw[:4] == b"DIRC",
          "source custody Git index header is malformed")
    checksum = hashlib.sha1(raw[:-width]).digest() if width == 20 else hashlib.sha256(raw[:-width]).digest()
    _need(checksum == raw[-width:], "source custody Git index checksum differs")
    version, count = struct.unpack_from(">II", raw, 4)
    _need(version in {2, 3} and count <= LIMITS["working_files"],
          "unsupported Git index version or staged inventory bound")
    offset, end, entries, previous = 12, len(raw) - width, [], None
    for _ in range(count):
        start = offset
        _need(offset + 42 + width <= end, "truncated Git index entry")
        mode = struct.unpack_from(">I", raw, offset + 24)[0]
        blob = raw[offset + 40:offset + 40 + width].hex()
        flags = struct.unpack_from(">H", raw, offset + 40 + width)[0]
        _need(not flags & 0x4000, "extended/sparse/intent-to-add Git index entries are unsupported")
        stage = (flags >> 12) & 3
        _need(mode in {0o100644, 0o100755} and stage == 0,
              "nonregular or unmerged Git index entries are unsupported")
        offset += 42 + width
        nul = raw.find(b"\0", offset, end)
        _need(nul >= offset, "unterminated Git index path")
        path = raw[offset:nul]
        name = _safe_raw_path(path)
        _need(name is not None and ".git" not in name.split("/")
              and (flags & 0xFFF) == min(len(path), 0xFFF), "malformed Git index path or name length")
        key = (path, stage)
        _need(previous is None or key > previous, "Git index paths are duplicate or unsorted")
        previous = key
        offset = start + ((nul + 1 - start + 7) // 8) * 8
        _need(offset <= end and not any(raw[nul:offset]), "Git index entry padding is malformed")
        entries.append({"path": name, "blob_oid": blob, "mode": mode, "stage": stage,
                        "assume_valid": bool(flags & 0x8000)})
    while offset < end:
        _need(offset + 8 <= end, "truncated Git index extension")
        kind, size = raw[offset:offset + 4], struct.unpack_from(">I", raw, offset + 4)[0]
        offset += 8
        _need(kind == b"TREE" and offset + size <= end,
              "unsupported Git index extension or truncated TREE cache")
        offset += size
    _need(offset == end, "Git index content boundary differs")
    return {"schema": "finite-integer-ordinary-git-staged-identity@1",
        "object_format": "sha1" if width == 20 else "sha256", "entries": entries,
        "scope": "path/blob/mode/stage/assume-valid; volatile stat and optional TREE cache excluded"}


def _projection(value):
    if value is None:
        return b"null"
    result = {}
    for field in fields(value):
        member = getattr(value, field.name)
        result[field.name] = ([row.to_dict() for row in member] if type(member) is tuple
                              else member.to_dict())
    return _plain_wire(result)


def _native(owner, manifest, checkpoint):
    checkpoint()
    _need(owner.index.catalog.current(owner.expected_head.repository_id) == owner.expected_head,
          "source custody native current head changed")
    result = []
    for entry in manifest.snapshot.entries:
        checkpoint()
        raw = _projection(owner.index.lookup(manifest, entry.path))
        result.append((entry.path, len(raw), hashlib.sha256(raw).hexdigest()))
    checkpoint()
    return tuple(result)


@dataclass(frozen=True, slots=True)
class FrozenSourceCustody:
    """Private native control. Serialized material alone cannot create custody."""

    _seal: Any
    _owner: RepositoryPlanPreviewOwner
    _manifest: Any
    _projection_rows: tuple
    _files: tuple[_File, ...]
    _working_inventory: tuple
    _git_inventory: tuple
    _git_index: bytes
    _exclusions: tuple[bytes, ...]
    _material: bytes

    @property
    def material_binding(self):
        _need(self._seal is _SEAL, "source custody must be acquired from the native owner")
        return json.loads(self._material)

    def require_current(self, checkpoint=lambda: None):
        _need(self._seal is _SEAL and callable(checkpoint), "native custody and checkpoint required")
        checkpoint()
        self._owner.__post_init__()
        _need(self._owner.expected_head.to_dict() == self.material_binding["head"],
              "source custody selected owner head changed")
        _need(_native(self._owner, self._manifest, checkpoint) == self._projection_rows,
              "source custody native AST projection changed")
        working = _working(self._owner, self._exclusions, checkpoint)
        git = _git(self._owner, checkpoint)
        if working != self._working_inventory:
            raise StaleCodebaseError("source custody repository inventory changed")
        _need(git == self._git_inventory, "source custody Git inventory changed")
        for expected in self._files:
            limit = LIMITS["native_object_bytes"] if expected.role.startswith("cas:") else LIMITS["working_file_bytes"]
            actual, raw = _read(expected.path, role=expected.role, bound=limit, checkpoint=checkpoint)
            if expected.role == "git:index":
                _need(actual.path == expected.path and actual.role == expected.role
                      and actual.witness[2] == expected.witness[2]
                      and canonical_dag_json_bytes(_index_identity(raw, self._manifest.snapshot.git_commit)) == self._git_index,
                      "source custody staged Git index identity changed")
            elif (actual.path, actual.role, actual.size, actual.sha, actual.witness[2]) != (
                    expected.path, expected.role, expected.size, expected.sha, expected.witness[2]):
                error = StaleCodebaseError if expected.role.startswith("working:") else SourceCustodyError
                raise error("source custody artifact or repository bytes changed: " + expected.role)
        # No application/native observer runs after these physical fences.
        _need(_working(self._owner, self._exclusions, checkpoint) == working
              and _git(self._owner, checkpoint) == git, "source custody inventory changed during physical checks")
        checkpoint()
        return self.material_binding


def capture_source_custody(owner, checkpoint=lambda: None):
    """Seal a bounded ordinary Git checkout and its admitted native evidence.

    Gitignored regular extras become explicit custody-only rows. Built-in and
    manifest exclusions retain the native raw-path semantics. Linked worktrees,
    opaque/deleted entries and larger repositories are explicitly unsupported.
    """
    _need(type(owner) is RepositoryPlanPreviewOwner and callable(checkpoint),
          "exact native preview owner and checkpoint required")
    checkpoint()
    owner.__post_init__()
    manifest = owner.index.load(owner.expected_head.manifest_cid)
    _need(manifest.snapshot.snapshot_cid == owner.expected_head.snapshot_cid
          and manifest.ast_revision_id == owner.expected_head.ast_revision_id
          and manifest.snapshot.repository_id == owner.expected_head.repository_id
          and manifest.snapshot.mode in {"git-clean", "git-working"},
          "source custody requires the exact admitted committed Git manifest")
    _need(len(manifest.snapshot.entries) <= LIMITS["working_files"], "admitted source inventory exceeds custody bound")
    projections = _native(owner, manifest, checkpoint)
    exclusions = _exclusion_raw(manifest.snapshot.exclusions)
    working, git = _working(owner, exclusions, checkpoint), _git(owner, checkpoint)
    files, raw_working, raw_git, native_size = [], {}, {}, 0
    for name, path in working[0]:
        row, raw = _read(path, role="working:" + name, bound=LIMITS["working_file_bytes"], checkpoint=checkpoint)
        files.append(row)
        raw_working[name] = raw
    for name, path in git[0]:
        row, raw = _read(path, role="git:" + name, bound=LIMITS["git_file_bytes"], checkpoint=checkpoint)
        files.append(row)
        raw_git[name] = raw
    _git_commit(raw_git, manifest.snapshot.git_commit)
    index_identity = _index_identity(raw_git["index"], manifest.snapshot.git_commit)
    cas_rows = {}

    def add_cas(cid, source, role, expected=None):
        nonlocal native_size
        checkpoint()
        path = owner.index.artifacts.path_for(cid, source=source)
        row, raw = _read(path, role="cas:" + role, bound=LIMITS["native_object_bytes"], checkpoint=checkpoint)
        _need((cid_for_bytes(raw) if source else cid_for_structured(json.loads(raw))) == cid,
              "source custody CAS content identity changed")
        if not source:
            _need(canonical_dag_json_bytes(json.loads(raw)) == raw, "source custody CAS JSON is not canonical")
        if expected is not None:
            _need(raw == expected, "source custody native content differs from frozen evidence")
        if path not in cas_rows:
            native_size += len(raw)
            _need(len(cas_rows) < LIMITS["native_objects"] and native_size <= LIMITS["native_total_bytes"],
                  "source custody native evidence closure exceeds byte/object bounds")
            cas_rows[path] = row
        return raw

    add_cas(manifest.cid, False, "manifest", canonical_dag_json_bytes(manifest.to_dict()))
    admitted = []
    for entry in manifest.snapshot.entries:
        checkpoint()
        _need(not entry.is_opaque and entry.source_cid is not None
              and entry.disposition not in {"staged_deleted", "unstaged_deleted", "conflicted", "opaque"}
              and _safe_raw_path(bytes.fromhex(entry.raw_path_hex)) == entry.path
              and entry.path in raw_working, "source custody rejects opaque, missing, conflicted or noncanonical source")
        raw = add_cas(entry.source_cid, True, "source:" + entry.path)
        if raw != raw_working[entry.path] or entry.size_bytes != len(raw):
            raise StaleCodebaseError("source custody current file differs from admitted captured bytes")
        admitted.append(entry.path)
        unit = next(item for item in manifest.units if item.source_key == entry.source_key)
        if unit.ast_cid is not None:
            record = owner.index.load_ast_artifact(manifest, entry.path)
            add_cas(unit.ast_cid, False, "ast:" + entry.path, canonical_dag_json_bytes(record.to_dict()))
    files.extend(cas_rows.values())
    files = tuple(sorted(files, key=lambda row: str(row.path)))
    body = {"schema": SCHEMA, "profile": "ordinary-git-small-detached-source@1",
        "head": owner.expected_head.to_dict(), "manifest_cid": manifest.cid,
        "semantic_state_cid": manifest.semantic_state.state_cid, "limits": dict(LIMITS),
        "scope": "selected native source/AST closure and bounded current inventory; no transitive Git object/environment attestation",
        "native_exclusions": list(manifest.snapshot.exclusions),
        "inventory_scope": "native raw exclusions; Gitignored extras explicitly sealed as custody-only files",
        "admitted_source_paths": sorted(admitted),
        "custody_only_paths": sorted(set(raw_working) - set(admitted)),
        "working_directories": [{"path": name, "physical_identity": list(witness)} for name, witness in working[1]],
        "git_directories": [{"path": name, "physical_identity": list(witness)} for name, witness in git[1]],
        "absent_git_controls": list(git[2]), "files": [row.material() for row in files],
        "git_staged_identity": index_identity,
        "native_ast_projections": [{"path": path, "size_bytes": size, "sha256": sha}
                                   for path, size, sha in projections], **_FALSE}
    body["custody_cid"] = cid_for_structured(body)
    result = FrozenSourceCustody(_SEAL, owner, manifest, projections, files, working, git,
                                canonical_dag_json_bytes(index_identity), exclusions,
                                canonical_dag_json_bytes(body))
    result.require_current(checkpoint)
    return result


__all__ = ["SCHEMA", "LIMITS", "SourceCustodyError", "FrozenSourceCustody", "capture_source_custody"]
