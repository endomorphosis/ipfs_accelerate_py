"""Bounded, reversible repository snapshots with optional compact Python views.

The package retains every included original byte. Compact views are a separate
representation, never a proof, execution authority, or substitute source of truth.
Only permission bits (0777), directories, bytes and relative symlink targets are
preserved; ownership, timestamps, ACLs, extended attributes and special bits are
outside this format. Git metadata named .git is excluded at every depth.
"""
from __future__ import annotations

import base64
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import uuid

from . import code_source_view

SCHEMA = "repository-source-view@1"
MAX_FILES = 4096
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_FILE_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_PATH_BYTES = 1024
MAX_DEPTH = 64
SCOPE = {
    "preserved": ["regular_file_bytes", "directory_entries", "permission_bits_0777", "relative_symlink_targets"],
    "not_preserved": ["ownership", "timestamps", "acls", "extended_attributes", "special_permission_bits"],
    "ast_check": "capture_time_only",
    "proof_authority": False,
    "behavioral_equivalence_verified": False,
    "semantic_cid": False,
}


class RepositorySourceViewError(ValueError):
    """Bounded, content-free error suitable for a tool response."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


RepositoryViewError = RepositorySourceViewError


def _fail(code):
    raise RepositorySourceViewError(code)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _is_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _integer(value, lower, upper):
    return type(value) is int and lower <= value <= upper


def _absolute(path):
    try:
        return Path(os.path.abspath(os.fspath(path)))
    except (TypeError, ValueError, OSError):
        _fail("invalid_location")


def _source_anchor(value):
    if (not isinstance(value, str) or not value.startswith("/") or len(value) > 4096
            or any(ord(c) < 32 or ord(c) == 127 for c in value) or str(_absolute(value)) != value):
        _fail("invalid_source_root")
    try:
        value.encode("utf-8")
    except UnicodeError:
        _fail("invalid_source_root")
    return Path(value)


def _disjoint(left, right):
    if left == right or left in right.parents or right in left.parents:
        _fail("overlapping_locations")


def _open_directory(path):
    """Open each absolute component without following symlinks."""
    path = _absolute(path)
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for piece in path.parts[1:]:
            child = os.open(piece, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except OSError:
        os.close(fd)
        _fail("unsafe_or_missing_directory")


def _relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        _fail("invalid_path")
    try:
        valid_size = len(value.encode("utf-8")) <= MAX_PATH_BYTES
    except UnicodeError:
        _fail("invalid_path")
    parts = value.split("/")
    if (not valid_size or len(parts) > MAX_DEPTH or PurePosixPath(value).is_absolute()
            or any(part in {"", ".", "..", ".git"} for part in parts)
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        _fail("invalid_path")
    return value


def _target(value):
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value:
        _fail("unsafe_symlink")
    try:
        if len(value.encode("utf-8")) > MAX_PATH_BYTES:
            _fail("unsafe_symlink")
    except UnicodeError:
        _fail("unsafe_symlink")
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        _fail("unsafe_symlink")
    return value


def _check_symlinks(entries):
    # Resolve through the manifest, not the live filesystem. Checking lexical
    # targets alone misses escapes through a second symlink followed by '..'.
    for path, entry in entries.items():
        if entry["kind"] != "symlink":
            continue
        queue = path.split("/")[:-1] + _target(entry["target"]).split("/")
        resolved = []
        hops = 0
        while queue:
            piece = queue.pop(0)
            if piece in {"", "."}:
                continue
            if piece == "..":
                if not resolved:
                    _fail("unsafe_symlink")
                resolved.pop()
                continue
            resolved.append(piece)
            existing = entries.get("/".join(resolved))
            if existing and existing["kind"] == "symlink":
                hops += 1
                if hops > 40:
                    _fail("symlink_cycle_or_depth")
                resolved.pop()
                queue = _target(existing["target"]).split("/") + queue
                if len(queue) > MAX_PATH_BYTES:
                    _fail("unsafe_symlink")


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns, info.st_nlink)


def _read_regular(parent_fd, name, limit):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _fail("unsupported_file")
        if before.st_size > limit:
            _fail("file_limit")
        chunks, size = [], 0
        while size <= limit:
            data = os.read(fd, min(65536, limit + 1 - size))
            if not data:
                break
            chunks.append(data)
            size += len(data)
        after = os.fstat(fd)
        if size > limit:
            _fail("file_limit")
        if _identity(before) != _identity(after) or size != after.st_size:
            _fail("source_changed_during_read")
        return b"".join(chunks), after
    finally:
        os.close(fd)


def _mode(info):
    if stat.S_IMODE(info.st_mode) & ~0o777:
        _fail("unsupported_permission_bits")
    return stat.S_IMODE(info.st_mode)


def _limits(max_files, max_total_bytes, max_file_bytes):
    result = dict(max_files=max_files, max_total_bytes=max_total_bytes, max_file_bytes=max_file_bytes)
    for key, maximum in (("max_files", MAX_FILES), ("max_total_bytes", MAX_TOTAL_BYTES), ("max_file_bytes", MAX_FILE_BYTES)):
        if not _integer(result[key], 1, maximum):
            _fail("invalid_limits")
    return result


def _write_new(parent_fd, name, data, mode=0o444):
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent_fd)
    try:
        offset = 0
        while offset < len(data):
            count = os.write(fd, data[offset:])
            if count <= 0:
                _fail("write_failed")
            offset += count
        os.fchmod(fd, mode)
        os.fsync(fd)
    finally:
        os.close(fd)


def _put_blob(blobs_fd, data, seen):
    digest = _sha(data)
    if digest not in seen:
        _write_new(blobs_fd, digest, data)
        seen.add(digest)
    return digest


def _compact(path, data):
    if not path.endswith(".py") or any(p.startswith(".") for p in path.split("/")):
        return data, "passthrough", False
    try:
        view = code_source_view.render_source_bytes(path, data, format="compact")
        candidate = view["content"].encode("utf-8")
        if view["format"] == "compact" and view["ast_equivalent"] is True and len(candidate) < len(data):
            return candidate, "python-ast-compact", True
    except (code_source_view.SourceViewError, UnicodeError, RecursionError, ValueError):
        pass
    return data, "passthrough", False


def _scan(root_fd, limits, blobs_fd=None):
    entries, identities, excluded, seen = {}, {}, [], set()
    total = 0

    def visit(directory_fd, prefix):
        nonlocal total
        before = os.fstat(directory_fd)
        identities[prefix] = _identity(before)
        mode = _mode(before)
        if prefix:
            entries[prefix] = dict(kind="directory", mode=mode)
        names = []
        with os.scandir(directory_fd) as iterator:
            for item in iterator:
                if item.name == ".git":
                    excluded.append((prefix + "/" if prefix else "") + ".git")
                    continue
                names.append(item.name)
                if len(names) + len(entries) > limits["max_files"]:
                    _fail("entry_limit")
        for name in sorted(names):
            relative = _relative((prefix + "/" if prefix else "") + name)
            if len(entries) >= limits["max_files"]:
                _fail("entry_limit")
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                child_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd)
                try:
                    if _identity(os.fstat(child_fd)) != _identity(info):
                        _fail("source_changed_during_read")
                    visit(child_fd, relative)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(info.st_mode):
                data, observed = _read_regular(directory_fd, name, limits["max_file_bytes"])
                if _identity(info) != _identity(observed):
                    _fail("source_changed_during_read")
                total += len(data)
                if total > limits["max_total_bytes"]:
                    _fail("total_bytes_limit")
                entry = dict(kind="file", mode=_mode(observed), original_sha256=_sha(data), original_bytes=len(data))
                identities[relative] = _identity(observed)
                if blobs_fd is not None:
                    compact, rendering, equivalent = _compact(relative, data)
                    _put_blob(blobs_fd, data, seen)
                    entry.update(compact_sha256=_put_blob(blobs_fd, compact, seen), compact_bytes=len(compact), rendering=rendering, ast_equivalent=equivalent)
                entries[relative] = entry
            elif stat.S_ISLNK(info.st_mode):
                if info.st_nlink != 1:
                    _fail("unsupported_file")
                target = _target(os.readlink(name, dir_fd=directory_fd))
                if _identity(os.stat(name, dir_fd=directory_fd, follow_symlinks=False)) != _identity(info):
                    _fail("source_changed_during_read")
                entries[relative] = dict(kind="symlink", target=target)
                identities[relative] = _identity(info)
            else:
                _fail("unsupported_file")
        if _identity(os.fstat(directory_fd)) != _identity(before):
            _fail("source_changed_during_read")
        return mode

    root_mode = visit(root_fd, "")
    _check_symlinks(entries)
    return entries, root_mode, sorted(excluded), identities


def _original_entries(entries):
    return {path: {key: value for key, value in entry.items()
                   if key not in {"compact_sha256", "compact_bytes", "rendering", "ast_equivalent"}}
            for path, entry in entries.items()}


def _tree_hash(root_mode, entries):
    return _sha(_json(dict(root_mode=root_mode, entries=_original_entries(entries))))


def assert_repository_current(repository: Path, manifest: dict) -> dict:
    """Fence a retained view against the complete current original source tree.

    This is an observation, not a filesystem lease: another process can write
    immediately afterwards. Callers serving results should check on both sides
    of their operation. The manifest must already belong to their bound package.
    """
    _validate_manifest(manifest)
    source = _absolute(repository)
    if str(source) != manifest["source_root"]:
        _fail("source_root_mismatch")
    fd = _open_directory(source)
    try:
        entries, mode, excluded, identities = _scan(fd, manifest["limits"])
        if (entries != _original_entries(manifest["entries"])
                or mode != manifest["root_mode"] or excluded != manifest["excluded_paths"]):
            _fail("stale_source")
        after, after_mode, after_excluded, after_identities = _scan(fd, manifest["limits"])
        if (after != entries or after_mode != mode or after_excluded != excluded
                or after_identities != identities):
            _fail("source_changed_during_read")
        current_fd = _open_directory(source)
        try:
            if _identity(os.fstat(current_fd)) != identities[""]:
                _fail("stale_source")
        finally:
            os.close(current_fd)
        return dict(source_tree_sha256=manifest["source_tree_sha256"], source_current=True)
    except OSError:
        _fail("stale_source")
    finally:
        os.close(fd)


def _counts(manifest):
    files = [entry for entry in manifest["entries"].values() if entry["kind"] == "file"]
    return dict(entry_count=len(manifest["entries"]), file_count=len(files),
                original_bytes=sum(item["original_bytes"] for item in files),
                compact_bytes=sum(item["compact_bytes"] for item in files),
                compacted_files=sum(item["rendering"] == "python-ast-compact" for item in files))


def _stage(parent_fd):
    name = ".repository-view-" + uuid.uuid4().hex
    os.mkdir(name, 0o700, dir_fd=parent_fd)
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    return name, fd


def _cleanup(parent_fd, name):
    def accessible(fd):
        os.fchmod(fd, 0o700)
        with os.scandir(fd) as iterator:
            names = [entry.name for entry in iterator if entry.is_dir(follow_symlinks=False)]
        for child in names:
            os.chmod(child, 0o700, dir_fd=fd, follow_symlinks=False)
            child_fd = os.open(child, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            try:
                accessible(child_fd)
            finally:
                os.close(child_fd)
    try:
        os.chmod(name, 0o700, dir_fd=parent_fd, follow_symlinks=False)
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        try:
            accessible(fd)
        finally:
            os.close(fd)
        shutil.rmtree(name, dir_fd=parent_fd)
    except FileNotFoundError:
        pass


def _publish(parent_fd, staged, name):
    # Linux renameat2 is an atomic no-overwrite publication. A check followed by
    # os.rename would incorrectly replace a concurrently created empty directory.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        _fail("atomic_publication_unavailable")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(parent_fd, os.fsencode(staged), parent_fd, os.fsencode(name), 1) != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            _fail("destination_exists")
        if error in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
            _fail("atomic_publication_unavailable")
        _fail("publication_failed")
    try:
        os.fsync(parent_fd)
    except OSError:
        # Rename is already committed. Report durability uncertainty instead of
        # returning an error that incorrectly implies an absent destination.
        return False
    return True


def _absent(parent_fd, name):
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    _fail("destination_exists")


def create_repository_view(repository: Path, package: Path, *, max_files=MAX_FILES,
                           max_total_bytes=MAX_TOTAL_BYTES, max_file_bytes=MAX_FILE_BYTES,
                           origin_source_root: Path | None = None) -> dict:
    """Capture one bounded repository into a new, atomically published package."""
    source, output = _absolute(repository), _absolute(package)
    _source_anchor(str(source))
    origin = source if origin_source_root is None else _source_anchor(os.fspath(origin_source_root))
    _disjoint(source, output)
    _disjoint(origin, output)
    limits = _limits(max_files, max_total_bytes, max_file_bytes)
    source_fd = _open_directory(source)
    parent_fd = None
    stage_name = stage_fd = blobs_fd = None
    published = False
    try:
        parent_fd = _open_directory(output.parent)
        _absent(parent_fd, output.name)
        stage_name, stage_fd = _stage(parent_fd)
        os.mkdir("blobs", 0o700, dir_fd=stage_fd)
        blobs_fd = os.open("blobs", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=stage_fd)
        entries, root_mode, excluded, identities = _scan(source_fd, limits, blobs_fd)
        manifest = dict(schema=SCHEMA, source_root=str(source), origin_source_root=str(origin), root_mode=root_mode, entries=entries,
                        source_tree_sha256=_tree_hash(root_mode, entries), limits=limits,
                        exclusions=[".git at any depth"], excluded_paths=excluded, scope=SCOPE,
                        renderer=dict(schema=code_source_view.SCHEMA,
                                      sha256=_sha(Path(code_source_view.__file__).read_bytes()),
                                      python=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"))
        encoded = _json(manifest)
        if len(encoded) > MAX_MANIFEST_BYTES:
            _fail("manifest_limit")
        _validate_manifest(manifest)
        current, current_mode, current_excluded, current_identities = _scan(source_fd, limits)
        if (current != _original_entries(entries) or current_mode != root_mode
                or current_excluded != excluded or current_identities != identities):
            _fail("snapshot_changed")
        # Check that the original source pathname still refers to the captured
        # root; a renamed root must not be published as a fresh snapshot.
        current_root_fd = _open_directory(source)
        try:
            if _identity(os.fstat(current_root_fd)) != identities[""]:
                _fail("snapshot_changed")
        finally:
            os.close(current_root_fd)
        _write_new(stage_fd, "manifest.json", encoded)
        os.fchmod(blobs_fd, 0o555)
        os.fsync(blobs_fd)
        os.fchmod(stage_fd, 0o555)
        os.fsync(stage_fd)
        durable = _publish(parent_fd, stage_name, output.name)
        published = True
        return dict(schema=SCHEMA, operation="create", manifest_sha256=_sha(encoded),
                    source_tree_sha256=manifest["source_tree_sha256"], package=str(output),
                    directory_fsync_complete=durable, **_counts(manifest))
    except OSError:
        _fail("filesystem_error")
    finally:
        for fd in (blobs_fd, stage_fd, source_fd):
            if fd is not None:
                os.close(fd)
        if stage_name is not None and not published:
            _cleanup(parent_fd, stage_name)
        if parent_fd is not None:
            os.close(parent_fd)


def _keys(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        _fail("invalid_manifest")


def _validate_manifest(manifest):
    _keys(manifest, {"schema", "source_root", "origin_source_root", "root_mode", "entries", "source_tree_sha256", "limits", "exclusions", "excluded_paths", "scope", "renderer"})
    if manifest["schema"] != SCHEMA or _json(manifest["scope"]) != _json(SCOPE) or manifest["exclusions"] != [".git at any depth"]:
        _fail("invalid_manifest")
    if not _integer(manifest["root_mode"], 0, 0o777) or not _is_hash(manifest["source_tree_sha256"]):
        _fail("invalid_manifest")
    _source_anchor(manifest["source_root"])
    _source_anchor(manifest["origin_source_root"])
    _keys(manifest["limits"], {"max_files", "max_total_bytes", "max_file_bytes"})
    limits = _limits(**manifest["limits"])
    _keys(manifest["renderer"], {"schema", "sha256", "python"})
    renderer = manifest["renderer"]
    if (renderer["schema"] != "code-source-view@1" or not _is_hash(renderer["sha256"])
            or not isinstance(renderer["python"], str) or not 1 <= len(renderer["python"]) <= 32
            or any(c not in "0123456789." for c in renderer["python"])):
        _fail("invalid_manifest")
    entries = manifest["entries"]
    if type(entries) is not dict or len(entries) > limits["max_files"]:
        _fail("entry_limit")
    total = 0
    for path, entry in entries.items():
        _relative(path)
        if type(entry) is not dict:
            _fail("invalid_manifest")
        kind = entry.get("kind")
        if kind == "file":
            _keys(entry, {"kind", "mode", "original_sha256", "original_bytes", "compact_sha256", "compact_bytes", "rendering", "ast_equivalent"})
            if (not _is_hash(entry["original_sha256"]) or not _is_hash(entry["compact_sha256"])
                    or not _integer(entry["original_bytes"], 0, limits["max_file_bytes"])
                    or not _integer(entry["compact_bytes"], 0, entry["original_bytes"])):
                _fail("invalid_manifest")
            total += entry["original_bytes"]
            if entry["rendering"] == "passthrough":
                if (entry["original_sha256"] != entry["compact_sha256"] or entry["original_bytes"] != entry["compact_bytes"]
                        or entry["ast_equivalent"] is not False):
                    _fail("invalid_manifest")
            elif entry["rendering"] == "python-ast-compact":
                if (not path.endswith(".py") or any(p.startswith(".") for p in path.split("/"))
                        or entry["compact_bytes"] >= entry["original_bytes"] or entry["ast_equivalent"] is not True
                        or entry["original_sha256"] == entry["compact_sha256"]):
                    _fail("invalid_manifest")
            else:
                _fail("invalid_manifest")
        elif kind == "directory":
            _keys(entry, {"kind", "mode"})
        elif kind == "symlink":
            _keys(entry, {"kind", "target"})
            _target(entry["target"])
        else:
            _fail("invalid_manifest")
        if kind != "symlink" and not _integer(entry["mode"], 0, 0o777):
            _fail("invalid_manifest")
        parent = PurePosixPath(path).parent.as_posix()
        if parent != "." and (type(entries.get(parent)) is not dict or entries[parent].get("kind") != "directory"):
            _fail("invalid_path_topology")
    if total > limits["max_total_bytes"]:
        _fail("total_bytes_limit")
    excluded = manifest["excluded_paths"]
    if (type(excluded) is not list or len(excluded) > len(entries) + 1
            or any(not isinstance(path, str) for path in excluded) or excluded != sorted(set(excluded))):
        _fail("invalid_manifest")
    for path in excluded:
        pieces = path.split("/")
        if pieces[-1] != ".git":
            _fail("invalid_manifest")
        if len(pieces) > 1:
            parent = _relative("/".join(pieces[:-1]))
            if type(entries.get(parent)) is not dict or entries[parent].get("kind") != "directory":
                _fail("invalid_manifest")
    _check_symlinks(entries)
    if _tree_hash(manifest["root_mode"], entries) != manifest["source_tree_sha256"]:
        _fail("tree_digest_mismatch")


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_manifest_key")
        result[key] = value
    return result


def _decode_manifest(data):
    try:
        # Bound parser depth and container/item count before allocating JSON
        # structures. Canonical validation below handles complete JSON grammar.
        depth = nodes = 0
        quoted = escaped = False
        for char in data:
            if quoted:
                if escaped:
                    escaped = False
                elif char == 92:
                    escaped = True
                elif char == 34:
                    quoted = False
            elif char == 34:
                quoted = True
            elif char in (91, 123):
                depth += 1
                nodes += 1
            elif char in (93, 125):
                depth -= 1
            elif char in (44, 58):
                nodes += 1
            if depth > 16 or nodes > 150000:
                _fail("manifest_structure_limit")
        manifest = json.loads(data.decode("utf-8"), object_pairs_hook=_object_pairs,
                              parse_constant=lambda _: _fail("invalid_manifest"))
        _validate_manifest(manifest)
        if _json(manifest) != data:
            _fail("noncanonical_manifest")
        return manifest
    except (UnicodeError, RecursionError, TypeError, OverflowError):
        _fail("invalid_manifest")
    except ValueError as exc:
        if isinstance(exc, RepositorySourceViewError):
            raise
        _fail("invalid_manifest")


def _read_blob(blobs_fd, digest, size):
    data, _ = _read_regular(blobs_fd, digest, size)
    if len(data) != size or _sha(data) != digest:
        _fail("blob_digest_mismatch")
    return data


def _package_fds(package):
    root_fd = _open_directory(package)
    try:
        blobs_fd = os.open("blobs", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        return root_fd, blobs_fd
    except OSError:
        os.close(root_fd)
        _fail("invalid_package")


def load_repository_view(package: Path, *, expected_manifest_sha256: str) -> dict:
    """Verify a pinned canonical manifest and every bounded content blob."""
    if not _is_hash(expected_manifest_sha256):
        _fail("invalid_manifest_digest")
    root_fd, blobs_fd = _package_fds(package)
    try:
        with os.scandir(root_fd) as iterator:
            names = set()
            for entry in iterator:
                names.add(entry.name)
                if len(names) > 2:
                    _fail("unexpected_package_entry")
        if names != {"manifest.json", "blobs"}:
            _fail("invalid_package")
        data, _ = _read_regular(root_fd, "manifest.json", MAX_MANIFEST_BYTES)
        if _sha(data) != expected_manifest_sha256:
            _fail("manifest_digest_mismatch")
        manifest = _decode_manifest(data)
        expected = {}
        for entry in manifest["entries"].values():
            if entry["kind"] == "file":
                for representation in ("original", "compact"):
                    digest, size = entry[representation + "_sha256"], entry[representation + "_bytes"]
                    if digest in expected and expected[digest] != size:
                        _fail("invalid_manifest")
                    expected[digest] = size
        seen = set()
        with os.scandir(blobs_fd) as iterator:
            for entry in iterator:
                if entry.name not in expected or entry.name in seen:
                    _fail("unexpected_blob")
                seen.add(entry.name)
                _read_blob(blobs_fd, entry.name, expected[entry.name])
        if seen != set(expected):
            _fail("missing_blob")
        return manifest
    except OSError:
        _fail("invalid_package")
    finally:
        os.close(blobs_fd)
        os.close(root_fd)


def _representation(value):
    if not isinstance(value, str) or value not in {"original", "compact"}:
        _fail("invalid_representation")


def read_repository_file(package: Path, *, expected_manifest_sha256: str, path: str,
                         representation="compact") -> dict:
    """Read one retained representation, without consulting the source tree."""
    _representation(representation)
    path = _relative(path)
    manifest = load_repository_view(package, expected_manifest_sha256=expected_manifest_sha256)
    entry = manifest["entries"].get(path)
    if not entry or entry["kind"] != "file":
        _fail("not_regular_file")
    root_fd, blobs_fd = _package_fds(package)
    try:
        digest, size = entry[representation + "_sha256"], entry[representation + "_bytes"]
        data = _read_blob(blobs_fd, digest, size)
    except OSError:
        _fail("invalid_package")
    finally:
        os.close(blobs_fd)
        os.close(root_fd)
    try:
        content, encoding = data.decode("utf-8"), "utf-8"
    except UnicodeError:
        content, encoding = base64.b64encode(data).decode("ascii"), "base64"
    return dict(schema=SCHEMA, operation="read", manifest_sha256=expected_manifest_sha256,
                source_tree_sha256=manifest["source_tree_sha256"], path=path, representation=representation,
                content=content, encoding=encoding, view_sha256=digest, view_bytes=size, **entry)


def restore_repository_view(package: Path, destination: Path, *, expected_manifest_sha256: str,
                            representation="original") -> dict:
    """Restore a complete tree into an absent, disjoint destination atomically."""
    _representation(representation)
    package, output = _absolute(package), _absolute(destination)
    _disjoint(package, output)
    manifest = load_repository_view(package, expected_manifest_sha256=expected_manifest_sha256)
    _disjoint(Path(manifest["source_root"]), output)
    _disjoint(Path(manifest["origin_source_root"]), output)
    parent_fd = _open_directory(output.parent)
    package_fd = blobs_fd = stage_fd = None
    stage_name = None
    published = False
    try:
        _absent(parent_fd, output.name)
        package_fd, blobs_fd = _package_fds(package)
        stage_name, stage_fd = _stage(parent_fd)
        children = {}
        for path, entry in manifest["entries"].items():
            parent = str(PurePosixPath(path).parent)
            children.setdefault("" if parent == "." else parent, []).append((path, entry))
        # Bound open descriptors by tree depth, not by repository width.
        def restore_at(fd, prefix):
            for path, entry in children.get(prefix, []):
                name = PurePosixPath(path).name
                if entry["kind"] == "directory":
                    os.mkdir(name, 0o700, dir_fd=fd)
                    child_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    try:
                        restore_at(child_fd, path)
                        os.fchmod(child_fd, entry["mode"])
                        os.fsync(child_fd)
                    finally:
                        os.close(child_fd)
                elif entry["kind"] == "file":
                    data = _read_blob(blobs_fd, entry[representation + "_sha256"], entry[representation + "_bytes"])
                    _write_new(fd, name, data, entry["mode"])
                else:
                    os.symlink(entry["target"], name, dir_fd=fd)
        restore_at(stage_fd, "")
        os.fchmod(stage_fd, manifest["root_mode"])
        os.fsync(stage_fd)
        durable = _publish(parent_fd, stage_name, output.name)
        published = True
        return dict(schema=SCHEMA, operation="restore", manifest_sha256=expected_manifest_sha256,
                    source_tree_sha256=manifest["source_tree_sha256"], representation=representation,
                    destination=str(output), directory_fsync_complete=durable, **_counts(manifest))
    except OSError:
        _fail("filesystem_error")
    finally:
        for fd in (blobs_fd, package_fd, stage_fd):
            if fd is not None:
                os.close(fd)
        if stage_name is not None and not published:
            _cleanup(parent_fd, stage_name)
        os.close(parent_fd)
